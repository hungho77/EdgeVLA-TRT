# TurboVLA

[TurboVLA](https://github.com/H-EmbodVis/TurboVLA) (0.2B) maps vision and language straight to actions without an
LLM: DINOv3 ViT-B/16 encodes each camera, BERT the instruction, six bidirectional vision-language fusion layers mix
them, and an ACT decoder regresses a 12-row action chunk (tanh) from the fused tokens and two state tokens. The
public checkpoint is [`H-EmbodVis/TurboVLA`](https://huggingface.co/H-EmbodVis/TurboVLA)
`checkpoints/libero/turbovla_libero.pth`, trained on all four LIBERO suites.

It runs as two engines built from the official modules through the legacy ONNX exporter:

- `text` (BERT and its projection) runs once per instruction; `TurbovlaPolicy` caches its output. It computes in
  FP32 behind FP16 inputs and outputs (5.9 ms, built with `--noTF32`), which keeps the text tokens within FP16
  rounding of the official ones.
- `policy` (DINOv3 on both views, the fusion and the decoder) runs every call in FP16.

`TurbovlaPolicy` reproduces the official evaluation path (`turbovla/evaluation/suite_policy.py` with
`libero_all4_stats.json`):

- BERT tokens from a C++ WordPiece tokenizer (identical ids to HF's on every LIBERO instruction, accents, CJK and
  Cyrillic). GroundingDINO's sub-sentence attention mask and position ids are built on the host, padded to the
  checkpoint's per-instruction length (11, 14 or 21; the BERT rows past an instruction's own length are zero, as
  the official encoder fills them).
- ImageNet normalization of 256x256 views (`primary`, `wrist`). The official policy rejects other sizes; the
  server resizes them with PIL's bicubic filter.
- The state normalized with the suite's mean / std, arm actions unnormalized from [-1, 1] with its min / max, the
  gripper set to +1 / -1 by sign.

**FP16 attention range.** DINOv3's first layer attends with scores up to 9.5e4 after scaling (its high-norm
register tokens), past FP16's 65504. PyTorch's FP16 attention kernels accumulate in FP32 and never store the
scores, so eager FP16 is fine; TensorRT's FP16 attention stores them, and the visual tokens came out 10% off
(relative RMS) with actions up to 0.09 off. The exporter computes DINOv3's QK^T and softmax in FP32 and the
probabilities times V in FP16. DINOv3's RoPE tables, which depend only on the image size, are folded to constants
(their coordinate arithmetic has a shape-dependent branch TensorRT cannot parse).

**Chunk blending instead of real-time chunking.** The decoder regresses the chunk in one pass, so there is no
denoising loop to inpaint. `TurbovlaBlend` blends the new chunk with the previous chunk's rows in normalized space,
with the weights of pi0.5's real-time chunking (1 on the frozen rows, then an exponential ramp over the overlap), so
the server speaks the same `overlap_frozen` RTC fields (its `ready` line says `"rtc_method": "blend"`).

## Accuracy and speed

AGX Orin 64 GB, JetPack 6.2, against the official policy in FP32 on the board's GPU (five LIBERO observations,
instruction padding 21, 11 and 14, and three typed instructions whose "." / "?" split the sub-sentence
masks; up to 0.0026 on those):

| Stage | max \|Δ\| | Reference spread |
|---|---|---|
| Split PyTorch modules, FP32 | 0 (text tokens and actions) | |
| Text engine | ≤ 0.007 on tokens up to 20 | |
| Policy engine on the official text tokens | ≤ 0.0023 | |
| `turbovla_policy_server`, normalized actions | ≤ 0.0032 (gripper signs identical) | official BF16 vs FP32: 0.0068 |

Inline frames give bit-identical actions to the same images by path. A policy call on a LIBERO observation takes
32 ms through the LIBERO harness, median of 20 (policy engine 14.2 ms; the text engine only when the instruction changes).

LIBERO-Spatial, 10 tasks x 10 episodes, the official rollout's conventions (`libero_eval.py --policy turbovla`):
**95%** (failures on tasks 1, 5, 7 and 8, all at the step limit), reported 97.0% (TurboVLA paper, 500 episodes).

The async loop at 30 Hz (`turbovla_async_control`, overlap 6, frozen 3) and `replay_robot.py` over TCP with SO101
frames: no stalls, planner lag 1-2 ticks, switch jump 0.0000 with blending (0.84 on the arm, 2.0 on the gripper
without).

```bash
# Official reference env: TurboVLA's code on a CUDA PyTorch with transformers 4.57 (timm is stubbed)
python experimental_models/turbovla/scripts/turbovla_reference.py --repo TurboVLA --checkpoint TurboVLA-hf \
    --dinov3 dinov3-vitb16 --bert bert-base-uncased --obs obs.npz --out ref.npz
python experimental_models/turbovla/scripts/export_turbovla.py --repo TurboVLA --checkpoint TurboVLA-hf \
    --dinov3 dinov3-vitb16 --bert bert-base-uncased --check ref.npz --obs obs.npz --out turbovla_onnx
trtexec --onnx=turbovla_onnx/text.onnx --saveEngine=engines/text.engine --stronglyTyped --noTF32
trtexec --onnx=turbovla_onnx/policy.onnx --saveEngine=engines/policy.engine --stronglyTyped
cp turbovla_onnx/config.json turbovla_onnx/tokenizer.json engines/
python experimental_models/turbovla/scripts/run_turbovla_engines.py --engines engines --reference ref.npz \
    --obs obs.npz                                         # stage by stage
turbovla_policy_server --engineDir engines [--port 5555]
turbovla_async_control --engineDir engines --inputFile obs.json [--rtc 1]
```

The checkpoint carries every weight, so the encoders are built from their configs only. `--dinov3` holds DINOv3
ViT-B/16's `config.json` and `preprocessor_config.json` (`facebook/dinov3-vitb16-pretrain-lvd1689m` is gated; the
public `onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX` has the same config files), `--bert` holds
`bert-base-uncased`'s config and tokenizer:

```bash
for f in config.json preprocessor_config.json; do
    hf download onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX $f --local-dir dinov3-vitb16; done
hf download google-bert/bert-base-uncased config.json tokenizer.json tokenizer_config.json vocab.txt \
    --local-dir bert-base-uncased
hf download H-EmbodVis/TurboVLA --include "*.json" "*.yaml" "checkpoints/libero/*" --local-dir TurboVLA-hf
```

The released checkpoint stores its weights as `model_state_dict`, which the official loader read until it switched
to requiring `ema_model_state_dict` (commit `ced2b0c`); `turbovla_reference.py` restores the earlier lookup. Its
parameters derive from DINOv3 and are distributed under the
[DINOv3 License](https://huggingface.co/H-EmbodVis/TurboVLA/blob/main/DINOv3_LICENSE.md), which carries over to the
engines built from it.

## Runs trained with the starVLA trainer (SO101)

The TurboVLA repo's starVLA trainer (`third_party/starvla_runtime`) builds the same core model and saves a run
directory instead: `config.yaml`, `checkpoints/*.pt` (the EMA weights), the run's data config, `modality.json` and
`dataset_statistics.json`, e.g. [`twanghcmut/TurboVLA-SO101-LeftArm-Multitask`](https://huggingface.co/twanghcmut/TurboVLA-SO101-LeftArm-Multitask)
(11 SO101 left-arm tasks, 6 joints, cameras `top` 640x360 and `wrist` 640x480, 16-row chunk).
`turbovla_reference.py` and `export_turbovla.py` take such a directory as `--checkpoint` and follow its
`predict_action` and open-loop evaluation, all recorded in the files the run ships:

- cameras in `modality.json`'s video order;
- state and actions normalized as the data config's `normalization_modes` declare (min_max over the statistics'
  min / max, a dim whose min equals its max passing through raw, actions clipped to [-1, 1] before unnormalizing; a
  continuous gripper);
- frames resized by DINOv3's fast image processor: antialiased bilinear in float, squashed to 224 x 224
  (Pillow's bicubic, the LIBERO path, is up to 18 levels off);
- the instruction at its own token count: the run pads to the longest instruction of a batch, and the action head
  attends to every text token, so padding to 256 moved the normalized actions by up to 0.16. The text and policy
  engines take the length as a dynamic axis (2 to 256 tokens), with the attention modules exported in a
  length-free form.

`config.json` records these as `cameras`, `state_normalization`, `binary_gripper`, `clip_actions`, `resize` and
`text_padding`; an engine directory without them keeps the LIBERO behaviour.

AGX Orin, two frames of `hungho77/so101_left_arm_pick_red_cube_into_pot`, `turbovla_policy_server` from raw frames
against the run's own `predict_action` in FP32: normalized actions within 0.0011, robot actions within 0.08-0.11 on
a ±98 range (the official BF16 vs FP32: 0.27-0.31), about 31 ms per call. Open loop over episode 0 through
the SO101 robot client (MAE of the first 16 rows, stride 16): 1.546, the model card's 1.534. The model hardly
tells its two views apart: swapping them moves the official actions by 0.2, feeding the top view twice by 19-35.

```bash
python experimental_models/turbovla/scripts/turbovla_reference.py --repo TurboVLA \
    --checkpoint TurboVLA-SO101-LeftArm-Multitask --dinov3 dinov3-vitb16 --bert bert-base-uncased \
    --obs-json obs_0.json obs_1.json --out ref.npz        # obs: {"task", "state", "cameras": {name: image}}
python experimental_models/turbovla/scripts/export_turbovla.py --repo TurboVLA \
    --checkpoint TurboVLA-SO101-LeftArm-Multitask --dinov3 dinov3-vitb16 --bert bert-base-uncased --out turbovla_onnx
T=input_ids:1xN,position_ids:1xN,self_attention:1xNxN,hidden_valid:1xN,attention:1xN
trtexec --onnx=turbovla_onnx/text.onnx --saveEngine=engines/text.engine --stronglyTyped --noTF32 \
    --minShapes=${T//N/2} --optShapes=${T//N/32} --maxShapes=${T//N/256}
T=text_tokens:1xNx256,attention:1xN,self_attention:1xNxN
trtexec --onnx=turbovla_onnx/policy.onnx --saveEngine=engines/policy.engine --stronglyTyped \
    --minShapes=${T//N/2} --optShapes=${T//N/32} --maxShapes=${T//N/256}
cp turbovla_onnx/config.json turbovla_onnx/tokenizer.json engines/
python experimental_models/turbovla/scripts/run_turbovla_engines.py --engines engines --reference ref.npz
```

