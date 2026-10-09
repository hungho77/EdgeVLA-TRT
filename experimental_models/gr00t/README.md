# GR00T (N1.5 / N1.6 / N1.7)

GR00T policies on EdgeVLA-TRT: N1.7 through Edge-LLM's Qwen3-VL path, N1.6 and N1.5 through the Eagle
backbone in `tensorrt_edgellm.models.eagle`, all three with the same split action head, RTC and servers.

## GR00T N1.7

NVIDIA GR00T N1.7 (SO101 fine-tune) runs as Edge-LLM backbone plus a vendored action head:

- **Backbone**: GR00T's Cosmos-Reason2 (Qwen3-VL) truncated to `select_layer` = 16 layers, exported through the
  regular Qwen3-VL path with `emit_hidden_states: "pre_norm"`, so the engine returns every position's layer-16
  output before the final norm. GR00T pops the extra layers off a full-depth model, and under transformers 4.57
  its `hidden_states[-1]` is then the unnormalized tensor; the post-norm one has cosine 0.08 to it.
- **Action head**, three engines for one embodiment: `vl_prep` (vlln, VL self-attention and the K/V of all 16 DiT
  cross-attention blocks, once per call), `state_encoder`, and `denoise_step` (action encoder, DiT on the cached
  K/V, action decoder, Euler update). The DiT's timestep conditioning is precomputed for the fixed schedule, the
  DiT linears can be exported as calibrated W8A8 (`--int8-weights`), and the four denoising steps replay as one
  CUDA graph.
- **Policy**: `processing.json` records the embodiment's state normalization and action decoding (percentile
  bounds, per-step relative-action bounds, relative-to-absolute joints) from the official processor;
  `gr00t_policy_server` takes camera frames, raw state and an instruction and returns absolute actions, with
  optional real-time chunking (RTC: the next chunk is inpainted from the tail of the previous one). The tail is
  kept in absolute joint space and re-encoded against the new state, so the frozen rows reproduce the actions
  already committed even after the arm moved; seeding the previous normalized rows directly, as the model-level
  API does, misses them by up to 2.8 on SO101 when the state moved 6.6 in 8 frames.
  `gr00t_policy_client.py` applies GR00T's image and language preprocessing on the robot side.

AGX Orin, two SO101 dataset frames from raw video and raw state, against the official `Gr00tPolicy` (fp32
PyTorch) with the same noise:

| | FP16 head | W8A8 DiT |
|---|---|---|
| Backbone hidden states vs PyTorch | per-token cosine 0.99986 (min 0.998) | same |
| Absolute SO101 actions, max \|Δ\| (joint range ±90) | 0.21 | 1.06 |
| Absolute actions with RTC (overlap 8, frozen 2), max \|Δ\| | 0.25 | |
| Policy step p50, CUDA graph (backbone 26.5 ms) | 96.4 ms | 73.1 ms |

A VLA-OPT quantization job shared the board during the timing runs. [`ducido/GR00T-N1.7-SO101-banana-all49`](https://huggingface.co/ducido/GR00T-N1.7-SO101-banana-all49) built
the same way matches the official policy within 0.13 (0.24 with RTC); its `config.json` names the fine-tuner's
local path as `model_name`, so the official reference needs a copy with `nvidia/Cosmos-Reason2-2B` there.

```bash
# backbone
python experimental_models/gr00t/scripts/extract_gr00t_n1_7_backbone.py --gr00t GR00T-N1.7-SO101-Multitask \
    --base <Cosmos-Reason2-2B snapshot> --out gr00t_backbone
tensorrt-edgellm-export gr00t_backbone gr00t_onnx
llm_build --onnxDir gr00t_onnx/llm --engineDir engines/llm --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 640
visual_build --onnxDir gr00t_onnx/visual --engineDir engines --minImageTokens 16 --maxImageTokens 512 \
    --maxImageTokensPerImage 256
# action head and processing (need the GR00T N1.7 source and its dependencies, e.g. transformers 4.57 and diffusers)
python experimental_models/gr00t/scripts/export_gr00t_action_head.py --gr00t-src <dir holding gr00t/> \
    --checkpoint GR00T-N1.7-SO101-Multitask --embodiment new_embodiment --out action_onnx \
    [--int8-weights --features <pre-norm backbone features .npy> --input-ids <input ids .npy>]
python experimental_models/gr00t/scripts/export_gr00t_processing.py --gr00t-src <dir holding gr00t/> \
    --checkpoint GR00T-N1.7-SO101-Multitask --embodiment new_embodiment --out engines/action/processing.json
# action engines: the backbone token axis needs a profile (--int8 as well for a W8A8 denoise_step)
cp action_onnx/config.json engines/action/
T=backbone_features:1xNx2048,image_mask:1xN,attention_mask:1xN
trtexec --onnx=action_onnx/vl_prep.onnx --saveEngine=engines/action/vl_prep.engine --fp16 \
    --minShapes=${T//N/16} --optShapes=${T//N/145} --maxShapes=${T//N/512}
trtexec --onnx=action_onnx/state_encoder.onnx --saveEngine=engines/action/state_encoder.engine --fp16
T=cross_keys:16x1xNx1536,cross_values:16x1xNx1536,text_bias:1x1x1xN,image_bias:1x1x1xN
trtexec --onnx=action_onnx/denoise_step.onnx --saveEngine=engines/action/denoise_step.engine --fp16 \
    --minShapes=${T//N/16} --optShapes=${T//N/145} --maxShapes=${T//N/512}
python experimental_models/gr00t/examples/gr00t_policy_client.py --server-cmd "gr00t_policy_server \
    --llmEngineDir engines/llm --multimodalEngineDir engines --actionEngineDir engines/action" ...
```

## GR00T N1.6

GR00T N1.6 (SO101 fine-tune) shares N1.7's action head and denoising loop; its backbone is NVIDIA's Eagle 3
(Eagle-Block2A-2B-v2: a SigLIP2 tower, 2x2 pixel unshuffle and an MLP connector into Qwen3-1.7B truncated to 16
layers), which Edge-LLM has no model for:

- **Backbone** (`tensorrt_edgellm.models.eagle`): the two halves in plain PyTorch ops, exported as a `visual` and
  a `prefix` engine (`trtexec --stronglyTyped`). Each camera is its own batch item: the official packed
  FlashAttention keeps the images apart, and its eager fallback would not (image-token cosine 0.77 between the
  two). GR00T casts the backbone to bf16 at load, which also rounds Qwen3's RoPE `inv_freq` buffer; the model was
  trained with those frequencies, so the export reproduces them. The fine-tuned top four LLM layers and the whole
  action head are stored in FP32, and the official policy rounds them to bf16 when it loads; the engines keep the
  stored weights.
- **Preprocessing** (`Gr00tEagleBackbone`, needs OpenCV): albumentations' shortest-edge INTER_AREA resize,
  95% centre crop and resize again (OpenCV), Eagle's resize to multiples of 28 (a port of Pillow's fixed-point
  bicubic), the lower-cased, punctuation-free instruction and Eagle's chat template. Pixel values are
  bit-identical to the official processor's and token ids identical.
- **Action head and policy**: `export_gr00t_action_head.py` and `export_gr00t_processing.py` take N1.6
  checkpoints too (`--check`: the split head matches the official `get_action` exactly), and
  `Gr00tN17Policy`, its RTC included, runs on the Eagle features unchanged.

AGX Orin, two SO101 dataset frames (300, 900) from raw video and raw state, against the official `Gr00tPolicy`
on the CPU with FP32 activations (its weights as it loads them) and the same x_0 (`scripts/official_reference.py`);
the official policy as it serves, bf16 throughout, is the precision floor:

| Frames 300 / 900 | FP16 engines | official bf16 |
|---|---|---|
| Backbone features, cosine | 0.99764 / 0.99980 | 0.99826 / 0.99119 |
| Normalized actions, max \|Δ\| | 0.0049 / 0.0029 | 0.0072 / 0.0047 |
| Absolute SO101 actions, max \|Δ\| (joint range ±90) | 0.148 / 0.202 | 0.287 / 0.159 |
| Policy step p50, preprocessing to actions (visual 53.9 ms, prefix 30.9 ms) | 171.7 ms | |

A few image tokens are very sensitive to precision in both (lowest per-token cosine 0.45 for the engines on frame
300, 0.21 for official bf16 on frame 900), and TensorRT rebuilds of the same graph moved the frame-300 backbone
cosine between 0.994 and 0.9988; the actions stayed at or below the bf16 spread in every build.
`gr00t_policy_server --eagleBackboneDir engines/backbone --actionEngineDir engines/action` serves N1.5 and N1.6: it
takes raw camera frames (in `processing.json`'s `video_keys` order) and the raw instruction, since the Eagle backbone
applies the official preprocessing, and replies with the same actions as `gr00t_eagle_policy_inference`.

`gr00t_async_control --eagleBackboneDir` at 30 Hz (horizon 16, overlap 10, frozen 8, drifting state): no stalls,
planner latency 5-6 ticks, switch jump at most 0.0105 joint units (11.2 with no frozen rows).

```bash
python experimental_models/gr00t/scripts/official_reference.py --gr00t-src Isaac-GR00T \
    --checkpoint GR00T-N1.6-SO101-Multitask --modality-config GR00T-N1.6-SO101-Multitask/so101_config.py \
    --dataset so101-multitask --frame 300 --eager-attention --out ref_f300.npz   # Isaac-GR00T's pins, transformers 4.51.3
python -m tensorrt_edgellm.models.eagle.export GR00T-N1.6-SO101-Multitask \
    Isaac-GR00T/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2 backbone_onnx
trtexec --onnx=backbone_onnx/visual/model.onnx --saveEngine=engines/backbone/visual.engine --stronglyTyped \
    --minShapes=pixel_values:1x3x252x336 --optShapes=pixel_values:2x3x252x336 --maxShapes=pixel_values:3x3x252x336
trtexec --onnx=backbone_onnx/prefix/model.onnx --saveEngine=engines/backbone/prefix.engine --stronglyTyped \
    --minShapes=token_ids:1x2,image_features:1x1x2048 --optShapes=token_ids:1x260,image_features:1x216x2048 \
    --maxShapes=token_ids:1x512,image_features:1x324x2048
cp backbone_onnx/*.json engines/backbone/
# action head and processing.json as for N1.7 (with the N1.6 source), into engines/action
gr00t_eagle_policy_inference --backboneDir engines/backbone --actionDir engines/action --inputFile obs.json
```

## GR00T N1.5

GR00T N1.5 (SO101 fine-tune, data config `so100_dualcam`) runs on the same pieces:

- **Backbone**: Eagle 2.5, SigLIP at 224x224 (256 tokens per camera, no pixel unshuffle), a linear connector and
  Qwen3 cut to 12 layers; `tensorrt_edgellm.models.eagle` picks the variant from the checkpoint. N1.5 loads with
  `from_pretrained(torch_dtype=bfloat16)`, which leaves the RoPE `inv_freq` buffer in FP32, so N1.5 keeps the exact
  frequencies (N1.6 rounds them).
- **Action head**: the same DiT with N1.5's 32 learned target-vision tokens between state and actions, a 4-layer
  VL self-attention, and cross-attention over every backbone token; the split head matches `get_action` exactly.
- **Preprocessing**: 95% centre crop and torch's antialiased bilinear resize to 224x224 on [0, 1] floats, truncated
  to 8 bits, then the instruction after the images. The truncation makes the resize's arithmetic visible: the port
  uses torch's FP32 weights, its width-then-height order and fused multiply-adds, which plain multiply-adds miss
  on ~1% of the pixels. Min/max state and action normalization without clipping (`clip_actions: false`).
- **Language**: N1.5's official clients pass the instruction as a one-element list, which the batched transform
  renders as `"['Pick up ...']"`, while training saw the plain text; the port feeds the plain text
  (`official_reference.py --n15-served-language` reproduces the served prompt; actions move up to 0.30).

| Frames 300 / 900, vs official FP32 | FP16 engines | official bf16 |
|---|---|---|
| Absolute SO101 actions, max \|Δ\| | 0.091 / 0.085 | 0.741 / 0.456 |
| Normalized actions, max \|Δ\| (frame 300) | 0.0016 | |
| Policy step p50 (visual 22.9 ms, prefix 39.9 ms) | 110.4 ms | |

Pixel values are bit-identical and token ids identical on both frames. `gr00t_async_control --eagleBackboneDir` at
30 Hz (overlap 8, frozen 5): no stalls, planner latency 4 ticks, switch jump 0.0000 (14.2 with no frozen rows).

```bash
python experimental_models/gr00t/scripts/official_reference.py --gr00t-src groot-n15 \
    --checkpoint GR00T-N1.5-SO101-Multitask --n15-data-config so100_dualcam --dataset so101-multitask \
    --frame 300 --eager-attention --out ref_f300.npz   # N1.5's pins; pytorch3d, decord, flash_attn unused
python -m tensorrt_edgellm.models.eagle.export GR00T-N1.5-SO101-Multitask \
    groot-n15/gr00t/model/backbone/eagle2_hg_model backbone_onnx
python experimental_models/gr00t/scripts/export_gr00t_processing.py --gr00t-src groot-n15 \
    --checkpoint GR00T-N1.5-SO101-Multitask --n15-data-config so100_dualcam --out engines/action/processing.json
# action head as for N1.6 (--max-backbone-tokens 1024); engines with trtexec as above, 224x224 images
```

## Faster Eagle backbone

The Eagle prefix (Qwen3 cut to `select_layer`) can run on Edge-LLM's LLM runtime with its attention plugin
instead of the plain-op `prefix.engine`. `Gr00tEagleBackbone` uses an `llm/` engine when the backbone directory
holds one, feeding the visual engine's features as precomputed image embeddings at the image-context tokens.
N1.6's policy rounds Qwen3's RoPE `inv_freq` to bf16 at load; the exported config carries `rope_inv_freq_bf16`
and the runtime rounds the same way.

```bash
python experimental_models/gr00t/scripts/extract_gr00t_eagle_llm.py --gr00t <checkpoint> \
    --eagle-dir <Eagle model dir> --tokenizer-dir engines/backbone --out eagle_llm
tensorrt-edgellm-export eagle_llm eagle_llm_onnx
python experimental_models/gr00t/scripts/extract_gr00t_eagle_llm.py --patch-onnx-config eagle_llm_onnx/llm \
    --out eagle_llm
llm_build --onnxDir eagle_llm_onnx/llm --engineDir engines/backbone/llm --maxBatchSize 1 \
    --maxInputLen 512 --maxKVCacheCapacity 640      # N1.5 (564 tokens for two views): 1024 / 1152
```

The host preprocessing runs the views concurrently and reuses the token ids while the task stays the same (pixels
bit-identical). AGX Orin, LIBERO-Spatial checkpoints, one policy-server call on an idle board:

| | Plain-op prefix | LLM-runtime prefix | LIBERO-Spatial |
|---|---|---|---|
| N1.6 | 149 ms | 120 ms (prefix 30 -> 25 ms) | 97% -> 100% |
| N1.5, 8 denoising steps | 244 ms | 196 ms (prefix 57 -> 46 ms) | 88% |
| N1.5, 4 denoising steps (`--denoising-steps 4`) | | 112 ms | 89% |

Against the plain-op prefix on the same inputs, the backbone features keep per-token cosine >= 0.9994 and the robot
actions move by at most 0.0009. The rest of the gap to N1.7 is the model: SigLIP2-so400m on 324 patches per view
spends 30 of its 42 ms in GEMMs that TensorRT already runs at about 17 TFLOP/s, with attention fused (5 ms).
