# MolmoAct2

[MolmoAct2](https://github.com/allenai/molmoact2) (AllenAI) is a Molmo2 vision-language model with a flow-matching
action expert attached to every layer: a SigLIP2 ViT (27 layers, features from layers -3 and -9) whose 27 x 27
patches are pooled 2 x 2 into 196 tokens per image, a 36-layer language model (hidden 2560, 8 KV heads) over the
prompt, and a 36-block action expert (hidden 768) that cross-attends to each language-model layer's keys and
values and denoises a 10 x 32 chunk in 10 Euler steps. The LIBERO checkpoint is LeRobot's
[`allenai/MolmoAct2-LIBERO-LeRobot`](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot) (trained on all four
suites); the server runs its continuous-action path (`inference_action_mode=continuous`), not the discrete FAST
tokens.

It runs as five engines:

| Engine | Built from | Per call |
|---|---|---|
| `vision` | the ViT, the 2 x 2 attention pooling and the projector, for the checkpoint's two cameras | once, 392 tokens |
| `prefix_a` | token embeddings with the visual features added at the image patches, layers 0-17 and their K / V | once, ~500 tokens |
| `prefix_b` | layers 18-35, their K / V (the last layer only computes K / V) | once |
| `context` | the expert's shared context projection and norm and every block's key norm over the 36 layers' K / V | once |
| `step` | one expert step, x_{t+dt} = x_t + dt * strength * v | 10 times, as a CUDA graph |

`MolmoAct2Policy` reproduces the LeRobot processor on the host. Each camera's frame is resized to 378 x 378 with
the processor's bilinear filter and cut into SigLIP2 patches (bit-identical to the official `pixel_values`). The
state is normalized with the masked q01 / q99 statistics (the gripper passes through), clipped and binned into 256
state tokens in the `robot_action` prompt with the setup and control-mode strings; the prompt is tokenized with the
checkpoint's tokenizer (ids identical to the official processor's). Image tokens attend to each other, text tokens
causally, and the expert cross-attends to every prompt token but the `<|im_end|>` ones. The chunk is clamped to
[-1, 1] and unnormalized on the masked dims (the gripper is returned as the model gives it).

**What FP16 could not hold.** The projected image features reach 1.7e4 and stay in the language model's residual
stream, and the pooling attention's unscaled scores reach 8e4: RMSNorm, every softmax and the pooling attention
run in FP32, the FP32 products built with `--noTF32`. The official model also runs the ViT's attention in FP32
(`float32_attention`); its scaled scores and the language model's stay below 64, so both compute QK^T in FP16 with
the scale on Q, which takes a call from 630 to 420 ms (on a shared GPU) at the same action error.
`export_molmoact2.py --fp32-attention` builds the official precision.

**TensorRT 10.3.** The expert's self-attention reads Q, K and V out of one fused projection viewed as
[B, S, 3, H, D], then applies per-head RMSNorm and RoPE. TensorRT 10.3 miscomputes that pattern in FP16 (the
first block's output was off by more than its own magnitude while onnxruntime matched PyTorch), and Edge-LLM's
`apply_trt103_workarounds` does not match it, so the export splits the projection into three GEMMs. The expert's
per-step modulations depend only on the step's time, so they are precomputed into a table the step engine indexes.

**Real-time chunking.** As GR00T's: the overlap rows start from the previous chunk's normalized rows from
`start_row`, and their velocity is scaled by 0 on the frozen rows, then by an exponential ramp.

## Accuracy and speed

AGX Orin 64 GB, JetPack 6.2, against the official LeRobot policy in FP32 (CPU), two LIBERO observations with
instructions of 17 and 3 words (497 and 482 prompt tokens), the same initial noise:

| Stage | FP16 QK^T (default) | `--fp32-attention` |
|---|---|---|
| Visual features (from the official `pixel_values`) | 0.10% / 0.27% | 0.13% / 0.14% |
| Language-model K / V, chained | 0.7% / 2.0-2.2% | 0.4% / 1.3% |
| Normalized chunk, context and steps on the official K / V, max \|Δ\| | 0.0008 / 0.0011 | 0.0008 / 0.0011 |
| Normalized chunk, chained, max \|Δ\| | 0.0021 / 0.0020 | 0.0030 / 0.0017 |
| `molmoact2_policy_server` actions from raw frames, max \|Δ\| | 0.0014 / 0.0020 | 0.0014 / 0.0016 |

Inline frames give bit-identical actions to the same images by path; with and without the CUDA graph the actions
are identical. A call takes about 410 ms (median of 20 on a LIBERO observation, with the desktop drawing about 20%
of the GPU): vision 79 ms, language model and expert context 200 ms, 10 expert steps 108 ms, host 6 ms. The host
splices each image's patch tokens in after tokenizing and prepares the prompt while the vision engine runs.

**Control rate.** `replay_robot.py` over TCP with raw frames at 5 Hz (`overlap=5, frozen=4`, 60 ticks): call
median 399 ms, planner lag 2-3 ticks, no stalls, switch jump 0.0000 with RTC and 1.96 without. LIBERO's 20 Hz would
need a call under about 250 ms with one request in flight (a 10-row chunk lasts 500 ms), which this model does not
reach on Orin.

LIBERO-Spatial, 10 tasks x 10 episodes, this harness's protocol (fixed initial states, 10 settle steps, 280 steps,
256 x 256 renders) with the checkpoint's conventions (both views flipped, 10 of 10 rows per call, gripper as
returned): **99%** (99 of 100; task 5 failed once at the step limit), and 100% with `--fp32-attention`. The checkpoint reports 98.4% over 50 episodes per task with LeRobot's `lerobot-eval`
(per-episode seeds from 1000).

```bash
# Reference env: LeRobot 0.6.1 (lerobot.policies.molmoact2) on CPU; --hf holds MolmoAct2-LIBERO's config,
# processor and tokenizer files, --fast the FAST tokenizer's (the LeRobot weights replace the base model's)
python experimental_models/molmoact2/scripts/molmoact2_reference.py --checkpoint MolmoAct2-LIBERO-LeRobot \
    --hf MolmoAct2-LIBERO-hf --fast MolmoAct2-FAST-Tokenizer --obs obs.npz --out ref.npz --precision fp32
for stage in vision prefix_a prefix_b context step; do
    python experimental_models/molmoact2/scripts/export_molmoact2.py --checkpoint MolmoAct2-LIBERO-LeRobot \
        --hf MolmoAct2-LIBERO-hf --fast MolmoAct2-FAST-Tokenizer --stage $stage --out onnx
done
python experimental_models/molmoact2/scripts/export_molmoact2.py --checkpoint MolmoAct2-LIBERO-LeRobot \
    --hf MolmoAct2-LIBERO-hf --fast MolmoAct2-FAST-Tokenizer --stage assets --out engines
trtexec --onnx=onnx/vision/model.onnx --saveEngine=engines/vision.engine --stronglyTyped --noTF32
trtexec --onnx=onnx/prefix_a/model.onnx --saveEngine=engines/prefix_a.engine --stronglyTyped --noTF32 \
    --minShapes=input_ids:400,image_flag:400,cos:400x128,sin:400x128 \
    --optShapes=input_ids:500,image_flag:500,cos:500x128,sin:500x128 \
    --maxShapes=input_ids:640,image_flag:640,cos:640x128,sin:640x128
trtexec --onnx=onnx/prefix_b/model.onnx --saveEngine=engines/prefix_b.engine --stronglyTyped --noTF32 \
    --minShapes=hidden:1x400x2560,image_flag:400,cos:400x128,sin:400x128 \
    --optShapes=hidden:1x500x2560,image_flag:500,cos:500x128,sin:500x128 \
    --maxShapes=hidden:1x640x2560,image_flag:640,cos:640x128,sin:640x128
trtexec --onnx=onnx/context/model.onnx --saveEngine=engines/context.engine --stronglyTyped --noTF32 \
    --minShapes=keys:36x400x1024,values:36x400x1024 --optShapes=keys:36x500x1024,values:36x500x1024 \
    --maxShapes=keys:36x640x1024,values:36x640x1024
trtexec --onnx=onnx/step/model.onnx --saveEngine=engines/step.engine --stronglyTyped --noTF32 \
    --minShapes=context_k:36x1x400x8x96,context_v:36x1x400x8x96,encoder_mask:1x400 \
    --optShapes=context_k:36x1x500x8x96,context_v:36x1x500x8x96,encoder_mask:1x500 \
    --maxShapes=context_k:36x1x640x8x96,context_v:36x1x640x8x96,encoder_mask:1x640
python experimental_models/molmoact2/scripts/run_molmoact2_engines.py --engines engines --reference ref.npz
molmoact2_policy_server --engineDir engines [--port 5555]
```

The profiles cover prompts of 400-640 tokens: two cameras take 396 image tokens and a LIBERO instruction about
another 90-100. `molmoact2_prompt_dump` and `molmoact2_image_dump` check the host prompt and patches on their
own. The checkpoint is distributed under the Apache License 2.0.
