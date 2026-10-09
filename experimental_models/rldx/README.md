# RLDX-1

[RLDX-1](https://github.com/RLWRLD/RLDX-1) (RLWRLD) pairs a Qwen3-VL-8B backbone, truncated to its first 18 decoder
layers and read through 64 learned cognition tokens, with MSAT, a flow-matching action transformer (4 dual-stream
and 8 single-stream blocks, 4 Euler steps). The public LIBERO checkpoint is
[`RLWRLD/RLDX-1-FT-LIBERO`](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) (trained on all four suites; its
memory module and physics stream are off). Each call sees, per camera, the frames at t-6, t-4, t-2 and t; the
backbone's video-token compression (VTC) replaces the three past frames' tokens by their mean at layer 4.

It runs as four engines:

| Engine | Built from | Per call |
|---|---|---|
| `visual/` | Edge-LLM's Qwen3-VL ViT (deepstack outputs), exported from a Qwen3-VL checkpoint holding RLDX's ViT weights | 8 images (4 frames x 2 views), 512 merged tokens |
| `llm_a` | layers 0-3 over the full prompt (~616 tokens) with the deepstack adds | once |
| `llm_b` | the compression, layers 4-17, the final norm, the last 64 rows (the cognition features) | once |
| `action` | one MSAT Euler step, x_{t+dt} = x_t + dt * strength * v | 4 times, as a CUDA graph |

`RldxPolicy` builds the prompt on the host and caches it per task: the formalized instruction (lowercase, word
characters) through the Qwen tokenizer (ids identical to HF's on every LIBERO instruction plus accented, CJK and
Cyrillic text), the image pads, interleaved MRoPE tables in FP32 (equal to the official `get_rope_index` within
FP32 rounding) and the compression's gather indices. Frames go through the checkpoint's AspectAreaResizeAndCrop
(OpenCV INTER_AREA; 256 x 256 is unchanged) and Edge-LLM's Qwen3-VL preprocessing; the state is min-max
normalized with the q01 / q99 statistics and clipped; the chunk is unnormalized the same way.

**What FP16 could not hold.** The single-stream blocks of MSAT carry the projected cognition tokens through their
residual stream with three channels near 6.2e4 (FP16's limit is 65504): that stream, the projection feeding it and
every LayerNorm run in FP32, the blocks' GEMMs in FP16. The language model's residual stream reaches 1.2e4 and
its key norm carries gains up to 34, so attention applies RoPE and computes QK^T and the softmax in FP32. Those
FP32 matrix products must not use TF32: the engines are built with `--noTF32`, which took the cognition features
from 0.86% to 0.27% relative error at the same speed. MSAT's complex-number RoPE is rewritten as a real rotation
of the interleaved pairs, with the official position ids (the double-stream blocks number the state and action
tokens from 1, the single-stream ones from 2).

**Real-time chunking.** The official RTC modes do not apply (`trained` needs a checkpoint trained for it, `guided`
needs autograd), so the server uses GR00T's: the overlap rows start from the previous chunk's normalized rows and
their velocity is scaled by 0 on the frozen rows, then by an exponential ramp.

**Frame history.** The server is stateless, as the official one: a request carries the past frames named
`camera@offset` (`front_view@-6`, ...), and its `ready` line lists `frame_history`. A missing past frame takes the
camera's oldest frame in the request, as the official evaluation buffer repeats the first observation. The
offsets are in the training data's steps; `AsyncChunkedController` keeps the frames per tick when a server
advertises a history, so the control loop should run at the dataset's rate.

## Accuracy and speed

AGX Orin 64 GB, JetPack 6.2, against the official policy in FP32 on the board's GPU, two LIBERO observations with
moving frames and instructions of 17 and 3 words, the same initial noise:

| Stage | Relative error | Official BF16 vs FP32 |
|---|---|---|
| Visual engine (from PNG, C++ preprocessing) | 2.4% (deepstack 0.7-1.9%) | 2.3-6.3% |
| Hidden state after layer 3 | 0.45% | 1.8-3.2% |
| Cognition features, chained | 0.5-0.6% | 0.4% |
| `rldx_policy_server` actions, max \|Δ\| | 0.0085 / 0.0035 (gripper commands identical) | 0.0085 / 0.0066 |

Inline frames give bit-identical actions to the same images by path. 480 x 640 frames (resized to 192 x 256,
6 x 8 tokens each, as the official preprocessing does) match the official policy within 0.0061. A call takes
about 300 ms on an idle board: visual 95 ms (preprocessing and the ViT on 8 images), language model 115 ms, 4
action steps 80 ms.

**Control rate.** With one request in flight, a chunk is adopted one policy latency after it was planned and the
next arrives one latency later, so a loop runs without stalls only while twice the latency in ticks stays within
the 16-row chunk: about 300 ms fits 20 Hz (6 ticks; LIBERO's rate, which the history offsets count in) but not
30 Hz (10 ticks). `replay_robot.py` over TCP with raw frames and the frame history runs the protocol end to end.

LIBERO-Spatial, 10 tasks x 10 episodes, this harness's protocol (fixed initial states, 10 settle steps, 220 steps)
with the checkpoint's conventions (8 of 16 rows per call, gripper `sign(2 g - 1)`): **97%** (failures on tasks
1, 4 and 5, all at the step limit; policy call median 297 ms). The paper
reports 98.6% on LIBERO-Short (the Spatial, Object and Goal average; no per-suite number) over 50 episodes per task,
with the official rollout's random resets, no settle steps and a 720-step limit.

```bash
# Reference env: RLDX's code on a CUDA PyTorch with transformers 4.57.0 (its pinned version)
RLDX_ATTN_IMPL=sdpa PYTHONPATH=RLDX-1 python experimental_models/rldx/scripts/rldx_reference.py \
    --checkpoint RLDX-1-FT-LIBERO --obs obs.npz --out ref.npz --precision fp32
# Visual: RLDX's ViT as a Qwen3-VL checkpoint (RLDX-1-VLM's config, processor and tokenizer files plus the
# checkpoint's backbone.qwen_model.model.visual.* tensors renamed model.visual.*), through Edge-LLM
tensorrt-edgellm-export visual_ckpt onnx --skip-llm
visual_build --onnxDir onnx/visual --engineDir engines --minImageTokens 16 --maxImageTokens 2048 \
    --maxImageTokensPerImage 256
# Language model halves, action step, config.json and tokenizer
for stage in llm_a llm_b action; do
    PYTHONPATH=RLDX-1 python experimental_models/rldx/scripts/export_rldx.py --checkpoint RLDX-1-FT-LIBERO \
        --vlm RLDX-1-VLM --stage $stage --out onnx
done
PYTHONPATH=RLDX-1 python experimental_models/rldx/scripts/export_rldx.py --checkpoint RLDX-1-FT-LIBERO \
    --vlm RLDX-1-VLM --stage assets --out engines
trtexec --onnx=onnx/llm_a/model.onnx --saveEngine=engines/llm_a.engine --stronglyTyped --noTF32 \
    --minShapes=input_ids:256,visual:128x4096,deepstack:3x128x4096,visual_index:256,cos:320x128,sin:320x128 \
    --optShapes=input_ids:560,visual:512x4096,deepstack:3x512x4096,visual_index:560,cos:624x128,sin:624x128 \
    --maxShapes=input_ids:720,visual:1024x4096,deepstack:3x1024x4096,visual_index:720,cos:784x128,sin:784x128
trtexec --onnx=onnx/llm_b/model.onnx --saveEngine=engines/llm_b.engine --stronglyTyped --noTF32 \
    --minShapes=hidden:1x320x4096,pool:320,keep_index:100,cos:100x128,sin:100x128 \
    --optShapes=hidden:1x624x4096,pool:624,keep_index:230,cos:230x128,sin:230x128 \
    --maxShapes=hidden:1x784x4096,pool:784,keep_index:400,cos:400x128,sin:400x128
trtexec --onnx=onnx/action/model.onnx --saveEngine=engines/action.engine --stronglyTyped --noTF32
PYTHONPATH=RLDX-1 python experimental_models/rldx/scripts/run_rldx_engines.py --engines engines \
    --vlm RLDX-1-VLM --reference ref.npz --state state.npz   # stage by stage
rldx_policy_server --engineDir engines [--port 5555]
```

`tokenizer.json` is generated from RLDX-1-VLM's `vocab.json` / `merges.txt` by the HF fast tokenizer. The profiles
cover prompts of 256-720 tokens: 256 x 256 LIBERO frames give about 550, 480 x 640 frames (6 x 8 tokens each)
about 420. `rldx_prompt_dump` and `rldx_vision_dump` check the host tables and the visual engine on their own. The
checkpoint is distributed under the
[RLWRLD Model License v1.0](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO/blob/main/LICENSE.md), which carries
over to the engines built from it.
