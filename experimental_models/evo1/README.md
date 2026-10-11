# Evo-1

[Evo-1](https://github.com/MINT-SJTU/Evo-1) (MINT-SJTU, 0.77B) pairs InternVL3-1B, its language model truncated to
the first 14 of Qwen2.5-0.5B's 24 layers, with a cross-modulated flow-matching head: 8 blocks of cross-attention
from 50 action tokens to the language model's hidden states plus a state token, a pooled projection of the whole
chunk to a velocity, and 50 Euler steps from uniform noise. The LIBERO checkpoint is
[`MINT-SJTU/Evo1_LIBERO`](https://huggingface.co/MINT-SJTU/Evo1_LIBERO) (all four suites, a DeepSpeed
`mp_rank_00_model_states.pt`); the server follows the official `Evo_1/scripts/Evo1_server.py`.

It runs as four engines:

| Engine | Built from | Per call |
|---|---|---|
| `vision` | InternViT-300M at 448 x 448, pixel shuffle and the projector | once, 256 tokens per camera |
| `prefix` | token embeddings with the image tokens in place, the 14 layers and the final norm over 1024 positions | once |
| `context` | the state token, and every block's cross-attention keys and values over the 1025 context rows | once |
| `step` | one Euler step, x_{t+dt} = x_t + dt * strength * v | 50 times, as a CUDA graph |

`Evo1Policy` reproduces the official server on the host. Each frame is resized to 448 x 448 with OpenCV's bilinear
filter and its red and blue channels are swapped: the official server converts the RGB frames its clients send with
`cv2.COLOR_BGR2RGB` (`swap_red_blue` in `config.json`). The model always reads three views; the missing ones are
black images whose 256 tokens the language model's attention mask excludes, and their vision tokens are computed
once at start-up. The prompt is `Image-i: <img>` + 256 image tokens + `</img>\n` per view, then the stripped
instruction, right-padded to 1024 tokens (ids identical to the official slow tokenizer's on all 40 LIBERO
instructions). The state is mapped to [-1, 1] by the checkpoint's min / max, clamped and zero-padded to 24; the
first action dims of the chunk are unnormalized the same way.

**The padded positions matter.** The head cross-attends to all 1024 + 1 positions with no key mask, so the hidden
states of the padding and of the black view feed the actions. The official setup installs flash-attn, and with it
InternVL3's language model runs FlashAttention-2, which gives a position the attention mask excludes no attention
output at all (only its residual and MLP). Without flash-attn the model falls back to eager attention, where such a
position attends to the earlier tokens instead: that moves the actions by up to 0.73 on LIBERO observations, and the
first four LIBERO-Spatial episodes all failed at the step limit. The export reproduces FlashAttention-2's result (`--attention flash`, the default; the
reference script takes the same flag).

**Precision.** RMSNorm, LayerNorm and every softmax run in FP32; QK^T is computed in FP16 with the scale on Q. The
step engine reads about 240 MB of weights per step (the 8 blocks and the 44800 x 896 pooled projection), which
bounds the 50 steps on Orin's memory bandwidth.

**Real-time chunking.** As GR00T's: the overlap rows start from the previous chunk's normalized rows from
`start_row`, and their velocity is scaled by 0 on the frozen rows, then by an exponential ramp.

## Accuracy and speed

AGX Orin 64 GB, JetPack 6.2, three LIBERO-Spatial observations rendered at 448 x 448, against the official policy
in FP32 (CUDA) with the same initial noise; the official BF16 server lands 0.018-0.024 from the same reference:

| Stage | max \|Δ\| |
|---|---|
| Vision tokens, from the official pixel values | 0.05 on values up to 8 |
| Language-model hidden states, from the official image tokens | 0.26-0.31 on values up to 185 |
| Normalized chunk, context and steps on the official hidden states | 0.0012-0.0013 |
| Normalized chunk, chained | 0.003-0.007 |
| `evo1_policy_server` robot actions from raw frames (PNG) | 0.003-0.006 on a ±0.94 range, cosine 0.999996 |
| The same on two SO101 frame pairs (360x640 and 480x640, resized by the runtime) | 0.0018-0.0025 |

Swapping the two cameras moves the server's actions by up to 0.95, and not swapping red and blue by up to 0.50.
A call takes about 250 ms (median of 10 on a LIBERO observation): vision 70 ms for two cameras, language model
33 ms, head context and 50 steps 133 ms, host 3 ms.

**Control rate.** `replay_robot.py` over TCP with raw SO101 frames at 30 Hz (the LIBERO checkpoint, state padded to
8; `overlap=20, frozen=14`, 300 ticks): call median 255 ms, planner lag 7-8 ticks, no stalls, switch jump 0.0000
with RTC and 1.01 without.

LIBERO-Spatial, 10 tasks x 10 episodes, this harness's protocol (fixed initial states, 10 settle steps) with the
official client's conventions (448 x 448 renders, both views flipped, 14 of 50 rows per call, 350 steps, seed 42):
**87%** (tasks 5 and 8 at 6 and 7 of 10). With the November 2025 server's sampler, which the published numbers came
from (`--sampler 2025-11`: 32 steps, time rows `int(t * 1000)`, as the checkpoint was trained), **89%** at about
200 ms per call. The paper reports 92.7% for LIBERO-Spatial (94.8% over the four suites), averaged over five runs of
10 episodes per task.

```bash
# Official reference env: Evo-1's code on a CUDA PyTorch with transformers 4.57 (timm is stubbed); --vlm is
# OpenGVLab/InternVL3-1B (tokenizer and model code; the checkpoint carries every weight)
python experimental_models/evo1/scripts/evo1_reference.py --repo Evo-1 --checkpoint Evo1_LIBERO \
    --vlm InternVL3-1B --obs obs.npz --out ref.npz        # obs: agentview_i, wrist_i (448 x 448), state_i, task_i
python experimental_models/evo1/scripts/export_evo1.py --repo Evo-1 --checkpoint Evo1_LIBERO \
    --vlm InternVL3-1B --out evo1_onnx [--check ref.npz] [--tasks instructions.txt] [--sampler 2025-11]
trtexec --onnx=evo1_onnx/vision.onnx --saveEngine=engines/vision.engine --stronglyTyped \
    --minShapes=pixels:1x3x448x448 --optShapes=pixels:2x3x448x448 --maxShapes=pixels:3x3x448x448
for g in prefix context step; do
    trtexec --onnx=evo1_onnx/$g.onnx --saveEngine=engines/$g.engine --stronglyTyped
done
cp evo1_onnx/config.json evo1_onnx/tokenizer.json engines/
python experimental_models/evo1/scripts/run_evo1_engines.py --engines engines --reference ref.npz
evo1_tokenize engines < instructions.txt               # the runtime's token ids, one JSON array per line
evo1_policy_server --engineDir engines [--port 5555]
```

`--tasks` makes the export check that every listed instruction tokenizes as the image blocks followed by the
instruction alone, which is how the runtime builds the prompt. Evo-1's code is under the MIT License; the
checkpoint's repository states no license.
