# LingBot-VLA 2.0

[LingBot-VLA 2.0](https://github.com/robbyant/lingbot-vla-v2) (Robbyant, 6B) pairs Qwen3-VL-4B with a 36-layer
action expert (hidden 768) that attends jointly to every language layer's keys and values, pi0-style: the prefix
(two camera views, the instruction in the Qwen3 chat template and 16 depth / video alignment queries) is encoded
once, then the expert integrates a 50 x 55 chunk from Gaussian noise at t = 1 down to 0 in 10 Euler steps. Every
expert layer is a sparse MoE (32 experts, top 4 by a sigmoid router, a shared expert) with adaRMSNorm on the flow
time. There is no official LIBERO checkpoint; the port is validated on the community fine-tune
[`openroboto-ai/lingbot-vla-v2-6b-libero`](https://huggingface.co/openroboto-ai/lingbot-vla-v2-6b-libero) (all four
suites, step 6000; it ships its training config, robot mapping and normalization statistics under `training/`).

It runs as four engines:

| Engine | Built from | Per call |
|---|---|---|
| `vision` | Qwen3-VL's ViT at the 256 x 256 training size, merger and the three deepstack mergers | once, 64 tokens per camera |
| `prefix_a` | the prefix (vision boundaries, image tokens, instruction, alignment queries) through language layers 0-17 with the deepstack adds, their K / V | once, 220 tokens |
| `prefix_b` | language layers 18-35, their K / V | once |
| `step` | one Euler step of the expert over the prefix K / V, with pi0.5's inpainting for real-time chunking | 10 times, as a CUDA graph |

`LingbotPolicy` reproduces the official `FeatureTransform` and processor on the host, checked bit-exact against
them on LIBERO observations: each frame is resized to the training size by an antialiased bilinear filter in float
(torchvision's) and cut into Qwen3-VL patches by the fast processor's path ((x - 127.5) / 127.5, two identical
temporal frames, merge-blocked order); the instruction is tokenized inside the chat template (identical ids on all
40 LIBERO instructions) and right-padded to 72 tokens; the multimodal RoPE ids follow `get_rope_index` with the
mask (padding at position 1, the expert's tokens after the largest valid one). The state and the actions map onto
the model's padded 55-dim layout by the robot config's slices and the training config's joint widths
(`state` / `actions` in `config.json`, with their q01 / q99 bounds).

**What the reference has to turn off.** The policy's constructor calls `torch.set_float32_matmul_precision("high")`,
so the official model in FP32 actually runs TF32 matmuls, and its MoE runs Triton group GEMMs (bf16 operands or TF32
dots). Under TF32 a model built on the CPU and one built on the GPU disagreed by 0.036 on the normalized chunk (the
size of the BF16 spread), and an FP32 re-implementation of the ViT landed 6.9e-3 from it. `lingbot_reference.py
--precision fp32` restores IEEE FP32 and evaluates the MoE densely in FP32; against that golden the same
re-implementation lands 2e-5 (ViT), 1.4e-4 (prefix K / V) and 1.2e-6 (the chunk after 10 steps). The checkpoint also asks for FlashAttention-2 for every sub-model; only the ViT
uses it (the prefix and the expert attend through the policy's own eager attention with an explicit mask), and its
SDPA fallback attends within each image as the varlen kernel does.

**Precision and the MoE.** The language model's hidden states reach 2.7e4 from layer 6 on, so every RMS norm runs
in FP32, as do RoPE and the attention (the official joint attention upcasts Q, K and V); the scaled scores stay
below 100. The router runs in FP32 from the FP16 hidden state (near-equal sigmoid scores otherwise flip the top-4).
All 32 experts are evaluated: over the 51 suffix tokens a layer routes to 28 of them on average, so a sparse kernel
would read nearly the same weights. The experts and the shared expert are concatenated along the intermediate
dimension and each expert's block is scaled by its routing weight before the (linear) down projection: three wide
GEMMs per layer. The step reads about 3.4 GB of weights, which bounds it on Orin's memory bandwidth.

**Real-time chunking.** As pi0.5's: after every Euler step, row r of the overlap moves towards
t * noise + (1 - t) * previous[start_row + r] by weight 1 on the frozen rows and an exponential ramp after them.

## Accuracy and speed

AGX Orin 64 GB, JetPack 6.2, three LIBERO-Spatial observations (256 x 256), against the official policy in true
FP32 (CUDA) with the same noise; the official BF16 server lands 0.026-0.056 from the same reference in robot actions:

| Stage | max \|Δ\| |
|---|---|
| Vision tokens / deepstack, from the official patches | 0.10-0.14 / 0.35-0.50 on values up to 20 / 64 |
| Prefix K / V of layers 0, 17, 35, from the official image tokens | 0.20 / 0.00, 0.06-0.07 / 0.18-0.20, 0.35-0.39 / 0.69-0.77 on values up to 266 / 115 |
| Normalized chunk (all 55 dims), chained | 0.036-0.056 (official BF16: 0.11-0.16) |
| `lingbot_policy_server` robot actions from raw frames | 0.004-0.007, cosine 0.999998 |

Swapping the two cameras moves the actions by up to 1.9. A call takes 520-675 ms on a LIBERO observation: vision
27 ms, language model 104 ms, 10 expert steps 383 ms, host 3 ms on a quiet GPU (median of 8); 675 ms (median of 20,
`libero_eval.py --bench-calls 20`) with the desktop drawing about 20% of the GPU, where every engine stage slows.

**Control rate.** `replay_robot.py` over TCP with raw SO101 frames at 30 Hz (the LIBERO checkpoint, state padded to
8; 300 ticks, the same GPU load): call median 681 ms, of which 669 ms in the server's engines, planner lag 20-22
ticks, no stalls; with `overlap=34, frozen=24` every switch
lands inside the frozen rows and jumps 0.0000. With 20 frozen rows the switches land past them (1.15, 2.02 without
RTC): the frozen rows must cover the lag.

LIBERO-Spatial, 10 tasks x 10 episodes, this harness's protocol (fixed initial states, 10 settle steps, seed 7,
256 x 256 renders, both views flipped, 5 of 50 rows per call, 220 steps, actions as returned; the fine-tune's own
evaluation records 5 rows per policy call and these step limits): **97%** (tasks 1 and 8 at 8 and 9 of 10). The
fine-tune reports 94% from 5 episodes per task.

```bash
# Official reference env: the lingbot-vla-v2 repo at the checkpoint's pinned commit (951475ae) on a CUDA PyTorch with
# transformers 4.57; --qwen3vl holds Qwen3-VL-4B-Instruct's config, processor and tokenizer files only
python experimental_models/lingbot/scripts/lingbot_reference.py --repo lingbot-vla-v2 \
    --checkpoint lingbot-vla-v2-6b-libero --qwen3vl Qwen3-VL-4B-Instruct --obs obs.npz --out ref.npz --capture
for stage in vision prefix_a prefix_b step; do
    python experimental_models/lingbot/scripts/export_lingbot.py --repo lingbot-vla-v2 \
        --checkpoint lingbot-vla-v2-6b-libero --qwen3vl Qwen3-VL-4B-Instruct --out onnx --stage $stage [--check ref.npz]
    trtexec --onnx=onnx/$stage/model.onnx --saveEngine=engines/$stage.engine --stronglyTyped --noTF32
done
python experimental_models/lingbot/scripts/export_lingbot.py ... --out engines --stage assets --tasks instructions.txt
python experimental_models/lingbot/scripts/run_lingbot_engines.py --engines engines --reference ref.npz
lingbot_inputs_dump engines out "task" image.png image2.png   # the host's ids, RoPE ids and patches
lingbot_policy_server --engineDir engines [--port 5555]
```

The FP32 model takes about 25 GB of the board's memory while a stage exports; it is built directly on the GPU (the
Orin shares DRAM, so a CPU copy moved over would double it) and loaded shard by shard. The released code and base
weights are under the Apache License 2.0; the fine-tune's original upload
(`Fisher-Wang/lingbot-vla-v2-6b-libero-full-step6000`, the same weights) is marked Apache-2.0.
