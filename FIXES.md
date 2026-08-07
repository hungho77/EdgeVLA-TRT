# Four fixes for garbled generation

This branch collects four independent defects that all present the same way — an engine that
builds cleanly and then generates nonsense — but have four different causes. Found and fixed
on **Jetson Thor (sm_110), JetPack 7.1, TensorRT 10.13.3.9**, against upstream `7f061f2`.

Related issues: [NVIDIA/TensorRT-Edge-LLM#151](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/151),
[#105](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/105).

Branches live at
[hungho77/TensorRT-Edge-LLM](https://github.com/hungho77/TensorRT-Edge-LLM/branches/all?query=fix).

| # | Symptom | Cause | Branch |
|---|---|---|---|
| 1 | FP16 engines emit gibberish | Myelin `fc_h_fusion` miscompiles; the existing workaround is version-gated to TRT ≥ 10.15 | `fix/fc-h-fusion-trt1013` |
| 2 | NVFP4 engines emit gibberish | CASK miscompiles when ≥ 2 epilogues fuse into one NVFP4 GEMM | `fix/nvfp4-cask-epilogue-trt1013` |
| 3 | INT4 AWQ emits degenerate text | The AWQ zero-point is folded into 4-bit weights and clamped — lossy | `fix/awq-asymmetric-zero-point` |
| 4 | InternVL3-9B "build failure" | InternLM2 layout unsupported; the exporter silently loads **nothing** | `fix/internvl3-internlm2-layout` |

Each branch is cut from `7f061f2` and stands alone, so they can be reviewed or taken
separately. This branch merges all four, plus this file: 645 insertions, 24 deletions,
5 files.

**Only fixes 1 and 2 are specific to this platform and TensorRT version.** Fixes 3 and 4 are
wrong everywhere — the AWQ zero-point fold breaks any asymmetric checkpoint on any GPU, and the
InternLM2 layout is unsupported regardless of backend. That is why the branch is named for the
shared symptom rather than for Thor.

---

## 1. FP16 — Myelin `fc_h_fusion` on TRT 10.13

`gate_proj` and `up_proj` share the `post_attention_layernorm` output, so Myelin fuses them
into a single horizontal GEMM — and that fusion is miscompiled on sm_110 at TRT 10.13.

The tree already knows about this. `applyMyelinCompileWorkarounds` sets
`-peep:fc_h_fusion=off`, but behind:

```cpp
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 15)
    if (maxBatchSize == 1) appendLunowudFlag(flags, "-peep:fc_h_fusion=off");
#endif
```

JetPack 7.1 ships **10.13.3.9**, which falls in the gap between the 10.13/10.14 branch (which
only applies `-peep:match_dual_gemm=off`) and this one, so the flag is never applied. The fix
widens the gate. **Only the version gate is wrong** — the `maxBatchSize == 1` condition is
correct and deliberate.

Same ONNX, only the flag changed:

| Engine | Without | With |
|---|---|---|
| Qwen2.5-0.5B FP16, text | `扫群AAST标的中共…` | `The capital of France is Paris.` |
| Qwen2.5-VL-3B FP16, image | `间STALLorest为自己跟%=…` | correct caption |

Costs nothing — `llm_bench`, 3B, batch 1, 100 iters: prefill 78.87 ± 5.71 → **74.14 ± 2.03 ms**
(and 3× less variance), decode 28.63 → 29.65 ms.

**Why FP8 was never affected:** Q/DQ nodes sit between the shared input and gate/up, so the
horizontal pattern cannot match. The bug is fusion-pattern-dependent, not precision-dependent.

Ruled out, each at the cost of a full rebuild — please don't re-suggest these: FP16 dynamic
range or GPU FP16 support (PyTorch FP16 is coherent on this GPU; peak activation 4 400 against
the 65 504 limit, 0/36 layers overflow); a missing unified-checkpoint conversion; violated
dependency pins (rebuilt with every pin exact — the ONNX graph *does* change, the output does
not); the attention kernel (both CuTe DSL FMHA and FMHA_v2 fail without the flag and both work
with it); engine size; and anything VL- or mRoPE-specific — a text-only 0.5B fails identically.

## 2. NVFP4 — CASK epilogue fusion

A different bug that looks identical, reported after the FP16 fix landed. `fc_h_fusion` is not
the cause: NVFP4 stays garbled with that fusion both on and off. That is what the Q/DQ argument
predicts — the NVFP4 graph has 792 `DequantizeLinear` and 144 `TRT_FP4DynamicQuantize` between
the shared layernorm output and the projections, so the horizontal pattern cannot match, the
same reason FP8 was always immune.

The decisive knob is one level below Myelin, in CASK:

```bash
export __LUNOWUD="-cask_fusion:max_num_epilogues=1"
```

Threshold test, same ONNX at batch 1, the value the only variable:

| `max_num_epilogues` | Output |
|---|---|
| 0 | correct |
| **1** | correct |
| 2 | garbage |
| 4 | garbage |

Corruption appears the moment CASK fuses **two or more** epilogues into a single NVFP4 GEMM.
Because the cap is engine-wide it must not be applied to FP16/FP8 graphs, which do not have the
bug, so it is gated on `maxBatchSize == 1` **and** an NVFP4 graph — detected by scanning the
ONNX proto (small; weights are external) for `TRT_FP4DynamicQuantize`.

**The defect is in generated epilogue code, not in tactic selection.** Holding `maxBatchSize 1`
fixed and varying only the cap, the correct build (`=1`) and the garbled build (`=2`) have
identical tactic pools — 700 relu / 120 swish each — and identical winning kernels; `diff` of
the chosen-kernel lists is empty. Two tempting readings of the tactic tables are therefore
wrong, and both were tested: this is not a GEMV or small-batch specialisation (the same
`cutlass3x_sm100_bstensorop_..._ue4m3xe2m1_...` family wins in both profiles), and bias+swish is
not the miscompiling epilogue (swish tracks batch size but not correctness). Source dumping does
not reach it either — `-cask_fusion:dump_source=on` yields only Myelin's NVRTC pointwise
kernels, because the fused NVFP4 GEMM is a precompiled CASK kernel inside `libnvinfer`.

**The fix costs nothing, despite appearances.** The uncapped 38.38 ms prefill is a *miscompiled*
kernel doing the wrong work; it is not a baseline. Every correct configuration lands at 51–58 ms:

| Engine (all run at batch 1) | Prefill (1024 tok) | Decode (pastKV 1024) | Output |
|---|---|---|---|
| `maxBatchSize 1` + cap | **50.93 ± 0.34 ms** | 14.00 ± 0.47 ms | correct |
| `maxBatchSize 2`, no flags | 56.84 ± 1.03 ms | **12.97 ± 0.12 ms** | correct |
| `maxBatchSize 4`, no flags | 57.54 ± 2.10 ms | 13.43 ± 0.15 ms | correct |
| `maxBatchSize 1`, no flags | 38.38 ± 2.60 ms | 13.68 ± 0.77 ms | **garbage** |

Holding the engine fixed at `maxBatchSize 2` and varying only the cap isolates what correct
multi-epilogue fusion is actually worth: 54.76 ± 2.50 vs 53.42 ± 2.15 ms prefill — identical
within noise. It buys nothing measurable, so capping it costs nothing. The fusion does fire
there (TRT reports 16 118 784 vs 8 610 816 bytes of activation memory on the same graph), it
simply does not pay.

**`--maxBatchSize 2` is an equally valid workaround** needing no flag at all. And do **not**
build a dense NVFP4 GEMM plugin to "win back" the 38 %: a plugin gets zero epilogue fusion, so
its ceiling is the `max_num_epilogues=0` row, which is slower than this fix.

## 3. INT4 AWQ — asymmetric zero-points clamped away

Reported against `Qwen/Qwen2.5-3B-Instruct-AWQ`. Not the Myelin family: AWQ's GEMM runs inside
`int4GroupwiseGemmPlugin`, so fusion never reaches it.

`repack_awq_to_plugin` folded the zero-point into the weight nibbles and clamped:

```python
nibbles = (nibbles - zeros_expanded + 8).clamp(0, 15)   # silently lossy
```

Within one group `nibble - qzero` spans the 16 values `[-qzero, 15-qzero]`, while the kernel's
fixed zero-point pins the representable window to `[-8, 7]`. The two coincide only at
`qzero == 8`, so any checkpoint written with `zero_point: true` pushes part of every group out
of range. Measured on that checkpoint (group size 128):

| | |
|---|---|
| zero-point range | 1–14, **48–66 % of groups away from 8** |
| span after folding | **[-5, 21]** (representable: `[0, 15]`) |
| weights clamped | **0.63 %**, error up to 7 quantization levels |
| output shift, one projection | **4–6 %** (cosine 0.9983) |

The truncated values are the largest-magnitude weights in each group — precisely what AWQ exists
to preserve.

**The fix splits the zero-point out of the GEMM instead of folding it:**

```
Σᵢ xᵢ(qᵢ − z_g)s_g  =  Σᵢ xᵢ(qᵢ − 8)s_g  +  Σ_g (8 − z_g)s_g · Σ_{i∈g} xᵢ
                       └─ kernel, unshifted nibbles ─┘   └─ run-time correction ─┘
```

`repack_awq_to_plugin` returns the constant factor `(8 − zeros) · scales`; `AWQLinear.forward`
adds it back from the group-wise sums of its input. Algebraically exact — **cosine 1.00000000**
against correct AWQ dequantization, residual 0.01 % from fp16 rounding of the correction, versus
5.97 % before — and **no kernel change**: the plugin still takes three inputs.

Cost is about 0.8 % of the layer's FLOPs, and less in practice: the ONNX optimizer dedupes the
reductions to **144** `ReduceSum` nodes for 252 corrections, because q/k/v share one layernorm
output and gate/up share the other. Symmetric checkpoints are unaffected — the correction is
all zeros.

Verified end to end through export → build → generate, the repacking code the only difference
(2.02 GB engine, batch 1, greedy, 80 tokens):

| | *"Explain in three sentences why the sky appears blue."* |
|---|---|
| before | `the sky appears blue because the sky appears blue. In three sentences, I can explain that…` — degenerate loop |
| after | `…a phenomenon called Rayleigh scattering, where sunlight is scattered in all directions by gases and small particles…` — correct |

Requantizing each group symmetrically was also tried and is **worse** (13 % output error),
because it degrades every weight rather than 0.6 %.

## 4. InternVL3-9B — unsupported text backbone, silently

9B and 9B-Instruct are the only InternVL3 sizes whose LLM is **InternLM2** rather than Qwen2.5,
and correspondingly the only ones with no `-hf` release — exactly the set that fails.

| | `-hf` sizes ✅ | 9B ❌ |
|---|---|---|
| Attention | `self_attn.q_proj/k_proj/v_proj/o_proj` | `attention.wqkv` (fused), `attention.wo` |
| MLP | `mlp.gate_proj/up_proj/down_proj` | `feed_forward.w1/w2/w3` |
| Norm | `input_layernorm`, `post_attention_layernorm` | `attention_norm`, `ffn_norm` |
| Embed / head | `embed_tokens`, `lm_head` | `tok_embeddings`, `output` |

**The export does not fail — it silently emits random weights.** It returns exit 0 and writes a
complete ONNX; the loader never matches the checkpoint keys and leaves the module
default-initialized, with no "keys not loaded" warning. On a 2-layer stand-in the exported
`embedding.safetensors` has `std = 1.0002` (fresh `randn`) against the checkpoint's `0.0200`.
**The reported "build failure" is a downstream symptom of exporting a randomly-initialized
graph.** Failing loudly when a checkpoint populates none of the model's parameters would be
worth doing independently of this script.

`tensorrt_edgellm/scripts/convert_internlm2_internvl.py` rewrites only the decoder into the
standard layout and retags `llm_config` as `llama` — InternLM2 *is* Llama with GQA (RMSNorm,
SwiGLU, RoPE, no attention bias), so the mapping is exact rather than approximate. The vision
tower is deliberately untouched: `vision_model.*` / `mlp1.*` under `model_type: internvl_chat`
is already supported, since `export.py` overrides the `intern_vit_6b` vision model_type to
`internvl`.

```bash
python -m tensorrt_edgellm.scripts.convert_internlm2_internvl <src> <dst>
python -m tensorrt_edgellm.scripts.convert_internlm2_internvl <src> <dst> --free-src
        # deletes each source shard once converted; peak disk is one shard, not two
        # 18 GB checkpoints
```

`wqkv` packs, per KV head, `num_heads // num_kv_heads` query heads followed by one K and one V,
so viewing it as `[num_kv, q_per_kv + 2, head_dim, hidden]` and slicing `[:, :q_per_kv]` /
`[:, -2]` / `[:, -1]` recovers the three projections. Flattening q in that order gives query
head `i` belonging to KV group `i // q_per_kv`, the grouping standard GQA assumes. Verified
**bit-exact** (`torch.equal`, max abs diff 0.0) against the InternLM2 reference forward on the
real layer-0 `wqkv`.

Confirmed on the full 18.28 GB checkpoint: 685 → 781 tensors (+96 = 48 layers whose `wqkv`
becomes q/k/v), export returns 0, the exported embedding matches the checkpoint (`std` 0.0198
both sides, `torch.allclose` True), and the resulting 15.61 GB FP16 engine generates correctly:

```
The capital of France is Paris.<|im_end|>

The sky appears blue due to a phenomenon called Rayleigh scattering, where shorter
wavelengths of light, such as blue, are scattered more by the Earth's atmosphere than
longer wavelengths like red...
```

**Two tokenizer traps sit between the conversion and a working engine.** Neither is a conversion
bug; both are silent, and the second one costs a full engine build to discover:

1. *No `tokenizer.json`.* The checkpoint ships only a SentencePiece `tokenizer.model` plus
   `InternLM3Tokenizer` remote code, while `llmBuilder.cpp` copies
   `{tokenizer_config.json, tokenizer.json, processed_chat_template.json}`. Generate a fast
   tokenizer first — `AutoTokenizer.from_pretrained(..., trust_remote_code=True,
   use_fast=True).save_pretrained(...)`, which needs `sentencepiece` and takes ~15 min for the
   128k vocab — and drop it next to the ONNX before building.
2. *A wrong chat template looks exactly like a broken model.* If `AutoTokenizer` cannot load the
   directory, `checkpoint_utils` falls back to `write_fallback_processed_chat_template`, a
   generic `"User: " / "Assistant: "` format. InternVL3-9B is ChatML
   (`<|im_start|>role\n…<|im_end|>`), so every prompt comes out malformed and the engine emits
   degenerate text — the first build here printed `1. 2. 3. 4. 5. …`, which reads like a
   quantization or conversion failure and is neither. The converter copies `tokenizer.model` and
   `tokenization_internlm3.py` for exactly this reason.

`processed_chat_template.json` is a plain runtime file that the engine never bakes in, so a
wrong one is fixable in seconds — regenerate with `process_chat_template(model_dir, out_dir)`
and copy it into the engine directory. **Do not rebuild the engine for this.** Check
`roles.user.prefix` before blaming the weights for garbled output.

---

## Applying the fixes

### All four at once

```bash
git remote add hungho77 https://github.com/hungho77/TensorRT-Edge-LLM.git
git fetch hungho77 fixes/garbled-output
git checkout -b fixes/garbled-output hungho77/fixes/garbled-output
```

Or merge them into a branch of your own:

```bash
git merge hungho77/fixes/garbled-output
```

### One at a time

Each branch is a single commit on top of `7f061f2`. Fixes 3 and 4 touch files nothing else
does; fixes 1 and 2 both edit `applyMyelinCompileWorkarounds` in `llmBuilder.cpp`, but in
adjacent, distinct blocks, so any order and any combination applies cleanly — both orders were
tested and produce a `llmBuilder.cpp` identical to `fixes/garbled-output`. Cherry-pick the ones
you want:

```bash
git fetch hungho77 fix/fc-h-fusion-trt1013 \
                   fix/nvfp4-cask-epilogue-trt1013 \
                   fix/awq-asymmetric-zero-point \
                   fix/internvl3-internlm2-layout

git cherry-pick hungho77/fix/awq-asymmetric-zero-point     # e.g. AWQ only
```

| Fix | Branch | Files touched |
|---|---|---|
| 1 — FP16 | `fix/fc-h-fusion-trt1013` | `cpp/builder/llmBuilder.cpp` |
| 2 — NVFP4 | `fix/nvfp4-cask-epilogue-trt1013` | `cpp/builder/llmBuilder.cpp` |
| 3 — AWQ | `fix/awq-asymmetric-zero-point` | `tensorrt_edgellm/checkpoint/repacking.py`, `tensorrt_edgellm/models/linear.py` |
| 4 — InternVL3 | `fix/internvl3-internlm2-layout` | `tensorrt_edgellm/scripts/convert_internlm2_internvl.py` (new file) |

### As patch files

If you would rather not add a remote, the four patches are attached to the issue and apply
with `git am` on a clean `7f061f2` — verified to reproduce `fixes/garbled-output` exactly:

```bash
git am < fc_h_fusion_trt1013.patch
git am < nvfp4_cask_epilogue_trt1013.patch
git am < awq_asymmetric_zero_point.patch
git am < internvl3_internlm2_layout.patch
```

### Which ones do you need?

- **Fix 3 (AWQ)** — take it regardless of platform or TensorRT version. It is wrong for every
  asymmetric AWQ checkpoint on every GPU. Pure Python; no rebuild of the C++ runtime needed.
- **Fix 4 (InternVL3)** — likewise platform-independent, and a new file only. No rebuild.
- **Fixes 1 and 2** — needed on TensorRT 10.13/10.14. On TRT ≥ 10.15 fix 1 is redundant
  (upstream already applies the flag); fix 2 has not been tested there. Both are C++ and do
  require rebuilding the plugin and `llm_build`.

So if you are only hitting the AWQ or InternVL3 bug, you can cherry-pick those two and skip
the rebuild entirely.

## Building

Only needed for fixes 1 and 2.

```bash
cmake -B build -S . \
  -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-thor -DCMAKE_CUDA_ARCHITECTURES=110a \
  -DTRT_PACKAGE_DIR=/usr -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The toolchain file is mandatory. Without it, `cpp/CMakeLists.txt` does not add SM 101 to the
required FMHA list — it only does so when 110 is in `CMAKE_CUDA_ARCHITECTURES`, since "SM 110
uses SM 101 FMHA cubins at runtime" — and defines `EXCLUDE_SM_101` instead, so the cubins vanish
and inference dies with `There must be one kernel to implement the MHA`. Verify before
compiling; both must hold:

```bash
grep -rl EXCLUDE_SM_101 build/cpp/CMakeFiles/*/flags.make | wc -l          # must be 0
strings build/libNvInfer_edgellm_plugin.so | grep -c '_fp16.*_sm101'       # must be ~324
```

Note that the toolchain sets `CMAKE_CUDA_ARCHITECTURES` as a *non-cache* variable, so it does
not appear in a correctly-configured `CMakeCache.txt` — and a stale entry from a bad configure
silently wins. Delete the build directory rather than reconfiguring over it.
