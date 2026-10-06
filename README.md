<div align="center">

# EdgeVLA

**Vision-Language-Action inference on NVIDIA Jetson, built on NVIDIA TensorRT Edge-LLM**

[![upstream](https://img.shields.io/badge/built%20on-TensorRT%20Edge--LLM%200.11.0-76b900)](https://github.com/NVIDIA/TensorRT-Edge-LLM)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](LICENSE)
[![platform](https://img.shields.io/badge/tested-AGX%20Orin%20JetPack%206.2-lightgrey)](JETPACK6.md)

</div>

EdgeVLA is a fork of [NVIDIA TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) focused on
Vision-Language-Action (VLA), vision-language navigation (VLN) and world-action models on edge devices. It keeps
Edge-LLM's export → engine build → C++ runtime pipeline and adds VLA models, VLA-oriented runtime work, and
support for platforms upstream no longer targets.

> EdgeVLA is an independent project. It is not an NVIDIA product and is not endorsed by NVIDIA. NVIDIA and
> TensorRT are trademarks of NVIDIA Corporation. Code from upstream keeps its NVIDIA copyright headers and is
> distributed under the Apache License 2.0.

## What this fork adds

| Area | Status | Details |
|---|---|---|
| InternVLA-N1-DualVLN (VLN) | Export → build → inference verified on AGX Orin (JetPack 6.2) | [Guide](docs/source/user_guide/examples/vla/internvla_n1.md) · [runtime](experimental_models/internvla_n1/README.md) |
| JetPack 6 (CUDA 12.6, TensorRT 10.3) | Runs 0.11 on AGX Orin | [JETPACK6.md](JETPACK6.md) |
| Exact asymmetric INT4 AWQ | Engine output matches the exactly dequantized checkpoint token-for-token | [FIXES.md](FIXES.md) |
| NVFP4 AWQ `pre_quant_scale`, NVFP4 CASK epilogue cap (TRT 10.13/10.14) | Fixed; needs Thor to verify | [FIXES.md](FIXES.md) |
| InternLM2-backed InternVL3 checkpoints | Converter | `tensorrt_edgellm/scripts/convert_internlm2_internvl.py` |
| GR00T N1.7 policy | Backbone through Edge-LLM, action head with cross-attention K/V cached per call; 114.5 ms per action chunk on Orin, matching the official model | [below](#gr00t-n17) |
| VLA replan caching | Encoder-cache partial hits and KV prefix reuse with tail-only hidden capture; ~2× faster InternVLA-N1 replans | [below](#kv-cache-reuse-for-vla-control-loops) |

Upstream VLA support (experimental pi0.5, Alpamayo, Cosmos3-Edge policy) is unchanged and documented under
[docs/source/user_guide/examples/vla](docs/source/user_guide/examples/vla/index.md).

### Measured on AGX Orin 64 GB, JetPack 6.2

InternVLA-N1-DualVLN, FP16 System 2 (Qwen2.5-VL-7B), BF16 System 1, `--ticks 40 --cadence 4`:

| Metric | Value |
|---|---|
| First plan | 181 ms (System 2 text-only, as in the CLI example) |
| System-1 tick, mean / worst | 188 ms (5.3 Hz) / 314 ms |

The LLM engine was built with `--maxInputLen 2048 --maxKVCacheCapacity 2560` to fit beside the other engines.

## KV-cache reuse for VLA control loops

A VLA replans every few control steps with a prompt that mostly repeats: the same instruction, history frames it
has already seen, and one new frame. Upstream 0.11 recomputed all of it: the encoder cache only restored when every
image of a request was cached, and hidden-state capture bypassed the context cache. This fork adds:

- **Encoder-cache partial hits**: only uncached images go through the vision encoder; the rest are assembled from
  the cache. On by default (`encoderEmbeddingCacheBudgetBytes`, 256 MiB).
- **Tail-only hidden capture with context reuse**: a request that reads only its last N hidden states sets
  `LLMGenerationRequest::hiddenCaptureTailTokens = N`; the context cache then restores matching prefix pages and
  always recomputes those N positions. Enable the cache with `--enableContextReuse` (InternVLA-N1 server and
  `internvla_n1_s2_bench`). Attention-only, non-speculative deployments; no engine rebuild needed.

InternVLA-N1 System-2 replans on AGX Orin, simulated 30-step InternNav episode (8 `linspace` history frames plus the
current frame, 384×384, about 1878 prompt tokens), measured with `internvla_n1_s2_bench`:

| Configuration | Mean replan | vs. no cache |
|---|---|---|
| No cache | 1734 ms | — |
| Encoder partial hits | 1205 ms | −30% |
| Encoder partial hits + KV prefix reuse | 749 ms | −57% |

KV reuse restores 41% of prompt tokens over the episode (256–768 per replan). `z_latents` are bit-identical to the
no-cache run in every configuration. Another GPU job shared the board during part of the measurement, so absolute
times vary between runs; the ordering held in every run.

The same comparison on real navigation input, R2R trajectory 1304 in Matterport3D scene `gZ6f7yhEvPG` rendered by
VLN-PE (102 frames, its R2R instruction; prepared with
[`prepare_vlnpe_episode.py`](experimental_models/internvla_n1/examples/prepare_vlnpe_episode.py)), averaged over the
94 nine-frame replans:

| Configuration | Mean replan | vs. no cache |
|---|---|---|
| No cache | 2009 ms | — |
| Encoder partial hits | 1429 ms | −29% |
| Encoder partial hits + KV prefix reuse | 938 ms | −53% |

KV reuse restores 38% of the episode's prompt tokens; `z_latents` are again bit-identical to the no-cache run.

Reuse applies to VLAs with a **causal** VLM backbone (InternVLA-N1, GR00T, Alpamayo). Models whose image+text prefix
attends bidirectionally (pi0.5, SmolVLA) can only skip repeated vision encoding.

### Roadmap

1. **InternVLA-N1 pilot**: done (above), measured on synthetic and real rendered navigation episodes.
2. **GR00T N1.7**: done ([above](#gr00t-n17)). Next: INT8 DiT weights, CUDA graphs, RTC chunking and SO101
   pre/post-processing; then a shared `vla/` layer (observation encoder, causal backbone on the core runtime, action
   head, dual-rate scheduler) extracted from InternVLA-N1 and GR00T.
3. **Bidirectional-prefix VLAs** (pi0.5, SmolVLA): encoder cache, persistent CUDA graphs, prefix KV pool in
   the core runtime.

## GR00T N1.7

NVIDIA GR00T N1.7 (SO101 fine-tune) runs as Edge-LLM backbone plus a vendored action head
([`experimental_models/gr00t`](experimental_models/gr00t)):

- **Backbone**: GR00T's Cosmos-Reason2 (Qwen3-VL) truncated to `select_layer` = 16 layers, exported through the
  regular Qwen3-VL path with `emit_hidden_states: "post_norm"`, so the engine returns every position's layer-16
  output after the final norm. Under transformers 4.57, the version GR00T N1.7 was saved with, `hidden_states[-1]`
  is that post-norm tensor; the pre-norm one has cosine 0.07 to it.
- **Action head**, three engines for one embodiment: `vl_prep` (vlln, VL self-attention and the K/V of all 16 DiT
  cross-attention blocks, once per call), `state_encoder`, and `denoise_step` (action encoder, DiT on the cached
  K/V, action decoder, Euler update). The DiT's timestep conditioning is precomputed for the fixed schedule.
  `export_gr00t_n1_7_action_head.py --check` confirms the split is bit-identical to the official `get_action`.

AGX Orin, FP16, two real SO101 camera frames, fixed state and noise:

| | Result |
|---|---|
| Backbone hidden states vs PyTorch | cosine 0.99963 |
| Actions vs official GR00T (fp32 PyTorch) | cosine 1.000000, max \|Δ\| 0.0039 on values up to ±4.2 |
| Policy step p50 | 114.5 ms (backbone 29 ms, action head 86 ms) |

Another GPU job shared the board during the measurement.

```bash
# backbone
python experimental_models/gr00t/scripts/extract_gr00t_n1_7_backbone.py --gr00t GR00T-N1.7-SO101-Multitask \
    --base <Cosmos-Reason2-2B snapshot> --out gr00t_backbone
tensorrt-edgellm-export gr00t_backbone gr00t_onnx
llm_build --onnxDir gr00t_onnx/llm --engineDir engines/llm --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 640
visual_build --onnxDir gr00t_onnx/visual --engineDir engines --minImageTokens 16 --maxImageTokens 512 \
    --maxImageTokensPerImage 256
# action head (needs the GR00T N1.7 source and its dependencies, e.g. transformers 4.57 and diffusers)
python experimental_models/gr00t/scripts/export_gr00t_n1_7_action_head.py --gr00t-src <dir holding gr00t/> \
    --checkpoint GR00T-N1.7-SO101-Multitask --embodiment new_embodiment --out action_onnx
# then trtexec --fp16 each of action_onnx/{vl_prep,state_encoder,denoise_step}.onnx into engines/action
gr00t_policy_inference --llmEngineDir engines/llm --multimodalEngineDir engines --actionEngineDir engines/action \
    --promptFile prompt.txt --images top.png,wrist.png --state state.f32 --noise noise.f32 --out actions.f32
```

## Getting started

The build, export and runtime workflow is upstream's:
[installation](docs/source/user_guide/getting_started/installation.md),
[quick start](docs/source/user_guide/getting_started/quick-start-guide.md),
[supported models](docs/source/user_guide/getting_started/supported-models.md).
On JetPack 6 follow [JETPACK6.md](JETPACK6.md) instead of the JetPack 7 build line.

```bash
git clone --recurse-submodules https://github.com/hungho77/edge-vla.git
```

Python packages, CLI names (`tensorrt-edgellm-*`) and C++ namespaces keep their upstream names so that
upstream releases merge cleanly.

## Upstream sync

`main` tracks NVIDIA TensorRT Edge-LLM releases by merging `upstream/main`
(`https://github.com/NVIDIA/TensorRT-Edge-LLM`). Fixes that apply upstream are also proposed there; see
[NVIDIA/TensorRT-Edge-LLM#193](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/193) for InternVLA-N1.

## License

Apache License 2.0; see [LICENSE](LICENSE). Built on
[NVIDIA TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM), Copyright NVIDIA Corporation.
