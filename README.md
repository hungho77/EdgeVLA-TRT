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

Upstream VLA support (experimental pi0.5, Alpamayo, Cosmos3-Edge policy) is unchanged and documented under
[docs/source/user_guide/examples/vla](docs/source/user_guide/examples/vla/index.md).

### Measured on AGX Orin 64 GB, JetPack 6.2

InternVLA-N1-DualVLN, FP16 System 2 (Qwen2.5-VL-7B), BF16 System 1, `--ticks 40 --cadence 4`:

| Metric | Value |
|---|---|
| First plan | 181 ms (System 2 text-only, as in the CLI example) |
| System-1 tick, mean / worst | 188 ms (5.3 Hz) / 314 ms |

The LLM engine was built with `--maxInputLen 2048 --maxKVCacheCapacity 2560` to fit beside the other engines.

## Roadmap: KV-cache reuse for VLA control loops

Edge-LLM 0.11 has page-granular context reuse (`--enableContextReuse`), an encoder embedding cache, cross-request
KV retention and FP8 KV. None of the VLA paths reuse KV across control steps today: each replan or action chunk
recomputes the full prefix. Reuse applies to VLAs with a **causal** VLM backbone (InternVLA-N1, GR00T, Alpamayo);
models whose image+text prefix attends bidirectionally (pi0.5, SmolVLA) can only skip repeated vision encoding.

1. **InternVLA-N1 pilot**: allow hidden-state capture with the context cache, lay the prompt out
   static-text → episode history → latent queries, enable the encoder cache for history frames, and measure
   replan latency on Orin with real frames.
2. **Shared `vla/` layer**: observation encoder, causal backbone on the core runtime, action head
   (flow matching / diffusion / autoregressive), and a dual-rate scheduler; port GR00T N1.6/N1.7 onto it.
3. **Bidirectional-prefix VLAs** (pi0.5, SmolVLA): encoder cache, persistent CUDA graphs, prefix KV pool in
   the core runtime.

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
