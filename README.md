<div align="center">

# EdgeVLA-TRT

**Vision-Language-Action inference on NVIDIA Jetson, built on NVIDIA TensorRT Edge-LLM**

[![upstream](https://img.shields.io/badge/built%20on-TensorRT%20Edge--LLM%200.11.0-76b900)](https://github.com/NVIDIA/TensorRT-Edge-LLM)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](LICENSE)
[![platform](https://img.shields.io/badge/tested-AGX%20Orin%20JetPack%206.2-lightgrey)](JETPACK6.md)

</div>

EdgeVLA-TRT is a fork of [NVIDIA TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) focused on
Vision-Language-Action (VLA), vision-language navigation (VLN) and world-action models on edge devices. It keeps
Edge-LLM's export → engine build → C++ runtime pipeline and adds VLA models, VLA-oriented runtime work, and
support for platforms upstream no longer targets.

> EdgeVLA-TRT is an independent project. It is not an NVIDIA product and is not endorsed by NVIDIA. NVIDIA and
> TensorRT are trademarks of NVIDIA Corporation. Code from upstream keeps its NVIDIA copyright headers and is
> distributed under the Apache License 2.0.

## Support matrix

VLA and VLN models on AGX Orin 64 GB, JetPack 6.2 (CUDA 12.6, TensorRT 10.3). Each port is verified export → engine
build → C++ inference against the official implementation (FP32, same inputs and noise); the numbers are in each
guide. Engines are FP16 unless noted.

| Model | Checkpoint verified | Latency on Orin | RTC | Async loop | Server | Guide |
|---|---|---|:---:|:---:|:---:|---|
| GR00T N1.7 | `GR00T-N1.7-SO101-Multitask`, `ducido/GR00T-N1.7-SO101-banana-all49` | 96.4 ms / chunk (73.1 ms W8A8 DiT) | ✓ | ✓ | ✓ | [GR00T](experimental_models/gr00t/README.md#gr00t-n17) |
| GR00T N1.6 | `GR00T-N1.6-SO101-Multitask` | 171.7 ms / chunk | ✓ | ✓ | ✓ | [GR00T](experimental_models/gr00t/README.md#gr00t-n16) |
| GR00T N1.5 | `GR00T-N1.5-SO101-Multitask` | 110.4 ms / chunk | ✓ | ✓ | ✓ | [GR00T](experimental_models/gr00t/README.md#gr00t-n15) |
| pi0.5 | openpi `pi05_so101` | 204.5 ms / chunk (177.9 ms INT8 prefix) | ✓ | ✓ | | [pi0.5](experimental_models/pi05/README.md#so101-pi05_so101) |
| SmolVLA | LeRobot 0.6.1 | 83-93 ms / chunk | ✓ | ✓ | | [SmolVLA](experimental_models/smolvla/DESIGN.md) |
| X-VLA | `lerobot/xvla-base` (LeRobot 0.6.1) | 155.8 ms / 30-step chunk | ✓ | ✓ | | [X-VLA](experimental_models/xvla/README.md) |
| OpenVLA | `openvla/openvla-7b` | 706 ms / action | | | | [OpenVLA](experimental_models/openvla/README.md) |
| InternVLA-N1 (VLN) | `InternVLA-N1-DualVLN` | first plan 181 ms, System-1 tick 188 ms | | dual-rate | ✓ | [InternVLA-N1](experimental_models/internvla_n1/README.md) |

- **RTC**: real-time chunking, where the next action chunk is inpainted from the unexecuted tail of the previous one,
  so chunk switches do not jump.
- **Async loop**: a 30 Hz control loop with the policy planning the next chunk in the background (`*_async_control`);
  InternVLA-N1 runs its planner and trajectory head at two rates in one process.
- **Server**: a long-lived process answering one JSON request per line (`gr00t_policy_server`,
  `internvla_n1_dual_system_server`).
- Latency is one policy call, preprocessing to actions, measured with other jobs sometimes sharing the board; see
  each guide for the breakdown and conditions.

Upstream's VLA support (pi0.5 `libero` / `droid` / `aloha`, Alpamayo, Cosmos3-Edge policy) and its LLMs and VLMs
are unchanged: [VLA examples](docs/source/user_guide/examples/vla/index.md),
[supported models](docs/source/user_guide/getting_started/supported-models.md).

## Runtime features

| Feature | Details |
|---|---|
| KV-cache reuse for VLA replans | Encoder-cache partial hits and KV prefix reuse with tail-only hidden capture; InternVLA-N1 replans 57% faster, outputs bit-identical. [Shared VLA runtime](experimental_models/vla/README.md#kv-cache-reuse-for-vla-control-loops) |
| Shared VLA layer | Engines with shared context memory, VLM-as-backbone prefill, dual-rate planner handoff, async action chunking. [Shared VLA runtime](experimental_models/vla/README.md#shared-vla-layer) |
| JetPack 6 (CUDA 12.6, TensorRT 10.3) | Runs 0.11 on AGX Orin, including an ONNX rewrite around a TensorRT 10.3 attention miscompilation. [JETPACK6.md](JETPACK6.md) |
| Precomputed image embeddings | `LLMGenerationRequest::precomputedImageEmbeddings` feeds an external vision engine into the LLM runtime (OpenVLA). |
| Exact asymmetric INT4 AWQ | Engine output matches the exactly dequantized checkpoint token-for-token. [FIXES.md](FIXES.md) |
| NVFP4 AWQ `pre_quant_scale`, NVFP4 CASK epilogue cap (TRT 10.13/10.14) | Fixed; needs Thor to verify. [FIXES.md](FIXES.md) |
| InternLM2-backed InternVL3 checkpoints | Converter: `tensorrt_edgellm/scripts/convert_internlm2_internvl.py` |

Open: the encoder cache and prefix KV pool for bidirectional-prefix VLAs (pi0.5, SmolVLA) in the core runtime.

## Getting started

The build, export and runtime workflow is upstream's:
[installation](docs/source/user_guide/getting_started/installation.md),
[quick start](docs/source/user_guide/getting_started/quick-start-guide.md),
[supported models](docs/source/user_guide/getting_started/supported-models.md).
On JetPack 6 follow [JETPACK6.md](JETPACK6.md) instead of the JetPack 7 build line.

```bash
git clone --recurse-submodules https://github.com/hungho77/EdgeVLA-TRT.git
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

