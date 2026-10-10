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

VLA and VLN models on AGX Orin 64 GB, JetPack 6.2 (CUDA 12.6, TensorRT 10.3), FP16 engines. Each row is verified on
a public checkpoint the model's authors or community publish: export → engine build → C++ policy server, driven in
LIBERO simulation by [`libero_eval.py`](experimental_models/vla/README.md#libero-evaluation) with each checkpoint's
own evaluation conventions (LIBERO-Spatial, 10 tasks x 10 episodes, fixed initial states).

| Model | Public checkpoint | LIBERO-Spatial on Orin (reported) | Policy call | RTC | Async loop | Guide |
|---|---|---|---|:---:|:---:|---|
| GR00T N1.7 | [`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) | 97% (97.7%) | 101 ms | ✓ | ✓ | [GR00T](experimental_models/gr00t/README.md#gr00t-n17) |
| GR00T N1.6 | [`0xAnkitSingh/GR00T-N1.6-LIBERO`](https://huggingface.co/0xAnkitSingh/GR00T-N1.6-LIBERO) | 100% (96.0%) | 120 ms | ✓ | ✓ | [GR00T](experimental_models/gr00t/README.md#gr00t-n16) |
| GR00T N1.5 | [`youliangtan/gr00t-n1.5-libero-spatial-posttrain`](https://huggingface.co/youliangtan/gr00t-n1.5-libero-spatial-posttrain) | 88% (92%); 89% at 4 steps | 196 ms; 112 ms at 4 steps | ✓ | ✓ | [GR00T](experimental_models/gr00t/README.md#gr00t-n15) |
| pi0.5 | [openpi `pi05_libero`](https://github.com/Physical-Intelligence/openpi) | 100% (98.8%) | 231 ms | ✓ | ✓ | [pi0.5](experimental_models/pi05/README.md) |
| SmolVLA | [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero) | 72% (90%) | 122 ms | ✓ | ✓ | [SmolVLA](experimental_models/smolvla/DESIGN.md) |
| X-VLA | [`lerobot/xvla-libero`](https://huggingface.co/lerobot/xvla-libero) | 98% (98.2%) | 198 ms | ✓ | ✓ | [X-VLA](experimental_models/xvla/README.md) |
| TurboVLA | [`H-EmbodVis/TurboVLA`](https://huggingface.co/H-EmbodVis/TurboVLA) (`checkpoints/libero`) | 95% (97.0%) | 32 ms | ✓ (blend) | ✓ | [TurboVLA](experimental_models/turbovla/README.md) |
| RLDX-1 | [`RLWRLD/RLDX-1-FT-LIBERO`](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) | 97% (98.6% LIBERO-Short) | 297 ms | ✓ | ✓ (20 Hz) | [RLDX-1](experimental_models/rldx/README.md) |
| MolmoAct2 | [`allenai/MolmoAct2-LIBERO-LeRobot`](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot) | 100% (98.4%) | 558 ms | ✓ | ✓ (4 Hz) | [MolmoAct2](experimental_models/molmoact2/README.md) |
| OpenVLA | [`openvla/openvla-7b-finetuned-libero-spatial`](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial) | 85% (84.7%) | 838 ms | | | [OpenVLA](experimental_models/openvla/README.md) |
| InternVLA-N1 (VLN) | [`InternRobotics/InternVLA-N1-DualVLN`](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN) | VLN: R2R episode in VLN-PE | first plan 181 ms, System-1 tick 188 ms | | dual-rate | [InternVLA-N1](experimental_models/internvla_n1/README.md) |

- **Reported**: the success rate the checkpoint's authors or paper give for LIBERO-Spatial, usually over more
  episodes; the [LIBERO guide](experimental_models/vla/README.md#libero-evaluation) lists the sources and the
  conventions each run follows. SmolVLA's engines match LeRobot's policy on LIBERO observations, so its gap is the
  checkpoint's under this protocol.
- **Policy call**: one request to the model's policy server on a LIBERO observation (two 256x256 views; OpenVLA
  one), image encoding to actions, median of 20 on an otherwise idle board.
- **RTC**: real-time chunking, where the next action chunk is inpainted from the unexecuted tail of the previous one,
  so chunk switches do not jump. **Async loop**: a 30 Hz control loop with the policy planning the next chunk in
  the background (`*_async_control`); InternVLA-N1 runs its planner and trajectory head at two rates in one process.
- RLDX-1 reads the frames at t-6, t-4, t-2 and t: its server takes them inline (`camera@offset`) and the Python
  controller keeps them per tick. Its paper reports LIBERO-Short (Spatial, Object, Goal averaged), not Spatial.
- MolmoAct2's async loop keeps up at 4 Hz at its latency, not LIBERO's 20 Hz.
- TurboVLA regresses its chunk in one pass (no denoising to inpaint), so its server blends the new chunk with the
  previous one under the same overlap / frozen fields instead.
- GR00T N1.5 / N1.6 run their Eagle prefix on the LLM runtime
  ([GR00T guide](experimental_models/gr00t/README.md#faster-eagle-backbone)). N1.5's official LIBERO evaluation
  denoises in 8 steps; its checkpoint default of 4 is a speed setting with the same success rate here.
- Every model has a policy server for a real robot: raw camera frames in over stdin or TCP, action chunks out, with a
  Python client and an asynchronous real-time-chunking control loop
  ([real-robot serving](experimental_models/vla/README.md#real-robot-serving)). The guides also cover SO101
  fine-tunes (GR00T N1.5 / N1.6 / N1.7, pi0.5, SmolVLA, X-VLA) checked against the official policies.

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

