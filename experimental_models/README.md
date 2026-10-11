# Experimental Models

This tree hosts experimental model integrations that are **not** first-class models in the core
`tensorrt_edgellm` (Python export) / `cpp` (C++ runtime) packages. Code here **links against and reuses** the
core libraries (KV cache, builders, kernels, ONNX export utilities, runner patterns) rather than duplicating
them, and follows the same runtime design conventions as `cpp/` (a runner owns its config, device buffers and
TensorRT context, and drives its loop in-process on the GPU).

It is built only when `-DBUILD_EXPERIMENTAL_MODELS=ON` is passed to CMake.

## Models

| Model | Path | Status |
|-------|------|--------|
| Cosmos3-Omni | `cosmos3/` | autoregressive text scaffolding + diffusion policy / action generation |
| Nemotron-3.5-ASR | `nemotron3_5_asr/` | offline (batch 1) RNN-T speech transcription (FastConformer encoder + LSTM prediction network; no LLM backbone) |
| Shared VLA runtime | `vla/` | engines with shared context memory, async action chunking, TCP policy-server framing, Python client, LIBERO evaluation |
| pi0.5 | `pi05/` | FP16 flow-matching VLA policy; openpi's `pi05_libero`, `pi05_droid`, `pi05_aloha` and SO101 contracts |
| GR00T N1.5 / N1.6 / N1.7 | `gr00t/` | Eagle or Qwen3-VL backbone + flow-matching DiT action head; LIBERO and SO101 |
| SmolVLA | `smolvla/` | LeRobot SmolVLA; LIBERO and SO101 |
| X-VLA | `xvla/` | LeRobot X-VLA, three engines |
| OpenVLA | `openvla/` | vision engine + Llama-2 on the core LLM runtime, action-token decoding |
| TurboVLA | `turbovla/` | single-pass regression policy (DINOv3 + BERT); H-EmbodVis and starVLA-trainer checkpoints |
| RLDX-1 | `rldx/` | truncated Qwen3-VL-8B backbone + action expert, frame history |
| MolmoAct2 | `molmoact2/` | Molmo2 VLM prefix + flow-matching action expert |
| Evo-1 | `evo1/` | InternVL3-1B (14 language layers) + cross-modulated flow-matching head; FlashAttention-2 padding semantics |
| InternVLA-N1 | `internvla_n1/` | dual-system vision-language navigation (Qwen2.5-VL planner served by the core runtime + flow-matching trajectory expert; asynchronous in-process handoff) |

See each model's `README.md` (SmolVLA: `smolvla/DESIGN.md`) for details, and the top-level `README.md` for the
support matrix and roadmap.
