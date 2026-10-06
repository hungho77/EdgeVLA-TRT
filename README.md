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
2. **GR00T N1.7**: done ([below](#gr00t-n17)): W8A8 DiT, CUDA-graph denoising, RTC chunking and SO101
   pre/post-processing.
3. **Shared VLA layer**: done ([below](#shared-vla-layer)), extracted from InternVLA-N1 and GR00T.
4. **Bidirectional-prefix VLAs** (pi0.5, SmolVLA): encoder cache, persistent CUDA graphs, prefix KV pool in
   the core runtime.

## GR00T N1.7

NVIDIA GR00T N1.7 (SO101 fine-tune) runs as Edge-LLM backbone plus a vendored action head
([`experimental_models/gr00t`](experimental_models/gr00t)):

- **Backbone**: GR00T's Cosmos-Reason2 (Qwen3-VL) truncated to `select_layer` = 16 layers, exported through the
  regular Qwen3-VL path with `emit_hidden_states: "pre_norm"`, so the engine returns every position's layer-16
  output before the final norm. GR00T pops the extra layers off a full-depth model, and under transformers 4.57
  its `hidden_states[-1]` is then the unnormalized tensor; the post-norm one has cosine 0.08 to it.
- **Action head**, three engines for one embodiment: `vl_prep` (vlln, VL self-attention and the K/V of all 16 DiT
  cross-attention blocks, once per call), `state_encoder`, and `denoise_step` (action encoder, DiT on the cached
  K/V, action decoder, Euler update). The DiT's timestep conditioning is precomputed for the fixed schedule, the
  DiT linears can be exported as calibrated W8A8 (`--int8-weights`), and the four denoising steps replay as one
  CUDA graph.
- **Policy**: `processing.json` records the embodiment's state normalization and action decoding (percentile
  bounds, per-step relative-action bounds, relative-to-absolute joints) from the official processor;
  `gr00t_policy_server` takes camera frames, raw state and an instruction and returns absolute actions, with
  optional real-time chunking (RTC: the next chunk is inpainted from the tail of the previous one). The tail is
  kept in absolute joint space and re-encoded against the new state, so the frozen rows reproduce the actions
  already committed even after the arm moved; seeding the previous normalized rows directly, as the model-level
  API does, misses them by up to 2.8 on SO101 when the state moved 6.6 in 8 frames.
  `gr00t_policy_client.py` applies GR00T's image and language preprocessing on the robot side.

AGX Orin, two SO101 dataset frames from raw video and raw state, against the official `Gr00tPolicy` (fp32
PyTorch) with the same noise:

| | FP16 head | W8A8 DiT |
|---|---|---|
| Backbone hidden states vs PyTorch | per-token cosine 0.99986 (min 0.998) | same |
| Absolute SO101 actions, max \|Δ\| (joint range ±90) | 0.21 | 1.06 |
| Absolute actions with RTC (overlap 8, frozen 2), max \|Δ\| | 0.25 | |
| Policy step p50, CUDA graph (backbone 26.5 ms) | 96.4 ms | 73.1 ms |

A VLA-OPT quantization job shared the board during the timing runs.

```bash
# backbone
python experimental_models/gr00t/scripts/extract_gr00t_n1_7_backbone.py --gr00t GR00T-N1.7-SO101-Multitask \
    --base <Cosmos-Reason2-2B snapshot> --out gr00t_backbone
tensorrt-edgellm-export gr00t_backbone gr00t_onnx
llm_build --onnxDir gr00t_onnx/llm --engineDir engines/llm --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 640
visual_build --onnxDir gr00t_onnx/visual --engineDir engines --minImageTokens 16 --maxImageTokens 512 \
    --maxImageTokensPerImage 256
# action head and processing (need the GR00T N1.7 source and its dependencies, e.g. transformers 4.57 and diffusers)
python experimental_models/gr00t/scripts/export_gr00t_n1_7_action_head.py --gr00t-src <dir holding gr00t/> \
    --checkpoint GR00T-N1.7-SO101-Multitask --embodiment new_embodiment --out action_onnx \
    [--int8-weights --features <pre-norm backbone features .npy> --input-ids <input ids .npy>]
python experimental_models/gr00t/scripts/export_gr00t_n1_7_processing.py --gr00t-src <dir holding gr00t/> \
    --checkpoint GR00T-N1.7-SO101-Multitask --embodiment new_embodiment --out engines/action/processing.json
# then trtexec --fp16 (and --int8 for the W8A8 denoise_step) each of action_onnx/*.onnx into engines/action
python experimental_models/gr00t/examples/gr00t_policy_client.py --server-cmd "gr00t_policy_server \
    --llmEngineDir engines/llm --multimodalEngineDir engines --actionEngineDir engines/action" ...
```

## Shared VLA layer

[`experimental_models/vla`](experimental_models/vla) holds what InternVLA-N1 and GR00T have in common; both now
build on it, with byte-identical outputs before and after the extraction:

- `vlaEngine`: TensorRT engines with user-managed context memory, checked binding and shapes, and one scratch
  allocation for engines that run one after another.
- `vlaBackbone`: the VLM as a VLA backbone, a prefill over a pre-templated prompt and images that keeps one
  layer's hidden states (optionally only the tail rows, so context reuse can restore the prefix).
- `vlaDualRate`: the latest-plan handoff and coalescing planner thread between a slow planner and a fast loop
  (InternVLA-N1's System 2 / System 1).
- `vlaAsyncChunker`: asynchronous action chunking on top of it. The control loop executes one row per tick; at
  `horizon - overlap` rows it snapshots the observation (on the control thread) and requests the next chunk,
  then switches to it at the row the planner latency consumed.

The flow-matching loops stay per model: their schedules, timestep encodings and guidance differ.

`gr00t_async_control` drives GR00T through the chunker at 30 Hz on AGX Orin (horizon 16, overlap 8, frozen 5)
with fixed frames and a drifting joint state: no stalls after the first chunk, planner latency 2 ticks with the
W8A8 head (3 with FP16), and a switch jump of at most 0.006 joint units. The frozen rows have to exceed the
worst-case latency in ticks: with 4 frozen rows, one FP16 switch landed 4 ticks in and jumped 11.3. Frozen rows are
only reproduced inside the model's per-step action range; a seed outside it (SO101 joint 4 spans about 1.9) is
clipped.

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
