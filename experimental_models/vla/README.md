# Shared VLA runtime

What the VLA ports share: the engine / backbone / async-chunking layer and the core-runtime caching that
makes VLA replans cheap.

## Shared VLA layer

This directory holds what InternVLA-N1 and GR00T have in common; both now
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
[`prepare_vlnpe_episode.py`](../internvla_n1/examples/prepare_vlnpe_episode.py)), averaged over the
94 nine-frame replans:

| Configuration | Mean replan | vs. no cache |
|---|---|---|
| No cache | 2009 ms | — |
| Encoder partial hits | 1429 ms | −29% |
| Encoder partial hits + KV prefix reuse | 938 ms | −53% |

KV reuse restores 38% of the episode's prompt tokens; `z_latents` are again bit-identical to the no-cache run.

Reuse applies to VLAs with a **causal** VLM backbone (InternVLA-N1, GR00T, Alpamayo). Models whose image+text prefix
attends bidirectionally (pi0.5, SmolVLA) can only skip repeated vision encoding.

