# SmolVLA on EdgeVLA (design notes)

Target: [`quangnd58/smolvla-so101-multitask`](https://huggingface.co/quangnd58/smolvla-so101-multitask)
(LeRobot `v0.6.1`, base `lerobot/smolvla_base`), validated export -> build -> inference against LeRobot's
own `SmolVLAPolicy` (reference venv with `lerobot[smolvla]==0.6.1`, CPU FP32, fixed noise), stage by stage
as for pi0.5.

## Model, as LeRobot 0.6.1 runs it

| Part | Shape / behaviour |
|---|---|
| Vision | SmolVLM2-500M SigLIP (768 wide), images `resize_with_pad` to 512x512, `x * 2 - 1`; pixel-shuffle connector (scale 4) -> 64 tokens per camera |
| VLM | SmolVLM2-500M text model truncated to 16 of 32 layers: hidden 960, 15 heads, 5 KV heads, head_dim 64, RoPE theta 100000 |
| Prefix | [camera tokens, language (48, right-padded, newline-appended), 1 state token]; state mean/std-normalized, zero-padded to 32, `state_proj` |
| Prefix mask | images + language bidirectional (pads masked); the state token attends to everything, nothing attends to it |
| Expert | 16 layers, width 720 (0.75x), same head layout; time is a pi0-style concat MLP (`action_time_mlp_in/out`) on the action embedding, no AdaRMS |
| Expert attention | even layers: self-attention over [VLM prefix K/V of that layer ; expert K/V], action tokens causal among themselves; odd layers: cross-attention, queries (RoPE from position 0) over the VLM's cached post-RoPE K and V re-projected by the expert's own `k_proj`/`v_proj` (no RoPE on those keys) |
| Sampling | 10 Euler steps, chunk 50; the cache is cropped back to the prefix after every step |
| Output | mean/std unnormalize; absolute joint targets (no delta step) |

## Mapping onto the runtime

- Visual and prefix: the pi0.5 component pattern (one engine each). The prefix emits per-layer post-RoPE K/V
  of the compact prefix (pads dropped, positions = cumsum of the pad mask, as LeRobot computes them). The
  state-token mask is two mask-free attentions: rows of images+language over themselves, and the state row
  over all keys, which keeps `EDGELLM_PORTABLE_ATTENTION` usable on JetPack 6.
- Cross K/V: odd layers' expert-projected K/V depend on the prefix only, so they are computed once per call
  (one small engine, or folded into the prefix engine's outputs), like GR00T's cached cross-attention K/V.
- Denoise step: one engine per step. Self-attention layers use the AttentionPlugin over the paged prefix pool
  with a causal tree mask over the 50 action rows (the pi0.5 path, QKV packed by one GEMM); cross-attention
  layers attend over the precomputed K/V (plain attention, no cache write).
- Policy: mean/std normalization, the camera rename (`top` -> camera1, `wrist` -> camera2), the tokenizer
  with the newline step; RTC as for pi0.5 but with no delta re-encoding (absolute actions), async through
  `vla::AsyncChunker`.

## Order of work

1. Reference capture script (LeRobot policy, fixed noise; dumps preprocessed images, tokens, prefix K/V,
   per-step velocity, actions).
2. Export (`tensorrt_edgellm/models/smolvla`): visual, prefix, cross-KV, denoise step; per-stage parity in
   PyTorch before ONNX.
3. C++ runtime (`experimental_models/smolvla`), golden test end to end, timing.
4. RTC / async, INT8 where the timing says it pays.
