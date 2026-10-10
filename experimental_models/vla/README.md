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


## LIBERO evaluation

[`scripts/libero_eval.py`](scripts/libero_eval.py) measures a policy's LIBERO success rate in simulation. The model
runs in its C++ policy server (one JSON request per line: `gr00t_policy_server`, `pi05_policy_server`,
`smolvla_policy_server`, `xvla_policy_server`, `turbovla_policy_server`, `rldx_policy_server`,
`molmoact2_policy_server`, `openvla_policy_server`), started as a subprocess, so the simulator
side needs only a LIBERO Python environment (robosuite 1.4 with mujoco 2.3, `MUJOCO_GL=egl`) and no TensorRT.

Protocol: LIBERO's fixed initial states (episode *i* of a task uses init state *i*), 10 no-op steps for the objects
to settle, then the policy until success or the step limit; 256x256 renders. Each adapter follows its checkpoint's
own LIBERO evaluation script, because these conventions move the success rate by tens of points:

| Policy | Checkpoint | Following | Views | State | Rows per call | Gripper | Step limit |
|---|---|---|---|---|---|---|---|
| `gr00t_n17` | `nvidia/GR00T-N1.7-LIBERO` (`libero_sim`) | Isaac-GR00T `examples/LIBERO` | both flipped | xyz, axis-angle, gripper qpos | 8 of 16 | binarized, inverted | 220 |
| `gr00t_n16` | `0xAnkitSingh/GR00T-N1.6-LIBERO` (`libero_panda`) | Isaac-GR00T | both flipped | as N1.7 | 8 of 16 | binarized, inverted | 220 |
| `gr00t_n15` | `youliangtan/gr00t-n1.5-libero-spatial-posttrain` (`LiberoDataConfig`, 8 denoising steps) | GR00T N1.5 `examples/Libero` | both flipped | as N1.7 | 1 of 16 | binarized, inverted | 220 |
| `pi05` | openpi `pi05_libero` (PyTorch conversion) | openpi `examples/libero` | both flipped, PIL bilinear to 224 | eef pos, axis-angle, gripper qpos | 5 of 10 | as returned | 220 |
| `smolvla` | `HuggingFaceVLA/smolvla_libero` | LeRobot `LiberoProcessorStep` | both flipped | as pi0.5 | 1 of 50 (its `n_action_steps`) | as returned | 280 |
| `xvla` | `lerobot/xvla-libero` | LeRobot `make_xvla_libero_pre_post_processors` | agent view flipped only | eef pos, rot6d of the controller orientation, 0, padded to 20 | 30 of 30 | `> 0.5` | 280, absolute control |
| `turbovla` | `H-EmbodVis/TurboVLA` (`checkpoints/libero`, all four suites) | TurboVLA's VLA-Adapter-derived rollout | both flipped | as pi0.5 | 12 of 12 | as returned (+-1) | 220 |
| `rldx` | `RLWRLD/RLDX-1-FT-LIBERO` (`general_embodiment`, all four suites) | RLDX-1 `run_scripts/eval/libero` | both flipped, frames t-6 / t-4 / t-2 / t | as pi0.5 | 8 of 16 | `sign(2 g - 1)` of gripper_close | 220 |
| `molmoact2` | `allenai/MolmoAct2-LIBERO-LeRobot` (all four suites, continuous actions) | LeRobot `LiberoProcessorStep` (`lerobot-eval`) | both flipped | as pi0.5 | 10 of 10 (its `n_action_steps`) | as returned | 280 |
| `openvla` | `openvla/openvla-7b-finetuned-libero-spatial` | OpenVLA `run_libero_eval.py` | agent view flipped, JPEG round trip, Lanczos to 224, 90% centre crop | none | 1 | binarized, inverted | 220 |

OpenVLA's TensorFlow image steps are reproduced with OpenCV, Pillow and NumPy (`tf.image.crop_and_resize` exactly;
JPEG and Lanczos to within a gray level).

LIBERO-Spatial, 10 tasks x 10 episodes, AGX Orin 64 GB, JetPack 6.2, FP16 engines:

| Policy | Success on Orin | Reported for the checkpoint |
|---|---|---|
| GR00T N1.7 | 97.0% | 97.65% (Isaac-GR00T, 200 episodes) |
| GR00T N1.6 | 97.0%; 100% with the LLM-runtime prefix | 96.0% (model card, 200 episodes) |
| GR00T N1.5 | 88.0%; 89% at 4 denoising steps | 92% (Isaac-GR00T N1.5, 50 episodes, 8 steps) |
| pi0.5 | 100.0% | 98.8% (openpi) |
| SmolVLA | 72.0% | 90% (SmolVLA paper) |
| X-VLA | 98.0% | 98.2% (X-VLA paper) |
| TurboVLA | 95.0% | 97.0% (TurboVLA paper, 500 episodes) |
| RLDX-1 | 97.0% | 98.6% LIBERO-Short, the Spatial / Object / Goal average (RLDX-1 paper; random resets, 720 steps) |
| MolmoAct2 | 100.0% | 98.4% (model card, LeRobot implementation, 50 episodes per task) |
| OpenVLA | 85.0% | 84.7% (OpenVLA paper) |

SmolVLA's engines match LeRobot on a LIBERO observation (robot actions max |Δ| 0.013 on a ±1 range, identical
tokens), and rendering at LeRobot's default 360x360 instead did not change the hardest tasks (45% vs 50% on tasks
3 and 9), so the gap is the public checkpoint's under this protocol rather than the runtime's.
[`scripts/lerobot_policy_server.py`](scripts/lerobot_policy_server.py) serves LeRobot's own PyTorch policy under
the same protocol for a direct comparison on a machine whose PyTorch build supports the GPU.

```bash
# engines: each model's guide, from the checkpoint above; N1.6 / N1.5 Eagle export with --camera-height 256
# --camera-width 256 (LIBERO frames), N1.5's action head with --denoising-steps 8
MUJOCO_GL=egl python experimental_models/vla/scripts/libero_eval.py --policy gr00t_n17 --suite libero_spatial \
    --episodes 10 --max-steps 220 --server-env LD_LIBRARY_PATH=... --server-env EDGELLM_PLUGIN_PATH=... \
    --server-cmd "gr00t_policy_server --llmEngineDir e/llm --multimodalEngineDir e --actionEngineDir e/action" \
    --out gr00t_n17_libero_spatial.json
```

`--out` is rewritten after every episode and a rerun with the same file resumes; `--tasks` and `--rows` narrow a
run or override the rows executed per call.

## Real-robot serving

Every VLA family has a policy server a robot can drive directly: `gr00t_policy_server` (N1.5 / N1.6 / N1.7),
`pi05_policy_server`, `smolvla_policy_server`, `xvla_policy_server`, `turbovla_policy_server`,
`rldx_policy_server`, `molmoact2_policy_server`, `openvla_policy_server`, and `internvla_n1_dual_system_server`
for navigation. They share one transport
([`cpp/vlaServer.h`](cpp/vlaServer.h)):

- **stdin / stdout by default**, or **TCP** with `--port N` (bound to 127.0.0.1; `--host 0.0.0.0` for the LAN). TCP
  serves one client at a time and resets the episode when a client connects.
- **Raw camera frames inline**: a JSON header line names each frame (`{"frames": [{"name", "height", "width",
  "bytes"}], "state": [...], "task": "..."}`) and the frames' bytes follow it, each packed `[H, W, 3]` uint8
  **RGB** (OpenCV images are BGR: convert them). The server applies the checkpoint's own image and language
  preprocessing; GR00T N1.7's eval image transform runs on the server too. Requests naming image files still
  work, which is what the simulator harnesses use.
- **A `ready` line on connect** with the family, the camera names in order, the state and action widths, the
  chunk size and the RTC scheme; `frame_history` when the policy also reads past frames (RLDX-1: [-6, -4, -2, 0]),
  which travel as `camera@offset` frames. `AsyncChunkedController` then keeps the frames of every tick and sends
  those at the offsets, so its loop should tick at the training data's rate.

The Python side is [`scripts/vla_policy_client.py`](scripts/vla_policy_client.py):

```python
from vla_policy_client import AsyncChunkedController, VlaPolicyClient

client = VlaPolicyClient(host="127.0.0.1", port=5555)          # or server_cmd=[...] to spawn it over stdio
print(client.info)                                            # cameras, state_dim, action_dim, chunk, rtc
chunk = client.act({"top": top_rgb, "wrist": wrist_rgb}, state, "pick up the banana", reset=True)

# A 30 Hz loop with the next chunk planned in the background and real-time chunking at the switches.
read = lambda tick: ({"top": camera_top(), "wrist": camera_wrist()}, joint_state(), "pick up the banana")
controller = AsyncChunkedController(client, read, overlap=10, frozen=7)
controller.warmup(read(0))                                    # first plain and RTC calls, before the loop
for tick in itertools.count():
    action = controller.step(tick)                            # None: no chunk covers this tick, hold the robot
    if action is not None:
        robot.send(action)
    sleep_until_next_tick()
```

`frozen` rows are reproduced exactly at a chunk switch, so they must exceed the worst planner latency in ticks
(`replay_robot.py` reports it); `chunk - overlap` rows are executed before the next chunk is requested.
[`scripts/replay_robot.py`](scripts/replay_robot.py) is this loop fed by a recorded LeRobot dataset instead of
hardware.

| Server | Cameras | State in | Actions out | Rows per call | RTC |
|---|---|---|---|---|---|
| GR00T N1.5 / N1.6 / N1.7 | `processing.json` `video_keys` | raw state groups in modality order | absolute raw actions per the embodiment (relative joints decoded against the state) | action horizon (16) | overlap / frozen |
| pi0.5 | the contract's camera slots | robot units, the embodiment's width | robot units after openpi's output transform | action horizon (10 LIBERO, 50 SO101) | overlap / frozen |
| SmolVLA | the dataset's camera names | the dataset's state | the dataset's actions, unnormalized | chunk (50) | delay / horizon |
| X-VLA | the checkpoint's camera names | proprio, zero-padded to 20 | ee6d absolute targets: position, rot6d, gripper through a sigmoid | chunk (30) | delay / horizon |
| RLDX-1 | `front_view`, `left_wrist_view`, each with `@-6`, `@-4`, `@-2` | eef position, axis-angle, gripper qpos (LIBERO) | eef position / rotation deltas and gripper_close, unnormalized | chunk (16) | overlap / frozen (GR00T's) |
| TurboVLA | `primary`, `wrist` | the checkpoint's (LIBERO: eef pos, axis-angle, gripper qpos) | environment actions: arm deltas unnormalized, gripper +1 / -1 | chunk (12) | overlap / frozen, by blending the new chunk with the previous one (no denoising to inpaint) |
| MolmoAct2 | the checkpoint's (LIBERO: `image`, `wrist_image`) | the checkpoint's (LIBERO: eef pos, axis-angle, gripper qpos) | the checkpoint's actions, unnormalized on the masked dims (LIBERO: deltas, gripper in [-1, 1]) | chunk (10) | overlap / frozen (GR00T's) |
| OpenVLA | `image` (third-person) | none | one unnormalized end-effector delta and gripper, per the `unnorm_key` dataset | 1 | none |

Verified by replay and in LIBERO over TCP, not on hardware. AGX Orin, SO101 dataset frames at 30 Hz wall clock
through `replay_robot.py` over TCP with raw frames, 300 ticks each unless noted, no stalls (after `warmup`):

| | Policy call | Planner lag (ticks) | Switch jump with RTC / without |
|---|---|---|---|
| GR00T N1.7 (frozen 7) | 141 ms | 5 | 0.0087 / 26.9 |
| SmolVLA (frozen 6) | 79 ms | 3 | 0.0000 / 6.08 |
| X-VLA (frozen 9) | 188 ms | 6 | 0.0000 / 0.036 |
| pi0.5 (frozen 12) | 272 ms | 8-11 | 0.0000 / 6.15 |
| TurboVLA (overlap 6, frozen 3; LIBERO checkpoint on SO101 frames, another job on the GPU) | 36-42 ms | 1-2 | 0.0000 / 2.0 |
| RLDX-1 (overlap 10, frozen 8; LIBERO checkpoint on SO101 frames with the frame history, 20 Hz, 200 ticks) | 292 ms | 6-7 | 0.0000 / 0.32 |
| MolmoAct2 (overlap 5, frozen 4; LIBERO checkpoint on SO101 frames, 4 Hz, 60 ticks) | 544 ms | 3 | 0.0000 / 1.95 |

Inline frames over stdio and TCP give bit-identical actions to the same images by path for every family, and a
LIBERO-Spatial run through a TCP server (`libero_eval.py --port`) succeeds as over stdio. OpenVLA answers about
once per second, so it suits slow, synchronous stepping rather than a 30 Hz loop.
