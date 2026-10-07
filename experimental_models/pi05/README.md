# pi0.5 (experimental runtime)

pi0.5 model definitions and ONNX export live in the main Python package under
`tensorrt_edgellm.models.pi05` (exported through the unified `tensorrt-edgellm-export`). This
directory keeps the separate experimental runtime: C++ component builder, policy runner and
inference CLI.

The supported input is an already-converted **PyTorch** pi0.5 checkpoint (`model.safetensors` +
`config.json`); the LeRobot releases are already in that form. This runtime supports openpi's
`pi05_libero`, `pi05_droid` and `pi05_aloha` contracts, in FP16.

## Layout

```text
pi05/
  cpp/        # component builder, policy runner, policy pre/post-processing
  examples/   # pi05_policy_build and pi05_policy_inference
  scripts/    # compare_pi05_actions.py, the numerical comparator
```

## Quickstart

The binaries come from the standard Edge-LLM experimental-model build (configure with
`-DBUILD_EXPERIMENTAL_MODELS=ON`); this assumes they and the plugin library are already built.

```bash
export CHECKPOINT=lerobot/pi05_libero_base ONNX_DIR=$HOME/pi05_onnx
export ENGINE_DIR=$HOME/pi05_engines BUILD_DIR=/path/to/tensorrt-edge-llm/build
export EDGELLM_PLUGIN_PATH="$BUILD_DIR/libNvInfer_edgellm_plugin.so"

# 1. Export ONNX + component contracts (x86 host, CPU-only). --pi05-policy-config names the
#    openpi contract, and is required for every configuration but pi05_libero.
PYTHONNOUSERSITE=1 tensorrt-edgellm-export "$CHECKPOINT" "$ONNX_DIR" --dtype float16 \
    --pi05-policy-config pi05_libero

# 2. Stage the configuration's normalization statistics. The checkpoint carries the
#    processor schemas but not their state, which stays in openpi's own checkpoint
#    assets. They have to be here before step 3, which copies them into the bundle.
mkdir -p "$ONNX_DIR/assets"
curl --fail --location --output "$ONNX_DIR/assets/norm_stats.json" \
  https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json

# 3. Build every component engine and stage the runtime sidecars.
"$BUILD_DIR/experimental_models/pi05/examples/pi05_policy_build" \
    --onnxDir "$ONNX_DIR" --engineDir "$ENGINE_DIR"

# 4. Run policy inference. See the example doc for the observation.json format.
"$BUILD_DIR/experimental_models/pi05/examples/pi05_policy_inference" \
    --engineDir "$ENGINE_DIR" --inputFile observation.json --output action.json
```

The export writes four components: `visual/`, `prefix/`, `action/` and `cond/`. `cond/` precomputes
the AdaRMS modulation schedule instead of rebuilding it inside the per-step action graph, and the
runtime re-evaluates it only when the denoise-step count or the batch changes. It is optional:
`--no-pi05-hoist-adarms-cond` leaves it out.

[pi0.5 example](../../docs/source/user_guide/examples/vla/pi05.md) has the statistics download, the
request format and the correctness check; [pi0.5 Design](../../docs/source/developer_guide/models/pi05.md)
has the component contracts.

## SO101 (`pi05_so101`)

`pi05_so101` serves [`hungho77/pi05-SO101-Multitask`](https://huggingface.co/hungho77/pi05-SO101-Multitask),
an openpi PyTorch fine-tune: the overhead and wrist cameras in the `observation/image` and
`observation/wrist_image` slots, a 6-dim state discretized into the prompt, a 50-step horizon, and the
five arm joints as deltas from the request's state (the gripper is absolute), which the policy adds
back so `robot_actions` are absolute joint targets.

That release's `config.json` carries only `discrete_state_input`; export it through a directory that
adds the openpi model fields next to the weights and assets:

```json
{"action_dim": 32, "action_horizon": 50, "paligemma_variant": "gemma_2b",
 "action_expert_variant": "gemma_300m", "precision": "bfloat16", "discrete_state_input": true}
```

On JetPack 6 (TensorRT 10.3) export with `EDGELLM_PORTABLE_ATTENTION=1`; see
[JETPACK6.md](../../JETPACK6.md).

`scripts/openpi_reference.py` runs openpi's own `create_trained_policy` on a
`pi05_policy_inference` request in FP32 and writes the chunk, the x_0 it drew and, with
`--inputs-dir`, the model inputs for the canonical-tensor mode; `compare_pi05_actions.py` then
scores either field. On AGX Orin (JetPack 6.2) against two raw SO101 dataset frames the robot actions
match at cosine 0.999999 (max |d| 0.31 on joint ranges of about ±100); the normalized chunk is at
cosine 0.99999 with max |d| 0.008, just outside the comparator's 5e-3 ceiling. pi05_libero passes
it (cosine 0.999991, max |d| 4.8e-3).

## INT8 prefix (opt-in)

The prefix tower dominates a pi0.5 call. `--pi05-prefix-int8 <stats>` exports its projections as W8A8
SmoothQuant (per-channel INT8 weights, per-tensor INT8 activations) from activation statistics that
`scripts/calibrate_pi05_prefix_int8.py` collects by running openpi on dataset frames:

```bash
python experimental_models/pi05/scripts/calibrate_pi05_prefix_int8.py --config pi05_so101 \
    --checkpoint <openpi checkpoint> --dataset <LeRobot v3 root> --num-frames 48 --out prefix_amax.safetensors
tensorrt-edgellm-export <checkpoint> <onnx_dir> --pi05-policy-config pi05_so101 \
    --pi05-prefix-int8 prefix_amax.safetensors --pi05-prefix-int8-alpha 0.8 \
    --pi05-prefix-fp16-projections o_proj,down_proj
```

Gemma's `down_proj` inputs carry outliers up to ~1e4 that per-tensor INT8 cannot hold, so the useful
setting keeps `o_proj` and `down_proj` in FP16 and quantizes q/k/v and gate/up. SO101 on AGX Orin
against openpi FP32 (held-out dataset frames):

| Prefix | Robot actions, max \|d\| (mean) | Note |
|---|---|---|
| FP16 | 0.21-0.31 | openpi's own bf16 vs FP32: 0.16-0.28 |
| W8A8 all projections, alpha 0.5 | 12.7-17.9 | unusable |
| W8A8 q/k/v + gate/up, alpha 0.8 | 0.78-1.45 (0.10-0.30) | about 5x the bf16 floor |

The full W8A8 prefix runs in 67 ms against 128 ms for FP16; the accurate setting saves less, since a third of
the prefix FLOPs stay FP16.
