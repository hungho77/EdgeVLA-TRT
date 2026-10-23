# X-VLA

X-VLA (LeRobot 0.6.1, `lerobot/xvla-base`) runs as three engines built from LeRobot's own modules through the
legacy ONNX exporter: `vision` (Florence-2's DaViT and
projection, one batch item per camera), `encoder` (the BART encoder over the primary view's tokens and the padded
instruction) and `step` (one flow step of the soft-prompted action transformer, x1 to the action over 10 steps).
The exported graphs go through `tensorrt_edgellm.onnx.trt_workarounds`: TensorRT 10.3 miscomputes attention whose
transposed key is scaled after a fused QKV projection was split into heads, which left actions 0.04 off on a 0.57
range, in FP32 too, while onnxruntime matched the eager model (see [JETPACK6.md](../../JETPACK6.md)). `XvlaPolicy`
reproduces LeRobot's processing: ImageNet normalization and resize_with_pad, BART tokens padded to the saved
length, the zero-padded state, and the ee6d action space (gripper channels zeroed in the inputs, sigmoid on the
output). The domain id is a runtime input (`setDomainId`, default: the preprocessor's), the denoising loop replays
as a CUDA graph, and `XvlaRtc` adds real-time chunking: each step's predicted clean action is pulled toward the
previous chunk's remaining rows with LeRobot's LINEAR prefix weights (X-VLA's ee6d actions are absolute, so the
rows are reused as they are).

AGX Orin, two cameras from raw SO101 frames, against LeRobot's `XVLAPolicy` in FP32 with the same x1: identical
token ids, bit-identical preprocessed views, actions max |Δ| 0.0008 / 0.0006 on two frames (cosine 1.000000);
155.8 ms per chunk (vision 25.4, encoder 3.2, 10 steps 127.3) with unlocked clocks. `xvla_async_control` at 30 Hz
(horizon 30, overlap 20, frozen 6, drifting state): no stalls, planner latency 5-6 ticks, switch jump 0.0000 with
RTC (0.0151 without).

```bash
python experimental_models/xvla/scripts/lerobot_xvla_reference.py --checkpoint xvla-base --observation obs.json \
    --out ref.npz                                         # LeRobot 0.6.1
python experimental_models/xvla/scripts/export_xvla.py --checkpoint xvla-base --check ref.npz --out xvla_onnx
trtexec --onnx=xvla_onnx/vision.onnx --saveEngine=engines/vision.engine --stronglyTyped \
    --minShapes=images:1x3x224x224 --optShapes=images:2x3x224x224 --maxShapes=images:3x3x224x224
trtexec --onnx=xvla_onnx/encoder.onnx --saveEngine=engines/encoder.engine --stronglyTyped
trtexec --onnx=xvla_onnx/step.onnx --saveEngine=engines/step.engine --stronglyTyped
cp -r xvla_onnx/config.json xvla_onnx/tokenizer engines/
xvla_policy_inference --engineDir engines --inputFile obs.json
xvla_async_control --engineDir engines --inputFile obs.json [--domain ID] [--rtc 1]
```

## Checkpoints fine-tuned in the official repository (SO101)

`twanghcmut/X-VLA-SO101-LeftArm-Multitask` was fine-tuned with [2toinf/X-VLA](https://github.com/2toinf/X-VLA)
(`6bc2513`, its SO101 handler in the checkpoint's `so101/`), not LeRobot. `--xvla-src` exports it from that
repository's modules and `xvla_reference.py` takes the golden from its `XVLA.generate_actions`. The engine dir's
config.json then selects that checkpoint's contract in `XvlaPolicy`:

- `auto` action space: 6 real dims out of the model's 20, no gripper sigmoid;
- `MIN_MAX`: the proprio and actions are scaled to [-1, 1] with the training data's `meta/stats.json` (for the
  11 SO101 task datasets, the per-dataset min / max combined); joints 0-4 (`delta_dims`) are predicted relative to the
  normalized proprio and RTC seeds them as absolute rows;
- `proprio_source: previous_action`: the proprio is the last commanded action (`action[t - 1]`), not the measured
  joints; the policy server advertises it as `"proprio"` in its ready line;
- views resized with PIL bicubic (no padding), the missing third view a zero image through the vision engine.

AGX Orin, against the official model in FP32 with the same x1 on two dataset frames: identical token ids, bit-identical
views, actions max |Δ| 0.04 / 0.11 degrees; 166 ms per chunk (vision 32.9, encoder 2.9, 10 steps 130.2). Open loop on
the model card's 11 trajectories (its `eval_so101_open_loop.py` protocol): MAE 2.384 (card 2.344).
`xvla_async_control` at 30 Hz (overlap 20, frozen 8): planner lag 6 ticks, no stalls, switch jump 0.0056 with RTC
(8.55 without).

```bash
python experimental_models/xvla/scripts/xvla_reference.py --xvla-src X-VLA --checkpoint X-VLA-SO101-LeftArm-Multitask \
    --stats stats.json --obs obs.npz --out ref.npz                  # transformers 4.51.3, as X-VLA pins
python experimental_models/xvla/scripts/export_xvla.py --xvla-src X-VLA --checkpoint X-VLA-SO101-LeftArm-Multitask \
    --stats stats.json --domain-id 19 --delta-dims 0 1 2 3 4 --check ref.npz --out xvla_onnx
```
