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
