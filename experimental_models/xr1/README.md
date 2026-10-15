# Xiaomi-Robotics-1

[Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1) (4.7B) pairs Qwen3-VL-4B with a 36-layer
DiT action head (hidden 1024, 8 heads over the language model's 8 KV heads, adaLN on the flow time) that attends to
every language layer's keys and values, pi0-style: the prompt (three views in a chat-template message with their
captions, the instruction, `<cot></cot>` as the assistant turn) is encoded once, then the DiT integrates a 10 x 60
chunk from noise at t = 0 to 1 in 5 Euler steps. The RoboCasa checkpoint is
[`XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa`](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa)
(remote code `MiBoTForActionGeneration`, the processor's `robocasa_mg` action statistics); the server follows
`deploy/server.py` and the RoboCasa client `eval_robocasa/main.py`.

It runs as four engines:

| Engine | Built from | Per call |
|---|---|---|
| `vision` | Qwen3-VL's ViT at 256 x 256, merger and the three deepstack mergers | once, 64 tokens per view |
| `prefix_a` | the prompt (image placeholders filled, deepstack adds) through language layers 0-17, their K / V | once, 249-258 tokens (dynamic) |
| `prefix_b` | language layers 18-35, their K / V | once |
| `step` | one Euler step of the DiT over the prompt K / V, with pi0.5's inpainting for real-time chunking | 5 times, as a CUDA graph |

`Xr1Policy` reproduces the RoboCasa client and the slow processor on the host, checked bit-exact against them on
RoboCasa renders: each view is centre-cropped by `crop_ratio` (0.95) and resized back with PIL's bilinear filter in
8 bits, then cut into Qwen3-VL patches by the slow processor's path (rescale in float64, cast, normalize in float32,
two identical temporal frames, merge-blocked order); the prompt is the chat template's ids around the instruction
(identical on every instruction checked) with `get_rope_index`'s multimodal RoPE ids. The state is zero-padded
unnormalized to 60; the chunk starts from the official request's fixed noise (`torch.manual_seed(42)` on a
bfloat16 action mask, which the official server redraws identically on every call) and the active dims (`std >
1e-5`) are decoded with the robot type's mean / std.

**Precision.** The language model's residual stream reaches 5e2 in layer 0's keys; RMS norms (including the
per-head q / k norms), RoPE and attention run in FP32, the time conditioning is a precomputed FP32 table per step,
and the DiT's fused QKV projection is split into three GEMMs (TensorRT 10.3 miscomputes a fused projection viewed
and indexed into Q / K / V). The official "FP32" needs care: `from_pretrained(dtype=float32)` leaves the language
model in bfloat16 (its sub-config pins it) and the remote code casts the time embedding to bfloat16, which
`xr1_reference.py --precision fp32` both undoes.

**Real-time chunking.** As pi0.5's, for a flow from noise at t = 0: after every Euler step, row r of the overlap
moves towards (1 - t) * noise + t * previous[start_row + r] by weight 1 on the frozen rows and an exponential ramp
after them.

## Accuracy and speed

AGX Orin 64 GB, JetPack 6.2, three RoboCasa observations (PnPCounterToCab, OpenSingleDoor, TurnOnSinkFaucet, seed
0-2), against the official policy in true FP32 (CUDA) with the official noise; the official BF16 server lands
0.002-0.022 from the same reference in robot actions:

| Stage | max \|Δ\| |
|---|---|
| Vision tokens / deepstack, from the official patches | 0.04-0.05 / 0.03 on values up to 5.7 |
| Prompt K / V of layers 0, 17, 35, from the official image tokens | 1.4-2.0 / 0.003, 0.24-0.38 / 0.06, 0.47-1.3 / 0.35-1.4 on values up to 537 |
| Normalized chunk (60 dims), chained | 0.007-0.067 (official BF16: 0.028-0.108) |
| `xr1_policy_server` robot actions from raw frames | 0.002-0.013, cosine 0.99997-0.999999 |
| Host inputs vs the official client and processor | ids, RoPE ids and patches identical |

Swapping the left base view and the wrist view moves the actions by up to 1.4. A call takes about 210 ms (median
of 10 on a RoboCasa observation, the desktop drawing about 20% of the GPU): vision 38 ms, language model 106 ms,
5 DiT steps 56 ms, host 4 ms.

RoboCasa v0.2, [`scripts/robocasa_eval.py`](../vla/scripts/robocasa_eval.py) running the official rollout (each
task's step limit, 10 rows per call, episode seeds from 0, the official noise) against the server, 24 tasks x 5
episodes on this board; the authors report 100 episodes per task from eight GPUs. Per task:

| Task | Orin (5 episodes) | Reported (100 episodes) |
|---|:---:|:---:|
| PnPCounterToCab | 5/5 | 56% |
| PnPCabToCounter | 4/5 | 64% |
| PnPCounterToSink | 3/5 | 59% |
| PnPSinkToCounter | 2/5 | 71% |
| PnPCounterToMicrowave | 1/5 | 30% |
| PnPMicrowaveToCounter | 3/5 | 37% |
| PnPCounterToStove | 2/5 | 73% |
| PnPStoveToCounter | 2/5 | 62% |
| OpenSingleDoor | 5/5 | 87% |
| CloseSingleDoor | 5/5 | 100% |
| OpenDoubleDoor | 5/5 | 99% |
| CloseDoubleDoor | 5/5 | 94% |
| OpenDrawer | 5/5 | 77% |
| CloseDrawer | 5/5 | 96% |
| TurnOnStove | 4/5 | 82% |
| TurnOffStove | 0/5 | 29% |
| TurnOnSinkFaucet | 5/5 | 91% |
| TurnOffSinkFaucet | 5/5 | 97% |
| TurnSinkSpout | 4/5 | 86% |
| CoffeeSetupMug | 2/5 | 42% |
| CoffeeServeMug | 3/5 | 77% |
| CoffeePressButton | 5/5 | 82% |
| TurnOnMicrowave | 5/5 | 91% |
| TurnOffMicrowave | 5/5 | 99% |
| **Average** | **75.0%** | **74.2%** |

Policy calls during the run: 197 ms median (vision 37, language model 101, DiT 55, host 4).

```bash
# Official reference env: a CUDA PyTorch with transformers 4.57 (the checkpoint's remote code); obs.npz holds
# views_i [3, 256, 256, 3] (the client's upright renders), state_i and task_i
python experimental_models/xr1/scripts/xr1_reference.py --checkpoint Xiaomi-Robotics-1-RoboCasa \
    --obs obs.npz --out ref.npz --capture
for stage in vision prefix_a prefix_b step; do
    python experimental_models/xr1/scripts/export_xr1.py --checkpoint Xiaomi-Robotics-1-RoboCasa \
        --out onnx --stage $stage [--check ref.npz]
done
trtexec --onnx=onnx/vision/model.onnx --saveEngine=engines/vision.engine --stronglyTyped --noTF32
P=input_ids:1xN,position_ids:3xN
trtexec --onnx=onnx/prefix_a/model.onnx --saveEngine=engines/prefix_a.engine --stronglyTyped --noTF32 \
    --minShapes=${P//N/200} --optShapes=${P//N/256} --maxShapes=${P//N/384}
P=hidden:1xNx2560,position_ids:3xN
trtexec --onnx=onnx/prefix_b/model.onnx --saveEngine=engines/prefix_b.engine --stronglyTyped --noTF32 \
    --minShapes=${P//N/200} --optShapes=${P//N/256} --maxShapes=${P//N/384}
P=prefix_keys:36x1xNx8x128,prefix_values:36x1xNx8x128,prefix_mask:1xN
trtexec --onnx=onnx/step/model.onnx --saveEngine=engines/step.engine --stronglyTyped --noTF32 \
    --minShapes=${P//N/200} --optShapes=${P//N/256} --maxShapes=${P//N/384}
python experimental_models/xr1/scripts/export_xr1.py --checkpoint Xiaomi-Robotics-1-RoboCasa \
    --out engines --stage assets --tasks instructions.txt
python experimental_models/xr1/scripts/run_xr1_engines.py --engines engines --reference ref.npz
xr1_inputs_dump engines out "open the microwave door" left.png right.png wrist.png   # the host's inputs
xr1_policy_server --engineDir engines [--port 5555]
MUJOCO_GL=egl python experimental_models/vla/scripts/robocasa_eval.py --official Xiaomi-Robotics-1 \
    --episodes 5 --server-cmd "xr1_policy_server --engineDir engines" --out robocasa.json
```

The profiles cover prompts of 200-384 tokens: three views take 192 image tokens and the template, the captions and
a RoboCasa instruction another 57-66. The checkpoint and code are under the Apache License 2.0.
