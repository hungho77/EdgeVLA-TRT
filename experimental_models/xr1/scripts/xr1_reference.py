# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Golden actions from the official Xiaomi-Robotics-1 policy (XiaomiRobotics/Xiaomi-Robotics-1, its RoboCasa
evaluation: eval_robocasa/main.py and deploy/server.py).

The checkpoint's own remote code (MiBoTForActionGeneration) and processor are loaded with trust_remote_code, and a
request is built as the RoboCasa client builds it: the three views (two base cameras, the left wrist) centre-cropped
by --crop-ratio and resized back with PIL's bilinear filter, the chat-template messages with the views' captions and
"<cot></cot>" as the assistant turn, the robot state zero-padded to the model's state width, the processor's action
mask for --robot-type. The official server seeds the flow's noise with the request's seed on every call
(torch.manual_seed then randn_like on its bfloat16 action mask), so the noise is the same at every step; it is
reproduced here and handed to every precision unchanged.

    python xr1_reference.py --checkpoint Xiaomi-Robotics-1-RoboCasa --obs obs.npz --out ref.npz \\
        [--precision fp32|bf16] [--capture]

obs.npz holds views_<i> [3, H, W, 3] uint8 (the client's renders, already flipped upright), state_<i> and task_<i>.
``bf16`` is the official server's precision. ``fp32`` upcasts the bfloat16 checkpoint, the language model included
(its sub-config pins bfloat16), keeps the time embedding in FP32 (the remote code casts it to bfloat16) and IEEE FP32
matmuls. ``--capture`` also keeps the stage tensors the
export checks against (vision tokens, deepstack features, K / V of language layers 0, 17 and 35, the RoPE ids and
the first velocity).
"""

import argparse

import numpy as np
import torch

PROMPT_HEAD = "The following observations are captured from multiple views.\n# Base View\n"
PROMPT_WRIST = "\n# Left-Wrist View\n"
PROMPT_TASK = "\nGenerate robot actions for the task:\n{task} /no_cot"
ASSISTANT = "<cot></cot>"


def load_policy(checkpoint, precision):
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(checkpoint,
                                              trust_remote_code=True,
                                              use_fast=False)
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    # Built on the GPU: the Orin shares DRAM, and a CPU copy moved over would double the model.
    with torch.device("cuda"):
        model = AutoModel.from_pretrained(checkpoint,
                                          trust_remote_code=True,
                                          attn_implementation="eager",
                                          dtype=dtype)
    model = model.cuda().eval()
    if precision == "fp32":
        # The language model's sub-config pins bfloat16, which from_pretrained(dtype=float32) keeps.
        model.float()
        torch.set_float32_matmul_precision("highest")
        model.t_embedder.dtype = torch.float32
    return model, processor


def center_crop(images, ratio):
    """eval_robocasa/main.py center_crop_pil."""
    from PIL import Image
    if ratio >= 1.0:
        return images
    out = []
    for img in images:
        w, h = img.size
        nw, nh = int(w * ratio), int(h * ratio)
        left, top = (w - nw) // 2, (h - nh) // 2
        out.append(
            img.crop((left, top, left + nw, top + nh)).resize((w, h),
                                                              Image.BILINEAR))
    return out


def build_request(processor, views, state, task, robot_type, crop_ratio,
                  state_width):
    from PIL import Image
    images = center_crop([Image.fromarray(v) for v in views], crop_ratio)
    padded = np.zeros(state_width, np.float32)
    padded[:len(state)] = state
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT_HEAD
            },
            {
                "type": "image",
                "image": images[0]
            },
            {
                "type": "image",
                "image": images[1]
            },
            {
                "type": "text",
                "text": PROMPT_WRIST
            },
            {
                "type": "image",
                "image": images[2]
            },
            {
                "type": "text",
                "text": PROMPT_TASK.format(task=task)
            },
        ],
    }, {
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": ASSISTANT
        }]
    }]
    return dict(
        processor.apply_chat_template(messages,
                                      tokenize=True,
                                      return_dict=True,
                                      return_tensors="pt",
                                      state=padded.reshape(1, 1, state_width),
                                      robot_type=robot_type))


def official_noise(shape, seed):
    """The server's noise: torch.manual_seed(seed), then randn_like on its bfloat16 action mask on the GPU."""
    torch.manual_seed(seed)
    return torch.randn_like(
        torch.empty(shape, dtype=torch.bfloat16, device="cuda")).float()


def install_capture(model, captured):
    to_np = lambda x: x.detach().float().cpu().numpy()
    vlm_forward = model.vlm.forward

    def vlm(*a, **k):
        out = vlm_forward(*a, **k)
        cache = out.past_key_values
        for layer in (0, 17, 35):
            keys, values = cache[layer][0], cache[layer][1]
            captured[f"k{layer}"] = to_np(keys)
            captured[f"v{layer}"] = to_np(values)
        captured["position_ids"] = out.position_ids.cpu().numpy()
        return out

    model.vlm.forward = vlm
    features = model.vlm.model.get_image_features

    def image_features(*a, **k):
        tokens, deepstack = features(*a, **k)
        captured["vision"] = np.concatenate([to_np(t) for t in tokens])
        captured["deepstack"] = np.stack([to_np(d) for d in deepstack])
        return tokens, deepstack

    model.vlm.model.get_image_features = image_features
    dit_forward = model.dit_forward

    def dit(*a, **k):
        v = dit_forward(*a, **k)
        captured.setdefault("velocity0", to_np(v))
        return v

    model.dit_forward = dit


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--obs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision",
                        choices=["fp32", "bf16"],
                        default="fp32")
    parser.add_argument("--robot-type", default="robocasa_mg")
    parser.add_argument("--crop-ratio", type=float, default=0.95)
    parser.add_argument("--steps",
                        type=int,
                        default=5,
                        help="Euler steps (the server's default)")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the request's seed (the RoboCasa client sends 42)")
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()

    model, processor = load_policy(args.checkpoint, args.precision)
    dtype = next(model.parameters()).dtype
    captured = {}
    if args.capture:
        install_capture(model, captured)
    state_width = int(model.config.state_dim)
    obs = np.load(args.obs)
    count = len([k for k in obs.files if k.startswith("state_")])
    out = {}
    for i in range(count):
        captured.clear()
        request = build_request(processor,
                                obs[f"views_{i}"], obs[f"state_{i}"],
                                str(obs[f"task_{i}"]), args.robot_type,
                                args.crop_ratio, state_width)
        noise = official_noise(request["action_mask"].shape, args.seed)
        inputs = {
            k: (v.to("cuda", dtype) if v.is_floating_point() else v.cuda())
            for k, v in request.items()
        }
        original = torch.randn_like
        torch.randn_like = lambda t, *a, **k: noise.to(t.device, t.dtype)
        try:
            with torch.no_grad():
                normalized = model(**inputs,
                                   num_steps=args.steps,
                                   seed=args.seed).actions
        finally:
            torch.randn_like = original
        robot = processor.decode_action(normalized.float().cpu(),
                                        robot_type=args.robot_type)
        out.update({
            f"{k}_{i}":
            v.float().numpy() if v.is_floating_point() else v.numpy()
            for k, v in request.items()
        })
        out.update({f"{k}_{i}": v for k, v in captured.items()})
        out[f"noise_{i}"] = noise[0].cpu().numpy()
        out[f"normalized_{i}"] = normalized[0].float().cpu().numpy()
        out[f"actions_{i}"] = np.asarray(robot[0], np.float32)
        print(
            f"{i}: {str(obs[f'task_{i}'])!r} ({request['input_ids'].shape[1]} tokens)\n"
            f"  robot row 0 {np.round(out[f'actions_{i}'][0][:8], 4)}")
    np.savez(args.out, **out)
    print(f"{count} samples ({args.precision}) -> {args.out}")


if __name__ == "__main__":
    main()
