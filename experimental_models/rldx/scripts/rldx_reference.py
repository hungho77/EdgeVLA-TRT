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
"""Golden actions from the official RLDX-1 policy (RLWRLD/RLDX-1, the LIBERO server path), stage by stage.

Runs RLDXSimPolicyWrapper(RLDXPolicy) on observations shaped as the official LIBERO client sends them: per camera
the frames at t-6, t-4, t-2 and t, the state as x/y/z, roll/pitch/yaw (axis-angle) and the two gripper qpos, and the
task. The initial noise is fixed (saved with the outputs) so the engines can be fed the same one.

    RLDX_ATTN_IMPL=sdpa PYTHONPATH=RLDX-1 python rldx_reference.py --checkpoint RLDX-1-FT-LIBERO --obs obs.npz \\
        --out ref.npz [--precision bf16|fp32]

obs.npz holds agentview_<i> / wrist_<i> ([7, H, W, 3] raw LIBERO renders of consecutive steps, the last being t;
rotated 180 degrees here as the official env does), state_<i> and task_<i>. Per sample the output holds the
processor's input_ids, pixel_values and image_grid_thw, the visual embeddings and deepstack features, the hidden
state after LLM layer 3 (before the video-token compression) and after the last layer (compressed), the 64
cognition features the action model reads, the state features, each Euler step's input and velocity, the
normalized action chunk, and the decoded actions. ``bf16`` is the official setting (BF16 weights under autocast);
``fp32`` upcasts the weights and turns autocast off.
"""

import argparse

import numpy as np
import torch

HISTORY = (-7, -5, -3, -1)  # t-6, t-4, t-2, t in a 7-frame window


def load_policy(args):
    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.policy import policy_runtime
    from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

    policy = RLDXPolicy(embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
                        model_path=args.checkpoint,
                        device=args.device,
                        strict=False)
    if args.precision == "fp32":
        policy.model.float()
        cast = policy_runtime._rec_to_dtype
        policy_runtime._rec_to_dtype = lambda value, dtype: cast(
            value, dtype=torch.float32)

        def forward_fp32(self, collated):
            with torch.inference_mode(), torch.autocast(device_type="cuda",
                                                        enabled=False):
                return self.model.get_action(**collated)

        policy_runtime.PolicyRuntime._forward = forward_fp32
    return policy, RLDXSimPolicyWrapper(policy, strict=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--obs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision",
                        default="bf16",
                        choices=("bf16", "fp32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy, wrapper = load_policy(args)
    model = policy.model
    backbone = model.backbone
    qwen = backbone.qwen_model.model
    captured = {}

    def keep(name):

        def hook(module, inputs, output):
            captured.setdefault(name, []).append(output)

        return hook

    qwen.visual.register_forward_hook(keep("visual"))
    layers = qwen.language_model.layers
    layers[3].register_forward_hook(keep("layer3"))
    layers[-1].register_forward_hook(keep("last_layer"))

    def keep_inputs(module, inputs):
        captured["backbone_inputs"] = [inputs[0]]

    def keep_step_input(module, inputs):
        captured.setdefault("step_input", []).append(inputs[0])

    backbone.register_forward_pre_hook(keep_inputs)
    backbone.register_forward_hook(keep("backbone"))
    model.action_model.action_encoder.register_forward_pre_hook(
        keep_step_input)
    model.action_model.action_decoder.register_forward_hook(
        keep("step_velocity"))
    model.action_model.state_encoder.register_forward_hook(keep("state"))

    randn = torch.randn

    def fixed_noise(*size, **kwargs):
        shape = tuple(kwargs.pop("size", size))
        if len(shape) == 3 and shape[1:] == (16, 64):
            return captured["noise"].to(kwargs.get("device"),
                                        kwargs.get("dtype"))
        return randn(*size, **kwargs)

    obs = np.load(args.obs)
    out = {}
    count = len([k for k in obs.files if k.startswith("state_")])
    generator = torch.Generator().manual_seed(args.seed)
    for i in range(count):
        captured.clear()
        captured["noise"] = randn((1, 16, 64), generator=generator)
        views = {
            name:
            np.stack([obs[f"{key}_{i}"][h][::-1, ::-1] for h in HISTORY])[None]
            for name, key in (("video.image", "agentview"),
                              ("video.wrist_image", "wrist"))
        }
        state = obs[f"state_{i}"].astype(np.float64)
        observation = {
            **views,
            "state.x": state[None, None, 0:1],
            "state.y": state[None, None, 1:2],
            "state.z": state[None, None, 2:3],
            "state.roll": state[None, None, 3:4],
            "state.pitch": state[None, None, 4:5],
            "state.yaw": state[None, None, 5:6],
            "state.gripper": state[None, None, 6:8],
            "annotation.human.action.task_description":
            (str(obs[f"task_{i}"]), ),
        }
        torch.randn = fixed_noise
        try:
            actions, _ = wrapper.get_action(observation, {
                "reset_memory": [True],
                "session_ids": ["golden"]
            })
        finally:
            torch.randn = randn

        def np32(t):
            return t.detach().float().cpu().numpy()

        inputs = captured["backbone_inputs"][0]
        for key in ("input_ids", "pixel_values", "image_grid_thw"):
            out[f"{key}_{i}"] = inputs[key].cpu().numpy(
            ) if not inputs[key].is_floating_point() else np32(inputs[key])
        embeds, deepstack = captured["visual"][0][:2]
        out[f"visual_{i}"] = np32(embeds)
        out[f"deepstack_{i}"] = np.stack([np32(d) for d in deepstack])
        out[f"layer3_{i}"] = np32(captured["layer3"][0][0] if isinstance(
            captured["layer3"][0], tuple) else captured["layer3"][0])
        last = captured["last_layer"][0]
        out[f"last_layer_{i}"] = np32(
            last[0] if isinstance(last, tuple) else last)
        out[f"cognition_{i}"] = np32(
            captured["backbone"][0]["backbone_features"])
        out[f"state_features_{i}"] = np32(captured["state"][0])
        out[f"noise_{i}"] = captured["noise"].numpy()
        out[f"step_input_{i}"] = np.stack(
            [np32(x) for x in captured["step_input"]])
        out[f"step_velocity_{i}"] = np.stack(
            [np32(v[:, -16:]) for v in captured["step_velocity"]])
        out[f"actions_{i}"] = np.concatenate([
            np.asarray(actions[f"action.{k}"], np.float32)[0]
            for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ], -1)
        print(
            f"{i}: {str(obs[f'task_{i}'])!r}  {out[f'input_ids_{i}'].shape[1]} tokens, "
            f"action row 0 {np.round(out[f'actions_{i}'][0], 4)}")
    np.savez(args.out, **out)
    print(f"{count} samples ({args.precision}) -> {args.out}")


if __name__ == "__main__":
    main()
