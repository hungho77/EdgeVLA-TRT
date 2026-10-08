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
"""LeRobot reference for one X-VLA request: raw observation in, actions and every stage's tensors out.

Runs LeRobot's own ``XVLAPolicy`` with the checkpoint's saved pre/post-processors in FP32 on the CPU. Its
sampler draws x1 with ``torch.randn`` and ignores a passed noise, so the seeded x1 is injected there. Records
the model inputs (token ids, resized views, view mask, domain id, padded state), the Florence-2 encoder output
and auxiliary view features, the first denoising step's inputs, the action-space output and the robot actions.

    python lerobot_xvla_reference.py --checkpoint xvla-base --observation obs.json --out ref.npz

obs.json: {"task": "...", "state": [...], "cameras": {"image": "a.png", "image2": "b.png"}} keyed by the
checkpoint's image feature names (observation.images.<name>).
"""

import argparse
import json

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.xvla import modeling_xvla
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = "cpu"
    policy = XVLAPolicy.from_pretrained(
        args.checkpoint, config=config).to("cpu").to(torch.float32).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {
            "device": "cpu"
        }})

    request = json.load(open(args.observation))
    batch = {
        f"observation.images.{name}":
        torch.from_numpy(np.asarray(
            Image.open(path).convert("RGB")).copy()).permute(2, 0, 1).float() /
        255.0
        for name, path in request["cameras"].items()
    }
    batch["observation.state"] = torch.tensor(request["state"],
                                              dtype=torch.float32)
    batch["task"] = request["task"]
    batch = preprocess(batch)

    model = policy.model
    noise = torch.from_numpy(
        np.random.default_rng(args.seed).standard_normal(
            (1, model.chunk_size, model.dim_action)).astype(np.float32))
    captured = {}
    real_randn = torch.randn

    def fixed_randn(*size, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(
            size[0], (tuple, list)) else tuple(size)
        if shape == tuple(noise.shape):
            return noise.clone().to(kwargs.get("dtype") or torch.float32)
        return real_randn(*size, **kwargs)

    generate_actions = model.generate_actions

    def record_generate(input_ids, image_input, image_mask, domain_id, proprio,
                        steps):
        captured.update(input_ids=input_ids[0].numpy(),
                        image_input=image_input[0].float().numpy(),
                        image_mask=image_mask[0].numpy(),
                        domain_id=int(domain_id[0]),
                        proprio=proprio[0].float().numpy())
        out = generate_actions(input_ids, image_input, image_mask, domain_id,
                               proprio, steps)
        captured["model_actions"] = out[0].float().numpy()
        return out

    forward_vlm = model.forward_vlm

    def record_vlm(*a, **kw):
        out = forward_vlm(*a, **kw)
        captured["vlm_features"] = out["vlm_features"][0].float().numpy()
        captured["aux_visual_inputs"] = out["aux_visual_inputs"][0].float(
        ).numpy()
        return out

    transformer_forward = model.transformer.forward

    def record_step(*a, **kw):
        out = transformer_forward(*a, **kw)
        if "step0_out" not in captured:
            captured["step0_x_t"] = kw["action_with_noise"][0].float().numpy()
            captured["step0_proprio"] = kw["proprio"][0].float().numpy()
            captured["step0_t"] = float(kw["t"][0])
            captured["step0_out"] = out[0].float().numpy()
        return out

    model.generate_actions = record_generate
    model.forward_vlm = record_vlm
    model.transformer.forward = record_step
    modeling_xvla.torch.randn = fixed_randn
    try:
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)
            robot = postprocess(actions)
    finally:
        modeling_xvla.torch.randn = real_randn
    robot = robot.detach().float().numpy() if hasattr(
        robot, "detach") else np.asarray(robot)
    np.savez(args.out, noise=noise.numpy(), robot_actions=robot[0], **captured)
    print(
        f"tokens {captured['input_ids'].shape}, views {captured['image_input'].shape} (mask "
        f"{captured['image_mask'].tolist()}), domain {captured['domain_id']}, vlm features "
        f"{captured['vlm_features'].shape}, aux {captured['aux_visual_inputs'].shape}, robot actions {robot[0].shape} "
        f"-> {args.out}")


if __name__ == "__main__":
    main()
