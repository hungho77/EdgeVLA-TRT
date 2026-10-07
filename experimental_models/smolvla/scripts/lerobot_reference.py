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
"""LeRobot reference for one SmolVLA request: raw observation in, actions and model inputs out.

Runs LeRobot's own ``SmolVLAPolicy`` with the checkpoint's saved pre/post-processors, in FP32 on
the CPU, from a seeded x_0 that is also written out, and records what the model saw (preprocessed
images, tokens, normalized state) and the normalized chunk next to the robot actions.

    python lerobot_reference.py --checkpoint smolvla-so101-multitask --observation obs.json \\
        --out ref.npz

obs.json: {"task": "...", "state": [...], "cameras": {"top": "top.png", "wrist": "wrist.png"}} with
the dataset's camera names (the preprocessor renames them). Needs ``lerobot[smolvla]``; the VLM
base it names must be in the Hugging Face cache (set HF_HUB_OFFLINE=1 when offline).
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
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # The checkpoint carries every VLM weight; the base model is needed for its config only.
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.load_vlm_weights = False
    config.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(
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
        torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(
            2, 0, 1).float() / 255.0
        for name, path in request["cameras"].items()
    }
    batch["observation.state"] = torch.tensor(request["state"],
                                              dtype=torch.float32)
    batch["task"] = request["task"]
    batch = preprocess(batch)

    config = policy.config
    noise = torch.from_numpy(
        np.random.default_rng(args.seed).standard_normal(
            (1, config.chunk_size, config.max_action_dim)).astype(np.float32))
    captured = {}
    model = policy.model
    sample_actions = model.sample_actions

    def record(images, img_masks, lang_tokens, lang_masks, state, *a, **kw):
        captured["images"] = [i.detach().float().numpy() for i in images]
        captured["img_masks"] = [bool(m.reshape(-1)[0]) for m in img_masks]
        captured["lang_tokens"] = lang_tokens.detach().numpy()
        captured["lang_masks"] = lang_masks.detach().numpy()
        captured["state"] = state.detach().float().numpy()
        out = sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                             *a, **kw)
        captured["normalized"] = out.detach().float().numpy()
        return out

    model.sample_actions = record
    with torch.no_grad():
        actions = policy.predict_action_chunk(batch, noise=noise)
        robot = postprocess(actions)
    robot = robot.detach().float().numpy() if hasattr(
        robot, "detach") else np.asarray(robot)
    np.savez(args.out,
             noise=noise.numpy(),
             images=np.stack(captured["images"]),
             img_masks=np.array(captured["img_masks"]),
             lang_tokens=captured["lang_tokens"],
             lang_masks=captured["lang_masks"],
             state=captured["state"],
             normalized=captured["normalized"],
             robot_actions=robot)
    print(
        f"images {np.stack(captured['images']).shape} (masks {captured['img_masks']}), "
        f"tokens {int(captured['lang_masks'].sum())}/{captured['lang_tokens'].shape[-1]}, "
        f"robot actions {robot.shape} -> {args.out}")


if __name__ == "__main__":
    main()
