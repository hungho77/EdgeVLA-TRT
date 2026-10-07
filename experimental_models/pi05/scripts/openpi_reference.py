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
"""openpi reference for one pi0.5 request, in the pi05_policy_inference request format.

Runs openpi's own ``create_trained_policy`` for a training configuration on the raw
observation (cameras keyed by their openpi names, as in policy.json), in FP32 on the
CPU, from a seeded x_0 that is also written out for ``pi05_policy_inference --noise``.

    python openpi_reference.py --config pi05_so101 --checkpoint <openpi PyTorch checkpoint> \\
        --observation obs.json --out ref.json --noise-out x0.bin

ref.json holds ``actions`` (the normalized [horizon, 32] chunk) and ``robot_actions``
(openpi's output transforms applied), so either compare_pi05_actions.py field can be scored.
Needs an environment with openpi installed (PyTorch model support enabled).
"""

import argparse
import json

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",
                        required=True,
                        help="openpi TrainConfig name")
    parser.add_argument("--checkpoint",
                        required=True,
                        help="directory with model.safetensors and assets/")
    parser.add_argument("--observation",
                        required=True,
                        help="pi05_policy_inference request JSON")
    parser.add_argument("--state-key", default="observation/state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps",
                        type=int,
                        default=None,
                        help="num_steps for sample_actions")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help=
        "run openpi's own bfloat16 precision instead of FP32 (its noise floor)"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--noise-out",
                        required=True,
                        help="x_0 as raw fp32 [1, horizon, 32]")
    parser.add_argument(
        "--inputs-dir",
        help=
        "also write the model inputs openpi built, for pi05_policy_inference's "
        "canonical mode: pixel_values.bin (fp16 [views, 3, S, S]) and token_ids.csv"
    )
    args = parser.parse_args()

    import importlib.util
    import sys
    import types

    import torch

    # openpi's checkpoint module imports its training data loader, which imports LeRobot at
    # module scope; inference never reaches it, so a missing LeRobot is stubbed out.
    if importlib.util.find_spec("lerobot") is None:
        for name in ("lerobot", "lerobot.common", "lerobot.common.datasets",
                     "lerobot.common.datasets.lerobot_dataset"):
            sys.modules[name] = types.ModuleType(name)
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    # create_trained_policy casts the model to bfloat16, which also rounds buffers such as the
    # RoPE inverse frequencies; casting back cannot restore them. Keep everything FP32 instead.
    if not args.bf16:
        to_precision = PaliGemmaWithExpertModel.to_bfloat16_for_selected_params
        PaliGemmaWithExpertModel.to_bfloat16_for_selected_params = (
            lambda self, precision="bfloat16": to_precision(self, "float32"))
    train_config = openpi_config.get_config(args.config)
    policy = policy_config.create_trained_policy(train_config,
                                                 args.checkpoint,
                                                 pytorch_device="cpu")
    if not args.bf16:
        policy._model.to(torch.float32)

    request = json.load(open(args.observation))
    observation = {
        name: np.asarray(Image.open(path).convert("RGB"))
        for name, path in request["cameras"].items()
    }
    observation[args.state_key] = np.asarray(request["state"],
                                             dtype=np.float32)
    observation["prompt"] = request["task"]

    model = train_config.model
    noise = np.random.default_rng(args.seed).standard_normal(
        (1, model.action_horizon, model.action_dim)).astype(np.float32)
    noise.tofile(args.noise_out)

    captured = {}
    sample_actions = policy._sample_actions

    def record(device, observation, **kw):
        if args.steps is not None:
            kw["num_steps"] = args.steps
        captured["observation"] = observation
        a = (device, observation)
        actions = sample_actions(*a, **kw)
        captured["actions"] = actions[0].detach().float().cpu().numpy()
        return actions

    policy._sample_actions = record
    with torch.no_grad():
        result = policy.infer(observation, noise=noise)
    if args.inputs_dir:
        import os
        os.makedirs(args.inputs_dir, exist_ok=True)
        obs = captured["observation"]
        views = [
            obs.images[k][0].float().cpu().numpy() for k in obs.images
            if bool(obs.image_masks[k][0])
        ]
        np.stack([
            v.transpose(2, 0, 1) if v.shape[-1] == 3 else v for v in views
        ]).astype(np.float16).tofile(
            os.path.join(args.inputs_dir, "pixel_values.bin"))
        tokens = obs.tokenized_prompt[0][
            obs.tokenized_prompt_mask[0]].cpu().numpy()
        open(os.path.join(args.inputs_dir, "token_ids.csv"),
             "w").write(",".join(map(str, tokens)))
        print(
            f"inputs: {len(views)} views {views[0].shape}, {len(tokens)} tokens -> {args.inputs_dir}"
        )
    json.dump(
        {
            "actions":
            captured["actions"].tolist(),
            "robot_actions":
            np.asarray(result["actions"], dtype=np.float64).tolist(),
        }, open(args.out, "w"))
    print(
        f"{args.config}: normalized {captured['actions'].shape}, robot "
        f"{np.asarray(result['actions']).shape} -> {args.out}; x_0 -> {args.noise_out}"
    )


if __name__ == "__main__":
    main()
