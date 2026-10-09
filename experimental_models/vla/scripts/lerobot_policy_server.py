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
"""LeRobot's own PyTorch policy behind the EdgeVLA-TRT policy-server protocol, as a reference.

Speaks the request / reply format of smolvla_policy_server and xvla_policy_server ({"cameras": {name: path},
"state", "task", "reset"} -> {"actions": [[...], ...]}), so libero_eval.py can run the reference and the engines
under one protocol and the same initial states:

    libero_eval.py --policy smolvla --server-cmd "python lerobot_policy_server.py --checkpoint smolvla_libero" ...

Runs the checkpoint's saved pre / post-processors and predict_action_chunk; needs a LeRobot install.
"""

import argparse
import json
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import (get_policy_class,
                                          make_pre_post_processors)
    from PIL import Image

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    if hasattr(config, "load_vlm_weights"):
        config.load_vlm_weights = False  # the checkpoint carries every weight
    config.device = args.device
    policy = get_policy_class(config.type).from_pretrained(args.checkpoint,
                                                           config=config)
    policy = policy.to(args.device).to(torch.float32).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {
            "device": args.device
        }})
    print(json.dumps({"ready": True, "policy": config.type}), flush=True)

    for line in sys.stdin:
        if not line.strip():
            break
        try:
            request = json.loads(line)
            batch = {
                f"observation.images.{name}":
                torch.from_numpy(
                    np.asarray(Image.open(path).convert("RGB"))).permute(
                        2, 0, 1).float()[None] / 255.0
                for name, path in request["cameras"].items()
            }
            batch["observation.state"] = torch.tensor(
                request["state"], dtype=torch.float32)[None]
            batch["task"] = [request["task"]]
            with torch.no_grad():
                actions = postprocess(
                    policy.predict_action_chunk(preprocess(batch)))
            reply = {
                "actions": actions[0].detach().float().cpu().numpy().tolist()
            }
        except Exception as error:  # noqa: BLE001 -- reported to the client
            reply = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
