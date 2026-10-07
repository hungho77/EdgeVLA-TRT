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
"""Eager parity of the SmolVLA export modules against a lerobot_reference.py capture.

Runs visual -> prefix -> 10 denoise steps in PyTorch on the model inputs LeRobot itself
built (preprocessed images, token ids, normalized state, x_0) and compares the normalized
chunk with LeRobot's. Needs only torch and safetensors.

    python check_smolvla_modules.py --checkpoint smolvla-so101-multitask --reference ref.npz
"""

import argparse
import importlib.util
import os
import sys

import numpy as np


def load_modeling():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        here, "../../../tensorrt_edgellm/models/smolvla/modeling_smolvla.py")
    spec = importlib.util.spec_from_file_location("modeling_smolvla", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modeling_smolvla"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file

    m = load_modeling()
    cfg = m.SmolVLAConfig()
    visual, prefix, denoise = m.SmolVLAVisual(cfg), m.SmolVLAPrefix(
        cfg), m.SmolVLADenoise(cfg)
    m.load_smolvla_weights(
        load_file(os.path.join(args.checkpoint, "model.safetensors")), visual,
        prefix, denoise)
    for module in (visual, prefix, denoise):
        module.eval()

    ref = np.load(args.reference)
    images = torch.from_numpy(ref["images"]).float()  # [cameras, B, 3, H, W]
    tokens = torch.from_numpy(ref["lang_tokens"])[torch.from_numpy(
        ref["lang_masks"]).bool()][None]
    state = torch.from_numpy(ref["state"]).float()
    x = torch.from_numpy(ref["noise"]).float()
    with torch.no_grad():
        features = torch.cat(
            [visual(images[c]) for c in range(images.shape[0])], dim=1)
        prefix_kv = prefix(features, tokens, state)
        dt = torch.tensor(-1.0 / args.steps)
        for step in range(args.steps):
            x = denoise(x, torch.full((x.shape[0], ), 1.0 + step * float(dt)),
                        dt, *prefix_kv)
    ours = x.numpy().astype(np.float64)
    theirs = ref["normalized"].astype(np.float64)
    cos = float(
        (ours * theirs).sum() / np.linalg.norm(ours) / np.linalg.norm(theirs))
    print(
        f"prefix {features.shape[1]} image + {tokens.shape[1]} language + 1 state tokens; normalized chunk "
        f"{ours.shape}: cosine {cos:.6f}, max |d| {np.abs(ours - theirs).max():.2e}"
    )


if __name__ == "__main__":
    main()
