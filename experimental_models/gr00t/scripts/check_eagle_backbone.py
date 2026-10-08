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
"""Eager parity of the plain-op Eagle backbone against the official GR00T N1.6 features.

Feeds the pixel values and token ids captured by ``official_reference.py`` to
``tensorrt_edgellm.models.eagle`` (FP32, or FP16 with --fp16) and compares the backbone features.
The official policy rounds every weight to bf16 when it loads (the fine-tuned top LLM layers are
stored in FP32); --bf16-weights does the same, to separate that rounding from porting errors.

    python check_eagle_backbone.py --checkpoint GR00T-N1.6-SO101-Multitask --reference ref_f300.npz
"""

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

from tensorrt_edgellm.models.eagle.modeling_eagle import (EagleConfig,
                                                          EaglePrefix,
                                                          EagleVisual,
                                                          load_eagle_weights)


def load_backbone_state(checkpoint):
    weight_map = json.load(
        open(os.path.join(checkpoint,
                          "model.safetensors.index.json")))["weight_map"]
    state = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(checkpoint, shard), "pt") as f:
            for key in f.keys():
                if key.startswith(
                        "backbone.model."
                ) and "lm_head" not in key and ".head." not in key:
                    state[key] = f.get_tensor(key).float()
    return state


def cosine(a, b):
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16-weights", action="store_true")
    args = parser.parse_args()

    ref = np.load(args.reference)
    pixels = np.concatenate([
        ref[k] for k in sorted(f for f in ref.files
                               if f.startswith("backbone_input.pixel_values."))
    ])
    cfg = EagleConfig(
        image_height=pixels.shape[2],
        image_width=pixels.shape[3],
        text_layers=int(
            json.load(open(os.path.join(args.checkpoint,
                                        "config.json")))["select_layer"]))
    visual, prefix = EagleVisual(cfg), EaglePrefix(cfg)
    state = load_backbone_state(args.checkpoint)
    if args.bf16_weights:
        state = {k: v.bfloat16().float() for k, v in state.items()}
    load_eagle_weights(state, visual, prefix)
    dtype = torch.float16 if args.fp16 else torch.float32
    device = "cuda" if args.fp16 and torch.cuda.is_available() else "cpu"
    visual, prefix = visual.to(device, dtype).eval(), prefix.to(device,
                                                                dtype).eval()

    with torch.no_grad():
        image = visual(torch.from_numpy(pixels).to(device, dtype))
        features = prefix(
            torch.from_numpy(ref["input_ids"])[None].to(device),
            image.reshape(1, -1, cfg.text_hidden))[0].float().cpu().numpy()
    expected, mask = ref["backbone_features"], ref["image_mask"]
    label = ("fp16" if args.fp16 else "fp32") + (", bf16 weights"
                                                 if args.bf16_weights else "")
    print(
        f"{label} backbone features vs official: cosine {cosine(features, expected):.6f}, "
        f"max|d| {np.abs(features - expected).max():.4f} (|ref| max {np.abs(expected).max():.1f}); "
        f"image tokens cosine {cosine(features[mask], expected[mask]):.6f}, "
        f"text tokens cosine {cosine(features[~mask], expected[~mask]):.6f}")


if __name__ == "__main__":
    main()
