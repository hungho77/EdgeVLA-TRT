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
"""Export OpenVLA's fused DINOv2 + SigLIP vision backbone and projector as one ONNX graph (FP16, batch 1).

Builds the towers as the checkpoint's own ``PrismaticVisionBackbone`` does (timm, second-to-last block's patch
tokens, no final norm) with timm's explicit attention, and the fused-GELU projector, then loads the checkpoint's
weights. Input ``pixel_values`` [1, 6, 224, 224]: the DINOv2- and SigLIP-normalized image stacked on channels;
output ``image_embeds`` [256, 4096], the rows the LLM receives after BOS. Needs timm 0.9.10 (OpenVLA's pin).
Also writes the runtime's config.json: image size and per-tower normalization (the processor's bf16-rounded
constants), prompt template, image placeholder id, action bins and every dataset's action statistics.

    python export_openvla_vision.py --checkpoint openvla-7b --out vision_onnx [--check ref.npz]
"""

import argparse
import json
import os
import sys

import torch
from torch import nn

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from tensorrt_edgellm.onnx.trt_workarounds import \
    apply_trt103_workarounds  # noqa: E402


class OpenVLAVision(nn.Module):

    def __init__(self, config: dict) -> None:
        super().__init__()
        import timm
        import timm.layers

        timm.layers.set_fused_attn(False)
        ids, sizes = config["timm_model_ids"], config["image_sizes"]
        acts = config.get("timm_override_act_layers") or [None, None]
        if not config.get("use_fused_vision_backbone") or len(ids) != 2:
            raise ValueError(
                "only the fused (two-tower) Prismatic backbone is supported")
        self.featurizer = timm.create_model(ids[0],
                                            pretrained=False,
                                            num_classes=0,
                                            img_size=sizes[0],
                                            act_layer=acts[0])
        self.fused_featurizer = timm.create_model(ids[1],
                                                  pretrained=False,
                                                  num_classes=0,
                                                  img_size=sizes[1],
                                                  act_layer=acts[1])
        vision_dim = self.featurizer.embed_dim + self.fused_featurizer.embed_dim
        llm_dim = 4096
        self.fc1 = nn.Linear(vision_dim, 4 * vision_dim)
        self.fc2 = nn.Linear(4 * vision_dim, llm_dim)
        self.fc3 = nn.Linear(llm_dim, llm_dim)

    @staticmethod
    def _penultimate(tower, x):
        return tower.get_intermediate_layers(x, n={len(tower.blocks) - 2})[0]

    def forward(self, pixel_values):
        image, fused = torch.split(pixel_values, [3, 3], dim=1)
        patches = torch.cat([
            self._penultimate(self.featurizer, image),
            self._penultimate(self.fused_featurizer, fused)
        ],
                            dim=2)
        gelu = nn.functional.gelu
        return self.fc3(gelu(self.fc2(gelu(self.fc1(patches)))))[0]


def load_vision(checkpoint: str) -> OpenVLAVision:
    from safetensors import safe_open

    config = json.load(open(os.path.join(checkpoint, "config.json")))
    model = OpenVLAVision(config)
    weight_map = json.load(
        open(os.path.join(checkpoint,
                          "model.safetensors.index.json")))["weight_map"]
    state = {}
    for shard in sorted({
            v
            for k, v in weight_map.items()
            if not k.startswith("language_model.")
    }):
        with safe_open(os.path.join(checkpoint, shard), "pt") as f:
            for key in f.keys():
                if key.startswith("vision_backbone."):
                    # OpenVLA renames timm's LayerScale.gamma to scale_factor (HF overwrites "gamma" parameters).
                    state[key[len("vision_backbone."):].replace(
                        ".scale_factor", ".gamma")] = f.get_tensor(key)
                elif key.startswith("projector."):
                    state[key[len("projector."):]] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # SigLIP's attention-pool head is not used by get_intermediate_layers.
    missing = [
        k for k in missing if ".attn_pool." not in k and ".head" not in k
    ]
    unexpected = [k for k in unexpected if ".attn_pool." not in k]
    if missing or unexpected:
        raise ValueError(
            f"weight mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}"
        )
    return model.float().eval()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out")
    parser.add_argument(
        "--check",
        help="openvla_reference.py output: compare the FP32 module first")
    args = parser.parse_args()

    model = load_vision(args.checkpoint)
    if args.check:
        import numpy as np
        ref = np.load(args.check)
        with torch.no_grad():
            out = model(torch.from_numpy(ref["pixel_values"])[None]).numpy()
        expected = ref["projected"]
        cosine = float(
            np.sum(out * expected) / np.linalg.norm(out) /
            np.linalg.norm(expected))
        print(
            f"FP32 vision + projector vs official: cosine {cosine:.6f}, max|d| {np.abs(out - expected).max():.2e} "
            f"(|ref| max {np.abs(expected).max():.2f})")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.onnx.export(model.half(),
                          (torch.zeros(1, 6, 224, 224, dtype=torch.float16), ),
                          os.path.join(args.out, "vision.onnx"),
                          input_names=["pixel_values"],
                          output_names=["image_embeds"],
                          opset_version=17,
                          dynamo=False)
        rewrites = apply_trt103_workarounds(
            os.path.join(args.out, "vision.onnx"))
        if rewrites:
            print(f"vision: {rewrites} TensorRT 10.3 attention rewrites")
        write_runtime_config(args.checkpoint, args.out)
        print(f"exported -> {args.out}/vision.onnx, config.json")


def write_runtime_config(checkpoint: str, out: str) -> None:
    config = json.load(open(os.path.join(checkpoint, "config.json")))
    processor = json.load(
        open(os.path.join(checkpoint, "preprocessor_config.json")))
    if processor["image_resize_strategy"] != "resize-naive" or processor.get(
            "tvf_do_letterbox"):
        raise ValueError("only the resize-naive image strategy is supported")
    if any(p["interpolation"] != 3 for p in processor["tvf_resize_params"]):
        raise ValueError("only bicubic resizing is supported")
    json.dump(
        {
            "model_family":
            "openvla",
            "image_size":
            int(config["image_sizes"][0]),
            "normalize":
            processor["tvf_normalize_params"],
            "num_patches":
            256,
            "image_token_id":
            int(config["pad_token_id"]),
            "prompt":
            "In: What action should the robot take to {instruction}?\nOut:",
            "empty_token_id":
            29871,
            "n_action_bins":
            int(config["n_action_bins"]),
            "action_vocab_size":
            int(config["text_config"]["vocab_size"]) -
            int(config["pad_to_multiple_of"]),
            "norm_stats": {
                key: stats["action"]
                for key, stats in config["norm_stats"].items()
            },
        },
        open(os.path.join(out, "config.json"), "w"),
        indent=1)


if __name__ == "__main__":
    main()
