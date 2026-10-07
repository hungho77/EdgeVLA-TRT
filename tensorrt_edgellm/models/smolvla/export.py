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
"""Export a LeRobot SmolVLA checkpoint as visual / prefix / denoise ONNX components (FP16, batch 1).

    python -m tensorrt_edgellm.models.smolvla.export <checkpoint> <out_dir> [--vlm <SmolVLM2 snapshot>]

Writes ``<out_dir>/{visual,prefix,denoise}/model.onnx`` and ``config.json`` (shapes, chunk and step
count, I/O names), and stages the checkpoint's normalizer statistics and the VLM's tokenizer so the
runtime needs nothing else.
"""

import argparse
import glob
import json
import logging
import os
import shutil

import torch
from safetensors.torch import load_file
from torch import nn

from ...onnx.export_encoder import _run_dynamo_export
from .modeling_smolvla import (SmolVLAConfig, SmolVLADenoise, SmolVLAPrefix,
                               SmolVLAVisual, load_smolvla_weights)

logger = logging.getLogger(__name__)

MAX_CAMERAS = 3
MAX_TOKENS = 48
MAX_PREFIX = MAX_CAMERAS * 64 + MAX_TOKENS + 1


class _DenoiseExport(nn.Module):
    """Named K/V inputs, so the ONNX graph has one input per tensor."""

    def __init__(self, denoise: SmolVLADenoise) -> None:
        super().__init__()
        self.denoise = denoise

    def forward(self, x_t, timestep, dt, kv):
        return self.denoise(x_t, timestep, dt, *kv)


def kv_names(cfg: SmolVLAConfig):
    names = []
    for i in range(cfg.num_layers):
        kind = "self" if cfg.is_self_attn(i) else "cross"
        names += [f"{kind}_k_layer{i:02d}", f"{kind}_v_layer{i:02d}"]
    return names


def export_smolvla(checkpoint: str, out_dir: str, vlm_dir: str) -> None:
    cfg = SmolVLAConfig()
    policy = json.load(open(os.path.join(checkpoint, "config.json")))
    cfg.chunk_size = int(policy["chunk_size"])
    cfg.max_state_dim = int(policy["max_state_dim"])
    cfg.max_action_dim = int(policy["max_action_dim"])
    cfg.num_layers = int(policy["num_vlm_layers"])
    cfg.self_attn_every_n_layers = int(policy["self_attn_every_n_layers"])
    if policy.get("attention_mode") != "cross_attn" or policy.get(
            "add_image_special_tokens"):
        raise ValueError(
            "only attention_mode=cross_attn without image special tokens is supported"
        )

    visual, prefix, denoise = SmolVLAVisual(cfg), SmolVLAPrefix(
        cfg), SmolVLADenoise(cfg)
    load_smolvla_weights(
        load_file(os.path.join(checkpoint, "model.safetensors")), visual,
        prefix, denoise)
    visual, prefix, denoise = visual.half().eval(), prefix.half().eval(
    ), denoise.half().eval()

    Dim = torch.export.Dim
    os.makedirs(out_dir, exist_ok=True)
    for name in ("visual", "prefix", "denoise"):
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    views = Dim("num_views", min=1, max=MAX_CAMERAS)
    _run_dynamo_export(visual, (torch.zeros(
        2, 3, cfg.image_size, cfg.image_size, dtype=torch.float16), ),
                       os.path.join(out_dir, "visual", "model.onnx"),
                       ["pixel_values"], ["image_features"], ({
                           0: views
                       }, ))

    image_tokens = Dim("image_tokens",
                       min=cfg.image_tokens,
                       max=MAX_CAMERAS * cfg.image_tokens)
    text = Dim("num_tokens", min=2, max=MAX_TOKENS)
    names = kv_names(cfg)
    _run_dynamo_export(
        prefix, (torch.zeros(
            1, 2 * cfg.image_tokens, cfg.text_hidden,
            dtype=torch.float16), torch.zeros(1, 16, dtype=torch.int64),
                 torch.zeros(1, cfg.max_state_dim, dtype=torch.float32)),
        os.path.join(out_dir, "prefix", "model.onnx"),
        ["image_features", "token_ids", "state"], names, ({
            1: image_tokens
        }, {
            1: text
        }, None))

    prefix_len = Dim("prefix_len", min=2, max=MAX_PREFIX)
    kv = tuple(
        torch.zeros(1,
                    2 * cfg.image_tokens + 17,
                    cfg.text_kv_heads,
                    cfg.head_dim,
                    dtype=torch.float16) for _ in names)
    _run_dynamo_export(_DenoiseExport(denoise),
                       (torch.zeros(1, cfg.chunk_size, cfg.max_action_dim),
                        torch.ones(1), torch.tensor(-0.1), kv),
                       os.path.join(out_dir, "denoise", "model.onnx"),
                       ["x_t", "timestep", "dt"] + names, ["x_next"],
                       (None, None, None, tuple({1: prefix_len}
                                                for _ in names)))

    assets = os.path.join(out_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    for path in glob.glob(os.path.join(checkpoint, "policy_*processor*")):
        shutil.copy(path, assets)
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"):
        if os.path.exists(os.path.join(vlm_dir, name)):
            shutil.copy(os.path.join(vlm_dir, name), assets)
    json.dump(
        {
            "model_family": "smolvla",
            "image_size": cfg.image_size,
            "image_tokens_per_view": cfg.image_tokens,
            "max_views": MAX_CAMERAS,
            "max_tokens": MAX_TOKENS,
            "max_prefix_len": MAX_PREFIX,
            "chunk_size": cfg.chunk_size,
            "max_action_dim": cfg.max_action_dim,
            "max_state_dim": cfg.max_state_dim,
            "num_steps": int(policy.get("num_steps", 10)),
            "kv_names": names,
            "kv_heads": cfg.text_kv_heads,
            "head_dim": cfg.head_dim,
            "camera_rename": {},
            "policy": {
                k: policy[k]
                for k in ("input_features", "output_features",
                          "normalization_mapping", "resize_imgs_with_padding",
                          "tokenizer_max_length")
            },
        },
        open(os.path.join(out_dir, "config.json"), "w"),
        indent=1)
    logger.info("SmolVLA export complete: %s", out_dir)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument(
        "--vlm",
        default=None,
        help=
        "SmolVLM2 snapshot holding tokenizer.json (default: the Hugging Face cache)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    vlm = args.vlm or sorted(
        glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/*"
            )))[-1]
    export_smolvla(args.checkpoint, args.out_dir, vlm)


if __name__ == "__main__":
    main()
