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
                               SmolVLAVisual, inpaint, load_smolvla_weights)

logger = logging.getLogger(__name__)

MAX_CAMERAS = 3
MAX_TOKENS = 48
MAX_PREFIX = MAX_CAMERAS * 64 + MAX_TOKENS + 1


class _DenoiseExport(nn.Module):
    """One Euler step plus RTC inpainting; named K/V inputs, one graph input per tensor."""

    def __init__(self, denoise: SmolVLADenoise) -> None:
        super().__init__()
        self.denoise = denoise

    def forward(self, x_t, timestep, dt, x_0, rtc_seed, rtc_weight, kv):
        x_next = self.denoise(x_t, timestep, dt, *kv)
        return inpaint(x_next, timestep + dt, x_0, rtc_seed, rtc_weight)


def expert_intermediate_size(hidden, ffn_dim_multiplier=4, multiple_of=256):
    """LeRobot's get_intermediate_size."""
    hidden = int(ffn_dim_multiplier * int(2 * hidden / 3))
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


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
    # LeRobot's SmolVLMWithExpertModel: num_vlm_layers <= 0 keeps the whole text model, the expert matches the VLM's
    # depth unless num_expert_layers says otherwise, and its width and MLP follow expert_width_multiplier.
    text_layers = json.load(open(os.path.join(
        vlm_dir, "config.json")))["text_config"]["num_hidden_layers"]
    num_vlm_layers = int(policy["num_vlm_layers"])
    cfg.num_layers = num_vlm_layers if num_vlm_layers > 0 else text_layers
    num_expert_layers = int(policy.get("num_expert_layers", -1))
    if num_expert_layers > 0 and num_expert_layers != cfg.num_layers:
        raise ValueError("only an expert as deep as the VLM is supported")
    cfg.expert_hidden = int(cfg.text_hidden *
                            float(policy["expert_width_multiplier"]))
    cfg.expert_intermediate = expert_intermediate_size(cfg.expert_hidden)
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
    # Distinct tensors: the exporter merges inputs that are the same object.
    chunks = [
        torch.zeros(1, cfg.chunk_size, cfg.max_action_dim) for _ in range(3)
    ]
    _run_dynamo_export(
        _DenoiseExport(denoise),
        (chunks[0], torch.ones(1), torch.tensor(-0.1), chunks[1], chunks[2],
         torch.zeros(1, cfg.chunk_size, 1), kv),
        os.path.join(out_dir, "denoise", "model.onnx"),
        ["x_t", "timestep", "dt", "x_0", "rtc_seed", "rtc_weight"] + names,
        ["x_next"], (None, None, None, None, None, None,
                     tuple({1: prefix_len} for _ in names)))

    stage_runtime_assets(checkpoint, out_dir, vlm_dir, cfg, policy, names)
    logger.info("SmolVLA export complete: %s", out_dir)


def stage_runtime_assets(checkpoint: str, out_dir: str, vlm_dir: str,
                         cfg: SmolVLAConfig, policy: dict, names) -> None:
    """config.json plus assets/ (tokenizer, normalization.json) -- everything the runtime reads."""
    assets = os.path.join(out_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"):
        if os.path.exists(os.path.join(vlm_dir, name)):
            shutil.copy(os.path.join(vlm_dir, name), assets)

    def processor_step(file_name: str, registry_name: str) -> dict:
        steps = json.load(open(os.path.join(checkpoint, file_name)))["steps"]
        return next(step for step in steps
                    if step["registry_name"] == registry_name)

    normalizer = processor_step("policy_preprocessor.json",
                                "normalizer_processor")
    unnormalizer = processor_step("policy_postprocessor.json",
                                  "unnormalizer_processor")
    pre = load_file(os.path.join(checkpoint, normalizer["state_file"]))
    post = load_file(os.path.join(checkpoint, unnormalizer["state_file"]))
    mapping = policy["normalization_mapping"]
    if mapping.get("STATE") != "MEAN_STD" or mapping.get(
            "ACTION") != "MEAN_STD":
        raise ValueError(
            f"only MEAN_STD state/action normalization is supported, got {mapping}"
        )
    json.dump(
        {
            "state_mean": pre["observation.state.mean"].tolist(),
            "state_std": pre["observation.state.std"].tolist(),
            "state_eps": normalizer["config"]["eps"],
            "action_mean": post["action.mean"].tolist(),
            "action_std": post["action.std"].tolist(),
            "action_eps": unnormalizer["config"]["eps"],
        },
        open(os.path.join(assets, "normalization.json"), "w"),
        indent=1)

    rename = processor_step(
        "policy_preprocessor.json",
        "rename_observations_processor")["config"]["rename_map"]
    # LeRobot feeds the cameras in the order its config lists image features.
    camera_features = [
        k for k, v in policy["input_features"].items() if v["type"] == "VISUAL"
    ]
    cameras = []
    for feature in camera_features:
        source = next((k for k, v in rename.items() if v == feature), feature)
        cameras.append(source.replace("observation.images.", ""))
    json.dump(
        {
            "model_family":
            "smolvla",
            "image_size":
            cfg.image_size,
            "image_tokens_per_view":
            cfg.image_tokens,
            "max_views":
            MAX_CAMERAS,
            "max_tokens":
            MAX_TOKENS,
            "max_prefix_len":
            MAX_PREFIX,
            "chunk_size":
            cfg.chunk_size,
            "max_action_dim":
            cfg.max_action_dim,
            "max_state_dim":
            cfg.max_state_dim,
            "state_dim":
            int(policy["input_features"]["observation.state"]["shape"][0]),
            "action_dim":
            int(policy["output_features"]["action"]["shape"][0]),
            "num_steps":
            int(policy.get("num_steps", 10)),
            "kv_names":
            names,
            "kv_heads":
            cfg.text_kv_heads,
            "head_dim":
            cfg.head_dim,
            "cameras":
            cameras,
            "image_pad_value":
            0.0,
            "prompt_suffix":
            "\n",
        },
        open(os.path.join(out_dir, "config.json"), "w"),
        indent=1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument(
        "--assets-only",
        action="store_true",
        help="rewrite config.json and assets/ without re-exporting the ONNX")
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
    if args.assets_only:
        cfg = SmolVLAConfig()
        policy = json.load(open(os.path.join(args.checkpoint, "config.json")))
        stage_runtime_assets(args.checkpoint, args.out_dir, vlm, cfg, policy,
                             kv_names(cfg))
    else:
        export_smolvla(args.checkpoint, args.out_dir, vlm)


if __name__ == "__main__":
    main()
