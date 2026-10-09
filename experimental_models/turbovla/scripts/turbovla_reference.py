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
"""Golden actions from the official TurboVLA LIBERO policy (H-EmbodVis/TurboVLA, its suite-stats evaluation path).

The released checkpoint carries every weight, DINOv3 and BERT included, so the encoders are built from their configs
only (--dinov3: a dinov3_vit ViT-B/16 config.json, --bert: bert-base-uncased's config and tokenizer) and the checkpoint
is loaded strictly over them. timm is stubbed: the fusion blocks import DropPath, an identity at inference. The
released LIBERO checkpoint stores its weights as ``model_state_dict``, which the loader read until the official repo
switched to requiring ``ema_model_state_dict`` (commit ced2b0c); that earlier lookup is restored here.

    python turbovla_reference.py --repo TurboVLA --checkpoint TurboVLA-hf --dinov3 dinov3-vitb16 \\
        --bert bert-base-uncased --obs obs.npz --out ref.npz [--precision fp32|bf16|official-fp32]

obs.npz holds agentview_<i>, wrist_<i> (raw LIBERO renders; rotated 180 degrees here as the official rollout does),
state_<i> ([eef pos, axis-angle, gripper qpos]) and task_<i>. The output adds, per sample, the BERT token ids and
masks, the projected text tokens, the projected visual tokens, the fused condition, the normalized chunk and the
environment actions. ``fp32`` runs every module in FP32; ``bf16`` is the official evaluation default; ``official-fp32``
is the official FP32 mode, which still autocasts DINOv3 to BF16 on CUDA.
"""

import argparse
import os
import sys
import types

import numpy as np
import torch
from torch import nn


def stub_timm():
    # transformers probes for timm at import; probed before the stub exists, it keeps treating timm as absent.
    import transformers  # noqa: F401

    class DropPath(nn.Module):

        def __init__(self, drop_prob=0.0):
            super().__init__()

        def forward(self, x):
            return x

    layers = types.ModuleType("timm.models.layers")
    layers.DropPath = DropPath
    models = types.ModuleType("timm.models")
    models.layers = layers
    timm = types.ModuleType("timm")
    timm.models = models
    sys.modules.update({
        "timm": timm,
        "timm.models": models,
        "timm.models.layers": layers
    })


def from_config(config):
    from transformers import AutoConfig, AutoModel
    return AutoModel.from_config(
        AutoConfig.from_pretrained(config.model_name_or_path))


def load_policy(args):
    sys.path.insert(0, os.path.abspath(args.repo))
    stub_timm()
    from turbovla.evaluation import policy as base_policy
    from turbovla.evaluation import suite_policy
    from turbovla.models import text_encoder, vision_encoder
    base_policy._checkpoint_state_dict = lambda checkpoint: checkpoint.get(
        "ema_model_state_dict", checkpoint["model_state_dict"])
    text_encoder._load_pretrained_model = from_config
    vision_encoder._load_pretrained_model = from_config
    precision = "bf16" if args.precision == "bf16" else "fp32"
    policy = suite_policy.TurboVLAPolicy(
        ckpt_path=os.path.join(args.checkpoint, "checkpoints", "libero",
                               "turbovla_libero.pth"),
        dinov3_path=args.dinov3,
        bert_path=args.bert,
        stats_path=os.path.join(args.checkpoint, "libero_all4_stats.json"),
        stats_key="libero_all4_no_noops",
        device=args.device,
        precision=precision,
        verbose=False)
    if args.precision == "fp32":
        policy.model.vision_encoder.config.compute_precision = "fp32"
        policy.model.config.interaction.compute_precision = "fp32"
    return policy


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo",
                        required=True,
                        help="a clone of H-EmbodVis/TurboVLA")
    parser.add_argument("--checkpoint",
                        required=True,
                        help="the H-EmbodVis/TurboVLA model repo")
    parser.add_argument("--dinov3", required=True)
    parser.add_argument("--bert", required=True)
    parser.add_argument("--obs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision",
                        default="fp32",
                        choices=("fp32", "bf16", "official-fp32"))
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    policy = load_policy(args)
    model = policy.model
    captured = {}
    model.text_encoder.register_forward_hook(
        lambda m, i, o: captured.update(text=o[0], key_padding=o[1]))
    original_encode_vision = model.encode_vision
    model.encode_vision = lambda pixels: captured.setdefault(
        "visual", original_encode_vision(pixels))
    model.action_head.register_forward_pre_hook(
        lambda m, i: captured.update(condition=i[0]))
    original_tokenize = model.text_encoder._tokenize_group

    def tokenize(*a, **k):
        tokenized, self_attention, position_ids = original_tokenize(*a, **k)
        captured.update(input_ids=tokenized["input_ids"],
                        self_attention=self_attention,
                        position_ids=position_ids)
        return tokenized, self_attention, position_ids

    model.text_encoder._tokenize_group = tokenize

    obs = np.load(args.obs)
    out = {}
    count = len([k for k in obs.files if k.startswith("state_")])
    for i in range(count):
        captured.clear()
        primary = policy_rotate(obs[f"agentview_{i}"])
        wrist = policy_rotate(obs[f"wrist_{i}"])
        task = str(obs[f"task_{i}"])
        normalized = policy.predict_normalized_action_chunk(
            primary, wrist, task, obs[f"state_{i}"])
        env_actions = np.stack(
            [policy._normalized_row_to_env_action(r) for r in normalized])
        for key, value in captured.items():
            out[f"{key}_{i}"] = value.detach().float().cpu().numpy(
            ) if value.is_floating_point() else value.cpu().numpy()
        out[f"normalized_{i}"] = normalized
        out[f"actions_{i}"] = env_actions
        print(
            f"{i}: {task!r}\n  normalized row 0 {np.round(normalized[0], 4)}")
    np.savez(args.out, **out)
    print(f"{count} samples ({args.precision}) -> {args.out}")


def policy_rotate(image):
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


if __name__ == "__main__":
    main()
