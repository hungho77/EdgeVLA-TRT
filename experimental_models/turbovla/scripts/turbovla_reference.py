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

A run trained with the repo's starVLA trainer (--checkpoint holding config.yaml, checkpoints/*.pt, the run's
*data_config*.py, modality.json and dataset_statistics.json, e.g. an SO101 fine-tune) goes through its own
predict_action instead: the cameras in modality.json's video order, state and actions normalized as the data config
declares (min_max over the statistics' min / max; a dim whose min equals its max passes through raw; actions clipped
to [-1, 1] before unnormalizing), as its open-loop evaluation does. Its observations are --obs-json files
{"task", "state", "cameras": {name: image path}}.
"""

import argparse
import glob
import json
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


def starvla_weights(checkpoint):
    """The EMA (else latest) weights of a starVLA run directory, or None for the released LIBERO layout."""
    if not os.path.exists(os.path.join(checkpoint, "config.yaml")):
        return None
    weights = sorted(glob.glob(os.path.join(checkpoint, "checkpoints",
                                            "*.pt")))
    ema = [w for w in weights if "ema" in os.path.basename(w)]
    return (ema or weights or [None])[-1]


def data_config_modes(checkpoint):
    """normalization_modes of the run's data config ({"state.single_arm": "min_max", ...}), read from its source:
    importing it would pull in the whole starVLA dataloader."""
    import ast
    path = (glob.glob(os.path.join(checkpoint, "*data_config*.py"))
            or [None])[0]
    if path is None:
        raise SystemExit(
            f"{checkpoint}: no *data_config*.py to read the normalization from"
        )
    modes = {}
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Dict) and node.keys and all(
                isinstance(k, ast.Constant) and isinstance(k.value, str)
                and k.value.startswith(("state.", "action."))
                and isinstance(v, ast.Constant) and isinstance(v.value, str)
                for k, v in zip(node.keys, node.values)):
            modes.update({
                k.value: v.value
                for k, v in zip(node.keys, node.values)
            })
    if not modes:
        raise SystemExit(f"{path}: no normalization_modes found")
    return modes


class StarvlaPolicy:
    """A starVLA-trained TurboVLA run with its data config's normalization."""

    def __init__(self, args, weights):
        from omegaconf import OmegaConf
        from starVLA.model.framework.VLM4A.TurboVLA import TurboVLAFramework
        cfg = OmegaConf.load(os.path.join(args.checkpoint, "config.yaml"))
        cfg.framework.vision.model_path = os.path.abspath(args.dinov3)
        cfg.framework.text.bert_path = os.path.abspath(args.bert)
        # Every weight is in the run's checkpoint; the GroundingDINO initialization is a training-time step.
        cfg.framework.initialization.load_pretrained = False
        self.framework = TurboVLAFramework(cfg)
        state = torch.load(weights, map_location="cpu")
        self.framework.load_state_dict(state, strict=True)
        self.framework.to(args.device).eval()
        self.model = self.framework.model
        if args.precision == "bf16":
            self.framework.to(torch.bfloat16)

        modality = json.load(
            open(os.path.join(args.checkpoint, "modality.json")))
        self.cameras = list(modality["video"])
        payload = json.load(
            open(os.path.join(args.checkpoint, "dataset_statistics.json")))
        stats = payload[args.embodiment] if args.embodiment else next(
            iter(payload.values()))
        modes = data_config_modes(args.checkpoint)

        def per_dim(section):
            out = [None] * sum(v["end"] - v["start"]
                               for v in modality[section].values())
            for name, span in modality[section].items():
                for d in range(span["start"], span["end"]):
                    out[d] = modes.get(f"{section}.{name}", "min_max")
            return out

        if set(per_dim("state") + per_dim("action")) != {"min_max"}:
            raise SystemExit(
                f"normalization {modes}: only min_max on every state and action dim is supported"
            )
        self.state_min = np.asarray(stats["state"]["min"], np.float64)
        self.state_max = np.asarray(stats["state"]["max"], np.float64)
        self.action_min = np.asarray(stats["action"]["min"], np.float64)
        self.action_max = np.asarray(stats["action"]["max"], np.float64)
        self.proprio_mean = np.asarray(stats["state"]["mean"], np.float64)
        self.proprio_std = np.asarray(stats["state"]["std"], np.float64)

    def normalize_state(self, state):
        lo, hi = self.state_min, self.state_max
        state = np.asarray(state, np.float64)
        return np.where(
            hi != lo, (state - lo) / np.where(hi == lo, 1.0, hi - lo) * 2 - 1,
            state)

    def unnormalize(self, normalized):
        lo, hi = self.action_min, self.action_max
        return 0.5 * (np.clip(normalized, -1, 1) + 1) * (hi - lo) + lo

    def predict(self, images, task, state):
        """images: {camera: [H, W, 3] uint8}; returns the normalized chunk and the robot actions."""
        from PIL import Image
        example = {
            "image": [
                Image.fromarray(np.asarray(images[c], np.uint8))
                for c in self.cameras
            ],
            "lang":
            task,
            "state":
            self.normalize_state(state),
        }
        with torch.no_grad():
            normalized = self.framework.predict_action(
                [example])["normalized_actions"][0]
        return normalized, self.unnormalize(normalized)

    def runtime_config(self):
        """The config.json fields EdgeVLA-TRT's TurbovlaPolicy reads for this normalization and preprocessing."""
        return {
            "cameras": self.cameras,
            "state_normalization": "min_max",
            "state_min": self.state_min.tolist(),
            "state_max": self.state_max.tolist(),
            "binary_gripper": False,
            "clip_actions": True,
            "resize": "bilinear_antialias",
        }


def load_policy(args):
    sys.path.insert(0, os.path.abspath(args.repo))
    stub_timm()
    weights = starvla_weights(args.checkpoint)
    if weights is not None:
        sys.path.insert(
            0,
            os.path.join(os.path.abspath(args.repo), "third_party",
                         "starvla_runtime"))
        from turbovla.models import text_encoder, vision_encoder
        text_encoder._load_pretrained_model = from_config
        vision_encoder._load_pretrained_model = from_config
        policy = StarvlaPolicy(args, weights)
        if args.precision == "fp32":
            policy.model.vision_encoder.config.compute_precision = "fp32"
            policy.model.config.interaction.compute_precision = "fp32"
        return policy
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
    parser.add_argument("--obs", help="LIBERO observations (.npz)")
    parser.add_argument(
        "--obs-json",
        nargs="+",
        help="starVLA runs: {task, state, cameras: {name: image}} per sample")
    parser.add_argument(
        "--embodiment",
        help="starVLA runs: the statistics key (default: the only one)")
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

    out = {}
    if isinstance(policy, StarvlaPolicy):
        from PIL import Image
        original_inputs = policy.framework._model_inputs

        def model_inputs(examples):
            instructions, samples, states = original_inputs(examples)
            captured.update(pixels=samples["dinov3"], state=states)
            return instructions, samples, states

        policy.framework._model_inputs = model_inputs
        for i, path in enumerate(args.obs_json):
            captured.clear()
            sample = json.load(open(path))
            images = {
                k: np.asarray(Image.open(v).convert("RGB"))
                for k, v in sample["cameras"].items()
            }
            normalized, actions = policy.predict(images, sample["task"],
                                                 sample["state"])
            for key, value in captured.items():
                out[f"{key}_{i}"] = value.detach().float().cpu().numpy(
                ) if value.is_floating_point() else value.cpu().numpy()
            out[f"normalized_{i}"] = normalized
            out[f"actions_{i}"] = actions
            print(
                f"{i}: {sample['task']!r}\n  normalized row 0 {np.round(normalized[0], 4)}"
            )
        np.savez(args.out, **out)
        print(f"{len(args.obs_json)} samples ({args.precision}) -> {args.out}")
        return

    obs = np.load(args.obs)
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
