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
"""Golden actions from the official Evo-1 policy (MINT-SJTU/Evo-1, ``Evo_1/scripts/Evo1_server.py``).

The policy is built by the repo's own ``EVO1`` class and the checkpoint directory's ``config.json``, with the DeepSpeed
``mp_rank_00_model_states.pt`` loaded strictly over it, and queried as the official server does: each frame resized
to 448 x 448 by OpenCV and passed through ``cv2.COLOR_BGR2RGB`` (the official clients send RGB, so the red and blue
channels reach the model swapped), the views padded to three with a black image masked out of the language model's
attention, the state normalized by the checkpoint's bounds and zero-padded to 24, 50 Euler steps from uniform noise,
and the first action dims unnormalized. timm is stubbed: InternViT imports DropPath, an identity at inference.

The official setup installs flash-attn, with which InternVL3's language model runs FlashAttention-2: a position the
attention mask excludes (the right padding and the black view's tokens) gets no attention output at all, only its
residual and MLP, while the action head still reads its hidden state. Without flash-attn the model falls back to
eager attention, where such a position attends to the earlier tokens instead. ``--attention flash`` (the default)
reproduces FlashAttention-2's result on the eager path by zeroing each language layer's attention output at the
excluded positions; ``--attention eager`` keeps the fallback.

    python evo1_reference.py --repo Evo-1 --checkpoint Evo1_LIBERO --vlm InternVL3-1B \\
        --obs obs.npz --out ref.npz [--precision fp32|bf16]

obs.npz holds agentview_<i>, wrist_<i> (raw LIBERO renders at 448 x 448; rotated 180 degrees here as the official
LIBERO client does), state_<i> ([eef pos, axis-angle, gripper qpos]) and task_<i>. The output adds, per sample, the
model's pixel values, the vision features, the token ids and attention mask, the language model's hidden states, the
normalized state, the noise, the normalized chunk and the robot actions. ``bf16`` is the official server's precision
(bf16 weights under bf16 autocast); ``fp32`` upcasts every weight and keeps the preprocessing in FP32.
"""

import argparse
import contextlib
import importlib.machinery
import json
import os
import sys
import types

import numpy as np
import torch
from torch import nn

IMAGE_SIZE = 448
MAX_VIEWS = 3
MAX_DIM = 24


def stub_timm():
    # transformers probes for timm at import; probed before the stub exists, it keeps treating timm as absent.
    import transformers  # noqa: F401

    class DropPath(nn.Identity):

        def __init__(self, *args, **kwargs):
            super().__init__()

    layers = types.ModuleType("timm.layers")
    layers.DropPath = DropPath
    timm = types.ModuleType("timm")
    timm.layers = layers
    for module in (timm, layers):
        module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)
    sys.modules.update({"timm": timm, "timm.layers": layers})


def load_policy(repo, checkpoint, vlm, precision, steps):
    try:
        import timm  # noqa: F401
    except ImportError:
        stub_timm()
    sys.path.insert(0, os.path.join(repo, "Evo_1"))
    from config import EvoConfig

    from scripts.Evo1 import EVO1

    config = json.load(open(os.path.join(checkpoint, "config.json")))
    config.update(finetune_vlm=False,
                  finetune_action_head=False,
                  enable_gradient_checkpointing=False,
                  num_inference_timesteps=steps)
    if vlm:
        config["vlm_name"] = vlm
    model = EVO1(EvoConfig.from_dict(config)).eval()
    state = torch.load(os.path.join(checkpoint, "mp_rank_00_model_states.pt"),
                       map_location="cpu",
                       weights_only=False)["module"]
    model.load_state_dict(state, strict=True)
    if precision == "fp32":
        model.float()
        embedder = model.embedder
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

        def preprocess(batches):
            pixels, tiles = [], []
            for images in batches:
                for image in images:
                    pixels.append(
                        (image.float().unsqueeze(0).cpu() - mean) / std)
                tiles.append([1] * len(images))
            return torch.cat(pixels).to(embedder.device), tiles

        embedder._preprocess_images = preprocess
    return model.cuda(), config


def bounds(stats, key, dims):
    return (np.asarray(stats[key]["min"], np.float32)[:dims],
            np.asarray(stats[key]["max"], np.float32)[:dims])


class Policy:
    """The official server's ``infer_from_json_dict`` with the noise and the intermediate tensors exposed."""

    def __init__(self, args):
        self.model, self.config = load_policy(args.repo, args.checkpoint,
                                              args.vlm, args.precision,
                                              args.steps)
        self.precision = args.precision
        stats = json.load(
            open(os.path.join(args.checkpoint, "norm_stats.json")))
        arm = args.arm_key or next(iter(stats))
        stats = stats[arm]
        if "observation.state" not in stats:
            stats = stats[args.dataset_key or next(iter(stats))]
        self.stats = stats
        self.state_dim = len(stats["observation.state"]["min"])
        self.action_dim = len(stats["action"]["min"])
        self.captured = {}
        embedder = self.model.embedder
        original_extract = embedder.model.extract_feature

        def extract_feature(pixels):
            features = original_extract(pixels)
            self.captured.update(pixels=pixels, visual=features)
            return features

        embedder.model.extract_feature = extract_feature
        original_fuse = embedder._prepare_and_fuse_embeddings

        def fuse(prompts, *a, **k):
            embeds, mask = original_fuse(prompts, *a, **k)
            ids = embedder.tokenizer(prompts,
                                     return_tensors="pt",
                                     padding="max_length",
                                     truncation=True,
                                     max_length=embedder.max_text_length)
            self.captured.update(input_ids=ids["input_ids"],
                                 attention_mask=mask)
            return embeds, mask

        embedder._prepare_and_fuse_embeddings = fuse
        if args.attention == "flash":
            for layer in embedder.model.language_model.model.layers:
                layer.self_attn.register_forward_hook(self.zero_excluded)
        original_vl = self.model.get_vl_embeddings

        def vl_embeddings(*a, **k):
            hidden = original_vl(*a, **k)
            self.captured.update(hidden=hidden)
            return hidden

        self.model.get_vl_embeddings = vl_embeddings

    def zero_excluded(self, module, inputs, outputs):
        keep = self.captured["attention_mask"][..., None].to(outputs[0].dtype)
        return (outputs[0] * keep, ) + tuple(outputs[1:])

    def frames(self, images):
        import cv2
        from torchvision import transforms
        out = []
        for image in images:
            image = cv2.resize(np.asarray(image, np.uint8),
                               (IMAGE_SIZE, IMAGE_SIZE))
            out.append(transforms.ToTensor()(cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB)).cuda())
        return out

    def predict(self, images, task, state, noise):
        views = len(images)
        frames = self.frames(images)
        frames += [
            torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, device="cuda")
            for _ in range(MAX_VIEWS - views)
        ]
        image_mask = torch.tensor([1] * views + [0] * (MAX_VIEWS - views),
                                  dtype=torch.int32,
                                  device="cuda")
        action_mask = torch.tensor([[1] * self.action_dim + [0] *
                                    (MAX_DIM - self.action_dim)],
                                   dtype=torch.int32,
                                   device="cuda")
        low, high = bounds(self.stats, "observation.state", self.state_dim)
        state = np.asarray(state, np.float32)[:self.state_dim]
        normalized_state = np.clip(2 * (state - low) / (high - low + 1e-8) - 1,
                                   -1, 1)
        normalized_state = np.concatenate(
            [normalized_state,
             np.zeros(MAX_DIM - self.state_dim, np.float32)])
        used = {}

        def rand(*shape, device=None, dtype=None, **kwargs):
            # get_action draws torch.rand(...) * 2 - 1; hand it the uniform draw behind the given noise.
            uniform = ((noise.reshape(shape) + 1) / 2).to(device=device,
                                                          dtype=dtype)
            used["noise"] = uniform * 2 - 1
            return uniform

        autocast = torch.autocast(
            "cuda", dtype=torch.bfloat16
        ) if self.precision == "bf16" else contextlib.nullcontext()
        original_rand = torch.rand
        torch.rand = rand
        try:
            with torch.no_grad(), autocast:
                action = self.model.run_inference(
                    images=frames,
                    image_mask=image_mask,
                    prompt=task,
                    state_input=torch.from_numpy(normalized_state).cuda(),
                    action_mask=action_mask)
        finally:
            torch.rand = original_rand
        normalized = action.float().reshape(-1, MAX_DIM).cpu().numpy()
        low, high = bounds(self.stats, "action", self.action_dim)
        robot = (normalized[:, :self.action_dim] + 1.0) / 2.0 * (high - low +
                                                                 1e-8) + low
        self.captured.update(state=torch.from_numpy(normalized_state),
                             noise=used["noise"])
        return normalized, robot


def policy_rotate(image):
    return np.ascontiguousarray(image[::-1, ::-1])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="MINT-SJTU/Evo-1 clone")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="config.json, norm_stats.json, mp_rank_00_model_states.pt")
    parser.add_argument(
        "--vlm",
        help="InternVL3-1B directory (default: the config's vlm_name)")
    parser.add_argument("--obs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision",
                        choices=["fp32", "bf16"],
                        default="fp32")
    parser.add_argument("--steps",
                        type=int,
                        default=50,
                        help="Euler steps (the official server sets 50)")
    parser.add_argument("--arm-key",
                        help="norm_stats.json arm (default: the first)")
    parser.add_argument("--dataset-key",
                        help="norm_stats.json dataset under the arm")
    parser.add_argument(
        "--attention",
        choices=["flash", "eager"],
        default="flash",
        help="the language model's attention at excluded positions (see above)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    policy = Policy(args)
    obs = np.load(args.obs)
    count = len([k for k in obs.files if k.startswith("state_")])
    generator = torch.Generator().manual_seed(args.seed)
    out = {}
    for i in range(count):
        policy.captured.clear()
        noise = torch.rand(
            1, MAX_DIM * policy.model.horizon, generator=generator) * 2 - 1
        task = str(obs[f"task_{i}"])
        normalized, robot = policy.predict([
            policy_rotate(obs[f"agentview_{i}"]),
            policy_rotate(obs[f"wrist_{i}"])
        ], task, obs[f"state_{i}"], noise)
        for key, value in policy.captured.items():
            value = value.detach().cpu()
            out[f"{key}_{i}"] = value.float().numpy(
            ) if value.is_floating_point() else value.numpy()
        out[f"normalized_{i}"] = normalized
        out[f"actions_{i}"] = robot
        print(f"{i}: {task!r}\n  robot row 0 {np.round(robot[0], 4)}")
    np.savez(args.out, **out)
    print(f"{count} samples ({args.precision}) -> {args.out}")


if __name__ == "__main__":
    main()
