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
"""Golden actions from the official X-VLA repository (github.com/2toinf/X-VLA) for a checkpoint fine-tuned there with
the ``auto`` action space and min-max normalized actions (twanghcmut/X-VLA-SO101-LeftArm-Multitask).

Builds the inputs as that checkpoint's SO101 handler and evaluation script do: each camera resized to 224x224 with
PIL bicubic and ImageNet-normalized, zero views up to --views (marked present, so they go through the vision tower),
the proprio min-max normalized to [-1, 1] and zero-padded, the BART tokens padded to the processor's length. Runs
``XVLA.generate_actions`` in FP32 on the CPU with a seeded x1, then adds the normalized proprio back to the delta
dims and unnormalizes.

    python xvla_reference.py --xvla-src X-VLA --checkpoint X-VLA-SO101-LeftArm-Multitask --stats stats.json \\
        --obs obs.npz --out ref.npz [--domain-id 19] [--delta-dims 0 1 2 3 4]

obs.npz: <camera>_<i> ([H, W, 3] or a stack whose last entry is used), state_<i> (the proprio in robot units; for
the SO101 checkpoint the previous action) and task_<i>. stats.json: LeRobot meta/stats.json ("action" min / max).
"""

import argparse
import json
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xvla-src", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--obs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--domain-id", type=int, default=19)
    parser.add_argument("--delta-dims",
                        type=int,
                        nargs="*",
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.xvla_src)
    import torch
    from models import modeling_xvla
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    model = XVLA.from_pretrained(args.checkpoint).to(torch.float32).eval()
    processor = XVLAProcessor.from_pretrained(args.checkpoint)
    stats = json.load(open(args.stats))["action"]
    lo = np.asarray(stats["min"], dtype=np.float32)
    hi = np.asarray(stats["max"], dtype=np.float32)
    span = np.where(hi - lo < 1e-6, 1.0, hi - lo).astype(np.float32)
    dims = lo.shape[0]
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    obs = np.load(args.obs)
    count = len([k for k in obs.files if k.startswith("state_")])
    generator = np.random.default_rng(args.seed)
    real_randn = torch.randn
    out = {"action_min": lo, "action_span": span}
    for i in range(count):

        def frame(camera):
            image = obs[f"{camera}_{i}"]
            return Image.fromarray(image[-1] if image.ndim == 4 else image)

        views = [transform(frame(c)) for c in args.cameras]
        views += [torch.zeros_like(views[0])] * (args.views - len(views))
        image_input = torch.stack(views)[None]
        image_mask = torch.ones(1, args.views, dtype=torch.bool)
        input_ids = processor.encode_language([str(obs[f"task_{i}"])
                                               ])["input_ids"]
        state = obs[f"state_{i}"].astype(np.float32)
        proprio_n = (state - lo) / span * 2.0 - 1.0
        proprio = torch.zeros(1, model.action_space.dim_action)
        proprio[0, :dims] = torch.from_numpy(proprio_n)
        domain_id = torch.tensor([args.domain_id])
        noise = torch.from_numpy(
            generator.standard_normal(
                (1, model.num_actions,
                 model.action_space.dim_action)).astype(np.float32))

        captured = {}
        forward_vlm = model.forward_vlm

        def record_vlm(*a, **kw):
            enc = forward_vlm(*a, **kw)
            captured["vlm_features"] = enc["vlm_features"][0].numpy()
            captured["aux_visual_inputs"] = enc["aux_visual_inputs"][0].numpy()
            return enc

        def fixed_randn(*size, **kwargs):
            return noise.clone().to(kwargs.get("dtype") or torch.float32)

        model.forward_vlm = record_vlm
        modeling_xvla.torch.randn = fixed_randn
        try:
            with torch.no_grad():
                actions = model.generate_actions(input_ids,
                                                 image_input,
                                                 image_mask,
                                                 domain_id,
                                                 proprio,
                                                 steps=args.steps)
        finally:
            modeling_xvla.torch.randn = real_randn
            del model.forward_vlm
        normalized = actions[0, :, :dims].numpy().copy()
        robot = normalized.copy()
        robot[:, args.delta_dims] += proprio_n[args.delta_dims]
        robot = (robot + 1.0) / 2.0 * span + lo
        out.update({
            f"input_ids_{i}": input_ids[0].numpy(),
            f"image_input_{i}": image_input[0].numpy(),
            f"image_mask_{i}": image_mask[0].numpy(),
            f"proprio_{i}": proprio[0].numpy(),
            f"domain_id_{i}": np.int64(args.domain_id),
            f"noise_{i}": noise[0].numpy(),
            f"vlm_features_{i}": captured["vlm_features"],
            f"aux_visual_inputs_{i}": captured["aux_visual_inputs"],
            f"model_actions_{i}": normalized,
            f"robot_actions_{i}": robot,
        })
        print(f"{i}: {str(obs[f'task_{i}'])!r}  row 0 {np.round(robot[0], 3)}")
    np.savez(args.out, **out)
    print(f"{count} samples -> {args.out}")


if __name__ == "__main__":
    main()
