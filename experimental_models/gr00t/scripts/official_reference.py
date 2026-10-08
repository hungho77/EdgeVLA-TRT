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
"""Official GR00T N1.6 reference for one raw dataset observation.

Runs the GR00T source tree's own ``Gr00tPolicy`` on the CPU from a seeded x_0, in FP32 (the
policy's bf16 weights, FP32 activations) or with --bf16 as the policy serves (the precision floor), and
saves what each stage of a port has to match: the backbone features the action head
receives (with the token ids and image mask), the normalized state, the normalized action
prediction and the absolute actions.

    python official_reference.py --gr00t-src Isaac-GR00T --checkpoint GR00T-N1.6-SO101-Multitask \\
        --modality-config GR00T-N1.6-SO101-Multitask/so101_config.py --dataset so101-multitask \\
        --frame 300 --eager-attention --out ref_f300.npz
"""

import argparse
import sys

import numpy as np


def load_frame(dataset, frame, video_keys):
    """Frames and state of one dataset row (LeRobot v3; rows of file-000 map to video file 0)."""
    import av
    import pandas as pd
    frames = {}
    for key in video_keys:
        container = av.open(
            f"{dataset}/videos/observation.images.{key}/chunk-000/file-000.mp4"
        )
        for index, decoded in enumerate(container.decode(video=0)):
            if index == frame:
                frames[key] = decoded.to_ndarray(format="rgb24")
                break
    table = pd.read_parquet(f"{dataset}/data/chunk-000/file-000.parquet")
    state = np.asarray(table["observation.state"].iloc[frame],
                       dtype=np.float32)
    tasks = pd.read_parquet(f"{dataset}/meta/tasks.parquet")
    return frames, state, tasks.index[int(table["task_index"].iloc[frame])]


def _packed_eager(eager):

    def forward(module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout=0.0,
                seq_len_list=None,
                **kwargs):
        if seq_len_list is not None and len(seq_len_list) > 1:
            import torch
            image = torch.repeat_interleave(torch.arange(len(seq_len_list)),
                                            torch.tensor(seq_len_list))
            blocked = torch.zeros(image.numel(),
                                  image.numel(),
                                  dtype=query.dtype)
            blocked[image[:, None] != image[None, :]] = float("-inf")
            attention_mask = blocked if attention_mask is None else attention_mask + blocked
        return eager(module, query, key, value, attention_mask, scaling,
                     dropout, **kwargs)

    return forward


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gr00t-src", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--modality-config",
        help="python file that registers the embodiment's modality config")
    parser.add_argument("--embodiment", default="new_embodiment")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frame", type=int, default=300)
    parser.add_argument("--video-keys", nargs="+", default=["top", "wrist"])
    parser.add_argument("--state-split",
                        default="single_arm:5,gripper:1",
                        help="state groups in order, name:dim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eager-attention",
        action="store_true",
        help=
        "build a backbone that requires FlashAttention 2 (N1.6's Eagle) without it and run "
        "its attention eagerly: the same function, on a CPU")
    parser.add_argument("--bf16",
                        action="store_true",
                        help="run as the policy serves, activations in bf16")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.gr00t_src)
    from unittest import mock

    import gr00t.model  # noqa: F401  registers the models and processors
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy import gr00t_policy
    if args.modality_config:
        exec(
            open(args.modality_config).read(), {"__name__": "modality_config"})

    if args.eager_attention:
        from transformers.modeling_utils import PreTrainedModel
        PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(
            lambda cls, config, *a, **kw: config)
    policy = gr00t_policy.Gr00tPolicy(EmbodimentTag(args.embodiment),
                                      args.checkpoint,
                                      device="cpu")
    if not args.bf16:
        policy.model.float()
    if args.eager_attention:
        for module in policy.model.modules():
            config = getattr(module, "config", None)
            if config is not None and getattr(config, "_attn_implementation",
                                              None) is not None:
                config._attn_implementation = "eager"
        # Eagle's SigLIP2 packs every image into one sequence; its FlashAttention path keeps the images apart
        # with per-image cu_seqlens, while its eager path would let them attend to each other.
        for name, module in list(sys.modules.items()):
            if name.endswith("modeling_siglip2") and hasattr(
                    module, "eager_attention_forward"):
                module.eager_attention_forward = _packed_eager(
                    module.eager_attention_forward)
    head = policy.model.action_head
    horizon, action_dim = head.config.action_horizon, head.action_dim
    noise = np.random.default_rng(args.seed).standard_normal(
        (1, horizon, action_dim)).astype(np.float32)

    captured = {}
    real_randn = torch.randn

    def fixed_randn(*size, **kwargs):
        shape = tuple(
            kwargs.get(
                "size", size[0] if len(size) == 1
                and isinstance(size[0], (tuple, list)) else size))
        if shape == noise.shape:
            return torch.from_numpy(noise).to(
                kwargs.get("dtype", torch.float32))
        return real_randn(*size, **kwargs)

    real_head_get_action = head.get_action

    def spy_head(backbone_output, action_input, *a, **kw):
        captured["backbone_features"] = backbone_output["backbone_features"][
            0].detach().float().numpy()
        captured["image_mask"] = backbone_output["image_mask"][0].detach(
        ).numpy()
        captured["state"] = action_input["state"][0].detach().float().numpy()
        captured["embodiment_id"] = int(action_input["embodiment_id"][0])
        out = real_head_get_action(backbone_output, action_input, *a, **kw)
        captured["action_pred"] = out["action_pred"][0].detach().float().numpy(
        )
        return out

    backbone = policy.model.backbone
    real_backbone_forward = backbone.forward

    def spy_backbone(vl_input, *a, **kw):

        def to_numpy(value):
            return value.detach().float().numpy() if value.is_floating_point(
            ) else value.detach().numpy()

        for key, value in vl_input.items():
            if isinstance(value, torch.Tensor):
                captured[f"backbone_input.{key}"] = to_numpy(value)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        captured[f"backbone_input.{key}.{index}"] = to_numpy(
                            item)
        return real_backbone_forward(vl_input, *a, **kw)

    eagle = getattr(backbone, "model", None)
    real_extract = getattr(eagle, "extract_feature", None)

    def spy_extract(*a, **kw):
        out = real_extract(*a, **kw)
        captured["image_features"] = out.detach().float().numpy()
        captured["vision_dtype"] = str(
            next(eagle.vision_model.parameters()).dtype)
        return out

    real_model_get_action = policy.model.get_action

    def spy_model(*a, **kw):
        inputs = kw.get("inputs", a[0] if a else None)
        if inputs is not None and "input_ids" in inputs:
            captured["input_ids"] = inputs["input_ids"][0].detach().numpy()
        return real_model_get_action(*a, **kw)

    frames, state, instruction = load_frame(args.dataset, args.frame,
                                            args.video_keys)
    split, offset = {}, 0
    for item in args.state_split.split(","):
        name, dim = item.split(":")
        split[name] = state[None, None, offset:offset + int(dim)]
        offset += int(dim)
    observation = {
        "video": {
            k: v[None, None]
            for k, v in frames.items()
        },
        "state": split,
        "language": {
            "annotation.human.task_description": [[instruction]]
        },
    }
    keep_dtype = mock.MagicMock() if args.bf16 else mock.patch.object(
        gr00t_policy, "_rec_to_dtype", lambda x, dtype: x)
    with keep_dtype, \
            mock.patch.object(head, "get_action", spy_head), \
            mock.patch.object(policy.model, "get_action", spy_model), \
            mock.patch.object(backbone, "forward", spy_backbone), \
            (mock.patch.object(eagle, "extract_feature", spy_extract) if real_extract else mock.MagicMock()), \
            mock.patch("torch.randn", side_effect=fixed_randn), torch.no_grad():
        actions, _ = policy.get_action(observation)
    absolute = np.concatenate([actions[k][0] for k in split if k in actions],
                              axis=-1)
    np.savez(args.out,
             noise=noise,
             raw_state=state,
             absolute_actions=absolute,
             **{
                 k: v
                 for k, v in captured.items() if v is not None
             })
    print(
        f"{args.checkpoint}: {captured['backbone_features'].shape[0]} backbone tokens "
        f"({int(captured['image_mask'].sum())} image), state {captured['state'].shape}, "
        f"prediction {captured['action_pred'].shape}, absolute actions {absolute.shape} -> {args.out}"
    )


if __name__ == "__main__":
    main()
