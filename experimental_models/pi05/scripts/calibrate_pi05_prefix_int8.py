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
"""Prefix activation statistics for ``tensorrt-edgellm-export --pi05-prefix-int8``.

Runs openpi's own policy (FP32, CPU) over observations sampled from a LeRobot v3
dataset and records, for every PaliGemma language-model projection, the per input
channel abs-max over the valid prefix rows of the prefix pass (openpi pads the
prefix with masked camera slots and prompt padding; those rows are skipped).

    python calibrate_pi05_prefix_int8.py --config pi05_so101 --checkpoint <openpi ckpt> \\
        --dataset <LeRobot v3 root> --num-frames 48 --exclude 300,900 --out prefix_amax.safetensors

Needs openpi installed with PyTorch model support (see openpi_reference.py).
"""

import argparse
import collections
import importlib.util
import sys
import types

import numpy as np

_PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                "mlp.down_proj")


def sample_observations(dataset, num_frames, exclude, cameras, seed=0):
    """Rows spread over the dataset's episodes (and so its tasks), decoded from its videos."""
    import av
    import pandas as pd

    episodes = pd.read_parquet(
        f"{dataset}/meta/episodes/chunk-000/file-000.parquet")
    table = pd.read_parquet(f"{dataset}/data/chunk-000/file-000.parquet")
    tasks = pd.read_parquet(f"{dataset}/meta/tasks.parquet")
    fps = 30.0
    rng = np.random.default_rng(seed)
    picks = []
    for _, row in episodes.iloc[np.linspace(
            0,
            len(episodes) - 1, num_frames).astype(int)].iterrows():
        lo, hi = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        index = int(rng.integers(lo, hi))
        while index in exclude:
            index = int(rng.integers(lo, hi))
        picks.append((index, row))

    by_video = collections.defaultdict(list)
    for index, row in picks:
        for camera in cameras:
            key = f"videos/observation.images.{camera}"
            file_index = int(row[f"{key}/file_index"])
            frame = int(round(float(row[f"{key}/from_timestamp"]) *
                              fps)) + index - int(row["dataset_from_index"])
            by_video[(camera, file_index)].append((frame, index))
    frames = {}
    for (camera, file_index), wanted in by_video.items():
        wanted = dict(wanted)
        container = av.open(
            f"{dataset}/videos/observation.images.{camera}/chunk-000/file-{file_index:03d}.mp4"
        )
        for position, decoded in enumerate(container.decode(video=0)):
            if position in wanted:
                frames[(camera,
                        wanted[position])] = decoded.to_ndarray(format="rgb24")
            if position > max(wanted):
                break
    observations = []
    for index, _ in picks:
        state = np.asarray(table["observation.state"].iloc[index],
                           dtype=np.float32)
        task = tasks.index[int(table["task_index"].iloc[index])]
        observations.append((index, {
            f"observation/{name}": frames[(camera, index)]
            for name, camera in (("image", cameras[0]),
                                 ("wrist_image", cameras[1]))
        }
                             | {
                                 "observation/state": state,
                                 "prompt": task
                             }))
    return observations


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--cameras",
        default="top,wrist",
        help="dataset cameras for observation/image, wrist_image")
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument(
        "--exclude",
        default="",
        help="dataset rows kept out of calibration (held-out frames)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file

    if importlib.util.find_spec("lerobot") is None:
        for name in ("lerobot", "lerobot.common", "lerobot.common.datasets",
                     "lerobot.common.datasets.lerobot_dataset"):
            sys.modules[name] = types.ModuleType(name)
    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    to_precision = PaliGemmaWithExpertModel.to_bfloat16_for_selected_params
    PaliGemmaWithExpertModel.to_bfloat16_for_selected_params = (
        lambda self, precision="bfloat16": to_precision(self, "float32"))
    train_config = openpi_config.get_config(args.config)
    policy = policy_config.create_trained_policy(train_config,
                                                 args.checkpoint,
                                                 pytorch_device="cpu")
    policy._model.to(torch.float32)
    pwe = policy._model.paligemma_with_expert

    state = {"valid": None}
    amax = {}

    forward = pwe.forward

    def track_prefix(*a, **kw):
        embeds = kw.get("inputs_embeds")
        is_prefix = embeds is not None and embeds[0] is not None
        if is_prefix:
            pos = kw["position_ids"][0]
            state["valid"] = torch.cat(
                [torch.ones(1, dtype=torch.bool), pos[1:] != pos[:-1]])
        out = forward(*a, **kw)
        state["valid"] = None
        return out

    pwe.forward = track_prefix

    def hook(name):

        def record(module, inputs, output):
            if state["valid"] is None:
                return None
            x = inputs[0][0][state["valid"]].abs().amax(dim=0).float()
            amax[name] = x if name not in amax else torch.maximum(
                amax[name], x)
            return None

        return record

    for i, layer in enumerate(pwe.paligemma.language_model.layers):
        for projection in _PROJECTIONS:
            module = layer.get_submodule(projection)
            module.register_forward_hook(
                hook(f"model.layers.{i}.{projection}"))

    exclude = {int(v) for v in args.exclude.split(",") if v}
    observations = sample_observations(args.dataset, args.num_frames, exclude,
                                       args.cameras.split(","))
    for n, (index, observation) in enumerate(observations):
        with torch.no_grad():
            policy.infer(observation)
        print(
            f"[{n + 1}/{len(observations)}] row {index}: {observation['prompt']!r}",
            flush=True)
    save_file({k: v.contiguous() for k, v in amax.items()}, args.out)
    worst = sorted(((float(v.max()), k) for k, v in amax.items()),
                   reverse=True)[:4]
    print(
        f"{len(amax)} projections -> {args.out}; largest input abs-max: {worst}"
    )


if __name__ == "__main__":
    main()
