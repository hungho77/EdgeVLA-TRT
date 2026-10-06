# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Python client for gr00t_policy_server: raw camera frames + raw robot state in, absolute actions out.

Applies GR00T N1.7's evaluation preprocessing on the robot side (letterbox, shortest
edge 256, 0.95 center crop, shortest edge 256 again; lowercase instruction without
punctuation) and talks to the C++ server over stdin/stdout.

    client = Gr00tPolicyClient(server_cmd, video_keys=["top", "wrist"])
    actions = client.act({"top": top_rgb, "wrist": wrist_rgb}, raw_state, "Pick blue cube ...")
    # actions: [action_horizon, raw action dim] absolute joint targets, e.g. [16, 6] for SO101

--golden runs the official GR00T Gr00tPolicy on the same raw inputs and noise and
compares its actions with the server's (needs the GR00T N1.7 source and its deps).
"""

import argparse
import json
import os
import re
import subprocess
import tempfile

import cv2
import numpy as np


def preprocess_image(rgb, shortest_edge=256, crop_fraction=0.95):
    """GR00T N1.7 eval image transform (LetterBoxPad, SmallestMaxSize, FractionalCenterCrop, SmallestMaxSize)."""

    def smallest_max_size(img):
        h, w = img.shape[:2]
        scale = shortest_edge / min(h, w)
        size = (round(w * scale), round(h * scale))
        return img if size == (w, h) else cv2.resize(
            img, size, interpolation=cv2.INTER_AREA)

    h, w = rgb.shape[:2]
    if h != w:
        side = max(h, w)
        pad_h, pad_w = side - h, side - w
        rgb = cv2.copyMakeBorder(rgb,
                                 pad_h // 2,
                                 pad_h - pad_h // 2,
                                 pad_w // 2,
                                 pad_w - pad_w // 2,
                                 cv2.BORDER_CONSTANT,
                                 value=0)
    rgb = smallest_max_size(rgb)
    h, w = rgb.shape[:2]
    ch, cw = max(1, int(h * crop_fraction)), max(1, int(w * crop_fraction))
    y, x = (h - ch) // 2, (w - cw) // 2
    return smallest_max_size(rgb[y:y + ch, x:x + cw])


def formalize_language(text):
    return re.sub(r"[^\w\s]", "", text.lower())


class Gr00tPolicyClient:

    def __init__(self, server_cmd, video_keys, env=None):
        self.video_keys = list(video_keys)
        self.proc = subprocess.Popen(server_cmd,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     env=env,
                                     text=True,
                                     bufsize=1)
        self.info = self._read()
        self.tmp = tempfile.mkdtemp(prefix="gr00t_client_")

    def _read(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("gr00t_policy_server exited")
            if line.startswith("{"):
                reply = json.loads(line)
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply

    def act(self,
            frames,
            raw_state,
            instruction,
            rtc=None,
            seed=None,
            noise_file=None,
            reset=False):
        """frames: {video_key: HxWx3 uint8 RGB}; raw_state: concatenated raw state groups."""
        paths = []
        for key in self.video_keys:
            path = os.path.join(self.tmp, f"{key}.png")
            cv2.imwrite(
                path,
                cv2.cvtColor(preprocess_image(frames[key]), cv2.COLOR_RGB2BGR))
            paths.append(path)
        request = {
            "images": paths,
            "state": [float(v) for v in np.asarray(raw_state).ravel()],
            "instruction": formalize_language(instruction),
            "reset": reset,
        }
        if rtc:
            request["rtc"] = rtc
        if seed is not None:
            request["seed"] = int(seed)
        if noise_file:
            request["noise_file"] = noise_file
            request["debug"] = True
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        self.last_timing = reply.get("timing_ms")
        self.last_model_actions = reply.get("model_actions")
        return np.asarray(reply["actions"], dtype=np.float32)

    def close(self):
        self.proc.stdin.write("\n")
        self.proc.stdin.flush()
        self.proc.wait(timeout=30)


def encode_actions(spec, absolute, raw_state):
    """Inverse action decoding for chunk rows [0, len(absolute)), as Gr00tProcessing::encodeActions."""
    out = np.zeros((len(absolute), spec["max_action_dim"]), dtype=np.float32)
    state_offsets, offset = {}, 0
    for g in spec["state"]:
        state_offsets[g["name"]] = offset
        offset += g["dim"]
    offset = 0
    for g in spec["action"]:
        lo, hi = np.asarray(g["min"]), np.asarray(g["max"])
        if lo.ndim == 2:
            lo, hi = lo[:len(absolute)], hi[:len(absolute)]
        v = absolute[:, offset:offset + g["dim"]].astype(np.float64)
        if g["relative"]:
            ref = state_offsets[g["reference_state"]]
            v = v - raw_state[ref:ref + g["dim"]]
        degenerate = np.isclose(hi, lo)
        a = np.clip(2 * (v - lo) / np.where(degenerate, 1, hi - lo) - 1, -1, 1)
        out[:, offset:offset + g["dim"]] = np.where(degenerate, 0.0, a)
        offset += g["dim"]
    return out


def load_dataset_frame(args, frame):
    import av
    import pandas as pd
    frames = {}
    for key in args.video_keys:
        container = av.open(
            f"{args.dataset}/videos/observation.images.{key}/chunk-000/{args.video_file}"
        )
        for index, decoded in enumerate(container.decode(video=0)):
            if index == frame:
                frames[key] = decoded.to_ndarray(format="rgb24")
                break
    table = pd.read_parquet(f"{args.dataset}/data/chunk-000/{args.data_file}")
    raw_state = np.asarray(table["observation.state"].iloc[frame],
                           dtype=np.float32)
    tasks = pd.read_parquet(f"{args.dataset}/meta/tasks.parquet")
    return frames, raw_state, tasks.index[int(table["task_index"].iloc[frame])]


def rtc_check(args):
    """Frozen RTC rows must reproduce the previous chunk's committed absolute actions after the arm moved."""
    rtc = {"overlap": 8, "frozen": 2, "ramp_rate": 6.0}
    client = Gr00tPolicyClient(args.server_cmd.split(), args.video_keys)
    previous = None
    for i, frame in enumerate((args.frame, args.frame + args.rtc_stride)):
        frames, raw_state, instruction = load_dataset_frame(args, frame)
        actions = client.act(frames,
                             raw_state,
                             instruction,
                             rtc=rtc if i else None,
                             seed=i,
                             reset=i == 0)
        if previous is not None:
            start = len(previous) - rtc["overlap"]
            committed = previous[start:start + rtc["frozen"]]
            diff = np.abs(actions[:rtc["frozen"]] - committed)
            print(
                f"frames {args.frame} -> {frame}, state moved by up to {np.abs(raw_state - state0).max():.2f}: "
                f"frozen rows vs committed actions max |d| = {diff.max():.4f}")
            print("  committed:", np.round(committed[0], 3).tolist())
            print("  new      :", np.round(actions[0], 3).tolist())
        previous, state0 = actions, raw_state
    client.close()


def golden(args):
    """Official Gr00tPolicy (fp32) vs the server on one raw dataset observation with the same noise."""
    import sys
    from unittest import mock

    import torch
    sys.path.insert(0, args.gr00t_src)
    import gr00t.model.gr00t_n1d7.gr00t_n1d7  # noqa: F401
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7  # noqa: F401
    from gr00t.model.gr00t_n1d7.image_augmentations import \
        build_image_transformations_albumentations
    from gr00t.policy import gr00t_policy

    frames, raw_state, instruction = load_dataset_frame(args, args.frame)

    _, official_eval = build_image_transformations_albumentations([256, 256],
                                                                  [230, 230],
                                                                  0, None, 256,
                                                                  0.95)
    for key, rgb in frames.items():
        ours = preprocess_image(rgb)
        theirs = official_eval(image=rgb)["image"]
        print(
            f"image '{key}' {rgb.shape} -> {ours.shape}: max |ours - official| = "
            f"{int(np.abs(ours.astype(int) - theirs.astype(int)).max())}")

    # Two calls: a fresh chunk, then an RTC chunk inpainted from it, each with its own fixed noise.
    rng = np.random.default_rng(args.noise_seed)
    noises = [
        rng.standard_normal((40, 132)).astype(np.float32) for _ in range(2)
    ]
    noise_files = []
    for i, noise in enumerate(noises):
        noise_files.append(os.path.join(tempfile.mkdtemp(), f"noise{i}.f32"))
        noise.tofile(noise_files[-1])
    rtc = {"overlap": 8, "frozen": 2, "ramp_rate": 6.0}

    # Server first: the fp32 reference model is large, and the board's memory is shared with the GPU.
    client = Gr00tPolicyClient(args.server_cmd.split(), args.video_keys)
    ours, ours_model = [], []
    for i in range(2):
        ours.append(
            client.act(frames,
                       raw_state,
                       instruction,
                       rtc=rtc if i else None,
                       noise_file=noise_files[i],
                       reset=i == 0))
        ours_model.append(
            np.asarray(client.last_model_actions, dtype=np.float32))
        print(f"server timing (ms): {client.last_timing}")
    client.close()

    policy = gr00t_policy.Gr00tPolicy(model_path=args.checkpoint,
                                      embodiment_tag="new_embodiment",
                                      device="cpu")
    policy.model.float()
    real_randn = torch.randn
    current = {}

    def fixed_randn(*size, **kwargs):
        shape = tuple(
            kwargs.get(
                "size", size[0] if len(size) == 1
                and isinstance(size[0], (tuple, list)) else size))
        if shape == (1, 40, 132):
            return torch.from_numpy(current["noise"])[None].to(
                kwargs.get("dtype", torch.float32))
        return real_randn(*size, **kwargs)

    state_split = {
        "single_arm": raw_state[None, None, :5],
        "gripper": raw_state[None, None, 5:6]
    }
    observation = {
        "video": {
            k: v[None, None]
            for k, v in frames.items()
        },
        "state": state_split,
        "language": {
            "annotation.human.task_description": [[instruction]]
        },
    }
    real_get_action = policy.model.get_action
    horizon, dim = ours[0].shape
    print(
        f"instruction: {instruction!r}; raw state {np.round(raw_state, 2).tolist()}"
    )
    spec = json.load(open(args.processing))
    previous = None
    for i in range(2):

        def spy(**kwargs):
            if previous is not None:
                # The official RTC path copies rows [horizon - overlap, horizon) of action_input["action"]; they
                # hold the previous absolute chunk re-encoded for this call, as Gr00tN17Policy does.
                seeded = torch.zeros(1, 40, 132)
                seeded[0, horizon - rtc["overlap"]:horizon] = torch.from_numpy(
                    encode_actions(spec, previous[horizon - rtc["overlap"]:],
                                   raw_state))
                kwargs["inputs"]["action"] = seeded
                kwargs["options"] = {
                    "action_horizon": horizon,
                    "rtc_overlap_steps": rtc["overlap"],
                    "rtc_frozen_steps": rtc["frozen"],
                    "rtc_ramp_rate": rtc["ramp_rate"],
                }
            out = real_get_action(**kwargs)
            current["action_pred"] = out["action_pred"].clone()
            return out

        current["noise"] = noises[i]
        with mock.patch.object(gr00t_policy, "_rec_to_dtype", lambda x, dtype: x), \
                mock.patch.object(policy.model, "get_action", spy), \
                mock.patch("torch.randn", side_effect=fixed_randn):
            official, _ = policy.get_action(observation)
        official = np.concatenate(
            [official["single_arm"][0], official["gripper"][0]], axis=-1)
        previous = official
        diff = np.abs(ours[i] - official)
        model_diff = np.abs(ours_model[i].reshape(horizon, -1)[:, :dim] -
                            current["action_pred"][0, :horizon, :dim].numpy())
        label = f"RTC {rtc}" if i else "fresh chunk"
        print(
            f"{label}: absolute actions {ours[i].shape} max |ours - official| = {diff.max():.4f}, "
            f"mean {diff.mean():.4f}, official range [{official.min():.2f}, {official.max():.2f}]; "
            f"normalized max |d| = {model_diff.max():.4f}")
        print("  first step, ours    :", np.round(ours[i][0], 3).tolist())
        print("  first step, official:", np.round(official[0], 3).tolist())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-cmd",
                        required=True,
                        help="gr00t_policy_server command line")
    parser.add_argument("--video-keys", nargs="+", default=["top", "wrist"])
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--gr00t-src")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset",
                        help="LeRobot v3 dataset root (videos/, data/, meta/)")
    parser.add_argument("--video-file", default="file-000.mp4")
    parser.add_argument("--data-file", default="file-000.parquet")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--processing",
                        help="processing.json, for the golden RTC round")
    parser.add_argument("--rtc-check", action="store_true")
    parser.add_argument("--rtc-stride", type=int, default=8)
    args = parser.parse_args()
    if args.golden:
        golden(args)
    if args.rtc_check:
        rtc_check(args)


if __name__ == "__main__":
    main()
