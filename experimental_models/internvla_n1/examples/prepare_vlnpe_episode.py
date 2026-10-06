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
"""Turn a rendered VLN-PE R2R trajectory into input for internvla_n1_s2_bench.

Downloads one scene archive and the R2R annotations from the ungated
InternRobotics/IROS-2025-Challenge-Nav dataset, writes the trajectory's RGB
frames resized to InternNav's 384x384 as 000000.png, 000001.png, ..., and its
R2R instruction to instruction.txt.

    python prepare_vlnpe_episode.py --scene gZ6f7yhEvPG --trajectory 1304 --out episode
    internvla_n1_s2_bench ... --framesDir episode --steps 102 \\
        --instruction "$(cat episode/instruction.txt)"
"""

import argparse
import gzip
import json
import os
import tarfile
import urllib.request

import numpy as np
from PIL import Image

BASE = "https://huggingface.co/datasets/InternRobotics/IROS-2025-Challenge-Nav/resolve/main/vln_pe"


def fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="gZ6f7yhEvPG")
    parser.add_argument("--trajectory", default="1304")
    parser.add_argument("--out", default="episode")
    parser.add_argument("--cache", default="vlnpe_cache")
    parser.add_argument("--size",
                        type=int,
                        default=384,
                        help="InternNav resize_w/resize_h")
    args = parser.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    archive = fetch(f"{BASE}/traj_data/r2r/{args.scene}.tar.gz",
                    os.path.join(args.cache, f"{args.scene}.tar.gz"))
    member = f"{args.scene}/{args.trajectory}/videos/chunk-000/observation.images.rgb/rgb.npy"
    with tarfile.open(archive) as tar:
        tar.extract(member, args.cache)
    rgb = np.load(os.path.join(args.cache, member))
    for i, frame in enumerate(rgb):
        Image.fromarray(frame).convert("RGB").resize(
            (args.size, args.size)).save(os.path.join(args.out,
                                                      f"{i:06d}.png"))

    instruction = None
    for split in ("train", "val_seen", "val_unseen"):
        annotations = fetch(f"{BASE}/raw_data/r2r/{split}/{split}.json.gz",
                            os.path.join(args.cache, f"r2r_{split}.json.gz"))
        for episode in json.load(gzip.open(annotations))["episodes"]:
            if str(episode.get("trajectory_id")
                   ) == args.trajectory and args.scene in episode["scene_id"]:
                instruction = episode["instruction"]["instruction_text"].strip(
                )
                break
        if instruction:
            break
    if instruction is None:
        raise SystemExit(
            f"no R2R instruction for trajectory {args.trajectory} in {args.scene}"
        )
    with open(os.path.join(args.out, "instruction.txt"), "w") as f:
        f.write(instruction)
    print(f"{len(rgb)} frames -> {args.out}; instruction: {instruction}")


if __name__ == "__main__":
    main()
