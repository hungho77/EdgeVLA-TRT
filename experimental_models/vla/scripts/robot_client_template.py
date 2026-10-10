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
"""A real-robot control loop against a policy server over TCP, with the robot I/O left as three hooks.

Copy this file and vla_policy_client.py (both need only NumPy) to the machine that drives the robot, fill in
``Robot.read_cameras`` / ``read_state`` / ``send_action`` with its camera and joint code, and point it at the
server (on the Orin: ``<family>_policy_server --engineDir ... --port 5555 --host 0.0.0.0``):

    python robot_client_template.py --host 192.168.1.20 --port 5555 \\
        --task "Pick up the red cube and place it into the steel pot." --dry-run

The loop ticks at --hz (the training data's rate); the next chunk is planned in the background with real-time
chunking at the switches (vla_policy_client.AsyncChunkedController). Every command is limited to --max-step from
the previous one, starting from the measured state, so a bad chunk cannot jump the arm. --dry-run reads the robot
and prints the commands without sending them. Ctrl-C stops the loop; the robot then holds its last command.
"""

import argparse
import time

import numpy as np
from vla_policy_client import AsyncChunkedController, VlaPolicyClient

# Robot camera -> server camera for families whose servers rename them; the others name the cameras as the SO101
# datasets record them ("top", "wrist").
CAMERA_NAMES = {
    "pi05": {
        "top": "observation/image",
        "wrist": "observation/wrist_image"
    },
}
# (overlap, frozen) in ticks: frozen must exceed the planner latency in ticks (LAN round trip included).
RTC_DEFAULTS = {
    "smolvla": (10, 6),
    "pi05": (20, 14),
    "gr00t_n17": (10, 7),
    "turbovla": (6, 3)
}


class Robot:
    """The robot's I/O. Replace the bodies with the robot's own code."""

    def read_cameras(self):
        """{"top": [H, W, 3] uint8 RGB, "wrist": [H, W, 3] uint8 RGB}: the views as the training data recorded them
        (OpenCV captures are BGR: convert them)."""
        raise NotImplementedError

    def read_state(self):
        """The joint state in the training data's units and order, e.g. SO101: shoulder_pan, shoulder_lift,
        elbow_flex, wrist_flex, wrist_roll, gripper."""
        raise NotImplementedError

    def send_action(self, action):
        """Command one action row (the same units and order as the state)."""
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--task",
                        required=True,
                        help="the instruction, verbatim as in training")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--overlap", type=int, help="RTC overlap rows")
    parser.add_argument("--frozen", type=int, help="RTC frozen rows")
    parser.add_argument("--no-rtc", action="store_true")
    parser.add_argument(
        "--max-step",
        type=float,
        default=8.0,
        help="largest change of any joint per tick, in action units")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="print the commands instead of sending them")
    args = parser.parse_args()

    robot = Robot()
    client = VlaPolicyClient(host=args.host, port=args.port)
    family = client.info["family"]
    print("server:", client.info, flush=True)
    names = CAMERA_NAMES.get(family, {"top": "top", "wrist": "wrist"})
    missing = [
        n for n in names.values() if n not in client.info.get("cameras", [])
    ]
    if missing:
        raise SystemExit(
            f"the server expects cameras {client.info.get('cameras')}; set CAMERA_NAMES for {family}"
        )
    overlap, frozen = RTC_DEFAULTS.get(family, (10, 6))
    overlap = args.overlap if args.overlap is not None else overlap
    frozen = args.frozen if args.frozen is not None else frozen

    def snapshot(tick):
        frames = {names[k]: v for k, v in robot.read_cameras().items()}
        return frames, np.asarray(robot.read_state(),
                                  dtype=np.float32), args.task

    controller = AsyncChunkedController(client,
                                        snapshot,
                                        overlap=overlap,
                                        frozen=frozen,
                                        rtc=not args.no_rtc)
    controller.warmup(snapshot(0))
    last = np.asarray(robot.read_state(), dtype=np.float32)
    period = 1.0 / args.hz
    held = 0
    start = time.perf_counter()
    try:
        for tick in range(int(args.seconds * args.hz)):
            action = controller.step(tick)
            if action is None:
                held += 1
            else:
                command = last + np.clip(
                    np.asarray(action, np.float32) - last, -args.max_step,
                    args.max_step)
                if args.dry_run:
                    print(tick, np.round(command, 2), flush=True)
                else:
                    robot.send_action(command)
                last = command
            time.sleep(
                max(0.0, start + (tick + 1) * period - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        controller.close()
        client.close()
    latency = np.median(
        controller.latencies_ms) if controller.latencies_ms else float("nan")
    print(
        f"{controller.chunks} chunks, policy call median {latency:.0f} ms, last lag {controller.last_lag} ticks, "
        f"{held} held ticks")


if __name__ == "__main__":
    main()
