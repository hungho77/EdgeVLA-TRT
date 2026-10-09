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
"""A real-robot control loop against a policy server, driven by a recorded LeRobot dataset instead of hardware.

Tick i at --hz wall-clock rate shows the policy dataset frame (start + i): its camera images and its recorded state,
as a robot would read them; the action the loop would send is recorded rather than executed. The loop is the one a
robot runs: replace ``snapshot`` with the robot's camera and joint reads and send ``action`` to its controller.

    python replay_robot.py --port 5555 --dataset so101-multitask --cameras top=top wrist=wrist --hz 30 --ticks 300

Reports stalls (ticks with no action once the first chunk landed), the planner latency in ticks, and at each chunk
switch the jump between the old chunk's action for that tick and the new one's, split into switches that landed
inside the frozen rows (which RTC reproduces exactly) and beyond them.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vla_policy_client  # noqa: E402


def load_episode(root,
                 cameras,
                 start,
                 count,
                 video_file="file-000.mp4",
                 data_file="file-000.parquet"):
    """Frames {server camera: [count, H, W, 3]}, states [count, D] and the task, from a LeRobot v3 dataset."""
    import av
    import pandas as pd
    frames = {}
    for dataset_key, server_name in cameras.items():
        container = av.open(
            f"{root}/videos/observation.images.{dataset_key}/chunk-000/{video_file}"
        )
        decoded = []
        for index, frame in enumerate(container.decode(video=0)):
            if index >= start + count:
                break
            if index >= start:
                decoded.append(frame.to_ndarray(format="rgb24"))
        frames[server_name] = decoded
    table = pd.read_parquet(f"{root}/data/chunk-000/{data_file}")
    states = np.stack(
        table["observation.state"].iloc[start:start +
                                        count].to_numpy()).astype(np.float32)
    tasks = pd.read_parquet(f"{root}/meta/tasks.parquet")
    task = tasks.index[int(table["task_index"].iloc[start])]
    return frames, states, task


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-cmd",
                        help="spawn the server over stdio (else --port)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dataset",
                        required=True,
                        help="LeRobot v3 dataset root")
    parser.add_argument("--cameras",
                        nargs="+",
                        required=True,
                        help="dataset_key=server_camera ...")
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--frozen", type=int, default=5)
    parser.add_argument("--no-rtc", action="store_true")
    parser.add_argument("--task", help="override the dataset task")
    parser.add_argument("--state-dim",
                        type=int,
                        help="pad or cut the recorded state to this width")
    args = parser.parse_args()

    cameras = dict(pair.split("=", 1) for pair in args.cameras)
    frames, states, task = load_episode(args.dataset, cameras, args.start,
                                        args.ticks)
    task = args.task or task
    if args.state_dim:
        states = np.pad(
            states, ((0, 0), (0, max(
                0, args.state_dim - states.shape[1]))))[:, :args.state_dim]
    client = (vla_policy_client.VlaPolicyClient(
        server_cmd=args.server_cmd.split()) if args.server_cmd
              else vla_policy_client.VlaPolicyClient(host=args.host,
                                                     port=args.port))
    print("server:", client.info, flush=True)

    def snapshot(tick):
        return {
            name: frames[name][tick]
            for name in frames
        }, states[tick], task

    controller = vla_policy_client.AsyncChunkedController(client,
                                                          snapshot,
                                                          args.overlap,
                                                          args.frozen,
                                                          rtc=not args.no_rtc)
    controller.warmup(snapshot(0))
    period = 1.0 / args.hz
    stalls, lags, seam_frozen, seam_beyond, late, stall_ticks = 0, [], [], [], 0, []
    previous_tick, previous_chunk = -1, None
    next_time = time.perf_counter()
    for tick in range(args.ticks):
        action = controller.step(tick)
        if controller.current_tick != previous_tick:
            if previous_chunk is not None:
                lag = tick - controller.current_tick
                old_row = tick - previous_tick
                if old_row < len(previous_chunk):
                    jump = float(
                        np.abs(controller.current[lag] -
                               previous_chunk[old_row]).max())
                    (seam_frozen
                     if lag < args.frozen else seam_beyond).append(jump)
                lags.append(lag)
            previous_tick, previous_chunk = controller.current_tick, controller.current
        if action is None and previous_chunk is not None:
            stalls += 1
            stall_ticks.append(tick)
        next_time += period
        delay = next_time - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            late += 1
    controller.close()
    client.close()
    latency = np.median(
        controller.latencies_ms) if controller.latencies_ms else float("nan")
    print(
        f"{args.ticks} ticks at {args.hz:.0f} Hz, chunk {controller.rows}, overlap {controller.overlap}, "
        f"frozen {args.frozen}, rtc {controller.scheme if not args.no_rtc else None}"
    )
    print(
        f"  chunks {controller.chunks}, policy call median {latency:.0f} ms, adoption lag ticks "
        f"{sorted(set(lags))}, stalls {stalls}{' at ticks ' + str(stall_ticks[:10]) if stall_ticks else ''}, "
        f"late ticks {late}")
    print("  policy calls ms:",
          [round(v) for v in controller.latencies_ms[:12]])
    fmt = lambda v: f"{max(v):.4f} over {len(v)}" if v else "none"
    print(
        f"  switch jump max |d|: within frozen rows {fmt(seam_frozen)}, beyond {fmt(seam_beyond)}"
    )


if __name__ == "__main__":
    main()
