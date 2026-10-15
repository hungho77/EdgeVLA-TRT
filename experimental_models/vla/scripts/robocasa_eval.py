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
"""RoboCasa (v0.2, 24 kitchen tasks) success rate of an EdgeVLA-TRT policy server.

The rollout is Xiaomi-Robotics-1's eval_robocasa/main.py, imported from its repository (--official) for the task
list, step limits, environment construction and observation rendering: per episode the environment is built with
seed start_seed + i, the three views (two base cameras and the left wrist, 256 x 256, flipped upright) and the robot
state (7 joints and the gripper) go to the server with the episode's instruction, and the first --action-length rows
of the returned chunk are stepped, until success or the task's step limit. The frames are sent raw; the server does
the client's centre crop and the processor's steps itself. The simulator runs here (RoboCasa's environment,
MUJOCO_GL=egl); the model runs in the C++ server, spawned over stdio or reached over TCP.

    MUJOCO_GL=egl python robocasa_eval.py --official Xiaomi-Robotics-1 --episodes 5 \\
        --server-cmd "xr1_policy_server --engineDir engines" --out robocasa.json

--out is rewritten after every episode, and a rerun with the same --out skips the episodes it already holds.
"""

import argparse
import json
import os
import shlex
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vla_policy_client import VlaPolicyClient  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--official",
                        required=True,
                        help="XiaomiRobotics/Xiaomi-Robotics-1 clone")
    parser.add_argument("--server-cmd")
    parser.add_argument("--port", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--server-env",
                        action="append",
                        default=[],
                        help="KEY=VALUE for the server only")
    parser.add_argument("--tasks",
                        default="all",
                        help="comma-separated task names (default: all)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--action-length", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.official, "eval_robocasa"))
    import main as official

    env = None
    if args.server_cmd:
        env = dict(os.environ)
        env.update(dict(kv.split("=", 1) for kv in args.server_env))
    client = VlaPolicyClient(
        server_cmd=shlex.split(args.server_cmd) if args.server_cmd else None,
        host=args.host if args.port else None,
        port=args.port,
        env=env)
    print("server:", client.info, flush=True)
    cameras = client.info["cameras"]
    tasks = list(official.TASK_MAX_STEPS
                 ) if args.tasks == "all" else args.tasks.split(",")
    results = json.load(open(args.out)) if os.path.exists(args.out) else {
        "episodes": {}
    }

    for task in tasks:
        for i in range(args.episodes):
            key = f"{task}/{i}"
            if key in results["episodes"]:
                continue
            seed = args.start_seed + i
            sim = official.create_env(env_name=task,
                                      render_onscreen=False,
                                      seed=seed)
            sim.reset()
            controller = sim.robots[0].composite_controller
            base = np.eye(4)
            base[:3, :3] = controller.part_controllers[
                controller.arms[0]].origin_ori
            base[:3, 3] = controller.part_controllers[
                controller.arms[0]].origin_pos
            limit = official.TASK_MAX_STEPS[task]
            instruction = sim.get_ep_meta()["lang"]
            step, success, calls, first = 0, False, [], True
            while step < limit and not success:
                rgbs, proprios, _ = official.render_obs(
                    sim, official.camera_names, base)
                t0 = time.perf_counter()
                actions = client.act(
                    {
                        c: rgbs[k]
                        for k, c in enumerate(cameras)
                    },
                    proprios,
                    instruction,
                    reset=first)
                calls.append(1000 * (time.perf_counter() - t0))
                first = False
                for row in actions[:args.action_length]:
                    action = np.zeros(sim.action_spec[0].shape)
                    action[:row.shape[0]] = row
                    sim.step(action)
                    step += 1
                    if sim._check_success():
                        success = True
                        break
                    if step >= limit:
                        break
            sim.close()
            results["episodes"][key] = {
                "success": success,
                "steps": step,
                "seed": seed,
                "instruction": instruction,
                "policy_ms": float(np.median(calls))
            }
            done = [v["success"] for v in results["episodes"].values()]
            print(
                f"{task} ep {i} (seed {seed}): {'success' if success else 'fail'} in {step} steps, policy "
                f"{np.median(calls):.0f} ms | running {100 * np.mean(done):.1f}% of {len(done)}",
                flush=True)
            json.dump(results, open(args.out, "w"), indent=1)
    client.close()
    per_task = {}
    for key, value in results["episodes"].items():
        per_task.setdefault(key.split("/")[0], []).append(value["success"])
    for task, values in per_task.items():
        print(f"{task}: {sum(values)}/{len(values)}")
    total = [v for values in per_task.values() for v in values]
    print(f"overall {100 * np.mean(total):.1f}% of {len(total)} episodes")


if __name__ == "__main__":
    main()
