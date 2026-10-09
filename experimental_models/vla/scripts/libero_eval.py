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
"""LIBERO success rate of an EdgeVLA-TRT policy server.

The simulator runs here; the model runs in a C++ policy server started as a subprocess (one JSON request per stdin
line), so this needs only a LIBERO Python environment (robosuite 1.4 with mujoco 2.3, MUJOCO_GL=egl) and no
TensorRT. Each policy adapter follows its checkpoint's own LIBERO evaluation conventions (camera flip, state layout,
gripper sign, chunk rows executed per call).

Protocol: LIBERO's fixed initial states (episode i uses init state i), 10 no-op steps to let objects settle, then
the policy until success or --max-steps.

    MUJOCO_GL=egl python libero_eval.py --policy gr00t_n17 --suite libero_spatial --episodes 10 \\
        --server-env LD_LIBRARY_PATH=... --server-env EDGELLM_PLUGIN_PATH=... \\
        --server-cmd "gr00t_policy_server --llmEngineDir e/llm --multimodalEngineDir e --actionEngineDir e/action" \\
        --out results.json

--out is rewritten after every episode, and a rerun with the same --out skips the episodes it already holds.
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "..")
sys.path.insert(0,
                os.path.join(REPO, "experimental_models", "gr00t", "examples"))

NO_OP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
SETTLE_STEPS = 10


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / den


class JsonServer:
    """A policy server speaking one JSON object per line; non-JSON stdout lines are runtime logs."""

    def __init__(self, cmd, env_overrides):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(env_overrides)
        self.proc = subprocess.Popen(shlex.split(cmd),
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     env=env,
                                     text=True,
                                     bufsize=1)
        self.info = self.read()

    def read(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("policy server exited")
            if line.startswith("{"):
                reply = json.loads(line)
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply

    def request(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        return self.read()

    def close(self):
        try:
            self.proc.stdin.write("\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001 -- the server may already be gone
            self.proc.kill()


def mat2quat(rmat):
    """robosuite's mat2quat (x, y, z, w): eigenvector of the largest eigenvalue, as LeRobot's X-VLA utils."""
    m = np.asarray(rmat, dtype=np.float32)[:3, :3]
    k = np.array([
        [m[0, 0] - m[1, 1] - m[2, 2], 0.0, 0.0, 0.0],
        [m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], 0.0, 0.0],
        [
            m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1],
            0.0
        ],
        [
            m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1],
            m[0, 0] + m[1, 1] + m[2, 2]
        ],
    ],
                 dtype=np.float32) / 3.0
    w, v = np.linalg.eigh(k)
    q = v[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        q = -q
    return q[[1, 2, 3, 0]]


def rotate6d_to_axis_angle(r6d):
    a1, a2 = r6d[0:3], r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-6)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-6)
    return quat2axisangle(
        mat2quat(np.stack([b1, b2, np.cross(b1, b2)], axis=-1)))


class Gr00tAdapter:
    """gr00t_policy_server, as Isaac-GR00T's LIBERO env: views flipped 180 degrees, state [xyz, axis-angle,
    gripper qpos], gripper action binarized from [0, 1] and inverted to LIBERO's -1 = open."""

    control_mode = "relative"

    def __init__(self, server, preprocess, rows, binarize_gripper=True):
        self.server = server
        self.preprocess = preprocess
        self.binarize_gripper = binarize_gripper
        self.rows = rows
        self.tmp = tempfile.mkdtemp(prefix="libero_eval_")
        self.first = True
        if preprocess:
            from gr00t_policy_client import (formalize_language,
                                             preprocess_image)
            self.formalize, self.transform = formalize_language, preprocess_image

    def reset(self):
        self.first = True

    def act(self, obs, instruction):
        paths = []
        for key, cam in (("image", "agentview_image"),
                         ("wrist_image", "robot0_eye_in_hand_image")):
            frame = np.ascontiguousarray(obs[cam][::-1, ::-1])
            if self.preprocess:
                frame = self.transform(frame)
            path = os.path.join(self.tmp, f"{key}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            paths.append(path)
        state = np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]
        ])
        reply = self.server.request({
            "images":
            paths,
            "state": [float(v) for v in state],
            "instruction":
            self.formalize(instruction) if self.preprocess else instruction,
            "reset":
            self.first,
        })
        self.first = False
        actions = np.asarray(reply["actions"],
                             dtype=np.float64)[:self.rows, :7].copy()
        if self.binarize_gripper:
            actions[:, 6] = -np.sign(2.0 * actions[:, 6] - 1.0)
        return actions


class XvlaAdapter:
    """xvla_policy_server, as LeRobot's X-VLA LIBERO processors: only the agent view flipped, state [eef pos,
    rot6d of the controller's orientation, 0] padded to 20, absolute end-effector control, ee6d actions turned
    into [pos, axis-angle, gripper +-1]."""

    control_mode = "absolute"

    def __init__(self, server, rows):
        self.server = server
        self.rows = rows
        self.tmp = tempfile.mkdtemp(prefix="libero_eval_")
        self.first = True

    def reset(self):
        self.first = True

    def act(self, obs, instruction):
        cameras = {}
        for key, frame in (("image", obs["agentview_image"][::-1, ::-1]),
                           ("image2", obs["robot0_eye_in_hand_image"])):
            path = os.path.join(self.tmp, f"{key}.png")
            cv2.imwrite(
                path,
                cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
            cameras[key] = path
        mat = np.asarray(obs["ee_ori_mat"], dtype=np.float32)
        state = np.zeros(20, dtype=np.float32)
        state[:3] = obs["robot0_eef_pos"]
        state[3:9] = np.concatenate([mat[:3, 0], mat[:3, 1]])
        reply = self.server.request({
            "cameras": cameras,
            "state": [float(v) for v in state],
            "task": instruction,
            "reset": self.first
        })
        self.first = False
        out = []
        for row in np.asarray(reply["actions"], dtype=np.float64)[:self.rows]:
            out.append(
                np.concatenate([
                    row[:3],
                    rotate6d_to_axis_angle(row[3:9]),
                    [1.0 if row[9] > 0.5 else -1.0]
                ]))
        return out


class Pi05Adapter:
    """pi05_policy_server with openpi's LIBERO client (examples/libero/main.py): both views flipped 180 degrees and
    resized to 224 with PIL bilinear (openpi's resize_with_pad on a square frame), state [eef pos, axis-angle,
    gripper qpos], actions sent to LIBERO as returned."""

    control_mode = "relative"

    def __init__(self, server, rows, size=224):
        from PIL import Image
        self.resize = lambda frame: np.asarray(
            Image.fromarray(frame).resize((size, size), Image.BILINEAR))
        self.server = server
        self.rows = rows
        self.tmp = tempfile.mkdtemp(prefix="libero_eval_")
        self.first = True

    def reset(self):
        self.first = True

    def act(self, obs, instruction):
        cameras = {}
        for slot, cam in (("observation/image", "agentview_image"),
                          ("observation/wrist_image",
                           "robot0_eye_in_hand_image")):
            path = os.path.join(self.tmp, slot.replace("/", "_") + ".png")
            frame = self.resize(np.ascontiguousarray(obs[cam][::-1, ::-1]))
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cameras[slot] = path
        state = np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]
        ])
        reply = self.server.request({
            "cameras": cameras,
            "state": [float(v) for v in state],
            "task": instruction,
            "reset": self.first
        })
        self.first = False
        return np.asarray(reply["actions"], dtype=np.float64)[:self.rows, :7]


class SmolvlaAdapter:
    """smolvla_policy_server with LeRobot's LIBERO processing (LiberoProcessorStep): both views flipped 180 degrees
    as cameras image / image2, state [eef pos, axis-angle, gripper qpos], actions sent to LIBERO as returned."""

    control_mode = "relative"

    def __init__(self, server, rows):
        self.server = server
        self.rows = rows
        self.tmp = tempfile.mkdtemp(prefix="libero_eval_")
        self.first = True

    def reset(self):
        self.first = True

    def act(self, obs, instruction):
        cameras = {}
        for key, cam in (("image", "agentview_image"),
                         ("image2", "robot0_eye_in_hand_image")):
            path = os.path.join(self.tmp, f"{key}.png")
            cv2.imwrite(
                path,
                cv2.cvtColor(np.ascontiguousarray(obs[cam][::-1, ::-1]),
                             cv2.COLOR_RGB2BGR))
            cameras[key] = path
        state = np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]
        ])
        reply = self.server.request({
            "cameras": cameras,
            "state": [float(v) for v in state],
            "task": instruction,
            "reset": self.first
        })
        self.first = False
        return np.asarray(reply["actions"], dtype=np.float64)[:self.rows, :7]


class TurbovlaAdapter:
    """turbovla_policy_server, as TurboVLA's LIBERO rollout: both views flipped 180 degrees as cameras primary /
    wrist, state [eef pos, axis-angle, gripper qpos], actions (gripper already +-1) sent to LIBERO as returned."""

    control_mode = "relative"

    def __init__(self, server, rows):
        self.server = server
        self.rows = rows
        self.tmp = tempfile.mkdtemp(prefix="libero_eval_")
        self.first = True

    def reset(self):
        self.first = True

    def act(self, obs, instruction):
        cameras = {}
        for key, cam in (("primary", "agentview_image"),
                         ("wrist", "robot0_eye_in_hand_image")):
            path = os.path.join(self.tmp, f"{key}.png")
            cv2.imwrite(
                path,
                cv2.cvtColor(np.ascontiguousarray(obs[cam][::-1, ::-1]),
                             cv2.COLOR_RGB2BGR))
            cameras[key] = path
        state = np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]
        ])
        reply = self.server.request({
            "cameras": cameras,
            "state": [float(v) for v in state],
            "task": instruction,
            "reset": self.first
        })
        self.first = False
        return np.asarray(reply["actions"], dtype=np.float64)[:self.rows, :7]


def tf_crop_and_resize(image, box, size):
    """tf.image.crop_and_resize (bilinear) of a float [H, W, C] image to a normalized [y1, x1, y2, x2] box."""
    h, w = image.shape[:2]
    ys = box[0] * (h - 1) + np.arange(size) * (box[2] -
                                               box[0]) * (h - 1) / (size - 1)
    xs = box[1] * (w - 1) + np.arange(size) * (box[3] -
                                               box[1]) * (w - 1) / (size - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    ly, lx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = image[y0][:, x0] * (1 - lx) + image[y0][:, x1] * lx
    bottom = image[y1][:, x0] * (1 - lx) + image[y1][:, x1] * lx
    return top * (1 - ly) + bottom * ly


class OpenvlaAdapter:
    """openvla_policy_server with OpenVLA's LIBERO evaluation (experiments/robot/libero/run_libero_eval.py), its
    TensorFlow image steps reproduced in NumPy / OpenCV / Pillow: agent view flipped 180 degrees, JPEG round trip
    (as the RLDS builder stored frames), Lanczos resize to 224, a centre crop of 90% area resized back to 224
    (the model was fine-tuned with random crops), one action per call, gripper binarized from [0, 1] and inverted."""

    control_mode = "relative"

    def __init__(self,
                 server,
                 rows,
                 unnorm_key="libero_spatial",
                 size=224,
                 crop_area=0.9):
        from PIL import Image
        self.lanczos = lambda frame: np.asarray(
            Image.fromarray(frame).resize((size, size), Image.LANCZOS))
        self.server = server
        self.unnorm_key = unnorm_key
        self.size = size
        side = math.sqrt(crop_area)
        self.box = ((1 - side) / 2, (1 - side) / 2, (1 + side) / 2,
                    (1 + side) / 2)
        self.path = os.path.join(tempfile.mkdtemp(prefix="libero_eval_"),
                                 "image.png")

    def reset(self):
        pass

    def act(self, obs, instruction):
        frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR),
                             cv2.COLOR_BGR2RGB)
        frame = self.lanczos(frame).astype(np.float32) / 255.0
        frame = np.clip(tf_crop_and_resize(frame, self.box, self.size), 0.0,
                        1.0)
        frame = np.clip(np.floor(frame * 255.5), 0, 255).astype(np.uint8)
        cv2.imwrite(self.path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        reply = self.server.request({
            "image": self.path,
            "instruction": instruction,
            "unnorm_key": self.unnorm_key
        })
        action = np.asarray(reply["actions"][0], dtype=np.float64)[:7].copy()
        action[6] = -np.sign(2.0 * action[6] - 1.0)
        return [action]


# Chunk rows executed per call follow each checkpoint's own LIBERO evaluation.
POLICIES = {
    "gr00t_n17":
    lambda server, rows: Gr00tAdapter(server, preprocess=True, rows=rows or 8),
    "gr00t_n16":
    lambda server, rows: Gr00tAdapter(server, preprocess=False, rows=rows or 8
                                      ),
    "gr00t_n15":
    lambda server, rows: Gr00tAdapter(server, preprocess=False, rows=rows or 1
                                      ),
    "xvla":
    lambda server, rows: XvlaAdapter(server, rows=rows or 30),
    "pi05":
    lambda server, rows: Pi05Adapter(server, rows=rows or 5),
    "smolvla":
    lambda server, rows: SmolvlaAdapter(server, rows=rows or 1),
    "turbovla":
    lambda server, rows: TurbovlaAdapter(server, rows=rows or 12),
    "openvla":
    lambda server, rows: OpenvlaAdapter(server, rows=1),
}


def parse_range(text, count):
    if text == "all":
        return list(range(count))
    ids = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        ids.extend(range(int(lo), int(hi or lo) + 1))
    return ids


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, choices=sorted(POLICIES))
    parser.add_argument(
        "--rows",
        type=int,
        default=0,
        help="chunk rows executed per call (default: per policy)")
    parser.add_argument("--server-cmd",
                        help="spawn the policy server over stdio")
    parser.add_argument("--port",
                        type=int,
                        help="or connect to one listening on TCP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--server-env",
                        action="append",
                        default=[],
                        help="KEY=VALUE for the server only")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", default="all", help="e.g. all, 0-9, 0,3,5")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution",
                        type=int,
                        default=256,
                        help="camera render size (LeRobot's env uses 360)")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--bench-calls",
        type=int,
        default=0,
        help=
        "time this many policy calls on task 0's first observation instead of evaluating"
    )
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    results = {
        "policy": args.policy,
        "suite": args.suite,
        "max_steps": args.max_steps,
        "episodes": {}
    }
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.port:
        from vla_policy_client import VlaPolicyClient
        server = VlaPolicyClient(host=args.host, port=args.port)
    else:
        server = JsonServer(args.server_cmd,
                            dict(kv.split("=", 1) for kv in args.server_env))
    adapter = POLICIES[args.policy](server, args.rows)
    print("server:", server.info, flush=True)

    if args.bench_calls:
        task = suite.get_task(0)
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(
            get_libero_path("bddl_files"), task.problem_folder,
            task.bddl_file),
                                 camera_heights=args.resolution,
                                 camera_widths=args.resolution)
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(NO_OP)
        obs["ee_ori_mat"] = env.robots[0].controller.ee_ori_mat
        times = []
        for _ in range(args.bench_calls + 3):
            t0 = time.perf_counter()
            adapter.act(obs, task.language)
            times.append(1000 * (time.perf_counter() - t0))
        result = {
            "policy": args.policy,
            "calls": args.bench_calls,
            "policy_ms_median": float(np.median(times[3:]))
        }
        json.dump(result, open(args.out, "w"), indent=1)
        print(
            f"{args.policy}: policy call median {result['policy_ms_median']:.1f} ms over {args.bench_calls} calls",
            flush=True)
        env.close()
        server.close()
        return

    for task_id in parse_range(args.tasks, suite.get_num_tasks()):
        task = suite.get_task(task_id)
        pending = [
            e for e in range(args.episodes)
            if f"{task_id}/{e}" not in results["episodes"]
        ]
        if not pending:
            continue
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(
            get_libero_path("bddl_files"), task.problem_folder,
            task.bddl_file),
                                 camera_heights=args.resolution,
                                 camera_widths=args.resolution)
        env.seed(args.seed)
        init_states = suite.get_task_init_states(task_id)
        for episode in pending:
            env.reset()
            obs = env.set_init_state(init_states[episode])
            for _ in range(SETTLE_STEPS):
                obs, _, _, _ = env.step(NO_OP)
            for robot in env.robots:
                robot.controller.use_delta = adapter.control_mode == "relative"
            adapter.reset()
            queue, calls, success, step = [], [], False, 0
            for step in range(args.max_steps):
                if not queue:
                    obs["ee_ori_mat"] = env.robots[0].controller.ee_ori_mat
                    t0 = time.perf_counter()
                    queue = list(adapter.act(obs, task.language))
                    calls.append(time.perf_counter() - t0)
                obs, _, done, _ = env.step(queue.pop(0))
                if done:
                    success = True
                    break
            results["episodes"][f"{task_id}/{episode}"] = {
                "success": success,
                "steps": step + 1,
                "policy_ms": 1000 * float(np.median(calls)),
            }
            done_eps = list(results["episodes"].values())
            rate = sum(e["success"] for e in done_eps) / len(done_eps)
            print(
                f"task {task_id} ep {episode}: {'success' if success else 'fail'} in {step + 1} steps, "
                f"policy {1000 * np.median(calls):.0f} ms | running {rate:.1%} of {len(done_eps)}",
                flush=True)
            json.dump(results, open(args.out, "w"), indent=1)
        env.close()

    server.close()
    done_eps = list(results["episodes"].values())
    per_task = {}
    for key, e in results["episodes"].items():
        per_task.setdefault(key.split("/")[0], []).append(e["success"])
    results["success_rate"] = sum(e["success"]
                                  for e in done_eps) / len(done_eps)
    results["per_task"] = {
        k: sum(v) / len(v)
        for k, v in sorted(per_task.items(), key=lambda kv: int(kv[0]))
    }
    results["policy_ms_median"] = float(
        np.median([e["policy_ms"] for e in done_eps]))
    json.dump(results, open(args.out, "w"), indent=1)
    print(
        f"{args.suite}: {results['success_rate']:.1%} over {len(done_eps)} episodes, policy call median "
        f"{results['policy_ms_median']:.0f} ms",
        flush=True)


if __name__ == "__main__":
    main()
