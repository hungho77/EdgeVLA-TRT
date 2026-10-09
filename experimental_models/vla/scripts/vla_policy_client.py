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
"""Robot-side client for the EdgeVLA-TRT policy servers (GR00T, pi0.5, SmolVLA, X-VLA, TurboVLA, OpenVLA).

Camera frames travel inline as raw RGB behind the request header (see experimental_models/vla/cpp/vlaServer.h), so
the robot never writes image files. The client either starts the server as a subprocess (stdin / stdout) or connects
to one already listening on TCP (``--port`` on the server):

    client = VlaPolicyClient(server_cmd=["xvla_policy_server", "--engineDir", "engines"])   # or host=..., port=...
    print(client.info)                       # camera names, state / action widths, chunk size, RTC scheme
    actions = client.act({"image": rgb_top, "image2": rgb_wrist}, state, "pick up the cube", reset=True)

Frames are HxWx3 uint8 **RGB** numpy arrays (OpenCV images are BGR: convert them). ``act`` returns the server's
action chunk as a float32 array [rows, action_dim] in the checkpoint's own action space; what that means per family
is documented in experimental_models/vla/README.md. AsyncChunkedController runs a fixed-rate control loop on top.
"""

import json
import os
import socket
import subprocess

import numpy as np


class PolicyError(RuntimeError):
    """The server rejected a request."""


class VlaPolicyClient:

    def __init__(self, server_cmd=None, host=None, port=None, env=None):
        if (server_cmd is None) == (port is None):
            raise ValueError(
                "give either server_cmd (spawn over stdio) or port (connect over TCP)"
            )
        self._proc = self._sock = None
        if server_cmd is not None:
            server_env = dict(os.environ if env is None else env)
            server_env.pop("PYTHONPATH", None)
            self._proc = subprocess.Popen(server_cmd,
                                          stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE,
                                          env=server_env)
            self._reader, self._writer = self._proc.stdout, self._proc.stdin
        else:
            self._sock = socket.create_connection((host or "127.0.0.1", port))
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._reader = self._sock.makefile("rb")
            self._writer = self._sock.makefile("wb")
        self.info = self._read()
        self.cameras = list(self.info.get("cameras", []))

    def _read(self):
        while True:
            line = self._reader.readline()
            if not line:
                raise PolicyError("policy server closed the connection")
            # Over stdio the runtime's own log lines share stdout; protocol replies are the JSON objects.
            if line.startswith(b"{"):
                reply = json.loads(line)
                if "error" in reply:
                    raise PolicyError(reply["error"])
                return reply

    def request(self, header, frames=None):
        """Send one request. ``frames``: {name: HxWx3 uint8 RGB}; sent inline in dict order."""
        header = dict(header)
        payload = []
        if frames:
            if self.cameras:
                unknown = [name for name in frames if name not in self.cameras]
                if unknown:
                    raise ValueError(
                        f"cameras {unknown} not among the server's {self.cameras}"
                    )
            header["frames"] = []
            for name, frame in frames.items():
                frame = np.ascontiguousarray(frame)
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[
                        2] != 3:
                    raise ValueError(
                        f"frame {name!r} must be HxWx3 uint8 RGB, got {frame.dtype} {frame.shape}"
                    )
                header["frames"].append({
                    "name": name,
                    "height": int(frame.shape[0]),
                    "width": int(frame.shape[1]),
                    "bytes": int(frame.nbytes)
                })
                payload.append(frame.tobytes())
        self._writer.write(json.dumps(header).encode() + b"\n")
        for chunk in payload:
            self._writer.write(chunk)
        self._writer.flush()
        return self._read()

    def act(self, frames, state, task, **fields):
        """One action chunk [rows, action_dim]. ``fields`` pass through (reset, rtc, seed, noise_file, ...)."""
        reply = self.request(
            {
                "state": [float(v) for v in np.asarray(state).ravel()],
                "task": task,
                **fields
            }, frames)
        self.last_reply = reply
        return np.asarray(reply["actions"], dtype=np.float32)

    def close(self):
        try:
            self._writer.write(b"\n")
            self._writer.flush()
        except (BrokenPipeError, OSError):
            pass
        if self._proc is not None:
            self._proc.wait(timeout=30)
        if self._sock is not None:
            self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class AsyncChunkedController:
    """Fixed-rate control on top of a policy server, as experimental_models/vla/cpp/vlaAsyncChunker.h.

    The control loop calls ``step(tick)`` once per tick and executes the returned row; the next chunk is planned on a
    background thread meanwhile. Chunk row r of a chunk planned at tick t is the action for tick t + r. Once
    ``chunk - overlap`` rows of the current chunk were issued, ``step`` snapshots the observation (on the control
    thread, via ``snapshot(tick) -> (frames, state, task)``) and requests the next chunk for that tick; with real-time
    chunking the server inpaints it from the current chunk starting at row ``tick - planned tick``. When the new chunk
    lands the loop continues in it at row ``tick - planned tick``, skipping the rows the planner latency consumed.
    ``frozen`` rows (reproduced exactly) must cover the worst-case latency in ticks, or switches jump.

    ``step`` returns None when no chunk covers the tick (before the first lands, or when the planner fell behind):
    hold the robot, never extrapolate.
    """

    def __init__(self,
                 client,
                 snapshot,
                 overlap,
                 frozen,
                 rtc=True,
                 ramp_rate=6.0):
        import threading
        self.client = client
        self.snapshot = snapshot
        self.rows = int(client.info["chunk"])
        self.overlap = min(int(overlap), self.rows - 1) if self.rows > 1 else 0
        self.frozen = int(frozen)
        self.scheme = client.info.get("rtc") if rtc else None
        self.ramp_rate = ramp_rate
        self.replan_after = self.rows - self.overlap if self.rows > 1 else 0
        self._lock = threading.Condition()
        self._request = None  # (tick, observation) waiting for the planner
        self._landed = None  # (planned tick, actions)
        self._stop = False
        self._error = None
        self.current_tick, self.current = -1, None
        self.requested_for = -1
        self.last_lag = 0
        self.chunks = 0
        self.latencies_ms = []
        self._last_plan_tick = -1
        self._thread = threading.Thread(target=self._plan_loop, daemon=True)
        self._thread.start()

    def _rtc(self, tick):
        if self.scheme is None or self._last_plan_tick < 0:
            return None
        start_row = int(tick - self._last_plan_tick)
        if self.scheme == "overlap_frozen":
            return {
                "overlap": self.overlap,
                "frozen": self.frozen,
                "ramp_rate": self.ramp_rate,
                "start_row": start_row
            }
        if self.scheme == "delay_horizon":
            return {
                "delay": self.frozen,
                "horizon": self.overlap,
                "start_row": start_row
            }
        return None

    def _plan_loop(self):
        import time
        while True:
            with self._lock:
                while self._request is None and not self._stop:
                    self._lock.wait()
                if self._stop:
                    return
                tick, (frames, state, task) = self._request
            try:
                fields = {"reset": self._last_plan_tick < 0}
                rtc = self._rtc(tick)
                if rtc is not None:
                    fields["rtc"] = rtc
                t0 = time.perf_counter()
                actions = self.client.act(frames, state, task, **fields)
                self.latencies_ms.append(1000 * (time.perf_counter() - t0))
            except Exception as error:  # noqa: BLE001 -- surfaced on the control thread
                with self._lock:
                    self._error, self._request = error, None
                return
            self._last_plan_tick = tick
            with self._lock:
                self._landed, self._request = (tick, actions), None
                self.chunks += 1

    def warmup(self, observation):
        """Run the server's first plain and RTC calls (engine warm-up, CUDA graph captures) before the control loop,
        so the first real chunk is not stale on arrival. The first real request still resets the episode."""
        frames, state, task = observation
        self.client.act(frames, state, task, reset=True)
        if self.scheme is not None:
            self._last_plan_tick = 0
            self.client.act(frames,
                            state,
                            task,
                            rtc=self._rtc(self.replan_after))
            self._last_plan_tick = -1

    def step(self, tick):
        with self._lock:
            if self._error is not None:
                raise self._error
            landed = self._landed
        if landed is not None and landed[0] != self.current_tick:
            self.current_tick, self.current = landed
            self.last_lag = tick - self.current_tick
        row = tick - self.current_tick if self.current is not None else -1
        # One request in flight at a time: the next one continues the chunk that just landed.
        if (row < 0 or row >= self.replan_after
            ) and self.requested_for <= self.current_tick:
            observation = self.snapshot(tick)
            with self._lock:
                self._request = (tick, observation)
                self._lock.notify()
            self.requested_for = tick
        if row < 0 or row >= len(self.current if self.current is not None else
                                 ()):
            return None
        return self.current[row]

    def close(self):
        with self._lock:
            self._stop = True
            self._lock.notify()
        self._thread.join(timeout=60)
