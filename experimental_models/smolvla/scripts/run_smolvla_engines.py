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
"""Run the SmolVLA engines on a lerobot_reference.py capture and score them against LeRobot.

Feeds the model inputs LeRobot built (preprocessed images, token ids, normalized state, x_0)
through visual -> prefix -> denoise x num_steps and compares the normalized chunk; with
--iters it also times each stage. Needs TensorRT's Python bindings and CUDA PyTorch.

    python run_smolvla_engines.py --engines <dir with *.engine and config.json> --reference ref.npz
"""

import argparse
import json
import os
import time

import numpy as np


class Engine:

    def __init__(self, path, trt, logger):
        import torch
        self.torch = torch
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(
            open(path, "rb").read())
        self.context = self.engine.create_execution_context()
        self.trt = trt

    def __call__(self, inputs, stream):
        trt, torch = self.trt, self.torch
        keep = []
        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            keep.append(tensor)
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                dtype = {
                    trt.float16: torch.float16,
                    trt.float32: torch.float32
                }[self.engine.get_tensor_dtype(name)]
                outputs[name] = torch.empty(tuple(
                    self.context.get_tensor_shape(name)),
                                            dtype=dtype,
                                            device="cuda")
                self.context.set_tensor_address(name, outputs[name].data_ptr())
        assert self.context.execute_async_v3(stream.cuda_stream)
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--iters", type=int, default=0)
    args = parser.parse_args()

    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.ERROR)
    config = json.load(open(os.path.join(args.engines, "config.json")))
    visual, prefix, denoise = (Engine(
        os.path.join(args.engines, f"{n}.engine"), trt, logger)
                               for n in ("visual", "prefix", "denoise"))
    ref = np.load(args.reference)
    images = torch.from_numpy(ref["images"][:, 0]).half().cuda()
    tokens = torch.from_numpy(ref["lang_tokens"][ref["lang_masks"].astype(
        bool)][None]).long().cuda()
    state = torch.from_numpy(ref["state"]).float().cuda()
    noise = torch.from_numpy(ref["noise"]).float().cuda()
    steps = config["num_steps"]
    stream = torch.cuda.Stream()

    def run():
        times = {}
        with torch.cuda.stream(stream):
            t0 = time.perf_counter()
            features = visual({"pixel_values": images},
                              stream)["image_features"]
            features = features.reshape(1, -1, features.shape[-1])
            stream.synchronize()
            times["visual"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            kv = prefix(
                {
                    "image_features": features,
                    "token_ids": tokens,
                    "state": state
                }, stream)
            stream.synchronize()
            times["prefix"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            x = noise
            dt = torch.tensor(-1.0 / steps, device="cuda")
            for step in range(steps):
                t = torch.full((1, ), 1.0 + step * float(dt), device="cuda")
                x = denoise(
                    {
                        "x_t": x,
                        "timestep": t,
                        "dt": dt,
                        **{
                            n: kv[n]
                            for n in config["kv_names"]
                        }
                    }, stream)["x_next"]
            stream.synchronize()
            times["denoise"] = time.perf_counter() - t0
        return x, times

    x, _ = run()
    ours = x.float().cpu().numpy().astype(np.float64)
    theirs = ref["normalized"].astype(np.float64)
    cos = float(
        (ours * theirs).sum() / np.linalg.norm(ours) / np.linalg.norm(theirs))
    print(
        f"normalized chunk {ours.shape}: cosine {cos:.6f}, max |d| {np.abs(ours - theirs).max():.4f}, "
        f"mean |d| {np.abs(ours - theirs).mean():.4f}")
    if args.iters:
        samples = [run()[1] for _ in range(args.iters)]
        print(
            "median ms: " +
            ", ".join(f"{k} {1000 * np.median([s[k] for s in samples]):.1f}"
                      for k in samples[0]) +
            f", total {1000 * np.median([sum(s.values()) for s in samples]):.1f}"
        )


if __name__ == "__main__":
    main()
