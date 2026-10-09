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
"""Run the TurboVLA engines on the inputs ``turbovla_reference.py`` captured, stage by stage: the text engine, the
policy engine on the official text tokens, then the two chained.

    python run_turbovla_engines.py --engines engines --reference ref.npz --obs obs.npz
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_turbovla import text_inputs  # noqa: E402
from turbovla_reference import policy_rotate  # noqa: E402


class Engine:

    def __init__(self, path, trt, logger):
        import torch
        self.torch, self.trt = torch, trt
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(
            open(path, "rb").read())
        self.context = self.engine.create_execution_context()

    def __call__(self, inputs, stream):
        trt, torch = self.trt, self.torch
        dtypes = {
            trt.float16: torch.float16,
            trt.float32: torch.float32,
            trt.int64: torch.int64,
            trt.int32: torch.int32
        }
        keep = []
        names = {
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        }
        for name, tensor in inputs.items():
            if name not in names:
                continue
            tensor = tensor.to(dtypes[self.engine.get_tensor_dtype(
                name)]).cuda().contiguous()
            keep.append(tensor)
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                outputs[name] = torch.empty(
                    tuple(self.context.get_tensor_shape(name)),
                    dtype=dtypes[self.engine.get_tensor_dtype(name)],
                    device="cuda")
                self.context.set_tensor_address(name, outputs[name].data_ptr())
        assert self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return {k: v.float().cpu().numpy() for k, v in outputs.items()}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--obs", required=True)
    args = parser.parse_args()

    import tensorrt as trt
    import torch
    config = json.load(open(os.path.join(args.engines, "config.json")))
    logger = trt.Logger(trt.Logger.WARNING)
    text = Engine(os.path.join(args.engines, "text.engine"), trt, logger)
    policy = Engine(os.path.join(args.engines, "policy.engine"), trt, logger)
    stream = torch.cuda.Stream()
    ref, obs = np.load(args.reference), np.load(args.obs)
    mean = np.asarray(config["image_mean"], np.float32)
    std = np.asarray(config["image_std"], np.float32)
    state_mean = np.asarray(config["state_mean"], np.float32)
    state_std = np.asarray(config["state_std"], np.float32)
    length = config["text_length"]

    count = len([k for k in ref.files if k.startswith("normalized_")])
    for i in range(count):
        ids, positions, self_attention, hidden_valid, attention = text_inputs(
            ref, i, length, torch.float16)
        tokens = text(
            {
                "input_ids": ids,
                "position_ids": positions,
                "self_attention": self_attention,
                "hidden_valid": hidden_valid,
                "attention": attention
            }, stream)["text_tokens"]
        views = [
            policy_rotate(obs[f"{camera}_{i}"])
            for camera in ("agentview", "wrist")
        ]
        pixels = np.stack([
            ((v.astype(np.float32) * np.float32(1 / 255.0)) - mean) / std
            for v in views
        ]).transpose(0, 3, 1, 2)
        state = (obs[f"state_{i}"] - state_mean) / (state_std + 1e-6)
        common = {
            "pixels": torch.from_numpy(np.ascontiguousarray(pixels[None])),
            "attention": attention,
            "self_attention": self_attention,
            "state": torch.from_numpy(state[None].astype(np.float32))
        }
        on_ref = policy(
            {
                **common, "text_tokens": torch.from_numpy(ref[f"text_{i}"])
            }, stream)["actions"][0]
        chained = policy({
            **common, "text_tokens": torch.from_numpy(tokens)
        }, stream)["actions"][0]
        expected = ref[f"normalized_{i}"]
        print(
            f"sample {i}: text tokens max|d| {np.abs(tokens - ref[f'text_{i}']).max():.3e}; actions max|d| "
            f"policy on official text {np.abs(on_ref - expected).max():.3e}, chained "
            f"{np.abs(chained - expected).max():.3e}")


if __name__ == "__main__":
    main()
