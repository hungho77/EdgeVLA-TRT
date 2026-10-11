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
"""Run the Evo-1 engines on the inputs ``evo1_reference.py`` captured, each on the official tensors and then chained,
and compare with the official outputs (pixel values, token ids, attention mask, normalized state and noise in;
vision features, hidden states and the normalized chunk out).

    python run_evo1_engines.py --engines engines --reference ref_fp32.npz [--baseline ref_bf16.npz]

``--baseline`` prints how far the official policy at another precision lands from the same reference.
"""

import argparse
import json
import os

import numpy as np


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
        for name, tensor in inputs.items():
            tensor = torch.as_tensor(tensor).to(dtypes[
                self.engine.get_tensor_dtype(name)]).cuda().contiguous()
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
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()

    import tensorrt as trt
    import torch
    config = json.load(open(os.path.join(args.engines, "config.json")))
    logger = trt.Logger(trt.Logger.WARNING)
    vision, prefix, context, step = (Engine(
        os.path.join(args.engines, f"{name}.engine"), trt,
        logger) for name in ("vision", "prefix", "context", "step"))
    stream = torch.cuda.Stream()
    ref = np.load(args.reference)
    base = np.load(args.baseline) if args.baseline else None
    dims = config["max_dim"]
    mask = torch.tensor([1.0] * config["action_dim"] + [0.0] *
                        (dims - config["action_dim"]))

    def chunk(kv, noise):
        action = torch.as_tensor(noise).cuda().float()
        strength = torch.ones(config["chunk_size"], 1)
        dt = torch.tensor([1.0 / config["steps"]])
        for index in config["time_indices"]:
            action = step(
                {
                    "action": action,
                    "time_index": torch.tensor([index], dtype=torch.int32),
                    "action_mask": mask,
                    "strength": strength,
                    "dt": dt,
                    **kv
                }, stream)["action_next"]
        return (action.reshape(-1, dims).cpu() * mask).numpy()

    err = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max())
    count = len([k for k in ref.files if k.startswith("normalized_")])
    for i in range(count):
        r = lambda k: ref[f"{k}_{i}"]
        visual = vision({"pixels": r("pixels")},
                        stream)["image_tokens"].float().cpu()
        prefix_in = {
            "input_ids": r("input_ids"),
            "attention": r("attention_mask")
        }
        hidden_alone = prefix(
            {
                **prefix_in, "image_tokens":
                r("visual").reshape(-1, visual.shape[-1])
            }, stream)["hidden"]
        kv = context({
            "hidden": r("hidden"),
            "state": r("state")[None]
        }, stream)
        alone = chunk(kv, r("noise"))
        hidden = prefix(
            {
                **prefix_in, "image_tokens": visual.reshape(
                    -1, visual.shape[-1])
            }, stream)["hidden"]
        kv = context({"hidden": hidden, "state": r("state")[None]}, stream)
        chained = chunk(kv, r("noise"))
        action_dim = config["action_dim"]
        low = np.asarray(config["action_min"], np.float32)
        high = np.asarray(config["action_max"], np.float32)
        robot = (chained[:, :action_dim] + 1) / 2 * (high - low + 1e-8) + low
        print(
            f"{i}: vision {err(visual, r('visual')):.4f} (|ref| {np.abs(r('visual')).max():.1f})  "
            f"prefix {err(hidden_alone.float().cpu(), r('hidden')):.4f} (|ref| {np.abs(r('hidden')).max():.1f})  "
            f"head {err(alone, r('normalized')):.4f}\n"
            f"   chained: normalized {err(chained, r('normalized')):.4f}, robot {err(robot, r('actions')):.4f}, "
            f"cos {np.dot(robot.ravel(), r('actions').ravel()) / np.linalg.norm(robot) / np.linalg.norm(r('actions')):.6f}"
        )
        if base is not None:
            print(
                f"   official {args.baseline}: normalized {err(base[f'normalized_{i}'][:, :action_dim], r('normalized')[:, :action_dim]):.4f}, "
                f"robot {err(base[f'actions_{i}'], r('actions')):.4f}")


if __name__ == "__main__":
    main()
