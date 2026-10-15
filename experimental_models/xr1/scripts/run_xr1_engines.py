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
"""Run the Xiaomi-Robotics-1 engines on the inputs ``xr1_reference.py --capture`` saved, each on the official tensors
and then chained, and compare with the official outputs (vision tokens and deepstack features, prompt K / V of layers
0, 17 and 35, the normalized chunk).

    python run_xr1_engines.py --engines engines --reference capture.npz [--baseline ref_bf16.npz]
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
    engines = {
        name: Engine(os.path.join(args.engines, f"{name}.engine"), trt, logger)
        for name in ("vision", "prefix_a", "prefix_b", "step")
    }
    stream = torch.cuda.Stream()
    ref = np.load(args.reference)
    base = np.load(args.baseline) if args.baseline else None
    err = lambda a, b: float(
        np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())
    count = len([k for k in ref.files if k.startswith("normalized_")])

    def prefix(r, tokens, deepstack):
        positions = r("position_ids")[:, 0]
        a = engines["prefix_a"]({
            "input_ids": r("input_ids"),
            "position_ids": positions,
            "image_tokens": tokens,
            "deepstack": deepstack
        }, stream)
        b = engines["prefix_b"]({
            "hidden": a["hidden_out"],
            "position_ids": positions
        }, stream)
        return (torch.cat([a["keys"],
                           b["keys"]]), torch.cat([a["values"], b["values"]]))

    def chunk(r, keys, values):
        x = torch.as_tensor(r("noise"))[None].float().cuda()
        length = r("input_ids").shape[1]
        common = {
            "state": r("state"),
            "prefix_keys": keys,
            "prefix_values": values,
            "prefix_mask": np.ones((1, length), np.float32),
            "suffix_start": np.array([r("position_ids").max() + 1], np.int32),
            "action_mask": r("action_mask"),
            "dt": np.array([1.0 / config["steps"]], np.float32),
            "noise": r("noise")[None],
            "seed": r("noise")[None],
            "weight": np.zeros((r("noise").shape[0], 1), np.float32)
        }
        for step in range(config["steps"]):
            x = engines["step"]({
                "x": x,
                "step": np.array([step], np.int32),
                **common
            }, stream)["x_next"]
        return x[0].float().cpu().numpy()

    for i in range(count):
        r = lambda k: ref[f"{k}_{i}"]
        vision = engines["vision"]({"pixels": r("pixel_values")}, stream)
        keys, values = prefix(r, r("vision"), r("deepstack"))
        kv = "  ".join(
            f"K{layer} {err(keys[layer, 0].float().cpu().numpy().transpose(1, 0, 2), r(f'k{layer}')[0]):.3f}"
            f" V{layer} {err(values[layer, 0].float().cpu().numpy().transpose(1, 0, 2), r(f'v{layer}')[0]):.3f}"
            for layer in (0, 17, 35))
        print(
            f"{i}: vision {err(vision['image_tokens'].float().cpu(), r('vision')):.4f} deepstack "
            f"{err(vision['deepstack'].float().cpu(), r('deepstack')):.4f} (|ref| {np.abs(r('vision')).max():.1f} / "
            f"{np.abs(r('deepstack')).max():.1f})\n   prefix from official tokens: {kv}"
        )
        alone = chunk(r, keys, values)
        keys_c, values_c = prefix(r, vision["image_tokens"],
                                  vision["deepstack"])
        chained = chunk(r, keys_c, values_c)
        print(
            f"   chunk on engine prefix of official tokens {err(alone, r('normalized')):.4f}; "
            f"chained {err(chained, r('normalized')):.4f}")
        if base is not None:
            print(f"   official {os.path.basename(args.baseline)}: "
                  f"{err(base[f'normalized_{i}'], r('normalized')):.4f}")


if __name__ == "__main__":
    main()
