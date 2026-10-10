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
"""Run MolmoAct2's engines on what ``molmoact2_reference.py`` captured, stage by stage and chained: each stage is fed
the reference's inputs (isolating its error), then the previous engine's output.

    python run_molmoact2_engines.py --engines engines --reference ref.npz
"""

import argparse
import json
import os

import numpy as np
import torch


class Engine:

    def __init__(self, path, trt, logger):
        self.trt = trt
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(
            open(path, "rb").read())
        self.context = self.engine.create_execution_context()

    def __call__(self, inputs, stream):
        trt = self.trt
        dtypes = {
            trt.float16: torch.float16,
            trt.float32: torch.float32,
            trt.int64: torch.int64,
            trt.int32: torch.int32
        }
        keep = []
        for name, tensor in inputs.items():
            tensor = torch.as_tensor(tensor).to(
                dtypes[self.engine.get_tensor_dtype(name)]).cuda().contiguous()
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


def rel(expected, actual):
    expected = np.asarray(expected, np.float64)
    actual = np.asarray(actual.float().cpu() if torch.is_tensor(actual) else
                        actual, np.float64).reshape(expected.shape)
    return float(np.sqrt(((expected - actual)**2).mean() /
                         (expected**2).mean()))


def rope(length, head_dim, theta):
    inv = 1.0 / (theta**(np.arange(0, head_dim, 2, dtype=np.float32) /
                         np.float32(head_dim)))
    freqs = np.outer(np.arange(length, dtype=np.float32),
                     inv.astype(np.float32))
    emb = np.concatenate([freqs, freqs], -1)
    return torch.from_numpy(np.cos(emb)), torch.from_numpy(np.sin(emb))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    engines = {
        name: Engine(os.path.join(args.engines, f"{name}.engine"), trt, logger)
        for name in ("vision", "prefix_a", "prefix_b", "context", "step")
    }
    config = json.load(open(os.path.join(args.engines, "config.json")))
    stream = torch.cuda.Stream()
    ref = np.load(args.reference)
    count = len([k for k in ref.files if k.startswith("normalized_")])
    for i in range(count):
        visual = engines["vision"]({
            "patches": torch.from_numpy(ref[f"pixel_values_{i}"])
        }, stream)["visual"]
        ids = torch.from_numpy(ref[f"input_ids_{i}"][0])
        flag = torch.from_numpy(ref[f"token_type_ids_{i}"][0]).float()
        cos, sin = rope(ids.shape[0], config["head_dim"], config["rope_theta"])

        def prefix(features):
            a = engines["prefix_a"](
                {
                    "input_ids": ids,
                    "visual": features,
                    "image_flag": flag,
                    "cos": cos,
                    "sin": sin
                }, stream)
            b = engines["prefix_b"](
                {
                    "hidden": a["hidden"],
                    "image_flag": flag,
                    "cos": cos,
                    "sin": sin
                }, stream)
            return torch.cat([a["keys"], b["keys"]]), torch.cat(
                [a["values"], b["values"]])

        keys, values = prefix(visual)
        ref_keys, ref_values = ref[f"k_{i}"][:, 0], ref[f"v_{i}"][:, 0]
        ref_visual_keys, _ = prefix(torch.from_numpy(ref[f"visual_{i}"]))
        print(f"sample {i}: visual rel {rel(ref[f'visual_{i}'], visual):.2e}; K rel on reference visual "
              f"{rel(ref_keys, ref_visual_keys):.2e}, chained K {rel(ref_keys, keys):.2e} V {rel(ref_values, values):.2e}")

        def denoise(k, v):
            ctx = engines["context"]({"keys": k, "values": v}, stream)
            x = torch.from_numpy(ref[f"noise_{i}"])
            for step in range(config["flow_steps"]):
                x = engines["step"](
                    {
                        "x": x,
                        "step": torch.tensor([step]),
                        "dt": torch.tensor([1.0 / config["flow_steps"]]),
                        "context_k": ctx["context_k"],
                        "context_v": ctx["context_v"],
                        "encoder_mask": torch.from_numpy(
                            ref[f"encoder_mask_{i}"]).float(),
                        "strength": torch.ones(1, x.shape[1], 1)
                    }, stream)["x_next"]
            return x[0, :, :config["action_dim"]].float().cpu().numpy()

        expected = ref[f"normalized_{i}"]
        on_ref = denoise(torch.from_numpy(ref_keys), torch.from_numpy(ref_values))
        chained = denoise(keys, values)
        print(f"   normalized actions max|d|: context + step on reference K / V {np.abs(on_ref - expected).max():.4f}, "
              f"chained {np.abs(chained - expected).max():.4f} (|ref| max {np.abs(expected).max():.2f})")


if __name__ == "__main__":
    main()
