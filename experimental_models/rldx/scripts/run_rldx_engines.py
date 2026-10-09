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
"""Run RLDX-1's llm_a / llm_b / action engines on what ``rldx_reference.py`` captured, stage by stage and chained.

Each stage is fed the reference's own inputs first (isolating its error), then the previous engine's output. The
visual engine is checked by rldx_vision_dump; pass its dumps with --visual-dumps to start the chain from them.

    PYTHONPATH=RLDX-1 python run_rldx_engines.py --engines engines --vlm RLDX-1-VLM --reference ref.npz \\
        --state state.npz [--visual-dumps dump_prefix]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rldx_llm  # noqa: E402


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


def rel(expected, actual):
    expected, actual = np.asarray(expected,
                                  np.float64), np.asarray(actual, np.float64)
    return float(
        np.sqrt(((expected - actual)**2).mean() / (expected**2).mean()))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", required=True)
    parser.add_argument("--vlm", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--state",
        required=True,
        help="normalized, padded states state_<i> [64] per sample")
    parser.add_argument("--visual-dumps")
    args = parser.parse_args()

    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    engines = {
        name: Engine(os.path.join(args.engines, f"{name}.engine"), trt, logger)
        for name in ("llm_a", "llm_b", "action")
    }
    stream = torch.cuda.Stream()
    ref, states = np.load(args.reference), np.load(args.state)
    config = rldx_llm.text_config(args.vlm)
    count = len([k for k in ref.files if k.startswith("cognition_")])
    for i in range(count):
        ids = torch.from_numpy(ref[f"input_ids_{i}"])
        positions = rldx_llm.position_ids(
            args.vlm, ids, torch.from_numpy(ref[f"image_grid_thw_{i}"]))
        cos, sin = rldx_llm.rope_tables(config, positions)
        visual, deepstack = ref[f"visual_{i}"], ref[f"deepstack_{i}"]
        if args.visual_dumps:
            prefix = f"{args.visual_dumps}_s{i}"
            visual = np.fromfile(f"{prefix}_visual.bin",
                                 np.float32).reshape(visual.shape)
            deepstack = np.stack([
                np.fromfile(f"{prefix}_deepstack{d}.bin",
                            np.float32).reshape(visual.shape)
                for d in range(deepstack.shape[0])
            ])
        hidden = engines["llm_a"]({
            "input_ids": ids[0],
            "visual": visual,
            "deepstack": deepstack,
            "visual_index": rldx_llm.visual_index(ids),
            "cos": cos,
            "sin": sin
        }, stream)["hidden"]
        pool, keep, rope = rldx_llm.compression(ids)
        cos_b, sin_b = rldx_llm.rope_tables(config, positions[:, :, rope])
        llm_b = lambda h: engines["llm_b"]({
            "hidden": h,
            "pool": pool,
            "keep_index": keep,
            "cos": cos_b,
            "sin": sin_b
        }, stream)["cognition"]
        cognition = llm_b(hidden)
        on_ref = llm_b(torch.from_numpy(ref[f"layer3_{i}"]))
        print(
            f"sample {i}: llm_a layer-3 hidden rel {rel(ref[f'layer3_{i}'], hidden.float().cpu()):.2e}; "
            f"cognition rel: llm_b on reference input {rel(ref[f'cognition_{i}'], on_ref.float().cpu()):.2e}, "
            f"chained {rel(ref[f'cognition_{i}'], cognition.float().cpu()):.2e}"
        )

        state = torch.from_numpy(states[f"state_{i}"])[None, None]
        ones = torch.ones(1, 16, 1)

        def denoise(features):
            x = torch.from_numpy(ref[f"noise_{i}"])
            for k in range(4):
                x = engines["action"]({
                    "x": x,
                    "t": torch.tensor([k / 4.0]),
                    "dt": torch.tensor([0.25]),
                    "cognition": features,
                    "state": state,
                    "strength": ones
                }, stream)["x_next"].float().cpu()
            return x[0, :, :7].numpy()

        expected = (ref[f"step_input_{i}"][3] +
                    0.25 * ref[f"step_velocity_{i}"][3])[0, :, :7]
        print(
            f"   normalized actions max|d|: action engine on reference cognition "
            f"{np.abs(denoise(torch.from_numpy(ref[f'cognition_{i}'])) - expected).max():.4f}, "
            f"chained from {'visual engine' if args.visual_dumps else 'reference visual'} "
            f"{np.abs(denoise(cognition) - expected).max():.4f}")


if __name__ == "__main__":
    main()
