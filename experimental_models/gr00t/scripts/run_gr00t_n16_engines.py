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
"""Run the GR00T N1.6 engines on the inputs ``official_reference.py`` captured, stage by stage.

The Eagle visual / prefix engines get the official pixel values and token ids; the action engines get
the resulting features (or, with --official-features, the official ones), the official normalized state
and the same x_0. Prints the backbone feature and normalized action agreement.

    python run_gr00t_n16_engines.py --engines engines --reference ref_f300.npz
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
            trt.int32: torch.int32,
            trt.bool: torch.bool
        }
        keep = []
        for name, tensor in inputs.items():
            tensor = tensor.to(
                dtypes[self.engine.get_tensor_dtype(name)]).contiguous()
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


def cosine(a, b):
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--official-features", action="store_true")
    args = parser.parse_args()

    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.ERROR)
    load = lambda name: Engine(os.path.join(args.engines, name), trt, logger)
    action = json.load(
        open(os.path.join(args.engines, "action", "config.json")))
    ref = np.load(args.reference)
    stream = torch.cuda.Stream()

    pixels = np.concatenate([
        ref[k] for k in sorted(f for f in ref.files
                               if f.startswith("backbone_input.pixel_values."))
    ])
    tokens = torch.from_numpy(ref["input_ids"])[None].cuda()
    image = load("backbone/visual.engine")(
        {
            "pixel_values": torch.from_numpy(pixels).cuda()
        }, stream)["image_features"]
    features = load("backbone/prefix.engine")(
        {
            "token_ids": tokens,
            "image_features": image.reshape(1, -1, image.shape[-1])
        }, stream)["backbone_features"].float()
    expected, mask = ref["backbone_features"], ref["image_mask"]
    got = features[0].cpu().numpy()
    per_token = np.sum(got * expected, 1) / (np.linalg.norm(got, axis=1) *
                                             np.linalg.norm(expected, axis=1))
    print(
        f"backbone features vs official: cosine {cosine(got, expected):.6f}, per-token min {per_token.min():.5f}, "
        f"image {cosine(got[mask], expected[mask]):.6f}, text {cosine(got[~mask], expected[~mask]):.6f}"
    )

    if args.official_features:
        features = torch.from_numpy(expected)[None].cuda()
    image_mask = torch.from_numpy(mask)[None].cuda()
    prep = load("action/vl_prep.engine")(
        {
            "backbone_features": features,
            "image_mask": image_mask,
            "attention_mask": torch.ones_like(image_mask)
        }, stream)
    state = load("action/state_encoder.engine")(
        {
            "state": torch.from_numpy(ref["state"])[None].cuda()
        }, stream)["state_features"]
    denoise = load("action/denoise_step.engine")
    actions = torch.from_numpy(ref["noise"]).cuda()
    steps, buckets = action["num_inference_timesteps"], action[
        "num_timestep_buckets"]
    for step in range(steps):
        out = denoise(
            {
                "actions": actions,
                "timestep": torch.tensor([int(step / steps * buckets)]).cuda(),
                "state_features": state,
                "cross_keys": prep["cross_keys"],
                "cross_values": prep["cross_values"],
                "text_bias": prep["text_bias"],
                "image_bias": prep["image_bias"],
                "vel_strength": torch.ones_like(actions),
                "dt": torch.tensor(1.0 / steps).cuda()
            }, stream)
        actions = next(iter(out.values())).float()
    pred, expected_pred = actions[0].cpu().numpy(), ref["action_pred"]
    rows = int(
        json.load(open(os.path.join(args.engines, "action",
                                    "processing.json")))["action_horizon"])
    dims = sum(a["dim"] for a in json.load(
        open(os.path.join(args.engines, "action", "processing.json")))
               ["action"])
    used, used_ref = pred[:rows, :dims], expected_pred[:rows, :dims]
    print(
        f"normalized actions ({'official' if args.official_features else 'engine'} features) vs official: "
        f"cosine {cosine(used, used_ref):.6f}, max|d| {np.abs(used - used_ref).max():.4f} over the {rows}x{dims} "
        f"rows/dims the robot uses")


if __name__ == "__main__":
    main()
