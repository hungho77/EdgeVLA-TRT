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
"""Run the X-VLA engines on the inputs ``lerobot_xvla_reference.py`` captured, stage by stage.

    python run_xvla_engines.py --engines engines --reference ref.npz
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
        names = {
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        }
        for name, tensor in inputs.items():
            if name not in names:
                continue
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


def max_abs(a, b):
    return float(np.abs(a - b).max())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--reference-features",
        action="store_true",
        help=
        "feed the reference encoder output and view features to the step engine"
    )
    args = parser.parse_args()

    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.ERROR)
    load = lambda name: Engine(os.path.join(args.engines, f"{name}.engine"),
                               trt, logger)
    config = json.load(open(os.path.join(args.engines, "config.json")))
    ref = np.load(args.reference)
    stream = torch.cuda.Stream()

    images, mask = torch.from_numpy(ref["image_input"]), torch.from_numpy(
        ref["image_mask"])
    feats = load("vision")({
        "images": images[mask].cuda()
    }, stream)["image_features"]
    all_feats = feats.new_zeros((images.shape[0], *feats.shape[1:]))
    all_feats[mask.cuda()] = feats
    vlm = load("encoder")(
        {
            "primary_features": all_feats[:1],
            "token_ids": torch.from_numpy(ref["input_ids"])[None].cuda()
        }, stream)["vlm_features"]
    aux = all_feats[1:].reshape(1, -1, all_feats.shape[-1])
    print(
        f"encoder output max|d| {max_abs(vlm[0].float().cpu().numpy(), ref['vlm_features']):.4f} "
        f"(|ref| {np.abs(ref['vlm_features']).max():.1f}), aux features max|d| "
        f"{max_abs(aux[0].float().cpu().numpy(), ref['aux_visual_inputs']):.4f}"
    )

    if args.reference_features:
        vlm = torch.from_numpy(ref["vlm_features"])[None].cuda()
        aux = torch.from_numpy(ref["aux_visual_inputs"])[None].cuda()
    step = load("step")
    x1 = torch.from_numpy(ref["noise"]).cuda()
    action = torch.zeros_like(x1)
    steps = config["num_steps"]
    for i in range(steps, 0, -1):
        action = step(
            {
                "x1": x1,
                "action": action,
                "t": torch.full((1, ), i / steps).cuda(),
                "proprio": torch.from_numpy(ref["proprio"])[None].cuda(),
                "domain_id": torch.tensor([int(ref["domain_id"])]).cuda(),
                "vlm_features": vlm,
                "aux_visual_inputs": aux,
                "rtc_seed": torch.zeros_like(x1),
                "rtc_weight": torch.zeros(1, x1.shape[1], 1).cuda()
            }, stream)["action_next"].float()
    out = action[0].cpu().numpy()
    gripper = config["gripper_idx"]
    out[:, gripper] = 1.0 / (1.0 + np.exp(-out[:, gripper]))
    expected = ref["model_actions"]
    cosine = float(
        np.sum(out * expected) / np.linalg.norm(out) /
        np.linalg.norm(expected))
    print(
        f"actions vs LeRobot: max|d| {max_abs(out, expected):.4f}, cosine {cosine:.6f} (|ref| max "
        f"{np.abs(expected).max():.3f})")


if __name__ == "__main__":
    main()
