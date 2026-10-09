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
"""Export MolmoAct2's continuous-inference graphs (molmoact2_modules.py) as FP16 ONNX, one stage per process.

    python export_molmoact2.py --checkpoint MolmoAct2-LIBERO-LeRobot --hf MolmoAct2-LIBERO-hf \\
        --fast MolmoAct2-FAST-Tokenizer --stage vision|prefix_a|prefix_b|context|step|assets --out DIR

The graphs take a dynamic prompt length; the vision graph takes the checkpoint's two cameras. ``assets`` writes
config.json (geometry, special tokens, the masked q01 / q99 statistics) and the tokenizer for MolmoAct2Policy. Every
graph goes through tensorrt_edgellm.onnx.trt_workarounds.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import molmoact2_modules as M  # noqa: E402
import molmoact2_reference as R  # noqa: E402

LENGTH = 500
VISUAL = 392


class PrefixFirst(nn.Module):

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def forward(self, input_ids, visual, image_flag, cos, sin):
        return self.prefix(input_ids, visual, image_flag, cos, sin)


class PrefixLater(nn.Module):

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def forward(self, hidden, image_flag, cos, sin):
        _, keys, values = self.prefix(hidden, None, image_flag, cos, sin)
        return keys, values


def load(args, workdir):
    import lerobot.policies.molmoact2.processor_molmoact2  # noqa: F401
    args.precision, args.device = "bf16", "cpu"
    checkpoint = R.local_checkpoint(args, workdir)
    R.from_config_only()
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    policy = MolmoAct2Policy.from_pretrained(checkpoint).eval()
    return policy._backbone().to(torch.float16)


def export(module, inputs, path, names, outputs, axes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module,
                          inputs,
                          path,
                          input_names=names,
                          output_names=outputs,
                          dynamic_axes=axes,
                          opset_version=17,
                          dynamo=False)
    from tensorrt_edgellm.onnx.trt_workarounds import apply_trt103_workarounds
    print(f"{path}: TensorRT 10.3 rewrites {apply_trt103_workarounds(path)}")


def rope_example(length, head_dim):
    f = torch.outer(torch.arange(length, dtype=torch.float32),
                    torch.rand(head_dim // 2))
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--fast", required=True)
    parser.add_argument("--stage",
                        required=True,
                        choices=("vision", "prefix_a", "prefix_b", "context",
                                 "step", "assets"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--split",
                        type=int,
                        default=18,
                        help="first block of prefix_b")
    args = parser.parse_args()
    if args.stage == "assets":
        export_assets(args)
        return

    workdir = tempfile.TemporaryDirectory(prefix="molmoact2_ckpt_")
    model = load(args, workdir.name)
    config = model.config
    hidden = model.transformer.wte.embedding.shape[1]
    blocks = len(model.transformer.blocks)
    kv_dim = config.text_config.num_key_value_heads * config.text_config.head_dim \
        if hasattr(config, "text_config") else 1024
    head_dim = config.text_config.head_dim if hasattr(config,
                                                      "text_config") else 128
    f16 = torch.float16
    seq = {0: "sequence"}
    out = lambda name: os.path.join(args.out, name, "model.onnx")
    if args.stage == "vision":
        # The pooling table only depends on the 378 x 378 crop: rebuild it with the official processor.
        from lerobot.policies.molmoact2.molmoact2_hf_model.image_processing_molmoact2 import \
            arange_for_pooling
        idx = torch.from_numpy(arange_for_pooling(torch.arange(729).reshape(27, 27).numpy(),
                                                  2, 2)).reshape(-1, 4)
        table = torch.cat([idx, idx])
        export(M.Vision(model.vision_backbone, table),
               (torch.rand(2, 729, 588) * 2 - 1, ), out("vision"), ["patches"],
               ["visual"], None)
    elif args.stage == "prefix_a":
        cos, sin = rope_example(LENGTH, head_dim)
        ids = torch.randint(0, 151000, (LENGTH, ))
        ids[10:10 + VISUAL] = config.image_patch_id
        export(PrefixFirst(
            M.Prefix(model.transformer, 0, args.split, False,
                     config.image_patch_id)),
               (ids, torch.randn(VISUAL, hidden, dtype=f16),
                (torch.arange(LENGTH) < 400).half(), cos, sin), out("prefix_a"),
               ["input_ids", "visual", "image_flag", "cos", "sin"],
               ["hidden", "keys", "values"], {
                   "input_ids": seq,
                   "image_flag": seq,
                   "cos": seq,
                   "sin": seq,
                   "hidden": {
                       1: "sequence"
                   },
                   "keys": {
                       1: "sequence"
                   },
                   "values": {
                       1: "sequence"
                   }
               })
    elif args.stage == "prefix_b":
        cos, sin = rope_example(LENGTH, head_dim)
        export(PrefixLater(
            M.Prefix(model.transformer, args.split, blocks, True,
                     config.image_patch_id)),
               (torch.randn(1, LENGTH, hidden, dtype=f16),
                (torch.arange(LENGTH) < 400).half(), cos, sin), out("prefix_b"),
               ["hidden", "image_flag", "cos", "sin"], ["keys", "values"], {
                   "hidden": {
                       1: "sequence"
                   },
                   "image_flag": seq,
                   "cos": seq,
                   "sin": seq,
                   "keys": {
                       1: "sequence"
                   },
                   "values": {
                       1: "sequence"
                   }
               })
    elif args.stage == "context":
        export(M.Context(model._require_action_expert()),
               (torch.randn(blocks, LENGTH, kv_dim, dtype=f16),
                torch.randn(blocks, LENGTH, kv_dim, dtype=f16)), out("context"),
               ["keys", "values"], ["context_k", "context_v"], {
                   "keys": {
                       1: "sequence"
                   },
                   "values": {
                       1: "sequence"
                   },
                   "context_k": {
                       2: "sequence"
                   },
                   "context_v": {
                       2: "sequence"
                   }
               })
    elif args.stage == "step":
        expert = model._require_action_expert()
        heads = expert.config.num_heads
        head = expert.config.hidden_size // heads
        horizon = int(config.max_action_horizon)
        dims = int(config.max_action_dim)
        steps = int(getattr(config, "flow_matching_num_steps", 10))
        export(M.Step(expert, args_action_dim(args), horizon, steps),
               (torch.randn(1, horizon, dims, dtype=f16),
                torch.tensor([3]), torch.full((1, ), 0.1, dtype=f16),
                torch.randn(blocks, 1, LENGTH, heads, head, dtype=f16),
                torch.randn(blocks, 1, LENGTH, heads, head, dtype=f16),
                torch.ones(1, LENGTH, dtype=f16),
                torch.ones(1, horizon, 1, dtype=f16)), out("step"), [
                    "x", "step", "dt", "context_k", "context_v",
                    "encoder_mask", "strength"
                ], ["x_next", "velocity"], {
                    "context_k": {
                        2: "sequence"
                    },
                    "context_v": {
                        2: "sequence"
                    },
                    "encoder_mask": {
                        1: "sequence"
                    }
                })


def args_action_dim(args):
    return int(
        json.load(open(os.path.join(args.checkpoint,
                                    "config.json")))["output_features"]
        ["action"]["shape"][0])


def export_assets(args):
    from safetensors.torch import load_file
    ck = json.load(open(os.path.join(args.checkpoint, "config.json")))
    pre = json.load(
        open(os.path.join(args.checkpoint, "policy_preprocessor.json")))
    pack = next(s["config"] for s in pre["steps"]
                if s["registry_name"] == "molmoact2_pack_inputs")
    hf = json.load(open(os.path.join(args.hf, "config.json")))
    text = hf.get("text_config", hf)

    def stats(prefix, feature):
        """The masked q01 / q99 normalization of one feature (masked-out dims pass through raw)."""
        name = next(f for f in os.listdir(args.checkpoint)
                    if f.startswith(prefix) and f.endswith(".safetensors"))
        values = load_file(os.path.join(args.checkpoint, name))
        return {
            key: values[f"{feature}.{key}"].tolist()
            for key in ("q01", "q99", "mask")
        }

    os.makedirs(args.out, exist_ok=True)
    config = {
        "model_family": "molmoact2",
        "cameras": [k.split(".")[-1] for k in ck["image_keys"]],
        "image_size": 378,
        "patch_size": 14,
        "image_tokens": 196,
        "setup": pack["setup_type"],
        "control_mode": pack["control_mode"],
        "state_bins": int(pack["num_state_tokens"]),
        "action_horizon": int(ck["chunk_size"]),
        "action_dim": args_action_dim(args),
        "max_action_dim": int(pack["max_action_dim"]),
        "state_dim": int(ck["input_features"]["observation.state"]["shape"][0]),
        "flow_steps": int(hf.get("flow_matching_num_steps", 10)),
        "head_dim": int(text.get("head_dim", 128)),
        "rope_theta": float(text.get("rope_theta", 5000000.0)),
        "bos_token_id": 151645,
        "image_patch_id": int(hf["image_patch_id"]),
        "state_normalization": stats("policy_preprocessor_step",
                                     "observation.state"),
        "action_normalization": stats("policy_postprocessor_step", "action"),
    }
    json.dump(config,
              open(os.path.join(args.out, "config.json"), "w"),
              indent=1)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(args.hf, name), args.out)
    print(f"assets -> {args.out}")


if __name__ == "__main__":
    main()
