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
"""Export RLDX-1's language model halves (rldx_llm.py) and its action step (rldx_action.py) as FP16 ONNX graphs,
one per process to bound memory.

    PYTHONPATH=RLDX-1 python export_rldx.py --checkpoint RLDX-1-FT-LIBERO --vlm RLDX-1-VLM --stage llm_a --out onnx
    PYTHONPATH=RLDX-1 python export_rldx.py --checkpoint RLDX-1-FT-LIBERO --vlm RLDX-1-VLM --stage llm_b --out onnx
    PYTHONPATH=RLDX-1 python export_rldx.py --checkpoint RLDX-1-FT-LIBERO --vlm RLDX-1-VLM --stage action --out onnx

The sequence length and the number of visual tokens are dynamic; the cos / sin tables stay FP32.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rldx_action  # noqa: E402
import rldx_llm  # noqa: E402

LLM_PREFIX = "backbone.qwen_model.model.language_model."


def export_llm_a(args, text_config):
    weights = rldx_llm.load_tensors(args.checkpoint, LLM_PREFIX)
    weights = {
        k: v
        for k, v in weights.items() if k == "embed_tokens.weight" or any(
            k.startswith(f"layers.{i}.") for i in range(4))
    }
    cog = rldx_llm.load_tensors(args.checkpoint, "backbone.cog_emb")[""]
    model = rldx_llm.LlmA(text_config, weights, cog, torch.float16).eval()
    del weights
    hidden, head = text_config.hidden_size, text_config.head_dim
    length, visual = 560, 512
    index = torch.full((length, ), -1, dtype=torch.int64)
    index[20:20 + visual] = torch.arange(visual)
    inputs = (torch.zeros(length, dtype=torch.int64),
              torch.randn(visual, hidden, dtype=torch.float16),
              torch.randn(3, visual, hidden, dtype=torch.float16), index,
              torch.randn(length + 64, head), torch.randn(length + 64, head))
    path = os.path.join(args.out, "llm_a", "model.onnx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(model,
                      inputs,
                      path,
                      input_names=[
                          "input_ids", "visual", "deepstack", "visual_index",
                          "cos", "sin"
                      ],
                      output_names=["hidden"],
                      dynamic_axes={
                          "input_ids": {
                              0: "prompt"
                          },
                          "visual": {
                              0: "visual_tokens"
                          },
                          "deepstack": {
                              1: "visual_tokens"
                          },
                          "visual_index": {
                              0: "prompt"
                          },
                          "cos": {
                              0: "sequence"
                          },
                          "sin": {
                              0: "sequence"
                          },
                          "hidden": {
                              1: "sequence"
                          }
                      },
                      opset_version=17,
                      dynamo=False)
    return path


def export_llm_b(args, text_config):
    weights = rldx_llm.load_tensors(args.checkpoint, LLM_PREFIX)
    weights = {
        k: v
        for k, v in weights.items()
        if not k.startswith("embed_tokens") and not any(
            k.startswith(f"layers.{i}.") for i in range(4))
    }
    layers = 1 + max(
        int(k.split(".")[1]) for k in weights if k.startswith("layers."))
    model = rldx_llm.LlmB(text_config,
                          weights,
                          layers,
                          torch.float16,
                          residual_dtype=getattr(torch, args.residual)).eval()
    del weights
    hidden, head = text_config.hidden_size, text_config.head_dim
    length, kept = 624, 230
    inputs = (torch.randn(1, length, hidden,
                          dtype=torch.float16), torch.rand(length),
              torch.arange(kept), torch.randn(kept,
                                              head), torch.randn(kept, head))
    path = os.path.join(args.out, "llm_b", "model.onnx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=["hidden", "pool", "keep_index", "cos", "sin"],
        output_names=["cognition"],
        dynamic_axes={
            "hidden": {
                1: "sequence"
            },
            "pool": {
                0: "sequence"
            },
            "keep_index": {
                0: "compressed"
            },
            "cos": {
                0: "compressed"
            },
            "sin": {
                0: "compressed"
            }
        },
        opset_version=17,
        dynamo=False)
    return path


def export_action(args, text_config):
    import json
    model, config = rldx_action.build_action_model(args.checkpoint)
    rldx_action.mixed_precision(model)
    embodiment = json.load(
        open(os.path.join(args.checkpoint,
                          "embodiment_id.json")))[args.embodiment]
    step = rldx_action.Step(model.half(), embodiment_id=int(embodiment)).eval()
    horizon, dim = 16, int(config.max_action_dim)
    f16 = torch.float16
    # Distinct example tensors: the exporter merges inputs that are the same object.
    inputs = (torch.randn(1, horizon, dim,
                          dtype=f16), torch.full((1, ), 0.25, dtype=f16),
              torch.full((1, ), 0.5, dtype=f16),
              torch.randn(1, 64, text_config.hidden_size, dtype=f16),
              torch.randn(1, 1, int(config.max_state_dim),
                          dtype=f16), torch.ones(1, horizon, 1, dtype=f16))
    path = os.path.join(args.out, "action", "model.onnx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(
        step,
        inputs,
        path,
        input_names=["x", "t", "dt", "cognition", "state", "strength"],
        output_names=["x_next", "velocity"],
        opset_version=17,
        dynamo=False)
    return path


def export_assets(args, text_config):
    """config.json for RldxPolicy and the tokenizer, next to the engines."""
    import json
    import shutil

    import rldx.model.core.rldx  # noqa: F401 -- registers RLDX-1 with AutoConfig
    from transformers import AutoConfig
    vlm = AutoConfig.from_pretrained(args.vlm)
    model = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    processor = json.load(
        open(os.path.join(args.checkpoint, "processor_config.json")))
    modality = processor["processor_kwargs"]["modality_configs"][
        args.embodiment]
    stats = json.load(open(os.path.join(args.checkpoint,
                                        "statistics.json")))[args.embodiment]

    def ranges(group):
        keys = modality[group]["modality_keys"]
        return [{
            "key": k,
            "q01": stats[group][k]["q01"],
            "q99": stats[group][k]["q99"]
        } for k in keys]

    merge = vlm.vision_config.spatial_merge_size
    grid = args.image_size // vlm.vision_config.patch_size // merge
    embodiments = json.load(
        open(os.path.join(args.checkpoint, "embodiment_id.json")))
    os.makedirs(args.out, exist_ok=True)
    config = {
        "model_family": "rldx",
        "embodiment": args.embodiment,
        "embodiment_id": int(embodiments[args.embodiment]),
        "cameras": modality["video"]["modality_keys"],
        "frame_history": modality["video"]["delta_indices"],
        "image_size": args.image_size,
        "grid": [grid, grid],
        "cognition_tokens": int(model.n_cog_tokens),
        "head_dim": int(text_config.head_dim),
        "rope_theta": float(text_config.rope_theta),
        "mrope_section": list(text_config.rope_scaling["mrope_section"]),
        "hidden_size": int(text_config.hidden_size),
        "action_horizon": 16,
        "max_action_dim": int(model.max_action_dim),
        "max_state_dim": int(model.max_state_dim),
        "denoising_steps": 4,
        "state": ranges("state"),
        "action": ranges("action"),
    }
    json.dump(config,
              open(os.path.join(args.out, "config.json"), "w"),
              indent=1)
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json", "vocab.json",
                 "merges.txt"):
        if os.path.exists(os.path.join(args.vlm, name)):
            shutil.copy(os.path.join(args.vlm, name), args.out)
    return os.path.join(args.out, "config.json")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vlm",
                        required=True,
                        help="RLWRLD/RLDX-1-VLM (its config.json)")
    parser.add_argument("--stage",
                        required=True,
                        choices=("llm_a", "llm_b", "action", "assets"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--embodiment", default="general_embodiment")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--residual",
        default="float16",
        choices=("float16", "float32"),
        help="llm_b's residual stream dtype (the GEMMs stay FP16)")
    args = parser.parse_args()

    text_config = rldx_llm.text_config(args.vlm)
    export = {
        "llm_a": export_llm_a,
        "llm_b": export_llm_b,
        "action": export_action,
        "assets": export_assets
    }[args.stage]
    with torch.no_grad():
        print(f"exported {args.stage} -> {export(args, text_config)}")


if __name__ == "__main__":
    main()
