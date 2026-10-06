# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Extract a GR00T N1.7 backbone as a Qwen3-VL checkpoint for tensorrt-edgellm-export.

GR00T N1.7 stores its fine-tuned Cosmos-Reason2 (Qwen3-VL) backbone under
``backbone.model.*`` with only the first ``select_layer`` decoder layers. This
writes those weights under Hugging Face Qwen3-VL names, next to the base
model's config (decoder truncated to ``select_layer``) and processor files, and
sets ``emit_hidden_states`` so the engine returns the full-sequence pre-norm
hidden states the action head cross-attends to.

    python extract_gr00t_n1_7_backbone.py --gr00t MODEL_ZOO/GR00T-N1.7-SO101-Multitask \\
        --base ~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/<rev> --out gr00t_backbone
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GR00T_PREFIX = "backbone.model."
PROCESSOR_FILES = ("chat_template.json", "generation_config.json",
                   "merges.txt", "preprocessor_config.json",
                   "tokenizer_config.json", "tokenizer.json",
                   "video_preprocessor_config.json", "vocab.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gr00t", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gr00t_config = json.load(open(os.path.join(args.gr00t, "config.json")))
    select_layer = int(gr00t_config["select_layer"])
    os.makedirs(args.out, exist_ok=True)

    config = json.load(open(os.path.join(args.base, "config.json")))
    config["text_config"]["num_hidden_layers"] = select_layer
    if "layer_types" in config["text_config"]:
        config["text_config"]["layer_types"] = config["text_config"][
            "layer_types"][:select_layer]
    # GR00T pops the extra layers off a full-depth model, and its hidden_states[-1] is then the last kept layer's
    # output before the final norm (checked against the official Gr00tPolicy). A checkpoint that is truncated
    # through num_hidden_layers instead returns the post-norm tensor from transformers, which the head was not
    # trained on.
    config["emit_hidden_states"] = "pre_norm"
    config["gr00t_select_layer"] = select_layer
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    for name in PROCESSOR_FILES:
        if os.path.exists(os.path.join(args.base, name)):
            shutil.copy(os.path.join(args.base, name), args.out)

    weight_map = json.load(
        open(os.path.join(args.gr00t,
                          "model.safetensors.index.json")))["weight_map"]
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(args.gr00t, shard), "pt") as f:
            for key in f.keys():
                if key.startswith(GR00T_PREFIX):
                    tensors[key[len(GR00T_PREFIX):]] = f.get_tensor(key).to(
                        torch.bfloat16).contiguous()
    layers = {
        int(k.split(".")[3])
        for k in tensors if k.startswith("model.language_model.layers.")
    }
    if layers != set(range(select_layer)):
        raise SystemExit(
            f"checkpoint has decoder layers {sorted(layers)}, expected 0..{select_layer - 1}"
        )
    save_file(tensors,
              os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})
    print(
        f"{len(tensors)} backbone tensors, {select_layer} decoder layers -> {args.out}"
    )


if __name__ == "__main__":
    main()
