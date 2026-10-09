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
"""Extract a GR00T N1.5 / N1.6 Eagle backbone's Qwen3 as a plain checkpoint for tensorrt-edgellm-export.

The Eagle prefix then runs on Edge-LLM's LLM runtime (attention plugin, paged KV) instead of the plain-op prefix
engine, fed the visual engine's features as precomputed image embeddings at the image-context tokens. GR00T keeps
the first ``select_layer`` decoder layers and reads their post-norm output, so the config truncates the decoder
and sets ``emit_hidden_states: "post_norm"``. N1.6's policy casts the backbone to bf16 at load, which rounds
Qwen3's RoPE ``inv_freq`` buffer; the model was trained with those frequencies, so ``rope_inv_freq_bf16`` asks the
runtime to round them too. Weights go to FP16 as the plain-op engines store them (N1.6 keeps its fine-tuned top
layers in FP32).

    python extract_gr00t_eagle_llm.py --gr00t GR00T-N1.6-LIBERO --eagle-dir <Eagle dir> \\
        --tokenizer-dir engines/backbone --out eagle_llm
    tensorrt-edgellm-export eagle_llm eagle_llm_onnx
    python extract_gr00t_eagle_llm.py --patch-onnx-config eagle_llm_onnx/llm --out eagle_llm

tensorrt-edgellm-export does not carry ``rope_inv_freq_bf16`` into the exported config the runtime reads; the
last step copies it there.
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json",
                   "vocab.json", "merges.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gr00t")
    parser.add_argument("--eagle-dir",
                        help="the Eagle model directory (its config.json)")
    parser.add_argument("--tokenizer-dir", help="the Eagle tokenizer files")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--patch-onnx-config",
        help=
        "exported LLM ONNX dir: copy rope_inv_freq_bf16 from --out into its config.json"
    )
    args = parser.parse_args()

    if args.patch_onnx_config:
        source = json.load(open(os.path.join(args.out, "config.json")))
        path = os.path.join(args.patch_onnx_config, "config.json")
        exported = json.load(open(path))
        exported["rope_inv_freq_bf16"] = bool(source["rope_inv_freq_bf16"])
        json.dump(exported, open(path, "w"), indent=2)
        print(f"rope_inv_freq_bf16={exported['rope_inv_freq_bf16']} -> {path}")
        return
    if not (args.gr00t and args.eagle_dir and args.tokenizer_dir):
        parser.error("--gr00t, --eagle-dir and --tokenizer-dir are required")

    gr00t = json.load(open(os.path.join(args.gr00t, "config.json")))
    if gr00t["model_type"] == "gr00t_n1_5":
        root = "backbone.eagle_model.language_model.model."
        layers = int(gr00t["backbone_cfg"]["select_layer"])
        round_rope = False
    elif gr00t["model_type"] == "Gr00tN1d6":
        root = "backbone.model.language_model.model."
        layers = int(gr00t["select_layer"])
        round_rope = True
    else:
        raise SystemExit(
            f"not a GR00T Eagle checkpoint: {gr00t['model_type']}")
    eagle = json.load(open(os.path.join(args.eagle_dir, "config.json")))

    config = dict(eagle["text_config"])
    config.update(architectures=["Qwen3ForCausalLM"],
                  model_type="qwen3",
                  num_hidden_layers=layers,
                  torch_dtype="float16",
                  tie_word_embeddings=True,
                  image_token_id=int(eagle["image_token_index"]),
                  emit_hidden_states="post_norm",
                  rope_inv_freq_bf16=round_rope)
    if "layer_types" in config:
        config["layer_types"] = config["layer_types"][:layers]
    os.makedirs(args.out, exist_ok=True)
    json.dump(config,
              open(os.path.join(args.out, "config.json"), "w"),
              indent=2)
    for name in TOKENIZER_FILES:
        if os.path.exists(os.path.join(args.tokenizer_dir, name)):
            shutil.copy(os.path.join(args.tokenizer_dir, name), args.out)

    weight_map = json.load(
        open(os.path.join(args.gr00t,
                          "model.safetensors.index.json")))["weight_map"]
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(args.gr00t, shard), "pt") as f:
            for key in f.keys():
                if not key.startswith(root):
                    continue
                name = key[len(root):]
                if name.startswith("layers.") and int(
                        name.split(".")[1]) >= layers:
                    continue
                tensors["model." + name] = f.get_tensor(key).to(
                    torch.float16).contiguous()
    kept = {
        int(k.split(".")[2])
        for k in tensors if k.startswith("model.layers.")
    }
    if kept != set(range(layers)):
        raise SystemExit(
            f"decoder layers {sorted(kept)}, expected 0..{layers - 1}")
    save_file(tensors,
              os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})
    print(
        f"{len(tensors)} tensors, {layers} decoder layers, rope_inv_freq_bf16={round_rope} -> {args.out}"
    )


if __name__ == "__main__":
    main()
