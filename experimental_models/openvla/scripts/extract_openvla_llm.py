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
"""Extract an OpenVLA checkpoint's Llama-2 as a plain LlamaForCausalLM checkpoint for tensorrt-edgellm-export.

The language model is stored under ``language_model.*``. Its config is ``LlamaConfig(**text_config)``, i.e. the
library defaults plus the checkpoint's overrides, which is what OpenVLA runs (rms_norm_eps 1e-6, not Llama-2's
published 1e-5). ``image_token_id`` is set to the pad id: the runtime fills those positions with the projected
patch embeddings, which OpenVLA places right after BOS.

    python extract_openvla_llm.py --checkpoint openvla-7b --out openvla_llm
    tensorrt-edgellm-export openvla_llm llm_onnx
    python extract_openvla_llm.py --checkpoint openvla-7b --patch-onnx-config llm_onnx/llm

tensorrt-edgellm-export writes image_token_id only for vision-language model types, and the runtime reads it from
the exported config, so the last step adds it there, together with the plain-concatenation chat template llm_build
requires (OpenVLA's requests are pre-tokenized and never templated).
"""

import argparse
import json
import os
import shutil

TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model",
                   "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "generation_config.json")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out")
    parser.add_argument(
        "--patch-onnx-config",
        help="exported LLM ONNX dir whose config.json gets image_token_id")
    args = parser.parse_args()
    config = json.load(open(os.path.join(args.checkpoint, "config.json")))
    if args.patch_onnx_config:
        path = os.path.join(args.patch_onnx_config, "config.json")
        exported = json.load(open(path))
        exported["image_token_id"] = int(config["pad_token_id"])
        json.dump(exported, open(path, "w"), indent=2)
        with open(os.path.join(args.patch_onnx_config, "chat_template.jinja"),
                  "w") as f:
            f.write(
                "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            )
        print(
            f"image_token_id {exported['image_token_id']}, chat_template.jinja -> {args.patch_onnx_config}"
        )
    if not args.out:
        return

    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.models.llama import LlamaConfig

    text = LlamaConfig(**config["text_config"]).to_dict()
    text.update(architectures=["LlamaForCausalLM"],
                model_type="llama",
                torch_dtype="bfloat16",
                image_token_id=int(config["pad_token_id"]))
    os.makedirs(args.out, exist_ok=True)
    json.dump(text, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    weight_map = json.load(
        open(os.path.join(args.checkpoint,
                          "model.safetensors.index.json")))["weight_map"]
    index = {}
    for number, shard in enumerate(sorted(set(weight_map.values()))):
        tensors = {}
        with safe_open(os.path.join(args.checkpoint, shard), "pt") as f:
            for key in f.keys():
                if key.startswith("language_model."):
                    tensors[key[len("language_model."):]] = f.get_tensor(key)
        if tensors:
            name = f"model-{number:05d}.safetensors"
            save_file(tensors,
                      os.path.join(args.out, name),
                      metadata={"format": "pt"})
            index.update({key: name for key in tensors})
    json.dump({
        "metadata": {},
        "weight_map": index
    },
              open(os.path.join(args.out, "model.safetensors.index.json"),
                   "w"),
              indent=2)
    for name in TOKENIZER_FILES:
        if os.path.exists(os.path.join(args.checkpoint, name)):
            shutil.copy(os.path.join(args.checkpoint, name), args.out)
    print(
        f"{len(index)} language-model tensors, image_token_id {text['image_token_id']}, "
        f"rms_norm_eps {text['rms_norm_eps']} -> {args.out}")


if __name__ == "__main__":
    main()
