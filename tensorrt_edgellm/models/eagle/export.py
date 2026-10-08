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
"""Export a GR00T N1.5 or N1.6 Eagle backbone as visual / prefix ONNX components (FP16, batch 1).

    python -m tensorrt_edgellm.models.eagle.export <gr00t checkpoint> <eagle dir> <out_dir>

``<eagle dir>`` is the Eagle directory of the matching GR00T source tree, which holds the tokenizer:
``gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2`` (N1.6) or ``gr00t/model/backbone/eagle2_hg_model``
(N1.5). Writes ``<out_dir>/{visual,prefix}/model.onnx``, ``tokenizer.json`` and ``config.json`` with
the image pipeline and prompt layout the runtime reproduces. N1.5's image pipeline comes from its data
config, which lives in the source tree: every N1.5 data config centre-crops 95% and resizes to 224x224
(--n15-crop-scale, --n15-image-size).
"""

import argparse
import json
import logging
import os

import torch

from ...onnx.export_encoder import _run_dynamo_export
from .modeling_eagle import (EagleConfig, EaglePrefix, EagleVisual,
                             gr00t_eagle_config, load_eagle_weights)

logger = logging.getLogger(__name__)

MAX_VIEWS = 3
MAX_TOKENS = 1024
PROMPT_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"


def load_backbone_state(checkpoint: str) -> dict:
    from safetensors import safe_open
    weight_map = json.load(
        open(os.path.join(checkpoint,
                          "model.safetensors.index.json")))["weight_map"]
    state = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(checkpoint, shard), "pt") as f:
            for key in f.keys():
                if key.startswith(
                        "backbone."
                ) and "lm_head" not in key and ".head." not in key:
                    state[key] = f.get_tensor(key).float()
    return state


def image_size(processor: dict, height: int, width: int):
    """The size SigLIP2 sees: albumentations SmallestMaxSize, the GR00T FractionalCenterCrop
    (int(side * fraction) per side), SmallestMaxSize again, then Eagle's smart_resize to multiples of
    28. Python's round() (half to even) throughout, as the official code."""
    edge, fraction = processor["shortest_image_edge"], processor[
        "crop_fraction"]

    def shortest(h, w):
        scale = edge / float(min(h, w))
        return (h, w) if scale == 1.0 else (round(h * scale), round(w * scale))

    h, w = shortest(height, width)
    h, w = shortest(max(1, int(h * fraction)), max(1, int(w * fraction)))
    return max(28, round(h / 28) * 28), max(28, round(w / 28) * 28)


def runtime_config(checkpoint: str, camera_height: int, camera_width: int,
                   n15_crop_scale: float, n15_image_size: int) -> dict:
    """The image pipeline and prompt layout of the checkpoint's version, as config.json fields."""
    model_config = json.load(open(os.path.join(checkpoint, "config.json")))
    if model_config["model_type"] == "gr00t_n1_5":
        # VideoCrop (eval: centre crop) and VideoResize (bilinear, antialiased) on [0, 1] floats, then
        # uint8; Eagle 2.5 then sees one 224 tile. The images come before the instruction, and the template
        # adds the assistant turn.
        return {
            "image_pipeline": "gr00t_n15",
            "crop_scale": n15_crop_scale,
            "image_height": n15_image_size,
            "image_width": n15_image_size,
            "formalize_language": False,
            "text_after_images": True,
            "prompt_suffix": "<|im_end|>\n<|im_start|>assistant\n",
        }
    processor = json.load(
        open(os.path.join(checkpoint,
                          "processor_config.json")))["processor_kwargs"]
    if model_config.get("model_name") != "nvidia/Eagle-Block2A-2B-v2":
        raise ValueError(
            f"only the Eagle-Block2A-2B-v2 backbone is supported, got {model_config.get('model_name')}"
        )
    if not processor.get("use_albumentations") or processor.get(
            "image_crop_size") is not None:
        raise ValueError(
            "only the albumentations shortest-edge / fractional-crop image pipeline is supported"
        )
    height, width = image_size(processor, camera_height, camera_width)
    return {
        "image_pipeline": "gr00t_n16",
        "shortest_image_edge": processor["shortest_image_edge"],
        "crop_fraction": processor["crop_fraction"],
        "image_height": height,
        "image_width": width,
        "formalize_language": bool(processor.get("formalize_language", True)),
        "text_after_images": False,
        "prompt_suffix": "<|im_end|>\n",
    }


def export_eagle(checkpoint: str, eagle_dir: str, out_dir: str,
                 runtime: dict) -> None:
    height, width = runtime["image_height"], runtime["image_width"]
    cfg = gr00t_eagle_config(checkpoint, height, width)
    visual, prefix = EagleVisual(cfg), EaglePrefix(cfg)
    load_eagle_weights(load_backbone_state(checkpoint), visual, prefix)
    visual, prefix = visual.half().eval(), prefix.half().eval()

    for name in ("visual", "prefix"):
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    Dim = torch.export.Dim
    _run_dynamo_export(
        visual, (torch.zeros(2, 3, height, width, dtype=torch.float16), ),
        os.path.join(out_dir, "visual", "model.onnx"), ["pixel_values"],
        ["image_features"], ({
            0: Dim("num_views", min=1, max=MAX_VIEWS)
        }, ))
    _run_dynamo_export(
        prefix,
        (torch.zeros(1, 2 * cfg.image_tokens + 40, dtype=torch.int64),
         torch.zeros(
             1, 2 * cfg.image_tokens, cfg.text_hidden, dtype=torch.float16)),
        os.path.join(out_dir, "prefix", "model.onnx"),
        ["token_ids", "image_features"], ["backbone_features"],
        ({
            1: Dim("num_tokens", min=2, max=MAX_TOKENS)
        }, {
            1: Dim("image_tokens", min=1, max=MAX_VIEWS * cfg.image_tokens)
        }))
    stage_runtime_assets(eagle_dir, out_dir, cfg, runtime)
    logger.info("Eagle backbone export complete: %s", out_dir)


def stage_runtime_assets(eagle_dir: str, out_dir: str, cfg: EagleConfig,
                         runtime: dict) -> None:
    from transformers import Qwen2TokenizerFast
    os.makedirs(out_dir, exist_ok=True)
    Qwen2TokenizerFast.from_pretrained(eagle_dir).backend_tokenizer.save(
        os.path.join(out_dir, "tokenizer.json"))
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        source = os.path.join(eagle_dir, name)
        if os.path.exists(source):
            with open(source) as f, open(os.path.join(out_dir, name),
                                         "w") as g:
                g.write(f.read())
    json.dump(
        {
            "model_family": "gr00t_eagle",
            **runtime,
            "image_tokens_per_view": cfg.image_tokens,
            "max_views": MAX_VIEWS,
            "max_tokens": MAX_TOKENS,
            "hidden_size": cfg.text_hidden,
            "image_token_id": cfg.image_token_id,
            # Eagle's chat template for one user turn of the instruction and the images.
            "prompt_prefix": PROMPT_PREFIX,
            "image_prefix": "<image {index}><img>",
            "image_context": "<IMG_CONTEXT>",
            "image_suffix": "</img>",
        },
        open(os.path.join(out_dir, "config.json"), "w"),
        indent=1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("eagle_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--n15-crop-scale", type=float, default=0.95)
    parser.add_argument("--n15-image-size", type=int, default=224)
    parser.add_argument(
        "--assets-only",
        action="store_true",
        help=
        "rewrite config.json and the tokenizer without re-exporting the ONNX")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    runtime = runtime_config(args.checkpoint, args.camera_height,
                             args.camera_width, args.n15_crop_scale,
                             args.n15_image_size)
    if args.assets_only:
        cfg = gr00t_eagle_config(args.checkpoint, runtime["image_height"],
                                 runtime["image_width"])
        stage_runtime_assets(args.eagle_dir, args.out_dir, cfg, runtime)
    else:
        export_eagle(args.checkpoint, args.eagle_dir, args.out_dir, runtime)


if __name__ == "__main__":
    main()
