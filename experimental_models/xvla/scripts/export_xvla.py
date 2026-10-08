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
"""Export a LeRobot X-VLA checkpoint as vision / encoder / step ONNX graphs (FP16, batch 1).

LeRobot's own modules provide every layer; the legacy TorchScript exporter lowers their attention to standard
ops, which TensorRT 10.3 builds (``trtexec --stronglyTyped``).

  vision.onnx   views [V, 3, 224, 224] (ImageNet-normalized) -> Florence-2 image features [V, tokens, D]
  encoder.onnx  primary view features [1, tokens, D] + token ids [1, L] -> Florence-2 encoder output [1, tokens + L, D]
  step.onnx     one denoising step for a domain id: x_t = x1 * t + action * (1 - t), gripper channels zeroed in x_t
                and the state, policy transformer -> predicted action (before the action space's postprocess),
                pulled toward rtc_seed by rtc_weight per row for real-time chunking (zero weights: the plain step)

Every graph goes through tensorrt_edgellm.onnx.trt_workarounds: the policy transformer's fused-QKV attention
otherwise hits a TensorRT 10.3 miscompilation (actions off by up to 0.04 on a 0.57 range, in FP32 too).

Writes config.json (shapes, steps, action space, state / action normalization) and the BART tokenizer.

    python export_xvla.py --checkpoint xvla-base --out xvla_onnx [--check ref.npz]
"""

import argparse
import json
import os
import sys

import torch
from torch import nn

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from tensorrt_edgellm.onnx.trt_workarounds import \
    apply_trt103_workarounds  # noqa: E402


class Vision(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.vlm = model.vlm

    def forward(self, images):
        return self.vlm.get_image_features(images).pooler_output


class Encoder(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.vlm = model.vlm

    def forward(self, primary, token_ids):
        # BartEncoder.forward without its mask construction, which does not trace: X-VLA attends to every position
        # (an all-ones mask), the same as passing no mask to the layers.
        embeds = torch.cat(
            [primary, self.vlm.get_input_embeddings()(token_ids)], dim=1)
        encoder = self.vlm.language_model.encoder
        hidden = encoder.layernorm_embedding(
            embeds + encoder.embed_positions(embeds[:, :, -1]))
        for layer in encoder.layers:
            hidden = layer(hidden, None)
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        return hidden


class Step(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        keep = torch.ones(model.dim_action)
        keep[list(model.action_space.gripper_idx)] = 0.0
        self.register_buffer("action_keep", keep)
        proprio_keep = torch.ones(model.dim_proprio)
        proprio_keep[[
            i for i in model.action_space.gripper_idx if i < model.dim_proprio
        ]] = 0.0
        self.register_buffer("proprio_keep", proprio_keep)

    def forward(self,
                x1,
                action,
                t,
                proprio,
                domain_id,
                vlm_features,
                aux_visual_inputs,
                rtc_seed=None,
                rtc_weight=None):
        x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        predicted = self.transformer(domain_id=domain_id,
                                     vlm_features=vlm_features,
                                     aux_visual_inputs=aux_visual_inputs,
                                     action_with_noise=x_t * self.action_keep,
                                     proprio=proprio * self.proprio_keep,
                                     t=t)
        if rtc_seed is None:
            return predicted
        return predicted + rtc_weight * (rtc_seed - predicted)


def load_policy(checkpoint):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cpu"
    policy = XVLAPolicy.from_pretrained(
        checkpoint, config=config).to("cpu").to(torch.float32).eval()
    # Eager attention traces to standard ONNX ops; it is the same function as SDPA.
    for module in policy.modules():
        module_config = getattr(module, "config", None)
        if module_config is not None and hasattr(module_config,
                                                 "_attn_implementation"):
            module_config._attn_implementation = "eager"
    if policy.config.action_mode.lower() != "ee6d":
        raise ValueError(
            f"only the ee6d action space is supported, got {policy.config.action_mode}"
        )
    return policy


def run_chain(vision, encoder, step, ref, steps):
    images = torch.from_numpy(ref["image_input"])
    mask = torch.from_numpy(ref["image_mask"])
    feats = vision(images[mask])
    all_feats = feats.new_zeros((images.shape[0], *feats.shape[1:]))
    all_feats[mask] = feats
    vlm = encoder(all_feats[:1], torch.from_numpy(ref["input_ids"])[None])
    aux = all_feats[1:].reshape(1, -1, all_feats.shape[-1])
    x1 = torch.from_numpy(ref["noise"])
    action = torch.zeros_like(x1)
    proprio = torch.from_numpy(ref["proprio"])[None]
    domain = torch.tensor([int(ref["domain_id"])])
    for i in range(steps, 0, -1):
        action = step(x1, action, torch.full((1, ), i / steps), proprio,
                      domain, vlm, aux)
    return vlm[0].numpy(), aux[0].numpy(), action[0].numpy()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out")
    parser.add_argument(
        "--check",
        help="lerobot_xvla_reference.py output: compare the split modules first"
    )
    args = parser.parse_args()

    import numpy as np
    policy = load_policy(args.checkpoint)
    model, config = policy.model, policy.config
    vision, encoder, step = Vision(model).eval(), Encoder(model).eval(), Step(
        model).eval()

    if args.check:
        ref = np.load(args.check)
        with torch.no_grad():
            vlm, aux, action = run_chain(vision, encoder, step, ref,
                                         config.num_denoising_steps)
        gripper = list(model.action_space.gripper_idx)
        action[:, gripper] = 1.0 / (1.0 + np.exp(-action[:, gripper]))
        for name, ours, expected in (("encoder output", vlm,
                                      ref["vlm_features"]),
                                     ("aux view features", aux,
                                      ref["aux_visual_inputs"]),
                                     ("actions", action,
                                      ref["model_actions"])):
            print(
                f"split {name} vs LeRobot: max|d| {np.abs(ours - expected).max():.3e} "
                f"(|ref| max {np.abs(expected).max():.2f})")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        # The saved tokenizer step pads to its own max_length, which is what the model sees.
        length = int(tokenizer_config(args.checkpoint)["max_length"])
        dim = model.vlm.config.vision_config.projection_dim
        with torch.no_grad():
            tokens = vision(torch.zeros(1, 3, 224, 224)).shape[1]
        vision, encoder, step = vision.half(), encoder.half(), step.half()
        f16 = torch.float16
        export = lambda m, a, name, i, o, axes=None: torch.onnx.export(
            m,
            a,
            os.path.join(args.out, f"{name}.onnx"),
            input_names=i,
            output_names=o,
            opset_version=17,
            dynamo=False,
            dynamic_axes=axes)
        export(vision, (torch.zeros(2, 3, 224, 224, dtype=f16), ), "vision",
               ["images"], ["image_features"], {
                   "images": {
                       0: "views"
                   },
                   "image_features": {
                       0: "views"
                   }
               })
        export(encoder, (torch.zeros(1, tokens, dim, dtype=f16),
                         torch.zeros(1, length, dtype=torch.int64)), "encoder",
               ["primary_features", "token_ids"], ["vlm_features"])
        views = int(config.num_image_views or 1)
        chunk, dim_action = model.chunk_size, model.dim_action
        # Distinct example tensors: the exporter merges inputs that are the same object.
        export(step,
               (torch.randn(1, chunk, dim_action, dtype=f16),
                torch.randn(1, chunk, dim_action,
                            dtype=f16), torch.ones(1, dtype=f16),
                torch.zeros(1, model.dim_proprio,
                            dtype=f16), torch.zeros(1, dtype=torch.int64),
                torch.zeros(1, tokens + length, dim, dtype=f16),
                torch.zeros(1, (views - 1) * tokens, dim, dtype=f16),
                torch.randn(1, chunk, dim_action,
                            dtype=f16), torch.zeros(1, chunk, 1, dtype=f16)),
               "step", [
                   "x1", "action", "t", "proprio", "domain_id", "vlm_features",
                   "aux_visual_inputs", "rtc_seed", "rtc_weight"
               ], ["action_next"])
        rewrites = {
            name:
            apply_trt103_workarounds(os.path.join(args.out, f"{name}.onnx"))
            for name in ("vision", "encoder", "step")
        }
        stage_runtime_assets(args.checkpoint, args.out, policy, tokens, length)
        print(
            f"exported vision / encoder / step -> {args.out}; TensorRT 10.3 rewrites {rewrites}"
        )


def tokenizer_config(checkpoint):
    steps = json.load(
        open(os.path.join(checkpoint, "policy_preprocessor.json")))["steps"]
    return next(s for s in steps
                if s["registry_name"] == "tokenizer_processor")["config"]


def stage_runtime_assets(checkpoint, out, policy, tokens, length):
    from transformers import AutoTokenizer

    config, model = policy.config, policy.model
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.backend_tokenizer.save(os.path.join(out, "tokenizer.json"))
    tokenizer.save_pretrained(os.path.join(out, "tokenizer"))
    steps = json.load(
        open(os.path.join(checkpoint, "policy_preprocessor.json")))["steps"]
    tokenizer_step = tokenizer_config(checkpoint)
    domain_step = next(
        (s for s in steps if s["registry_name"] == "xvla_add_domain_id"), None)
    cameras = [
        key.removeprefix("observation.images.")
        for key in config.image_features
    ]
    stats = {}
    post = json.load(
        open(os.path.join(checkpoint, "policy_postprocessor.json")))["steps"]
    unnormalizer = next(
        (s for s in post if s["registry_name"] == "unnormalizer_processor"),
        None)
    if unnormalizer and unnormalizer.get("state_file"):
        from safetensors.torch import load_file
        state = load_file(os.path.join(checkpoint, unnormalizer["state_file"]))
        stats = {
            "action_mean": state["action.mean"].tolist(),
            "action_std": state["action.std"].tolist()
        }
    json.dump(
        {
            **stats,
            "model_family":
            "xvla",
            "cameras":
            cameras,
            "num_views":
            int(config.num_image_views or len(cameras)),
            "image_size":
            list(config.resize_imgs_with_padding or (224, 224)),
            "image_tokens_per_view":
            tokens,
            "max_tokens":
            length,
            "padding_side":
            tokenizer_step["padding_side"],
            "pad_token_id":
            int(tokenizer.pad_token_id),
            "chunk_size":
            int(model.chunk_size),
            "action_dim":
            int(model.dim_action),
            "proprio_dim":
            int(model.dim_proprio),
            "state_dim":
            int(config.robot_state_feature.shape[0]),
            "gripper_idx":
            list(model.action_space.gripper_idx),
            "num_steps":
            int(config.num_denoising_steps),
            "domain_id":
            int(domain_step["config"]["domain_id"]) if domain_step else 0,
            "num_domains":
            int(config.num_domains),
            "state_normalization":
            getattr(config.normalization_mapping["STATE"], "value",
                    str(config.normalization_mapping["STATE"])),
            "action_normalization":
            getattr(config.normalization_mapping["ACTION"], "value",
                    str(config.normalization_mapping["ACTION"])),
        },
        open(os.path.join(out, "config.json"), "w"),
        indent=1)


if __name__ == "__main__":
    main()
