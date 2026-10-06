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
"""Export the GR00T N1.7 action head as three ONNX graphs for one embodiment.

The DiT's cross-attention keys and values depend only on the backbone
features, which are fixed for a whole action-chunk call, so they are computed
once instead of once per denoising step:

  vl_prep.onnx        backbone hidden states -> vlln -> vl_self_attention ->
                      K/V of every cross-attention block, plus additive
                      text/image attention biases            (once per call)
  state_encoder.onnx  state -> state token                   (once per call)
  denoise_step.onnx   action encoder -> DiT with the cached K/V -> action
                      decoder -> Euler update with the RTC velocity mask
                                                             (num_inference_timesteps times)

The official GR00T modules provide the weights and every layer's math; only
the cross-attention of the DiT blocks is re-expressed to take K/V as inputs.
Embodiment-specific weight banks are sliced to the chosen embodiment.

Needs the GR00T N1.7 source (--gr00t-src: the directory holding the ``gr00t``
package) and its dependencies (transformers 4.57, diffusers).

    python export_gr00t_n1_7_action_head.py --gr00t-src <dir> --checkpoint MODEL_ZOO/GR00T-N1.7-SO101-Multitask \\
        --embodiment new_embodiment --check
    python export_gr00t_n1_7_action_head.py ... --out action_onnx --max-backbone-tokens 512
"""

import argparse
import json
import os
import sys
import tempfile

import torch
import torch.nn.functional as F
from torch import nn

#: Additive bias for masked backbone positions; finite so FP16 engines stay NaN-free.
MASKED_BIAS = -30000.0


def load_action_head(gr00t_src, checkpoint):
    sys.path.insert(0, gr00t_src)
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from safetensors import safe_open

    config_dict = json.load(open(os.path.join(checkpoint, "config.json")))
    config = Gr00tN1d7Config(**{
        k: v
        for k, v in config_dict.items() if v is not None
    })
    head = Gr00tN1d7ActionHead(config)
    weight_map = json.load(
        open(os.path.join(checkpoint,
                          "model.safetensors.index.json")))["weight_map"]
    state = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(checkpoint, shard), "pt") as f:
            for key in f.keys():
                if key.startswith("action_head."):
                    state[key[len("action_head."):]] = f.get_tensor(
                        key).float()
    head.load_state_dict(state, strict=True)
    return head.float().eval(), config


def embodiment_index(checkpoint, embodiment):
    mapping = json.load(open(os.path.join(checkpoint, "embodiment_id.json")))
    return int(mapping[embodiment])


def slice_embodiment(module, index):
    """Keep one embodiment's slice of every CategorySpecificLinear; callers then pass category 0."""
    for child in module.modules():
        if hasattr(child, "W") and hasattr(child, "b") and child.W.dim() == 3:
            child.W = nn.Parameter(child.W.data[index:index + 1].clone())
            child.b = nn.Parameter(child.b.data[index:index + 1].clone())
            child.num_categories = 1


def cross_blocks(head):
    blocks = head.model.transformer_blocks
    return [blocks[i] for i in range(0, len(blocks), 2)]


class VLPrep(nn.Module):

    def __init__(self, head):
        super().__init__()
        self.vlln = head.vlln
        self.vl_self_attention = head.vl_self_attention
        self.cross = nn.ModuleList(cross_blocks(head))

    def forward(self, backbone_features, image_mask, attention_mask):
        features = self.vl_self_attention(self.vlln(backbone_features))
        keys = torch.stack([b.attn1.to_k(features) for b in self.cross])
        values = torch.stack([b.attn1.to_v(features) for b in self.cross])
        text = (~image_mask) & attention_mask
        image = image_mask & attention_mask
        zero = torch.zeros_like(backbone_features[..., 0])
        text_bias = torch.where(text, zero, zero + MASKED_BIAS)[:, None,
                                                                None, :]
        image_bias = torch.where(image, zero, zero + MASKED_BIAS)[:, None,
                                                                  None, :]
        return keys, values, text_bias, image_bias


class StateEncoder(nn.Module):

    def __init__(self, head):
        super().__init__()
        self.state_encoder = head.state_encoder

    def forward(self, state):
        category = torch.zeros(state.shape[0],
                               dtype=torch.long,
                               device=state.device)
        return self.state_encoder(state, category)


class DenoiseStep(nn.Module):

    def __init__(self, head, attend_text_every_n_blocks):
        super().__init__()
        self.action_encoder = head.action_encoder
        self.action_decoder = head.action_decoder
        self.position_embedding = head.position_embedding if head.config.add_pos_embed else None
        self.dit = head.model
        self.action_horizon = head.action_horizon
        self.text_period = 2 * attend_text_every_n_blocks

    @staticmethod
    def _cross_block(block, hidden, temb, keys, values, bias):
        attn = block.attn1
        query = attn.to_q(block.norm1(hidden, temb))
        batch, length, inner = query.shape
        head_dim = inner // attn.heads

        def split(x):
            return x.view(batch, -1, attn.heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(split(query),
                                             split(keys),
                                             split(values),
                                             attn_mask=bias)
        out = attn.to_out[0](out.transpose(1, 2).reshape(batch, length, inner))
        hidden = hidden + out
        return hidden + block.ff(block.norm3(hidden))

    def forward(self, actions, timestep, state_features, keys, values,
                text_bias, image_bias, vel_strength, dt):
        category = torch.zeros(actions.shape[0],
                               dtype=torch.long,
                               device=actions.device)
        timesteps = timestep.expand(actions.shape[0])
        action_features = self.action_encoder(actions, timesteps, category)
        if self.position_embedding is not None:
            action_features = action_features + self.position_embedding.weight[:actions.shape[
                1]][None]
        hidden = torch.cat((state_features, action_features), dim=1)
        temb = self.dit.timestep_encoder(timesteps)
        cross_index = 0
        for index, block in enumerate(self.dit.transformer_blocks):
            if index % 2 == 1:
                hidden = block(hidden, temb=temb)
            else:
                bias = text_bias if index % self.text_period == 0 else image_bias
                hidden = self._cross_block(block, hidden, temb,
                                           keys[cross_index],
                                           values[cross_index], bias)
                cross_index += 1
        shift, scale = self.dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden = self.dit.norm_out(hidden) * (1 + scale[:, None]) + shift[:,
                                                                          None]
        velocity = self.action_decoder(self.dit.proj_out_2(hidden),
                                       category)[:, -self.action_horizon:]
        return actions + dt * velocity * vel_strength


def split_modules(head, config):
    return (VLPrep(head), StateEncoder(head),
            DenoiseStep(head, config.attend_text_every_n_blocks))


def run_split(vl_prep, state_encoder, denoise, features, image_mask,
              attention_mask, state, noise, steps, buckets):
    keys, values, text_bias, image_bias = vl_prep(features, image_mask,
                                                  attention_mask)
    state_features = state_encoder(state)
    actions = noise
    vel = torch.ones_like(actions)
    dt = torch.tensor(1.0 / steps)
    for step in range(steps):
        t = torch.tensor([int(step / steps * buckets)], dtype=torch.long)
        actions = denoise(actions, t, state_features, keys, values, text_bias,
                          image_bias, vel, dt)
    return actions


def check(head, config, index, features_path, input_ids_path, image_token_id):
    from transformers.feature_extraction_utils import BatchFeature

    features = torch.from_numpy(
        __import__("numpy").load(features_path))[None].float()
    input_ids = torch.from_numpy(
        __import__("numpy").load(input_ids_path))[None]
    image_mask = input_ids == image_token_id
    attention_mask = torch.ones_like(image_mask)
    torch.manual_seed(0)
    state = torch.randn(1, config.state_history_length, config.max_state_dim)

    with torch.no_grad():
        embodiment = torch.tensor([index])
        torch.manual_seed(1)
        reference = head.get_action(
            BatchFeature({
                "backbone_features": features.clone(),
                "image_mask": image_mask,
                "backbone_attention_mask": attention_mask
            }), BatchFeature({
                "state": state,
                "embodiment_id": embodiment
            }))["action_pred"]
        slice_embodiment(head, index)
        vl_prep, state_encoder, denoise = split_modules(head, config)
        torch.manual_seed(1)
        noise = torch.randn(1, config.action_horizon, config.max_action_dim)
        ours = run_split(vl_prep, state_encoder, denoise, features, image_mask,
                         attention_mask, state, noise,
                         config.num_inference_timesteps,
                         config.num_timestep_buckets)
    err = (ours - reference).abs().max().item()
    print(f"split vs official get_action: max|d| {err:.3e}, |ref| max "
          f"{reference.abs().max().item():.3f}")
    return err


def _export_onnx(module, args, out, name, **kwargs):
    """Export, then store each graph as <name>.onnx plus one <name>.onnx.data."""
    import onnx

    with tempfile.TemporaryDirectory() as scratch:
        tmp = os.path.join(scratch, f"{name}.onnx")
        torch.onnx.export(module,
                          args,
                          tmp,
                          opset_version=17,
                          dynamo=False,
                          **kwargs)
        model = onnx.load(tmp)
    onnx.save(model,
              os.path.join(out, f"{name}.onnx"),
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=f"{name}.onnx.data")


def export(head, config, index, out, max_tokens):
    slice_embodiment(head, index)
    vl_prep, state_encoder, denoise = split_modules(head, config)
    os.makedirs(out, exist_ok=True)
    seq = 145
    num_cross = len(cross_blocks(head))
    inner = head.model.inner_dim
    features = torch.randn(1, seq, config.backbone_embedding_dim)
    image_mask = torch.zeros(1, seq, dtype=torch.bool)
    image_mask[:, 4:132] = True
    attention_mask = torch.ones(1, seq, dtype=torch.bool)
    seq_axis = {1: "backbone_tokens"}
    with torch.no_grad():
        _export_onnx(
            vl_prep, (features, image_mask, attention_mask),
            out,
            "vl_prep",
            input_names=["backbone_features", "image_mask", "attention_mask"],
            output_names=[
                "cross_keys", "cross_values", "text_bias", "image_bias"
            ],
            dynamic_axes={
                "backbone_features": seq_axis,
                "image_mask": seq_axis,
                "attention_mask": seq_axis,
                "cross_keys": {
                    2: "backbone_tokens"
                },
                "cross_values": {
                    2: "backbone_tokens"
                },
                "text_bias": {
                    3: "backbone_tokens"
                },
                "image_bias": {
                    3: "backbone_tokens"
                }
            })
        state = torch.randn(1, config.state_history_length,
                            config.max_state_dim)
        _export_onnx(state_encoder, (state, ),
                     out,
                     "state_encoder",
                     input_names=["state"],
                     output_names=["state_features"])
        actions = torch.randn(1, config.action_horizon, config.max_action_dim)
        keys = torch.randn(num_cross, 1, seq, inner)
        bias = torch.zeros(1, 1, 1, seq)
        _export_onnx(
            denoise,
            (actions, torch.tensor([0]), state_encoder(state), keys,
             keys.clone(), bias, bias.clone(), torch.ones_like(actions),
             torch.tensor(1.0 / config.num_inference_timesteps)),
            out,
            "denoise_step",
            input_names=[
                "actions", "timestep", "state_features", "cross_keys",
                "cross_values", "text_bias", "image_bias", "vel_strength", "dt"
            ],
            output_names=["next_actions"],
            dynamic_axes={
                "cross_keys": {
                    2: "backbone_tokens"
                },
                "cross_values": {
                    2: "backbone_tokens"
                },
                "text_bias": {
                    3: "backbone_tokens"
                },
                "image_bias": {
                    3: "backbone_tokens"
                }
            })
    meta = {
        "action_horizon": config.action_horizon,
        "action_dim": config.max_action_dim,
        "state_dim": config.max_state_dim * config.state_history_length,
        "num_inference_timesteps": config.num_inference_timesteps,
        "num_timestep_buckets": config.num_timestep_buckets,
        "num_cross_blocks": num_cross,
        "cross_inner_dim": inner,
        "backbone_embedding_dim": config.backbone_embedding_dim,
        "max_backbone_tokens": max_tokens,
        "embodiment_index": index,
    }
    json.dump(meta, open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"exported vl_prep / state_encoder / denoise_step -> {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gr00t-src", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embodiment", default="new_embodiment")
    parser.add_argument("--out")
    parser.add_argument("--max-backbone-tokens", type=int, default=512)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--features", help="[tokens, 2048] .npy for --check")
    parser.add_argument("--input-ids", help="[tokens] .npy for --check")
    parser.add_argument("--image-token-id", type=int, default=151655)
    args = parser.parse_args()

    head, config = load_action_head(args.gr00t_src, args.checkpoint)
    index = embodiment_index(args.checkpoint, args.embodiment)
    if args.check:
        check(head, config, index, args.features, args.input_ids,
              args.image_token_id)
        head, config = load_action_head(args.gr00t_src, args.checkpoint)
    if args.out:
        export(head, config, index, args.out, args.max_backbone_tokens)


if __name__ == "__main__":
    main()
