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
"""RLDX-1's truncated Qwen3-VL language model as two plain-op graphs split at the video-token compression.

RLDX keeps the first ``select_layer`` (18) decoder layers. Its LayerWrapper compresses the video tokens on the input
of layer 4: every token from the first ``<|vision_start|>`` up to the current frame's first view is replaced by
their mean, placed at the first one's position, and the rest of the stack runs causally on the shorter sequence.
The deepstack visual features are added to the outputs of layers 0-2, before the compression.

  llm_a: token ids, visual features [N, D] and their deepstack features [3, N, D] scattered by a per-position
         visual index, the 64 learned cognition embeddings appended, MRoPE cos / sin [S, head_dim] (FP32) ->
         the hidden state after layer 3 [1, S, D]
  llm_b: that hidden state, a pooling weight per position (1 / count on the compressed tokens), the gather index
         of the compressed sequence (S standing for the pooled token) and its cos / sin -> the final-norm output of
         the last 64 positions (the cognition features the action model reads) [1, 64, D]

The host builds every index and table per instruction (the compression bounds move with the prompt length).
Attention applies RoPE and computes QK^T and the softmax in FP32 (the residual stream reaches 1.2e4 and the
query / key norms carry gains up to 34), the probabilities times V in the model dtype; RMSNorm is the official
module, which normalizes in FP32.
"""

import json
import os

import torch
from torch import nn

IMAGE_PAD = 151655
VISION_START = 151652


def load_tensors(checkpoint, prefix, dtype=None):
    """The checkpoint's tensors under ``prefix``, with the prefix stripped."""
    from safetensors import safe_open
    index = json.load(
        open(os.path.join(checkpoint,
                          "model.safetensors.index.json")))["weight_map"]
    out = {}
    for shard in sorted({s for k, s in index.items() if k.startswith(prefix)}):
        with safe_open(os.path.join(checkpoint, shard), "pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    t = f.get_tensor(key)
                    out[key[len(prefix):]] = t.to(dtype) if dtype else t
    return out


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def attention_forward(self, hidden_states, cos, sin, mask):
    """Qwen3VLTextAttention with RoPE, QK^T and the softmax in FP32."""
    batch, length = hidden_states.shape[:2]
    shape = (batch, length, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    q, k = q.float(), k.float()
    cos, sin = cos[None, None], sin[None, None]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    k = k.repeat_interleave(self.num_key_value_groups, dim=1)
    v = v.repeat_interleave(self.num_key_value_groups, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling + mask
    probs = scores.softmax(dim=-1).to(v.dtype)
    out = torch.matmul(probs, v).transpose(1, 2).reshape(batch, length, -1)
    return self.o_proj(out)


def layer_forward(layer, x, cos, sin, mask):
    """A decoder layer whose residual stream keeps x's dtype; the norms feed the projections in the weights' dtype."""
    dtype = layer.mlp.down_proj.weight.dtype
    h = attention_forward(layer.self_attn,
                          layer.input_layernorm(x).to(dtype), cos, sin, mask)
    x = x + h.to(x.dtype)
    return x + layer.mlp(layer.post_attention_layernorm(x).to(dtype)).to(
        x.dtype)


def causal_mask(length, like):
    """Additive FP32 causal mask [1, 1, L, L], built in the graph from the sequence length."""
    row = torch.arange(length, device=like.device)
    return torch.where(row[None, :] > row[:, None],
                       torch.tensor(-1e9, device=like.device),
                       torch.tensor(0.0, device=like.device))[None, None]


def build_layers(text_config, indices, weights, dtype):
    from rldx.model.modules.backbone.modeling_qwen3_vl import \
        Qwen3VLTextDecoderLayer
    layers = nn.ModuleList()
    for i in indices:
        layer = Qwen3VLTextDecoderLayer(text_config, i)
        state = {
            k[len(f"layers.{i}.layer."):]: v
            for k, v in weights.items() if k.startswith(f"layers.{i}.layer.")
        }
        layer.load_state_dict(state, strict=True)
        layers.append(layer.to(dtype))
    return layers


class LlmA(nn.Module):

    def __init__(self,
                 text_config,
                 weights,
                 cog_emb,
                 dtype,
                 residual_dtype=None):
        super().__init__()
        self.residual_dtype = residual_dtype or dtype
        self.embed = nn.Embedding.from_pretrained(
            weights["embed_tokens.weight"].to(dtype))
        self.cog = nn.Parameter(cog_emb.to(dtype), requires_grad=False)
        self.layers = build_layers(text_config, range(4), weights, dtype)

    def forward(self, input_ids, visual, deepstack, visual_index, cos, sin):
        """input_ids [S0], visual_index [S0] (-1: text) -> hidden [1, S0 + 64, D]."""
        is_visual = (visual_index >= 0)[:, None]
        rows = visual_index.clamp(min=0)
        x = torch.where(is_visual, visual[rows].to(self.cog.dtype),
                        self.embed(input_ids))
        x = torch.cat([x, self.cog], 0)[None].to(self.residual_dtype)
        pad = torch.zeros(self.cog.shape, dtype=x.dtype, device=x.device)
        mask = causal_mask(x.shape[1], x)
        for i, layer in enumerate(self.layers):
            x = layer_forward(layer, x, cos, sin, mask)
            if i < deepstack.shape[0]:
                extra = torch.where(is_visual, deepstack[i][rows].to(x.dtype),
                                    torch.zeros_like(visual[:1]).to(x.dtype))
                x = x + torch.cat([extra, pad], 0)[None]
        return x


class LlmB(nn.Module):

    def __init__(self,
                 text_config,
                 weights,
                 num_layers,
                 dtype,
                 residual_dtype=None):
        super().__init__()
        self.dtype = dtype
        self.residual_dtype = residual_dtype or dtype
        from rldx.model.modules.backbone.modeling_qwen3_vl import \
            Qwen3VLTextRMSNorm
        self.layers = build_layers(text_config, range(4, num_layers), weights,
                                   dtype)
        self.norm = Qwen3VLTextRMSNorm(text_config.hidden_size,
                                       eps=text_config.rms_norm_eps)
        self.norm.load_state_dict({"weight": weights["norm.weight"]})
        self.norm.to(dtype)

    def forward(self, hidden, pool, keep_index, cos, sin):
        """hidden [1, S, D], pool [S] (weights of the pooled token), keep_index [S'] (S = the pooled token)."""
        pooled = (hidden.float() * pool[None, :, None]).sum(1, keepdim=True)
        x = torch.cat([hidden, pooled.to(hidden.dtype)],
                      1)[:, keep_index].to(self.residual_dtype)
        mask = causal_mask(x.shape[1], x)
        for layer in self.layers:
            x = layer_forward(layer, x, cos, sin, mask)
        return self.norm(x[:, -64:]).to(self.dtype)


def text_config(vlm_dir):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(vlm_dir).text_config


def rope_tables(text_config, position_ids):
    """Official MRoPE cos / sin [S, head_dim] in FP32 for position ids [3, 1, S]."""
    from rldx.model.modules.backbone.modeling_qwen3_vl import \
        Qwen3VLTextRotaryEmbedding
    rotary = Qwen3VLTextRotaryEmbedding(text_config)
    cos, sin = rotary(torch.zeros(1, dtype=torch.float32), position_ids)
    return cos[0].float(), sin[0].float()


def position_ids(vlm_dir, input_ids, image_grid_thw, cognition_tokens=64):
    """The official get_rope_index over the prompt with the cognition placeholders appended."""
    from types import SimpleNamespace

    from rldx.model.modules.backbone.modeling_qwen3_vl import Qwen3VLModel
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(vlm_dir)
    ids = torch.cat([
        input_ids,
        torch.full((1, cognition_tokens), 248068, dtype=input_ids.dtype)
    ], 1)
    positions, _ = Qwen3VLModel.get_rope_index(SimpleNamespace(config=config),
                                               ids, image_grid_thw, None,
                                               torch.ones_like(ids))
    return positions


def compression(input_ids, num_views=2, cognition_tokens=64):
    """The LayerWrapper's bounds: from the first <|vision_start|> to the num_views-th from last one. Returns the
    pooling weights [S], the compressed gather index [S'] (S = the pooled token) and its rope gather index."""
    ids = input_ids[0].tolist()
    starts = [i for i, t in enumerate(ids) if t == VISION_START]
    begin, end = starts[0], starts[-num_views]
    total = len(ids) + cognition_tokens
    pool = torch.zeros(total)
    pool[begin:end] = 1.0 / (end - begin)
    keep = list(range(begin)) + [total] + list(range(end, total))
    rope = list(range(begin)) + [begin] + list(range(end, total))
    return pool, torch.tensor(keep), torch.tensor(rope)


def visual_index(input_ids):
    is_pad = input_ids[0] == IMAGE_PAD
    return torch.where(is_pad,
                       torch.cumsum(is_pad.long(), 0) - 1,
                       torch.full_like(input_ids[0], -1))
