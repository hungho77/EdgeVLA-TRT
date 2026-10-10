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
"""MolmoAct2's continuous-inference path as exportable modules built on LeRobot's own (lerobot.policies.molmoact2).

  vision:  SigLIP2 patches of every image [N, 729, 588] (FP32, [-1, 1]) -> the features added to the <im_patch>
           tokens [N * 196, D]: 25 ViT blocks, layers 24 and 18 concatenated, 2 x 2 attention pooling, SwiGLU
           projector. The pooling index table is fixed by the 378 x 378 crop and baked in.
  prefix:  a contiguous range of the language model's blocks: token ids and the visual features (first range) or the
           running hidden state (later ranges), a per-position image flag and RoPE cos / sin (FP32) -> the hidden
           state and, per block, K (after the qk norm and RoPE) and V as [S, kv_heads * head_dim]. Image tokens
           attend to every image token, text tokens causally, as _build_native_attention_bias does. The last block
           of the model only needs its K / V.
  context: every block's K / V -> the action expert's per-block cross-attention K / V (its shared context
           projections and norm, then each block's key norm), once per call.
  step:    one flow step (indexed by its step: the expert's modulations depend only on the time and are precomputed),
           x_{t+dt} = x_t + dt * strength * v with the padded action dims zeroed before and after;
           strength 1 everywhere is the plain Euler step, real-time chunking scales the overlap rows as GR00T does.

The checkpoint's projected image features reach 1.7e4 and stay in the residual stream, and the pooling attention's
unscaled scores reach 8e4: RMSNorm, the softmaxes and the pooling attention (the official float32_attention, kept on
its SDPA path: the eager path ignores the pooling mask) run in FP32. The ViT's and the language model's scaled scores
stay below 64, so their QK^T is FP16 unless fp32_attention is set.
"""

import torch
from torch import nn


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Vision(nn.Module):

    def __init__(self, vision_backbone, pooling_index, fp32_attention=False):
        super().__init__()
        self.vb = vision_backbone
        # The ViT's scaled scores stay below 64, so its attention can stay in FP16; the pooling attention's reach
        # 8e4 and keep the official FP32.
        for block in vision_backbone.image_vit.transformer.resblocks:
            block.attention.float32_attention = fp32_attention
        # The processor numbers each image's patches from 0; the images' features are concatenated.
        groups, patches = pooling_index.shape[0] // 2, 729
        offset = (torch.arange(pooling_index.shape[0]) // groups *
                  patches)[:, None]
        index = torch.where(pooling_index >= 0, pooling_index + offset,
                            pooling_index)
        self.register_buffer("index", index, persistent=False)

    def forward(self, patches):
        """patches [N, 729, 588] FP32 -> [N * 196, D]."""
        vb = self.vb
        images = torch.round((patches + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)
        images = (images / 255.0 * 2.0 - 1.0).to(
            vb.image_projector.w1.weight.dtype)
        features = vb.encode_image(images[None])
        dim = features.shape[-1]
        valid = self.index >= 0
        to_pool = features.reshape(-1, dim)[self.index.clamp(min=0)]
        to_pool = to_pool * valid[:, :, None].to(to_pool.dtype)
        denom = valid.float().sum(-1).clamp(min=1)
        query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(
            to_pool.dtype)
        pooled = vb.image_pooling_2d(query,
                                     to_pool,
                                     attn_mask=valid[:, None, None, :])
        return vb.image_projector(pooled.reshape(-1, pooled.shape[-1]))


def attention_kv(attn, x, cos, sin, mask, need_output=True, fp32=False):
    """MolmoAct2Attention returning the output and the cached K / V. ``fp32`` applies RoPE and computes QK^T in FP32;
    otherwise they stay in FP16 with the scale on Q (scaled scores stay below 64) and only the softmax is FP32."""
    batch, length = x.shape[:2]
    shape = (batch, length, -1, attn.head_dim)
    dtype = torch.float32 if fp32 else x.dtype
    q, k, v = attn.att_proj(x).split(attn.fused_dims, dim=-1)
    q = attn.q_norm(q.view(shape)).transpose(1, 2).to(dtype)
    k = attn.k_norm(k.view(shape)).transpose(1, 2).to(dtype)
    v = v.view(shape).transpose(1, 2)
    cos, sin = cos[None, None].to(dtype), sin[None, None].to(dtype)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    k_cache = k.to(v.dtype).transpose(1, 2).reshape(batch, length, -1)
    v_cache = v.transpose(1, 2).reshape(batch, length, -1)
    if not need_output:
        return None, k_cache, v_cache
    groups = attn.num_key_value_groups
    scores = torch.matmul(q * attn.head_dim**-0.5,
                          k.repeat_interleave(groups, 1).transpose(-1, -2))
    probs = (scores.float() + mask).softmax(-1).to(v.dtype)
    out = torch.matmul(probs, v.repeat_interleave(groups, 1))
    return attn.attn_out(out.transpose(1, 2).reshape(batch, length,
                                                     -1)), k_cache, v_cache


class Prefix(nn.Module):

    def __init__(self,
                 transformer,
                 first,
                 last,
                 final,
                 image_patch_id,
                 fp32_attention=False):
        super().__init__()
        self.fp32_attention = fp32_attention
        self.blocks = nn.ModuleList(transformer.blocks[first:last])
        self.final = final
        self.wte = transformer.wte if first == 0 else None
        self.image_patch_id = image_patch_id

    def forward(self, tokens_or_hidden, visual, image_flag, cos, sin):
        """First range: token ids [S] and visual [V, D]; later ranges: hidden [1, S, D] (visual unused)."""
        if self.wte is not None:
            ids = tokens_or_hidden
            x = self.wte(ids[None])
            is_patch = ids == self.image_patch_id
            row = (torch.cumsum(is_patch.long(), 0) - 1).clamp(min=0)
            x = x + torch.where(is_patch[:, None], visual[row].to(x.dtype),
                                torch.zeros_like(visual[:1]).to(x.dtype))[None]
        else:
            x = tokens_or_hidden
        length = x.shape[1]
        position = torch.arange(length, device=x.device)
        image = image_flag > 0.5
        allowed = (position[None, :] <= position[:, None]) | (image[:, None]
                                                              & image[None, :])
        mask = torch.where(allowed, torch.tensor(0.0, device=x.device),
                           torch.tensor(-1e9, device=x.device))[None, None]
        keys, values = [], []
        for i, block in enumerate(self.blocks):
            last = self.final and i == len(self.blocks) - 1
            h = block.attn_norm(x)
            out, k, v = attention_kv(block.self_attn,
                                     h,
                                     cos,
                                     sin,
                                     mask,
                                     need_output=not last,
                                     fp32=self.fp32_attention)
            keys.append(k[0])
            values.append(v[0])
            if last:
                break
            x = x + out
            x = x + block.mlp(block.ff_norm(x))
        return x, torch.stack(keys), torch.stack(values)


class Context(nn.Module):

    def __init__(self, expert):
        super().__init__()
        self.expert = expert

    def forward(self, keys, values):
        """keys / values [L, S, kv_dim] -> per-block cross-attention K / V [L, 1, S, heads, head_dim]."""
        contexts = self.expert._prepare_kv_context([
            (keys[i][None], values[i][None]) for i in range(keys.shape[0])
        ])
        return torch.stack([k for k, _ in contexts
                            ]), torch.stack([v for _, v in contexts])


def scaled_dot_product_attention_query_scaled(query,
                                              key,
                                              value,
                                              attn_mask=None,
                                              dropout_p=0.0,
                                              is_causal=False,
                                              scale=None,
                                              **kwargs):
    """F.scaled_dot_product_attention with the scale on the query. TensorRT 10.3 miscomputes the legacy exporter's
    SDPA lowering (the scale split across the query and the transposed key, the key split out of a fused QKV
    projection) when a norm and RoPE sit between the split and the transpose, which the trt_workarounds pass does
    not match. The expert's scores stay below 10, so the attention keeps the model dtype."""
    scale = query.shape[-1]**-0.5 if scale is None else scale
    scores = torch.matmul(query * scale, key.transpose(-1, -2))
    if attn_mask is not None:
        scores = scores + attn_mask
    return torch.matmul(scores.softmax(-1), value)


def separate_qkv(attn):
    """ActionExpertSelfAttention with its fused QKV projection as three GEMMs. TensorRT 10.3 miscomputes the fused
    projection viewed as [B, S, 3, H, D] and indexed per role (the self-attention output was off by more than its own
    size while onnxruntime matched PyTorch); separate projections are exact."""
    hidden = attn.hidden_size
    weight, bias = attn.qkv.weight, attn.qkv.bias
    for i, role in enumerate(("q", "k", "v")):
        linear = nn.Linear(hidden, hidden).to(weight.dtype)
        linear.weight.data = weight.data[i * hidden:(i + 1) * hidden].clone()
        linear.bias.data = bias.data[i * hidden:(i + 1) * hidden].clone()
        setattr(attn, f"{role}_proj_split", linear)

    def forward(x, *, attn_mask=None, is_causal=False, rope_cache=None):
        batch, length, _ = x.shape
        shape = (batch, length, attn.num_heads, attn.head_dim)
        q = attn.q_proj_split(x).view(shape).transpose(1, 2)
        k = attn.k_proj_split(x).view(shape).transpose(1, 2)
        v = attn.v_proj_split(x).view(shape).transpose(1, 2)
        q, k = attn._apply_qk_norm(q, k)
        if attn.rope is not None:
            q, k = attn.rope(q, k, rope_cache=rope_cache)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        return attn.out_proj(out.transpose(1, 2).reshape(batch, length, -1))

    attn.forward = forward


class Step(nn.Module):
    """One flow step with the expert's per-step modulations precomputed: they depend only on the step's time, so
    the graph indexes a table by the step instead of embedding the time."""

    def __init__(self, expert, action_dim, horizon, steps):
        super().__init__()
        self.expert = expert
        dims = torch.arange(expert.action_embed.in_features)
        self.register_buffer("keep", (dims < action_dim).float(),
                             persistent=False)
        self.horizon = horizon
        for block in expert.blocks:
            separate_qkv(block.self_attn)
        with torch.no_grad():
            cache = expert.prepare_modulation_cache([
                torch.full((1, ), k / steps, dtype=torch.float32)
                for k in range(steps)
            ])
        self.register_buffer("conditioning",
                             torch.stack([c.conditioning for c in cache]),
                             persistent=False)
        self.register_buffer(
            "blocks",
            torch.stack([
                torch.stack([torch.stack(m) for m in c.block_modulations])
                for c in cache
            ]),
            persistent=False)
        self.register_buffer(
            "final",
            torch.stack([torch.stack(c.final_modulation) for c in cache]),
            persistent=False)

    def forward(self, x, step, dt, context_k, context_v, encoder_mask,
                strength):
        """x [1, H, A]; step [1] int64; dt [1]; context K / V [L, 1, S, heads, head_dim]; encoder_mask [1, S]."""
        from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import (
            ActionExpertContext, ActionExpertStepModulation)
        expert = self.expert
        dtype = x.dtype
        blocks = self.blocks.index_select(0, step)[0]
        modulation = ActionExpertStepModulation(
            conditioning=self.conditioning.index_select(0, step)[0],
            block_modulations=[
                tuple(blocks[i, j] for j in range(blocks.shape[1]))
                for i in range(blocks.shape[0])
            ],
            final_modulation=tuple(self.final.index_select(0, step)[0]))
        context = ActionExpertContext(
            kv_contexts=[(context_k[i], context_v[i])
                         for i in range(context_k.shape[0])],
            cross_mask=expert._build_cross_attention_mask(
                encoder_mask, 1, dtype),
            self_mask=None,
            valid_action=None,
            rope_cache=expert.blocks[0].self_attn.rope.build_cache(
                seq_len=self.horizon, device=x.device, dtype=dtype))
        keep = self.keep.to(dtype)
        x = x * keep
        sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention_query_scaled
        try:
            velocity = expert.forward_with_context(
                x,
                modulation.conditioning,
                context=context,
                modulation=modulation) * keep
        finally:
            torch.nn.functional.scaled_dot_product_attention = sdpa
        return (x + dt * strength * velocity) * keep, velocity
