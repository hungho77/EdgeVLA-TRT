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
"""RLDX-1's action model (MSAT flow matching) as one Euler step, from the official modules.

  step: x_t [1, horizon, 64], t and dt [1] (the flow time, 0 = noise, 1 = data), the 64 cognition features [1, 64, D],
        the normalized, zero-padded state [1, 1, 64] and a per-row velocity strength [1, horizon, 1] ->
        x_{t+dt} = x_t + dt * strength * v, and v. Strength 1 everywhere is the plain Euler step; real-time chunking
        starts the overlap rows from the previous chunk and scales their velocity (0 on the frozen rows, then a ramp),
        as GR00T's RTC does.

The embodiment is fixed at export (its slice of the category-specific encoders). ``mixed_precision`` keeps what
FP16 cannot hold in FP32: the single-stream blocks carry the projected cognition tokens with a few channels near
6.2e4 through their residual stream (FP16's limit is 65504), so that stream, the projection feeding it and every
LayerNorm run in FP32, with the blocks' GEMMs in FP16. MSAT's RoPE multiplies complex
numbers; the step rotates the interleaved pairs with real cos / sin tables of the same frequencies instead. The
position ids, which the model builds from constants (the double-stream blocks number the state and action tokens
from 1, the single-stream blocks from 2), are unchanged.
"""

import torch
from torch import nn


def apply_rotary_emb_real(xq, xk, freqs):
    """ops.apply_rotary_emb with freqs [B, N, D/2, 2] (cos, sin) in place of complex frequencies."""
    cos, sin = freqs[..., 0].unsqueeze(1), freqs[..., 1].unsqueeze(1)

    def rotate(x):
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        re, im = pairs[..., 0], pairs[..., 1]
        return torch.stack([re * cos - im * sin, re * sin + im * cos],
                           -1).flatten(3).type_as(x)

    return rotate(xq), rotate(xk)


def use_real_rope(action_model):
    """Swap the complex RoPE for apply_rotary_emb_real on every embedder and attention block."""
    from rldx.model.modules.action_model import blocks, ops

    blocks.apply_rotary_emb = apply_rotary_emb_real
    for module in action_model.modules():
        if isinstance(module, ops.RoPEEmbedder1D):
            for i in range(module.n_axes):
                freqs = getattr(module, f"freqs_cis_{i}")
                module.register_buffer(f"cos_sin_{i}",
                                       torch.stack([freqs.real, freqs.imag],
                                                   -1),
                                       persistent=False)

            def forward(ids, module=module):
                return torch.cat([
                    getattr(module, f"cos_sin_{i}")[ids[..., i]]
                    for i in range(ids.shape[-1])
                ], -2)

            module.forward = forward


def layer_norm_fp32(module):
    """nn.LayerNorm computed in FP32, returned in the input dtype."""

    def forward(x):
        y = torch.nn.functional.layer_norm(
            x.float(), module.normalized_shape,
            None if module.weight is None else module.weight.float(),
            None if module.bias is None else module.bias.float(), module.eps)
        return y.to(x.dtype)

    module.forward = forward


def single_block_forward(block, x, temb, pe=None, time_token=None, **kwargs):
    """SingleStreamBlock with identity modulation (a time token is present): the residual stream x stays in FP32."""
    from rldx.model.modules.action_model import blocks
    dtype = block.linear1.weight.dtype
    h = block.pre_norm(x.float()).to(dtype)
    if block.pos_embed is not None:
        h = block.pos_embed(h)
    mlp_in = 2 * int(block.hidden_size * block.mlp_ratio)
    qkv, mlp = torch.split(block.linear1(h), [3 * block.inner_dim, mlp_in],
                           dim=-1)
    q, k, v = (blocks._split_heads(t, block.num_heads)
               for t in qkv.chunk(3, dim=-1))
    batch, heads, length, dim = q.shape
    q = block.q_norm(q.reshape(batch * heads, length,
                               dim)).reshape(batch, heads, length, dim)
    k = block.k_norm(k.reshape(batch * heads, length,
                               dim)).reshape(batch, heads, length, dim)
    if block.use_rope and pe is not None:
        q, k = apply_rotary_emb_real(q, k, pe)
    attn = blocks._merge_heads(
        torch.nn.functional.scaled_dot_product_attention(q, k, v))
    gate, up = mlp.chunk(2, dim=-1)
    mlp_out = block.mlp_proj(torch.nn.functional.silu(gate) * up)
    out = block.post_norm(block.linear2(torch.cat([attn, mlp_out], -1)))
    return x.float() + out.float()


def mixed_precision(action_model):
    """FP32 where FP16 overflows (see the module docstring); everything else stays in the model dtype."""
    msat = action_model.model
    for module in action_model.modules():
        if isinstance(module, nn.LayerNorm):
            layer_norm_fp32(module)
    projection = msat.vl_proj_to_sa
    projection.forward = lambda x: torch.nn.functional.linear(
        x.float(), projection.weight.float(), projection.bias.float())
    count = len(msat.single_blocks)
    for i, block in enumerate(msat.single_blocks):
        assert block.use_swiglu, "only the SwiGLU single-stream block is supported"

        def forward(x, temb, *args, block=block, last=i == count - 1, **kw):
            y = single_block_forward(block, x, temb, *args, **kw)
            if not last:
                return y
            # Only the trailing state / action rows are read after the last block; the cognition rows, which may
            # exceed FP16, are zeroed before the cast.
            keep = torch.arange(y.shape[1], device=y.device) >= msat_vl_rows[0]
            return torch.where(keep[None, :, None], y, torch.zeros_like(y)).to(
                block.linear1.weight.dtype)

        block.forward = forward
    msat_vl_rows = [0]
    original = msat.forward

    def msat_forward(*args, encoder_hidden_states=None, **kwargs):
        msat_vl_rows[0] = encoder_hidden_states.shape[1]
        return original(*args,
                        encoder_hidden_states=encoder_hidden_states,
                        **kwargs)

    msat.forward = msat_forward


def build_action_model(checkpoint, dtype=torch.float32):
    import rldx_llm
    from rldx.model.core.rldx import RLDXActionModel
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    model = RLDXActionModel(config)
    state = rldx_llm.load_tensors(checkpoint, "action_model.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "rope" not in k and "freqs" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"action model weights: missing {missing}, unexpected {unexpected}"
        )
    use_real_rope(model)
    return model.to(dtype).eval(), config


class Step(nn.Module):

    def __init__(self, action_model, embodiment_id=0):
        super().__init__()
        self.model = action_model
        self.register_buffer("embodiment", torch.tensor([embodiment_id]))

    def forward(self, x, t, dt, cognition, state, strength):
        am = self.model
        horizon = x.shape[1]
        state_features = am.state_encoder(state, self.embodiment)
        features = am.action_encoder(x, t.expand(1, horizon), self.embodiment)
        features = features + am.position_embedding(
            torch.arange(horizon, device=x.device))[None]
        out = am.model(hidden_states=torch.cat([state_features, features], 1),
                       encoder_hidden_states=cognition,
                       timestep=t,
                       encoder_attention_mask=None,
                       physics_embs=None,
                       physics_attention_mask=None)
        out = out["action"] if isinstance(out, dict) else out
        velocity = am.action_decoder(out, self.embodiment)[:, -horizon:]
        return x + dt * strength * velocity, velocity
