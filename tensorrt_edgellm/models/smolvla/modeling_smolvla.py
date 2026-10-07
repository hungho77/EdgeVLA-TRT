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
"""SmolVLA (LeRobot 0.6.1) as three export components: visual, prefix, denoise step.

Transcribed from LeRobot's ``SmolVLMWithExpertModel`` / ``VLAFlowMatching``. Every
attention is written with plain tensor ops and an explicit mask so the modules run eagerly
for parity against LeRobot and export to standard ONNX (no plugins, no TRT-native ops).

- ``SmolVLAVisual``: SigLIP tower + pixel-shuffle connector, scaled by sqrt(hidden) as
  ``embed_prefix`` does. pixel_values [N, 3, 512, 512] in [-1, 1] -> [N, 64, 960].
- ``SmolVLAPrefix``: the 16-layer VLM over [image features, language, state token] (pads
  dropped). Images and language attend bidirectionally; the state token attends to
  everything and nothing attends to it. Emits each self-attention layer's post-RoPE K/V
  and, for cross-attention layers, the K/V the expert re-projects from them.
- ``SmolVLADenoise``: one Euler step of the action expert over those K/V.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SmolVLAConfig:
    text_hidden: int = 960
    text_heads: int = 15
    text_kv_heads: int = 5
    head_dim: int = 64
    text_intermediate: int = 2560
    num_layers: int = 16
    rms_eps: float = 1e-5
    vocab_size: int = 49280
    vision_hidden: int = 768
    vision_layers: int = 12
    vision_heads: int = 12
    vision_intermediate: int = 3072
    vision_eps: float = 1e-6
    image_size: int = 512
    patch_size: int = 16
    scale_factor: int = 4
    expert_hidden: int = 720
    expert_intermediate: int = 2048
    self_attn_every_n_layers: int = 2
    max_state_dim: int = 32
    max_action_dim: int = 32
    chunk_size: int = 50
    min_period: float = 4e-3
    max_period: float = 4.0
    rope_max_wavelength: float = 10_000.0

    @property
    def image_tokens(self) -> int:
        return (self.image_size // self.patch_size)**2 // self.scale_factor**2

    def is_self_attn(self, layer: int) -> bool:
        return layer % self.self_attn_every_n_layers == 0


def apply_rope(x: torch.Tensor, positions: torch.Tensor,
               max_wavelength: float) -> torch.Tensor:
    """LeRobot's ``apply_rope``: x [B, L, H, D], positions [B, L], half-split, in FP32."""
    d_half = x.shape[-1] // 2
    dtype = x.dtype
    x = x.float()
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        d_half, dtype=torch.float32, device=x.device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].float() / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin],
                     dim=-1).to(dtype)


def attention(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              mask: "torch.Tensor | None",
              fp32_scores: bool = True) -> torch.Tensor:
    """LeRobot's eager attention: q [B, Lq, H, D], k/v [B, Lk, Hkv, D], mask [B or 1, Lq, Lk] bool.

    KV heads are repeated onto their query group; scale and softmax run in FP32. LeRobot
    upcasts Q and K too, which the text attention keeps; ``fp32_scores=False`` multiplies in
    the input precision (FP32 accumulation on tensor cores), as the SigLIP tower runs in HF.
    """
    groups = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(groups, dim=2)
    v = v.repeat_interleave(groups, dim=2)
    if fp32_scores:
        q, k = q.float(), k.float()
    scores = torch.matmul(q.transpose(1, 2),
                          k.transpose(1, 2).transpose(2, 3)).float()
    scores = scores * (q.shape[-1]**-0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None],
                                    torch.finfo(torch.float32).min)
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.matmul(probs, v.transpose(1, 2)).transpose(1, 2)
    return out.reshape(out.shape[0], out.shape[1], -1)


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class SwiGLU(nn.Module):

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------------------------
# Visual
# --------------------------------------------------------------------------------------------


class _VisionAttention(nn.Module):

    def __init__(self, cfg: SmolVLAConfig) -> None:
        super().__init__()
        d = cfg.vision_hidden
        self.heads = cfg.vision_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        shape = (b, n, self.heads, d // self.heads)
        out = attention(self.q_proj(x).view(shape),
                        self.k_proj(x).view(shape),
                        self.v_proj(x).view(shape),
                        None,
                        fp32_scores=False)
        return self.out_proj(out.to(x.dtype))


class _VisionLayer(nn.Module):

    def __init__(self, cfg: SmolVLAConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.vision_hidden, eps=cfg.vision_eps)
        self.self_attn = _VisionAttention(cfg)
        self.layer_norm2 = nn.LayerNorm(cfg.vision_hidden, eps=cfg.vision_eps)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(cfg.vision_hidden, cfg.vision_intermediate)
        self.mlp.fc2 = nn.Linear(cfg.vision_intermediate, cfg.vision_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp.fc2(gelu_tanh(self.mlp.fc1(self.layer_norm2(x))))


class SmolVLAVisual(nn.Module):

    def __init__(self, cfg: SmolVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patches = (cfg.image_size // cfg.patch_size)**2
        self.patch_embedding = nn.Conv2d(3,
                                         cfg.vision_hidden,
                                         cfg.patch_size,
                                         stride=cfg.patch_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(patches, cfg.vision_hidden))
        self.layers = nn.ModuleList(
            [_VisionLayer(cfg) for _ in range(cfg.vision_layers)])
        self.post_layernorm = nn.LayerNorm(cfg.vision_hidden,
                                           eps=cfg.vision_eps)
        self.connector = nn.Linear(cfg.vision_hidden * cfg.scale_factor**2,
                                   cfg.text_hidden,
                                   bias=False)

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        s = self.cfg.scale_factor
        b, seq, d = x.shape
        side = int(seq**0.5)
        x = x.view(b, side, side // s, d * s).permute(0, 2, 1, 3)
        x = x.reshape(b, side // s, side // s, d * s * s).permute(0, 2, 1, 3)
        return x.reshape(b, seq // (s * s), d * s * s)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.patch_embedding.weight.dtype)
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        x = x + self.position_embedding[None]
        for layer in self.layers:
            x = layer(x)
        x = self.connector(self.pixel_shuffle(self.post_layernorm(x)))
        return x * math.sqrt(self.cfg.text_hidden)


# --------------------------------------------------------------------------------------------
# Prefix (VLM)
# --------------------------------------------------------------------------------------------


class _TextAttention(nn.Module):

    def __init__(self, hidden: int, cfg: SmolVLAConfig, kv_in: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden,
                                cfg.text_heads * cfg.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(kv_in,
                                cfg.text_kv_heads * cfg.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(kv_in,
                                cfg.text_kv_heads * cfg.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(cfg.text_heads * cfg.head_dim,
                                hidden,
                                bias=False)


class _TextLayer(nn.Module):

    def __init__(self, hidden: int, intermediate: int, cfg: SmolVLAConfig,
                 kv_in: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, cfg.rms_eps)
        self.self_attn = _TextAttention(hidden, cfg, kv_in)
        self.post_attention_layernorm = RMSNorm(hidden, cfg.rms_eps)
        self.mlp = SwiGLU(hidden, intermediate)

    def finish(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.o_proj(attn.to(x.dtype))
        return x + self.mlp(self.post_attention_layernorm(x))


def _expert_layers(cfg: SmolVLAConfig) -> nn.ModuleList:
    kv = cfg.text_kv_heads * cfg.head_dim
    return nn.ModuleList([
        _TextLayer(cfg.expert_hidden, cfg.expert_intermediate, cfg,
                   cfg.expert_hidden if cfg.is_self_attn(i) else kv)
        for i in range(cfg.num_layers)
    ])


class SmolVLAPrefix(nn.Module):
    """[image features, token ids, state] -> per-layer K/V the denoise step reads.

    Outputs, in layer order: self-attention layers give the VLM's post-RoPE K and V;
    cross-attention layers give the expert's k_proj/v_proj of them. Each is [B, L, 5, 64].
    """

    def __init__(self, cfg: SmolVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.text_hidden)
        self.state_proj = nn.Linear(cfg.max_state_dim, cfg.text_hidden)
        self.layers = nn.ModuleList([
            _TextLayer(cfg.text_hidden, cfg.text_intermediate, cfg,
                       cfg.text_hidden) for _ in range(cfg.num_layers)
        ])
        # Only the cross-attention layers' k/v projections are used here.
        self.expert_layers = _expert_layers(cfg)

    def forward(self, image_features: torch.Tensor, token_ids: torch.Tensor,
                state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        cfg = self.cfg
        lang = self.embed_tokens(token_ids) * math.sqrt(cfg.text_hidden)
        dtype = self.state_proj.weight.dtype
        x = torch.cat([
            image_features.to(dtype), lang,
            self.state_proj(state.to(dtype))[:, None]
        ],
                      dim=1)
        b, length, _ = x.shape
        positions = torch.arange(length, device=x.device)[None].expand(b, -1)
        # The state token (last) is visible to itself only among the rows; it sees everything.
        cols = torch.arange(length, device=x.device)
        rows = cols[:, None]
        mask = ((cols[None, :] < length - 1) | (rows == length - 1))[None]

        outputs: List[torch.Tensor] = []
        kv_shape = (b, length, cfg.text_kv_heads, cfg.head_dim)
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q = apply_rope(
                attn.q_proj(h).view(b, length, cfg.text_heads, cfg.head_dim),
                positions, cfg.rope_max_wavelength)
            k = apply_rope(
                attn.k_proj(h).view(kv_shape), positions,
                cfg.rope_max_wavelength)
            v = attn.v_proj(h).view(kv_shape)
            if cfg.is_self_attn(i):
                outputs += [k, v]
            else:
                expert = self.expert_layers[i].self_attn
                flat = (b, length, cfg.text_kv_heads * cfg.head_dim)
                outputs += [
                    expert.k_proj(k.reshape(flat)).view(kv_shape),
                    expert.v_proj(v.reshape(flat)).view(kv_shape)
                ]
            if i + 1 < len(self.layers):
                x = layer.finish(x, attention(q, k, v, mask))
        return tuple(outputs)


# --------------------------------------------------------------------------------------------
# Denoise step (expert)
# --------------------------------------------------------------------------------------------


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, min_period: float,
                              max_period: float) -> torch.Tensor:
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=t.device)
    period = min_period * (max_period / min_period)**fraction
    angle = (2.0 * math.pi / period)[None, :] * t[:, None].float()
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)


class SmolVLADenoise(nn.Module):
    """One Euler step: (x_t, t, dt, prefix K/V) -> x_t + dt * v_t."""

    def __init__(self, cfg: SmolVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_in_proj = nn.Linear(cfg.max_action_dim, cfg.expert_hidden)
        self.action_out_proj = nn.Linear(cfg.expert_hidden, cfg.max_action_dim)
        self.action_time_mlp_in = nn.Linear(2 * cfg.expert_hidden,
                                            cfg.expert_hidden)
        self.action_time_mlp_out = nn.Linear(cfg.expert_hidden,
                                             cfg.expert_hidden)
        self.layers = _expert_layers(cfg)
        self.norm = RMSNorm(cfg.expert_hidden, cfg.rms_eps)

    def velocity(self, x_t: torch.Tensor, timestep: torch.Tensor,
                 prefix_kv: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        cfg = self.cfg
        b, horizon, _ = x_t.shape
        prefix_len = prefix_kv[0].shape[1]
        action = self.action_in_proj(x_t.to(self.action_in_proj.weight.dtype))
        time = sinusoidal_time_embedding(timestep, cfg.expert_hidden,
                                         cfg.min_period, cfg.max_period)
        time = time.to(action.dtype)[:, None, :].expand_as(action)
        x = self.action_time_mlp_out(
            F.silu(self.action_time_mlp_in(torch.cat([action, time], dim=2))))

        steps = torch.arange(horizon, device=x.device)[None].expand(b, -1)
        self_positions = steps + prefix_len
        index = torch.arange(horizon, device=x.device)
        causal = index[None, :] <= index[:, None]
        self_mask = torch.cat([
            torch.ones(horizon, prefix_len, dtype=torch.bool, device=x.device),
            causal
        ],
                              dim=1)[None]
        kv_shape = (b, horizon, cfg.text_kv_heads, cfg.head_dim)
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_proj(h).view(b, horizon, cfg.text_heads, cfg.head_dim)
            k_prefix, v_prefix = prefix_kv[2 * i], prefix_kv[2 * i + 1]
            if cfg.is_self_attn(i):
                q = apply_rope(q, self_positions, cfg.rope_max_wavelength)
                k = apply_rope(
                    attn.k_proj(h).view(kv_shape), self_positions,
                    cfg.rope_max_wavelength)
                v = attn.v_proj(h).view(kv_shape)
                out = attention(q, torch.cat([k_prefix, k], dim=1),
                                torch.cat([v_prefix, v], dim=1), self_mask)
            else:
                q = apply_rope(q, steps, cfg.rope_max_wavelength)
                out = attention(q, k_prefix, v_prefix, None)
            x = layer.finish(x, out)
        return self.action_out_proj(self.norm(x)).float()

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor,
                dt: torch.Tensor, *prefix_kv: torch.Tensor) -> torch.Tensor:
        return x_t + dt * self.velocity(x_t, timestep, prefix_kv)


# --------------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------------

_VWE = "model.vlm_with_expert."


def load_smolvla_weights(state: Dict[str, torch.Tensor], visual: SmolVLAVisual,
                         prefix: SmolVLAPrefix,
                         denoise: SmolVLADenoise) -> None:
    """Load a LeRobot SmolVLA ``model.safetensors`` into the three components (strict)."""
    vision = _VWE + "vlm.model.vision_model."
    mapped: Dict[nn.Module, Dict[str, torch.Tensor]] = {
        visual: {},
        prefix: {},
        denoise: {}
    }
    for key, tensor in state.items():
        tensor = tensor.float()
        if key.startswith(vision):
            name = key[len(vision):]
            name = name.replace("embeddings.patch_embedding.",
                                "patch_embedding.")
            name = name.replace("encoder.layers.", "layers.")
            if name == "embeddings.position_embedding.weight":
                name = "position_embedding"
            mapped[visual][name] = tensor
        elif key == _VWE + "vlm.model.connector.modality_projection.proj.weight":
            mapped[visual]["connector.weight"] = tensor
        elif key.startswith(_VWE + "vlm.model.text_model.layers."):
            mapped[prefix][key[len(_VWE + "vlm.model.text_model."):]] = tensor
        elif key == _VWE + "vlm.model.text_model.embed_tokens.weight":
            mapped[prefix]["embed_tokens.weight"] = tensor
        elif key.startswith("model.state_proj."):
            mapped[prefix][key[len("model."):]] = tensor
        elif key.startswith(_VWE + "lm_expert.layers."):
            name = key[len(_VWE + "lm_expert."):]
            mapped[denoise][name] = tensor
            if ".self_attn.k_proj." in name or ".self_attn.v_proj." in name:
                mapped[prefix]["expert_" + name] = tensor
        elif key == _VWE + "lm_expert.norm.weight":
            mapped[denoise]["norm.weight"] = tensor
        elif key.startswith("model.action_"):
            mapped[denoise][key[len("model."):]] = tensor
    for module, weights in mapped.items():
        if module is prefix:
            # Only the cross layers' expert k/v are loaded into the prefix; the rest stay unused.
            own = {
                k
                for k in module.state_dict()
                if not k.startswith("expert_layers.")
            }
            expert = {
                k: v
                for k, v in weights.items() if k.startswith("expert_")
            }
            for k in list(expert):
                layer = int(k.split(".")[1])
                if prefix.cfg.is_self_attn(layer):
                    weights.pop(k)
            missing = own - set(weights)
            if missing:
                raise KeyError(
                    f"SmolVLA prefix weights missing: {sorted(missing)[:5]}")
            module.load_state_dict(weights, strict=False)
        else:
            module.load_state_dict(weights, strict=True)
