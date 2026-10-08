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
"""GR00T's Eagle backbones in plain PyTorch ops: N1.6's Eagle 3 (Eagle-Block2A-2B-v2) and N1.5's Eagle 2.5
(SigLIP 224, a linear connector, Qwen3 cut to 12 layers).

Two components with explicit shapes, so they run eagerly for parity and export to standard ONNX:

* ``EagleVisual``: the SigLIP / SigLIP2 tower on fixed-size images, one image per batch item (N1.6's
  packed FlashAttention keeps images apart the same way), N1.6's 2x2 pixel unshuffle and the
  ``mlp1`` connector. N1.6's config disables windowed attention and 2D RoPE, so the window split its
  code performs is a permutation it undoes, and is left out.
* ``EaglePrefix``: Qwen3 truncated to ``select_layer`` decoder layers over the prompt with the image
  features scattered into the image-context positions, causal, returning the final-norm hidden
  states GR00T's action head reads (``hidden_states[-1]`` of transformers 4.51).

GR00T N1.6 trains and serves the backbone after ``.to(torch.bfloat16)``, which also rounds Qwen3's
non-persistent RoPE ``inv_freq`` buffer; the rounded frequencies are part of that model, so they are
reproduced (``rope_inv_freq_bf16``). N1.5 loads with ``from_pretrained(torch_dtype=bfloat16)``,
which leaves the buffer in FP32.
"""

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..smolvla.modeling_smolvla import RMSNorm, SwiGLU, attention, gelu_tanh


@dataclass
class EagleConfig:
    vision_hidden: int = 1152
    vision_layers: int = 27
    vision_heads: int = 16
    vision_intermediate: int = 4304
    vision_eps: float = 1e-6
    patch_size: int = 14
    position_grid: int = 16
    image_height: int = 252
    image_width: int = 336
    text_hidden: int = 2048
    text_layers: int = 16
    text_heads: int = 16
    text_kv_heads: int = 8
    head_dim: int = 128
    text_intermediate: int = 6144
    text_eps: float = 1e-6
    rope_theta: float = 1e6
    rope_inv_freq_bf16: bool = True
    vocab_size: int = 151680
    image_token_id: int = 151669
    pixel_unshuffle: bool = True
    connector_layers: int = 2
    weight_root: str = "backbone.model."

    @property
    def grid(self):
        return self.image_height // self.patch_size, self.image_width // self.patch_size

    @property
    def image_tokens(self):
        h, w = self.grid
        return (h // 2) * (w // 2) if self.pixel_unshuffle else h * w


def gr00t_eagle_config(checkpoint: str, image_height: int,
                       image_width: int) -> EagleConfig:
    """The backbone of a GR00T N1.5 or N1.6 checkpoint, for images of the given size."""
    config = json.load(open(os.path.join(checkpoint, "config.json")))
    if config["model_type"] == "gr00t_n1_5":
        return EagleConfig(image_height=image_height,
                           image_width=image_width,
                           text_layers=int(
                               config["backbone_cfg"]["select_layer"]),
                           rope_inv_freq_bf16=False,
                           pixel_unshuffle=False,
                           connector_layers=1,
                           weight_root="backbone.eagle_model.")
    if config["model_type"] == "Gr00tN1d6":
        return EagleConfig(image_height=image_height,
                           image_width=image_width,
                           text_layers=int(config["select_layer"]))
    raise ValueError(
        f"no Eagle backbone for model_type {config['model_type']!r}")


class _VisionLayer(nn.Module):

    def __init__(self, cfg: EagleConfig) -> None:
        super().__init__()
        d = cfg.vision_hidden
        self.heads = cfg.vision_heads
        self.layer_norm1 = nn.LayerNorm(d, eps=cfg.vision_eps)
        self.layer_norm2 = nn.LayerNorm(d, eps=cfg.vision_eps)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, cfg.vision_intermediate)
        self.fc2 = nn.Linear(cfg.vision_intermediate, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.layer_norm1(x)
        split = lambda t: t.view(b, n, self.heads, d // self.heads)
        h = attention(split(self.q_proj(h)),
                      split(self.k_proj(h)),
                      split(self.v_proj(h)),
                      None,
                      fp32_scores=False)
        x = x + self.out_proj(h)
        return x + self.fc2(gelu_tanh(self.fc1(self.layer_norm2(x))))


class EagleVisual(nn.Module):
    """pixel_values [views, 3, H, W] in [-1, 1] -> image features [views, image_tokens, text_hidden]."""

    def __init__(self, cfg: EagleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.vision_hidden
        self.patch_embedding = nn.Linear(3 * cfg.patch_size**2, d)
        self.position_embedding = nn.Parameter(
            torch.zeros(cfg.position_grid**2, d))
        self.layers = nn.ModuleList(
            _VisionLayer(cfg) for _ in range(cfg.vision_layers))
        self.post_layernorm = nn.LayerNorm(d, eps=cfg.vision_eps)
        if cfg.connector_layers == 2:
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(4 * d), nn.Linear(4 * d, cfg.text_hidden),
                nn.GELU(), nn.Linear(cfg.text_hidden, cfg.text_hidden))
        else:
            self.mlp1 = nn.Sequential(nn.Linear(d, cfg.text_hidden))
        self.register_buffer("positions",
                             torch.zeros(cfg.grid[0] * cfg.grid[1], d),
                             persistent=False)

    def resize_positions(self) -> None:
        """SigLIP2's antialiased bilinear resize of the 16x16 table to this image grid (in FP32)."""
        g, d = self.cfg.position_grid, self.cfg.vision_hidden
        h, w = self.cfg.grid
        table = self.position_embedding.detach().float().reshape(
            g, g, d).permute(2, 0, 1)[None]
        table = F.interpolate(table,
                              size=(h, w),
                              mode="bilinear",
                              align_corners=False,
                              antialias=True)
        self.positions = table.reshape(d, h * w).transpose(0, 1).to(
            self.positions.dtype)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        v, p = pixel_values.shape[0], cfg.patch_size
        h, w = cfg.grid
        # Patch features in (row, col, channel) order, as the official convert_images_to_patches.
        patches = pixel_values.reshape(v, 3, h, p, w,
                                       p).permute(0, 2, 4, 3, 5, 1)
        x = self.patch_embedding(patches.reshape(v, h * w,
                                                 3 * p * p)) + self.positions
        for layer in self.layers:
            x = layer(x)
        x = self.post_layernorm(x)
        if cfg.pixel_unshuffle:
            # F.pixel_unshuffle(2) on [C, h, w]: channel c * 4 + dy * 2 + dx.
            x = x.reshape(v, h // 2, 2, w // 2, 2,
                          cfg.vision_hidden).permute(0, 1, 3, 5, 2, 4)
            x = x.reshape(v, (h // 2) * (w // 2), 4 * cfg.vision_hidden)
        return self.mlp1(x)


def rope_inv_freq(cfg: EagleConfig) -> torch.Tensor:
    """Qwen3's inv_freq, computed as transformers does."""
    d = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta
                      **(torch.arange(0, d, 2, dtype=torch.int64).float() / d))
    return inv_freq.bfloat16().float() if cfg.rope_inv_freq_bf16 else inv_freq


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """x [B, L, H, D]; cos/sin [L, D/2] in FP32; half-split rotation."""
    dtype, half = x.dtype, x.shape[-1] // 2
    x = x.float()
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos[None, :, None], sin[None, :, None]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin],
                     dim=-1).to(dtype)


class _TextLayer(nn.Module):

    def __init__(self, cfg: EagleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, hd = cfg.text_hidden, cfg.head_dim
        self.input_layernorm = RMSNorm(d, cfg.text_eps)
        self.post_attention_layernorm = RMSNorm(d, cfg.text_eps)
        self.q_proj = nn.Linear(d, cfg.text_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, cfg.text_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, cfg.text_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.text_heads * hd, d, bias=False)
        self.q_norm = RMSNorm(hd, cfg.text_eps)
        self.k_norm = RMSNorm(hd, cfg.text_eps)
        self.mlp = SwiGLU(d, cfg.text_intermediate)

    def forward(self, x, cos, sin, mask):
        cfg = self.cfg
        b, n, _ = x.shape
        h = self.input_layernorm(x)
        q = self.q_norm(
            self.q_proj(h).view(b, n, cfg.text_heads, cfg.head_dim))
        k = self.k_norm(
            self.k_proj(h).view(b, n, cfg.text_kv_heads, cfg.head_dim))
        v = self.v_proj(h).view(b, n, cfg.text_kv_heads, cfg.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        x = x + self.o_proj(attention(q, k, v, mask))
        return x + self.mlp(self.post_attention_layernorm(x))


class EaglePrefix(nn.Module):
    """token_ids [1, L] and image_features [1, N, text_hidden] (N = image-context tokens in the
    prompt, in prompt order) -> backbone features [1, L, text_hidden]."""

    def __init__(self, cfg: EagleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.text_hidden)
        self.layers = nn.ModuleList(
            _TextLayer(cfg) for _ in range(cfg.text_layers))
        self.norm = RMSNorm(cfg.text_hidden, cfg.text_eps)
        self.register_buffer("inv_freq", rope_inv_freq(cfg), persistent=False)

    def forward(self, token_ids: torch.Tensor,
                image_features: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(token_ids)
        is_image = token_ids == self.cfg.image_token_id
        slot = (torch.cumsum(is_image.to(torch.int64), dim=1) - 1).clamp(min=0)
        x = torch.where(is_image[..., None], image_features[0][slot[0]][None],
                        x)
        n = token_ids.shape[1]
        angles = torch.arange(n, device=token_ids.device).float(
        )[:, None] * self.inv_freq.float()[None]
        cos, sin = torch.cos(angles), torch.sin(angles)
        causal = torch.ones(n, n, dtype=torch.bool,
                            device=token_ids.device).tril()[None]
        for layer in self.layers:
            x = layer(x, cos, sin, causal)
        return self.norm(x)


def load_eagle_weights(state: dict, visual: EagleVisual,
                       prefix: EaglePrefix) -> None:
    """Load GR00T's backbone weights; the language model keeps its first layers."""
    root = visual.cfg.weight_root
    vision = root + "vision_model.vision_model."
    text = root + "language_model.model."
    v, t = {}, {}
    for key, value in state.items():
        if key.startswith(vision + "embeddings."):
            name = key[len(vision + "embeddings."):]
            if name.startswith("position_embedding"):
                v["position_embedding"] = value
            elif name == "patch_embedding.weight" and value.dim() == 4:
                # SigLIP's Conv2d patch embedding as a linear over (row, col, channel) patch features.
                v[name] = value.permute(0, 2, 3, 1).reshape(value.shape[0], -1)
            else:
                v[name] = value
        elif key.startswith(vision + "encoder.layers."):
            name = key[len(vision + "encoder."):]
            v[name.replace("self_attn.", "").replace("mlp.", "")] = value
        elif key.startswith(vision + "post_layernorm."):
            v[key[len(vision):]] = value
        elif key.startswith(root + "mlp1."):
            v[key[len(root):]] = value
        elif key.startswith(text + "layers."):
            name = key[len(text):]
            if int(name.split(".")[1]) < prefix.cfg.text_layers:
                t[name.replace("self_attn.", "")] = value
        elif key.startswith(text + "embed_tokens.") or key.startswith(text +
                                                                      "norm."):
            t[key[len(text):]] = value
    visual.load_state_dict(v, strict=True)
    prefix.load_state_dict(t, strict=True)
    visual.resize_positions()
