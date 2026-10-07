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
"""W8A8 SmoothQuant for the pi0.5 prefix tower.

The prefix's projections become ``INT8SQLinear``: per-channel INT8 weights and a
per-tensor INT8 activation, after a SmoothQuant scale that moves activation outliers
into the weights. Projections reading the same input (q/k/v, gate/up) share one
smoothing vector, so a single smoothed activation feeds all of them.

Activation statistics are per input channel abs-max over the valid prefix rows of
calibration observations, keyed by module name (``model.layers.N.self_attn.q_proj``)
in a safetensors file; ``calibrate_pi05_prefix_int8.py`` writes it from openpi.
"""

import logging
from typing import Dict, List, Optional

import torch
from torch import nn

from ..linear import INT8SQLinear

logger = logging.getLogger(__name__)

# Projections that read the same activation, in prefix-layer order.
_SHARED_INPUT_GROUPS = (("self_attn", ("q_proj", "k_proj",
                                       "v_proj")), ("self_attn", ("o_proj", )),
                        ("mlp", ("gate_proj", "up_proj")), ("mlp",
                                                            ("down_proj", )))


def _int8_linear(linear: nn.Linear, smooth: torch.Tensor,
                 input_scale: float) -> INT8SQLinear:
    weight = linear.weight.detach().float() * smooth[None, :]
    weight_scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    out = INT8SQLinear(linear.in_features,
                       linear.out_features,
                       bias=linear.bias is not None)
    out.weight.copy_(
        torch.round(weight / weight_scale[:, None]).clamp(-127,
                                                          127).to(torch.int8))
    out.weight_scale.copy_(weight_scale)
    out.input_scale.fill_(input_scale)
    out.pre_quant_scale.copy_((1.0 / smooth).to(torch.float16))
    if linear.bias is not None:
        out.bias.copy_(linear.bias.detach().float())
    return out


def quantize_prefix_int8(model: nn.Module,
                         act_amax: Optional[Dict[str, torch.Tensor]],
                         alpha: float = 0.5,
                         keep_fp16: Optional[List[str]] = None) -> int:
    """Replace the prefix projections in place; returns how many were quantized.

    Without ``act_amax`` the scales are placeholders (no smoothing, unit activation
    scale): the graph has the INT8 structure but not usable accuracy.
    """
    keep_fp16 = set(keep_fp16 or [])
    count = 0
    for layer_name, layer in model.named_modules():
        if not layer_name.endswith(tuple(f"layers.{i}" for i in range(1024))):
            continue
        for parent_name, members in _SHARED_INPUT_GROUPS:
            parent = getattr(layer, parent_name)
            names = [f"{layer_name}.{parent_name}.{m}" for m in members]
            # An entry is a full module name or a projection name for every layer ("down_proj").
            if any(n in keep_fp16 or n.rsplit(".", 1)[-1] in keep_fp16
                   for n in names):
                continue
            linears = [getattr(parent, m) for m in members]
            if act_amax is None:
                smooth = torch.ones(linears[0].in_features)
                input_scale = 1.0
            else:
                x_max = act_amax[names[0]].float().clamp_min(1e-5)
                w_max = torch.stack([
                    l.weight.detach().float().abs().amax(dim=0)
                    for l in linears
                ]).amax(dim=0).clamp_min(1e-5)
                smooth = (x_max.pow(alpha) /
                          w_max.pow(1.0 - alpha)).clamp_min(1e-5)
                input_scale = float((x_max / smooth).max()) / 127.0
            for member, linear in zip(members, linears):
                setattr(parent, member,
                        _int8_linear(linear, smooth, input_scale))
                count += 1
    logger.info("pi0.5 prefix: %d projections in W8A8 (alpha %.2f%s)", count,
                alpha, "" if act_amax else ", placeholder scales")
    return count
