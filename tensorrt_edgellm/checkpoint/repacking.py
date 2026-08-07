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
"""
Post-load weight repacking for quantized linear layers.

Transforms checkpoint weight formats (AWQ column-packed int32, GPTQ row-packed
int32, ModelOpt uint8, ModelOpt NVFP4) into the per-plugin layout expected by
TensorRT (int4 GEMM plugin or NVFP4 MoE plugin).  All functions operate
in-place on ``module._buffers`` or return new tensors; they are called by
:func:`~loader.load_weights` after all checkpoint tensors have been assigned.
"""

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "repack_awq_to_plugin",
    "repack_gptq_to_plugin",
    "decode_modelopt_nvfp4",
    "repack_nvfp4_qwen3_moe_experts",
    "repack_nvfp4_nemotron_moe_experts",
]

# ---------------------------------------------------------------------------
# AWQ weight swizzle
# ---------------------------------------------------------------------------


def repack_awq_to_plugin(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Repack AWQ qweight from [in, out//8] int32 to [out//2, in] int8.

    AWQ packs 8 int4 nibbles per int32 along the output axis::

        int32 = (n7 << 28) | (n6 << 24) | ... | (n0 << 0)
        where n_k = output channel (8*col + k), value in [0, 15]

    The int4 GEMM kernel uses ``(nibble - 8) * scale`` while AWQ dequantizes as
    ``(nibble - qzero) * scale``. The zero-point cannot be folded into the nibbles:
    within one group ``nibble - qzero`` spans the 16 values ``[-qzero, 15-qzero]``,
    but the kernel's fixed zero-point pins the representable window to ``[-8, 7]``,
    and the two coincide only at ``qzero == 8``. Shifting and clamping (the previous
    approach) silently truncated whichever end fell outside — on
    Qwen2.5-3B-Instruct-AWQ that is 0.63% of weights, always the largest-magnitude
    ones in their group, moving a projection's output by 4-6%.

    Instead the zero-point is split out of the GEMM exactly::

        sum_i x_i (q_i - z_g) s_g  =  sum_i x_i (q_i - 8) s_g
                                    + sum_g (8 - z_g) s_g * sum_{i in g} x_i

    The first term is what the kernel already computes from the *unshifted*
    nibbles. The second is a per-group correction that :class:`AWQLinear` applies
    at run time from the group-wise sums of its input; this function returns its
    constant factor ``(8 - zeros) * scales``. It costs one reduction plus a
    ``[G, out]`` GEMV — about 0.8% of the layer's FLOPs — and is exact.

    Args:
        qweight: ``[in, out//8]`` int32, AWQ column-packed.
        qzeros:  ``[in//g, out//8]`` int32, AWQ column-packed zero-points.
        scales:  ``[in//g, out]`` dequantization scales. When given, the
            zero-point correction is returned; when ``None`` only the packed
            weights are returned (correction is ``None``), which is correct only
            for a symmetric checkpoint.

    Returns:
        ``(qweight_out, zero_correction)`` — ``[out//2, in]`` int8 in the plugin
        layout (K-block permute, even/odd shuffle within 8, N-row interleave,
        four nibbles per int16 viewed as two int8 rows), and ``[in//g, out]``
        float16 ``(8 - zeros) * scales`` or ``None``.
    """
    in_features, out_div8 = qweight.shape
    out_features = out_div8 * 8
    group_size = in_features // qzeros.shape[0]

    qw = qweight.cpu().to(torch.int32)
    qz = qzeros.cpu().to(torch.int32)

    # AutoAWQ packs 8 nibbles per int32 in non-sequential output-channel order.
    # Bit position k within each packed int32 stores the value for output channel
    # _AWQ_BIT_TO_CH[k] within that group of 8, derived from AutoAWQ's
    # AWQ_REVERSE_ORDER = [0,4,1,5,2,6,3,7] (packing_utils.py): the inverse
    # permutation gives the output channel encoded at each bit position.
    # Without this reorder, output channels within each group of 8 are scrambled.
    _AWQ_BIT_TO_CH = [0, 2, 4, 6, 1, 3, 5, 7]

    # Extract weight nibbles: nibbles[in, out] = uint4 value in [0, 15]
    nibbles = torch.zeros(in_features, out_features, dtype=torch.int32)
    for k in range(8):
        nibbles[:, _AWQ_BIT_TO_CH[k]::8] = (qw >> (4 * k)) & 0xF

    # Extract zero-point nibbles: zeros[in//g, out] = uint4 in [0, 15]
    zeros = torch.zeros(in_features // group_size,
                        out_features,
                        dtype=torch.int32)
    for k in range(8):
        zeros[:, _AWQ_BIT_TO_CH[k]::8] = (qz >> (4 * k)) & 0xF

    # The nibbles go to the kernel unshifted, so it computes (nibble - 8) * scale and
    # the (8 - qzero) * scale remainder becomes the run-time correction below.
    zero_correction = None
    if scales is not None:
        zero_correction = ((8 - zeros).to(torch.float32) *
                           scales.detach().cpu().to(torch.float32)).to(
                               torch.float16).to(scales.device)
    elif not bool(torch.all(zeros == 8)):
        logger.warning(
            "AWQ checkpoint has asymmetric zero-points but repack_awq_to_plugin "
            "was called without scales, so no zero-point correction can be built. "
            "The dequantized weights will be wrong.")

    # Transpose [in, out] -> [out, in] = [N, K] for pack_intweights
    nibbles_nk = nibbles.t().contiguous().numpy().astype(np.int16)  # [N, K]

    packed_int16 = _pack_intweights(nibbles_nk)  # [N//4, K] int16
    packed_int8 = packed_int16.view(np.int8).reshape(
        packed_int16.shape[0] * 2, packed_int16.shape[1])  # [N//2, K]

    return (torch.tensor(packed_int8,
                         dtype=torch.int8).to(qweight.device), zero_correction)


def _pack_intweights(unpacked_qweight: np.ndarray) -> np.ndarray:
    """Pack nibbles ``[N, K]`` int16 in ``[0, 15]`` to ``[N//4, K]`` int16 (int4 GEMM layout).

    Steps: permute within each 32-wide K block; even/odd reorder within each 8;
    interleave every four N rows across 64-wide K stripes; pack four nibbles per int16.
    """
    interleave = 4
    kstride = 64
    N, K = unpacked_qweight.shape

    # Step 1: Permute within K-blocks of 32
    # np.arange(32).reshape(4,4,2).transpose(1,0,2) -> [0,1,8,9,16,17,24,25,...]
    pk = unpacked_qweight.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    pk = pk.reshape(N, K // 32, 32)

    # Step 2: Within each group of 8, reorder [0,1,2,3,4,5,6,7] -> [0,2,4,6,1,3,5,7]
    pk = pk.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    pk = pk.reshape(N, K)

    # Step 3: Interleave every 4 rows (N dimension) across K-blocks of 64
    pk = pk.reshape(N // interleave, interleave, K // kstride, kstride)
    pk = pk.transpose(0, 2, 1, 3)  # [N//4, K//64, 4, 64]
    pk = pk.reshape(N // interleave, K // kstride, kstride, interleave)

    # Step 4: Pack 4 nibbles per int16 (little-endian nibble order)
    pk = (pk[..., 0]
          | (pk[..., 1] << 4)
          | (pk[..., 2] << 8)
          | (pk[..., 3] << 12))
    return pk.reshape(N // interleave, K).astype(np.int16)


def _gather_rows_by_gidx_order(
    weight: torch.Tensor,
    g_idx: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reorder rows of ``weight`` (K major) so channels with the same ``g_idx`` group are contiguous."""
    group_num = int(weight.shape[0] / group_size)
    gmax = int(torch.max(g_idx).item())
    assert group_num == gmax + 1, (
        f"Group number {group_num} != max(g_idx)+1 ({gmax + 1})")
    indices_list = []
    for i in range(group_num):
        indices = torch.nonzero(g_idx == i, as_tuple=False).squeeze(1)
        indices_list.append(indices)
    permute_idx = torch.cat(indices_list, dim=0)
    new_weight = weight.index_select(0, permute_idx)
    assert new_weight.shape[0] == weight.shape[0]
    return new_weight, permute_idx


# ---------------------------------------------------------------------------
# GPTQ weight swizzle
# ---------------------------------------------------------------------------


def repack_gptq_to_plugin(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: Optional[torch.Tensor] = None,
    zero_point_offset: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repack GPTQ ``qweight`` ``[in//8, out]`` int32 to plugin ``[out//2, in]`` int8.

    Unpacks eight nibbles per int32 along K, applies GPTQ zero-point offset,
    optionally reorders K rows by ``g_idx`` (``desc_act``), transposes to ``[N, K]``,
    then :func:`_pack_intweights`.

    Returns:
        ``(qweight_out, int4_act_perm)`` — permute activations with
        ``x.index_select(-1, int4_act_perm)`` before the int4 GEMM op when non-trivial.
    """
    in_div8, out_features = qweight.shape
    in_features = in_div8 * 8

    qw = qweight.cpu().to(torch.int32)
    qz = qzeros.cpu().to(torch.int32)

    # Some symmetric GPTQ checkpoints (e.g. Qwen3.5 int4) omit zero points
    # entirely, storing ``qzeros`` as an empty ``[num_groups, 0]`` tensor.
    # Treat these as symmetric quantization with the implicit midpoint zero (8).
    symmetric = qz.numel() == 0
    if symmetric:
        if qz.dim() >= 1 and qz.shape[0] > 0:
            num_groups = qz.shape[0]
        elif g_idx is not None and g_idx.numel() > 0:
            num_groups = int(g_idx.max().item()) + 1
        else:
            num_groups = 1
    else:
        num_groups = qz.shape[0]
    group_size = in_features // num_groups

    # Extract weight nibbles: nibbles[in, out] = uint4 value in [0, 15]
    # GPTQ row-packs: bit k of column `in` is in row `in//8`, bit position 4*k
    nibbles = torch.zeros(in_features, out_features, dtype=torch.int32)
    for k in range(8):
        nibbles[k::8, :] = (qw >> (4 * k)) & 0xF

    # Extract zero-point nibbles: zeros[group, out] = uint4 in [0, 15]
    # qzeros is [num_groups, out//8] -- same column packing as AWQ qzeros
    if symmetric:
        # No stored zeros: actual_zero is the 4-bit midpoint 8, so stored_zero =
        # 8 - zero_point_offset makes the offset adjustment below a no-op.
        zeros = torch.full((num_groups, out_features),
                           8 - int(zero_point_offset),
                           dtype=torch.int32)
    else:
        zeros = torch.zeros(num_groups, out_features, dtype=torch.int32)
        for k in range(8):
            zeros[:, k::8] = (qz >> (4 * k)) & 0xF

    if g_idx is None:
        g_idx_t = torch.arange(in_features, dtype=torch.int32) // group_size
    else:
        g_idx_t = g_idx.cpu().to(torch.int32)
    # Expand zeros from [num_groups, out] -> [in, out] using per-channel group ids.
    zeros_expanded = zeros[g_idx_t.to(torch.int64)]  # [in, out]

    # GPTQ checkpoints differ on whether qzeros stores zero or zero-1.
    # actual_zero = stored_zero + zero_point_offset.
    # Adjust nibbles: kernel does (nibble - 8) * scale; GPTQ does (nibble - actual_zero) * scale
    # -> repacked = nibble - (stored_zero + zero_point_offset) + 8
    nibbles = (nibbles - zeros_expanded - int(zero_point_offset) + 8).clamp(
        0, 15)

    # Gather K rows by group (identity order when ``g_idx`` is sequential).
    nibbles, permute_idx = _gather_rows_by_gidx_order(nibbles, g_idx_t,
                                                      group_size)

    # Transpose [in, out] -> [out, in] = [N, K] for pack_intweights
    nibbles_nk = nibbles.t().contiguous().numpy().astype(np.int16)
    packed_int16 = _pack_intweights(nibbles_nk)
    packed_int8 = packed_int16.view(np.int8).reshape(packed_int16.shape[0] * 2,
                                                     packed_int16.shape[1])

    qw_out = torch.tensor(packed_int8, dtype=torch.int8).to(qweight.device)
    perm = permute_idx.to(torch.int64)
    return qw_out, perm


# ---------------------------------------------------------------------------
# Post-load cast / format fixups (called by load_weights)
# ---------------------------------------------------------------------------


def apply_all_repacking(model: nn.Module) -> None:
    """Apply all quantization repacking passes after checkpoint load.

    MoE expert stacking runs FIRST because it needs the original GPTQ int32
    weights (before regular repacking converts them to the swizzled plugin
    format).  After stacking, the per-expert GPTQLinear modules have their
    qweight set to None so ``_repack_gptq_weights`` skips them.
    """
    _stack_moe_experts(model)
    _repack_awq_weights(model)
    _repack_gptq_weights(model)
    _cast_modelopt_awq_prepacked(model)
    _cast_fp8_linear_scales(model)
    _cast_nvfp4_weights(model)


def _cast_modelopt_awq_prepacked(model: nn.Module) -> None:
    """Post-process W4A16 prepacked AWQ linear buffers after load.

    1. Unpack ``[N//2, K] uint8`` (two nibbles per byte) to nibbles, apply
       :func:`_pack_intweights`, store ``[N//2, K] int8``.
    2. Cast optional ``pre_quant_scale`` to float16 so forward() Mul stays in fp16.
    3. Transpose scales to ``[K//g, N]`` float16 for the int4 GEMM custom op.
    """
    from ..models.linear import ModelOptAWQPrepackedLinear  # local import
    for module in model.modules():
        if isinstance(module, ModelOptAWQPrepackedLinear):
            # 1. Repack weight: ModelOpt uint8[N//2, K] -> swizzled int8[N//2, K]
            w = module._buffers.get("weight")
            if w is not None and w.dtype == torch.uint8:
                w_cpu = w.cpu()
                N_half, K = w_cpu.shape
                N = N_half * 2
                # Unpack 2 nibbles per byte: low nibble -> even N rows, high -> odd N rows
                # ModelOpt pack_int4_in_uint8 stores weights using two's complement masking:
                # s in [-8,7] -> u = s & 0xF (so s=-8 -> u=8, s=0 -> u=0, s=7 -> u=7)
                # The plugin kernel uses (nibble - 8) * scale, so nibble must be s+8 in [0,15].
                # Convert: plugin_nibble = (u + 8) % 16
                w_i16 = w_cpu.to(torch.int16)
                nibbles = torch.zeros(N, K, dtype=torch.int16)
                nibbles[0::2] = w_i16 & 0xF  # even N channels = low nibble
                nibbles[1::2] = (
                    w_i16 >> 4) & 0xF  # odd N channels = high nibble
                nibbles = (nibbles +
                           8) % 16  # two's complement -> plugin convention
                nibbles_np = nibbles.numpy().astype(np.int16)
                packed_int16 = _pack_intweights(nibbles_np)  # [N//4, K] int16
                packed_int8 = packed_int16.view(np.int8).reshape(
                    packed_int16.shape[0] * 2, packed_int16.shape[1])
                module._buffers["weight"] = torch.tensor(packed_int8,
                                                         dtype=torch.int8).to(
                                                             w.device)

            sc = module._buffers.get("weight_scale")
            if sc is None:
                continue

            # 2. Cast pre_quant_scale to float16 so forward() Mul stays in fp16.
            # pre_quant_scale is an AWQ activation smoothing scale applied as
            # x_smooth = x * pqs before the GEMM (matches the reference pipeline
            # where DQ+MatMul patterns include a leading Mul(x, pqs) node).
            pqs = module._buffers.get("pre_quant_scale")
            sc_f32 = sc.to(torch.float32)  # work in fp32 for precision
            if pqs is not None and pqs.dtype != torch.float16:
                module._buffers["pre_quant_scale"] = pqs.to(torch.float16)

            # 3. Transpose [N, K//g] -> [K//g, N] and cast to float16
            module._buffers["weight_scale"] = sc_f32.t().contiguous().to(
                torch.float16)


def _cast_fp8_linear_scales(model: nn.Module) -> None:
    """Cast FP8Linear ``input_scale`` / ``weight_scale`` to float16 if needed."""
    from ..models.linear import FP8Linear  # local import to avoid circular dep
    for module in model.modules():
        if not isinstance(module, FP8Linear):
            continue
        for name in ("weight_scale", "input_scale"):
            t = module._buffers.get(name)
            if t is None or t.dtype == torch.float16:
                continue
            module._buffers[name] = t.to(torch.float16)


def _cast_nvfp4_weights(model: nn.Module) -> None:
    """View-cast NVFP4 weight buffers from uint8 to int8 in-place.

    Packed FP4 nibbles have the same bit pattern in both types.
    Some ONNX importers mishandle UINT8 weight initializers for block DQ; int8 works.
    """
    from ..models.linear import \
        is_nvfp4_linear  # local import to avoid circular dep
    for module in model.modules():
        if is_nvfp4_linear(module):
            w = module._buffers.get("weight")
            if w is not None and w.dtype == torch.uint8:
                module._buffers["weight"] = w.view(torch.int8)


def _repack_awq_weights(model: nn.Module) -> None:
    """Swizzle ``AWQLinear.qweight`` after load and build the zero-point correction.

    Scales should already be ``[K//g, N]``; cast to float16 if needed. The
    zero-point is not folded into the nibbles (see :func:`repack_awq_to_plugin`);
    it lands in ``zero_correction``, which :class:`AWQLinear` applies at run time.
    """
    from ..models.linear import AWQLinear  # local import to avoid circular dep
    for module in model.modules():
        if isinstance(module, AWQLinear):
            sc = module._buffers.get("scales")
            if sc is not None and sc.dtype != torch.float16:
                sc = sc.to(torch.float16)
                module._buffers["scales"] = sc
            qw = module._buffers.get("qweight")
            qz = module._buffers.get("qzeros")
            if qw is not None and qw.dtype == torch.int32 and qz is not None:
                packed, correction = repack_awq_to_plugin(qw, qz, sc)
                module._buffers["qweight"] = packed
                module._buffers["zero_correction"] = correction
                logger.debug("Repacked AWQ qweight: %s -> %s", list(qw.shape),
                             list(packed.shape))


def _repack_gptq_weights(model: nn.Module) -> None:
    """Swizzle ``GPTQLinear.qweight`` after load; set ``int4_act_perm`` for ``desc_act``."""
    from ..models.linear import \
        GPTQLinear  # local import to avoid circular dep
    for module in model.modules():
        if isinstance(module, GPTQLinear):
            qw = module._buffers.get("qweight")
            qz = module._buffers.get("qzeros")
            if qw is not None and qw.dtype == torch.int32 and qz is not None:
                g_idx_buf = module._buffers.get("g_idx")
                packed, perm = repack_gptq_to_plugin(
                    qw, qz, g_idx_buf, getattr(module, "zero_point_offset", 1))
                module._buffers["qweight"] = packed
                module._buffers["int4_act_perm"] = perm
                logger.debug("Repacked GPTQ qweight: %s -> %s", list(qw.shape),
                             list(packed.shape))
            sc = module._buffers.get("scales")
            if sc is not None and sc.dtype != torch.float16:
                module._buffers["scales"] = sc.to(torch.float16)
    logger.info("Repacked GPTQ weights")


def _stack_moe_experts(model: nn.Module) -> None:
    """Stack per-expert weights into the layout required by the active MoE plugin.

    Walks every ``nn.Module`` in *model* and invokes ``_prepare_moe_weights``
    on every block that defines it. Each block decides its own packing path
    (Marlin for ``Int4MoePlugin``; CuTeDSL 6D MMA for
    ``Nvfp4MoePlugin``); this helper is backend-agnostic.

    Must run BEFORE ``_repack_gptq_weights`` because the GPTQ path needs
    the original int32-packed weights.  After extracting, per-expert
    qweight buffers are set to ``None`` so the regular GPTQ repacking
    skips them.
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "_prepare_moe_weights"):
            module._prepare_moe_weights()
            count += 1
    if count:
        logger.info("Stacked expert weights for %d MoE block(s)", count)


# ---------------------------------------------------------------------------
# Marlin INT4 repacking for MoE experts
# ---------------------------------------------------------------------------
# Adapted from tensorrt_edgellm/llm_models/layers/int4_moe_plugin.py.
# These functions convert GPTQ int32-packed weights → Marlin layout consumed
# by trt_edgellm::Int4MoePlugin.
# ---------------------------------------------------------------------------


def _unpack_int4_gptq(qweight: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ ``[K//8, N]`` int32 → ``[K, N]`` int16 nibbles."""
    pack_factor = 8
    wf = torch.tensor(list(range(0, 32, 4)),
                      dtype=torch.int32).unsqueeze(0).to(qweight.device)
    weight = torch.bitwise_and(
        torch.bitwise_right_shift(
            qweight.unsqueeze(1).expand(-1, pack_factor, -1),
            wf.unsqueeze(-1).to(qweight.device)).to(torch.int16), 15)
    return weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])


def _unpack_qzeros_moe(qzeros: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ qzeros ``[num_groups, N//8]`` → ``[num_groups, N]``."""
    device = qzeros.device
    wf = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28],
                      dtype=torch.int64,
                      device=device).view(1, 1, -1)
    z = qzeros.unsqueeze(2).expand(-1, -1, 8).to(torch.int64)
    return torch.bitwise_and(torch.bitwise_right_shift(z, wf),
                             15).reshape(qzeros.shape[0], -1)


def _extract_gptq_for_marlin(
    proj: nn.Module,
    group_size: int,
    zero_point_offset: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(weights [N, K] int16, scales [N, num_groups] fp16)`` from a
    GPTQ linear module, remapping zero-points so Marlin's ``(q - 8) * scale``
    equals GPTQ's ``(q - zero) * scale``.

    GPTQ checkpoints differ on whether qzeros stores ``zero_point`` or
    ``zero_point - 1``.  The adjustment is therefore::

        q_marlin = q - (stored_zero + zero_point_offset) + 8
    """
    unpacked = _unpack_int4_gptq(proj.qweight)  # [K, N]

    # Symmetric GPTQ checkpoints may omit zero points (``qzeros`` is ``None`` or
    # an empty ``[num_groups, 0]`` tensor); the implicit midpoint zero (8)
    # already matches Marlin's ``(q - 8) * scale``, so no remapping is needed.
    qzeros = getattr(proj, "qzeros", None)
    if qzeros is not None and qzeros.numel() > 0:
        zeros = _unpack_qzeros_moe(qzeros)  # [num_groups, N]
        K, N = unpacked.shape
        group_ids = torch.arange(K, device=unpacked.device) // group_size
        zeros_expanded = zeros[group_ids.clamp(max=zeros.shape[0] - 1)]
        # actual_zero = stored_zero + zero_point_offset
        unpacked = torch.clamp(
            unpacked.to(torch.int32) - zeros_expanded.to(torch.int32) -
            int(zero_point_offset) + 8, 0, 15).to(torch.int16)

    weights = unpacked.transpose(0, 1).contiguous()  # [N, K]
    scales = proj.scales.data.to(torch.float16).transpose(0, 1).contiguous()
    return weights, scales


# Pre-computed Marlin tensor core layout indices (from int4_moe_plugin.py).
_MARLIN_PACK_IDX = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int32)

# fmt: off
_MARLIN_OUT_IDX = np.array([
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
    1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
    65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125,
    2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
    66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126,
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
    67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127
], dtype=np.int32)

_ROW_PATTERN = np.array([
    [0, 1, 8, 9, 0, 1, 8, 9], [2, 3, 10, 11, 2, 3, 10, 11],
    [4, 5, 12, 13, 4, 5, 12, 13], [6, 7, 14, 15, 6, 7, 14, 15]
], dtype=np.int32)
_MARLIN_ROW_IDX = np.tile(_ROW_PATTERN, (32, 1))

_MARLIN_COL_IDX = np.array([
    [0,0,0,0,8,8,8,8],[0,0,0,0,8,8,8,8],[0,0,0,0,8,8,8,8],[0,0,0,0,8,8,8,8],
    [1,1,1,1,9,9,9,9],[1,1,1,1,9,9,9,9],[1,1,1,1,9,9,9,9],[1,1,1,1,9,9,9,9],
    [2,2,2,2,10,10,10,10],[2,2,2,2,10,10,10,10],[2,2,2,2,10,10,10,10],[2,2,2,2,10,10,10,10],
    [3,3,3,3,11,11,11,11],[3,3,3,3,11,11,11,11],[3,3,3,3,11,11,11,11],[3,3,3,3,11,11,11,11],
    [4,4,4,4,12,12,12,12],[4,4,4,4,12,12,12,12],[4,4,4,4,12,12,12,12],[4,4,4,4,12,12,12,12],
    [5,5,5,5,13,13,13,13],[5,5,5,5,13,13,13,13],[5,5,5,5,13,13,13,13],[5,5,5,5,13,13,13,13],
    [6,6,6,6,14,14,14,14],[6,6,6,6,14,14,14,14],[6,6,6,6,14,14,14,14],[6,6,6,6,14,14,14,14],
    [7,7,7,7,15,15,15,15],[7,7,7,7,15,15,15,15],[7,7,7,7,15,15,15,15],[7,7,7,7,15,15,15,15],
    [16,16,16,16,24,24,24,24],[16,16,16,16,24,24,24,24],[16,16,16,16,24,24,24,24],[16,16,16,16,24,24,24,24],
    [17,17,17,17,25,25,25,25],[17,17,17,17,25,25,25,25],[17,17,17,17,25,25,25,25],[17,17,17,17,25,25,25,25],
    [18,18,18,18,26,26,26,26],[18,18,18,18,26,26,26,26],[18,18,18,18,26,26,26,26],[18,18,18,18,26,26,26,26],
    [19,19,19,19,27,27,27,27],[19,19,19,19,27,27,27,27],[19,19,19,19,27,27,27,27],[19,19,19,19,27,27,27,27],
    [20,20,20,20,28,28,28,28],[20,20,20,20,28,28,28,28],[20,20,20,20,28,28,28,28],[20,20,20,20,28,28,28,28],
    [21,21,21,21,29,29,29,29],[21,21,21,21,29,29,29,29],[21,21,21,21,29,29,29,29],[21,21,21,21,29,29,29,29],
    [22,22,22,22,30,30,30,30],[22,22,22,22,30,30,30,30],[22,22,22,22,30,30,30,30],[22,22,22,22,30,30,30,30],
    [23,23,23,23,31,31,31,31],[23,23,23,23,31,31,31,31],[23,23,23,23,31,31,31,31],[23,23,23,23,31,31,31,31],
    [32,32,32,32,40,40,40,40],[32,32,32,32,40,40,40,40],[32,32,32,32,40,40,40,40],[32,32,32,32,40,40,40,40],
    [33,33,33,33,41,41,41,41],[33,33,33,33,41,41,41,41],[33,33,33,33,41,41,41,41],[33,33,33,33,41,41,41,41],
    [34,34,34,34,42,42,42,42],[34,34,34,34,42,42,42,42],[34,34,34,34,42,42,42,42],[34,34,34,34,42,42,42,42],
    [35,35,35,35,43,43,43,43],[35,35,35,35,43,43,43,43],[35,35,35,35,43,43,43,43],[35,35,35,35,43,43,43,43],
    [36,36,36,36,44,44,44,44],[36,36,36,36,44,44,44,44],[36,36,36,36,44,44,44,44],[36,36,36,36,44,44,44,44],
    [37,37,37,37,45,45,45,45],[37,37,37,37,45,45,45,45],[37,37,37,37,45,45,45,45],[37,37,37,37,45,45,45,45],
    [38,38,38,38,46,46,46,46],[38,38,38,38,46,46,46,46],[38,38,38,38,46,46,46,46],[38,38,38,38,46,46,46,46],
    [39,39,39,39,47,47,47,47],[39,39,39,39,47,47,47,47],[39,39,39,39,47,47,47,47],[39,39,39,39,47,47,47,47],
    [48,48,48,48,56,56,56,56],[48,48,48,48,56,56,56,56],[48,48,48,48,56,56,56,56],[48,48,48,48,56,56,56,56],
    [49,49,49,49,57,57,57,57],[49,49,49,49,57,57,57,57],[49,49,49,49,57,57,57,57],[49,49,49,49,57,57,57,57],
    [50,50,50,50,58,58,58,58],[50,50,50,50,58,58,58,58],[50,50,50,50,58,58,58,58],[50,50,50,50,58,58,58,58],
    [51,51,51,51,59,59,59,59],[51,51,51,51,59,59,59,59],[51,51,51,51,59,59,59,59],[51,51,51,51,59,59,59,59],
    [52,52,52,52,60,60,60,60],[52,52,52,52,60,60,60,60],[52,52,52,52,60,60,60,60],[52,52,52,52,60,60,60,60],
    [53,53,53,53,61,61,61,61],[53,53,53,53,61,61,61,61],[53,53,53,53,61,61,61,61],[53,53,53,53,61,61,61,61],
    [54,54,54,54,62,62,62,62],[54,54,54,54,62,62,62,62],[54,54,54,54,62,62,62,62],[54,54,54,54,62,62,62,62],
    [55,55,55,55,63,63,63,63],[55,55,55,55,63,63,63,63],[55,55,55,55,63,63,63,63],[55,55,55,55,63,63,63,63],
], dtype=np.int32)
# fmt: on


def _marlin_permute_scales(s, size_k, size_n, group_size):
    """Permute scale columns for Marlin kernel shared-memory read pattern."""
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def pack_int4_awq_marlin(
    weights_q: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack INT4 ``[E, N, K]`` weights + ``[E, N, num_groups]`` scales to Marlin.

    Returns ``(weights_marlin [E, K//16, 2*N] int32,
               scales_marlin  [E, num_groups, N] fp16)``.
    """
    num_experts, N, K = weights_q.shape
    device = weights_q.device
    weights_marlin_list = []

    for expert_id in range(num_experts):
        w_np = weights_q[expert_id].transpose(
            0, 1).contiguous().cpu().numpy().astype(np.uint32)  # [K, N]

        k_tiles, n_tiles = K // 16, N // 64
        tiles = w_np.reshape(k_tiles, 16, n_tiles, 64).transpose(0, 2, 1, 3)
        gathered = tiles[:, :, _MARLIN_ROW_IDX,
                         _MARLIN_COL_IDX][:, :, :,
                                          _MARLIN_PACK_IDX].astype(np.uint32)

        packed_out = (gathered[:, :, :, 0] | (gathered[:, :, :, 1] << 4)
                      | (gathered[:, :, :, 2] << 8)
                      | (gathered[:, :, :, 3] << 12)
                      | (gathered[:, :, :, 4] << 16)
                      | (gathered[:, :, :, 5] << 20)
                      | (gathered[:, :, :, 6] << 24)
                      | (gathered[:, :, :, 7] << 28))

        out = np.zeros((k_tiles, n_tiles * 128), dtype=np.uint32)
        for n_tile_id in range(n_tiles):
            out[:,
                n_tile_id * 128 + _MARLIN_OUT_IDX] = packed_out[:,
                                                                n_tile_id, :]
        weights_marlin_list.append(
            torch.from_numpy(out.view(np.int32)).to(device))

    weights_marlin = torch.stack(weights_marlin_list, dim=0)

    scales_marlin = scales.transpose(1, 2).contiguous()  # [E, num_groups, N]
    for e in range(num_experts):
        scales_marlin[e] = _marlin_permute_scales(scales_marlin[e], K, N,
                                                  group_size)

    return weights_marlin, scales_marlin


# ---------------------------------------------------------------------------
# NVFP4 MoE Marlin tile pack
# ---------------------------------------------------------------------------

_FP8_MAX = 448.0
_FP4_E2M1_POSITIVE_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                     dtype=np.float32)
# Midpoints between consecutive E2M1 levels (for searchsorted-based quantization).
_E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                        dtype=np.float32)


def decode_modelopt_nvfp4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    group_size: int = 16,
) -> np.ndarray:
    """Dequantize one ModelOpt NVFP4 weight tensor to dense fp32 ``[out, in]``.

    ``weight`` is ``[out, in//2]`` int8/uint8 with two FP4 E2M1 nibbles per
    byte (low nibble = even index).  ``weight_scale`` is ``[out, in//group_size]``
    FP8 E4M3 (accepts ``float8_e4m3fn``, an int8 view of it, or a float cast).
    ``weight_scale_2`` is ``[1]`` fp32 per-tensor scale-of-scale.
    """
    w = weight.detach().cpu().numpy()
    if w.dtype == np.int8:
        w = w.view(np.uint8)
    if w.dtype != np.uint8:
        raise TypeError(f"unexpected weight dtype {w.dtype}")
    out_f, half = w.shape

    lo = w & np.uint8(0x0F)
    hi = (w >> np.uint8(4)) & np.uint8(0x0F)
    nibbles = np.empty((out_f, half * 2), dtype=np.uint8)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi
    sign = (nibbles & np.uint8(0x08)) != 0
    magnitude = nibbles & np.uint8(0x07)
    values = _FP4_E2M1_POSITIVE_LEVELS[magnitude]
    values = np.where(sign, -values, values).astype(np.float32)

    if weight_scale.dtype == torch.float8_e4m3fn:
        ws_fp32 = weight_scale.detach().to(torch.float32).cpu().numpy()
    elif weight_scale.dtype == torch.int8:
        ws_fp32 = (weight_scale.detach().view(torch.float8_e4m3fn).to(
            torch.float32).cpu().numpy())
    elif weight_scale.dtype in (torch.float32, torch.float16, torch.bfloat16):
        ws_fp32 = weight_scale.detach().to(torch.float32).cpu().numpy()
    else:
        raise TypeError(f"unsupported weight_scale dtype {weight_scale.dtype}")

    ws2 = float(weight_scale_2.detach().reshape(-1)[0].item())

    num_groups = ws_fp32.shape[-1]
    in_f = num_groups * group_size
    if values.shape != (out_f, in_f):
        raise ValueError(f"nibble shape {values.shape} does not match "
                         f"(out={out_f}, num_groups*group_size={in_f})")
    values_grouped = values.reshape(out_f, num_groups, group_size)
    dense = values_grouped * ws_fp32[..., np.newaxis]
    dense = dense.reshape(out_f, in_f)
    dense *= ws2
    return dense.astype(np.float32)


def _round_dense_to_bf16(dense: np.ndarray) -> np.ndarray:
    """Round a dense fp32 weight through BF16 while returning fp32 storage."""
    dense_t = torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32))
    return dense_t.to(torch.bfloat16).to(torch.float32).cpu().numpy()


def _swizzle_nvfp4_mma_scales(scale_bytes: np.ndarray, m_dim: int,
                              k_sf_dim: int) -> np.ndarray:
    """Swizzle linear FP8 block scales to CuTeDSL's 6D MMA layout."""
    if scale_bytes.dtype == np.int8:
        sf = scale_bytes.view(np.uint8)
    elif scale_bytes.dtype == np.uint8:
        sf = scale_bytes
    else:
        raise TypeError(f"unexpected scale dtype {scale_bytes.dtype}")
    if sf.shape != (m_dim, k_sf_dim):
        raise ValueError(f"scale shape {sf.shape} != ({m_dim}, {k_sf_dim})")

    m_tiles = (m_dim + 127) // 128
    k_tiles = (k_sf_dim + 3) // 4
    padded_m = m_tiles * 128
    padded_k_sf = k_tiles * 4
    sf_padded = np.zeros((padded_m, padded_k_sf), dtype=np.uint8)
    sf_padded[:m_dim, :k_sf_dim] = sf

    sf_5d = sf_padded.reshape(m_tiles, 4, 32, k_tiles, 4)
    return sf_5d.transpose(0, 3, 2, 1, 4).copy().view(np.int8)


def _pack_nvfp4_moe_weight(
        dense_w_mk: np.ndarray,
        group_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack dense ``[M, K]`` weights for ``Nvfp4MoePlugin``.

    Returns ``(qweights [M, K/2] int8,
    blocks_scale [m_tiles, k_tiles, 32, 4, 4] int8)``.  The scale tensor
    stores raw FP8 E4M3 block scales in the physical CuTeDSL MMA layout.
    """
    if group_size != 16:
        raise NotImplementedError("Nvfp4MoePlugin requires group_size=16")

    m_dim, k_dim = dense_w_mk.shape
    if k_dim % group_size != 0 or k_dim % 2 != 0:
        raise ValueError(
            f"K ({k_dim}) must be a multiple of {group_size} and even")

    dense = _round_dense_to_bf16(dense_w_mk)
    k_sf_dim = k_dim // group_size
    dense_blocks = dense.reshape(m_dim, k_sf_dim, group_size)
    block_scales = np.maximum(np.abs(dense_blocks).max(axis=-1) / 6.0,
                              1e-12).astype(np.float32)

    scaled = (dense_blocks / block_scales[..., np.newaxis]).clip(-6.0, 6.0)
    abs_idx = np.searchsorted(_E2M1_BOUNDS, np.abs(scaled)).astype(np.uint8)
    sign_bit = (scaled < 0).astype(np.uint8) << np.uint8(3)
    nibbles = (abs_idx | sign_bit).reshape(m_dim, k_dim)

    lo = nibbles[:, 0::2]
    hi = nibbles[:, 1::2]
    qweights = (lo | (hi << np.uint8(4))).astype(np.uint8).view(np.int8)

    sf_bytes = torch.from_numpy(block_scales.copy()).to(
        torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
    blocks_scale = _swizzle_nvfp4_mma_scales(sf_bytes, m_dim, k_sf_dim)

    return (torch.from_numpy(qweights.copy()),
            torch.from_numpy(blocks_scale.copy()))


def _interleave_qwen3_swiglu_fc1(
    gate_dense: np.ndarray,
    up_dense: np.ndarray,
    hidden_size: int,
    moe_inter_size: int,
) -> np.ndarray:
    """Build FC1 dense weight as 64-row interleaved up/gate chunks.

    Layout: ``[up_chunk(64), gate_chunk(64), up_chunk(64), gate_chunk(64), ...]``
    along the M axis. Consumed natively by the SM100/101/110 ``Nvfp4MoePlugin`` split
    FC1 kernel.
    """
    if gate_dense.shape != up_dense.shape:
        raise ValueError(
            f"gate dense shape {gate_dense.shape} != up dense shape {up_dense.shape}"
        )
    expected_shape = (moe_inter_size, hidden_size)
    if gate_dense.shape != expected_shape:
        raise ValueError(
            f"gate/up dense shape {gate_dense.shape} != {expected_shape}")

    swiglu_interleave_rows = 64
    if moe_inter_size % swiglu_interleave_rows != 0:
        raise ValueError(
            f"moe_inter_size ({moe_inter_size}) must be a multiple of "
            f"{swiglu_interleave_rows} for SwiGLU FC1 layout")

    n_chunks = moe_inter_size // swiglu_interleave_rows
    up_chunks = up_dense.reshape(n_chunks, swiglu_interleave_rows, hidden_size)
    gate_chunks = gate_dense.reshape(n_chunks, swiglu_interleave_rows,
                                     hidden_size)
    return np.stack([up_chunks, gate_chunks],
                    axis=1).reshape(2 * moe_inter_size, hidden_size)


def _concat_qwen3_swiglu_fc1(
    gate_dense: np.ndarray,
    up_dense: np.ndarray,
    hidden_size: int,
    moe_inter_size: int,
) -> np.ndarray:
    """Build FC1 dense weight as plain ``[up_all, gate_all]`` concat.

    Layout: all ``moe_inter_size`` up rows followed by all ``moe_inter_size``
    gate rows along the M axis. Consumed natively by the SM12x
    ``NvFP4MoEPluginGeforce`` fused kernel.
    """
    if gate_dense.shape != up_dense.shape:
        raise ValueError(
            f"gate dense shape {gate_dense.shape} != up dense shape {up_dense.shape}"
        )
    expected_shape = (moe_inter_size, hidden_size)
    if gate_dense.shape != expected_shape:
        raise ValueError(
            f"gate/up dense shape {gate_dense.shape} != {expected_shape}")
    return np.concatenate([up_dense, gate_dense],
                          axis=0).reshape(2 * moe_inter_size, hidden_size)


def repack_nvfp4_qwen3_moe_experts(
    experts: Iterable[nn.Module],
    hidden_size: int,
    moe_inter_size: int,
    group_size: int = 16,
    fc1_layout: str = "interleave",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack Qwen3 NVFP4 experts for the active NVFP4 MoE plugin.

    Each expert is expected to contain ModelOpt NVFP4 gate/up/down
    projection tensors.  Dense weights are decoded, rounded through BF16, and
    repacked for the CuTeDSL plugin.

    Args:
        experts: per-expert ``nn.Module`` containers exposing
            ``gate_proj`` / ``up_proj`` / ``down_proj``.
        hidden_size: model hidden size ``H``.
        moe_inter_size: per-expert intermediate size ``I``.
        group_size: NVFP4 K-axis group size (must be ``16``).
        fc1_layout: SwiGLU FC1 row layout.
            * ``"interleave"`` (default) -- ``Nvfp4MoePlugin`` (SM100/101/110): 64-row
              up/gate interleaved chunks along the M axis.
            * ``"concat"`` -- ``NvFP4MoEPluginGeforce`` (SM12x): plain
              ``[up_all, gate_all]`` concat along the M axis.
    """
    from ..models.linear import \
        is_nvfp4_linear  # local import to avoid circular dep

    if fc1_layout == "interleave":
        build_fc1_dense = _interleave_qwen3_swiglu_fc1
    elif fc1_layout == "concat":
        build_fc1_dense = _concat_qwen3_swiglu_fc1
    else:
        raise ValueError(
            f"fc1_layout={fc1_layout!r} not recognized; use "
            "'interleave' (SM100/101/110 Nvfp4MoePlugin) or 'concat' (SM12x "
            "NvFP4MoEPluginGeforce)")

    fc1_qweights = []
    fc1_blocks_scale = []
    fc2_qweights = []
    fc2_blocks_scale = []

    for expert in experts:
        gate = expert.gate_proj
        up = expert.up_proj
        down = expert.down_proj
        if not (is_nvfp4_linear(gate) and is_nvfp4_linear(up)
                and is_nvfp4_linear(down)):
            raise TypeError("Qwen3 NVFP4 MoE experts must use NVFP4 quant")

        gate_dense = decode_modelopt_nvfp4(gate.weight, gate.weight_scale,
                                           gate.weight_scale_2, group_size)
        up_dense = decode_modelopt_nvfp4(up.weight, up.weight_scale,
                                         up.weight_scale_2, group_size)
        down_dense = decode_modelopt_nvfp4(down.weight, down.weight_scale,
                                           down.weight_scale_2, group_size)

        if down_dense.shape != (hidden_size, moe_inter_size):
            raise ValueError(f"down dense shape {down_dense.shape} != "
                             f"({hidden_size}, {moe_inter_size})")

        fc1_dense = build_fc1_dense(gate_dense, up_dense, hidden_size,
                                    moe_inter_size)
        fc1_qw, fc1_sf = _pack_nvfp4_moe_weight(fc1_dense, group_size)
        fc2_qw, fc2_sf = _pack_nvfp4_moe_weight(down_dense, group_size)
        fc1_qweights.append(fc1_qw)
        fc1_blocks_scale.append(fc1_sf)
        fc2_qweights.append(fc2_qw)
        fc2_blocks_scale.append(fc2_sf)

    return (torch.stack(fc1_qweights,
                        dim=0), torch.stack(fc1_blocks_scale, dim=0),
            torch.stack(fc2_qweights,
                        dim=0), torch.stack(fc2_blocks_scale, dim=0))


def repack_nvfp4_nemotron_moe_experts(
    experts: Iterable[nn.Module],
    hidden_size: int,
    moe_inter_size: int,
    group_size: int = 16,
    hidden_size_alignment: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, int, int]:
    """Pack Nemotron-H NVFP4 experts for the active NVFP4 MoE plugin.

    Nemotron routed experts are ReLU2 MLPs, so FC1 is the raw ``up_proj``
    only.  Unlike Qwen3, no gate projection must be concatenated into a SwiGLU
    FC1 tensor.  The checkpoint's FP4 nibbles and FP8 block-scale bytes already
    match the plugin weight orientation; this helper only changes the scale
    layout to CuTeDSL's 6D MMA order and exposes ``weight_scale_2`` as the
    per-expert alpha.

    Both kernels require the FC1 N axis (``moe_inter_size``) be a multiple of
    128, so the routed intermediate is always padded to the next 128 boundary.
    The SM12x ``NvFP4MoEPluginGeforce`` kernel additionally requires the FC1 K
    axis (``hidden_size``) be a multiple of 256 (``kCuteDslTileK * kStaticAbStage``)
    to drain the mainloop pipeline cleanly. Pass ``hidden_size_alignment=256``
    to also zero-pad along the H axis for that target; SM110 keeps
    ``hidden_size_alignment=1`` so ``H`` is unchanged.

    Math: ``relu2(0) = 0`` makes the zero-padded FC1 N rows produce zero
    intermediate activations; zero-padded FC1 K columns receive zero hidden
    activations (the caller is expected to zero-pad ``hidden_states`` to match
    ``padded_hidden_size`` before the plugin call). Zero-padded FC2 M rows
    emit zero contributions to the corresponding output channels, which the
    caller slices away.

    Returns the per-expert FC1/FC2 weights, scales, alphas, plus the resolved
    ``padded_inter_size`` and ``padded_hidden_size`` (both >= the originals).
    """
    from ..models.linear import \
        is_nvfp4_linear  # local import to avoid circular dep

    fc1_qweights = []
    fc1_blocks_scale = []
    fc1_alpha = []
    fc2_qweights = []
    fc2_blocks_scale = []
    fc2_alpha = []
    padded_inter_size = ((moe_inter_size + 127) // 128) * 128
    if hidden_size_alignment <= 0:
        raise ValueError(
            f"hidden_size_alignment ({hidden_size_alignment}) must be >= 1")
    padded_hidden_size = ((hidden_size + hidden_size_alignment - 1) //
                          hidden_size_alignment) * hidden_size_alignment
    if padded_hidden_size % group_size != 0:
        raise ValueError(
            f"padded_hidden_size ({padded_hidden_size}) must be a multiple of "
            f"group_size ({group_size})")
    h_padded_k = padded_hidden_size // 2  # FC1 K dim (FP4 packed)
    h_padded_sf = padded_hidden_size // group_size  # FC1/FC2 SF count along H

    for expert in experts:
        up = expert.up_proj
        down = expert.down_proj
        if not (is_nvfp4_linear(up) and is_nvfp4_linear(down)):
            raise TypeError("Nemotron NVFP4 MoE experts must use NVFP4 quant")

        if tuple(up.weight.shape) != (moe_inter_size, hidden_size // 2):
            raise ValueError(f"up weight shape {tuple(up.weight.shape)} != "
                             f"({moe_inter_size}, {hidden_size // 2})")
        if tuple(up.weight_scale.shape) != (moe_inter_size,
                                            hidden_size // group_size):
            raise ValueError(
                f"up scale shape {tuple(up.weight_scale.shape)} != "
                f"({moe_inter_size}, {hidden_size // group_size})")
        if tuple(down.weight.shape) != (hidden_size, moe_inter_size // 2):
            raise ValueError(
                f"down weight shape {tuple(down.weight.shape)} != "
                f"({hidden_size}, {moe_inter_size // 2})")
        if tuple(down.weight_scale.shape) != (hidden_size,
                                              moe_inter_size // group_size):
            raise ValueError(
                f"down scale shape {tuple(down.weight_scale.shape)} != "
                f"({hidden_size}, {moe_inter_size // group_size})")

        up_weight = up.weight.detach().cpu()
        down_weight = down.weight.detach().cpu()
        if up_weight.dtype == torch.uint8:
            up_weight = up_weight.view(torch.int8)
        if down_weight.dtype == torch.uint8:
            down_weight = down_weight.view(torch.int8)
        if up_weight.dtype != torch.int8 or down_weight.dtype != torch.int8:
            raise TypeError("Nemotron NVFP4 weights must be int8/uint8")

        needs_pad = (padded_inter_size != moe_inter_size
                     or padded_hidden_size != hidden_size)
        if needs_pad:
            padded_up_weight = torch.zeros((padded_inter_size, h_padded_k),
                                           dtype=torch.int8)
            padded_up_weight[:moe_inter_size, :hidden_size // 2] = up_weight
            up_weight = padded_up_weight

            padded_down_weight = torch.zeros(
                (padded_hidden_size, padded_inter_size // 2), dtype=torch.int8)
            padded_down_weight[:hidden_size, :moe_inter_size //
                               2] = down_weight
            down_weight = padded_down_weight

            up_sf_bytes = _sf_bytes_from_checkpoint(up.weight_scale)
            padded_up_sf = np.zeros((padded_inter_size, h_padded_sf),
                                    dtype=np.uint8)
            padded_up_sf[:moe_inter_size, :hidden_size //
                         group_size] = up_sf_bytes

            down_sf_bytes = _sf_bytes_from_checkpoint(down.weight_scale)
            padded_down_sf = np.zeros(
                (padded_hidden_size, padded_inter_size // group_size),
                dtype=np.uint8)
            padded_down_sf[:hidden_size, :moe_inter_size //
                           group_size] = down_sf_bytes
        else:
            padded_up_sf = _sf_bytes_from_checkpoint(up.weight_scale)
            padded_down_sf = _sf_bytes_from_checkpoint(down.weight_scale)

        up_sf = _swizzle_nvfp4_mma_scales(padded_up_sf, padded_inter_size,
                                          h_padded_sf)
        down_sf = _swizzle_nvfp4_mma_scales(padded_down_sf, padded_hidden_size,
                                            padded_inter_size // group_size)

        fc1_qweights.append(up_weight.contiguous())
        fc1_blocks_scale.append(torch.from_numpy(up_sf))
        fc1_alpha.append(float(up.weight_scale_2.detach().reshape(-1)[0]))
        fc2_qweights.append(down_weight.contiguous())
        fc2_blocks_scale.append(torch.from_numpy(down_sf))
        fc2_alpha.append(float(down.weight_scale_2.detach().reshape(-1)[0]))

    return (torch.stack(fc1_qweights,
                        dim=0), torch.stack(fc1_blocks_scale, dim=0),
            torch.tensor(fc1_alpha,
                         dtype=torch.float32), torch.stack(fc2_qweights,
                                                           dim=0),
            torch.stack(fc2_blocks_scale,
                        dim=0), torch.tensor(fc2_alpha, dtype=torch.float32),
            padded_inter_size, padded_hidden_size)


def _sf_bytes_from_checkpoint(raw_sf: torch.Tensor) -> np.ndarray:
    """Extract raw FP8-E4M3 bytes from a checkpoint ``weight_scale`` tensor."""
    if raw_sf.dtype == torch.float8_e4m3fn:
        return raw_sf.detach().cpu().view(torch.uint8).numpy()
    if raw_sf.dtype == torch.int8:
        return raw_sf.detach().cpu().view(torch.uint8).numpy()
    if raw_sf.dtype in (torch.float32, torch.float16, torch.bfloat16):
        # Float-cast fallback (non-ModelOpt checkpoints). Go through FP8 E4M3.
        return raw_sf.detach().to(torch.float8_e4m3fn).cpu().view(
            torch.uint8).numpy()
    raise TypeError(f"unsupported weight_scale dtype {raw_sf.dtype}")
