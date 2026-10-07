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
"""Standard-op lowering of trt::Attention / trt::RotaryEmbedding (EDGELLM_PORTABLE_ATTENTION)."""

import numpy as np
import onnxscript
import pytest
from onnx.reference import ReferenceEvaluator
from onnxscript import script

from tensorrt_edgellm.onnx.dynamo_translations import (
    _portable_attention_dispatch, _portable_rope_translation,
    _portable_unmasked_attention_translation)

SCALE = 0.25


@script()
def _attention_at_test_scale(
    query: onnxscript.FLOAT16,
    key: onnxscript.FLOAT16,
    value: onnxscript.FLOAT16,
) -> onnxscript.FLOAT16:
    return _portable_unmasked_attention_translation(query,
                                                    key,
                                                    value,
                                                    scale=0.25)


def _run(function, **inputs):
    return ReferenceEvaluator(function.to_model_proto()).run(None, inputs)[0]


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("heads,kv_heads", [(8, 1), (4, 4), (8, 2)])
def test_unmasked_attention_matches_reference(heads, kv_heads):
    rng = np.random.default_rng(0)
    batch, seq, kv_seq, dim = 2, 5, 7, 16
    q = rng.standard_normal((batch, heads, seq, dim)).astype(np.float16)
    k = rng.standard_normal((batch, kv_heads, kv_seq, dim)).astype(np.float16)
    v = rng.standard_normal((batch, kv_heads, kv_seq, dim)).astype(np.float16)

    out = _run(_attention_at_test_scale, query=q, key=k, value=v)

    group = heads // kv_heads
    k_full = np.repeat(k.astype(np.float64), group, axis=1)
    v_full = np.repeat(v.astype(np.float64), group, axis=1)
    expected = _softmax(
        SCALE * q.astype(np.float64) @ k_full.transpose(0, 1, 3, 2)) @ v_full
    assert out.shape == q.shape
    np.testing.assert_allclose(out.astype(np.float64),
                               expected,
                               atol=5e-3,
                               rtol=5e-3)


def test_rope_matches_rotate_half():
    rng = np.random.default_rng(1)
    batch, heads, seq, dim, positions = 2, 3, 4, 8, 10
    x = rng.standard_normal((batch, heads, seq, dim)).astype(np.float16)
    angles = rng.uniform(0, np.pi, (positions, dim // 2))
    cos, sin = np.cos(angles).astype(np.float16), np.sin(angles).astype(
        np.float16)
    position_ids = rng.integers(0, positions, (batch, seq)).astype(np.int32)

    out = _run(_portable_rope_translation,
               x=x,
               cos=cos,
               sin=sin,
               position_ids=position_ids)

    c = cos[position_ids][:, None].astype(np.float64)
    s = sin[position_ids][:, None].astype(np.float64)
    x1, x2 = np.split(x.astype(np.float64), 2, axis=-1)
    expected = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)
    np.testing.assert_allclose(out.astype(np.float64),
                               expected,
                               atol=5e-3,
                               rtol=5e-3)


def test_masked_or_causal_attention_is_refused():
    with pytest.raises(NotImplementedError):
        _portable_attention_dispatch(None,
                                     None,
                                     None,
                                     attn_mask=object(),
                                     is_causal=False,
                                     scale=1.0)
    with pytest.raises(NotImplementedError):
        _portable_attention_dispatch(None,
                                     None,
                                     None,
                                     attn_mask=None,
                                     is_causal=True,
                                     scale=1.0)
