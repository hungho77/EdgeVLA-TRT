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
"""TensorRT 10.3 rewrite: a scalar on a split-out transposed key moves onto the query."""

import numpy as np
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from tensorrt_edgellm.onnx.trt_workarounds import move_key_scale_to_query


def _attention_scores(split_key: bool):
    """q @ (k^T * s) with k split out of a fused [3, heads, len, dim] tensor, or taken from its own input."""
    nodes, inputs = [], [
        helper.make_tensor_value_info("qkv", TensorProto.FLOAT, [3, 2, 5, 4])
    ]
    nodes.append(
        helper.make_node("Constant", [], ["axes"],
                         value=helper.make_tensor("a", TensorProto.INT64, [1],
                                                  [0])))
    nodes.append(
        helper.make_node("Split", ["qkv"], ["qs", "ks", "vs"],
                         axis=0,
                         num_outputs=3))
    nodes.append(helper.make_node("Squeeze", ["qs", "axes"], ["q"]))
    if split_key:
        nodes.append(helper.make_node("Squeeze", ["ks", "axes"], ["k"]))
    else:
        inputs.append(
            helper.make_tensor_value_info("k", TensorProto.FLOAT, [2, 5, 4]))
    nodes.append(helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 2, 1]))
    nodes.append(
        helper.make_node("Constant", [], ["s"],
                         value=helper.make_tensor("v", TensorProto.FLOAT, [],
                                                  [0.5])))
    nodes.append(helper.make_node("Mul", ["kt", "s"], ["kts"]))
    nodes.append(helper.make_node("MatMul", ["q", "kts"], ["scores"]))
    graph = helper.make_graph(
        nodes, "g", inputs,
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, None)])
    return helper.make_model(graph,
                             opset_imports=[helper.make_opsetid("", 18)])


def test_split_key_scale_moves_to_query():
    model = _attention_scores(split_key=True)
    feed = {
        "qkv":
        np.random.default_rng(0).standard_normal(
            (3, 2, 5, 4)).astype(np.float32)
    }
    before = ReferenceEvaluator(model).run(None, feed)[0]
    assert move_key_scale_to_query(model) == 1
    matmul = next(n for n in model.graph.node if n.op_type == "MatMul")
    assert matmul.input[1] == "kt"
    np.testing.assert_allclose(ReferenceEvaluator(model).run(None, feed)[0],
                               before,
                               rtol=1e-6)


def test_key_not_split_from_a_fused_projection_is_kept():
    model = _attention_scores(split_key=False)
    assert move_key_scale_to_query(model) == 0
