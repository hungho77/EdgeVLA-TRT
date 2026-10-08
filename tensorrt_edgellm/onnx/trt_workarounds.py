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
"""ONNX rewrites that keep exported graphs out of TensorRT 10.3 (JetPack 6) miscompilations.

``move_key_scale_after_matmul``: TensorRT 10.3 miscomputes attention whose transposed key is multiplied by a scalar
when the key comes from a fused QKV projection split into heads (Reshape / Transpose / Split / Squeeze) -- the
pattern PyTorch's scaled_dot_product_attention export emits for a fused-QKV module. A single such block was off by
0.10 on outputs up to 10, in FP32 with TF32 off, while onnxruntime matched PyTorch. ``MatMul(q, Mul(k^T, s))`` is
rewritten to ``MatMul(Mul(q, s), k^T)``, the same function for a scalar ``s``: TensorRT computes a scaled query
correctly, and it is the smaller operand to scale. Only keys split out of a fused projection are rewritten, so graphs TensorRT already computes stay
unchanged.
"""

import onnx
from onnx import helper


def _scalar_inputs(model: onnx.ModelProto) -> set:
    """Names of tensors known to hold a single element (rank 0 or all dims 1)."""
    scalars = set()

    def single(dims):
        return all(d == 1 for d in dims)

    for init in model.graph.initializer:
        if single(init.dims):
            scalars.add(init.name)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and single(attr.t.dims):
                    scalars.add(node.output[0])
    for value in list(model.graph.value_info) + list(model.graph.input):
        shape = value.type.tensor_type.shape
        if value.type.tensor_type.HasField("shape") and all(
                d.HasField("dim_value") and d.dim_value == 1
                for d in shape.dim):
            scalars.add(value.name)
    return scalars


def _split_from_fused_projection(name: str, producers: dict) -> bool:
    """Whether ``name`` is a Transpose of a head tensor taken out of a fused projection (Squeeze / Gather of Split)."""
    node = producers.get(name)
    if node is None or node.op_type != "Transpose":
        return False
    for _ in range(2):
        node = producers.get(node.input[0])
        if node is None:
            return False
        if node.op_type == "Split":
            return True
        if node.op_type not in ("Squeeze", "Gather"):
            return False
    return False


def move_key_scale_to_query(model: onnx.ModelProto) -> int:
    """Rewrite MatMul(q, Mul(Transpose(k), s)) with scalar s to MatMul(Mul(q, s), Transpose(k)). Returns the count."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:  # noqa: BLE001 -- models above 2 GB: fall back to constants only
        inferred = model
    scalars = _scalar_inputs(inferred)
    producers = {out: node for node in model.graph.node for out in node.output}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    graph_outputs = {o.name for o in model.graph.output}

    rewritten = 0
    for matmul in list(model.graph.node):
        if matmul.op_type != "MatMul" or len(matmul.input) != 2:
            continue
        mul = producers.get(matmul.input[1])
        if mul is None or mul.op_type != "Mul" or mul.output[
                0] in graph_outputs:
            continue
        if len(consumers.get(mul.output[0], [])) != 1:
            continue
        operands = list(mul.input)
        transposed = [
            i for i, name in enumerate(operands)
            if _split_from_fused_projection(name, producers)
        ]
        if len(transposed) != 1 or operands[1 - transposed[0]] not in scalars:
            continue
        key, scale = operands[transposed[0]], operands[1 - transposed[0]]
        query = matmul.input[0] + "_key_scale"
        scaled = helper.make_node("Mul", [matmul.input[0], scale], [query],
                                  name=mul.name + "_on_query")
        nodes = model.graph.node
        nodes.insert(list(nodes).index(matmul), scaled)
        matmul.input[0] = query
        matmul.input[1] = key
        nodes.remove(mul)
        rewritten += 1
    return rewritten


def apply_trt103_workarounds(path: str) -> int:
    """Apply every rewrite to the ONNX file at ``path`` in place (external data stays where it is)."""
    model = onnx.load(path, load_external_data=False)
    count = move_key_scale_to_query(model)
    if count:
        onnx.save(model, path)
    return count
