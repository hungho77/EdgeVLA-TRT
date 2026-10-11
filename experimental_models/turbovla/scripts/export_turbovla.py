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
"""Export the TurboVLA LIBERO checkpoint as text / policy ONNX graphs (FP16, batch 1).

The official modules provide every layer; the legacy TorchScript exporter lowers them to standard ops.

  text.onnx    BERT token ids, position ids and the sub-sentence self-attention mask [1, L, L] (the masks the
               official generate_masks_with_special_tokens builds, computed on the host) -> projected text tokens
               [1, L, D]. ``hidden_valid`` zeroes the BERT output past an instruction's own padding length (the
               checkpoint's padding_length_by_instruction), as the official encoder fills those rows with zeros.
  policy.onnx  ImageNet-normalized views [1, V, 3, S, S], text tokens, the tokenizer's attention mask, the
               sub-sentence mask and the normalized state -> the normalized action chunk [1, chunk, action_dim] (tanh)

The text graph only changes with the instruction, so the runtime caches its output; it computes in FP32 behind FP16
inputs and outputs. Both graphs go through
tensorrt_edgellm.onnx.trt_workarounds (the decoder's nn.MultiheadAttention splits a fused QKV projection).

Writes config.json (shapes, the padding layout, the special tokens the masks split on, the suite statistics) and the
BERT tokenizer.json.

    python export_turbovla.py --repo TurboVLA --checkpoint TurboVLA-hf --dinov3 dinov3-vitb16 \\
        --bert bert-base-uncased --out turbovla_onnx [--check ref.npz --obs obs.npz]
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import turbovla_reference  # noqa: E402

from tensorrt_edgellm.onnx.trt_workarounds import \
    apply_trt103_workarounds  # noqa: E402


class Text(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.encoder = model.text_encoder
        self.zero_padded = bool(model.config.text.zero_padded_tokens)

    def forward(self, input_ids, position_ids, self_attention, hidden_valid,
                attention):
        io_dtype = self_attention.dtype
        dtype = self.encoder.text_projection.weight.dtype
        self_attention, hidden_valid, attention = (t.to(dtype)
                                                   for t in (self_attention,
                                                             hidden_valid,
                                                             attention))
        hidden = self.encoder.bert(input_ids=input_ids,
                                   attention_mask=self_attention,
                                   token_type_ids=torch.zeros_like(input_ids),
                                   position_ids=position_ids).last_hidden_state
        hidden = hidden * hidden_valid.unsqueeze(-1)
        tokens = self.encoder.text_projection(hidden)
        if self.zero_padded:
            tokens = tokens * attention.unsqueeze(-1)
        return tokens.to(io_dtype)


class Policy(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixels, text_tokens, attention, self_attention, state):
        model = self.model
        visual = model.encode_vision(pixels)
        visual, text = model.vision_language_interaction(
            visual_tokens=visual,
            text_tokens=text_tokens,
            text_key_padding_mask=attention < 0.5,
            text_self_attention_masks=self_attention > 0.5)
        return model.action_head(torch.cat([visual, text], dim=1), state)


def length_free_attention(model):
    """Every nn.MultiheadAttention written so the text length stays a graph dimension (the text layers attend over
    the text, the action head over the visual, text and state tokens): the legacy exporter records
    F.multi_head_attention_forward's reshapes with the example length. Separate Q / K / V projections and the scale
    on Q (TensorRT 10.3 miscomputes a scaled K split out of a fused projection)."""
    for module in model.modules():
        if not isinstance(module, nn.MultiheadAttention):
            continue

        def forward(query,
                    key,
                    value,
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=True,
                    mha=module,
                    **kw):
            if mha.batch_first:
                query, key, value = (t.transpose(0, 1)
                                     for t in (query, key, value))
            embed, heads = mha.embed_dim, mha.num_heads
            head_dim = embed // heads
            w, b = mha.in_proj_weight, mha.in_proj_bias
            q = nn.functional.linear(query, w[:embed],
                                     b[:embed]) * head_dim**-0.5
            k = nn.functional.linear(key, w[embed:2 * embed],
                                     b[embed:2 * embed])
            v = nn.functional.linear(value, w[2 * embed:], b[2 * embed:])
            batch_heads = query.shape[1] * heads
            q, k, v = (t.reshape(-1, batch_heads, head_dim).transpose(0, 1)
                       for t in (q, k, v))
            scores = torch.matmul(q, k.transpose(1, 2))
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float("-inf")) if attn_mask.dtype == torch.bool \
                    else scores + attn_mask
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask.repeat_interleave(heads, 0)[:, None, :],
                    float("-inf"))
            out = mha.out_proj(
                torch.matmul(scores.softmax(-1),
                             v).transpose(0,
                                          1).reshape(-1, query.shape[1],
                                                     embed))
            return (out.transpose(0, 1) if mha.batch_first else out), None

        module.forward = forward


def freeze_rope(model, size):
    """DINOv3's RoPE cos / sin depend only on the image size; computed once in FP32 (as the module does) they replace
    the coordinate arithmetic, whose shape-dependent branch TensorRT cannot parse."""
    rope = model.vision_encoder.backbone.rope_embeddings
    with torch.no_grad():
        cos, sin = rope(torch.zeros(1, 3, size, size))
    rope.forward = lambda pixel_values: (cos.to(pixel_values.dtype),
                                         sin.to(pixel_values.dtype))


def fp32_attention_scores(model):
    """DINOv3's first layer attends with scores up to ~1e5 after scaling (its high-norm register tokens), past FP16's
    range; PyTorch's FP16 attention kernels accumulate in FP32 and never store them, TensorRT's FP16 attention does.
    QK^T and the softmax run in FP32, the probabilities times V in FP16."""
    from transformers import AttentionInterface

    def attention(module,
                  query,
                  key,
                  value,
                  attention_mask,
                  scaling,
                  dropout=0.0,
                  **kwargs):
        scores = torch.matmul(query.float(),
                              key.float().transpose(-1, -2)) * scaling
        probs = scores.softmax(dim=-1).to(value.dtype)
        return torch.matmul(probs, value).transpose(1, 2).contiguous(), None

    AttentionInterface.register("turbovla_fp32_scores", attention)
    model.vision_encoder.backbone.config._attn_implementation = "turbovla_fp32_scores"


def text_inputs(ref, i, length, dtype):
    """The text graph's inputs for reference sample i, padded to ``length`` as the official encoder pads them."""
    ids = torch.zeros(1, length, dtype=torch.int64)
    positions = torch.zeros(1, length, dtype=torch.int64)
    self_attention = torch.eye(length, dtype=dtype)[None].clone()
    hidden_valid = torch.zeros(1, length, dtype=dtype)
    own = ref[f"input_ids_{i}"].shape[1]
    ids[0, :own] = torch.from_numpy(ref[f"input_ids_{i}"][0])
    positions[0, :own] = torch.from_numpy(ref[f"position_ids_{i}"][0])
    self_attention[0, :own, :own] = torch.from_numpy(
        ref[f"self_attention_{i}"][0].astype(np.float32))
    hidden_valid[0, :own] = 1
    valid = (~ref[f"key_padding_{i}"][0]).astype(np.float32)
    attention = torch.zeros(1, length, dtype=dtype)
    attention[0, :valid.shape[0]] = torch.from_numpy(valid)
    return ids, positions, self_attention, hidden_valid, attention


def check(policy, text, graph, ref_path, obs_path, length, dtype, device):
    ref, obs = np.load(ref_path), np.load(obs_path)
    count = len([k for k in ref.files if k.startswith("normalized_")])
    for i in range(count):
        ids, positions, self_attention, hidden_valid, attention = (
            t.to(device) for t in text_inputs(ref, i, length, dtype))
        samples, states = policy._build_batch(
            [turbovla_reference.policy_rotate(obs[f"agentview_{i}"])],
            [turbovla_reference.policy_rotate(obs[f"wrist_{i}"])],
            [obs[f"state_{i}"]])
        with torch.no_grad():
            tokens = text(ids, positions, self_attention, hidden_valid,
                          attention)
            actions = graph(samples["dinov3"].to(dtype), tokens, attention,
                            self_attention, states.to(dtype))
        for name, ours, expected in (("text tokens", tokens, ref[f"text_{i}"]),
                                     ("normalized actions", actions[0],
                                      ref[f"normalized_{i}"])):
            ours = ours.float().cpu().numpy()
            print(
                f"sample {i} split {name} vs official: max|d| {np.abs(ours - expected).max():.3e} "
                f"(|ref| max {np.abs(expected).max():.2f})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dinov3", required=True)
    parser.add_argument("--bert", required=True)
    parser.add_argument("--out")
    parser.add_argument(
        "--check",
        help="turbovla_reference.py output: compare the split graphs first")
    parser.add_argument("--obs", help="the observations --check was made from")
    parser.add_argument(
        "--embodiment",
        help="starVLA runs: the statistics key (default: the only one)")
    parser.add_argument("--check-dtype",
                        default="float32",
                        choices=("float32", "float16"))
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.precision = "fp32"

    torch.backends.mha.set_fastpath_enabled(False)
    policy = turbovla_reference.load_policy(args)
    model = policy.model
    config = model.config
    length = int(config.text.padding_length or config.text.max_length)
    text, graph = Text(model).eval(), Policy(model).eval()

    if args.check:
        if not hasattr(policy, "_build_batch"):
            raise SystemExit(
                "--check compares the released LIBERO checkpoint; check a starVLA run with "
                "run_turbovla_engines.py instead")
        dtype = getattr(torch, args.check_dtype)
        check(policy, text.to(dtype), graph.to(dtype), args.check, args.obs,
              length, dtype, args.device)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        text, graph = text.cpu(), graph.cpu()
        views, size = int(config.vision.num_views), int(
            config.vision.image_size)
        freeze_rope(model, size)
        fp32_attention_scores(model)
        f16 = torch.float16
        hidden = int(config.interaction.hidden_dim)
        # An instruction padded to the longest of its batch keeps its own token count at batch 1, and the action head
        # attends to every text token, padding included: such a run's graphs take the instruction's length.
        longest = config.text.padding_length is None and not config.text.padding_length_by_instruction
        text_axes = {
            "input_ids": {
                1: "length"
            },
            "position_ids": {
                1: "length"
            },
            "self_attention": {
                1: "length",
                2: "length"
            },
            "hidden_valid": {
                1: "length"
            },
            "attention": {
                1: "length"
            },
            "text_tokens": {
                1: "length"
            },
        } if longest else None
        policy_axes = {
            "text_tokens": {
                1: "length"
            },
            "attention": {
                1: "length"
            },
            "self_attention": {
                1: "length",
                2: "length"
            },
        } if longest else None
        example = min(length, 32) if longest else length
        if longest:
            length_free_attention(model)
        export = lambda m, a, name, i, o, axes: torch.onnx.export(
            m,
            a,
            os.path.join(args.out, f"{name}.onnx"),
            input_names=i,
            output_names=o,
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False)
        # The text graph runs once per instruction (the runtime caches its output), so it keeps FP32 weights and
        # math behind FP16 inputs and outputs; it is exported before the shared model goes to FP16.
        # Distinct example tensors: the exporter merges inputs that are the same object.
        export(text, (torch.zeros(1, example, dtype=torch.int64),
                      torch.arange(example)[None].clone(),
                      torch.eye(example, dtype=f16)[None].clone(),
                      torch.ones(1, example, dtype=f16),
                      torch.ones(1, example, dtype=f16) * 0.5), "text", [
                          "input_ids", "position_ids", "self_attention",
                          "hidden_valid", "attention"
                      ], ["text_tokens"], text_axes)
        graph = graph.half()
        export(
            graph, (torch.randn(1, views, 3, size, size, dtype=f16),
                    torch.randn(1, example, hidden,
                                dtype=f16), torch.ones(1, example, dtype=f16),
                    torch.eye(example, dtype=f16)[None].clone(),
                    torch.randn(1, int(config.action.state_dim), dtype=f16)),
            "policy",
            ["pixels", "text_tokens", "attention", "self_attention", "state"],
            ["actions"], policy_axes)
        rewrites = {
            name:
            apply_trt103_workarounds(os.path.join(args.out, f"{name}.onnx"))
            for name in ("text", "policy")
        }
        stage_runtime_assets(args, policy, length)
        print(
            f"exported text / policy -> {args.out}; TensorRT 10.3 rewrites {rewrites}"
        )


def stage_runtime_assets(args, policy, length):
    config = policy.model.config
    tokenizer = policy.model.text_encoder.tokenizer
    shutil.copy(os.path.join(args.bert, "tokenizer.json"),
                os.path.join(args.out, "tokenizer.json"))
    preprocessor = json.load(
        open(os.path.join(args.dinov3, "preprocessor_config.json")))
    config_json = {
        "model_family":
        "turbovla",
        "cameras": ["primary", "wrist"],
        "num_views":
        int(config.vision.num_views),
        "image_size":
        int(config.vision.image_size),
        "image_mean":
        preprocessor.get("image_mean", [0.485, 0.456, 0.406]),
        "image_std":
        preprocessor.get("image_std", [0.229, 0.224, 0.225]),
        "text_length":
        length,
        "text_length_by_instruction":
        dict(config.text.padding_length_by_instruction),
        "split_tokens":
        [int(t) for t in policy.model.text_encoder.special_tokens],
        "hidden_dim":
        int(config.interaction.hidden_dim),
        "chunk_size":
        int(config.action.horizon),
        "action_dim":
        int(config.action.action_dim),
        "state_dim":
        int(config.action.state_dim),
        "state_mean": [float(v) for v in policy.proprio_mean],
        "state_std": [float(v) for v in policy.proprio_std],
        "action_min": [float(v) for v in policy.action_min],
        "action_max": [float(v) for v in policy.action_max],
        "binary_gripper_index":
        int(config.action.action_dim) - 1,
        "tokenizer_lowercase":
        bool(getattr(tokenizer, "do_lower_case", True)),
    }
    if config.text.padding_length is None and not config.text.padding_length_by_instruction:
        # The official text encoder pads to the longest instruction of the batch: at batch 1, none.
        config_json["text_padding"] = "longest"
    if hasattr(policy, "runtime_config"):
        config_json.update(policy.runtime_config())
    json.dump(config_json,
              open(os.path.join(args.out, "config.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
