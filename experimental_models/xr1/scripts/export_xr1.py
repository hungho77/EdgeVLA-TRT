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
"""Export Xiaomi-Robotics-1 (MiBoTForActionGeneration) to FP16 ONNX graphs for TensorRT, from the checkpoint's modules:

  vision.onnx    the processor's patches of every view [views * 256, 1536] -> Qwen3-VL's ViT, merger and the three
                 deepstack mergers: image tokens [views * 64, 2560] and deepstack [3, views * 64, 2560].
  prefix_a.onnx  the prompt ids [1, L] (the image placeholders in place), the image tokens, the deepstack features
                 and the multimodal RoPE ids [3, L] -> language layers 0-17 over the prompt: hidden [1, L, 2560] and
                 post-RoPE K / V [18, 1, L, 8, 128]. L is a dynamic axis (the instruction's length).
  prefix_b.onnx  the hidden states and RoPE ids -> layers 18-35's K / V.
  step.onnx      one Euler step of the DiT head: the chunk [1, H, D] (FP32), the step index, the robot state, every
                 layer's prompt K / V, the prompt mask, the first head RoPE position, the action mask, dt and the
                 real-time-chunking inputs (the initial noise, the seed chunk, a per-row weight [H, 1]) -> the next
                 chunk. Rows with weight w move towards (1 - t) * noise + t * seed by w after the step (the flow runs
                 from noise at t = 0); weight 0 leaves the official step.

RMS norms, RoPE and attention run in FP32; the time conditioning (TimestepEmbedder and t_projector) is precomputed per
step in FP32. The DiT's fused QKV projection is split into three GEMMs (TensorRT 10.3 miscomputes a fused projection
viewed and indexed into Q / K / V). ``--check`` runs these graphs in PyTorch FP32 against
``xr1_reference.py --capture`` first. Each stage is written to OUT/<stage>/model.onnx; ``--stage assets`` writes the
runtime's config.json and tokenizer into OUT.

    python export_xr1.py --checkpoint Xiaomi-Robotics-1-RoboCasa --out onnx \\
        --stage vision|prefix_a|prefix_b|step [--check capture.npz]
    python export_xr1.py --checkpoint Xiaomi-Robotics-1-RoboCasa --out engines --stage assets
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import xr1_reference  # noqa: E402

from tensorrt_edgellm.onnx.trt_workarounds import \
    apply_trt103_workarounds  # noqa: E402

# Additive mask for excluded keys; finite, so a fully masked row averages instead of producing NaN.
MASKED = -1e30
GRID = (1, 16, 16)


def norm32(norm, x):
    if isinstance(norm, nn.LayerNorm):
        y = F.layer_norm(x.float(), norm.normalized_shape, norm.weight.float(),
                         norm.bias.float(), norm.eps)
    else:
        h = x.float()
        y = h * torch.rsqrt(
            h.pow(2).mean(-1, keepdim=True) +
            norm.variance_epsilon) * norm.weight.float()
    return y.to(x.dtype)


def attend(q, k, v, bias=None):
    """Grouped-query attention over [B, heads, L, D] tensors in FP32; returns [B, Lq, heads * D] in q's dtype."""
    dtype = q.dtype
    groups = q.shape[1] // k.shape[1]
    q, k, v = q.float(), k.float(), v.float()
    if groups > 1:
        k, v = (t[:, :, None].expand(-1, -1, groups, -1, -1).flatten(1, 2)
                for t in (k, v))
    scores = (q * q.shape[-1]**-0.5) @ k.transpose(-1, -2)
    if bias is not None:
        scores = scores + bias
    out = scores.softmax(-1) @ v
    return out.transpose(1, 2).flatten(2).to(dtype)


def rotate(q, k, cos, sin):
    """RoPE on [B, heads, L, D] in FP32 (apply_rotary_pos_emb with unsqueeze_dim 1)."""
    cos, sin = cos[:, None].float(), sin[:, None].float()

    def half(x):
        a, b = x.chunk(2, -1)
        return torch.cat([-b, a], -1)

    q, k = q.float(), k.float()
    return q * cos + half(q) * sin, k * cos + half(k) * sin


class Vision(nn.Module):
    """The ViT at a fixed grid per view: position embeddings and RoPE precomputed, attention within each image (the
    varlen split of the official path)."""

    def __init__(self, visual, views):
        super().__init__()
        grid = torch.tensor([GRID] * views,
                            device=visual.patch_embed.proj.weight.device)
        with torch.no_grad():
            pos = visual.fast_pos_embed_interpolate(grid)
            rotary = visual.rot_pos_emb(grid).reshape(pos.shape[0], -1)
            emb = torch.cat([rotary, rotary], -1)
        self.register_buffer("pos", pos.float())
        self.register_buffer("cos", emb.cos().float())
        self.register_buffer("sin", emb.sin().float())
        self.visual = visual
        self.views = views

    def forward(self, pixels):
        visual = self.visual
        x = visual.patch_embed(pixels) + self.pos.to(pixels.dtype)
        features = []
        cos, sin = self.cos[:, None], self.sin[:, None]
        for index, block in enumerate(visual.blocks):
            attn = block.attn
            h = norm32(block.norm1, x)
            dim = h.shape[-1]
            w, b = attn.qkv.weight, attn.qkv.bias
            q, k, v = (F.linear(h, w[i * dim:(i + 1) * dim],
                                b[i * dim:(i + 1) * dim]).unflatten(
                                    -1, (attn.num_heads, -1))
                       for i in range(3))
            half = lambda t: torch.cat(
                [-t[..., t.shape[-1] // 2:], t[..., :t.shape[-1] // 2]], -1)
            q = q.float() * cos + half(q.float()) * sin
            k = k.float() * cos + half(k.float()) * sin
            per_image = lambda t: t.reshape(self.views, -1, *t.shape[1:]
                                            ).transpose(1, 2)
            o = attend(
                per_image(q).to(x.dtype),
                per_image(k).to(x.dtype), per_image(v))
            x = x + attn.proj(o.reshape(x.shape[0], -1))
            x = x + block.mlp(norm32(block.norm2, x))
            if index in visual.deepstack_visual_indexes:
                merger = visual.deepstack_merger_list[
                    visual.deepstack_visual_indexes.index(index)]
                features.append(merger(x))
        return visual.merger(x), torch.stack(features)


class Prefix(nn.Module):
    """The prompt through a range of Qwen3-VL language layers (causal, batch 1), keeping each layer's K / V."""

    def __init__(self, model, layers, first):
        super().__init__()
        language = model.vlm.model.language_model
        self.language = language
        self.layers = nn.ModuleList(language.layers[i] for i in layers)
        self.first = first
        self.image_token = int(model.config.vlm_config.image_token_id)
        self.deepstack_layers = len(
            model.vlm.model.visual.deepstack_visual_indexes)

    def forward(self,
                ids_or_hidden,
                position_ids,
                image_tokens=None,
                deepstack=None):
        if self.first:
            ids = ids_or_hidden
            is_image = ids[0] == self.image_token
            index = (torch.cumsum(is_image.to(torch.int32), 0) -
                     1).clamp(min=0)
            x = torch.where(is_image[:, None], image_tokens[index],
                            self.language.embed_tokens(ids[0]))[None]
        else:
            x = ids_or_hidden
        length = x.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool,
                            device=x.device).tril()
        bias = torch.where(causal, 0.0, MASKED)[None, None]
        cos, sin = self.language.rotary_emb(x.float(), position_ids[:, None])
        keys, values = [], []
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = norm32(layer.input_layernorm, x)
            shape = (1, length, -1, attn.head_dim)
            q = norm32(attn.q_norm, attn.q_proj(h).view(shape)).transpose(1, 2)
            k = norm32(attn.k_norm, attn.k_proj(h).view(shape)).transpose(1, 2)
            v = attn.v_proj(h).view(shape).transpose(1, 2)
            q, k = rotate(q, k, cos, sin)
            keys.append(k.to(x.dtype).transpose(1, 2))
            values.append(v.transpose(1, 2))
            o = attend(q.to(x.dtype), k.to(x.dtype), v, bias)
            x = x + attn.o_proj(o)
            x = x + layer.mlp(norm32(layer.post_attention_layernorm, x))
            if self.first and i < self.deepstack_layers:
                x = x + torch.where(is_image[:, None], deepstack[i][index],
                                    0.0)[None].to(x.dtype)
        return x, torch.stack(keys), torch.stack(values)


def time_table(model, steps):
    """t_projector(t_embedder(t * 1000)) for t = step / steps, in FP32, as dit_forward computes it per step."""
    embedder = model.t_embedder
    dtype = embedder.dtype
    embedder.dtype = torch.float32
    try:
        t = torch.arange(steps, dtype=torch.float32, device="cuda") / steps
        with torch.no_grad():
            table = model.t_projector(embedder(t * 1000).float()).view(
                steps, 6, -1)
    finally:
        embedder.dtype = dtype
    return table.float()


class Step(nn.Module):

    def __init__(self, model, steps):
        super().__init__()
        self.model = model
        self.steps = steps
        self.register_buffer("time", time_table(model, steps))
        self.register_buffer(
            "next_t",
            torch.arange(1, steps + 1, dtype=torch.float32) / steps)

    def forward(self, x, step, state, prefix_keys, prefix_values, prefix_mask,
                suffix_start, action_mask, dt, noise, seed, weight):
        model = self.model
        dtype = prefix_keys.dtype
        t = self.time[step]  # [1, 6, hidden]
        actions = model.action_projector((x * action_mask).to(dtype))
        h = torch.cat([
            model.sink.weight[None].to(dtype),
            model.state_projector(state), actions
        ], 1)
        length = h.shape[1]
        positions = suffix_start + torch.arange(length, device=x.device)
        cos, sin = model.rotary_emb(h.float(),
                                    positions[None, None].expand(3, 1, -1))
        causal = torch.ones(length, length, dtype=torch.bool,
                            device=x.device).tril()
        bias = torch.cat([
            torch.where(prefix_mask > 0.5, 0.0,
                        MASKED)[:, None, None, :].expand(-1, -1, length, -1),
            torch.where(causal, 0.0, MASKED)[None, None]
        ], -1)
        for i, layer in enumerate(model.dit.layers):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                layer.adaln_table[None].float() + t).to(dtype).chunk(6, dim=1)
            attn = layer.attn
            hn = norm32(layer.input_layernorm, h) * (1 + scale_msa) + shift_msa
            dim = hn.shape[-1]
            w, b = attn.qkv_proj.weight, attn.qkv_proj.bias
            q, k, v = (F.linear(hn, w[j * dim:(j + 1) * dim],
                                b[j * dim:(j + 1) * dim]).view(
                                    1, length, attn.num_heads, attn.head_dim)
                       for j in range(3))
            q = norm32(attn.q_norm, q).transpose(1, 2)
            k = norm32(attn.k_norm, k).transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = rotate(q, k, cos, sin)
            keys = torch.cat([prefix_keys[i].transpose(1, 2).float(), k], 2)
            values = torch.cat([prefix_values[i].transpose(1, 2), v], 2)
            o = attend(q.to(dtype), keys.to(dtype), values, bias)
            h = h + gate_msa * attn.o_proj(o)
            hn = norm32(layer.post_layernorm, h) * (1 + scale_mlp) + shift_mlp
            h = h + gate_mlp * layer.mlp(hn)
        velocity = model.action_output_layer(h[:, -x.shape[1]:]).float()
        x = x + dt * velocity
        t_next = self.next_t[step]
        return x + weight * ((1 - t_next) * noise + t_next * seed - x)


def export(module, inputs, path, input_names, output_names, axes=None):
    torch.onnx.export(module,
                      inputs,
                      path,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=axes,
                      opset_version=17,
                      dynamo=False)
    return apply_trt103_workarounds(path)


def check(model, stage, graph, ref, steps, half):
    t = lambda k: torch.from_numpy(ref[f"{k}_0"]).cuda()
    err = lambda a, b: float((a.float() - b.float()).abs().max())
    with torch.no_grad():
        if stage == "vision":
            tokens, deep = graph(t("pixel_values"))
            print(
                f"vision {err(tokens, t('vision')):.2e} deepstack {err(deep, t('deepstack')):.2e} "
                f"(|ref| {float(t('vision').abs().max()):.1f})")
            return
        layers = model.config.vlm_config.text_config.num_hidden_layers
        a = Prefix(model, range(0, half), True).eval()
        b = Prefix(model, range(half, layers), False).eval()
        ids = t("input_ids").int()
        positions = t("position_ids")[:, 0].int()
        hidden, ka, va = a(ids, positions, t("vision"), t("deepstack"))
        _, kb, vb = b(hidden, positions)
        keys, values = torch.cat([ka, kb]), torch.cat([va, vb])
        for layer in (0, 17, 35):
            print(
                f"prefix layer {layer}: K {err(keys[layer].transpose(1, 2), t(f'k{layer}')):.2e} "
                f"V {err(values[layer].transpose(1, 2), t(f'v{layer}')):.2e} "
                f"(|K| {float(t(f'k{layer}').abs().max()):.1f})")
        if stage != "step":
            return
        x = t("noise")[None].float()
        noise = x.clone()
        start = positions.max().reshape(1) + 1
        mask = torch.ones(1, ids.shape[1], device="cuda")
        action_mask = t("action_mask").float()
        weight = torch.zeros(x.shape[1], 1, device="cuda")
        dt = torch.tensor([1.0 / steps], device="cuda")
        for i in range(steps):
            previous = x
            x = graph(x, torch.tensor([i], dtype=torch.int32, device="cuda"),
                      t("state").float(), keys, values, mask, start,
                      action_mask, dt, noise, noise, weight)
            if i == 0:
                print(
                    f"velocity step 0: {err((x - previous) / dt, t('velocity0')):.2e}"
                )
        print(f"chunk after {steps} steps: {err(x[0], t('normalized')):.2e}")


def prompt_ids(processor, robot_type, crop_ratio, state_width, tasks):
    """The prompt's ids before and after the instruction (the head carries every view's image placeholders), checked
    to tokenize as head + instruction + tail for every task."""
    views = np.zeros((3, 256, 256, 3), np.uint8)
    tokenizer = processor.tokenizer
    head = tail = None
    for task in tasks:
        ids = xr1_reference.build_request(
            processor, views, [0.0], task, robot_type, crop_ratio,
            state_width)["input_ids"][0].tolist()
        alone = tokenizer(task, add_special_tokens=False)["input_ids"]
        starts = [
            i for i in range(len(ids) - len(alone) + 1)
            if ids[i:i + len(alone)] == alone
        ]
        if len(starts) != 1:
            raise SystemExit(
                f"the prompt does not tokenize as head + instruction + tail for {task!r}"
            )
        h, t = ids[:starts[0]], ids[starts[0] + len(alone):]
        if head is not None and (h != head or t != tail):
            raise SystemExit(
                f"the prompt around {task!r} differs from the other instructions'"
            )
        head, tail = h, t
    return head, tail


def write_assets(args, model, processor):
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "special_tokens_map.json", "added_tokens.json",
                 "preprocessor_config.json", "chat_template.jinja"):
        source = os.path.join(args.checkpoint, name)
        if os.path.exists(source):
            shutil.copy(source, os.path.join(args.out, name))
    config = processor.action_config[args.robot_type]
    tasks = open(args.tasks).read().splitlines() if args.tasks else []
    head, tail = prompt_ids(
        processor, args.robot_type, args.crop_ratio,
        int(model.config.state_dim),
        tasks + ["pick the apple from the counter and place it in the sink"])
    image = processor.image_processor
    vlm = model.config.vlm_config
    noise = xr1_reference.official_noise(
        processor.get_action_mask(args.robot_type).shape, args.seed)
    json.dump(
        {
            "model_family":
            "xiaomi_robotics_1",
            "robot_type":
            args.robot_type,
            "views":
            3,
            "image_size":
            GRID[1] * int(image.patch_size),
            "crop_ratio":
            args.crop_ratio,
            "patch_size":
            int(image.patch_size),
            "temporal_patch_size":
            int(image.temporal_patch_size),
            "merge_size":
            int(image.merge_size),
            "image_mean": [float(v) for v in image.image_mean],
            "image_std": [float(v) for v in image.image_std],
            "image_token_id":
            int(vlm.image_token_id),
            "prompt_head_ids":
            head,
            "prompt_tail_ids":
            tail,
            "cameras": [
                "robot0_agentview_left", "robot0_agentview_right",
                "robot0_eye_in_hand"
            ],
            "layers":
            int(vlm.text_config.num_hidden_layers),
            "kv_heads":
            int(vlm.text_config.num_key_value_heads),
            "head_dim":
            int(vlm.text_config.head_dim),
            "chunk_size":
            int(config["std"].shape[-2]) if config["std"].dim() > 1 else 0,
            "max_dim":
            int(model.config.action_dim),
            "state_dim":
            int(model.config.state_dim),
            "steps":
            args.steps,
            "seed":
            args.seed,
            "noise":
            noise.reshape(-1).tolist(),
            "action_mean":
            config["mean"].reshape(-1).float().tolist(),
            "action_std":
            config["std"].reshape(-1).float().tolist(),
            "action_mask":
            processor.get_action_mask(
                args.robot_type).reshape(-1).float().tolist(),
        },
        open(os.path.join(args.out, "config.json"), "w"),
        indent=1)
    print(f"assets -> {args.out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["vision", "prefix_a", "prefix_b", "step", "assets"])
    parser.add_argument("--check", help="xr1_reference.py --capture output")
    parser.add_argument("--robot-type", default="robocasa_mg")
    parser.add_argument("--crop-ratio", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks",
        help=
        "assets: file of instructions, one per line, whose tokenization the runtime's split must match"
    )
    parser.add_argument("--max-prompt",
                        type=int,
                        default=384,
                        help="the prefix engines' longest prompt")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model, processor = xr1_reference.load_policy(args.checkpoint, "fp32")
    if args.stage == "assets":
        return write_assets(args, model, processor)
    views = 3
    layers = model.config.vlm_config.text_config.num_hidden_layers
    half = layers // 2
    ref = np.load(args.check) if args.check else None
    os.makedirs(os.path.join(args.out, args.stage), exist_ok=True)
    path = os.path.join(args.out, args.stage, "model.onnx")
    f16, gpu = torch.float16, dict(device="cuda")
    text = model.config.vlm_config.text_config
    hidden = text.hidden_size
    tokens = views * (GRID[1] * GRID[2]) // 4
    with torch.no_grad():
        if args.stage == "vision":
            graph = Vision(model.vlm.model.visual, views).eval()
            if ref is not None:
                check(model, "vision", graph, ref, args.steps, half)
            rewrites = export(
                graph.half(),
                (torch.randn(views * 256, 1536, dtype=f16, **gpu), ), path,
                ["pixels"], ["image_tokens", "deepstack"])
        elif args.stage in ("prefix_a", "prefix_b"):
            first = args.stage == "prefix_a"
            graph = Prefix(model,
                           range(0, half) if first else range(half, layers),
                           first).eval()
            if ref is not None:
                check(model, "prefix", graph, ref, args.steps, half)
            graph = graph.half()
            length = 256
            positions = torch.zeros(3, length, dtype=torch.int32, **gpu)
            if first:
                ids = torch.zeros(1, length, dtype=torch.int32, **gpu)
                ids[0, 10:10 + tokens] = int(
                    model.config.vlm_config.image_token_id)
                inputs = (ids, positions,
                          torch.randn(tokens, hidden, dtype=f16, **gpu),
                          torch.randn(3, tokens, hidden, dtype=f16, **gpu))
                names = [
                    "input_ids", "position_ids", "image_tokens", "deepstack"
                ]
                axes = {
                    "input_ids": {
                        1: "length"
                    },
                    "position_ids": {
                        1: "length"
                    }
                }
            else:
                inputs = (torch.randn(1, length, hidden, dtype=f16,
                                      **gpu), positions)
                names = ["hidden", "position_ids"]
                axes = {"hidden": {1: "length"}, "position_ids": {1: "length"}}
            axes.update({
                "hidden_out": {
                    1: "length"
                },
                "keys": {
                    2: "length"
                },
                "values": {
                    2: "length"
                }
            })
            rewrites = export(graph, inputs, path, names,
                              ["hidden_out", "keys", "values"], axes)
        else:
            graph = Step(model, args.steps).eval().cuda()
            if ref is not None:
                check(model, "step", graph, ref, args.steps, half)
            graph = graph.half()
            length = 256
            chunk = int(processor.get_action_mask(args.robot_type).shape[1])
            dim = int(model.config.action_dim)
            kv = (layers, 1, length, text.num_key_value_heads, text.head_dim)
            rewrites = export(
                graph,
                (torch.randn(1, chunk, dim, **
                             gpu), torch.zeros(1, dtype=torch.int32, **gpu),
                 torch.randn(
                     1, 1, int(model.config.state_dim), dtype=f16, **gpu),
                 torch.randn(*kv, dtype=f16, **
                             gpu), torch.randn(*kv, dtype=f16, **gpu),
                 torch.ones(1, length, dtype=f16, **
                            gpu), torch.zeros(1, dtype=torch.int32, **gpu),
                 torch.ones(1, chunk, dim, **gpu), torch.full(
                     (1, ), 0.2, **gpu), torch.randn(1, chunk, dim, **gpu),
                 torch.randn(1, chunk, dim, **gpu), torch.zeros(
                     chunk, 1, **gpu)), path, [
                         "x", "step", "state", "prefix_keys", "prefix_values",
                         "prefix_mask", "suffix_start", "action_mask", "dt",
                         "noise", "seed", "weight"
                     ], ["x_next"], {
                         "prefix_keys": {
                             2: "length"
                         },
                         "prefix_values": {
                             2: "length"
                         },
                         "prefix_mask": {
                             1: "length"
                         }
                     })
    print(f"exported {path}; TensorRT 10.3 rewrites {rewrites}")


if __name__ == "__main__":
    main()
