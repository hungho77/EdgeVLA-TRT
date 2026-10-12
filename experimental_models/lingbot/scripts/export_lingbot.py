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
"""Export LingBot-VLA 2.0 (Robbyant/lingbot-vla-v2) to FP16 ONNX graphs for TensorRT, from the official modules:

  vision.onnx    the processor's patches of every camera [cameras * 256, 1536] -> Qwen3-VL's ViT, merger and the
                 three deepstack mergers: image tokens [cameras, 64, 2560] and deepstack [3, cameras, 64, 2560].
  prefix_a.onnx  the instruction ids and mask [1, 72], the image tokens, the deepstack features and the multimodal
                 RoPE ids [3, P] -> the prefix (vision boundaries, image tokens, instruction, the depth / video
                 alignment queries) through language layers 0-17: hidden [1, P, 2560] and K / V [18, 1, P, 8, 128].
  prefix_b.onnx  the hidden states, mask and RoPE ids -> layers 18-35's K / V.
  step.onnx      one Euler step of the action expert: the chunk [1, 50, 55] (FP32), the step index, the padded
                 state, every layer's prefix K / V, the prefix mask, the first suffix RoPE position, dt and the
                 real-time-chunking inputs (the initial noise, the seed chunk and a per-row weight [50, 1]) -> the
                 next chunk. Rows with weight w move towards t * noise + (1 - t) * seed by w after the step, as
                 pi0.5's inpainting does; weight 0 leaves the official step. The expert's 36 layers attend jointly to the
                 prefix K / V and their own state / action tokens; every MoE layer evaluates all its experts
                 and mixes them with the top-k routing weights (the router in FP32).

Every RMS / layer norm, the RoPE rotation and the attention (scores, softmax and the weighted values) run in FP32, as
the official joint attention does. ``--check`` first runs these graphs in PyTorch FP32 against the captures of
``lingbot_reference.py --capture``. Each stage is written to OUT/<stage>/model.onnx (the prefix halves and the step
carry external weights); ``--stage assets`` writes the runtime's config.json and tokenizer into OUT, checking that
the chat-template prompt tokenizes as template + instruction for every line of ``--tasks``.

    python export_lingbot.py --repo lingbot-vla-v2 --checkpoint lingbot-vla-v2-6b-libero \\
        --qwen3vl Qwen3-VL-4B-Instruct --out onnx --stage vision|prefix_a|prefix_b|step [--check capture.npz]
    python export_lingbot.py ... --out engines --stage assets [--tasks instructions.txt]
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
import lingbot_reference  # noqa: E402

from tensorrt_edgellm.onnx.trt_workarounds import \
    apply_trt103_workarounds  # noqa: E402

# Additive mask for excluded keys; finite, so a fully masked row (a padded query) averages instead of producing NaN,
# as the official finite fill does.
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
    """Grouped-query attention over [B, L, heads, D] tensors in FP32; returns [B, Lq, heads * D] in q's dtype."""
    groups = q.shape[2] // k.shape[2]
    q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))
    if groups > 1:
        k, v = (t[:, :, None].expand(-1, -1, groups, -1, -1).flatten(1, 2)
                for t in (k, v))
    scores = (q * q.shape[-1]**-0.5) @ k.transpose(-1, -2)
    if bias is not None:
        scores = scores + bias
    out = scores.softmax(-1) @ v
    return out.transpose(1, 2).flatten(2)


def use_dense_onnx_moe():
    """Every expert evaluated on every token, mixed with the official top-k weights: the routed set per layer covers
    28 of the 32 experts on average over the 51 suffix tokens, so this reads no more weights than a sparse kernel.
    The experts and the shared expert are concatenated along the intermediate dim (see prepare_moe) and each
    expert's intermediate block is scaled by its routing weight before the down projection, which is linear: three
    wide GEMMs per layer instead of 32 small ones."""
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import \
        Qwen2TokenMoeBlock

    def forward(self, hidden_states):
        batch, length, dim = hidden_states.shape
        x = hidden_states.reshape(-1, dim)
        scores = (x.float() @ self.gate.weight.float().t()).sigmoid()
        _, selected = torch.topk(scores + self.e_score_correction_bias[None],
                                 self.top_k,
                                 dim=-1)
        weights = torch.gather(scores, 1, selected)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        ids = torch.arange(self.num_experts, device=x.device)[None, None]
        dense = ((ids == selected[..., None]).float() *
                 weights[..., None]).sum(1)  # [tokens, experts]
        scale = torch.cat([
            dense[..., None].expand(-1, -1, self.expert_width).flatten(1),
            dense.new_ones(x.shape[0], self.shared_width)
        ], 1)
        h = F.silu(F.linear(x, self.gate_all)) * F.linear(x, self.up_all)
        out = F.linear((h.float() * scale).to(x.dtype), self.down_all)
        return out.reshape(batch, length, dim), None

    Qwen2TokenMoeBlock.forward = forward


def prepare_moe(model):
    """Concatenate each MoE block's experts and shared expert into [experts * I + I_shared, D] gate / up and
    [D, experts * I + I_shared] down weights (the originals are released: the FP32 model fills most of the board)."""
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import \
        Qwen2TokenMoeBlock
    for module in model.modules():
        if isinstance(module, Qwen2TokenMoeBlock):
            if getattr(module, "_use_shared_expert_gate", False):
                raise SystemExit("a gated shared expert is not supported")
            experts, shared = module.experts, module.shared_expert
            count, width, dim = experts.gate_proj.shape
            with torch.no_grad():
                gate = torch.cat([
                    experts.gate_proj.reshape(count * width, dim),
                    shared.gate_proj.weight
                ])
                up = torch.cat([
                    experts.up_proj.reshape(count * width, dim),
                    shared.up_proj.weight
                ])
                down = torch.cat([
                    experts.down_proj.permute(1, 0, 2).reshape(
                        dim, count * width), shared.down_proj.weight
                ], 1)
            module.experts = None
            module.shared_expert = None
            module.gate_all = nn.Parameter(gate, requires_grad=False)
            module.up_all = nn.Parameter(up, requires_grad=False)
            module.down_all = nn.Parameter(down, requires_grad=False)
            module.expert_width = width
            module.shared_width = shared.gate_proj.weight.shape[0]


class Vision(nn.Module):
    """Qwen3VLVisionModel at a fixed grid per camera: position embeddings and RoPE precomputed, attention within each
    image (the varlen split of the official path)."""

    def __init__(self, visual, cameras):
        super().__init__()
        grid = torch.tensor([GRID] * cameras,
                            device=visual.patch_embed.proj.weight.device)
        with torch.no_grad():
            pos = visual.fast_pos_embed_interpolate(grid)
            _, (cos, sin), _, _, _ = visual.preprcess_grid_thw(grid)
        self.register_buffer("pos", pos.float())
        self.register_buffer("cos", cos.float())
        self.register_buffer("sin", sin.float())
        self.visual = visual
        self.cameras = cameras

    def forward(self, pixels):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import \
            apply_rotary_pos_emb_vision
        visual = self.visual
        x = visual.patch_embed(pixels) + self.pos.to(pixels.dtype)
        features = []
        for index, block in enumerate(visual.blocks):
            attn = block.attn
            h = norm32(block.norm1, x)
            dim = h.shape[-1]
            w, b = attn.qkv.weight, attn.qkv.bias
            q, k, v = (F.linear(h, w[i * dim:(i + 1) * dim],
                                b[i * dim:(i + 1) * dim]).unflatten(
                                    -1, (attn.num_heads, -1))
                       for i in range(3))
            q, k = apply_rotary_pos_emb_vision(q, k, self.cos, self.sin)
            per_image = lambda t: t.reshape(self.cameras, -1, *t.shape[1:])
            o = attend(per_image(q), per_image(k), per_image(v))
            x = x + attn.proj(o.reshape(x.shape[0], -1).to(x.dtype))
            x = x + block.mlp(norm32(block.norm2, x))
            if index in visual.deepstack_visual_indexes:
                merger = visual.deepstack_merger_list[
                    visual.deepstack_visual_indexes.index(index)]
                features.append(merger(x))
        tokens = visual.merger(x)
        shape = (self.cameras, -1, tokens.shape[-1])
        return tokens.reshape(shape), torch.stack(
            [f.reshape(shape) for f in features])


class Prefix(nn.Module):
    """The prefix through a range of Qwen3-VL language layers, keeping each layer's post-RoPE K / V."""

    def __init__(self, policy, layers, cameras, first):
        super().__init__()
        model = policy.model
        qe = model.qwenvl_with_expert
        self.language = qe.qwenvl.model.language_model
        self.layers = nn.ModuleList(self.language.layers[i] for i in layers)
        self.first = first
        self.deepstack_layers = len(qe.qwenvl.visual.deepstack_visual_indexes)
        self.cameras = cameras
        config = qe.qwenvl.config
        with torch.no_grad():
            embed = self.language.embed_tokens.weight
            self.register_buffer("vision_start",
                                 embed[config.vision_start_token_id].clone())
            self.register_buffer("vision_end",
                                 embed[config.vision_end_token_id].clone())
            self.register_buffer("align", alignment_queries(model).clone())

    def assemble(self, input_ids, image_tokens):
        parts = []
        for c in range(self.cameras):
            parts += [
                self.vision_start[None], image_tokens[c], self.vision_end[None]
            ]
        lang = self.language.embed_tokens(input_ids)[0]
        return torch.cat(parts + [lang, self.align.to(lang.dtype)])[None]

    def deepstack_add(self, deepstack, layer, length):
        rows = []
        zero = deepstack.new_zeros(1, deepstack.shape[-1])
        for c in range(self.cameras):
            rows += [zero, deepstack[layer, c], zero]
        visual = torch.cat(rows)
        return F.pad(visual, (0, 0, 0, length - visual.shape[0]))[None]

    def forward(self,
                hidden_or_ids,
                lang_mask,
                position_ids,
                image_tokens=None,
                deepstack=None):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import \
            apply_rotary_pos_emb
        x = self.assemble(hidden_or_ids,
                          image_tokens) if self.first else hidden_or_ids
        length = x.shape[1]
        images = length - lang_mask.shape[1] - self.align.shape[0]
        valid = torch.cat([
            lang_mask.new_ones(1, images), lang_mask,
            lang_mask.new_ones(1, self.align.shape[0])
        ], 1).float() > 0.5
        causal = torch.ones(length, length, dtype=torch.bool,
                            device=x.device).tril()
        allowed = causal[None] & valid[:, None, :] & valid[:, :, None]
        bias = torch.where(allowed, 0.0, MASKED)[:, None]
        cos, sin = self.language.rotary_emb(x.float(), position_ids[:, None])
        keys, values = [], []
        for i, layer in enumerate(self.layers):
            q, k, v = layer(x, compute_kqv=True)
            q, k = apply_rotary_pos_emb(q.float(),
                                        k.float(),
                                        cos,
                                        sin,
                                        unsqueeze_dim=2)
            keys.append(k.to(x.dtype))
            values.append(v)
            o = attend(q, k, v, bias)
            x = layer(x, o.to(x.dtype), 0, length, output_atten=True)
            if self.first and i < self.deepstack_layers:
                x = x + self.deepstack_add(deepstack, i, length)
        return x, torch.stack(keys), torch.stack(values)


def alignment_queries(model):
    """The prefix's alignment query tokens after the instruction: learned constants (see embed_prefix)."""
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import \
        prefix_query_segments

    def task_tokens(tokens):
        return tokens.view(model.num_task_tokens,
                           tokens.shape[0] // model.num_task_tokens,
                           tokens.shape[1]).mean(1)

    current = task_tokens(model.depth_align_embs)
    if getattr(model, "use_future_video", False) and getattr(
            model, "use_current_video_patch", False) and getattr(
                model, "use_current_shared_task_proj", False):
        current = model.current_shared_task_proj(
            torch.cat(
                [current, task_tokens(model.current_video_align_embs)], -1))
    future = None
    if model.use_future_depth:
        future = task_tokens(model.future_depth_align_embs)
        if getattr(model, "use_future_video", False) and getattr(
                model, "use_future_video_patch", True) and getattr(
                    model, "future_video_share_future_depth_query",
                    False) and getattr(model, "use_shared_future_task_proj",
                                       False):
            future = model.future_shared_task_proj(
                torch.cat([future,
                           task_tokens(model.future_video_align_embs)], -1))
    segments = []
    for name in prefix_query_segments(
            use_depth_align=True,
            use_future_depth=model.use_future_depth,
            use_future_video=getattr(model, "use_future_video", False),
            use_future_video_cls=getattr(model, "use_future_video_cls", False),
            use_future_video_patch=getattr(model, "use_future_video_patch",
                                           True),
            future_video_share_future_depth_query=getattr(
                model, "future_video_share_future_depth_query", False)):
        if name == "language":
            continue
        if name == "current_depth":
            segments.append(current)
        elif name == "future_depth":
            segments.append(future)
        elif name == "future_video":
            segments.append(task_tokens(model.future_video_align_embs))
        elif name == "future_video_cls":
            segments.append(model.future_video_cls_align_emb.weight)
        else:
            raise SystemExit(f"unsupported prefix segment {name}")
    return torch.cat(segments).float()


def time_table(steps):
    """The flow times the official loop visits (t = 1, then t += -1 / steps in FP32) and their sinusoidal
    embeddings, as embed_suffix computes them."""
    dt = torch.tensor(-1.0 / steps, dtype=torch.float32)
    t = torch.tensor(1.0, dtype=torch.float32)
    times = []
    while t >= -dt / 2:
        times.append(t.clone())
        t += dt
    times = torch.stack(times)
    return times, dt


class Step(nn.Module):

    def __init__(self, policy, prefix_length):
        super().__init__()
        from lingbotvla.models.vla.lingbot_vla.utils import \
            create_sinusoidal_pos_embedding
        model = policy.model
        self.model = model
        qe = model.qwenvl_with_expert
        self.expert = qe.qwen_expert.model
        self.rotary = qe.qwenvl.model.language_model.rotary_emb
        self.chunk = model.config.n_action_steps
        times, dt = time_table(model.config.num_steps)
        with torch.no_grad():
            table = create_sinusoidal_pos_embedding(times,
                                                    model.config.proj_width,
                                                    min_period=4e-3,
                                                    max_period=4.0)
        self.register_buffer("time_table", table.float())
        # The time after each step, for the inpainting target.
        self.register_buffer(
            "next_time",
            torch.cat([times[1:], (times[-1] + dt)[None]]).float())
        suffix = self.chunk + 1
        # Suffix blocks: the state token sees itself, the action tokens see the state and each other.
        block = torch.tensor([1] + [2] * self.chunk)
        allowed = block[None, :] <= block[:, None]
        self.register_buffer("suffix_bias",
                             torch.where(allowed, 0.0, MASKED)[None, None])
        self.register_buffer("offsets", torch.arange(suffix))

    def forward(self, x, step, state, prefix_keys, prefix_values, prefix_mask,
                suffix_start, dt, noise, seed, weight):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import \
            apply_rotary_pos_emb
        model = self.model
        dtype = prefix_keys.dtype
        time = self.time_table[step].to(dtype)  # [1, width]
        state_emb = model.state_proj(state)
        action_emb = model.action_in_proj(x.to(dtype))
        fused = torch.cat(
            [action_emb, time[:, None].expand(-1, self.chunk, -1)], -1)
        fused = model.action_time_mlp_out(
            F.silu(model.action_time_mlp_in(fused)))
        h = torch.cat([state_emb[:, None], fused], 1)
        positions = (suffix_start + self.offsets)[None, None].expand(3, 1, -1)
        cos, sin = self.rotary(h.float(), positions)
        prefix_bias = torch.where(prefix_mask > 0.5, 0.0, MASKED)[:, None,
                                                                  None, :]
        bias = torch.cat(
            [prefix_bias.expand(-1, -1, h.shape[1], -1), self.suffix_bias], -1)
        for i, layer in enumerate(self.expert.layers):
            q, k, v = layer(h, compute_kqv=True, ada_cond=time)
            q, k = apply_rotary_pos_emb(q.float(),
                                        k.float(),
                                        cos,
                                        sin,
                                        unsqueeze_dim=2)
            keys = torch.cat([prefix_keys[i].float(), k], 1)
            values = torch.cat([prefix_values[i], v], 1)
            o = attend(q, keys, values, bias)
            h, _ = layer(h,
                         o.to(dtype),
                         0,
                         h.shape[1],
                         output_atten=True,
                         ada_cond=time)
        h = self.expert.norm(h)
        velocity = model.action_out_proj(h[:, -self.chunk:]).float()
        x = x + dt * velocity
        t = self.next_time[step]
        return x + weight * (t * noise + (1 - t) * seed - x)


def export(module, inputs, path, input_names, output_names):
    torch.onnx.export(module,
                      inputs,
                      path,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=17,
                      dynamo=False)
    return apply_trt103_workarounds(path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen3vl", required=True)
    parser.add_argument("--train-config")
    parser.add_argument("--robot-config")
    parser.add_argument("--norm-stats")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["vision", "prefix_a", "prefix_b", "step", "assets"])
    parser.add_argument("--check",
                        help="lingbot_reference.py --capture output")
    parser.add_argument(
        "--tasks",
        help=
        "assets: file of instructions, one per line, whose tokenization the runtime's split must match"
    )
    args = parser.parse_args()
    training = os.path.join(args.checkpoint, "training")
    args.train_config = args.train_config or os.path.join(
        training, "libero_full.yaml")
    args.robot_config = args.robot_config or os.path.join(
        training, "libero.yaml")
    args.norm_stats = args.norm_stats or os.path.join(training,
                                                      "norm_stats.json")
    args.precision = "fp32"
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.repo)
    policy, transform, config, data_config = lingbot_reference.load_policy(
        args)
    # After load_policy, which installs the reference's FP32 einsum MoE.
    use_dense_onnx_moe()
    prepare_moe(policy)
    qe = policy.model.qwenvl_with_expert
    cameras = len(transform.org_features["images"])
    layers = qe.qwenvl.config.text_config.num_hidden_layers
    half = layers // 2
    prefix_length = cameras * (GRID[1] * GRID[2] // 4 + 2) + int(
        config.tokenizer_max_length) + alignment_queries(policy.model).shape[0]
    if args.stage == "assets":
        return write_assets(args, policy, transform, config, data_config,
                            cameras, prefix_length)
    ref = np.load(args.check) if args.check else None
    t = (lambda k: torch.from_numpy(ref[f"{k}_0"]).cuda()
         ) if ref is not None else None
    os.makedirs(os.path.join(args.out, args.stage), exist_ok=True)
    path = os.path.join(args.out, args.stage, "model.onnx")
    f16 = torch.float16
    gpu = dict(device="cuda")
    with torch.no_grad():
        if args.stage == "vision":
            graph = Vision(qe.qwenvl.visual, cameras).eval()
            if ref is not None:
                tokens, deep = graph(t("pixels"))
                print(
                    f"vision {float((tokens - t('vision')).abs().max()):.2e}"
                    f" deepstack {float((deep - t('deepstack')).abs().max()):.2e}"
                    f" (|ref| {float(t('vision').abs().max()):.1f})")
            graph = graph.half()
            rewrites = export(
                graph, (torch.randn(cameras * 256, 1536, dtype=f16, **gpu), ),
                path, ["pixels"], ["image_tokens", "deepstack"])
        elif args.stage in ("prefix_a", "prefix_b"):
            first = args.stage == "prefix_a"
            span = range(0, half) if first else range(half, layers)
            graph = Prefix(policy, span, cameras, first).eval()
            if ref is not None:
                check_prefix(policy, cameras, t, half, layers)
            graph = graph.half()
            ids = torch.zeros(1,
                              int(config.tokenizer_max_length),
                              dtype=torch.int32,
                              **gpu)
            mask = torch.ones(1,
                              int(config.tokenizer_max_length),
                              dtype=f16,
                              **gpu)
            positions = torch.zeros(3, prefix_length, dtype=torch.int32, **gpu)
            hidden = qe.qwenvl.config.text_config.hidden_size
            if first:
                inputs = (ids, mask, positions,
                          torch.randn(cameras, 64, hidden, dtype=f16, **gpu),
                          torch.randn(3, cameras, 64, hidden, dtype=f16,
                                      **gpu))
                names = [
                    "input_ids", "lang_mask", "position_ids", "image_tokens",
                    "deepstack"
                ]
            else:
                inputs = (torch.randn(1,
                                      prefix_length,
                                      hidden,
                                      dtype=f16,
                                      **gpu), mask, positions)
                names = ["hidden", "lang_mask", "position_ids"]
            rewrites = export(graph, inputs, path, names,
                              ["hidden_out", "keys", "values"])
        else:
            graph = Step(policy, prefix_length).eval().cuda()
            if ref is not None:
                check_step(policy, graph, t, config)
            graph = graph.half()
            text = qe.qwenvl.config.text_config
            kv = (layers, 1, prefix_length, text.num_key_value_heads,
                  text.head_dim)
            chunk, dim = config.n_action_steps, config.max_action_dim
            rewrites = export(
                graph,
                (torch.randn(1, chunk, dim, **
                             gpu), torch.zeros(1, dtype=torch.int32, **gpu),
                 torch.randn(1, dim, dtype=f16, **
                             gpu), torch.randn(*kv, dtype=f16, **gpu),
                 torch.randn(*kv, dtype=f16, **gpu),
                 torch.ones(1, prefix_length, dtype=f16, **
                            gpu), torch.zeros(1, dtype=torch.int32, **gpu),
                 torch.full(
                     (1, ), -0.1, **gpu), torch.randn(1, chunk, dim, **gpu),
                 torch.randn(1, chunk, dim, **gpu), torch.zeros(
                     chunk, 1, **gpu)), path, [
                         "x", "step", "state", "prefix_keys", "prefix_values",
                         "prefix_mask", "suffix_start", "dt", "noise", "seed",
                         "weight"
                     ], ["x_next"])
    print(f"exported {path}; TensorRT 10.3 rewrites {rewrites}")


def check_prefix(policy, cameras, t, half, layers):
    """Both halves in PyTorch FP32 on the reference's own embedding inputs, against its K / V captures."""
    policy.model.qwenvl_with_expert
    a = Prefix(policy, range(0, half), cameras, True).eval()
    b = Prefix(policy, range(half, layers), cameras, False).eval()
    ids = t("lang_tokens")[None].int()
    mask = t("lang_masks")[None].float()
    positions = t("position_ids")[:, 0].int()
    hidden, ka, va = a(ids, mask, positions, t("vision"), t("deepstack"))
    _, kb, vb = b(hidden, mask, positions)
    keys = torch.cat([ka, kb]).float()
    values = torch.cat([va, vb]).float()
    valid = t("prefix_pad")[0].bool()
    for layer in (0, 17, 35):
        dk = (keys[layer, 0][valid] - t(f"k{layer}")[0][valid]).abs().max()
        dv = (values[layer, 0][valid] - t(f"v{layer}")[0][valid]).abs().max()
        print(f"prefix layer {layer}: K {float(dk):.2e} V {float(dv):.2e} "
              f"(|K| {float(t(f'k{layer}').abs().max()):.1f})")
    return keys, values


def check_step(policy, graph, t, config):
    keys, values = check_prefix(policy, 2, t, config.num_hidden_layers // 2,
                                config.num_hidden_layers)
    _, dt = time_table(config.num_steps)
    x = t("noise")[None].float()
    mask = t("prefix_pad").float()
    start = t("position_ids")[:, 0].max().int().reshape(1) + 1
    weight = torch.zeros(config.n_action_steps, 1, device="cuda")
    noise = x.clone()
    for i in range(config.num_steps):
        x = graph(x, torch.tensor([i], dtype=torch.int32, device="cuda"),
                  t("state")[None], keys, values, mask, start,
                  dt.cuda().reshape(1), noise, noise, weight)
        if i == 0:
            v0 = (x - t("noise")[None]) / dt.cuda()
            print(
                f"velocity step 0: {float((v0 - t('velocity0')).abs().max()):.2e}"
            )
    print(
        f"chunk after {config.num_steps} steps: {float((x[0] - t('normalized')).abs().max()):.2e}"
    )


def feature_layout(transform, data_config, robot_config, kind):
    """Where each robot feature sits in the model's padded vector, from the robot config's origin slices and the
    training config's joint widths: [{source, start, count, slot, low, high}] with the bounds_99 statistics."""
    import ast

    import yaml
    robot = yaml.safe_load(open(robot_config))
    prefix = "observation.state." if kind == "states" else "action."
    widths = transform.feature_config.joints_max_dim
    slots, offset = {}, 0
    for joint in transform.feature_config.joints:
        slots[joint] = offset
        offset += widths[joint]
    norm_types = {
        k: v
        for d in data_config.norm_type
        for k, v in ast.literal_eval(d).items()
    }
    stats = transform.normalizer.norm_stats
    layout = []
    for entry in robot[kind]:
        (feature, spec), = entry.items()
        joint = feature.split(prefix)[-1]
        if norm_types[joint] not in ("bounds_99_woclip", "bounds_99"):
            raise SystemExit(
                f"unsupported normalization {norm_types[joint]} for {feature}")
        (source, span), = spec["origin_keys"][0].items()
        low = [float(v) for v in stats[feature]["q01"]]
        high = [float(v) for v in stats[feature]["q99"]]
        layout.append({
            "source": source,
            "start": int(span["start"]),
            "count": int(span["end"]) - int(span["start"]),
            "slot": slots[joint],
            "low": low,
            "high": high,
            "clip": norm_types[joint] == "bounds_99",
        })
    return layout


def prompt_ids(tokenizer, tasks):
    """The chat template's ids around the instruction, checked to tokenize as prefix + instruction + suffix."""
    marker = "\x00"
    text = tokenizer.apply_chat_template([{
        "role": "user",
        "content": marker
    }],
                                         tokenize=False,
                                         add_generation_prompt=False)
    head, tail = text.split(marker)
    prefix = tokenizer(head, add_special_tokens=False)["input_ids"]
    suffix = tokenizer(tail, add_special_tokens=False)["input_ids"]
    for task in tasks:
        joint = tokenizer(head + task + tail,
                          add_special_tokens=False)["input_ids"]
        split = prefix + tokenizer(
            task, add_special_tokens=False)["input_ids"] + suffix
        if joint != split:
            raise SystemExit(
                f"the prompt does not tokenize as template + instruction for {task!r}"
            )
    return prefix, suffix


def write_assets(args, policy, transform, config, data_config, cameras,
                 prefix_length):
    import yaml
    qe = policy.model.qwenvl_with_expert
    text = qe.qwenvl.config.text_config
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "preprocessor_config.json", "chat_template.json",
                 "vocab.json", "merges.txt"):
        source = os.path.join(args.qwen3vl, name)
        if os.path.exists(source):
            shutil.copy(source, os.path.join(args.out, name))
    times, dt = time_table(config.num_steps)
    tokenizer = transform.tokenizer
    if not getattr(config, "use_qwen3_chat_template", False):
        raise SystemExit("only the Qwen3 chat-template prompt is supported")
    tasks = open(args.tasks).read().splitlines() if args.tasks else []
    head, tail = prompt_ids(tokenizer, tasks)
    robot = yaml.safe_load(open(args.robot_config))
    origin = {}
    for entry in robot["images"]:
        (feature,
         source), = entry.items() if isinstance(entry, dict) else ((entry,
                                                                    entry), )
        origin[feature] = source["origin_keys"] if isinstance(source,
                                                              dict) else source
    processor = json.load(
        open(os.path.join(args.qwen3vl, "preprocessor_config.json")))
    json.dump(
        {
            "model_family":
            "lingbot_vla_v2",
            "cameras": [origin[f] for f in transform.feature_config.images],
            "image_size":
            int(getattr(data_config, "img_size", 256)),
            "patch_size":
            int(processor["patch_size"]),
            "temporal_patch_size":
            int(processor["temporal_patch_size"]),
            "merge_size":
            int(processor["merge_size"]),
            "image_mean":
            processor["image_mean"],
            "image_std":
            processor["image_std"],
            "prefix_length":
            prefix_length,
            "text_length":
            int(config.tokenizer_max_length),
            "prompt_prefix_ids":
            head,
            "prompt_suffix_ids":
            tail,
            "pad_id":
            int(tokenizer.pad_token_id),
            "align_tokens":
            int(alignment_queries(policy.model).shape[0]),
            "layers":
            int(text.num_hidden_layers),
            "kv_heads":
            int(text.num_key_value_heads),
            "head_dim":
            int(text.head_dim),
            "chunk_size":
            int(config.n_action_steps),
            "max_dim":
            int(config.max_action_dim),
            "steps":
            int(config.num_steps),
            "dt":
            float(dt),
            "state":
            feature_layout(transform, data_config, args.robot_config,
                           "states"),
            "actions":
            feature_layout(transform, data_config, args.robot_config,
                           "actions"),
        },
        open(os.path.join(args.out, "config.json"), "w"),
        indent=1)
    print(f"assets -> {args.out}")


if __name__ == "__main__":
    main()
