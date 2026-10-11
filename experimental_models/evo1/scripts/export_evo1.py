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
"""Export Evo-1 (MINT-SJTU/Evo-1) to four FP16 ONNX graphs for TensorRT, from the official modules' weights:

  vision.onnx   ImageNet-normalized views [V, 3, 448, 448] -> InternViT-300M, pixel shuffle and the projector:
                [V, 256, 896] image tokens.
  prefix.onnx   token ids [1, L], the image tokens of every view [3 * 256, 896] and the attention mask [1, L] ->
                the truncated Qwen2.5 (the checkpoint's 14 layers and final norm) over the whole padded sequence:
                hidden states [1, L, 896]. The image tokens replace the <IMG_CONTEXT> ids in order.
  context.onnx  the hidden states and the normalized state [1, 24] -> the action head's cross-attention keys and
                values over the 1024 + 1 context tokens, split into heads: kv_<b> [2 (key, value), 1, heads, L + 1,
                head dim] for every block b. They do not change across the denoising steps.
  step.onnx     one Euler step: the flattened chunk [1, H * 24] (FP32), the time-table row index [1], the action-dim
                mask [24], a per-row velocity scale [H, 1] (1, or real-time chunking's strengths), dt [1] and the
                keys / values -> the next chunk [1, H * 24].

The official head attends to every context token, the right padding and the masked-out black view included, so the
language model runs on the full padded length and its output is taken at every position; at the positions the mask
excludes the attention output is zero, as with the official setup's FlashAttention-2 (``--attention``, see
evo1_reference.py). Attention is computed with
separate query / key / value projections, the scale on the query, and a softmax in FP32; layer and RMS norms run in
FP32. ``--check obs-reference.npz`` (``evo1_reference.py --precision fp32``) first runs these graphs in PyTorch FP32
against the official policy's captured tensors.

    python export_evo1.py --repo Evo-1 --checkpoint Evo1_LIBERO --vlm InternVL3-1B --out evo1_onnx \\
        [--check ref_fp32.npz]
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
import evo1_reference  # noqa: E402

from tensorrt_edgellm.onnx.trt_workarounds import \
    apply_trt103_workarounds  # noqa: E402

# Additive mask for excluded keys; finite, so that the FP32 softmax never subtracts infinities.
MASKED = -1e30


def norm32(norm, x):
    """``norm`` (LayerNorm or RMSNorm) evaluated in FP32, returned in ``x``'s dtype."""
    if isinstance(norm, nn.LayerNorm):
        y = F.layer_norm(x.float(), norm.normalized_shape, norm.weight.float(),
                         norm.bias.float(), norm.eps)
    else:
        h = x.float()
        y = h * torch.rsqrt(
            h.pow(2).mean(-1, keepdim=True) +
            norm.variance_epsilon) * norm.weight.float()
    return y.to(x.dtype)


def attend(q, k, v, scale, bias=None):
    """softmax(q k^T * scale + bias) v over [B, heads, N, D] tensors, scaled on the query, softmax in FP32."""
    scores = ((q * scale) @ k.transpose(-1, -2)).float()
    if bias is not None:
        scores = scores + bias
    return scores.softmax(-1).to(v.dtype) @ v


class Vision(nn.Module):
    """InternVLChatModel.extract_feature at 448 x 448 (select_layer -1, pixel shuffle v2, mlp1)."""

    def __init__(self, internvl):
        super().__init__()
        vision = internvl.vision_model
        embeddings = vision.embeddings
        self.patch = embeddings.patch_embedding
        grid = embeddings.image_size // embeddings.patch_size
        with torch.no_grad():
            position = torch.cat([
                embeddings.position_embedding[:, :1],
                embeddings._get_pos_embed(embeddings.position_embedding[:, 1:],
                                          grid, grid)
            ], 1)
        self.register_buffer("class_token",
                             embeddings.class_embedding + position[:, :1])
        self.register_buffer("position", position[:, 1:].clone())
        self.layers = vision.encoder.layers
        self.heads = vision.config.num_attention_heads
        self.grid = grid
        self.mlp1 = internvl.mlp1

    def forward(self, pixels):
        x = self.patch(pixels).flatten(2).transpose(1, 2) + self.position
        x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], 1)
        for layer in self.layers:
            attn = layer.attn
            h = norm32(layer.norm1, x)
            dim = h.shape[-1]
            w, b = attn.qkv.weight, attn.qkv.bias
            q, k, v = (F.linear(
                h, w[i * dim:(i + 1) * dim],
                None if b is None else b[i * dim:(i + 1) * dim]).unflatten(
                    -1, (self.heads, -1)).transpose(1, 2) for i in range(3))
            o = attend(q, k, v, attn.scale).transpose(1, 2).flatten(2)
            x = x + attn.proj(o) * layer.ls1
            x = x + layer.mlp(norm32(layer.norm2, x)) * layer.ls2
        x = x[:, 1:]
        g, c = self.grid, x.shape[-1]
        # pixel_shuffle(scale 0.5, ps_version v2) on [N, W, H, C]
        x = x.reshape(-1, g, g // 2, c * 2).permute(0, 2, 1, 3)
        x = x.reshape(-1, g // 2, g // 2, c * 4).permute(0, 2, 1, 3)
        return self.mlp1(x.reshape(-1, (g // 2)**2, c * 4))


class Prefix(nn.Module):
    """The embedder's language model on inputs_embeds over the padded sequence, final norm included."""

    def __init__(self, embedder, length, flash):
        super().__init__()
        self.flash = flash
        language = embedder.model.language_model.model
        self.embed = language.embed_tokens
        self.layers = language.layers
        self.norm = language.norm
        self.image_token = embedder.tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>")
        config = language.config
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.heads
        positions = torch.arange(length)[None]
        with torch.no_grad():
            cos, sin = language.rotary_emb(torch.zeros(1, dtype=torch.float32),
                                           positions)
        self.register_buffer("cos", cos[0].float())
        self.register_buffer("sin", sin[0].float())
        causal = torch.full((length, length), MASKED).triu(1)
        self.register_buffer("causal", causal)

    def rope(self, x):
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
        return (x.float() * self.cos + rotated.float() * self.sin).to(x.dtype)

    def forward(self, input_ids, visual, attention):
        is_image = input_ids == self.image_token
        index = (torch.cumsum(is_image.to(torch.int32), 1) - 1).clamp(min=0)
        x = torch.where(is_image[..., None], visual[index[0]][None],
                        self.embed(input_ids))
        bias = self.causal + torch.where(attention > 0.5, 0.0, MASKED)[:, None,
                                                                       None, :]
        for layer in self.layers:
            attn = layer.self_attn
            h = norm32(layer.input_layernorm, x)
            q = attn.q_proj(h).unflatten(-1, (self.heads, -1)).transpose(1, 2)
            k = attn.k_proj(h).unflatten(-1,
                                         (self.kv_heads, -1)).transpose(1, 2)
            v = attn.v_proj(h).unflatten(-1,
                                         (self.kv_heads, -1)).transpose(1, 2)
            repeat = self.heads // self.kv_heads
            k = self.rope(k).repeat_interleave(repeat, 1)
            v = v.repeat_interleave(repeat, 1)
            o = attend(self.rope(q), k, v, self.head_dim**-0.5, bias)
            if self.flash:
                o = o * attention[:, None, :, None].to(o.dtype)
            x = x + attn.o_proj(o.transpose(1, 2).flatten(2))
            x = x + layer.mlp(norm32(layer.post_attention_layernorm, x))
        return norm32(self.norm, x)


def linear(category_linear):
    return category_linear.linear


class Context(nn.Module):
    """The state token appended to the hidden states, and every block's cross-attention keys and values."""

    def __init__(self, head):
        super().__init__()
        self.state_fc1 = linear(head.state_encoder.fc1)
        self.state_fc2 = linear(head.state_encoder.fc2)
        self.blocks = head.transformer_blocks

    def forward(self, hidden, state):
        state_token = self.state_fc2(F.relu(self.state_fc1(state)))
        context = torch.cat([hidden, state_token[:, None]], 1)
        kv = []
        for block in self.blocks:
            dim, heads = block.attn.embed_dim, block.attn.num_heads
            w, b = block.attn.in_proj_weight, block.attn.in_proj_bias
            split = lambda t: t.unflatten(-1, (heads, -1)).transpose(1, 2)
            kv.append(
                torch.stack([
                    split(F.linear(context, w[rows], b[rows]))
                    for rows in (slice(dim, 2 * dim), slice(2 * dim, None))
                ]))
        return tuple(kv)


class Step(nn.Module):
    """One Euler step of FlowmatchingActionHead.get_action (embodiment 0), the velocity scaled per row."""

    def __init__(self, head):
        super().__init__()
        self.head = head
        self.horizon = head.horizon
        self.per_action_dim = head.per_action_dim
        self.register_buffer("time_table", head.time_pos_enc.pe[0].clone())
        self.register_buffer("action_position",
                             head.action_encoder.pos_encoding.pe[0].clone())

    def forward(self, action, time_index, action_mask, strength, dt, *kv):
        head = self.head
        encoder = head.action_encoder
        seq = action.reshape(1, self.horizon,
                             self.per_action_dim) * action_mask
        seq = seq.to(kv[0].dtype)
        x = F.relu(linear(encoder.W1)(seq)) + self.action_position
        x = linear(encoder.W3)(F.relu(linear(encoder.W2)(x)))
        time = self.time_table[time_index].to(kv[0].dtype)
        for i, block in enumerate(head.transformer_blocks):
            attn = block.attn
            dim, heads = attn.embed_dim, attn.num_heads
            q = F.linear(norm32(block.norm1, x), attn.in_proj_weight[:dim],
                         attn.in_proj_bias[:dim])
            q = q.unflatten(-1, (heads, -1)).transpose(1, 2)
            o = attend(q, kv[i][0], kv[i][1], (dim // heads)**-0.5)
            x = x + attn.out_proj(o.transpose(1, 2).flatten(2))
            x = x + block.ff(norm32(block.norm2, x) + time[:, None])
        x = norm32(head.norm_out, x)
        pooled = head.seq_pool_proj(x.reshape(1, -1))
        mlp = head.mlp_head
        velocity = linear(mlp.fc2)(F.relu(linear(mlp.fc1)(pooled))).float()
        velocity = velocity.reshape(1, self.horizon, -1) * strength
        return action + dt * velocity.reshape(action.shape)


def time_indices(steps, sampler="current"):
    """The time-table row of each Euler step: the current get_action's, or the November 2025 release's (the one the
    published checkpoints were trained and evaluated with)."""
    if sampler == "2025-11":
        return [int(i / steps * 1000) for i in range(steps)]
    return [min(int(i / steps * 999), 999) for i in range(steps)]


def kv_names(blocks):
    return [f"kv_{b}" for b in range(blocks)]


def integrate(step, kv, noise, action_mask, steps):
    action = noise.clone()
    ones = torch.ones(step.horizon, 1, device=action.device)
    dt = torch.tensor([1.0 / steps], device=action.device)
    for index in time_indices(steps):
        action = step(action, torch.tensor([index], device=action.device),
                      action_mask, ones, dt, *kv)
    return action.reshape(-1, action_mask.shape[-1]) * action_mask


def check(policy, graphs, reference, steps):
    vision, prefix, context, step = graphs
    ref = np.load(reference)
    t = lambda k: torch.from_numpy(ref[k]).cuda()
    mask = torch.tensor([1.0] * policy.action_dim + [0.0] *
                        (evo1_reference.MAX_DIM - policy.action_dim)).cuda()
    count = len([k for k in ref.files if k.startswith("normalized_")])
    with torch.no_grad():
        for i in range(count):
            visual = vision(t(f"pixels_{i}"))
            hidden = prefix(t(f"input_ids_{i}"),
                            t(f"visual_{i}").flatten(0, 1),
                            t(f"attention_mask_{i}").float())
            kv = context(t(f"hidden_{i}"), t(f"state_{i}")[None])
            chunk = integrate(step, kv, t(f"noise_{i}"), mask, steps)
            err = lambda a, b: float((a - b).abs().max())
            print(
                f"{i}: visual {err(visual, t(f'visual_{i}')):.2e}  hidden {err(hidden, t(f'hidden_{i}')):.2e} "
                f"(|hidden| {float(t(f'hidden_{i}').abs().max()):.1f})  chunk "
                f"{err(chunk, t(f'normalized_{i}')):.2e}")


def stage_runtime_assets(args, policy, length, steps):
    embedder = policy.model.embedder
    tokenizer = embedder.tokenizer
    prompt = embedder._build_multimodal_prompt(
        [[1] * evo1_reference.MAX_VIEWS], [""])[0]
    prefix_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    head = policy.model.action_head
    stats = policy.stats
    tasks = open(args.tasks).read().splitlines() if args.tasks else []
    for task in tasks:
        joint = tokenizer(embedder._build_multimodal_prompt([[1] * 3],
                                                            [task])[0],
                          add_special_tokens=False)["input_ids"]
        split = prefix_ids + tokenizer(task.strip(),
                                       add_special_tokens=False)["input_ids"]
        if joint != split:
            raise SystemExit(
                f"the prompt does not tokenize as prefix + task for {task!r}")
    tokenizer_dir = args.vlm or policy.config["vlm_name"]
    shutil.copy(os.path.join(tokenizer_dir, "tokenizer.json"),
                os.path.join(args.out, "tokenizer.json"))
    config_json = {
        "model_family": "evo1",
        "cameras": ["image_1", "image_2"],
        "max_views": evo1_reference.MAX_VIEWS,
        "image_size": evo1_reference.IMAGE_SIZE,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        # The official server converts the client's frames with cv2.COLOR_BGR2RGB before the model.
        "swap_red_blue": True,
        "resize": "bilinear",
        "image_tokens": int(embedder.model.num_image_token),
        "text_length": length,
        "prefix_ids": prefix_ids,
        "image_context_id":
        int(tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")),
        "pad_id": int(tokenizer.pad_token_id),
        "hidden_dim": int(head.embed_dim),
        "blocks": len(head.transformer_blocks),
        "chunk_size": int(head.horizon),
        "max_dim": evo1_reference.MAX_DIM,
        "state_dim": policy.state_dim,
        "action_dim": policy.action_dim,
        "steps": steps,
        "attention": args.attention,
        "time_indices": time_indices(steps, args.sampler),
        "state_min": [float(v) for v in stats["observation.state"]["min"]],
        "state_max": [float(v) for v in stats["observation.state"]["max"]],
        "action_min": [float(v) for v in stats["action"]["min"]],
        "action_max": [float(v) for v in stats["action"]["max"]],
    }
    json.dump(config_json,
              open(os.path.join(args.out, "config.json"), "w"),
              indent=1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vlm", help="InternVL3-1B directory")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--sampler",
        choices=["current", "2025-11"],
        default="current",
        help=
        "Euler time rows: the current server's (50 steps by default) or the November 2025 server's (32 steps)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Euler steps (default: 50, or 32 with --sampler 2025-11)")
    parser.add_argument(
        "--attention",
        choices=["flash", "eager"],
        default="flash",
        help=
        "the language model at excluded positions: FlashAttention-2's zero output (the official setup) or eager's"
    )
    parser.add_argument("--arm-key")
    parser.add_argument("--dataset-key")
    parser.add_argument(
        "--check",
        help="evo1_reference.py --precision fp32 output to compare against")
    parser.add_argument(
        "--tasks",
        help=
        "file of instructions, one per line, whose tokenization the runtime's split must match"
    )
    args = parser.parse_args()
    args.precision = "fp32"
    if args.steps is None:
        args.steps = 32 if args.sampler == "2025-11" else 50
    os.makedirs(args.out, exist_ok=True)

    policy = evo1_reference.Policy(args)
    model = policy.model
    embedder = model.embedder
    length = int(embedder.max_text_length)
    graphs = tuple(
        g.eval().cuda()
        for g in (Vision(embedder.model),
                  Prefix(embedder, length, args.attention == "flash"),
                  Context(model.action_head), Step(model.action_head)))
    if args.check:
        check(policy, graphs, args.check, args.steps)

    f16 = torch.float16
    vision, prefix, context, step = (g.half() for g in graphs)
    gpu = dict(device="cuda")
    hidden = model.action_head.embed_dim
    views, tokens = evo1_reference.MAX_VIEWS, int(
        embedder.model.num_image_token)
    blocks, rows = len(model.action_head.transformer_blocks), int(
        model.action_head.horizon) * evo1_reference.MAX_DIM
    heads = model.action_head.transformer_blocks[0].attn.num_heads

    def export(module, inputs, name, input_names, output_names, axes=None):
        path = os.path.join(args.out, f"{name}.onnx")
        torch.onnx.export(module,
                          inputs,
                          path,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=axes,
                          opset_version=17,
                          dynamo=False)
        return apply_trt103_workarounds(path)

    size = evo1_reference.IMAGE_SIZE
    rewrites = {
        "vision":
        export(vision, (torch.randn(views, 3, size, size, dtype=f16, **gpu), ),
               "vision", ["pixels"], ["image_tokens"], {
                   "pixels": {
                       0: "views"
                   },
                   "image_tokens": {
                       0: "views"
                   }
               }),
        "prefix":
        export(prefix, (torch.zeros(1, length, dtype=torch.int32, **gpu),
                        torch.randn(views * tokens, hidden, dtype=f16, **gpu),
                        torch.ones(1, length, dtype=f16, **gpu)), "prefix",
               ["input_ids", "image_tokens", "attention"], ["hidden"]),
        "context":
        export(context,
               (torch.randn(1, length, hidden, dtype=f16, **gpu),
                torch.randn(1, evo1_reference.MAX_DIM, dtype=f16, **gpu)),
               "context", ["hidden", "state"], kv_names(blocks)),
        "step":
        export(
            step, (torch.randn(1, rows, **
                               gpu), torch.zeros(1, dtype=torch.int32, **gpu),
                   torch.ones(evo1_reference.MAX_DIM, **gpu),
                   torch.ones(int(model.action_head.horizon), 1, **
                              gpu), torch.full((1, ), 0.02, **gpu)) +
            tuple(
                torch.randn(
                    2, 1, heads, length + 1, hidden // heads, dtype=f16, **gpu)
                for _ in range(blocks)), "step",
            ["action", "time_index", "action_mask", "strength", "dt"] +
            kv_names(blocks), ["action_next"]),
    }
    stage_runtime_assets(args, policy, length, args.steps)
    print(
        f"exported vision / prefix / context / step -> {args.out}; TensorRT 10.3 rewrites {rewrites}"
    )


if __name__ == "__main__":
    main()
