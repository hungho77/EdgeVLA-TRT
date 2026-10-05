#!/usr/bin/env python3
"""Convert an InternLM2-backed InternVL3 checkpoint to the naming the exporter expects.

InternVL3 ships two text backbones. The 1B/2B/8B/14B sizes use Qwen2.5 and are
republished by HuggingFace as ``-hf`` checkpoints with standard
``self_attn.q_proj`` / ``mlp.gate_proj`` naming, which TensorRT-Edge-LLM's default
CausalLM path loads directly. The 9B size uses InternLM2 instead, has no ``-hf``
counterpart, and stores its decoder as::

    language_model.model.layers.N.attention.wqkv        (Q, K and V fused)
    language_model.model.layers.N.attention.wo
    language_model.model.layers.N.attention_norm
    language_model.model.layers.N.feed_forward.w1/w2/w3
    language_model.model.layers.N.ffn_norm
    language_model.model.tok_embeddings / model.norm / output

None of those names exist in the exporter, so the build fails. This script rewrites
just the decoder into the standard layout and retags ``llm_config`` as ``llama`` --
InternLM2 is Llama with GQA (RMSNorm, SwiGLU, RoPE, no attention bias), so the
mapping is exact rather than approximate.

The vision tower is left untouched: ``vision_model.*`` / ``mlp1.*`` with
``model_type: internvl_chat`` is already a supported path (``export.py`` overrides
the ``intern_vit_6b`` vision model_type to ``internvl``).

The wqkv fusion packs, per KV head, ``num_attention_heads // num_key_value_heads``
query heads followed by one K head and one V head::

    wqkv: [num_kv * (q_per_kv + 2) * head_dim, hidden]
        -> view [num_kv, q_per_kv + 2, head_dim, hidden]
        -> q = [:, :q_per_kv], k = [:, -2], v = [:, -1]

Flattening q in that order yields query head i belonging to KV group
``i // q_per_kv``, which is exactly the grouping standard GQA assumes.

Usage:
    python scripts/convert_internlm2_internvl.py <src_dir> <dst_dir>
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

# Files copied verbatim: tokenizer, preprocessor and template state must survive
# untouched, exactly as the quantization export flow in this repo treats them.
#
# The SentencePiece model and the remote tokenizer class matter as much as the JSON:
# InternVL3-9B declares `InternLM3Tokenizer` through `auto_map` and ships no
# tokenizer.json, so dropping either one makes AutoTokenizer fail on the converted
# directory. The exporter reacts to that by silently falling back to a generic
# "User: / Assistant: " chat template instead of the model's ChatML one, which
# formats every prompt wrongly and makes the engine emit degenerate text.
_COPY_VERBATIM = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "preprocessor_config.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenization_internlm2.py",
    "tokenization_internlm2_fast.py",
    "tokenization_internlm3.py",
    "conversation.py",
)

_LAYER_RENAMES = {
    "attention_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attention.wo": "self_attn.o_proj",
    "feed_forward.w1": "mlp.gate_proj",  # SwiGLU gate
    "feed_forward.w3": "mlp.up_proj",  # SwiGLU up
    "feed_forward.w2": "mlp.down_proj",
}


def split_wqkv(wqkv: torch.Tensor, num_heads: int, num_kv_heads: int,
               head_dim: int):
    """Split a fused InternLM2 ``wqkv`` into ``(q, k, v)`` projections."""
    q_per_kv = num_heads // num_kv_heads
    expected = num_kv_heads * (q_per_kv + 2) * head_dim
    if wqkv.shape[0] != expected:
        raise ValueError(
            f"wqkv has {wqkv.shape[0]} rows, expected {expected} for "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}")
    hidden = wqkv.shape[1]
    v = wqkv.view(num_kv_heads, q_per_kv + 2, head_dim, hidden)
    q = v[:, :q_per_kv].reshape(num_kv_heads * q_per_kv * head_dim, hidden)
    k = v[:, -2].reshape(num_kv_heads * head_dim, hidden)
    val = v[:, -1].reshape(num_kv_heads * head_dim, hidden)
    return q.contiguous(), k.contiguous(), val.contiguous()


def convert_weights(state, num_heads, num_kv_heads, head_dim):
    """Rewrite one checkpoint shard's decoder tensors; pass everything else through."""
    out = {}
    for key, tensor in state.items():
        if not key.startswith("language_model."):
            out[key] = tensor  # vision_model.*, mlp1.* stay as they are
            continue

        if key == "language_model.model.tok_embeddings.weight":
            out["language_model.model.embed_tokens.weight"] = tensor
            continue
        if key == "language_model.output.weight":
            out["language_model.lm_head.weight"] = tensor
            continue
        if key == "language_model.model.norm.weight":
            out[key] = tensor
            continue

        if ".attention.wqkv.weight" in key:
            prefix = key[:key.index(".attention.wqkv.weight")]
            q, k, v = split_wqkv(tensor, num_heads, num_kv_heads, head_dim)
            out[f"{prefix}.self_attn.q_proj.weight"] = q
            out[f"{prefix}.self_attn.k_proj.weight"] = k
            out[f"{prefix}.self_attn.v_proj.weight"] = v
            continue

        for old, new in _LAYER_RENAMES.items():
            token = f".{old}."
            if token in key:
                out[key.replace(token, f".{new}.")] = tensor
                break
        else:
            out[key] = tensor
    return out


def convert_config(config):
    """Retag ``llm_config`` from internlm2 to llama; leave the vision side alone."""
    llm = config.get("llm_config")
    if llm is None:
        raise ValueError("config.json has no 'llm_config' - not an "
                         "internvl_chat checkpoint?")
    if llm.get("model_type") not in ("internlm2", "llama"):
        raise ValueError(
            f"llm_config.model_type is {llm.get('model_type')!r}; this script "
            f"only converts InternLM2-backed checkpoints. Qwen2-backed sizes "
            f"already have an -hf release that the exporter loads directly.")
    llm["model_type"] = "llama"
    llm["architectures"] = ["LlamaForCausalLM"]
    # InternLM2 exposes a single `bias` flag for all projections; Llama has none.
    if llm.pop("bias", False):
        raise ValueError("llm_config.bias is true; attention biases are not "
                         "part of the Llama layout this script targets.")
    llm.pop("auto_map", None)
    # The custom modeling code no longer applies once the layout is standard.
    config.pop("auto_map", None)
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="InternVL3-9B checkpoint directory")
    ap.add_argument("dst", help="output directory")
    ap.add_argument(
        "--free-src",
        action="store_true",
        help="delete each source shard once its converted copy is written. Peak "
        "usage becomes one shard rather than two full checkpoints, which is the "
        "difference between fitting and not fitting for the 9B (18 GB each). "
        "Destroys the source checkpoint - only pass it if you can re-download.")
    args = ap.parse_args()

    with open(os.path.join(args.src, "config.json")) as f:
        config = json.load(f)
    llm = config["llm_config"]
    num_heads = llm["num_attention_heads"]
    num_kv = llm["num_key_value_heads"]
    head_dim = llm.get("head_dim", llm["hidden_size"] // num_heads)
    print(f"[cfg] {llm['model_type']}: heads={num_heads} kv={num_kv} "
          f"head_dim={head_dim} layers={llm['num_hidden_layers']}")

    os.makedirs(args.dst, exist_ok=True)

    index_path = os.path.join(args.src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    new_map, total = {}, 0
    for shard in shards:
        src_shard = os.path.join(args.src, shard)
        state = load_file(src_shard)
        converted = convert_weights(state, num_heads, num_kv, head_dim)
        save_file(converted,
                  os.path.join(args.dst, shard),
                  metadata={"format": "pt"})
        for k in converted:
            new_map[k] = shard
        n_in, n_out = len(state), len(converted)
        total += n_out
        freed = ""
        if args.free_src:
            del state, converted
            os.remove(src_shard)
            freed = "  (source shard removed)"
        print(f"[shard] {shard}: {n_in} -> {n_out} tensors{freed}")

    if os.path.exists(index_path):
        size = sum(
            os.path.getsize(os.path.join(args.dst, s)) for s in shards)
        with open(os.path.join(args.dst, "model.safetensors.index.json"),
                  "w") as f:
            json.dump({
                "metadata": {
                    "total_size": size
                },
                "weight_map": new_map
            }, f, indent=2)

    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(convert_config(config), f, indent=2)

    for name in _COPY_VERBATIM:
        src = os.path.join(args.src, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.dst, name))

    print(f"[done] {total} tensors -> {args.dst}")


if __name__ == "__main__":
    sys.exit(main())
