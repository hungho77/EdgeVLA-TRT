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
"""Golden actions from the official LingBot-VLA 2.0 policy (Robbyant/lingbot-vla-v2, deploy/lingbot_vla_v2_policy.py).

The policy is built as ``LingbotVLAv2Server.load_vla`` / ``reset`` build it, from explicit paths instead of the
directory layout the server assumes: the training config (``--train-config``, the run's ``lingbotvla_cli.yaml``;
a released LIBERO fine-tune ships it as ``training/libero_full.yaml``), the robot config mapping the dataset's
features (``--robot-config``, e.g. ``training/libero.yaml``), the normalization statistics (``--norm-stats``) and
Qwen3-VL-4B-Instruct's processor and tokenizer (``--qwen3vl``). The checkpoint is loaded strictly; attention runs
eager, as the server sets it.

    python lingbot_reference.py --repo lingbot-vla-v2 --checkpoint lingbot-vla-v2-6b-libero \\
        --qwen3vl Qwen3-VL-4B-Instruct --obs obs.npz --out ref.npz [--precision fp32|bf16]

obs.npz holds agentview_<i>, wrist_<i> (raw LIBERO renders; rotated 180 degrees here, as LeRobot's LIBERO dataset
stores them), state_<i> ([eef pos, axis-angle, gripper qpos]) and task_<i>. The output adds, per sample, the
processor's pixel values and grid, the token ids and mask, the padded state, the noise, the normalized chunk and the
robot actions; ``--capture`` also keeps the stage tensors the export checks against (vision tokens, deepstack
features, the prefix's multimodal RoPE ids and mask, K / V of language layers 0, 17 and 35, the first velocity).
``bf16`` is the official server's default precision. ``fp32`` is true FP32: the policy's constructor switches FP32
matmuls to TF32, which is turned back off, and the action expert's MoE is evaluated densely in FP32 instead of the
released Triton kernels.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

TEXT_KEYS = ("hidden_size", "intermediate_size", "num_hidden_layers",
             "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
             "rope_theta", "vocab_size", "max_position_embeddings",
             "hidden_act", "tie_word_embeddings", "tokenizer_path")


def use_sdpa_without_flash_attn():
    """The model requests FlashAttention-2 for every sub-model. The VLM and the expert attend jointly through the
    policy's own eager attention with an explicit mask, so the request only matters to the ViT, whose non-flash path
    attends within each image as the varlen kernel does: without flash-attn, fall back to SDPA."""
    import importlib.util
    if importlib.util.find_spec("flash_attn") is not None:
        return
    from transformers.modeling_utils import PreTrainedModel
    original = PreTrainedModel._check_and_adjust_attn_implementation

    def check(self, attn_implementation, *args, **kwargs):
        if attn_implementation == "flash_attention_2":
            attn_implementation = "sdpa"
        return original(self, attn_implementation, *args, **kwargs)

    PreTrainedModel._check_and_adjust_attn_implementation = check


def use_dense_fp32_moe():
    """The released fused MoE runs Triton group GEMMs (bf16 operands, or TF32 dots on FP32 ones). The FP32 golden
    computes every expert in FP32 instead and mixes them with the official top-k routing weights, the formula of the
    repo's eager MoE path."""
    import torch.nn.functional as F
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import \
        Qwen2TokenMoeBlock

    def forward(self, hidden_states):
        batch, length, dim = hidden_states.shape
        x = hidden_states.reshape(-1, dim).float()
        router_logits = F.linear(x, self.gate.weight.float())
        if self._router_activation == "sigmoid":
            scores = router_logits.sigmoid()
        else:
            scores = F.softmax(router_logits, dim=1)
        _, selected = torch.topk(scores + self.e_score_correction_bias[None],
                                 self.top_k,
                                 dim=-1)
        weights = scores.gather(1, selected)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        dense = torch.zeros_like(scores).scatter(1, selected, weights)
        experts = self.experts
        gate = torch.einsum("td,eid->eti", x, experts.gate_proj.float())
        up = torch.einsum("td,eid->eti", x, experts.up_proj.float())
        out = torch.einsum("eti,edi->etd",
                           F.silu(gate) * up, experts.down_proj.float())
        routed = torch.einsum("etd,te->td", out, dense)
        shared = self.shared_expert.float()(x)
        if self._use_shared_expert_gate:
            shared = torch.sigmoid(self.shared_expert_gate.float()(x)) * shared
        result = (routed + shared).to(hidden_states.dtype)
        return result.reshape(batch, length, dim), router_logits

    Qwen2TokenMoeBlock.forward = forward


def load_policy(args):
    sys.path.insert(0, args.repo)
    from glob import glob

    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.models import build_processor
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import \
        LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import \
        LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import \
        apply_lingbot_qwen3_vl_patch
    from safetensors import safe_open
    from transformers import AutoConfig

    apply_lingbot_qwen3_vl_patch()
    use_sdpa_without_flash_attn()
    if args.precision == "fp32":
        use_dense_fp32_moe()
    training = yaml.safe_load(open(args.train_config))
    model_config = dict(training["model"])
    model_config.update(training["train"])
    config = LingbotVLAV2Config(**model_config)
    for key, value in model_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    config.attention_implementation = "eager"
    config.tokenizer_path = args.qwen3vl
    qwen = AutoConfig.from_pretrained(args.qwen3vl).to_dict()
    for key in TEXT_KEYS:
        if key in qwen.get("text_config", {}):
            setattr(config, key, qwen["text_config"][key])
        elif key in qwen:
            setattr(config, key, qwen[key])
    config.vision_config = AutoConfig.from_pretrained(
        args.qwen3vl).vision_config
    if training["model"].get("vocab_size"):
        config.vocab_size = training["model"]["vocab_size"]
    config.use_cache = True
    processor = build_processor(args.qwen3vl)
    # Built on the GPU: on Jetson the GPU shares DRAM, and a CPU copy moved over would double the 25 GB.
    with torch.device("cuda"):
        policy = LingbotVlaV2Policy(config, eval=True)
    # Shard by shard (the FP32 checkpoint is 25 GB), strict over the union of the shards.
    expected = set(policy.state_dict().keys())
    loaded = set()
    for path in sorted(glob(os.path.join(args.checkpoint, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            shard = {key: f.get_tensor(key) for key in f.keys()}
        unexpected = policy.load_state_dict(shard,
                                            strict=False).unexpected_keys
        if unexpected:
            raise SystemExit(f"unexpected checkpoint keys: {unexpected[:5]}")
        loaded.update(shard)
        del shard
    if expected - loaded:
        raise SystemExit(
            f"missing checkpoint keys: {sorted(expected - loaded)[:5]}")
    if args.precision == "fp32":
        # The policy's constructor sets torch.set_float32_matmul_precision("high"): FP32 matmuls would run as TF32.
        torch.set_float32_matmul_precision("highest")
    if args.precision == "bf16":
        policy = policy.to(torch.bfloat16)
    else:
        policy.model.float()
    policy = policy.cuda().eval()
    # The trainer's CLI hands FeatureTransform these lists as "{'name': value}" strings; the saved YAML holds dicts.
    data = {
        key: [str(v) if isinstance(v, dict) else v
              for v in value] if key in ("joints", "norm_type") else value
        for key, value in training["data"].items()
    }
    data_config = SimpleNamespace(**data)
    transform = FeatureTransform(args.robot_config,
                                 data_config,
                                 config,
                                 processor,
                                 chunk_size=config.chunk_size,
                                 norm_stats_path=args.norm_stats)
    return policy, transform, config, data_config


def install_capture(policy):
    model = policy.model
    qe = model.qwenvl_with_expert
    captured = {}
    to_np = lambda x: x.detach().float().cpu().numpy()
    embed_image = qe.embed_image

    def image(*a, **k):
        tokens, deepstack = embed_image(*a, **k)
        captured.update(vision=to_np(tokens),
                        deepstack=np.stack([to_np(d) for d in deepstack]))
        return tokens, deepstack

    embed_prefix = model.embed_prefix

    def prefix(*a, **k):
        result = embed_prefix(*a, **k)
        captured.update(prefix_pad=result[1].cpu().numpy(),
                        position_ids=result[3].cpu().numpy())
        return result

    forward = qe.forward

    def joint(*a, **k):
        result = forward(*a, **k)
        if k.get("fill_kv_cache"):
            for layer in (0, 17, 35):
                captured[f"k{layer}"] = to_np(result[1][layer]["key_states"])
                captured[f"v{layer}"] = to_np(result[1][layer]["value_states"])
        return result

    predict_velocity = model.predict_velocity

    def velocity(*a, **k):
        v = predict_velocity(*a, **k)
        captured.setdefault("velocity0", to_np(v))
        return v

    qe.embed_image, model.embed_prefix, qe.forward, model.predict_velocity = image, prefix, joint, velocity
    return captured


def policy_rotate(image):
    return np.ascontiguousarray(image[::-1, ::-1])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo",
                        required=True,
                        help="Robbyant/lingbot-vla-v2 clone")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--train-config",
        help=
        "the run's training config (default: CHECKPOINT/training/libero_full.yaml)"
    )
    parser.add_argument(
        "--robot-config",
        help=
        "the robot feature mapping (default: CHECKPOINT/training/libero.yaml)")
    parser.add_argument("--norm-stats",
                        help="default: CHECKPOINT/training/norm_stats.json")
    parser.add_argument("--qwen3vl",
                        required=True,
                        help="Qwen3-VL-4B-Instruct processor / tokenizer")
    parser.add_argument("--obs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision",
                        choices=["fp32", "bf16"],
                        default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture",
                        action="store_true",
                        help="also save the stage tensors (see above)")
    args = parser.parse_args()
    training = os.path.join(args.checkpoint, "training")
    args.train_config = args.train_config or os.path.join(
        training, "libero_full.yaml")
    args.robot_config = args.robot_config or os.path.join(
        training, "libero.yaml")
    args.norm_stats = args.norm_stats or os.path.join(training,
                                                      "norm_stats.json")

    policy, transform, config, data_config = load_policy(args)
    from torchvision.transforms.v2 import Resize
    size = getattr(data_config, "img_size", 256)
    resize = Resize((size, size))
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    cameras = sorted(transform.org_features["images"])
    state_key = sorted(transform.org_features["states"])[0]
    obs = np.load(args.obs)
    count = len([k for k in obs.files if k.startswith("state_")])
    generator = torch.Generator().manual_seed(args.seed)
    captured = install_capture(policy) if args.capture else {}
    out = {}
    for i in range(count):
        captured.clear()
        frames = {
            cameras[0]: policy_rotate(obs[f"agentview_{i}"]),
            cameras[1]: policy_rotate(obs[f"wrist_{i}"])
        }
        item = {
            name:
            resize(
                torch.as_tensor(frame).permute(2, 0, 1).contiguous().float())
            for name, frame in frames.items()
        }
        item[state_key] = torch.as_tensor(obs[f"state_{i}"],
                                          dtype=torch.float32)
        item["task"] = str(obs[f"task_{i}"])
        applied = transform.apply(item, policy_eval=True)
        noise = torch.randn(1,
                            config.n_action_steps,
                            config.max_action_dim,
                            generator=generator)
        with torch.no_grad():
            chunk = policy.model.sample_actions(
                applied["images"].unsqueeze(0).to("cuda", dtype),
                applied["img_masks"].unsqueeze(0).cuda(),
                applied["lang_tokens"].unsqueeze(0).cuda(),
                applied["lang_masks"].unsqueeze(0).cuda(),
                applied["state"].unsqueeze(0).to("cuda", dtype),
                noise=noise.to("cuda", dtype),
                image_grid_thw=applied["image_grid_thw"].cuda())
        normalized = chunk[0].float().cpu()
        result = dict(applied)
        result["actions"] = normalized
        result["state"] = applied["state"].float()
        robot = transform.unapply(result)
        action_keys = sorted(transform.org_features["actions"])
        robot = np.concatenate([
            np.asarray(robot[k], np.float32).reshape(config.n_action_steps, -1)
            for k in action_keys
        ], -1)
        out.update({
            f"pixels_{i}": applied["images"].float().numpy(),
            f"image_grid_thw_{i}": applied["image_grid_thw"].numpy(),
            f"img_masks_{i}": applied["img_masks"].numpy(),
            f"lang_tokens_{i}": applied["lang_tokens"].numpy(),
            f"lang_masks_{i}": applied["lang_masks"].numpy(),
            f"state_{i}": applied["state"].float().numpy(),
            f"noise_{i}": noise[0].numpy(),
            f"normalized_{i}": normalized.numpy(),
            f"actions_{i}": robot,
        })
        out.update({f"{k}_{i}": v for k, v in captured.items()})
        print(f"{i}: {item['task']!r}\n  robot row 0 {np.round(robot[0], 4)}")
    np.savez(args.out, **out)
    print(f"{count} samples ({args.precision}) -> {args.out}")


if __name__ == "__main__":
    main()
