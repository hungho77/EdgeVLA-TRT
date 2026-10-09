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
"""Golden actions from LeRobot's MolmoAct2 policy (allenai/MolmoAct2-LIBERO-LeRobot), stage by stage.

Runs the checkpoint's own preprocessor, policy (continuous inference) and postprocessor on LIBERO observations with a
fixed initial noise. LeRobot resolves the policy's HF base checkpoint and the FAST action tokenizer by name; --hf and
--fast point them at local copies holding only their config, processor and tokenizer files (the LeRobot weights
replace the base model's, so it is built from its config).

    python molmoact2_reference.py --checkpoint MolmoAct2-LIBERO-LeRobot --hf MolmoAct2-LIBERO-hf \\
        --fast MolmoAct2-FAST-Tokenizer --obs obs.npz --out ref.npz [--precision fp32|bf16] [--preprocess-only]

obs.npz holds agentview_<i> / wrist_<i> (raw LIBERO renders, [H, W, 3] or a stack whose last entry is used; flipped
180 degrees here as LeRobot's LIBERO processor does), state_<i> ([eef pos, axis-angle, gripper qpos]) and task_<i>.
Per sample the output holds the packed model inputs, the visual features added to the image-patch tokens, every
layer's K / V, the action expert's cross-attention mask, the noise, each step's velocity, the normalized chunk and
the environment actions. ``bf16`` is the official setting (BF16 weights under autocast).
"""

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import torch


def local_checkpoint(args):
    """A copy of the LeRobot checkpoint's small files whose base-model and tokenizer references are local."""
    out = tempfile.mkdtemp(prefix="molmoact2_ckpt_")
    for name in os.listdir(args.checkpoint):
        src = os.path.join(args.checkpoint, name)
        if name.endswith(".safetensors") and os.path.getsize(src) > 1 << 20:
            os.symlink(os.path.abspath(src), os.path.join(out, name))
        elif os.path.isfile(src):
            shutil.copy(src, out)
    config = json.load(open(os.path.join(out, "config.json")))
    config.update(checkpoint_path=os.path.abspath(args.hf),
                  discrete_action_tokenizer=os.path.abspath(args.fast),
                  inference_action_mode="continuous",
                  device=args.device,
                  model_dtype="bfloat16"
                  if args.precision == "bf16" else "float32",
                  enable_inference_cuda_graph=False)
    json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=1)
    pre = json.load(open(os.path.join(out, "policy_preprocessor.json")))
    for step in pre["steps"]:
        if step["registry_name"] == "molmoact2_pack_inputs":
            step["config"].update(
                checkpoint_path=os.path.abspath(args.hf),
                discrete_action_tokenizer=os.path.abspath(args.fast))
        if step["registry_name"] == "device_processor":
            step["config"]["device"] = args.device
    json.dump(pre,
              open(os.path.join(out, "policy_preprocessor.json"), "w"),
              indent=1)
    post = json.load(open(os.path.join(out, "policy_postprocessor.json")))
    for step in post["steps"]:
        if step["registry_name"] == "device_processor":
            step["config"]["device"] = "cpu"
    json.dump(post,
              open(os.path.join(out, "policy_postprocessor.json"), "w"),
              indent=1)
    return out


def from_config_only():
    """Build the HF base model from its config: the LeRobot checkpoint's weights are loaded over it."""
    from lerobot.policies.molmoact2 import modeling_molmoact2 as lm

    cls = lm.MolmoAct2ForConditionalGeneration

    def from_pretrained(location, config=None, dtype=None, **kwargs):
        return cls._from_config(config, dtype=dtype)

    cls.from_pretrained = staticmethod(from_pretrained)
    lm._strict_load_safetensors_weights = lambda model, location: None


def observation(obs, i):

    def frame(key):
        image = obs[f"{key}_{i}"]
        image = image[-1] if image.ndim == 4 else image
        flipped = np.ascontiguousarray(image[::-1, ::-1])
        return torch.from_numpy(flipped).permute(2, 0, 1).float()[None] / 255

    return {
        "observation.images.image":
        frame("agentview"),
        "observation.images.wrist_image":
        frame("wrist"),
        "observation.state":
        torch.from_numpy(obs[f"state_{i}"].astype(np.float32))[None],
        "task": [str(obs[f"task_{i}"])],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hf",
                        required=True,
                        help="allenai/MolmoAct2-LIBERO's small files")
    parser.add_argument("--fast",
                        required=True,
                        help="allenai/MolmoAct2-FAST-Tokenizer")
    parser.add_argument("--obs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision",
                        default="fp32",
                        choices=("fp32", "bf16"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    import lerobot.policies.molmoact2.processor_molmoact2  # noqa: F401 -- registers its processor steps
    from lerobot.processor import PolicyProcessorPipeline
    checkpoint = local_checkpoint(args)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_preprocessor.json")
    obs = np.load(args.obs)
    count = len([k for k in obs.files if k.startswith("state_")])
    out = {}
    batches = []
    for i in range(count):
        batch = preprocessor(observation(obs, i))
        batches.append(batch)
        for key in ("input_ids", "token_type_ids", "pixel_values",
                    "image_token_pooling", "image_grids", "image_num_crops",
                    "action_dim_is_pad", "observation.state"):
            value = batch.get(key)
            if torch.is_tensor(value):
                out[f"{key.replace('.', '_')}_{i}"] = value.detach().cpu(
                ).float().numpy() if value.is_floating_point(
                ) else value.cpu().numpy()
        print(f"{i}: {str(obs[f'task_{i}'])!r}  {batch['input_ids'].shape[1]} tokens")
    if args.preprocess_only:
        np.savez(args.out, **out)
        print(f"{count} samples (preprocessing) -> {args.out}")
        return

    from_config_only()
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_postprocessor.json")
    policy = MolmoAct2Policy.from_pretrained(checkpoint).to(args.device)
    policy.eval()
    backbone = policy._backbone()
    captured = {}
    original_extract = backbone._extract_kv_states
    backbone._extract_kv_states = lambda cache: captured.setdefault(
        "kv", original_extract(cache))
    original_mask = backbone._get_encoder_attention_mask

    def mask(*a, **k):
        captured["mask"] = original_mask(*a, **k)
        return captured["mask"]

    backbone._get_encoder_attention_mask = mask
    backbone.model.vision_backbone.register_forward_hook(
        lambda m, i, o: captured.setdefault("visual", o))
    expert = backbone._require_action_expert()
    original_forward = expert.forward_with_context

    def forward_with_context(trajectory, *a, **k):
        velocity = original_forward(trajectory, *a, **k)
        captured.setdefault("steps", []).append((trajectory, velocity))
        return velocity

    expert.forward_with_context = forward_with_context
    randn = torch.randn
    generator = torch.Generator().manual_seed(args.seed)

    def fixed_noise(*size, **kwargs):
        shape = tuple(kwargs.get("size", size[0] if len(size) == 1 and
                                 isinstance(size[0], tuple) else size))
        if len(shape) == 3 and shape[1:] == (10, 32):
            return captured["noise"].to(kwargs.get("device"),
                                        kwargs.get("dtype"))
        return randn(*size, **kwargs)

    for i, batch in enumerate(batches):
        captured.clear()
        captured["noise"] = randn((1, 10, 32), generator=generator)
        torch.randn = fixed_noise
        try:
            with torch.no_grad():
                normalized = policy.predict_action_chunk(batch)
        finally:
            torch.randn = randn
        actions = postprocessor({"action": normalized})["action"]

        def np32(t):
            return t.detach().float().cpu().numpy()

        visual = captured["visual"]
        out[f"visual_{i}"] = np32(visual[0] if isinstance(
            visual, (tuple, list)) else visual)
        out[f"k_{i}"] = np.stack([np32(k) for k, _ in captured["kv"]])
        out[f"v_{i}"] = np.stack([np32(v) for _, v in captured["kv"]])
        out[f"encoder_mask_{i}"] = captured["mask"].cpu().numpy()
        out[f"noise_{i}"] = captured["noise"].numpy()
        out[f"step_input_{i}"] = np.stack(
            [np32(x) for x, _ in captured["steps"]])
        out[f"step_velocity_{i}"] = np.stack(
            [np32(v) for _, v in captured["steps"]])
        out[f"normalized_{i}"] = np32(normalized[0])
        out[f"actions_{i}"] = np32(actions[0])
        print(f"   action row 0 {np.round(out[f'actions_{i}'][0], 4)}")
    np.savez(args.out, **out)
    print(f"{count} samples ({args.precision}) -> {args.out}")


if __name__ == "__main__":
    main()
