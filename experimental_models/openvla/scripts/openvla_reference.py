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
"""Official OpenVLA reference for one image and instruction.

Runs the checkpoint's own ``OpenVLAForActionPrediction.predict_action`` (trust_remote_code, transformers 4.40.1)
on the CPU, in FP32 or with --bf16 as it is usually served, and saves what each stage of a port has to match:
the 6-channel pixel values, the fused DINOv2 + SigLIP patch features, the projected patch embeddings, the prompt
token ids, the greedy action tokens with their logits, and the unnormalized actions.

    python openvla_reference.py --checkpoint openvla-7b --image frame.png \\
        --instruction "pick up the banana" --unnorm-key bridge_orig --out ref.npz
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    processor = AutoProcessor.from_pretrained(args.checkpoint,
                                              trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True).eval()

    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"
    image = Image.open(args.image).convert("RGB")
    inputs = processor(prompt, image)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    # predict_action appends the empty token to input_ids but not to the mask; the mask is all ones, so generate's
    # default is the same mask at the right length.
    inputs.pop("attention_mask", None)

    captured = {}

    def keep(name):

        def hook(module, inputs, output):
            # A forward hook that returns a value replaces the module's output.
            if name not in captured:
                captured[name] = output[0].detach().float().numpy()

        return hook

    vla.vision_backbone.register_forward_hook(keep("patches"))
    vla.projector.register_forward_hook(keep("projected"))
    real_generate = vla.generate

    def generate(input_ids, **kwargs):
        captured["input_ids"] = input_ids[0].numpy()
        out = real_generate(input_ids,
                            output_scores=True,
                            return_dict_in_generate=True,
                            **kwargs)
        captured["action_logits"] = torch.stack(out.scores)[:,
                                                            0].float().numpy()
        captured["generated_ids"] = out.sequences[0,
                                                  input_ids.shape[1]:].numpy()
        return out.sequences

    vla.generate = generate
    with torch.no_grad():
        actions = vla.predict_action(**inputs,
                                     unnorm_key=args.unnorm_key,
                                     do_sample=False)
    logits = captured["action_logits"]
    top2 = np.sort(logits, axis=-1)[:, -2:]
    np.savez(args.out,
             pixel_values=inputs["pixel_values"][0].float().numpy(),
             actions=np.asarray(actions),
             top2_margin=top2[:, 1] - top2[:, 0],
             **captured)
    print(
        f"{args.checkpoint}: {len(captured['input_ids'])} prompt ids, patches {captured['patches'].shape}, "
        f"action tokens {captured['generated_ids'].tolist()}, smallest top-2 logit margin "
        f"{(top2[:, 1] - top2[:, 0]).min():.3f}, actions {np.round(actions, 4).tolist()} -> {args.out}"
    )


if __name__ == "__main__":
    main()
