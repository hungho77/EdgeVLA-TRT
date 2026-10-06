# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Write one embodiment's GR00T N1.7 state/action processing as processing.json.

The official processor resolves which statistics apply (percentiles,
relative-action overrides, per-step bounds); this records the result so the C++
policy only does arithmetic:

  state:  2 * (x - min) / (max - min) - 1 per group (0 where max == min),
          clipped to [-1, 1] when clip_outliers; groups concatenated in order
          and zero-padded to max_state_dim.
  action: model output rows [0, action_horizon), columns split per group in
          order; clip to [-1, 1], then (a + 1) / 2 * (max - min) + min with
          (T, D) or (D,) bounds; relative groups add the raw reference state.

    python export_gr00t_n1_7_processing.py --gr00t-src <dir> --checkpoint GR00T-N1.7-SO101-Multitask \\
        --embodiment new_embodiment --out engines/action/processing.json
"""

import argparse
import json
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gr00t-src", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embodiment", default="new_embodiment")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.gr00t_src)
    import gr00t.model.gr00t_n1d7.gr00t_n1d7  # noqa: F401  registers the config
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7  # noqa: F401  registers the processor
    from gr00t.data.types import ActionRepresentation, ActionType
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    sap = processor.state_action_processor
    tag = args.embodiment
    configs = sap.modality_configs[tag]
    params = sap.norm_params[tag]

    for modality in ("state", "action"):
        cfg = configs[modality]
        if cfg.mean_std_embedding_keys or cfg.sin_cos_embedding_keys:
            raise SystemExit(
                f"{modality}: mean/std or sin/cos encoded groups are not supported"
            )
    if getattr(processor, "use_mean_std", False) or getattr(
            sap, "use_mean_std", False):
        raise SystemExit("mean/std normalization is not supported")

    state = []
    for key in configs["state"].modality_keys:
        p = params["state"][key]
        state.append({
            "name": key,
            "dim": int(p["dim"]),
            "min": np.asarray(p["min"], dtype=np.float64).tolist(),
            "max": np.asarray(p["max"], dtype=np.float64).tolist(),
        })

    action = []
    action_configs = configs["action"].action_configs or [None] * len(
        configs["action"].modality_keys)
    for key, action_config in zip(configs["action"].modality_keys,
                                  action_configs):
        p = params["action"][key]
        relative = bool(action_config is not None
                        and action_config.rep == ActionRepresentation.RELATIVE
                        and sap.use_relative_action)
        if relative and action_config.type != ActionType.NON_EEF:
            raise SystemExit(
                f"action '{key}': only joint-space relative actions are supported"
            )
        action.append({
            "name":
            key,
            "dim":
            int(p["dim"]),
            "min":
            np.asarray(p["min"], dtype=np.float64).tolist(),
            "max":
            np.asarray(p["max"], dtype=np.float64).tolist(),
            "relative":
            relative,
            "reference_state":
            (action_config.state_key or key) if relative else None,
        })

    spec = {
        "embodiment": tag,
        "max_state_dim": int(processor.max_state_dim),
        "max_action_dim": int(processor.max_action_dim),
        "action_horizon": len(configs["action"].delta_indices),
        "clip_state": bool(sap.clip_outliers),
        "state": state,
        "action": action,
        "video_keys": list(configs["video"].modality_keys),
        "formalize_language": bool(processor.formalize_language),
    }
    with open(args.out, "w") as f:
        json.dump(spec, f, indent=1)
    print(
        f"{tag}: state {[s['name'] for s in state]}, action "
        f"{[(a['name'], 'relative' if a['relative'] else 'absolute') for a in action]}, "
        f"horizon {spec['action_horizon']} -> {args.out}")


if __name__ == "__main__":
    main()
