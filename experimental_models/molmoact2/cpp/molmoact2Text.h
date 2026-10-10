/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace molmoact2
{

//! MolmoAct2's _normalize_question_text: whitespace runs collapsed; surrounding quotes / brackets, a leading
//! "task:" / "the task is to" style prefix and trailing sentence punctuation / closers stripped until nothing
//! changes; several sentences joined with "; "; lowercased.
std::string normalizeQuestion(std::string const& text);

//! "<state_start><state_k>...<state_end>": each normalized value clipped to [-1, 1] and binned into \p bins levels
//! with round-half-to-even, as np.rint.
std::string stateTokens(std::vector<float> const& normalizedState, int32_t bins);

//! The robot_action prompt of MolmoAct2's processor: "Image k<|image|>" per image (one "<|image|>" alone for a single
//! image), the user turn with the task, the setup, the state tokens and the control mode, then the assistant turn
//! up to "<action_output>". \p imageTokens replaces each "<|image|>".
std::string robotPrompt(std::string const& task, std::string const& state, std::string const& setup,
    std::string const& controlMode, int32_t images, std::string const& imageTokens);

} // namespace molmoact2
} // namespace trt_edgellm
