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
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rldx
{

//! Qwen3-VL text-model geometry the prompt tables depend on.
struct RldxTextGeometry
{
    int32_t headDim{128};
    float ropeTheta{5000000.0F};
    int32_t mropeSection[3]{24, 20, 20};
};

//! The host-side inputs of RLDX's llm_a / llm_b engines for one prompt (see rldx_llm.py): the token ids with each
//! image's pads, the 64 cognition positions appended after them, interleaved MRoPE tables in FP32, and the
//! video-token compression of the LayerWrapper.
struct RldxPrompt
{
    std::vector<int64_t> inputIds;    //!< [S0] the templated prompt (no cognition tokens)
    std::vector<int64_t> visualIndex; //!< [S0] row of the visual features at an image pad, -1 elsewhere
    std::vector<float> cos;           //!< [S, headDim], S = S0 + cognition tokens
    std::vector<float> sin;
    std::vector<float> pool;          //!< [S] 1 / count on the compressed (past-frame) tokens
    std::vector<int64_t> keepIndex;   //!< [S'] rows of the compressed sequence; S stands for the pooled token
    std::vector<float> cosCompressed; //!< [S', headDim]
    std::vector<float> sinCompressed;
    int64_t sequence() const noexcept
    {
        return static_cast<int64_t>(pool.size());
    }
    int64_t compressed() const noexcept
    {
        return static_cast<int64_t>(keepIndex.size());
    }
};

//! "<|im_start|>user\n" + text + images x ("<|vision_start|>" + pads + "<|vision_end|>") + "<|im_end|>\n", as
//! RLDX's processor templates it (text first), then \p cognitionTokens placeholders. Images are frame-major,
//! \p views per frame; \p grids holds each image's merged-token grid (height, width). The compression keeps the
//! last frame.
RldxPrompt buildPrompt(std::vector<int32_t> const& textTokens, std::vector<std::pair<int32_t, int32_t>> const& grids,
    int32_t views, int32_t cognitionTokens, RldxTextGeometry const& geometry);

} // namespace rldx
} // namespace trt_edgellm
