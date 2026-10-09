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

#include "rldxPrompt.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace trt_edgellm
{
namespace rldx
{
namespace
{

constexpr int64_t kImStart = 151644;
constexpr int64_t kUser = 872;
constexpr int64_t kNewline = 198;
constexpr int64_t kImEnd = 151645;
constexpr int64_t kVisionStart = 151652;
constexpr int64_t kVisionEnd = 151653;
constexpr int64_t kImagePad = 151655;

//! Qwen3-VL's interleaved MRoPE: frequency j reads the temporal position, except j % 3 == 1 (height) and j % 3 == 2
//! (width) inside 3 x their sections; the table is [cos(freqs) | cos(freqs)] like HF's rotary embedding.
void ropeRows(std::vector<std::array<int64_t, 3>> const& positions, RldxTextGeometry const& g, std::vector<float>& cos,
    std::vector<float>& sin)
{
    int32_t const half = g.headDim / 2;
    std::vector<float> invFreq(half);
    for (int32_t j = 0; j < half; ++j)
    {
        invFreq[j] = 1.0F / std::pow(g.ropeTheta, static_cast<float>(2 * j) / static_cast<float>(g.headDim));
    }
    cos.assign(positions.size() * g.headDim, 0.0F);
    sin.assign(positions.size() * g.headDim, 0.0F);
    for (size_t p = 0; p < positions.size(); ++p)
    {
        for (int32_t j = 0; j < half; ++j)
        {
            int32_t axis = 0;
            if (j % 3 == 1 && j < 3 * g.mropeSection[1])
            {
                axis = 1;
            }
            else if (j % 3 == 2 && j < 3 * g.mropeSection[2])
            {
                axis = 2;
            }
            float const angle = static_cast<float>(positions[p][axis]) * invFreq[j];
            size_t const row = p * g.headDim;
            cos[row + j] = cos[row + j + half] = std::cos(angle);
            sin[row + j] = sin[row + j + half] = std::sin(angle);
        }
    }
}

} // namespace

RldxPrompt buildPrompt(std::vector<int32_t> const& textTokens, std::vector<std::pair<int32_t, int32_t>> const& grids,
    int32_t views, int32_t cognitionTokens, RldxTextGeometry const& geometry)
{
    auto const images = static_cast<int32_t>(grids.size());
    ELLM_CHECK(images > views && images % views == 0, "rldx: the prompt needs whole frames, more than one");
    RldxPrompt prompt;
    std::vector<std::array<int64_t, 3>> positions;
    int64_t next = 0;
    auto text = [&](int64_t id) {
        prompt.inputIds.push_back(id);
        prompt.visualIndex.push_back(-1);
        positions.push_back({next, next, next});
        ++next;
    };
    text(kImStart);
    text(kUser);
    text(kNewline);
    for (int32_t token : textTokens)
    {
        text(token);
    }
    std::vector<int64_t> starts;
    int64_t visualRow = 0;
    for (auto const& [gridH, gridW] : grids)
    {
        starts.push_back(static_cast<int64_t>(prompt.inputIds.size()));
        text(kVisionStart);
        int64_t const base = next;
        for (int32_t h = 0; h < gridH; ++h)
        {
            for (int32_t w = 0; w < gridW; ++w)
            {
                prompt.inputIds.push_back(kImagePad);
                prompt.visualIndex.push_back(visualRow++);
                positions.push_back({base, base + h, base + w});
            }
        }
        next = base + std::max(gridH, gridW);
        text(kVisionEnd);
    }
    text(kImEnd);
    text(kNewline);
    for (int32_t c = 0; c < cognitionTokens; ++c)
    {
        positions.push_back({next, next, next});
        ++next;
    }
    ropeRows(positions, geometry, prompt.cos, prompt.sin);

    // LayerWrapper: from the first <|vision_start|> up to the views-th from last one, pooled into one token.
    int64_t const total = static_cast<int64_t>(positions.size());
    int64_t const begin = starts.front();
    int64_t const end = starts[starts.size() - views];
    prompt.pool.assign(total, 0.0F);
    std::fill(prompt.pool.begin() + begin, prompt.pool.begin() + end, 1.0F / static_cast<float>(end - begin));
    std::vector<std::array<int64_t, 3>> compressed;
    for (int64_t i = 0; i < total; ++i)
    {
        if (i == begin)
        {
            prompt.keepIndex.push_back(total);
            compressed.push_back(positions[begin]);
        }
        if (i < begin || i >= end)
        {
            prompt.keepIndex.push_back(i);
            compressed.push_back(positions[i]);
        }
    }
    ropeRows(compressed, geometry, prompt.cosCompressed, prompt.sinCompressed);
    return prompt;
}

} // namespace rldx
} // namespace trt_edgellm
