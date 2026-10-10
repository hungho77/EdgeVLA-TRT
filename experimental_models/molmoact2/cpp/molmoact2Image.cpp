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

#include "molmoact2Image.h"

#include <algorithm>
#include <cmath>

namespace trt_edgellm
{
namespace molmoact2
{
namespace
{

struct Tap
{
    int32_t i0, i1;
    float w0, w1;
};

//! upsample_bilinear2d's source index and weights (align_corners=False), in FP32 as PyTorch computes them.
std::vector<Tap> taps(int32_t in, int32_t out)
{
    float const scale = static_cast<float>(in) / static_cast<float>(out);
    std::vector<Tap> t(out);
    for (int32_t d = 0; d < out; ++d)
    {
        float const src = std::max((static_cast<float>(d) + 0.5F) * scale - 0.5F, 0.0F);
        int32_t const i0 = std::min(static_cast<int32_t>(std::floor(src)), in - 1);
        float const w1 = src - static_cast<float>(i0);
        t[d] = Tap{i0, std::min(i0 + 1, in - 1), 1.0F - w1, w1};
    }
    return t;
}

} // namespace

void siglipPatches(unsigned char const* rgb, int32_t height, int32_t width, int32_t size, int32_t patch, float* out)
{
    std::vector<Tap> const rows = taps(height, size);
    std::vector<Tap> const cols = taps(width, size);
    int32_t const perSide = size / patch;
    for (int32_t y = 0; y < size; ++y)
    {
        Tap const& r = rows[y];
        unsigned char const* row0 = rgb + static_cast<size_t>(r.i0) * width * 3;
        unsigned char const* row1 = rgb + static_cast<size_t>(r.i1) * width * 3;
        for (int32_t x = 0; x < size; ++x)
        {
            Tap const& c = cols[x];
            float* dst = out + (static_cast<size_t>(y / patch) * perSide + x / patch) * patch * patch * 3
                + (static_cast<size_t>(y % patch) * patch + x % patch) * 3;
            for (int32_t ch = 0; ch < 3; ++ch)
            {
                float const a = row0[c.i0 * 3 + ch];
                float const b = row0[c.i1 * 3 + ch];
                float const cc = row1[c.i0 * 3 + ch];
                float const d = row1[c.i1 * 3 + ch];
                float const top = std::fma(c.w0, a, c.w1 * b);
                float const bottom = std::fma(c.w0, cc, c.w1 * d);
                float const value = std::clamp(std::nearbyint(std::fma(r.w0, top, r.w1 * bottom)), 0.0F, 255.0F);
                dst[ch] = value / 255.0F * 2.0F - 1.0F;
            }
        }
    }
}

} // namespace molmoact2
} // namespace trt_edgellm
