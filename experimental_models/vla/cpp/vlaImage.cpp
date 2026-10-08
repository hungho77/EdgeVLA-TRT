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

#include "vlaImage.h"

#include <algorithm>
#include <cmath>

namespace trt_edgellm
{
namespace vla
{

namespace
{

constexpr int32_t kPrecisionBits = 32 - 8 - 2;

double bicubicFilter(double x)
{
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
    {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0)
    {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    }
    return 0.0;
}

//! Pillow's precompute_coeffs + normalize_coeffs_8bpc for one axis.
struct Coefficients
{
    int32_t ksize{};
    std::vector<int32_t> bounds;  //!< [out, 2]: first source index, count
    std::vector<int32_t> weights; //!< [out, ksize], fixed point
};

Coefficients pilCoefficients(int32_t inSize, int32_t outSize)
{
    double const scale = static_cast<double>(inSize) / outSize;
    double const filterScale = std::max(scale, 1.0);
    double const support = 2.0 * filterScale;
    Coefficients c;
    c.ksize = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    c.bounds.resize(static_cast<size_t>(outSize) * 2);
    c.weights.assign(static_cast<size_t>(outSize) * c.ksize, 0);
    std::vector<double> k(c.ksize);
    for (int32_t xx = 0; xx < outSize; ++xx)
    {
        double const center = (xx + 0.5) * scale;
        int32_t const xmin = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        int32_t const xmax = std::min(static_cast<int32_t>(center + support + 0.5), inSize) - xmin;
        double sum = 0.0;
        for (int32_t x = 0; x < xmax; ++x)
        {
            k[x] = bicubicFilter((x + xmin - center + 0.5) / filterScale);
            sum += k[x];
        }
        for (int32_t x = 0; x < xmax; ++x)
        {
            double const w = sum != 0.0 ? k[x] / sum : k[x];
            c.weights[static_cast<size_t>(xx) * c.ksize + x]
                = static_cast<int32_t>(w < 0 ? -0.5 + w * (1 << kPrecisionBits) : 0.5 + w * (1 << kPrecisionBits));
        }
        c.bounds[static_cast<size_t>(xx) * 2] = xmin;
        c.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    return c;
}

inline unsigned char clip8(int32_t value)
{
    return static_cast<unsigned char>(std::clamp(value >> kPrecisionBits, 0, 255));
}

} // namespace

std::vector<unsigned char> resizeBicubicPil(
    unsigned char const* rgb, int32_t height, int32_t width, int32_t outHeight, int32_t outWidth)
{
    if (height == outHeight && width == outWidth)
    {
        return std::vector<unsigned char>(rgb, rgb + static_cast<size_t>(height) * width * 3);
    }
    // Pillow resamples horizontally, then vertically, rounding to 8 bits in between; it skips a pass whose
    // axis keeps its size.
    std::vector<unsigned char> horizontal;
    unsigned char const* source = rgb;
    if (width != outWidth)
    {
        Coefficients const c = pilCoefficients(width, outWidth);
        horizontal.resize(static_cast<size_t>(height) * outWidth * 3);
        for (int32_t y = 0; y < height; ++y)
        {
            for (int32_t xx = 0; xx < outWidth; ++xx)
            {
                int32_t const xmin = c.bounds[xx * 2];
                int32_t const count = c.bounds[xx * 2 + 1];
                int32_t const* k = &c.weights[static_cast<size_t>(xx) * c.ksize];
                for (int32_t ch = 0; ch < 3; ++ch)
                {
                    int32_t sum = 1 << (kPrecisionBits - 1);
                    for (int32_t x = 0; x < count; ++x)
                    {
                        sum += rgb[(static_cast<size_t>(y) * width + xmin + x) * 3 + ch] * k[x];
                    }
                    horizontal[(static_cast<size_t>(y) * outWidth + xx) * 3 + ch] = clip8(sum);
                }
            }
        }
        source = horizontal.data();
    }
    if (height == outHeight)
    {
        return horizontal;
    }
    Coefficients const c = pilCoefficients(height, outHeight);
    std::vector<unsigned char> out(static_cast<size_t>(outHeight) * outWidth * 3);
    for (int32_t yy = 0; yy < outHeight; ++yy)
    {
        int32_t const ymin = c.bounds[yy * 2];
        int32_t const count = c.bounds[yy * 2 + 1];
        int32_t const* k = &c.weights[static_cast<size_t>(yy) * c.ksize];
        for (int32_t x = 0; x < outWidth * 3; ++x)
        {
            int32_t sum = 1 << (kPrecisionBits - 1);
            for (int32_t y = 0; y < count; ++y)
            {
                sum += source[static_cast<size_t>(ymin + y) * outWidth * 3 + x] * k[y];
            }
            out[static_cast<size_t>(yy) * outWidth * 3 + x] = clip8(sum);
        }
    }
    return out;
}

} // namespace vla
} // namespace trt_edgellm
