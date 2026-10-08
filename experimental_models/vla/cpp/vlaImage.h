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
#include <vector>

namespace trt_edgellm
{
namespace vla
{

//! PIL's Image.resize(BICUBIC) on 8-bit RGB [height, width, 3]: separable, antialiased, 22-bit fixed point,
//! horizontal pass then vertical, rounded to 8 bits in between; bit-identical to Pillow.
std::vector<unsigned char> resizeBicubicPil(
    unsigned char const* rgb, int32_t height, int32_t width, int32_t outHeight, int32_t outWidth);

} // namespace vla
} // namespace trt_edgellm
