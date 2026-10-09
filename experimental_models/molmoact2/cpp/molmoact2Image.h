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
namespace molmoact2
{

//! MolmoAct2's SigLIP2 input for one 8-bit RGB frame [height, width, 3]: resized to \p size x \p size as torchvision's
//! Resize(BILINEAR, antialias=False) on uint8 does (float bilinear, align_corners=False, round half to even; the
//! fused multiply-adds reproduce PyTorch's CPU kernel bit for bit), scaled to [-1, 1] and cut into \p patch-pixel
//! patches, each flattened (row, column, channel): [(size / patch)^2, patch * patch * 3] written to \p out.
void siglipPatches(unsigned char const* rgb, int32_t height, int32_t width, int32_t size, int32_t patch, float* out);

} // namespace molmoact2
} // namespace trt_edgellm
