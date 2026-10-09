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

//! Write MolmoAct2's SigLIP2 patches for a PNG as raw FP32 [729, 588], for checking against the official processor:
//!   molmoact2_image_dump image.png out.bin

#include "molmoact2Image.h"
#include "runtime/imageUtils.h"

#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s IMAGE OUT\n", argv[0]);
        return 2;
    }
    auto const image = trt_edgellm::rt::imageUtils::loadRgbImageFromFile(argv[1]);
    std::vector<float> patches(729 * 588);
    trt_edgellm::molmoact2::siglipPatches(image.data(), static_cast<int32_t>(image.height),
        static_cast<int32_t>(image.width), 378, 14, patches.data());
    std::ofstream(argv[2], std::ios::binary)
        .write(reinterpret_cast<char const*>(patches.data()), patches.size() * sizeof(float));
    return 0;
}
