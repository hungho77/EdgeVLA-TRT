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
//! The host inputs LingbotPolicy builds for one observation, for checking them against the official processor:
//!   lingbot_inputs_dump ENGINE_DIR OUT_PREFIX "task" camera.png [camera.png ...]
//! prints {"ids": [...], "positions": [[...], [...], [...]]} and writes OUT_PREFIX.f32, every camera's patches
//! [patches, patch dim] in the engine directory's camera order (FP32, before the FP16 cast).

#include "lingbotPolicy.h"
#include "runtime/imageUtils.h"

#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

int main(int argc, char** argv)
{
    using namespace trt_edgellm;
    if (argc < 5)
    {
        std::fprintf(stderr, "usage: %s ENGINE_DIR OUT_PREFIX TASK CAMERA.png [CAMERA.png ...]\n", argv[0]);
        return 2;
    }
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    {
        lingbot::LingbotPolicy policy(argv[1], stream);
        std::vector<int32_t> ids = policy.promptIds(argv[3]);
        nlohmann::json out;
        out["ids"] = ids;
        std::vector<int32_t> const positions = policy.positionIds(static_cast<int32_t>(ids.size()));
        size_t const length = positions.size() / 3;
        out["positions"] = {std::vector<int32_t>(positions.begin(), positions.begin() + length),
            std::vector<int32_t>(positions.begin() + length, positions.begin() + 2 * length),
            std::vector<int32_t>(positions.begin() + 2 * length, positions.end())};
        std::ofstream file(std::string(argv[2]) + ".f32", std::ios::binary);
        for (int i = 4; i < argc; ++i)
        {
            auto const image = rt::imageUtils::loadRgbImageFromFile(argv[i]);
            std::vector<float> const p
                = policy.patches(image.data(), static_cast<int32_t>(image.height), static_cast<int32_t>(image.width));
            file.write(reinterpret_cast<char const*>(p.data()), static_cast<std::streamsize>(p.size() * sizeof(float)));
        }
        std::printf("%s\n", out.dump().c_str());
    }
    cudaStreamDestroy(stream);
    return 0;
}
