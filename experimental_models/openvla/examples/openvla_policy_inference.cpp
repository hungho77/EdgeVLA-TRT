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

//! OpenVLA policy call: one image and an instruction in, one action out.

#include "openvlaPolicy.h"

#include "common/trtUtils.h"
#include "runtime/imageUtils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

std::string argOf(int argc, char** argv, char const* flag, std::string const& fallback = "")
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::strcmp(argv[i], flag) == 0)
        {
            return argv[i + 1];
        }
    }
    return fallback;
}

float median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char** argv)
{
    std::string const visionDir = argOf(argc, argv, "--visionDir");
    std::string const llmDir = argOf(argc, argv, "--llmEngineDir");
    std::string const imagePath = argOf(argc, argv, "--image");
    std::string const instruction = argOf(argc, argv, "--instruction");
    std::string const unnormKey = argOf(argc, argv, "--unnormKey");
    if (visionDir.empty() || llmDir.empty() || imagePath.empty() || instruction.empty() || unnormKey.empty())
    {
        std::fprintf(stderr,
            "usage: %s --visionDir DIR --llmEngineDir DIR --image frame.png --instruction TEXT --unnormKey KEY\n"
            "          [--output out.json] [--iters N] [--secondImage frame2.png]\n",
            argv[0]);
        return 2;
    }
    auto const pluginHandle = loadEdgellmPluginLib();
    rt::imageUtils::ImageData const image = rt::imageUtils::loadRgbImageFromFile(imagePath);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    {
        openvla::OpenvlaPolicy policy(visionDir, llmDir, stream);
        auto const height = static_cast<int32_t>(image.height);
        auto const width = static_cast<int32_t>(image.width);
        openvla::OpenvlaStep const step = policy.act(image.data(), height, width, instruction, unnormKey);

        int32_t const iters = std::stoi(argOf(argc, argv, "--iters", "0"));
        if (iters > 0)
        {
            std::vector<float> vision, llm;
            for (int32_t i = 0; i < iters; ++i)
            {
                openvla::OpenvlaStep const timed = policy.act(image.data(), height, width, instruction, unnormKey);
                vision.push_back(timed.visionMs);
                llm.push_back(timed.llmMs);
            }
            std::printf("median ms over %d calls: vision %.1f, LLM prefill + %zu tokens %.1f, total %.1f\n", iters,
                median(vision), step.actionIds.size(), median(llm), median(vision) + median(llm));
        }

        // A second call in the same process: its result must not depend on the first (no reuse of the previous
        // image's KV for the identical placeholder ids).
        std::string const secondPath = argOf(argc, argv, "--secondImage");
        if (!secondPath.empty())
        {
            rt::imageUtils::ImageData const second = rt::imageUtils::loadRgbImageFromFile(secondPath);
            openvla::OpenvlaStep const next = policy.act(second.data(), static_cast<int32_t>(second.height),
                static_cast<int32_t>(second.width), instruction, unnormKey);
            std::printf("second image action tokens:");
            for (int32_t id : next.actionIds)
            {
                std::printf(" %d", id);
            }
            std::printf(" (%.1f ms)\n", next.visionMs + next.llmMs);
        }

        std::string const output = argOf(argc, argv, "--output");
        if (!output.empty())
        {
            nlohmann::json out;
            out["actions"] = step.actions;
            out["action_ids"] = step.actionIds;
            out["prompt_ids"] = step.promptIds;
            out["pixel_values"] = policy.preprocess(image.data(), height, width);
            std::ofstream(output) << out.dump();
        }
        std::printf("action tokens:");
        for (int32_t id : step.actionIds)
        {
            std::printf(" %d", id);
        }
        std::printf("\nactions:");
        for (float a : step.actions)
        {
            std::printf(" %.4f", a);
        }
        std::printf("\n");
    }
    cudaStreamDestroy(stream);
    return 0;
}
