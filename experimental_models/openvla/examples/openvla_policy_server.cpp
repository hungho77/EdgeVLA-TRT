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

//! OpenVLA policy server: one JSON request per stdin line, one JSON reply per stdout line.
//!
//! Request:  {"image": "frame.png",                    the primary camera frame
//!            "instruction": "pick up the banana",
//!            "unnorm_key": "bridge_orig"}             the dataset statistics that unnormalize the actions
//! Reply:    {"actions": [[...]], "action_tokens": [...], "timing_ms": {...}}  or {"error": "..."}

#include "openvlaPolicy.h"

#include "common/trtUtils.h"
#include "runtime/imageUtils.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

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

} // namespace

int main(int argc, char** argv)
{
    std::string const visionDir = argOf(argc, argv, "--visionDir");
    std::string const llmDir = argOf(argc, argv, "--llmEngineDir");
    if (visionDir.empty() || llmDir.empty())
    {
        std::fprintf(stderr, "usage: %s --visionDir DIR --llmEngineDir DIR\n", argv[0]);
        return 2;
    }

    auto const pluginHandle = loadEdgellmPluginLib();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    openvla::OpenvlaPolicy policy(visionDir, llmDir, stream);

    std::printf("{\"ready\":true}\n");
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
        {
            break;
        }
        Json reply;
        try
        {
            Json const in = Json::parse(line);
            rt::imageUtils::ImageData const image
                = rt::imageUtils::loadRgbImageFromFile(in.at("image").get<std::string>());
            openvla::OpenvlaStep const step
                = policy.act(image.data(), static_cast<int32_t>(image.height), static_cast<int32_t>(image.width),
                    in.at("instruction").get<std::string>(), in.at("unnorm_key").get<std::string>());
            reply["actions"] = Json::array({step.actions});
            reply["action_tokens"] = step.actionIds;
            reply["timing_ms"] = {{"vision", step.visionMs}, {"llm", step.llmMs}};
        }
        catch (std::exception const& e)
        {
            reply = {{"error", e.what()}};
        }
        std::printf("%s\n", reply.dump().c_str());
        std::fflush(stdout);
    }
    cudaStreamDestroy(stream);
    return 0;
}
