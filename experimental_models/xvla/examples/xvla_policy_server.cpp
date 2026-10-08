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

//! X-VLA policy server: one JSON request per stdin line, one JSON reply per stdout line.
//!
//! Request:  {"cameras": {"image": "top.png", ...},     raw frames keyed by the checkpoint's camera names
//!            "state": [...],                          raw proprio state
//!            "task": "pick the cube",
//!            "rtc": {"delay": 4, "horizon": 20,       optional, real-time chunking from the previous reply
//!                    "start_row": 10},
//!            "seed": 0,                               optional, reseeds x1
//!            "reset": true}                           optional, start of an episode
//! Reply:    {"actions": [[...], ...], "timing_ms": {...}}  or {"error": "..."}

#include "xvlaPolicy.h"

#include "runtime/imageUtils.h"

#include <nlohmann/json.hpp>

#include <chrono>
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
    std::string const engineDir = argOf(argc, argv, "--engineDir");
    if (engineDir.empty())
    {
        std::fprintf(stderr, "usage: %s --engineDir DIR [--domain ID] [--cudaGraph 1]\n", argv[0]);
        return 2;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    xvla::XvlaPolicy policy(engineDir, stream);
    policy.setUseCudaGraph(argOf(argc, argv, "--cudaGraph", "1") != "0");
    std::string const domain = argOf(argc, argv, "--domain");
    if (!domain.empty())
    {
        policy.setDomainId(std::stoi(domain));
    }

    std::printf("{\"ready\":true,\"chunk\":%d,\"action_dim\":%d,\"domain\":%d}\n", policy.chunkSize(),
        policy.actionDim(), policy.domainId());
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
            auto const t0 = std::chrono::steady_clock::now();
            if (in.value("reset", false))
            {
                policy.resetEpisode();
            }
            if (in.contains("seed"))
            {
                policy.setNoiseSeed(in.at("seed").get<uint64_t>());
            }
            std::vector<rt::imageUtils::ImageData> images;
            std::vector<std::string> names;
            for (auto const& [camera, path] : in.at("cameras").items())
            {
                images.push_back(rt::imageUtils::loadRgbImageFromFile(path.get<std::string>()));
                names.push_back(camera);
            }
            std::vector<xvla::XvlaView> views;
            for (size_t i = 0; i < images.size(); ++i)
            {
                views.push_back(xvla::XvlaView{names[i], images[i].data(), static_cast<int32_t>(images[i].height),
                    static_cast<int32_t>(images[i].width)});
            }
            xvla::XvlaRtc rtc;
            bool const useRtc = in.contains("rtc");
            if (useRtc)
            {
                auto const& r = in.at("rtc");
                rtc.inferenceDelay = r.at("delay").get<int32_t>();
                rtc.executionHorizon = r.at("horizon").get<int32_t>();
                rtc.startRow = r.value("start_row", -1);
            }
            xvla::XvlaChunk const chunk = policy.act(views, in.at("state").get<std::vector<float>>(),
                in.at("task").get<std::string>(), {}, useRtc ? &rtc : nullptr);

            Json rows = Json::array();
            for (int32_t t = 0; t < policy.chunkSize(); ++t)
            {
                rows.push_back(std::vector<float>(chunk.actions.begin() + static_cast<int64_t>(t) * policy.actionDim(),
                    chunk.actions.begin() + static_cast<int64_t>(t + 1) * policy.actionDim()));
            }
            reply["actions"] = rows;
            reply["timing_ms"] = {{"vision", chunk.visionMs}, {"encoder", chunk.encoderMs},
                {"denoise", chunk.denoiseMs},
                {"total", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()}};
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
