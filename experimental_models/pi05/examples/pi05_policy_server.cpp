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

//! pi0.5 policy server: one JSON request per stdin line, one JSON reply per stdout line.
//!
//! Request:  {"cameras": {"observation/image": "top.png", ...},  frames keyed by the contract's camera slots
//!            "state": [...],                                    robot units, the embodiment's width
//!            "task": "pick the cube",
//!            "rtc": {"overlap": 5, "frozen": 2,                 optional, real-time chunking from the previous reply
//!                    "ramp_rate": 6.0, "start_row": 5},
//!            "reset": true}                                     optional, start of an episode
//! Reply:    {"actions": [[...], ...], "timing_ms": {...}}  or {"error": "..."}

#include "common/trtUtils.h"
#include "runtime/pi05Policy.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
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
        std::fprintf(stderr, "usage: %s --engineDir DIR [--steps N] [--seed N] [--cudaGraph 1]\n", argv[0]);
        return 2;
    }

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // Only the action engine carries a plugin node.
    auto const pluginHandles = loadEdgellmPluginLib();
    pi05::Pi05Policy policy(engineDir, stream);
    pi05::Pi05Runtime& runtime = policy.runtime();
    runtime.setNoiseSeed(std::stoull(argOf(argc, argv, "--seed", "0")));
    runtime.setUseCudaGraph(argOf(argc, argv, "--cudaGraph", "1") != "0");
    std::string const steps = argOf(argc, argv, "--steps");
    if (!steps.empty())
    {
        runtime.setNumDenoiseSteps(std::stoi(steps));
    }

    std::printf("{\"ready\":true,\"horizon\":%d,\"robot_action_dim\":%d}\n", policy.contract().actionHorizon,
        policy.robotActionDim());
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
            if (in.value("reset", false))
            {
                policy.resetEpisode();
            }
            pi05::Pi05Observation observation;
            for (auto const& [slot, path] : in.at("cameras").items())
            {
                pi05::Pi05CameraView view;
                view.name = slot;
                view.imagePath = path.get<std::string>();
                observation.cameras.push_back(view);
            }
            observation.state = in.at("state").get<std::vector<float>>();
            observation.task = in.at("task").get<std::string>();

            pi05::Pi05Rtc rtc;
            bool const useRtc = in.contains("rtc");
            if (useRtc)
            {
                auto const& r = in.at("rtc");
                rtc.overlapSteps = r.at("overlap").get<int32_t>();
                rtc.frozenSteps = r.value("frozen", 0);
                rtc.rampRate = r.value("ramp_rate", 6.0F);
                rtc.startRow = r.value("start_row", -1);
            }
            pi05::Pi05ActionChunk const chunk = policy.infer(observation, useRtc ? &rtc : nullptr);

            int32_t const dim = policy.robotActionDim();
            Json rows = Json::array();
            for (int32_t t = 0; t < chunk.horizon; ++t)
            {
                rows.push_back(std::vector<float>(chunk.robotActions.begin() + static_cast<int64_t>(t) * dim,
                    chunk.robotActions.begin() + static_cast<int64_t>(t + 1) * dim));
            }
            reply["actions"] = rows;
            reply["timing_ms"] = {{"engines", chunk.timings.engineMs}, {"total", chunk.timings.policyMs}};
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
