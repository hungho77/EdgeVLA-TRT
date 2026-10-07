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

//! SmolVLA asynchronous control loop: a fixed-rate loop executes one action per tick while the policy, with
//! real-time chunking, runs on the planner thread (vla::AsyncChunker). The robot is simulated: fixed camera
//! frames and a state that drifts every tick (--drift), so the RTC seed is re-encoded against a moving arm. The
//! report shows stalls, planner latency in ticks, and the action jump at each chunk switch.

#include "smolvlaPolicy.h"

#include "runtime/imageUtils.h"

#include "common/checkMacros.h"
#include "common/trtUtils.h"
#include "vlaAsyncChunker.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace trt_edgellm;

namespace
{

struct Observation
{
    int64_t tick{-1};
    std::vector<float> state;
};

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
    std::string const inputFile = argOf(argc, argv, "--inputFile");
    if (engineDir.empty() || inputFile.empty())
    {
        std::fprintf(stderr,
            "usage: %s --engineDir DIR --inputFile observation.json [--ticks 300] [--controlHz 30] [--overlap 20]\n"
            "          [--frozen 6] [--drift 0.3] [--rtc 1]\n"
            "  observation.json is smolvla_policy_inference's request; its frames stay fixed, its state drifts.\n",
            argv[0]);
        return 2;
    }
    int32_t const ticks = std::stoi(argOf(argc, argv, "--ticks", "300"));
    double const controlHz = std::stod(argOf(argc, argv, "--controlHz", "30"));
    int32_t const overlap = std::stoi(argOf(argc, argv, "--overlap", "20"));
    int32_t const frozen = std::stoi(argOf(argc, argv, "--frozen", "6"));
    float const drift = std::stof(argOf(argc, argv, "--drift", "0.3"));
    bool const useRtc = argOf(argc, argv, "--rtc", "1") != "0";

    nlohmann::json const request = nlohmann::json::parse(std::ifstream(inputFile));
    std::vector<rt::imageUtils::ImageData> images;
    smolvla::SmolvlaObservation base;
    for (auto const& [name, path] : request.at("cameras").items())
    {
        images.push_back(rt::imageUtils::loadRgbImageFromFile(path.get<std::string>()));
        base.views.push_back(smolvla::SmolvlaView{name, images.back().data(),
            static_cast<int32_t>(images.back().height), static_cast<int32_t>(images.back().width)});
    }
    base.state = request.at("state").get<std::vector<float>>();
    base.task = request.at("task").get<std::string>();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    smolvla::SmolvlaPolicy policy(engineDir, stream);
    // The first call pays TensorRT's lazy initialization and the CUDA-graph capture, and turning inpainting on
    // re-captures once; both happen here rather than inside the control loop.
    policy.act(base);
    policy.resetEpisode();
    int32_t const horizon = policy.chunkSize();
    int32_t const actionDim = policy.actionDim();

    auto snapshot = [&](int64_t tick) {
        Observation observation;
        observation.tick = tick;
        observation.state = base.state;
        for (size_t d = 0; d < observation.state.size(); ++d)
        {
            observation.state[d] += drift * std::sin(0.05F * static_cast<float>(tick) + static_cast<float>(d));
        }
        return observation;
    };
    int64_t lastPlanTick = -1;
    auto planner = [&](int64_t tick, Observation const& observation) {
        ELLM_CHECK(observation.tick == tick, "the planner got another tick's observation");
        smolvla::SmolvlaObservation request = base;
        request.state = observation.state;
        smolvla::SmolvlaRtc rtc;
        rtc.inferenceDelay = frozen;
        rtc.executionHorizon = overlap;
        rtc.startRow = static_cast<int32_t>(tick - lastPlanTick);
        smolvla::SmolvlaChunk const chunk = policy.act(request, {}, useRtc && lastPlanTick >= 0 ? &rtc : nullptr);
        lastPlanTick = tick;
        return chunk.robot;
    };
    vla::AsyncChunker<Observation> chunker(horizon, actionDim, horizon - overlap, snapshot, planner);

    auto const period = std::chrono::duration<double>(1.0 / controlHz);
    auto next = std::chrono::steady_clock::now();
    int32_t executed = 0;
    int32_t stalls = 0;
    int64_t chunkTick = -1;
    std::vector<float> previousChunk;
    std::vector<int64_t> lags;
    double worstSeamFrozen = 0.0;
    double worstSeam = 0.0;
    for (int64_t tick = 0; tick < ticks; ++tick)
    {
        std::this_thread::sleep_until(next);
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        float const* action = chunker.step(tick);
        auto const& current = chunker.current();
        if (current.observationIndex != chunkTick)
        {
            int64_t const lag = chunker.lastAdoptionLag();
            int64_t const oldRow = tick - chunkTick;
            if (!previousChunk.empty() && action != nullptr && oldRow < horizon)
            {
                double seam = 0.0;
                for (int32_t d = 0; d < actionDim; ++d)
                {
                    seam = std::max(
                        seam, std::fabs(static_cast<double>(action[d]) - previousChunk[oldRow * actionDim + d]));
                }
                double& worst = lag < frozen ? worstSeamFrozen : worstSeam;
                worst = std::max(worst, seam);
            }
            if (chunkTick >= 0)
            {
                lags.push_back(lag);
                if (lag >= frozen)
                {
                    std::printf("warning: tick %lld switched %lld ticks into the new chunk, past its %d frozen rows\n",
                        static_cast<long long>(tick), static_cast<long long>(lag), frozen);
                }
            }
            chunkTick = current.observationIndex;
            previousChunk = current.actions;
        }
        if (action != nullptr)
        {
            ++executed;
        }
        else if (chunkTick >= 0)
        {
            ++stalls;
        }
    }
    chunker.stop();

    std::sort(lags.begin(), lags.end());
    std::printf("control %.0f Hz, %d ticks, horizon %d, overlap %d, frozen %d, RTC %s\n", controlHz, ticks, horizon,
        overlap, frozen, useRtc ? "on" : "off");
    std::printf(
        "  actions executed      : %d (%d ticks before the first chunk)\n", executed, ticks - executed - stalls);
    std::printf("  stalls after start    : %d\n", stalls);
    std::printf("  chunks                : %lld\n", static_cast<long long>(chunker.chunksCompleted()));
    if (!lags.empty())
    {
        std::printf("  planner lag (ticks)   : min %lld, median %lld, max %lld\n", static_cast<long long>(lags.front()),
            static_cast<long long>(lags[lags.size() / 2]), static_cast<long long>(lags.back()));
    }
    std::printf("  switch jump, max |d|  : %.4f within frozen rows, %.4f beyond\n", worstSeamFrozen, worstSeam);
    cudaStreamDestroy(stream);
    return 0;
}
