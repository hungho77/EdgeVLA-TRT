/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! GR00T N1.7 asynchronous control loop: a fixed-rate loop executes one action per tick while backbone, action
//! head and real-time chunking run on the planner thread (vla::AsyncChunker). The robot is simulated: fixed camera
//! frames and a state that drifts every tick (--drift), so the RTC seed is re-encoded against a moving arm. The
//! report shows stalls, planner latency in ticks, and the action jump at each chunk switch.

#include "gr00tN17Policy.h"

#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "vlaAsyncChunker.h"
#include "vlaBackbone.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kCaptureSlot = 1;

struct Observation
{
    int64_t tick{-1};
    std::vector<float> rawState;
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

std::vector<std::string> splitComma(std::string const& s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
        {
            out.push_back(item);
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    std::string const llmDir = argOf(argc, argv, "--llmEngineDir");
    std::string const visDir = argOf(argc, argv, "--multimodalEngineDir");
    std::string const actionDir = argOf(argc, argv, "--actionEngineDir");
    std::string const promptFile = argOf(argc, argv, "--promptFile");
    std::vector<std::string> const images = splitComma(argOf(argc, argv, "--images"));
    std::vector<float> rawState;
    for (auto const& v : splitComma(argOf(argc, argv, "--rawState")))
    {
        rawState.push_back(std::stof(v));
    }
    if (llmDir.empty() || visDir.empty() || actionDir.empty() || promptFile.empty() || images.empty()
        || rawState.empty())
    {
        std::fprintf(stderr,
            "usage: %s --llmEngineDir DIR --multimodalEngineDir DIR --actionEngineDir DIR --promptFile FILE\n"
            "          --images a.png,b.png --rawState v0,v1,... [--ticks 120] [--controlHz 30] [--overlap 8]\n"
            "          [--frozen 5] [--drift 0.3] [--imageTokenId 151655]\n",
            argv[0]);
        return 2;
    }
    int32_t const ticks = std::stoi(argOf(argc, argv, "--ticks", "120"));
    double const controlHz = std::stod(argOf(argc, argv, "--controlHz", "30"));
    int32_t const overlap = std::stoi(argOf(argc, argv, "--overlap", "8"));
    int32_t const frozen = std::stoi(argOf(argc, argv, "--frozen", "5"));
    // Small enough for SO101: joint 4's relative bounds span ~1.9, and a seed outside them is clipped.
    float const drift = std::stof(argOf(argc, argv, "--drift", "0.3"));
    int32_t const imageTokenId = std::stoi(argOf(argc, argv, "--imageTokenId", "151655"));
    std::ifstream promptStream(promptFile);
    std::string const prompt((std::istreambuf_iterator<char>(promptStream)), std::istreambuf_iterator<char>());

    auto const pluginHandles = loadEdgellmPluginLib();
    cudaStream_t planStream;
    cudaStreamCreate(&planStream);
    std::unordered_map<std::string, std::string> const noLora;
    rt::LLMInferenceRuntime backbone(llmDir, visDir, noLora, planStream);
    gr00t::Gr00tN17Policy policy(actionDir, planStream);
    auto const& cfg = policy.runner().config();
    int32_t const horizon = policy.processing().actionHorizon();
    int32_t const actionDim = policy.processing().rawActionDim();

    rt::LLMGenerationRequest const request = vla::makeBackboneRequest(prompt, vla::loadImages(images), kCaptureSlot);
    rt::Tensor noiseHost(rt::Coords(std::vector<int64_t>{cfg.actionHorizon, cfg.actionDim}), rt::DeviceType::kCPU,
        nvinfer1::DataType::kFLOAT, "gr00t::noiseHost");
    rt::Tensor noise(rt::Coords(std::vector<int64_t>{cfg.actionHorizon, cfg.actionDim}), rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "gr00t::noise");

    // A slow sinusoid per joint stands in for the arm moving under the executed actions.
    auto snapshot = [&](int64_t tick) {
        Observation observation;
        observation.tick = tick;
        observation.rawState = rawState;
        for (size_t d = 0; d < rawState.size(); ++d)
        {
            observation.rawState[d] += drift * std::sin(0.05F * static_cast<float>(tick) + static_cast<float>(d));
        }
        return observation;
    };
    int64_t lastPlanTick = -1;
    auto planner = [&](int64_t tick, Observation const& observation) {
        ELLM_CHECK(observation.tick == tick, "the planner got another tick's observation");
        std::mt19937_64 generator(static_cast<uint64_t>(tick));
        std::normal_distribution<float> normal(0.0F, 1.0F);
        float* values = noiseHost.dataPointer<float>();
        std::generate_n(values, cfg.actionHorizon * cfg.actionDim, [&] { return normal(generator); });
        cudaMemcpyAsync(noise.rawPointer(), noiseHost.rawPointer(),
            static_cast<size_t>(cfg.actionHorizon) * cfg.actionDim * sizeof(float), cudaMemcpyHostToDevice, planStream);

        rt::Tensor const* features = vla::runBackbone(backbone, request, planStream);
        ELLM_CHECK(features != nullptr, "backbone request failed");
        gr00t::Gr00tN17Policy::Rtc rtc;
        rtc.overlapSteps = overlap;
        rtc.frozenSteps = frozen;
        rtc.startRow = static_cast<int32_t>(tick - lastPlanTick);
        std::vector<float> chunk = policy.act(*features, vla::tokenMask(backbone, imageTokenId), observation.rawState,
            noise, planStream, lastPlanTick < 0 ? nullptr : &rtc);
        lastPlanTick = tick;
        return chunk;
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
                (lag < frozen ? worstSeamFrozen : worstSeam)
                    = std::max(lag < frozen ? worstSeamFrozen : worstSeam, seam);
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
    std::printf(
        "control %.0f Hz, %d ticks, horizon %d, overlap %d, frozen %d\n", controlHz, ticks, horizon, overlap, frozen);
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
    cudaStreamDestroy(planStream);
    return 0;
}
