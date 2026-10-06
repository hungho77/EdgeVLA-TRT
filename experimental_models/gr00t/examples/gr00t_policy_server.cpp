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

//! GR00T N1.7 policy server: one JSON request per stdin line, one JSON reply per stdout line.
//!
//! Request:  {"images": ["top.png", "wrist.png"],   camera frames after GR00T's eval image transform
//!            "state": [...],                        raw state, groups concatenated in modality order
//!            "instruction": "pick the cube",        after GR00T's language formalization
//!            "rtc": {"overlap": 8, "frozen": 2, "ramp_rate": 6.0},   optional, chunk inpainting
//!            "seed": 0, "noise_file": "noise.f32",  optional; noise_file holds [max_horizon, max_action_dim]
//!            "reset": true,                         optional, start of an episode
//!            "debug": true}                         optional, also return the normalized model actions
//! Reply:    {"actions": [[...], ...],               absolute raw actions [action_horizon][raw action dim]
//!            "timing_ms": {...}}  or {"error": "..."}

#include "gr00tN17ActionRunner.h"
#include "gr00tProcessing.h"

#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{

constexpr int32_t kCaptureSlot = 1;
constexpr char const* kImagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>";

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

//! GR00T N1.7's prompt: images first, then the instruction, in one user turn without a system prompt.
std::string buildPrompt(std::string const& instruction, size_t numImages)
{
    std::string text = "<|im_start|>user\n";
    for (size_t i = 0; i < numImages; ++i)
    {
        text += kImagePlaceholder;
    }
    return text + instruction + "<|im_end|>\n";
}

double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, char** argv)
{
    std::string const llmDir = argOf(argc, argv, "--llmEngineDir");
    std::string const visDir = argOf(argc, argv, "--multimodalEngineDir");
    std::string const actionDir = argOf(argc, argv, "--actionEngineDir");
    if (llmDir.empty() || visDir.empty() || actionDir.empty())
    {
        std::fprintf(stderr,
            "usage: %s --llmEngineDir DIR --multimodalEngineDir DIR --actionEngineDir DIR [--cudaGraph 1]\n"
            "  actionEngineDir holds the action engines, config.json and processing.json.\n",
            argv[0]);
        return 2;
    }

    auto const pluginHandles = loadEdgellmPluginLib();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::unordered_map<std::string, std::string> const noLora;
    rt::LLMInferenceRuntime backbone(llmDir, visDir, noLora, stream);
    gr00t::Gr00tN17ActionRunner action(actionDir, stream);
    action.setUseCudaGraph(argOf(argc, argv, "--cudaGraph", "1") != "0");
    gr00t::Gr00tProcessing const processing(actionDir + "/processing.json");
    auto const& cfg = action.config();
    int32_t const imageTokenId = std::stoi(argOf(argc, argv, "--imageTokenId", "151655"));

    size_t const noiseCount = static_cast<size_t>(cfg.actionHorizon) * cfg.actionDim;
    rt::Tensor noiseHost(rt::Coords(std::vector<int64_t>{cfg.actionHorizon, cfg.actionDim}), rt::DeviceType::kCPU,
        nvinfer1::DataType::kFLOAT, "gr00t::noiseHost");
    rt::Tensor noise(rt::Coords(std::vector<int64_t>{cfg.actionHorizon, cfg.actionDim}), rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "gr00t::noise");
    rt::Tensor actionsHost(rt::Coords(std::vector<int64_t>{cfg.actionHorizon, cfg.actionDim}), rt::DeviceType::kCPU,
        nvinfer1::DataType::kFLOAT, "gr00t::actionsHost");
    std::mt19937_64 generator(0);

    std::printf("{\"ready\":true,\"raw_state_dim\":%d,\"raw_action_dim\":%d,\"action_horizon\":%d}\n",
        processing.rawStateDim(), processing.rawActionDim(), processing.actionHorizon());
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
                action.resetEpisode();
            }
            std::vector<std::string> const images = in.at("images").get<std::vector<std::string>>();
            std::vector<float> const rawState = in.at("state").get<std::vector<float>>();

            // Noise: from a file (reproducing a reference), else N(0, 1) from the request seed or the stream.
            if (in.contains("noise_file"))
            {
                std::ifstream f(in.at("noise_file").get<std::string>(), std::ios::binary);
                f.read(static_cast<char*>(noiseHost.rawPointer()),
                    static_cast<std::streamsize>(noiseCount * sizeof(float)));
                if (!f)
                {
                    throw std::runtime_error("noise_file is shorter than [max_horizon, max_action_dim] floats");
                }
            }
            else
            {
                if (in.contains("seed"))
                {
                    generator.seed(in.at("seed").get<uint64_t>());
                }
                std::normal_distribution<float> normal(0.0F, 1.0F);
                float* values = noiseHost.dataPointer<float>();
                for (size_t i = 0; i < noiseCount; ++i)
                {
                    values[i] = normal(generator);
                }
            }
            cudaMemcpyAsync(
                noise.rawPointer(), noiseHost.rawPointer(), noiseCount * sizeof(float), cudaMemcpyHostToDevice, stream);

            rt::LLMGenerationRequest request;
            request.requests.resize(1);
            rt::Message msg;
            msg.role = "user";
            msg.contents.push_back({"text", buildPrompt(in.at("instruction").get<std::string>(), images.size())});
            request.requests[0].messages.push_back(std::move(msg));
            for (auto const& path : images)
            {
                request.requests[0].imageBuffers.push_back(rt::imageUtils::loadRgbImageFromFile(path));
            }
            request.applyChatTemplate = false;
            request.maxGenerateLength = 1;
            request.acceptHiddenLayer = kCaptureSlot;
            request.temperature = 1.0F;
            request.topP = 1.0F;
            request.topK = 1;
            rt::LLMGenerationResponse response;
            if (!backbone.handleRequest(request, response, stream, /*outputThinkerEmbeddings=*/true))
            {
                throw std::runtime_error("backbone request failed");
            }
            rt::Tensor const* features = backbone.getBaseModelHiddenStates(kCaptureSlot);
            std::vector<uint8_t> imageMask;
            for (int32_t id : backbone.getBaseModelInputTokenIds().at(0))
            {
                imageMask.push_back(id == imageTokenId ? 1 : 0);
            }
            cudaStreamSynchronize(stream);
            double const backboneMs = msSince(t0);

            auto const t1 = std::chrono::steady_clock::now();
            action.prepare(*features, imageMask, stream);
            action.encodeState(processing.normalizeState(rawState), stream);
            gr00t::Gr00tN17ActionRunner::RtcOptions rtc;
            bool const useRtc = in.contains("rtc");
            if (useRtc)
            {
                auto const& r = in.at("rtc");
                rtc.horizon = processing.actionHorizon();
                rtc.overlapSteps = r.at("overlap").get<int32_t>();
                rtc.frozenSteps = r.value("frozen", 0);
                rtc.rampRate = r.value("ramp_rate", 6.0F);
            }
            rt::Tensor const& sampled = action.sample(noise, stream, useRtc ? &rtc : nullptr);
            cudaMemcpyAsync(actionsHost.rawPointer(), sampled.rawPointer(), noiseCount * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            double const actionMs = msSince(t1);

            std::vector<float> const absolute = processing.decodeActions(actionsHost.dataPointer<float>(), rawState);
            Json rows = Json::array();
            for (int32_t t = 0; t < processing.actionHorizon(); ++t)
            {
                rows.push_back(
                    std::vector<float>(absolute.begin() + static_cast<int64_t>(t) * processing.rawActionDim(),
                        absolute.begin() + static_cast<int64_t>(t + 1) * processing.rawActionDim()));
            }
            reply["actions"] = rows;
            if (in.value("debug", false))
            {
                // Normalized model output rows [0, action_horizon), every padded column.
                float const* raw = actionsHost.dataPointer<float>();
                reply["model_actions"]
                    = std::vector<float>(raw, raw + static_cast<int64_t>(processing.actionHorizon()) * cfg.actionDim);
            }
            reply["timing_ms"] = {{"backbone", backboneMs}, {"action_head", actionMs}, {"total", msSince(t0)}};
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
