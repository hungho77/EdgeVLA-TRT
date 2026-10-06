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

//! GR00T N1.7 policy step: Edge-LLM backbone -> Gr00tN17ActionRunner -> one action chunk (normalized space).
//!
//! `--features` replaces the backbone with precomputed [tokens, embed] FP32 features (plus `--inputIds` for the
//! image mask), which isolates the action head when comparing against a reference.

#include "gr00tN17ActionRunner.h"

#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kCaptureSlot = 1;

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

template <typename T>
std::vector<T> readRaw(std::string const& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
    {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<T> data(static_cast<size_t>(f.tellg()) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    return data;
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
    std::string const promptFile = argOf(argc, argv, "--promptFile");
    std::vector<std::string> const images = splitComma(argOf(argc, argv, "--images"));
    std::string const featuresFile = argOf(argc, argv, "--features");
    std::string const inputIdsFile = argOf(argc, argv, "--inputIds");
    std::string const stateFile = argOf(argc, argv, "--state");
    std::string const noiseFile = argOf(argc, argv, "--noise");
    std::string const out = argOf(argc, argv, "--out");
    int32_t const iters = std::stoi(argOf(argc, argv, "--iters", "1"));
    int32_t const imageTokenId = std::stoi(argOf(argc, argv, "--imageTokenId", "151655"));
    bool const useBackbone = featuresFile.empty();
    if (actionDir.empty() || stateFile.empty() || noiseFile.empty()
        || (useBackbone && (llmDir.empty() || visDir.empty() || promptFile.empty() || images.empty()))
        || (!useBackbone && inputIdsFile.empty()))
    {
        std::fprintf(stderr,
            "usage: %s --actionEngineDir DIR --state state.f32 --noise noise.f32 [--out actions.f32] [--iters 1]\n"
            "          (--llmEngineDir DIR --multimodalEngineDir DIR --promptFile FILE --images a.png,b.png\n"
            "           | --features features.f32 --inputIds ids.i64)\n",
            argv[0]);
        return 1;
    }

    auto const pluginHandles = loadEdgellmPluginLib();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    gr00t::Gr00tN17ActionRunner action(actionDir, stream);
    action.setUseCudaGraph(argOf(argc, argv, "--cudaGraph", "1") != "0");
    auto const& cfg = action.config();

    std::vector<float> const state = readRaw<float>(stateFile);
    std::vector<float> const noiseHost = readRaw<float>(noiseFile);
    rt::Tensor noise(rt::Coords(std::vector<int64_t>{cfg.actionHorizon, cfg.actionDim}), rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "gr00t::noise");
    cudaMemcpy(noise.rawPointer(), noiseHost.data(), noiseHost.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::unique_ptr<rt::LLMInferenceRuntime> backbone;
    rt::LLMGenerationRequest request;
    rt::Tensor fixedFeatures;
    std::vector<uint8_t> fixedMask;
    if (useBackbone)
    {
        std::unordered_map<std::string, std::string> const noLora;
        backbone = std::make_unique<rt::LLMInferenceRuntime>(llmDir, visDir, noLora, stream);
        std::ifstream promptStream(promptFile);
        std::string const prompt((std::istreambuf_iterator<char>(promptStream)), std::istreambuf_iterator<char>());
        request.requests.resize(1);
        rt::Message msg;
        msg.role = "user";
        msg.contents.push_back({"text", prompt});
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
    }
    else
    {
        std::vector<float> const features = readRaw<float>(featuresFile);
        std::vector<int64_t> const ids = readRaw<int64_t>(inputIdsFile);
        fixedFeatures
            = rt::Tensor(rt::Coords(std::vector<int64_t>{static_cast<int64_t>(ids.size()), cfg.backboneEmbeddingDim}),
                rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "gr00t::fixedFeatures");
        cudaMemcpy(
            fixedFeatures.rawPointer(), features.data(), features.size() * sizeof(float), cudaMemcpyHostToDevice);
        for (int64_t id : ids)
        {
            fixedMask.push_back(id == imageTokenId ? 1 : 0);
        }
    }

    std::vector<double> backboneMs;
    std::vector<double> actionMs;
    std::vector<double> totalMs;
    rt::Tensor const* actions = nullptr;
    for (int32_t i = 0; i < iters; ++i)
    {
        auto const t0 = std::chrono::steady_clock::now();
        rt::Tensor const* features = &fixedFeatures;
        std::vector<uint8_t> imageMask = fixedMask;
        if (useBackbone)
        {
            rt::LLMGenerationResponse response;
            if (!backbone->handleRequest(request, response, stream, /*outputThinkerEmbeddings=*/true))
            {
                std::fprintf(stderr, "backbone request failed\n");
                return 1;
            }
            features = backbone->getBaseModelHiddenStates(kCaptureSlot);
            auto const& ids = backbone->getBaseModelInputTokenIds().at(0);
            imageMask.clear();
            for (int32_t id : ids)
            {
                imageMask.push_back(id == imageTokenId ? 1 : 0);
            }
        }
        cudaStreamSynchronize(stream);
        auto const t1 = std::chrono::steady_clock::now();
        backboneMs.push_back(msSince(t0));

        action.prepare(*features, imageMask, stream);
        action.encodeState(state, stream);
        actions = &action.sample(noise, stream);
        cudaStreamSynchronize(stream);
        actionMs.push_back(msSince(t1));
        totalMs.push_back(msSince(t0));
        if (i == 0)
        {
            int64_t const imageTokens = std::count(imageMask.begin(), imageMask.end(), 1);
            std::printf("backbone tokens %zu (image %lld)\n", imageMask.size(), static_cast<long long>(imageTokens));
        }
    }

    std::vector<float> host(static_cast<size_t>(cfg.actionHorizon) * cfg.actionDim);
    cudaMemcpy(host.data(), actions->rawPointer(), host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (!out.empty())
    {
        std::ofstream(out, std::ios::binary)
            .write(
                reinterpret_cast<char const*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(float)));
    }
    auto const p50 = [](std::vector<double> v) {
        if (v.size() > 1)
        {
            v.erase(v.begin());
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::printf("p50 over %d calls (first excluded): backbone %.1f ms, action head %.1f ms, total %.1f ms\n", iters,
        p50(backboneMs), p50(actionMs), p50(totalMs));
    cudaStreamDestroy(stream);
    return 0;
}
