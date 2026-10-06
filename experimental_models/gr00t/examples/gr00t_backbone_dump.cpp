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

//! Runs a GR00T backbone engine (exported with `emit_hidden_states`) on one observation and writes the
//! full-sequence hidden states the action head cross-attends to, as raw float32 [tokens, hidden].
//! `--iters` repeats the call and reports latency.

#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

//! Registry slot the runtime publishes the captured hidden states under; any non-zero layer index works.
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
    std::string const promptFile = argOf(argc, argv, "--promptFile");
    std::vector<std::string> const images = splitComma(argOf(argc, argv, "--images"));
    std::string const out = argOf(argc, argv, "--out");
    int32_t const iters = std::stoi(argOf(argc, argv, "--iters", "1"));
    if (llmDir.empty() || visDir.empty() || promptFile.empty() || images.empty())
    {
        std::fprintf(stderr,
            "usage: %s --llmEngineDir DIR --multimodalEngineDir DIR --promptFile FILE --images a.png,b.png\n"
            "          [--out hidden.bin] [--iters 1]\n"
            "  promptFile holds the already-templated prompt, one <|vision_start|><|image_pad|><|vision_end|>\n"
            "  per image in order.\n",
            argv[0]);
        return 1;
    }
    std::ifstream promptStream(promptFile);
    std::string const prompt((std::istreambuf_iterator<char>(promptStream)), std::istreambuf_iterator<char>());

    auto const pluginHandles = loadEdgellmPluginLib();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::unordered_map<std::string, std::string> const noLora;
    rt::LLMInferenceRuntime runtime(llmDir, visDir, noLora, stream);

    rt::LLMGenerationRequest request;
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

    std::vector<double> latencies;
    rt::Tensor const* hidden = nullptr;
    for (int32_t i = 0; i < iters; ++i)
    {
        rt::LLMGenerationResponse response;
        auto const t0 = std::chrono::steady_clock::now();
        bool const ok = runtime.handleRequest(request, response, stream, /*outputThinkerEmbeddings=*/true);
        cudaStreamSynchronize(stream);
        latencies.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        hidden = ok ? runtime.getBaseModelHiddenStates(kCaptureSlot) : nullptr;
        if (hidden == nullptr || hidden->isEmpty())
        {
            std::fprintf(stderr, "backbone request failed\n");
            return 1;
        }
    }

    auto const shape = hidden->getShape();
    int64_t const width = shape[shape.getNumDims() - 1];
    int64_t const tokens = shape.volume() / width;
    size_t const count = static_cast<size_t>(shape.volume());
    std::vector<float> host(count);
    if (hidden->getDataType() == nvinfer1::DataType::kHALF)
    {
        std::vector<__half> raw(count);
        cudaMemcpy(raw.data(), hidden->rawPointer(), count * sizeof(__half), cudaMemcpyDefault);
        std::transform(raw.begin(), raw.end(), host.begin(), [](__half h) { return __half2float(h); });
    }
    else
    {
        cudaMemcpy(host.data(), hidden->rawPointer(), count * sizeof(float), cudaMemcpyDefault);
    }
    std::printf("hidden states: %lld tokens x %lld\n", static_cast<long long>(tokens), static_cast<long long>(width));
    if (!out.empty())
    {
        std::ofstream(out, std::ios::binary)
            .write(reinterpret_cast<char const*>(host.data()), static_cast<std::streamsize>(count * sizeof(float)));
    }
    if (iters > 1)
    {
        std::vector<double> steady(latencies.begin() + 1, latencies.end());
        std::sort(steady.begin(), steady.end());
        double sum = 0;
        for (double v : steady)
        {
            sum += v;
        }
        std::printf("backbone latency over %zu calls after warm-up: mean %.1f ms  p50 %.1f ms\n", steady.size(),
            sum / static_cast<double>(steady.size()), steady[steady.size() / 2]);
    }
    cudaStreamDestroy(stream);
    return 0;
}
