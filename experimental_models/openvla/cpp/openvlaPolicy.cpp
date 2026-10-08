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

#include "openvlaPolicy.h"

#include "common/checkMacros.h"
#include "vlaImage.h"

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>

namespace trt_edgellm
{
namespace openvla
{

namespace
{

using Json = nlohmann::json;
using nvinfer1::DataType;

Json readJson(std::string const& path)
{
    std::ifstream file(path);
    ELLM_CHECK(file.good(), "OpenvlaPolicy: missing " + path);
    return Json::parse(file);
}

rt::Tensor makeTensor(
    std::vector<int64_t> const& shape, DataType type, char const* name, rt::DeviceType device = rt::DeviceType::kGPU)
{
    return rt::Tensor(rt::Coords(shape), device, type, name);
}

} // namespace

OpenvlaPolicy::OpenvlaPolicy(std::string const& visionDir, std::string const& llmEngineDir, cudaStream_t stream)
    : mStream(stream)
{
    Json const config = readJson(visionDir + "/config.json");
    ELLM_CHECK(config.at("model_family").get<std::string>() == "openvla", "OpenvlaPolicy: not an OpenVLA vision dir");
    mImageSize = config.at("image_size").get<int32_t>();
    auto const& normalize = config.at("normalize");
    ELLM_CHECK(normalize.size() == 2, "OpenvlaPolicy: expected two vision towers");
    for (int32_t t = 0; t < 2; ++t)
    {
        mMean[t] = normalize[t].at("mean").get<std::vector<float>>();
        mStd[t] = normalize[t].at("std").get<std::vector<float>>();
    }
    mNumPatches = config.at("num_patches").get<int32_t>();
    mImageTokenId = config.at("image_token_id").get<int32_t>();
    mPrompt = config.at("prompt").get<std::string>();
    mEmptyTokenId = config.at("empty_token_id").get<int32_t>();
    mBins = config.at("n_action_bins").get<int32_t>();
    mActionVocab = config.at("action_vocab_size").get<int32_t>();
    for (auto const& [key, stats] : config.at("norm_stats").items())
    {
        ActionStats s;
        s.low = stats.at("q01").get<std::vector<double>>();
        s.high = stats.at("q99").get<std::vector<double>>();
        s.mask = stats.contains("mask") ? stats.at("mask").get<std::vector<bool>>()
                                        : std::vector<bool>(s.low.size(), true);
        mStats.emplace(key, std::move(s));
    }

    mRuntime = vla::createTrtRuntime();
    mVision = vla::TrtEngine(*mRuntime, visionDir + "/vision.engine", stream);
    mContextMemory = vla::allocateSharedContextMemory({&mVision}, "openvla::visionContextMemory");
    int64_t const hidden = mVision.engine().getTensorShape("image_embeds").d[1];
    mPixelsHost = makeTensor({6, mImageSize, mImageSize}, DataType::kHALF, "openvla::pixelsHost", rt::DeviceType::kCPU);
    mPixels = makeTensor({1, 6, mImageSize, mImageSize}, DataType::kHALF, "openvla::pixels");
    mEmbeds = makeTensor({mNumPatches, hidden}, DataType::kHALF, "openvla::imageEmbeds");
    mVision.bind("pixel_values", mPixels.rawPointer());
    mVision.bind("image_embeds", mEmbeds.rawPointer());

    std::unordered_map<std::string, std::string> const noLora;
    mLlm = std::make_unique<rt::LLMInferenceRuntime>(llmEngineDir, "", noLora, stream);
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    ELLM_CHECK(mTokenizer->loadFromHF(llmEngineDir), "OpenvlaPolicy: failed to load the tokenizer");
    for (cudaEvent_t& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

OpenvlaPolicy::~OpenvlaPolicy() noexcept
{
    for (cudaEvent_t event : mEvents)
    {
        cudaEventDestroy(event);
    }
}

std::vector<float> OpenvlaPolicy::preprocess(unsigned char const* rgb, int32_t height, int32_t width) const
{
    // torchvision resize on a PIL image is PIL's bicubic resize; to_tensor, then each tower's normalization.
    std::vector<unsigned char> const resized = vla::resizeBicubicPil(rgb, height, width, mImageSize, mImageSize);
    size_t const plane = static_cast<size_t>(mImageSize) * mImageSize;
    std::vector<float> pixels(6 * plane);
    for (size_t i = 0; i < plane; ++i)
    {
        for (int32_t c = 0; c < 3; ++c)
        {
            float const value = static_cast<float>(resized[i * 3 + c]) / 255.0F;
            for (int32_t t = 0; t < 2; ++t)
            {
                pixels[(static_cast<size_t>(t) * 3 + c) * plane + i] = (value - mMean[t][c]) / mStd[t][c];
            }
        }
    }
    return pixels;
}

std::vector<int32_t> OpenvlaPolicy::promptIds(std::string const& instruction) const
{
    std::string lower = instruction;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    std::string prompt = mPrompt;
    prompt.replace(prompt.find("{instruction}"), 13, lower);
    std::vector<int32_t> text = mTokenizer->encode(prompt, /*addBos=*/true, /*addEos=*/false);
    ELLM_CHECK(
        !text.empty() && text.front() == mTokenizer->getBosId(), "OpenvlaPolicy: the prompt must start with BOS");
    if (text.back() != mEmptyTokenId)
    {
        text.push_back(mEmptyTokenId);
    }
    // OpenVLA splices the projected patches in right after BOS.
    std::vector<int32_t> ids{text.front()};
    ids.insert(ids.end(), mNumPatches, mImageTokenId);
    ids.insert(ids.end(), text.begin() + 1, text.end());
    return ids;
}

int32_t OpenvlaPolicy::actionDim(std::string const& unnormKey) const
{
    auto const it = mStats.find(unnormKey);
    ELLM_CHECK(it != mStats.end(), "OpenvlaPolicy: unknown unnorm key " + unnormKey);
    return static_cast<int32_t>(it->second.low.size());
}

std::vector<float> OpenvlaPolicy::decodeActions(
    std::vector<int32_t> const& actionIds, std::string const& unnormKey) const
{
    ActionStats const& stats = mStats.at(unnormKey);
    std::vector<float> actions(actionIds.size());
    for (size_t d = 0; d < actionIds.size(); ++d)
    {
        // Bin centers of linspace(-1, 1, bins); token vocab - 1 - k maps to bin k.
        int32_t const bin = std::clamp(mActionVocab - actionIds[d] - 1, 0, mBins - 2);
        double const lo = -1.0 + 2.0 * bin / (mBins - 1);
        double const hi = -1.0 + 2.0 * (bin + 1) / (mBins - 1);
        double const normalized = (lo + hi) / 2.0;
        actions[d] = static_cast<float>(
            stats.mask[d] ? 0.5 * (normalized + 1.0) * (stats.high[d] - stats.low[d]) + stats.low[d] : normalized);
    }
    return actions;
}

OpenvlaStep OpenvlaPolicy::act(unsigned char const* rgb, int32_t height, int32_t width, std::string const& instruction,
    std::string const& unnormKey)
{
    int32_t const dims = actionDim(unnormKey);
    OpenvlaStep step;
    std::vector<float> const pixels = preprocess(rgb, height, width);
    auto* staged = static_cast<__half*>(mPixelsHost.rawPointer());
    for (size_t i = 0; i < pixels.size(); ++i)
    {
        staged[i] = __float2half(pixels[i]);
    }
    CUDA_CHECK(
        cudaMemcpyAsync(mPixels.rawPointer(), staged, pixels.size() * sizeof(__half), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaEventRecord(mEvents[0], mStream));
    ELLM_CHECK(mVision.enqueue(mStream), "OpenvlaPolicy: vision enqueue failed");
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));

    step.promptIds = promptIds(instruction);
    rt::LLMGenerationRequest request;
    request.requests.resize(1);
    rt::Message message;
    message.role = "user";
    message.contents.push_back({"text", ""});
    request.requests[0].messages.push_back(std::move(message));
    request.preTokenizedInputIds = {step.promptIds};
    request.precomputedImageEmbeddings = &mEmbeds;
    request.applyChatTemplate = false;
    request.maxGenerateLength = dims;
    request.temperature = 1.0F;
    request.topP = 1.0F;
    request.topK = 1;
    rt::LLMGenerationResponse response;
    auto const start = std::chrono::steady_clock::now();
    ELLM_CHECK(mLlm->handleRequest(request, response, mStream), "OpenvlaPolicy: LLM request failed");
    step.llmMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    CUDA_CHECK(cudaEventElapsedTime(&step.visionMs, mEvents[0], mEvents[1]));

    ELLM_CHECK(!response.outputIds.empty() && static_cast<int32_t>(response.outputIds[0].size()) >= dims,
        "OpenvlaPolicy: the LLM stopped before producing every action token");
    step.actionIds.assign(response.outputIds[0].end() - dims, response.outputIds[0].end());
    step.actions = decodeActions(step.actionIds, unnormKey);
    return step;
}

} // namespace openvla
} // namespace trt_edgellm
