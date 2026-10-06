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

#include "vlaBackbone.h"

#include <utility>

namespace trt_edgellm
{
namespace vla
{

std::vector<rt::imageUtils::ImageData> loadImages(std::vector<std::string> const& paths)
{
    std::vector<rt::imageUtils::ImageData> images;
    for (auto const& path : paths)
    {
        images.push_back(rt::imageUtils::loadRgbImageFromFile(path));
    }
    return images;
}

rt::LLMGenerationRequest makeBackboneRequest(
    std::string const& prompt, std::vector<rt::imageUtils::ImageData> images, int32_t captureLayer, int32_t tailTokens)
{
    rt::LLMGenerationRequest request;
    request.requests.resize(1);
    rt::Message message;
    message.role = "user";
    message.contents.push_back({"text", prompt});
    request.requests[0].messages.push_back(std::move(message));
    request.requests[0].imageBuffers = std::move(images);
    request.applyChatTemplate = false;
    request.maxGenerateLength = 1;
    request.acceptHiddenLayer = captureLayer;
    request.hiddenCaptureTailTokens = tailTokens;
    // No default initializers; an uninitialized topK sizes the sampler workspace from garbage.
    request.temperature = 1.0F;
    request.topP = 1.0F;
    request.topK = 1;
    return request;
}

rt::Tensor const* runBackbone(
    rt::LLMInferenceRuntime& runtime, rt::LLMGenerationRequest const& request, cudaStream_t stream)
{
    rt::LLMGenerationResponse response;
    if (!runtime.handleRequest(request, response, stream, /*outputThinkerEmbeddings=*/true))
    {
        return nullptr;
    }
    rt::Tensor const* hidden = runtime.getBaseModelHiddenStates(request.acceptHiddenLayer);
    return hidden == nullptr || hidden->isEmpty() ? nullptr : hidden;
}

std::vector<uint8_t> tokenMask(rt::LLMInferenceRuntime const& runtime, int32_t tokenId)
{
    std::vector<uint8_t> mask;
    for (int32_t id : runtime.getBaseModelInputTokenIds().at(0))
    {
        mask.push_back(id == tokenId ? 1 : 0);
    }
    return mask;
}

} // namespace vla
} // namespace trt_edgellm
