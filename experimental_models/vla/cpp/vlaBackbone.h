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

#pragma once

#include "common/tensor.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace vla
{

std::vector<rt::imageUtils::ImageData> loadImages(std::vector<std::string> const& paths);

//! A VLM prefill used as a VLA backbone: \p prompt is already templated (the runtime's chat template is off),
//! one token is generated, and the hidden states of \p captureLayer are kept. With \p tailTokens > 0 only the
//! last tailTokens positions are read, which lets context reuse restore the prompt before them.
rt::LLMGenerationRequest makeBackboneRequest(std::string const& prompt, std::vector<rt::imageUtils::ImageData> images,
    int32_t captureLayer, int32_t tailTokens = 0);

//! Runs \p request and returns the captured hidden states (owned by \p runtime, valid until its next request),
//! or nullptr when the request fails or captured nothing.
rt::Tensor const* runBackbone(
    rt::LLMInferenceRuntime& runtime, rt::LLMGenerationRequest const& request, cudaStream_t stream);

//! 1 where the last request's first sequence holds \p tokenId, else 0; one entry per prompt token.
std::vector<uint8_t> tokenMask(rt::LLMInferenceRuntime const& runtime, int32_t tokenId);

} // namespace vla
} // namespace trt_edgellm
