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

#pragma once

#include "common/tensor.h"
#include "runtime/llmInferenceRuntime.h"
#include "tokenizer/tokenizer.h"
#include "vlaEngine.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace openvla
{

struct OpenvlaStep
{
    std::vector<float> actions;     //!< unnormalized, one value per action dimension
    std::vector<int32_t> actionIds; //!< the greedy action tokens
    std::vector<int32_t> promptIds; //!< BOS, image placeholders, prompt, empty token
    float visionMs{0.0F};
    float llmMs{0.0F};
};

//! OpenVLA on two engines: the fused DINOv2 + SigLIP vision backbone with its projector
//! (experimental_models/openvla/scripts/export_openvla_vision.py) and the Llama-2 LLM through Edge-LLM's runtime,
//! which takes the projected patches as precomputed image embeddings. Reproduces the checkpoint's
//! predict_action: PIL bicubic resize to 224, the processor's per-tower normalization, the "In: What action should
//! the robot take to ...?\nOut:" prompt with the empty token appended, greedy decoding of one token per action
//! dimension, the token-to-bin mapping and the dataset's q01 / q99 unnormalization.
//!
//! NOT thread-safe: one instance, one stream.
class OpenvlaPolicy
{
public:
    OpenvlaPolicy(std::string const& visionDir, std::string const& llmEngineDir, cudaStream_t stream);
    ~OpenvlaPolicy() noexcept;

    OpenvlaPolicy(OpenvlaPolicy const&) = delete;
    OpenvlaPolicy& operator=(OpenvlaPolicy const&) = delete;

    //! \p rgb is [height, width, 3] 8-bit RGB; \p unnormKey names the dataset statistics (e.g. "bridge_orig").
    OpenvlaStep act(unsigned char const* rgb, int32_t height, int32_t width, std::string const& instruction,
        std::string const& unnormKey);

    //! [6, size, size]: the image normalized for each tower, stacked on channels.
    std::vector<float> preprocess(unsigned char const* rgb, int32_t height, int32_t width) const;
    std::vector<int32_t> promptIds(std::string const& instruction) const;
    std::vector<float> decodeActions(std::vector<int32_t> const& actionIds, std::string const& unnormKey) const;
    int32_t actionDim(std::string const& unnormKey) const;
    //! The dataset statistics the checkpoint carries (valid unnormKey values).
    std::vector<std::string> unnormKeys() const
    {
        std::vector<std::string> keys;
        for (auto const& entry : mStats)
        {
            keys.push_back(entry.first);
        }
        return keys;
    }

private:
    struct ActionStats
    {
        std::vector<double> low;  //!< q01
        std::vector<double> high; //!< q99
        std::vector<bool> mask;   //!< false: the dimension is not normalized (e.g. a binary gripper)
    };

    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    vla::TrtEngine mVision;
    rt::Tensor mContextMemory;
    std::unique_ptr<rt::LLMInferenceRuntime> mLlm;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;

    int32_t mImageSize{224};
    std::vector<float> mMean[2], mStd[2];
    int32_t mNumPatches{256};
    int32_t mImageTokenId{32000};
    std::string mPrompt;
    int32_t mEmptyTokenId{29871};
    int32_t mBins{256};
    int32_t mActionVocab{32000};
    std::map<std::string, ActionStats> mStats;

    rt::Tensor mPixelsHost; //!< pinned FP16 [6, size, size]
    rt::Tensor mPixels;
    rt::Tensor mEmbeds; //!< [numPatches, hidden] FP16
    cudaEvent_t mEvents[2]{};
};

} // namespace openvla
} // namespace trt_edgellm
