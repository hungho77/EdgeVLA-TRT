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
#include "tokenizer/tokenizer.h"
#include "vlaEngine.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace smolvla
{

//! One camera frame, tightly packed row-major [height, width, 3] 8-bit RGB, borrowed for the call.
struct SmolvlaView
{
    std::string camera; //!< the dataset's camera name (config.json "cameras"), e.g. "top"
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct SmolvlaObservation
{
    std::vector<SmolvlaView> views;
    std::vector<float> state; //!< robot units, state_dim values
    std::string task;
};

struct SmolvlaChunk
{
    std::vector<float> normalized; //!< [chunk, maxActionDim], the denoised model output
    std::vector<float> robot;      //!< [chunk, actionDim], robot units
    std::vector<int64_t> tokenIds;
    float visualMs{0.0F};
    float prefixMs{0.0F};
    float denoiseMs{0.0F};
};

//! SmolVLA (LeRobot 0.6.1) on the visual / prefix / denoise engines written by
//! ``tensorrt_edgellm.models.smolvla.export``, with LeRobot's pre- and post-processing: resize with
//! top-left padding to the model's square size, [-1, 1] pixels, task + newline tokens, mean/std
//! state normalization, and the inverse on the actions.
//!
//! NOT thread-safe: one instance, one stream; every buffer is allocated at construction.
class SmolvlaPolicy
{
public:
    SmolvlaPolicy(std::string const& engineDir, cudaStream_t stream);
    ~SmolvlaPolicy() noexcept;

    SmolvlaPolicy(SmolvlaPolicy const&) = delete;
    SmolvlaPolicy& operator=(SmolvlaPolicy const&) = delete;

    //! \p noise is x_0, [chunk, maxActionDim]; when empty one is drawn from the seeded generator.
    SmolvlaChunk act(SmolvlaObservation const& observation, std::vector<float> const& noise = {});

    void setNoiseSeed(uint64_t seed) noexcept
    {
        mNoiseGen.seed(seed);
    }
    //! Replay the denoise loop as a CUDA graph, captured once per prefix length (default on).
    void setUseCudaGraph(bool enable) noexcept
    {
        mUseCudaGraph = enable;
    }

    int32_t chunkSize() const noexcept
    {
        return mChunk;
    }
    int32_t actionDim() const noexcept
    {
        return mActionDim;
    }

    //! LeRobot's resize_with_pad (bilinear, align_corners=False, zeros on top and left) then x * 2 - 1,
    //! as planar [3, size, size].
    static std::vector<float> preprocessView(unsigned char const* rgb, int32_t height, int32_t width, int32_t size);
    std::vector<int64_t> tokenize(std::string const& task) const;
    std::vector<float> normalizeState(std::vector<float> const& state) const;

private:
    void enqueueDenoiseLoop(int64_t prefixLen);

    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    vla::TrtEngine mVisual;
    vla::TrtEngine mPrefix;
    vla::TrtEngine mDenoise;
    rt::Tensor mContextMemory;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;

    int32_t mImageSize{512};
    int32_t mImageTokens{64};
    int32_t mHidden{960};
    int32_t mMaxViews{3};
    int32_t mMaxTokens{48};
    int32_t mMaxPrefix{0};
    int32_t mChunk{50};
    int32_t mMaxActionDim{32};
    int32_t mMaxStateDim{32};
    int32_t mStateDim{0};
    int32_t mActionDim{0};
    int32_t mNumSteps{10};
    int32_t mKVHeads{5};
    int32_t mHeadDim{64};
    std::vector<std::string> mCameras;
    std::vector<std::string> mKVNames;
    std::string mPromptSuffix{"\n"};
    std::vector<float> mStateMean, mStateStd, mActionMean, mActionStd;
    float mStateEps{1e-8F};

    rt::Tensor mPixelsHost; //!< pinned fp16 [maxViews, 3, size, size]
    rt::Tensor mPixels;
    rt::Tensor mImageFeatures; //!< [maxViews * imageTokens, hidden] fp16, viewed as [1, n, hidden]
    rt::Tensor mTokensHost;    //!< pinned int64
    rt::Tensor mTokens;
    rt::Tensor mStateHost; //!< pinned fp32 [maxStateDim]
    rt::Tensor mState;
    std::vector<rt::Tensor> mKV; //!< [1, maxPrefix, kvHeads, headDim] fp16 each
    rt::Tensor mX[2];            //!< x_t ping-pong, fp32 [1, chunk, maxActionDim]
    rt::Tensor mNoiseHost;       //!< pinned
    rt::Tensor mTimesteps;       //!< fp32 [numSteps], one per step
    rt::Tensor mDt;              //!< fp32 scalar
    rt::Tensor mOutHost;         //!< pinned fp32 [chunk, maxActionDim]

    std::mt19937_64 mNoiseGen{0};
    bool mUseCudaGraph{true};
    std::map<int64_t, cudaGraphExec_t> mGraphs; //!< keyed by prefix length
    cudaEvent_t mEvents[4]{};
};

} // namespace smolvla
} // namespace trt_edgellm
