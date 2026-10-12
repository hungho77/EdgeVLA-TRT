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

#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace lingbot
{

//! One camera frame, packed [H, W, 3] 8-bit RGB, borrowed for the call.
struct LingbotView
{
    std::string camera; //!< one of the engine directory's cameras, the dataset's image keys
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct LingbotChunk
{
    std::vector<float> actions;    //!< [horizon, robot action dim] unnormalized
    std::vector<float> normalized; //!< [horizon, maxDim] the flow output
    int32_t promptTokens{0};       //!< instruction tokens before the padding
    float hostMs{0.0F};            //!< frames, then the prompt, positions and noise (overlapping the vision engine)
    float visionMs{0.0F};
    float prefixMs{0.0F};
    float actionMs{0.0F};
};

//! Real-time chunking as pi0.5's: after every Euler step, row r < overlapSteps moves towards
//! t * noise + (1 - t) * previous[startRow + r] by weight 1 on the frozen rows, then an exponential ramp down to 0.
struct LingbotRtc
{
    int32_t overlapSteps{};
    int32_t frozenSteps{};
    float rampRate{6.0F};
    int32_t startRow{-1}; //!< -1: horizon - overlapSteps
};

//! How a slice of a robot vector maps onto the model's padded vector, with its bounds_99 statistics.
struct LingbotFeature
{
    int32_t start{0}; //!< first element in the robot vector
    int32_t count{0};
    int32_t slot{0}; //!< first element in the model vector
    std::vector<float> low, high;
    bool clip{false};
};

//! LingBot-VLA 2.0 (Robbyant/lingbot-vla-v2, deploy/lingbot_vla_v2_policy.py) on the engines written by
//! experimental_models/lingbot/scripts/export_lingbot.py. Each camera frame is resized to the training size (an
//! antialiased bilinear resize in float, as torchvision's), then cut into Qwen3-VL patches by the processor's fast
//! path ((x - 127.5) / 127.5, two identical temporal frames, merge-blocked order). The instruction goes through the
//! Qwen3 chat template and is right-padded; the prefix's multimodal RoPE ids follow get_rope_index with the mask
//! (padding at 1). The state is mapped to [-1, 1] by its q01 / q99 into the model's padded layout, the chunk
//! integrates the expert from Gaussian noise at t = 1 down to 0 and is unnormalized per action feature.
//!
//! NOT thread-safe: one instance, one stream; every buffer is allocated at construction.
class LingbotPolicy
{
public:
    LingbotPolicy(std::string const& engineDir, cudaStream_t stream);
    ~LingbotPolicy() noexcept;

    LingbotPolicy(LingbotPolicy const&) = delete;
    LingbotPolicy& operator=(LingbotPolicy const&) = delete;

    //! \p noise is the initial [horizon, maxDim] Gaussian sample, drawn from the seeded generator when empty.
    LingbotChunk act(std::vector<LingbotView> const& views, std::vector<float> const& state, std::string const& task,
        std::vector<float> const& noise = {}, LingbotRtc const* rtc = nullptr);

    void resetEpisode() noexcept
    {
        mPrevious.clear();
    }
    void setNoiseSeed(uint64_t seed) noexcept
    {
        mNoiseGen.seed(seed);
    }
    void setUseCudaGraph(bool enable) noexcept
    {
        mUseCudaGraph = enable;
    }
    std::vector<std::string> const& cameras() const noexcept
    {
        return mCameras;
    }
    int32_t stateDim() const noexcept
    {
        return mStateDim;
    }
    int32_t actionDim() const noexcept
    {
        return mActionDim;
    }
    int32_t chunkSize() const noexcept
    {
        return mHorizon;
    }

    //! The instruction's token ids inside the chat template, before padding.
    std::vector<int32_t> promptIds(std::string const& task) const;
    //! The prefix's [3, prefixLength] RoPE ids for \p validText instruction tokens.
    std::vector<int32_t> positionIds(int32_t validText) const;
    //! The processor's patches of one frame, [patches, 3 * temporal * patch * patch].
    std::vector<float> patches(unsigned char const* rgb, int32_t height, int32_t width) const;

private:
    void enqueueDenoise();

    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    vla::TrtEngine mVision, mPrefixA, mPrefixB, mStep;
    rt::Tensor mContextMemory;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;

    std::vector<std::string> mCameras;
    int32_t mImageSize{256}, mPatch{16}, mTemporal{2}, mMerge{2};
    float mImageMean[3]{}, mImageStd[3]{};
    int32_t mPrefixLength{220}, mTextLength{72}, mAlignTokens{16}, mPadId{151643};
    std::vector<int32_t> mPromptPrefix, mPromptSuffix;
    int32_t mLayers{36}, mLayersA{18}, mKvHeads{8}, mHeadDim{128}, mHorizon{50}, mMaxDim{55}, mSteps{10};
    float mDt{-0.1F};
    int32_t mStateDim{0}, mActionDim{0};
    std::vector<LingbotFeature> mStateLayout, mActionLayout;

    rt::Tensor mPixelsHost, mPixels;     //!< FP16 [cameras * patches, patch dim]
    rt::Tensor mImageTokens, mDeepstack; //!< FP16 vision outputs
    rt::Tensor mPromptHost;              //!< pinned: ids (INT32), positions (INT32), masks (FP16), state (FP16)
    rt::Tensor mInputIds, mLangMask, mPositionIds, mPrefixMask, mState;
    rt::Tensor mHidden, mHiddenOut; //!< FP16 [1, P, hidden]
    rt::Tensor mKeys, mValues;      //!< FP16 [layers, 1, P, kvHeads, headDim]
    rt::Tensor mX[2], mNoise, mSeed, mWeight;
    rt::Tensor mScalars; //!< step indices (INT32), suffix start (INT32), dt (FP32): 256-byte slots
    rt::Tensor mStageHost, mOutHost;
    std::vector<float> mPrevious;
    std::mt19937_64 mNoiseGen{0};
    bool mUseCudaGraph{true};
    cudaGraphExec_t mGraph{nullptr};
    cudaEvent_t mEvents[4]{};
};

} // namespace lingbot
} // namespace trt_edgellm
