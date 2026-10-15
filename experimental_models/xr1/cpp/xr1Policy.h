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
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace xr1
{

//! One camera frame, packed [H, W, 3] 8-bit RGB, borrowed for the call.
struct Xr1View
{
    std::string camera; //!< one of the engine directory's cameras
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct Xr1Chunk
{
    std::vector<float> actions;    //!< [horizon, action dim] decoded with the robot type's mean / std
    std::vector<float> normalized; //!< [horizon, maxDim] the flow output
    int32_t promptTokens{0};
    float hostMs{0.0F}; //!< frames, then the prompt and RoPE ids (overlapping the vision engine)
    float visionMs{0.0F};
    float prefixMs{0.0F};
    float actionMs{0.0F};
};

//! Real-time chunking as pi0.5's, for a flow from noise at t = 0: after every Euler step, row r < overlapSteps
//! moves towards (1 - t) * noise + t * previous[startRow + r] by weight 1 on the frozen rows, then an exponential
//! ramp down to 0.
struct Xr1Rtc
{
    int32_t overlapSteps{};
    int32_t frozenSteps{};
    float rampRate{6.0F};
    int32_t startRow{-1}; //!< -1: horizon - overlapSteps
};

//! Xiaomi-Robotics-1 (MiBoTForActionGeneration, its RoboCasa client) on the engines written by
//! experimental_models/xr1/scripts/export_xr1.py. Each view is centre-cropped by crop_ratio and resized back with
//! PIL's bilinear filter (8-bit), then cut into Qwen3-VL patches by the slow processor's path (rescale in float64 to
//! float32, (x - mean) / std in float32). The prompt is the chat template's ids around the instruction (the image
//! placeholders in place) with get_rope_index's multimodal RoPE ids; the robot state is zero-padded unnormalized; the
//! DiT integrates the chunk from the official request's fixed noise (seed 42) at t = 0 to 1, and the active action
//! dims are decoded with the robot type's mean / std.
//!
//! NOT thread-safe: one instance, one stream; buffers are allocated at construction for the engines' longest prompt.
class Xr1Policy
{
public:
    Xr1Policy(std::string const& engineDir, cudaStream_t stream);
    ~Xr1Policy() noexcept;

    Xr1Policy(Xr1Policy const&) = delete;
    Xr1Policy& operator=(Xr1Policy const&) = delete;

    //! \p noise is the initial [horizon, maxDim] sample, the official request's fixed noise when empty.
    Xr1Chunk act(std::vector<Xr1View> const& views, std::vector<float> const& state, std::string const& task,
        std::vector<float> const& noise = {}, Xr1Rtc const* rtc = nullptr);

    void resetEpisode() noexcept
    {
        mPrevious.clear();
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

    std::vector<int32_t> promptIds(std::string const& task) const;
    //! get_rope_index over \p ids: [3, ids.size()].
    std::vector<int32_t> positionIds(std::vector<int32_t> const& ids) const;
    //! The slow processor's patches of one cropped view.
    std::vector<float> patches(unsigned char const* rgb, int32_t height, int32_t width) const;

private:
    void enqueueDenoise(int64_t length);

    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    vla::TrtEngine mVision, mPrefixA, mPrefixB, mStep;
    rt::Tensor mContextMemory;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;

    std::vector<std::string> mCameras;
    int32_t mImageSize{256}, mPatch{16}, mTemporal{2}, mMerge{2};
    float mCropRatio{0.95F};
    float mImageMean[3]{}, mImageStd[3]{};
    int32_t mImageTokenId{151655};
    std::vector<int32_t> mPromptHead, mPromptTail;
    int32_t mLayers{36}, mLayersA{18}, mKvHeads{8}, mHeadDim{128}, mHorizon{10}, mMaxDim{60}, mSteps{5};
    int32_t mStateDim{8}, mStateWidth{60}, mActionDim{7};
    int64_t mMaxPrompt{384};
    std::vector<float> mNoise, mActionMean, mActionStd, mActionMask;

    rt::Tensor mPixelsHost, mPixels;
    rt::Tensor mImageTokens, mDeepstack;
    rt::Tensor mPromptHost; //!< pinned: ids, positions (INT32), state, prompt mask (FP16)
    rt::Tensor mInputIds, mPositionIds, mState, mPrefixMask;
    rt::Tensor mHidden, mHiddenOut;
    rt::Tensor mKeys, mValues;
    rt::Tensor mX[2], mNoiseDevice, mSeed, mWeight, mActionMaskDevice;
    rt::Tensor mScalars; //!< step indices (INT32), suffix start (INT32), dt (FP32): 256-byte slots
    rt::Tensor mStageHost, mOutHost;
    std::vector<float> mPrevious;
    bool mUseCudaGraph{true};
    std::map<int64_t, cudaGraphExec_t> mGraphs; //!< per prompt length
    cudaEvent_t mEvents[4]{};
};

} // namespace xr1
} // namespace trt_edgellm
