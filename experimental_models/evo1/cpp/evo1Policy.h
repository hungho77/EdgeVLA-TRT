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

#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace evo1
{

//! One camera frame, packed [H, W, 3] 8-bit RGB, borrowed for the call.
struct Evo1View
{
    std::string camera; //!< one of the engine directory's cameras ("image_1", "image_2", ...)
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct Evo1Chunk
{
    std::vector<float> actions;    //!< [horizon, actionDim] unnormalized
    std::vector<float> normalized; //!< [horizon, maxDim] the flow output, padded dims zero
    int32_t promptTokens{0};       //!< tokens before the padding
    float hostMs{0.0F};            //!< frames, then the prompt and noise (the latter overlap the vision engine)
    float visionMs{0.0F};
    float prefixMs{0.0F};
    float actionMs{0.0F}; //!< the head's keys / values and the Euler steps
};

//! Real-time chunking as GR00T's: rows [0, overlapSteps) start from the previous chunk's normalized rows from
//! startRow; their velocity is scaled by 0 on the first frozenSteps rows, then by an exponential ramp to 1.
struct Evo1Rtc
{
    int32_t overlapSteps{};
    int32_t frozenSteps{};
    float rampRate{6.0F};
    int32_t startRow{-1}; //!< -1: horizon - overlapSteps
};

//! Evo-1 (MINT-SJTU/Evo-1, its Evo1_server.py) on the engines written by
//! experimental_models/evo1/scripts/export_evo1.py. Each frame is resized to 448 x 448 with OpenCV's bilinear filter,
//! its red and blue channels swapped as the official server's cv2.COLOR_BGR2RGB does to the RGB frames its clients
//! send (swap_red_blue), and ImageNet-normalized. The views missing up to max_views are black images whose tokens the
//! language model's attention skips; their vision tokens are computed once. The prompt is the image blocks followed by
//! the instruction, right-padded to the full text length: the action head cross-attends to every position, padding
//! included. The state is normalized to [-1, 1] by the checkpoint's bounds and zero-padded; the chunk integrates 50
//! Euler steps from uniform noise and its first action dims are unnormalized.
//!
//! NOT thread-safe: one instance, one stream; every buffer is allocated at construction.
class Evo1Policy
{
public:
    Evo1Policy(std::string const& engineDir, cudaStream_t stream);
    ~Evo1Policy() noexcept;

    Evo1Policy(Evo1Policy const&) = delete;
    Evo1Policy& operator=(Evo1Policy const&) = delete;

    //! \p noise is the initial [horizon, maxDim] sample in [-1, 1], drawn from the seeded generator when empty.
    Evo1Chunk act(std::vector<Evo1View> const& views, std::vector<float> const& state, std::string const& task,
        std::vector<float> const& noise = {}, Evo1Rtc const* rtc = nullptr);

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
        return static_cast<int32_t>(mStateMin.size());
    }
    int32_t actionDim() const noexcept
    {
        return static_cast<int32_t>(mActionMin.size());
    }
    int32_t chunkSize() const noexcept
    {
        return mHorizon;
    }

    //! The prompt's token ids before padding: the image blocks, then the stripped instruction.
    std::vector<int32_t> promptIds(std::string const& task) const;

private:
    void stageFrame(int32_t view, unsigned char const* rgb, int32_t height, int32_t width);
    void enqueueDenoise();

    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    vla::TrtEngine mVision, mPrefix, mContext, mStep;
    rt::Tensor mContextMemory;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;

    std::vector<std::string> mCameras;
    int32_t mMaxViews{3}, mImageSize{448}, mImageTokens{256}, mTextLength{1024};
    int32_t mHidden{896}, mBlocks{8}, mHorizon{50}, mMaxDim{24}, mSteps{50};
    int32_t mImageContextId{151667}, mPadId{151643};
    bool mSwapRedBlue{true};
    std::vector<int32_t> mPrefixIds;
    std::vector<int32_t> mTimeIndices;
    std::vector<float> mStateMin, mStateMax, mActionMin, mActionMax;
    std::array<std::array<float, 256>, 3> mPixelLut{}; //!< input u8 -> normalized value, per output channel

    rt::Tensor mPixelsHost; //!< pinned FP16 [cameras, 3, size, size]
    rt::Tensor mPixels;
    rt::Tensor mVisual;     //!< FP16 [maxViews * imageTokens, hidden]; the black views' rows filled once
    rt::Tensor mPromptHost; //!< pinned: ids (INT32), then the attention mask (FP16)
    rt::Tensor mInputIds, mAttention;
    rt::Tensor mHiddenState;
    rt::Tensor mStateHost, mState; //!< FP16 [1, maxDim]
    rt::Tensor mKv;                //!< FP16 [blocks][2, 1, heads, context, headDim]
    rt::Tensor mX[2];              //!< FP32 [1, horizon * maxDim]
    rt::Tensor mScalars;           //!< per-step time index (INT32), then dt (FP32), one 256-byte slot each
    rt::Tensor mActionMask, mStrength;
    rt::Tensor mStageHost; //!< pinned: x0, strengths
    rt::Tensor mOutHost;
    std::vector<float> mPrevious;
    std::mt19937_64 mNoiseGen{0};
    bool mUseCudaGraph{true};
    cudaGraphExec_t mGraph{nullptr};
    cudaEvent_t mEvents[4]{};
};

} // namespace evo1
} // namespace trt_edgellm
