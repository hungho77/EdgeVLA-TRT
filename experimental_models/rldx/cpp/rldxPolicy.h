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
#include "rldxPrompt.h"
#include "runtime/imageUtils.h"
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
namespace rt
{
class MultimodalRunner;
}
namespace rldx
{

//! One camera frame at a history offset (0 = now, -2 = two data steps earlier), packed [H, W, 3] 8-bit RGB.
struct RldxFrame
{
    std::string camera;
    int32_t offset{0};
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct RldxChunk
{
    std::vector<float> actions;    //!< [horizon, actionDim] unnormalized: eef position / rotation deltas, gripper_close
    std::vector<float> normalized; //!< [horizon, maxActionDim] the flow output, before unnormalization
    int32_t historyFilled{0};      //!< history frames the request lacked, filled from its oldest frame per camera
    float hostMs{0.0F};
    float visionMs{0.0F};
    float llmMs{0.0F};
    float actionMs{0.0F};
};

//! Real-time chunking as GR00T's: rows [0, overlapSteps) start from the previous chunk's normalized rows from
//! startRow; their velocity is scaled by 0 on the first frozenSteps rows, then by an exponential ramp to 1.
struct RldxRtc
{
    int32_t overlapSteps{};
    int32_t frozenSteps{};
    float rampRate{6.0F};
    int32_t startRow{-1}; //!< -1: horizon - overlapSteps
};

//! RLDX-1 (RLWRLD, the LIBERO checkpoint's server path) on the engines written by
//! experimental_models/rldx/scripts/export_rldx.py and Edge-LLM's Qwen3-VL visual engine: per camera the frames at
//! the checkpoint's history offsets (t-6, t-4, t-2, t), each resized as its AspectAreaResizeAndCrop (area <= 256^2,
//! sides multiples of 32; 256 x 256 is unchanged, 480 x 640 becomes 192 x 256 and 6 x 8 visual tokens); the task
//! formalized (lowercase, word characters and spaces); the state min-max normalized with the q01 / q99 statistics and
//! clipped; the 16-row chunk flowed from noise in 4 Euler steps and unnormalized the same way. The prompt tables are
//! cached per task and image grids.
//!
//! NOT thread-safe: one instance, one stream; every buffer is allocated at construction.
class RldxPolicy
{
public:
    RldxPolicy(std::string const& engineDir, cudaStream_t stream);
    ~RldxPolicy() noexcept;

    RldxPolicy(RldxPolicy const&) = delete;
    RldxPolicy& operator=(RldxPolicy const&) = delete;

    //! \p state in the checkpoint's layout (LIBERO: eef position, axis-angle, the two gripper qpos). \p noise is the
    //! initial [horizon, maxActionDim] sample, drawn from the seeded generator when empty.
    RldxChunk act(std::vector<RldxFrame> const& frames, std::vector<float> const& state, std::string const& task,
        std::vector<float> const& noise = {}, RldxRtc const* rtc = nullptr);

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
    std::vector<int32_t> const& frameHistory() const noexcept
    {
        return mHistory;
    }
    int32_t stateDim() const noexcept
    {
        return static_cast<int32_t>(mStateLow.size());
    }
    int32_t actionDim() const noexcept
    {
        return static_cast<int32_t>(mActionLow.size());
    }
    int32_t chunkSize() const noexcept
    {
        return mHorizon;
    }

    //! AspectAreaResizeAndCrop (OpenCV INTER_AREA resize, centre crop), copied into a new host image.
    rt::imageUtils::ImageData preprocessFrame(unsigned char const* rgb, int32_t height, int32_t width) const;

private:
    RldxPrompt const& prompt(std::string const& task, std::vector<std::pair<int32_t, int32_t>> const& grids);
    void enqueueDenoise();

    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<rt::MultimodalRunner> mVision;
    vla::TrtEngine mLlmA;
    vla::TrtEngine mLlmB;
    vla::TrtEngine mAction;
    rt::Tensor mContextMemory;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;

    std::vector<std::string> mCameras;
    std::vector<int32_t> mHistory;
    int32_t mImageSize{256};
    int32_t mGrid[2]{8, 8};
    int32_t mCognition{64};
    int32_t mHidden{4096};
    int32_t mHorizon{16};
    int32_t mMaxActionDim{64};
    int32_t mMaxStateDim{64};
    int32_t mSteps{4};
    RldxTextGeometry mGeometry;
    std::vector<float> mStateLow, mStateHigh, mActionLow, mActionHigh;

    std::string mLastTask;
    std::vector<std::pair<int32_t, int32_t>> mLastGrids;
    RldxPrompt mPrompt;
    rt::Tensor mPromptHost; //!< pinned staging for the prompt tables
    rt::Tensor mInputIds, mVisualIndex, mCos, mSin, mPool, mKeepIndex, mCosB, mSinB;
    rt::Tensor mDeepstack; //!< [3, visual tokens, hidden] FP16
    rt::Tensor mHiddenA;   //!< [1, S, hidden] FP16
    rt::Tensor mCognitionFeatures;
    rt::Tensor mX[2]; //!< FP16 [1, horizon, maxActionDim] ping-pong
    rt::Tensor mVelocity;
    rt::Tensor mTimes;            //!< FP16 t per step, then dt, one 256-byte slot each
    rt::Tensor mState;            //!< FP16 [1, 1, maxStateDim]
    rt::Tensor mStrength;         //!< FP16 [1, horizon, 1]
    rt::Tensor mStageHost;        //!< pinned FP16: x0, state, strength
    rt::Tensor mOutHost;          //!< pinned FP16 [horizon, maxActionDim]
    std::vector<float> mPrevious; //!< last chunk, normalized [horizon, maxActionDim]
    std::mt19937_64 mNoiseGen{0};
    bool mUseCudaGraph{true};
    cudaGraphExec_t mGraph{nullptr};
    cudaEvent_t mEvents[4]{};
};

} // namespace rldx
} // namespace trt_edgellm
