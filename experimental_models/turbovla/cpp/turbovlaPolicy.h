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

#include "bertWordPiece.h"
#include "common/tensor.h"
#include "vlaEngine.h"

#include <NvInfer.h>
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace turbovla
{

//! One camera frame, tightly packed row-major [height, width, 3] 8-bit RGB, borrowed for the call.
struct TurbovlaView
{
    std::string camera; //!< "primary" or "wrist"
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct TurbovlaChunk
{
    std::vector<float> actions;    //!< [chunk, actionDim] environment actions (arm unnormalized, gripper +-1)
    std::vector<float> normalized; //!< [chunk, actionDim] the decoder's tanh output, after blending
    bool textCached{false};        //!< the instruction's text tokens came from the cache
    float hostMs{0.0F};
    float textMs{0.0F};
    float policyMs{0.0F};
};

//! Chunk blending for real-time control. TurboVLA's decoder regresses a chunk in one pass, so there is no denoising
//! loop to inpaint; the new chunk is instead blended, in normalized space, with the previous chunk's rows from
//! startRow (the rows the robot executes while this chunk is computed): weight 1 on the first frozenSteps rows, then
//! the exponential ramp of pi0.5's real-time chunking down to the new chunk over the rest of overlapSteps.
struct TurbovlaBlend
{
    int32_t overlapSteps{};
    int32_t frozenSteps{};
    float rampRate{6.0F};
    int32_t startRow{-1}; //!< -1: chunk - overlapSteps
};

//! The text graph's inputs for one instruction, as the official encoder builds them.
struct TurbovlaTextInputs
{
    std::vector<int64_t> inputIds;      //!< [L]
    std::vector<int64_t> positionIds;   //!< [L]
    std::vector<uint8_t> selfAttention; //!< [L, L]
    std::vector<uint8_t> hiddenValid;   //!< [L]
    std::vector<uint8_t> attention;     //!< [L]
};

//! TurboVLA (H-EmbodVis/TurboVLA, the LIBERO checkpoint's suite-stats evaluation path) on the text / policy engines
//! written by experimental_models/turbovla/scripts/export_turbovla.py. Two views (primary, wrist) at the model's
//! image size, ImageNet-normalized; frames of another size are resized with PIL's bicubic filter, which the official
//! policy does not do (it requires the model size). The state is normalized with the suite's mean / std, the arm
//! actions unnormalized from [-1, 1] with its min / max and the gripper set to +1 / -1 by sign (+1 at 0).
//!
//! NOT thread-safe: one instance, one stream; every buffer is allocated at construction.
class TurbovlaPolicy
{
public:
    TurbovlaPolicy(std::string const& engineDir, cudaStream_t stream);
    ~TurbovlaPolicy() noexcept;

    TurbovlaPolicy(TurbovlaPolicy const&) = delete;
    TurbovlaPolicy& operator=(TurbovlaPolicy const&) = delete;

    //! \p state is the robot state in the checkpoint's layout (LIBERO: [eef pos, axis-angle, gripper qpos]).
    TurbovlaChunk act(std::vector<TurbovlaView> const& views, std::vector<float> const& state, std::string const& task,
        TurbovlaBlend const* blend = nullptr);

    //! Forget the previous chunk, e.g. at the start of an episode.
    void resetEpisode() noexcept
    {
        mPrevious.clear();
    }

    TurbovlaTextInputs textInputs(std::string const& task) const;

    //! Blend weights for rows [0, overlap): 1 on the frozen rows, then 1 - ramp.
    static std::vector<float> blendWeights(int32_t overlap, int32_t frozen, float rampRate);

    std::vector<std::string> const& cameras() const noexcept
    {
        return mCameras;
    }
    int32_t chunkSize() const noexcept
    {
        return mChunk;
    }
    int32_t actionDim() const noexcept
    {
        return mActionDim;
    }
    int32_t stateDim() const noexcept
    {
        return mStateDim;
    }

private:
    void stageView(int32_t view, unsigned char const* rgb, int32_t height, int32_t width);

    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    vla::TrtEngine mText;
    vla::TrtEngine mPolicy;
    rt::Tensor mContextMemory;
    std::unique_ptr<BertWordPiece> mTokenizer;

    std::vector<std::string> mCameras;
    int32_t mImageSize{256};
    int32_t mTextLength{21};
    int32_t mHidden{256};
    int32_t mChunk{12};
    int32_t mActionDim{7};
    int32_t mStateDim{8};
    std::map<std::string, int32_t> mTextLengthByInstruction;
    std::vector<int64_t> mSplitTokens;
    std::vector<float> mStateMean, mStateStd, mActionMin, mActionMax;
    std::array<std::array<float, 256>, 3> mPixelLut{}; //!< u8 -> (u8 * rescale - mean) / std, per channel

    std::string mLastTask;  //!< the task mText's output in mTextTokens belongs to
    rt::Tensor mPixelsHost; //!< pinned FP16 [views, 3, size, size]
    rt::Tensor mPixels;
    rt::Tensor mTextHost; //!< pinned: ids, positions (INT64), then masks (FP16)
    rt::Tensor mInputIds;
    rt::Tensor mPositionIds;
    rt::Tensor mSelfAttention; //!< FP16 [1, L, L]
    rt::Tensor mHiddenValid;   //!< FP16 [1, L]
    rt::Tensor mAttention;     //!< FP16 [1, L]
    rt::Tensor mTextTokens;    //!< FP16 [1, L, hidden]
    rt::Tensor mState;         //!< FP16 [1, stateDim]
    rt::Tensor mStateHost;
    rt::Tensor mActions; //!< FP16 [1, chunk, actionDim]
    rt::Tensor mActionsHost;
    std::vector<float> mPrevious; //!< last chunk, normalized
    cudaEvent_t mEvents[3]{};
};

} // namespace turbovla
} // namespace trt_edgellm
