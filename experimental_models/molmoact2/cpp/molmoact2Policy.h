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
namespace molmoact2
{

//! One camera frame, packed [H, W, 3] 8-bit RGB, borrowed for the call.
struct MolmoAct2View
{
    std::string camera; //!< the checkpoint's camera key, e.g. "image" / "wrist_image"
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct MolmoAct2Chunk
{
    std::vector<float> actions;    //!< [horizon, actionDim] unnormalized (masked dims raw, clamped to [-1, 1])
    std::vector<float> normalized; //!< [horizon, maxActionDim] the flow output
    int32_t promptTokens{0};
    float hostMs{0.0F}; //!< frame patches, then the prompt and noise (the latter overlap the vision engine)
    float visionMs{0.0F};
    float prefixMs{0.0F};
    float actionMs{0.0F};
};

//! Real-time chunking as GR00T's: rows [0, overlapSteps) start from the previous chunk's normalized rows from
//! startRow; their velocity is scaled by 0 on the first frozenSteps rows, then by an exponential ramp to 1.
struct MolmoAct2Rtc
{
    int32_t overlapSteps{};
    int32_t frozenSteps{};
    float rampRate{6.0F};
    int32_t startRow{-1}; //!< -1: horizon - overlapSteps
};

//! MolmoAct2 (AllenAI, LeRobot's continuous-inference path) on the engines written by
//! experimental_models/molmoact2/scripts/export_molmoact2.py: each camera's frame resized to 378 x 378 and cut into
//! SigLIP2 patches as the official processor does (bit-identical); the state normalized with the masked q01 / q99
//! statistics, clipped and binned into 256 state tokens in the robot_action prompt; image tokens attending to each
//! other, text causally; the 36 layers' K / V feeding the flow expert, 10 Euler steps from noise; the chunk clamped
//! to [-1, 1] and unnormalized on the masked dims.
//!
//! NOT thread-safe: one instance, one stream; buffers are allocated at construction for the engines' largest prompt.
class MolmoAct2Policy
{
public:
    MolmoAct2Policy(std::string const& engineDir, cudaStream_t stream);
    ~MolmoAct2Policy() noexcept;

    MolmoAct2Policy(MolmoAct2Policy const&) = delete;
    MolmoAct2Policy& operator=(MolmoAct2Policy const&) = delete;

    //! \p noise is the initial [horizon, maxActionDim] sample, drawn from the seeded generator when empty.
    MolmoAct2Chunk act(std::vector<MolmoAct2View> const& views, std::vector<float> const& state,
        std::string const& task, std::vector<float> const& noise = {}, MolmoAct2Rtc const* rtc = nullptr);

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
        return static_cast<int32_t>(mStateLow.size());
    }
    int32_t actionDim() const noexcept
    {
        return mActionDim;
    }
    int32_t chunkSize() const noexcept
    {
        return mHorizon;
    }

    //! The token ids of the prompt for \p task and the normalized \p state, with the leading BOS.
    std::vector<int32_t> promptIds(std::string const& task, std::vector<float> const& normalizedState) const;
    std::vector<float> normalizeState(std::vector<float> const& state) const;

private:
    void enqueueDenoise(int64_t sequence);

    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    vla::TrtEngine mVision, mPrefixA, mPrefixB, mContext, mStep;
    rt::Tensor mContextMemory;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;

    std::vector<std::string> mCameras;
    std::string mSetup, mControlMode;
    int32_t mImageTokenCount{196}, mImagePatchId{154626}, mImageStartId{154624};
    int32_t mImageSize{378}, mPatch{14}, mStateBins{256}, mHorizon{10}, mActionDim{7}, mMaxActionDim{32};
    int32_t mSteps{10}, mHeadDim{128}, mLayers{36}, mLayersA{18}, mKvDim{1024}, mHidden{2560};
    int32_t mContextHeads{8}, mContextHeadDim{96};
    float mRopeTheta{5000000.0F};
    int32_t mBos{151645};
    std::vector<float> mStateLow, mStateHigh, mStateMask, mActionLow, mActionHigh, mActionMask;

    rt::Tensor mPatchesHost; //!< pinned FP32 [cameras, patches, patch pixels]
    rt::Tensor mPatches;
    rt::Tensor mPromptHost; //!< pinned: ids (INT64), then the image flag and encoder mask (FP16)
    rt::Tensor mInputIds, mImageFlag, mEncoderMask;
    rt::Tensor mCos, mSin; //!< FP32 [max S, headDim], filled once
    rt::Tensor mVisual, mHiddenState;
    rt::Tensor mKeys, mValues;       //!< FP16 [layers, S, kvDim]
    rt::Tensor mContextK, mContextV; //!< FP16 [layers, 1, S, heads, headDim]
    rt::Tensor mX[2], mVelocity;
    rt::Tensor mScalars; //!< step index (INT64) per step and dt (FP16), one 256-byte slot each
    rt::Tensor mStrength;
    rt::Tensor mStageHost, mOutHost;
    std::vector<float> mPrevious;
    std::mt19937_64 mNoiseGen{0};
    bool mUseCudaGraph{true};
    std::map<int64_t, cudaGraphExec_t> mGraphs; //!< per prompt length
    cudaEvent_t mEvents[4]{};
};

} // namespace molmoact2
} // namespace trt_edgellm
