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
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace xvla
{

//! One camera frame, tightly packed row-major [height, width, 3] 8-bit RGB, borrowed for the call.
struct XvlaView
{
    std::string camera; //!< the checkpoint's camera name (config.json "cameras"), e.g. "image"
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct XvlaChunk
{
    std::vector<float> actions; //!< [chunk, actionDim] robot actions: grippers through the sigmoid, unnormalized
    std::vector<int64_t> tokenIds;
    float visionMs{0.0F};
    float encoderMs{0.0F};
    float denoiseMs{0.0F};
};

//! Real-time chunking. The new chunk continues the previous one from row startRow (the robot executed startRow of its
//! actions when this one starts): its first inferenceDelay rows reproduce the previous chunk's remaining rows and the
//! weight then falls linearly to 0 at executionHorizon (LeRobot's LINEAR prefix weights). X-VLA predicts the clean
//! action at every step, so each step's prediction is pulled toward the previous rows by those weights. Its ee6d
//! actions are absolute end-effector targets, so the previous rows are reused as they are.
struct XvlaRtc
{
    int32_t inferenceDelay{};
    int32_t executionHorizon{};
    int32_t startRow{-1}; //!< -1: chunk - executionHorizon
};

//! X-VLA (LeRobot 0.6.1) on the vision / encoder / step engines written by
//! experimental_models/xvla/scripts/export_xvla.py, with LeRobot's processing: ImageNet normalization and
//! resize_with_pad (bilinear, zeros on top and left in normalized space), BART tokens padded to the saved
//! length, the state zero-padded to the model's proprio width, x1 -> action over num_steps (t = 1 .. 1/steps),
//! and the ee6d action space (gripper channels zeroed in the inputs, sigmoid on the output). Cameras without a
//! frame are masked out (zero features), as LeRobot pads missing views.
//!
//! NOT thread-safe: one instance, one stream; every buffer is allocated at construction.
class XvlaPolicy
{
public:
    XvlaPolicy(std::string const& engineDir, cudaStream_t stream);
    ~XvlaPolicy() noexcept;

    XvlaPolicy(XvlaPolicy const&) = delete;
    XvlaPolicy& operator=(XvlaPolicy const&) = delete;

    //! \p state is zero-padded or truncated to the proprio width, as LeRobot's pad_vector; the dataset's stateDim
    //! values, or what an environment processor builds (LIBERO: [eef pos, rot6d, 0] padded to 20). \p noise is x1,
    //! [chunk, actionDim], drawn from the seeded generator when empty.
    XvlaChunk act(std::vector<XvlaView> const& views, std::vector<float> const& state, std::string const& task,
        std::vector<float> const& noise = {}, XvlaRtc const* rtc = nullptr);

    //! Forget the previous chunk, e.g. at the start of an episode.
    void resetEpisode() noexcept
    {
        mPrevious.clear();
    }

    //! The embodiment's domain (soft prompts and domain-specific projections); defaults to the checkpoint's
    //! preprocessor domain.
    void setDomainId(int32_t domainId);
    int32_t domainId() const noexcept
    {
        return mDomainId;
    }

    //! LeRobot's get_prefix_weights (LINEAR): 1 on [0, delay), linspace(1, 0) inside [delay, horizon), 0 after.
    static std::vector<float> prefixWeights(int32_t delay, int32_t horizon, int32_t total);

    void setNoiseSeed(uint64_t seed) noexcept
    {
        mNoiseGen.seed(seed);
    }
    //! Replay the denoising loop as a CUDA graph, captured on the first call (default on).
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
    //! The checkpoint's camera names, in the order LeRobot feeds them.
    std::vector<std::string> const& cameras() const noexcept
    {
        return mCameras;
    }
    int32_t proprioDim() const noexcept
    {
        return mProprioDim;
    }

    //! LeRobot's ImageNet normalization then resize_with_pad, as planar [3, size, size].
    std::vector<float> preprocessView(unsigned char const* rgb, int32_t height, int32_t width) const;
    std::vector<int64_t> tokenize(std::string const& task) const;

private:
    void enqueueDenoiseLoop();

    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    vla::TrtEngine mVision;
    vla::TrtEngine mEncoder;
    vla::TrtEngine mStep;
    rt::Tensor mContextMemory;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;

    std::vector<std::string> mCameras;
    int32_t mNumViews{0};
    std::string mLastTask; //!< the task mLastTokens was tokenized from
    std::vector<int64_t> mLastTokens;
    int32_t mImageSize{224};
    int32_t mImageTokens{50};
    int32_t mMaxTokens{50};
    int64_t mPadTokenId{1};
    int32_t mChunk{30};
    int32_t mActionDim{20};
    int32_t mProprioDim{20};
    int32_t mNumSteps{10};
    int32_t mHidden{1024};
    std::vector<int32_t> mGripper;
    int32_t mDomainId{0};
    int32_t mNumDomains{1};
    std::vector<float> mActionMean, mActionStd; //!< empty: the checkpoint has no action statistics (identity)

    rt::Tensor mPixelsHost; //!< pinned FP16 [views, 3, size, size]
    rt::Tensor mPixels;
    rt::Tensor mFeatures; //!< [views, imageTokens, hidden] FP16; view 0 feeds the encoder, the rest are aux
    rt::Tensor mTokensHost;
    rt::Tensor mTokens;
    rt::Tensor mVlm;              //!< [1, imageTokens + maxTokens, hidden] FP16
    rt::Tensor mX1;               //!< [1, chunk, actionDim] FP16
    rt::Tensor mAction[2];        //!< ping-pong
    rt::Tensor mTimes;            //!< FP16 [numSteps]
    rt::Tensor mProprio;          //!< FP16 [1, proprioDim]
    rt::Tensor mStageHost;        //!< pinned FP16 staging for x1 and the state
    rt::Tensor mOutHost;          //!< pinned FP16 [chunk, actionDim]
    rt::Tensor mDomain;           //!< INT64 [1]
    rt::Tensor mRtcSeed;          //!< FP16 [1, chunk, actionDim], model space
    rt::Tensor mRtcWeight;        //!< FP16 [1, chunk, 1]
    rt::Tensor mRtcHost;          //!< pinned staging: domain id (INT64), then seed and weights (FP16)
    std::vector<float> mPrevious; //!< last chunk in model space (gripper logits, before unnormalization)
    std::mt19937_64 mNoiseGen{0};
    bool mUseCudaGraph{true};
    cudaGraphExec_t mGraph{nullptr};
    cudaEvent_t mEvents[4]{};
};

} // namespace xvla
} // namespace trt_edgellm
