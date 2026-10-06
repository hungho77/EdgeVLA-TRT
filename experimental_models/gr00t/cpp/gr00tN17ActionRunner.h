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

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace gr00t
{

//! GR00T N1.7 action head on the engines written by export_gr00t_n1_7_action_head.py.
//!
//! One action chunk is prepare() (backbone features -> cached cross-attention K/V), encodeState(), then
//! sample(), which runs the flow-matching Euler steps. Every buffer is allocated at construction.
class Gr00tN17ActionRunner
{
public:
    struct Config
    {
        int32_t actionHorizon{};
        int32_t actionDim{};
        int32_t stateDim{};
        int32_t numInferenceTimesteps{};
        int32_t numTimestepBuckets{};
        int32_t numCrossBlocks{};
        int32_t crossInnerDim{};
        int32_t backboneEmbeddingDim{};
        int32_t maxBackboneTokens{};
    };

    Gr00tN17ActionRunner(std::string const& engineDir, cudaStream_t stream);
    ~Gr00tN17ActionRunner() noexcept;

    Gr00tN17ActionRunner(Gr00tN17ActionRunner const&) = delete;
    Gr00tN17ActionRunner& operator=(Gr00tN17ActionRunner const&) = delete;

    //! Real-time chunking (GR00T RTC): start the new chunk from the tail of the previous one so consecutive
    //! chunks join smoothly while the robot keeps executing.
    struct RtcOptions
    {
        int32_t horizon{};      //!< the embodiment's action horizon (rows of a chunk that are real actions)
        int32_t overlapSteps{}; //!< leading rows seeded from the previous chunk's rows [horizon - overlap, horizon)
        int32_t frozenSteps{};  //!< leading rows kept exactly (policy latency, in control steps)
        float rampRate{6.0F};   //!< exponential ramp of the velocity between frozen and overlap rows
    };

    //! Forget the previous chunk, e.g. at the start of an episode.
    void resetEpisode() noexcept
    {
        mHasPrevious = false;
    }

    //! Replay the denoising loop as a CUDA graph, captured once per backbone token count (default on).
    void setUseCudaGraph(bool enable) noexcept
    {
        mUseCudaGraph = enable;
    }

    Config const& config() const noexcept
    {
        return mConfig;
    }

    //! \p backboneFeatures is [tokens, backboneEmbeddingDim] FP16 or FP32 on the GPU; \p imageMask marks the
    //! image-token positions (one entry per token). Valid until the next prepare().
    void prepare(rt::Tensor const& backboneFeatures, std::vector<uint8_t> const& imageMask, cudaStream_t stream);

    //! \p state is the normalized state, stateDim values.
    void encodeState(std::vector<float> const& state, cudaStream_t stream);

    //! Denoises \p noise ([actionHorizon, actionDim] FP32 on the GPU) into an action chunk of the same shape,
    //! returned in a runner-owned buffer valid until the next sample().
    //! With \p rtc and a previous chunk, the start of the new chunk is inpainted from it (see RtcOptions).
    rt::Tensor const& sample(rt::Tensor const& noise, cudaStream_t stream, RtcOptions const* rtc = nullptr);

private:
    struct Engine
    {
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;
    };

    void loadEngine(std::string const& path, Engine& engine, cudaStream_t stream);
    void bind(Engine& engine, char const* name, void const* address);
    void setShape(Engine& engine, char const* name, std::vector<int64_t> const& shape);
    void enqueueDenoiseLoop(cudaStream_t stream);

    Config mConfig;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    Engine mVlPrep;
    Engine mStateEncoder;
    Engine mDenoise;
    rt::Tensor mContextMemory;

    int64_t mTokens{0};
    rt::Tensor mFeatures;      //!< [1, maxTokens, embed] FP32
    rt::Tensor mImageMask;     //!< [1, maxTokens] bool
    rt::Tensor mAttentionMask; //!< [1, maxTokens] bool, all true
    rt::Tensor mCrossKeys;     //!< [numCross, 1, maxTokens, inner]
    rt::Tensor mCrossValues;
    rt::Tensor mTextBias;  //!< [1, 1, 1, maxTokens]
    rt::Tensor mImageBias; //!< [1, 1, 1, maxTokens]
    rt::Tensor mState;     //!< [1, 1, stateDim]
    rt::Tensor mStateHost; //!< pinned staging for mState
    rt::Tensor mMaskHost;  //!< pinned staging for mImageMask
    rt::Tensor mStateFeatures;
    rt::Tensor mActions[2]; //!< [1, horizon, actionDim] ping-pong
    rt::Tensor mVelStrength;
    rt::Tensor mTimesteps; //!< [numInferenceTimesteps] INT64, one bucket per step
    rt::Tensor mDt;        //!< scalar FP32

    rt::Tensor mPrevious;     //!< last sampled chunk, for RTC
    rt::Tensor mVelocityHost; //!< pinned staging for mVelStrength
    bool mHasPrevious{false};
    bool mVelocityIsOnes{true};

    bool mUseCudaGraph{true};
    std::unordered_map<int64_t, cudaGraphExec_t> mDenoiseGraphs; //!< keyed by backbone token count
};

} // namespace gr00t
} // namespace trt_edgellm
