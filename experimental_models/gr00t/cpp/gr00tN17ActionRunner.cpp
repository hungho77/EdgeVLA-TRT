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

#include "gr00tN17ActionRunner.h"
#include "gr00tKernels.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace gr00t
{
namespace
{

using Json = nlohmann::json;

rt::Tensor makeTensor(
    std::vector<int64_t> const& shape, DataType type, char const* name, rt::DeviceType device = rt::DeviceType::kGPU)
{
    return rt::Tensor(rt::Coords(shape), device, type, name);
}

} // namespace

Gr00tN17ActionRunner::Gr00tN17ActionRunner(std::string const& engineDir, cudaStream_t stream)
{
    std::ifstream configFile(engineDir + "/config.json");
    ELLM_CHECK(configFile.good(), "Gr00tN17ActionRunner: missing " + engineDir + "/config.json");
    Json const json = Json::parse(configFile);
    mConfig.actionHorizon = json.at("action_horizon").get<int32_t>();
    mConfig.actionDim = json.at("action_dim").get<int32_t>();
    mConfig.stateDim = json.at("state_dim").get<int32_t>();
    mConfig.numInferenceTimesteps = json.at("num_inference_timesteps").get<int32_t>();
    mConfig.numTimestepBuckets = json.at("num_timestep_buckets").get<int32_t>();
    mConfig.numCrossBlocks = json.at("num_cross_blocks").get<int32_t>();
    mConfig.crossInnerDim = json.at("cross_inner_dim").get<int32_t>();
    mConfig.backboneEmbeddingDim = json.at("backbone_embedding_dim").get<int32_t>();
    mConfig.maxBackboneTokens = json.at("max_backbone_tokens").get<int32_t>();

    mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    ELLM_CHECK(mRuntime, "Gr00tN17ActionRunner: failed to create TensorRT runtime");
    loadEngine(engineDir + "/vl_prep.engine", mVlPrep, stream);
    loadEngine(engineDir + "/state_encoder.engine", mStateEncoder, stream);
    loadEngine(engineDir + "/denoise_step.engine", mDenoise, stream);

    // The three engines run back to back on one stream, so they can share one scratch allocation.
    int64_t const contextBytes = std::max({mVlPrep.engine->getDeviceMemorySizeV2(),
        mStateEncoder.engine->getDeviceMemorySizeV2(), mDenoise.engine->getDeviceMemorySizeV2()});
    mContextMemory = makeTensor({contextBytes}, DataType::kUINT8, "gr00t::contextMemory");
    for (Engine* e : {&mVlPrep, &mStateEncoder, &mDenoise})
    {
        e->context->setDeviceMemoryV2(mContextMemory.rawPointer(), contextBytes);
    }

    int64_t const maxTokens = mConfig.maxBackboneTokens;
    int64_t const horizon = mConfig.actionHorizon;
    int64_t const actionDim = mConfig.actionDim;
    mFeatures = makeTensor({1, maxTokens, mConfig.backboneEmbeddingDim}, DataType::kFLOAT, "gr00t::features");
    mImageMask = makeTensor({1, maxTokens}, DataType::kBOOL, "gr00t::imageMask");
    mAttentionMask = makeTensor({1, maxTokens}, DataType::kBOOL, "gr00t::attentionMask");
    mMaskHost = makeTensor({1, maxTokens}, DataType::kBOOL, "gr00t::maskHost", rt::DeviceType::kCPU);
    mCrossKeys = makeTensor(
        {mConfig.numCrossBlocks, 1, maxTokens, mConfig.crossInnerDim}, DataType::kFLOAT, "gr00t::crossKeys");
    mCrossValues = makeTensor(
        {mConfig.numCrossBlocks, 1, maxTokens, mConfig.crossInnerDim}, DataType::kFLOAT, "gr00t::crossValues");
    mTextBias = makeTensor({1, 1, 1, maxTokens}, DataType::kFLOAT, "gr00t::textBias");
    mImageBias = makeTensor({1, 1, 1, maxTokens}, DataType::kFLOAT, "gr00t::imageBias");
    mState = makeTensor({1, 1, mConfig.stateDim}, DataType::kFLOAT, "gr00t::state");
    mStateHost = makeTensor({1, 1, mConfig.stateDim}, DataType::kFLOAT, "gr00t::stateHost", rt::DeviceType::kCPU);
    mStateFeatures = makeTensor({1, 1, mConfig.crossInnerDim}, DataType::kFLOAT, "gr00t::stateFeatures");
    mActions[0] = makeTensor({1, horizon, actionDim}, DataType::kFLOAT, "gr00t::actions0");
    mActions[1] = makeTensor({1, horizon, actionDim}, DataType::kFLOAT, "gr00t::actions1");
    mVelStrength = makeTensor({1, horizon, actionDim}, DataType::kFLOAT, "gr00t::velStrength");
    mPrevious = makeTensor({1, horizon, actionDim}, DataType::kFLOAT, "gr00t::previousActions");
    mVelocityHost = makeTensor({1, horizon, actionDim}, DataType::kFLOAT, "gr00t::velocityHost", rt::DeviceType::kCPU);
    mTimesteps = makeTensor({mConfig.numInferenceTimesteps}, DataType::kINT64, "gr00t::timesteps");
    mDt = makeTensor({1}, DataType::kFLOAT, "gr00t::dt");

    // Constants: all-true attention mask, unit velocity strength (no RTC), the timestep buckets
    // int(step / steps * buckets) the reference uses, and dt = 1 / steps.
    auto stage = makeTensor(
        {std::max<int64_t>(maxTokens, horizon * actionDim)}, DataType::kFLOAT, "gr00t::stage", rt::DeviceType::kCPU);
    std::memset(stage.rawPointer(), 1, static_cast<size_t>(maxTokens));
    CUDA_CHECK(
        cudaMemcpyAsync(mAttentionMask.rawPointer(), stage.rawPointer(), maxTokens, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::fill_n(stage.dataPointer<float>(), horizon * actionDim, 1.0F);
    CUDA_CHECK(cudaMemcpyAsync(mVelStrength.rawPointer(), stage.rawPointer(),
        static_cast<size_t>(horizon * actionDim) * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto* buckets = reinterpret_cast<int64_t*>(stage.rawPointer());
    for (int32_t s = 0; s < mConfig.numInferenceTimesteps; ++s)
    {
        buckets[s]
            = static_cast<int64_t>(static_cast<double>(s) / mConfig.numInferenceTimesteps * mConfig.numTimestepBuckets);
    }
    CUDA_CHECK(cudaMemcpyAsync(mTimesteps.rawPointer(), stage.rawPointer(),
        static_cast<size_t>(mConfig.numInferenceTimesteps) * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stage.dataPointer<float>()[0] = 1.0F / static_cast<float>(mConfig.numInferenceTimesteps);
    CUDA_CHECK(cudaMemcpyAsync(mDt.rawPointer(), stage.rawPointer(), sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Gr00tN17ActionRunner::loadEngine(std::string const& path, Engine& engine, cudaStream_t stream)
{
    engine.engine = deserializeCudaEngineFromFile(*mRuntime, path);
    ELLM_CHECK(engine.engine, "Gr00tN17ActionRunner: failed to load " + path);
    engine.context = std::unique_ptr<IExecutionContext>(
        engine.engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(engine.context, "Gr00tN17ActionRunner: failed to create a context for " + path);
    ELLM_CHECK(engine.context->setOptimizationProfileAsync(0, stream),
        "Gr00tN17ActionRunner: failed to set the optimization profile for " + path);
}

void Gr00tN17ActionRunner::bind(Engine& engine, char const* name, void const* address)
{
    ELLM_CHECK(engine.context->setTensorAddress(name, const_cast<void*>(address)),
        std::string("Gr00tN17ActionRunner: failed to bind ") + name);
}

void Gr00tN17ActionRunner::setShape(Engine& engine, char const* name, std::vector<int64_t> const& shape)
{
    Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    std::copy(shape.begin(), shape.end(), dims.d);
    ELLM_CHECK(engine.context->setInputShape(name, dims),
        std::string("Gr00tN17ActionRunner: failed to set the shape of ") + name);
}

void Gr00tN17ActionRunner::prepare(
    rt::Tensor const& backboneFeatures, std::vector<uint8_t> const& imageMask, cudaStream_t stream)
{
    int64_t const embed = mConfig.backboneEmbeddingDim;
    int64_t const tokens = backboneFeatures.getShape().volume() / embed;
    ELLM_CHECK(tokens > 0 && tokens <= mConfig.maxBackboneTokens,
        "Gr00tN17ActionRunner::prepare: backbone token count is outside the engine's range");
    ELLM_CHECK(static_cast<int64_t>(imageMask.size()) == tokens,
        "Gr00tN17ActionRunner::prepare: image mask length differs from the backbone token count");
    mTokens = tokens;

    if (backboneFeatures.getDataType() == DataType::kHALF)
    {
        launchHalfToFloat(backboneFeatures.rawPointer(), mFeatures.dataPointer<float>(), tokens * embed, stream);
    }
    else
    {
        CUDA_CHECK(cudaMemcpyAsync(mFeatures.rawPointer(), backboneFeatures.rawPointer(),
            static_cast<size_t>(tokens * embed) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
    std::copy(imageMask.begin(), imageMask.end(), static_cast<uint8_t*>(mMaskHost.rawPointer()));
    CUDA_CHECK(cudaMemcpyAsync(
        mImageMask.rawPointer(), mMaskHost.rawPointer(), static_cast<size_t>(tokens), cudaMemcpyHostToDevice, stream));

    setShape(mVlPrep, "backbone_features", {1, tokens, embed});
    setShape(mVlPrep, "image_mask", {1, tokens});
    setShape(mVlPrep, "attention_mask", {1, tokens});
    bind(mVlPrep, "backbone_features", mFeatures.rawPointer());
    bind(mVlPrep, "image_mask", mImageMask.rawPointer());
    bind(mVlPrep, "attention_mask", mAttentionMask.rawPointer());
    bind(mVlPrep, "cross_keys", mCrossKeys.rawPointer());
    bind(mVlPrep, "cross_values", mCrossValues.rawPointer());
    bind(mVlPrep, "text_bias", mTextBias.rawPointer());
    bind(mVlPrep, "image_bias", mImageBias.rawPointer());
    ELLM_CHECK(mVlPrep.context->enqueueV3(stream), "Gr00tN17ActionRunner: vl_prep enqueue failed");
}

void Gr00tN17ActionRunner::encodeState(std::vector<float> const& state, cudaStream_t stream)
{
    ELLM_CHECK(static_cast<int32_t>(state.size()) == mConfig.stateDim,
        "Gr00tN17ActionRunner::encodeState: state has the wrong width");
    std::copy(state.begin(), state.end(), mStateHost.dataPointer<float>());
    CUDA_CHECK(cudaMemcpyAsync(
        mState.rawPointer(), mStateHost.rawPointer(), state.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    bind(mStateEncoder, "state", mState.rawPointer());
    bind(mStateEncoder, "state_features", mStateFeatures.rawPointer());
    ELLM_CHECK(mStateEncoder.context->enqueueV3(stream), "Gr00tN17ActionRunner: state_encoder enqueue failed");
}

Gr00tN17ActionRunner::~Gr00tN17ActionRunner() noexcept
{
    for (auto& [tokens, graph] : mDenoiseGraphs)
    {
        cudaGraphExecDestroy(graph);
    }
}

void Gr00tN17ActionRunner::enqueueDenoiseLoop(cudaStream_t stream)
{
    int64_t const inner = mConfig.crossInnerDim;
    setShape(mDenoise, "cross_keys", {mConfig.numCrossBlocks, 1, mTokens, inner});
    setShape(mDenoise, "cross_values", {mConfig.numCrossBlocks, 1, mTokens, inner});
    setShape(mDenoise, "text_bias", {1, 1, 1, mTokens});
    setShape(mDenoise, "image_bias", {1, 1, 1, mTokens});
    bind(mDenoise, "state_features", mStateFeatures.rawPointer());
    bind(mDenoise, "cross_keys", mCrossKeys.rawPointer());
    bind(mDenoise, "cross_values", mCrossValues.rawPointer());
    bind(mDenoise, "text_bias", mTextBias.rawPointer());
    bind(mDenoise, "image_bias", mImageBias.rawPointer());
    bind(mDenoise, "vel_strength", mVelStrength.rawPointer());
    bind(mDenoise, "dt", mDt.rawPointer());
    int32_t current = 0;
    for (int32_t step = 0; step < mConfig.numInferenceTimesteps; ++step)
    {
        bind(mDenoise, "actions", mActions[current].rawPointer());
        bind(mDenoise, "timestep", mTimesteps.dataPointer<int64_t>() + step);
        bind(mDenoise, "next_actions", mActions[1 - current].rawPointer());
        ELLM_CHECK(mDenoise.context->enqueueV3(stream), "Gr00tN17ActionRunner: denoise_step enqueue failed");
        current = 1 - current;
    }
}

rt::Tensor const& Gr00tN17ActionRunner::sample(rt::Tensor const& noise, cudaStream_t stream, RtcOptions const* rtc)
{
    ELLM_CHECK(mTokens > 0, "Gr00tN17ActionRunner::sample: call prepare() first");
    int64_t const actionDim = mConfig.actionDim;
    size_t const rowBytes = static_cast<size_t>(actionDim) * sizeof(float);
    size_t const actionBytes = static_cast<size_t>(mConfig.actionHorizon) * rowBytes;
    // Initial actions and the velocity mask are set outside the graph, so callers may pass any noise buffer and
    // switch RTC on or off between calls.
    CUDA_CHECK(
        cudaMemcpyAsync(mActions[0].rawPointer(), noise.rawPointer(), actionBytes, cudaMemcpyDeviceToDevice, stream));
    if (rtc != nullptr && mHasPrevious && rtc->overlapSteps > 0)
    {
        ELLM_CHECK(rtc->horizon <= mConfig.actionHorizon && rtc->overlapSteps <= rtc->horizon && rtc->frozenSteps >= 0
                && rtc->frozenSteps <= rtc->overlapSteps,
            "Gr00tN17ActionRunner::sample: inconsistent RTC options");
        CUDA_CHECK(cudaMemcpyAsync(mActions[0].rawPointer(),
            mPrevious.dataPointer<float>() + static_cast<int64_t>(rtc->horizon - rtc->overlapSteps) * actionDim,
            static_cast<size_t>(rtc->overlapSteps) * rowBytes, cudaMemcpyDeviceToDevice, stream));
        // GR00T: ramp = 1 - exp(-rate * linspace(0, 1, n + 2)), normalized by its last value, interior n points.
        int32_t const ramped = rtc->overlapSteps - rtc->frozenSteps;
        double const last = std::max(1.0 - std::exp(-static_cast<double>(rtc->rampRate)), 1e-8);
        CUDA_CHECK(cudaStreamSynchronize(stream)); // the staging buffer may still feed the previous upload
        float* velocity = mVelocityHost.dataPointer<float>();
        for (int32_t row = 0; row < mConfig.actionHorizon; ++row)
        {
            float value = 1.0F;
            if (row < rtc->frozenSteps)
            {
                value = 0.0F;
            }
            else if (row < rtc->overlapSteps)
            {
                double const t = static_cast<double>(row - rtc->frozenSteps + 1) / (ramped + 1);
                value = static_cast<float>((1.0 - std::exp(-rtc->rampRate * t)) / last);
            }
            std::fill_n(velocity + static_cast<int64_t>(row) * actionDim, actionDim, value);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            mVelStrength.rawPointer(), mVelocityHost.rawPointer(), actionBytes, cudaMemcpyHostToDevice, stream));
        mVelocityIsOnes = false;
    }
    else if (!mVelocityIsOnes)
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::fill_n(mVelocityHost.dataPointer<float>(), mConfig.actionHorizon * actionDim, 1.0F);
        CUDA_CHECK(cudaMemcpyAsync(
            mVelStrength.rawPointer(), mVelocityHost.rawPointer(), actionBytes, cudaMemcpyHostToDevice, stream));
        mVelocityIsOnes = true;
    }

    auto const cached = mDenoiseGraphs.find(mTokens);
    if (mUseCudaGraph && cached != mDenoiseGraphs.end())
    {
        CUDA_CHECK(cudaGraphLaunch(cached->second, stream));
    }
    else
    {
        // TensorRT needs one regular enqueue for these shapes before its kernels can be captured.
        enqueueDenoiseLoop(stream);
        if (mUseCudaGraph)
        {
            cudaGraph_t graph{};
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            enqueueDenoiseLoop(stream);
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            cudaGraphExec_t exec{};
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            mDenoiseGraphs.emplace(mTokens, exec);
        }
    }
    rt::Tensor const& result = mActions[mConfig.numInferenceTimesteps % 2];
    CUDA_CHECK(
        cudaMemcpyAsync(mPrevious.rawPointer(), result.rawPointer(), actionBytes, cudaMemcpyDeviceToDevice, stream));
    mHasPrevious = true;
    return result;
}

} // namespace gr00t
} // namespace trt_edgellm
