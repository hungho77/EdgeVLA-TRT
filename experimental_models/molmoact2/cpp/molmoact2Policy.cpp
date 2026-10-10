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

#include "molmoact2Policy.h"

#include "common/checkMacros.h"
#include "molmoact2Image.h"
#include "molmoact2Text.h"

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>

namespace trt_edgellm
{
namespace molmoact2
{
namespace
{

using Json = nlohmann::json;
using nvinfer1::DataType;

constexpr int64_t kSlotBytes = 256;

rt::Tensor makeTensor(
    std::vector<int64_t> const& shape, DataType type, char const* name, rt::DeviceType device = rt::DeviceType::kGPU)
{
    return rt::Tensor(rt::Coords(shape), device, type, name);
}

void readRange(Json const& group, std::vector<float>& low, std::vector<float>& high, std::vector<float>& mask)
{
    low = group.at("q01").get<std::vector<float>>();
    high = group.at("q99").get<std::vector<float>>();
    mask = group.at("mask").get<std::vector<float>>();
}

//! LeRobot's QUANTILES denominator: q99 - q01, or its eps (1e-8) when they are equal.
float span(float low, float high)
{
    float const d = high - low;
    return d == 0.0F ? 1e-8F : d;
}

bool isImageToken(int32_t id)
{
    return id >= 154624 && id <= 154628;
}

} // namespace

MolmoAct2Policy::MolmoAct2Policy(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    std::ifstream file(engineDir + "/config.json");
    ELLM_CHECK(file.good(), "MolmoAct2Policy: missing " + engineDir + "/config.json");
    Json const config = Json::parse(file);
    ELLM_CHECK(config.at("model_family").get<std::string>() == "molmoact2", "MolmoAct2Policy: not a MolmoAct2 dir");
    mCameras = config.at("cameras").get<std::vector<std::string>>();
    mImageSize = config.at("image_size").get<int32_t>();
    mPatch = config.at("patch_size").get<int32_t>();
    mSetup = config.at("setup").get<std::string>();
    mControlMode = config.at("control_mode").get<std::string>();
    mStateBins = config.at("state_bins").get<int32_t>();
    mHorizon = config.at("action_horizon").get<int32_t>();
    mActionDim = config.at("action_dim").get<int32_t>();
    mMaxActionDim = config.at("max_action_dim").get<int32_t>();
    mSteps = config.at("flow_steps").get<int32_t>();
    mHeadDim = config.at("head_dim").get<int32_t>();
    mRopeTheta = config.at("rope_theta").get<float>();
    mBos = config.at("bos_token_id").get<int32_t>();
    readRange(config.at("state_normalization"), mStateLow, mStateHigh, mStateMask);
    readRange(config.at("action_normalization"), mActionLow, mActionHigh, mActionMask);
    int32_t const imageTokens = config.at("image_tokens").get<int32_t>();
    mImageTokens = "<im_start>";
    for (int32_t i = 0; i < imageTokens; ++i)
    {
        mImageTokens += "<im_patch>";
    }
    mImageTokens += "<im_end>";

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir), "MolmoAct2Policy: failed to load the tokenizer");
    mRuntime = vla::createTrtRuntime();
    mVision = vla::TrtEngine(*mRuntime, engineDir + "/vision.engine", stream);
    mPrefixA = vla::TrtEngine(*mRuntime, engineDir + "/prefix_a.engine", stream);
    mPrefixB = vla::TrtEngine(*mRuntime, engineDir + "/prefix_b.engine", stream);
    mContext = vla::TrtEngine(*mRuntime, engineDir + "/context.engine", stream);
    mStep = vla::TrtEngine(*mRuntime, engineDir + "/step.engine", stream);
    mContextMemory = vla::allocateSharedContextMemory(
        {&mVision, &mPrefixA, &mPrefixB, &mContext, &mStep}, "molmoact2::contextMemory");
    for (vla::TrtEngine* engine : {&mVision, &mPrefixA, &mPrefixB, &mContext, &mStep})
    {
        engine->setDeviceMemory(mContextMemory);
    }

    auto const dims = [](vla::TrtEngine const& e, char const* name) { return e.engine().getTensorShape(name); };
    mLayersA = static_cast<int32_t>(dims(mPrefixA, "keys").d[0]);
    mLayers = mLayersA + static_cast<int32_t>(dims(mPrefixB, "keys").d[0]);
    mKvDim = static_cast<int32_t>(dims(mPrefixA, "keys").d[2]);
    mHidden = static_cast<int32_t>(dims(mPrefixA, "hidden").d[2]);
    mContextHeads = static_cast<int32_t>(dims(mContext, "context_k").d[3]);
    mContextHeadDim = static_cast<int32_t>(dims(mContext, "context_k").d[4]);
    int64_t const visualRows = dims(mVision, "visual").d[0];
    int64_t const maxS = mPrefixA.engine().getProfileShape("input_ids", 0, nvinfer1::OptProfileSelector::kMAX).d[0];
    int64_t const cameras = static_cast<int64_t>(mCameras.size());
    int64_t const patches = static_cast<int64_t>(mImageSize / mPatch) * (mImageSize / mPatch);
    int64_t const patchPixels = static_cast<int64_t>(mPatch) * mPatch * 3;

    mPatchesHost
        = makeTensor({cameras, patches, patchPixels}, DataType::kFLOAT, "molmoact2::patchesHost", rt::DeviceType::kCPU);
    mPatches = makeTensor({cameras, patches, patchPixels}, DataType::kFLOAT, "molmoact2::patches");
    mPromptHost = makeTensor(
        {maxS * (8 + 2 + 2 + 8 * mHeadDim)}, DataType::kUINT8, "molmoact2::promptHost", rt::DeviceType::kCPU);
    mInputIds = makeTensor({maxS}, DataType::kINT64, "molmoact2::inputIds");
    mImageFlag = makeTensor({maxS}, DataType::kHALF, "molmoact2::imageFlag");
    mEncoderMask = makeTensor({1, maxS}, DataType::kHALF, "molmoact2::encoderMask");
    mCos = makeTensor({maxS, mHeadDim}, DataType::kFLOAT, "molmoact2::cos");
    mSin = makeTensor({maxS, mHeadDim}, DataType::kFLOAT, "molmoact2::sin");
    mVisual = makeTensor({visualRows, mHidden}, DataType::kHALF, "molmoact2::visual");

    mHiddenState = makeTensor({1, maxS, mHidden}, DataType::kHALF, "molmoact2::hidden");
    mKeys = makeTensor({mLayers, maxS, mKvDim}, DataType::kHALF, "molmoact2::keys");
    mValues = makeTensor({mLayers, maxS, mKvDim}, DataType::kHALF, "molmoact2::values");
    mContextK = makeTensor({mLayers, 1, maxS, mContextHeads, mContextHeadDim}, DataType::kHALF, "molmoact2::ctxK");
    mContextV = makeTensor({mLayers, 1, maxS, mContextHeads, mContextHeadDim}, DataType::kHALF, "molmoact2::ctxV");
    int64_t const chunkElems = static_cast<int64_t>(mHorizon) * mMaxActionDim;
    mX[0] = makeTensor({1, mHorizon, mMaxActionDim}, DataType::kHALF, "molmoact2::x0");
    mX[1] = makeTensor({1, mHorizon, mMaxActionDim}, DataType::kHALF, "molmoact2::x1");
    mVelocity = makeTensor({1, mHorizon, mMaxActionDim}, DataType::kHALF, "molmoact2::velocity");
    mScalars = makeTensor({(mSteps + 1) * kSlotBytes}, DataType::kUINT8, "molmoact2::scalars");
    mStrength = makeTensor({1, mHorizon, 1}, DataType::kHALF, "molmoact2::strength");
    mStageHost = makeTensor({(mSteps + 1) * kSlotBytes + (chunkElems + mHorizon) * 2}, DataType::kUINT8,
        "molmoact2::stage", rt::DeviceType::kCPU);
    mOutHost = makeTensor({chunkElems}, DataType::kHALF, "molmoact2::out", rt::DeviceType::kCPU);

    // Step indices (INT64) and dt (FP16), each in its own 256-byte slot: TensorRT wants aligned binding addresses.
    auto* slots = static_cast<char*>(mStageHost.rawPointer());
    std::memset(slots, 0, (mSteps + 1) * kSlotBytes);
    for (int32_t k = 0; k < mSteps; ++k)
    {
        *reinterpret_cast<int64_t*>(slots + k * kSlotBytes) = k;
    }
    *reinterpret_cast<__half*>(slots + mSteps * kSlotBytes) = __float2half(1.0F / static_cast<float>(mSteps));
    CUDA_CHECK(
        cudaMemcpyAsync(mScalars.rawPointer(), slots, (mSteps + 1) * kSlotBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    mVision.bind("patches", mPatches.rawPointer());
    mVision.bind("visual", mVisual.rawPointer());
    mPrefixA.bind("input_ids", mInputIds.rawPointer());
    mPrefixA.bind("visual", mVisual.rawPointer());
    mPrefixA.bind("image_flag", mImageFlag.rawPointer());
    mPrefixA.bind("cos", mCos.rawPointer());
    mPrefixA.bind("sin", mSin.rawPointer());
    mPrefixA.bind("hidden", mHiddenState.rawPointer());
    mPrefixB.bind("hidden", mHiddenState.rawPointer());
    mPrefixB.bind("image_flag", mImageFlag.rawPointer());
    mPrefixB.bind("cos", mCos.rawPointer());
    mPrefixB.bind("sin", mSin.rawPointer());
    mContext.bind("keys", mKeys.rawPointer());
    mContext.bind("values", mValues.rawPointer());
    mContext.bind("context_k", mContextK.rawPointer());
    mContext.bind("context_v", mContextV.rawPointer());
    for (cudaEvent_t& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

MolmoAct2Policy::~MolmoAct2Policy() noexcept
{
    for (auto& [length, graph] : mGraphs)
    {
        cudaGraphExecDestroy(graph);
    }
    for (cudaEvent_t event : mEvents)
    {
        if (event != nullptr)
        {
            cudaEventDestroy(event);
        }
    }
}

std::vector<float> MolmoAct2Policy::normalizeState(std::vector<float> const& state) const
{
    std::vector<float> out(state.size());
    for (size_t i = 0; i < state.size(); ++i)
    {
        float value = state[i];
        if (mStateMask[i] != 0.0F)
        {
            value = 2.0F * (state[i] - mStateLow[i]) / span(mStateLow[i], mStateHigh[i]) - 1.0F;
        }
        out[i] = std::clamp(value, -1.0F, 1.0F);
    }
    return out;
}

std::vector<int32_t> MolmoAct2Policy::promptIds(
    std::string const& task, std::vector<float> const& normalizedState) const
{
    std::string const prompt = robotPrompt(normalizeQuestion(task), stateTokens(normalizedState, mStateBins), mSetup,
        mControlMode, static_cast<int32_t>(mCameras.size()), mImageTokens);
    auto const ids = mTokenizer->encode(prompt);
    std::vector<int32_t> out{mBos};
    out.insert(out.end(), ids.begin(), ids.end());
    return out;
}

void MolmoAct2Policy::enqueueDenoise(int64_t sequence)
{
    auto* scalars = static_cast<char*>(mScalars.rawPointer());
    for (int32_t k = 0; k < mSteps; ++k)
    {
        mStep.bind("x", mX[k % 2].rawPointer());
        mStep.bind("step", scalars + k * kSlotBytes);
        mStep.bind("dt", scalars + mSteps * kSlotBytes);
        mStep.bind("context_k", mContextK.rawPointer());
        mStep.bind("context_v", mContextV.rawPointer());
        mStep.bind("encoder_mask", mEncoderMask.rawPointer());
        mStep.bind("strength", mStrength.rawPointer());
        mStep.bind("x_next", mX[(k + 1) % 2].rawPointer());
        mStep.bind("velocity", mVelocity.rawPointer());
        mStep.setShape("context_k", {mLayers, 1, sequence, mContextHeads, mContextHeadDim});
        mStep.setShape("context_v", {mLayers, 1, sequence, mContextHeads, mContextHeadDim});
        mStep.setShape("encoder_mask", {1, sequence});
        ELLM_CHECK(mStep.enqueue(mStream), "MolmoAct2Policy: step engine failed");
    }
}

MolmoAct2Chunk MolmoAct2Policy::act(std::vector<MolmoAct2View> const& views, std::vector<float> const& state,
    std::string const& task, std::vector<float> const& noise, MolmoAct2Rtc const* rtc)
{
    auto const start = std::chrono::steady_clock::now();
    ELLM_CHECK(static_cast<int32_t>(state.size()) == stateDim(),
        "MolmoAct2Policy: state needs " + std::to_string(stateDim()) + " values");
    MolmoAct2Chunk chunk;
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    // Cameras in the checkpoint's order, each patched on its own thread.
    size_t const cameraFloats = mPatchesHost.getShape().volume() / mCameras.size();
    std::vector<std::future<void>> patching;
    for (size_t c = 0; c < mCameras.size(); ++c)
    {
        auto const it
            = std::find_if(views.begin(), views.end(), [&](MolmoAct2View const& v) { return v.camera == mCameras[c]; });
        ELLM_CHECK(it != views.end(), "MolmoAct2Policy: missing camera " + mCameras[c]);
        float* dst = static_cast<float*>(mPatchesHost.rawPointer()) + c * cameraFloats;
        patching.push_back(std::async(std::launch::async,
            [this, it, dst] { siglipPatches(it->rgb, it->height, it->width, mImageSize, mPatch, dst); }));
    }

    std::vector<int32_t> const ids = promptIds(task, normalizeState(state));
    auto const sequence = static_cast<int64_t>(ids.size());
    ELLM_CHECK(sequence <= mInputIds.getShape()[0], "MolmoAct2Policy: the prompt exceeds the engines' profile");
    chunk.promptTokens = static_cast<int32_t>(sequence);
    auto* stage = static_cast<char*>(mPromptHost.rawPointer());
    auto* hostIds = reinterpret_cast<int64_t*>(stage);
    auto* hostFlag = reinterpret_cast<__half*>(stage + sequence * 8);
    auto* hostMask = hostFlag + sequence;
    auto* hostCos = reinterpret_cast<float*>(stage + sequence * 12);
    auto* hostSin = hostCos + sequence * mHeadDim;
    int32_t const half = mHeadDim / 2;
    for (int64_t p = 0; p < sequence; ++p)
    {
        hostIds[p] = ids[p];
        hostFlag[p] = __float2half(isImageToken(ids[p]) ? 1.0F : 0.0F);
        // The expert cross-attends to every prompt token except the <|im_end|> ones (BOS and turn ends).
        hostMask[p] = __float2half(ids[p] == mBos ? 0.0F : 1.0F);
        for (int32_t j = 0; j < half; ++j)
        {
            float const invFreq = 1.0F / std::pow(mRopeTheta, static_cast<float>(2 * j) / static_cast<float>(mHeadDim));
            float const angle = static_cast<float>(p) * invFreq;
            hostCos[p * mHeadDim + j] = hostCos[p * mHeadDim + j + half] = std::cos(angle);
            hostSin[p * mHeadDim + j] = hostSin[p * mHeadDim + j + half] = std::sin(angle);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), hostIds, sequence * 8, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mImageFlag.rawPointer(), hostFlag, sequence * 2, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mEncoderMask.rawPointer(), hostMask, sequence * 2, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mCos.rawPointer(), hostCos, sequence * mHeadDim * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mSin.rawPointer(), hostSin, sequence * mHeadDim * 4, cudaMemcpyHostToDevice, mStream));

    // Initial sample (padded dims zero; the step graph zeroes them too) and the RTC strengths.
    int64_t const chunkElems = static_cast<int64_t>(mHorizon) * mMaxActionDim;
    std::vector<float> x0(static_cast<size_t>(chunkElems));
    if (noise.empty())
    {
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::generate(x0.begin(), x0.end(), [&] { return normal(mNoiseGen); });
    }
    else
    {
        ELLM_CHECK(static_cast<int64_t>(noise.size()) == chunkElems,
            "MolmoAct2Policy: noise must be [horizon, max action dim]");
        x0 = noise;
    }
    std::vector<float> strength(mHorizon, 1.0F);
    if (rtc != nullptr && rtc->overlapSteps > 0 && !mPrevious.empty())
    {
        int32_t const startRow = rtc->startRow >= 0 ? rtc->startRow : mHorizon - rtc->overlapSteps;
        ELLM_CHECK(rtc->overlapSteps <= mHorizon && startRow >= 0 && startRow <= mHorizon && rtc->frozenSteps >= 0,
            "MolmoAct2Policy: RTC overlap and start row must lie inside the chunk");
        int32_t const overlap = std::min(rtc->overlapSteps, mHorizon - startRow);
        int32_t const frozen = std::min(rtc->frozenSteps, overlap);
        std::copy_n(mPrevious.begin() + static_cast<int64_t>(startRow) * mMaxActionDim,
            static_cast<int64_t>(overlap) * mMaxActionDim, x0.begin());
        double const last = std::max(1.0 - std::exp(-static_cast<double>(rtc->rampRate)), 1e-8);
        for (int32_t row = 0; row < overlap; ++row)
        {
            double const t = static_cast<double>(row - frozen + 1) / (overlap - frozen + 1);
            strength[row] = row < frozen ? 0.0F : static_cast<float>((1.0 - std::exp(-rtc->rampRate * t)) / last);
        }
    }
    auto* stageX = reinterpret_cast<__half*>(static_cast<char*>(mStageHost.rawPointer()) + (mSteps + 1) * kSlotBytes);
    std::transform(x0.begin(), x0.end(), stageX, [](float v) { return __float2half(v); });
    std::transform(strength.begin(), strength.end(), stageX + chunkElems, [](float v) { return __float2half(v); });
    CUDA_CHECK(cudaMemcpyAsync(mX[0].rawPointer(), stageX, chunkElems * 2, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(
        cudaMemcpyAsync(mStrength.rawPointer(), stageX + chunkElems, mHorizon * 2, cudaMemcpyHostToDevice, mStream));
    for (auto& f : patching)
    {
        f.get();
    }
    CUDA_CHECK(cudaMemcpyAsync(mPatches.rawPointer(), mPatchesHost.rawPointer(), mPatchesHost.getMemoryCapacity(),
        cudaMemcpyHostToDevice, mStream));
    chunk.hostMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();

    CUDA_CHECK(cudaEventRecord(mEvents[0], mStream));
    ELLM_CHECK(mVision.enqueue(mStream), "MolmoAct2Policy: vision engine failed");
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));
    size_t const layerElems = static_cast<size_t>(sequence) * mKvDim;
    mPrefixA.bind("keys", mKeys.rawPointer());
    mPrefixA.bind("values", mValues.rawPointer());
    mPrefixB.bind("keys", static_cast<__half*>(mKeys.rawPointer()) + mLayersA * layerElems);
    mPrefixB.bind("values", static_cast<__half*>(mValues.rawPointer()) + mLayersA * layerElems);
    mPrefixA.setShape("input_ids", {sequence});
    mPrefixA.setShape("image_flag", {sequence});
    mPrefixA.setShape("cos", {sequence, mHeadDim});
    mPrefixA.setShape("sin", {sequence, mHeadDim});
    ELLM_CHECK(mPrefixA.enqueue(mStream), "MolmoAct2Policy: prefix_a engine failed");
    mPrefixB.setShape("hidden", {1, sequence, mHidden});
    mPrefixB.setShape("image_flag", {sequence});
    mPrefixB.setShape("cos", {sequence, mHeadDim});
    mPrefixB.setShape("sin", {sequence, mHeadDim});
    ELLM_CHECK(mPrefixB.enqueue(mStream), "MolmoAct2Policy: prefix_b engine failed");
    mContext.setShape("keys", {mLayers, sequence, mKvDim});
    mContext.setShape("values", {mLayers, sequence, mKvDim});
    ELLM_CHECK(mContext.enqueue(mStream), "MolmoAct2Policy: context engine failed");
    CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));

    auto const graph = mGraphs.find(sequence);
    if (mUseCudaGraph && graph != mGraphs.end())
    {
        CUDA_CHECK(cudaGraphLaunch(graph->second, mStream));
    }
    else
    {
        // TensorRT needs one regular enqueue for these shapes before its kernels can be captured.
        enqueueDenoise(sequence);
        if (mUseCudaGraph)
        {
            CUDA_CHECK(cudaMemcpyAsync(mX[0].rawPointer(), stageX, chunkElems * 2, cudaMemcpyHostToDevice, mStream));
            cudaGraph_t captured{};
            CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
            try
            {
                enqueueDenoise(sequence);
            }
            catch (...)
            {
                cudaStreamEndCapture(mStream, &captured);
                if (captured != nullptr)
                {
                    cudaGraphDestroy(captured);
                }
                throw;
            }
            CUDA_CHECK(cudaStreamEndCapture(mStream, &captured));
            cudaGraphExec_t exec{};
            CUDA_CHECK(cudaGraphInstantiate(&exec, captured, 0));
            CUDA_CHECK(cudaGraphDestroy(captured));
            mGraphs.emplace(sequence, exec);
            CUDA_CHECK(cudaGraphLaunch(exec, mStream));
        }
    }
    CUDA_CHECK(cudaEventRecord(mEvents[3], mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        mOutHost.rawPointer(), mX[mSteps % 2].rawPointer(), chunkElems * 2, cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.visionMs, mEvents[0], mEvents[1]));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.prefixMs, mEvents[1], mEvents[2]));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.actionMs, mEvents[2], mEvents[3]));

    auto const* out = static_cast<__half const*>(mOutHost.rawPointer());
    chunk.normalized.resize(static_cast<size_t>(chunkElems));
    std::transform(out, out + chunkElems, chunk.normalized.begin(), [](__half h) { return __half2float(h); });
    mPrevious = chunk.normalized;
    chunk.actions.resize(static_cast<size_t>(mHorizon) * mActionDim);
    for (int32_t t = 0; t < mHorizon; ++t)
    {
        for (int32_t d = 0; d < mActionDim; ++d)
        {
            float const v = std::clamp(chunk.normalized[static_cast<size_t>(t) * mMaxActionDim + d], -1.0F, 1.0F);
            chunk.actions[static_cast<size_t>(t) * mActionDim + d]
                = mActionMask[d] != 0.0F ? (v + 1.0F) * span(mActionLow[d], mActionHigh[d]) / 2.0F + mActionLow[d] : v;
        }
    }
    return chunk;
}

} // namespace molmoact2
} // namespace trt_edgellm
