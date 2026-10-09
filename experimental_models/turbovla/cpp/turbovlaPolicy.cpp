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

#include "turbovlaPolicy.h"

#include "common/checkMacros.h"
#include "vlaImage.h"

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>

namespace trt_edgellm
{
namespace turbovla
{

namespace
{

using Json = nlohmann::json;
using nvinfer1::DataType;

Json readJson(std::string const& path)
{
    std::ifstream file(path);
    ELLM_CHECK(file.good(), "TurbovlaPolicy: missing " + path);
    return Json::parse(file);
}

rt::Tensor makeTensor(
    std::vector<int64_t> const& shape, DataType type, char const* name, rt::DeviceType device = rt::DeviceType::kGPU)
{
    return rt::Tensor(rt::Coords(shape), device, type, name);
}

//! The exporter drops graph inputs a checkpoint does not use (e.g. the text attention without zero_padded_tokens).
void bindIfPresent(vla::TrtEngine& engine, char const* name, void const* address)
{
    if (engine.engine().getTensorIOMode(name) != nvinfer1::TensorIOMode::kNONE)
    {
        engine.bind(name, address);
    }
}

} // namespace

TurbovlaPolicy::TurbovlaPolicy(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    Json const config = readJson(engineDir + "/config.json");
    ELLM_CHECK(config.at("model_family").get<std::string>() == "turbovla", "TurbovlaPolicy: not a TurboVLA engine dir");
    mCameras = config.at("cameras").get<std::vector<std::string>>();
    ELLM_CHECK(static_cast<int32_t>(mCameras.size()) == config.at("num_views").get<int32_t>(),
        "TurbovlaPolicy: one camera name per view");
    mImageSize = config.at("image_size").get<int32_t>();
    mTextLength = config.at("text_length").get<int32_t>();
    mTextLengthByInstruction = config.at("text_length_by_instruction").get<std::map<std::string, int32_t>>();
    mSplitTokens = config.at("split_tokens").get<std::vector<int64_t>>();
    mHidden = config.at("hidden_dim").get<int32_t>();
    mChunk = config.at("chunk_size").get<int32_t>();
    mActionDim = config.at("action_dim").get<int32_t>();
    mStateDim = config.at("state_dim").get<int32_t>();
    mStateMean = config.at("state_mean").get<std::vector<float>>();
    mStateStd = config.at("state_std").get<std::vector<float>>();
    mActionMin = config.at("action_min").get<std::vector<float>>();
    mActionMax = config.at("action_max").get<std::vector<float>>();
    auto const mean = config.at("image_mean").get<std::vector<float>>();
    auto const std = config.at("image_std").get<std::vector<float>>();
    float const rescale = 1.0F / 255.0F;
    for (int32_t c = 0; c < 3; ++c)
    {
        for (int32_t v = 0; v < 256; ++v)
        {
            mPixelLut[c][v] = (static_cast<float>(v) * rescale - mean[c]) / std[c];
        }
    }

    mTokenizer = std::make_unique<BertWordPiece>(engineDir + "/tokenizer.json");
    mRuntime = vla::createTrtRuntime();
    mText = vla::TrtEngine(*mRuntime, engineDir + "/text.engine", stream);
    mPolicy = vla::TrtEngine(*mRuntime, engineDir + "/policy.engine", stream);
    mContextMemory = vla::allocateSharedContextMemory({&mText, &mPolicy}, "turbovla::contextMemory");

    int64_t const views = static_cast<int64_t>(mCameras.size());
    int64_t const size = mImageSize;
    int64_t const length = mTextLength;
    mPixelsHost = makeTensor({views * 3 * size * size}, DataType::kHALF, "turbovla::pixelsHost", rt::DeviceType::kCPU);
    mPixels = makeTensor({1, views, 3, size, size}, DataType::kHALF, "turbovla::pixels");
    // ids and positions (2 x L INT64 = 8 L halves), then the self-attention, hidden-valid and attention masks.
    mTextHost = makeTensor(
        {8 * length + length * length + 2 * length}, DataType::kHALF, "turbovla::textHost", rt::DeviceType::kCPU);
    mInputIds = makeTensor({1, length}, DataType::kINT64, "turbovla::inputIds");
    mPositionIds = makeTensor({1, length}, DataType::kINT64, "turbovla::positionIds");
    mSelfAttention = makeTensor({1, length, length}, DataType::kHALF, "turbovla::selfAttention");
    mHiddenValid = makeTensor({1, length}, DataType::kHALF, "turbovla::hiddenValid");
    mAttention = makeTensor({1, length}, DataType::kHALF, "turbovla::attention");
    mTextTokens = makeTensor({1, length, mHidden}, DataType::kHALF, "turbovla::textTokens");
    mState = makeTensor({1, mStateDim}, DataType::kHALF, "turbovla::state");
    mStateHost = makeTensor({mStateDim}, DataType::kHALF, "turbovla::stateHost", rt::DeviceType::kCPU);
    mActions = makeTensor({1, mChunk, mActionDim}, DataType::kHALF, "turbovla::actions");
    mActionsHost = makeTensor(
        {static_cast<int64_t>(mChunk) * mActionDim}, DataType::kHALF, "turbovla::actionsHost", rt::DeviceType::kCPU);

    mText.bind("input_ids", mInputIds.rawPointer());
    mText.bind("position_ids", mPositionIds.rawPointer());
    mText.bind("self_attention", mSelfAttention.rawPointer());
    bindIfPresent(mText, "hidden_valid", mHiddenValid.rawPointer());
    bindIfPresent(mText, "attention", mAttention.rawPointer());
    mText.bind("text_tokens", mTextTokens.rawPointer());
    mPolicy.bind("pixels", mPixels.rawPointer());
    mPolicy.bind("text_tokens", mTextTokens.rawPointer());
    bindIfPresent(mPolicy, "attention", mAttention.rawPointer());
    bindIfPresent(mPolicy, "self_attention", mSelfAttention.rawPointer());
    mPolicy.bind("state", mState.rawPointer());
    mPolicy.bind("actions", mActions.rawPointer());
    for (cudaEvent_t& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

TurbovlaPolicy::~TurbovlaPolicy() noexcept
{
    for (cudaEvent_t event : mEvents)
    {
        if (event != nullptr)
        {
            cudaEventDestroy(event);
        }
    }
}

TurbovlaTextInputs TurbovlaPolicy::textInputs(std::string const& task) const
{
    auto const layout = mTextLengthByInstruction.find(task);
    int32_t const own = layout == mTextLengthByInstruction.end() ? mTextLength : layout->second;
    ELLM_CHECK(own <= mTextLength, "TurbovlaPolicy: instruction padding length exceeds the text length");
    auto const L = static_cast<size_t>(mTextLength);

    TurbovlaTextInputs in;
    std::vector<uint8_t> ownAttention;
    std::vector<int64_t> const ids = mTokenizer->encode(task, own, &ownAttention);
    in.inputIds.assign(L, 0);
    std::copy(ids.begin(), ids.end(), in.inputIds.begin());
    in.positionIds.assign(L, 0);
    in.selfAttention.assign(L * L, 0);
    for (size_t i = 0; i < L; ++i)
    {
        in.selfAttention[i * L + i] = 1;
    }
    in.hiddenValid.assign(L, 0);
    std::fill_n(in.hiddenValid.begin(), own, 1);
    in.attention.assign(L, 0);
    std::copy(ownAttention.begin(), ownAttention.end(), in.attention.begin());

    // GroundingDINO's generate_masks_with_special_tokens over the instruction's own length: each span that ends at a
    // split token attends within itself, with positions restarting at 0; a split token at either end stands alone.
    int32_t previous = 0;
    for (int32_t col = 0; col < own; ++col)
    {
        if (std::find(mSplitTokens.begin(), mSplitTokens.end(), ids[col]) == mSplitTokens.end())
        {
            continue;
        }
        if (col == 0 || col == own - 1)
        {
            in.positionIds[col] = 0;
        }
        else
        {
            for (int32_t r = previous + 1; r <= col; ++r)
            {
                for (int32_t c = previous + 1; c <= col; ++c)
                {
                    in.selfAttention[static_cast<size_t>(r) * L + c] = 1;
                }
                in.positionIds[r] = r - previous - 1;
            }
        }
        previous = col;
    }
    return in;
}

std::vector<float> TurbovlaPolicy::blendWeights(int32_t overlap, int32_t frozen, float rampRate)
{
    frozen = std::min(frozen, overlap);
    int32_t const ramped = overlap - frozen;
    double const last = std::max(1.0 - std::exp(-static_cast<double>(rampRate)), 1e-8);
    std::vector<float> weights;
    for (int32_t t = 0; t < overlap; ++t)
    {
        double ramp = 0.0;
        if (t >= frozen)
        {
            double const u = static_cast<double>(t - frozen + 1) / (ramped + 1);
            ramp = (1.0 - std::exp(-rampRate * u)) / last;
        }
        weights.push_back(static_cast<float>(1.0 - ramp));
    }
    return weights;
}

void TurbovlaPolicy::stageView(int32_t view, unsigned char const* rgb, int32_t height, int32_t width)
{
    std::vector<unsigned char> resized;
    if (height != mImageSize || width != mImageSize)
    {
        resized = vla::resizeBicubicPil(rgb, height, width, mImageSize, mImageSize);
        rgb = resized.data();
    }
    size_t const plane = static_cast<size_t>(mImageSize) * mImageSize;
    auto* out = static_cast<__half*>(mPixelsHost.rawPointer()) + static_cast<size_t>(view) * 3 * plane;
    for (size_t p = 0; p < plane; ++p)
    {
        for (int32_t c = 0; c < 3; ++c)
        {
            out[c * plane + p] = __float2half(mPixelLut[c][rgb[p * 3 + c]]);
        }
    }
}

TurbovlaChunk TurbovlaPolicy::act(std::vector<TurbovlaView> const& views, std::vector<float> const& state,
    std::string const& task, TurbovlaBlend const* blend)
{
    auto const start = std::chrono::steady_clock::now();
    ELLM_CHECK(static_cast<int32_t>(state.size()) == mStateDim,
        "TurbovlaPolicy: state needs " + std::to_string(mStateDim) + " values");
    std::vector<std::future<void>> staging;
    std::vector<bool> seen(mCameras.size(), false);
    for (auto const& view : views)
    {
        auto const it = std::find(mCameras.begin(), mCameras.end(), view.camera);
        ELLM_CHECK(it != mCameras.end(), "TurbovlaPolicy: unknown camera " + view.camera);
        auto const index = static_cast<int32_t>(it - mCameras.begin());
        ELLM_CHECK(!seen[index], "TurbovlaPolicy: camera " + view.camera + " given twice");
        seen[index] = true;
        staging.push_back(std::async(
            std::launch::async, [this, index, view] { stageView(index, view.rgb, view.height, view.width); }));
    }
    ELLM_CHECK(
        std::all_of(seen.begin(), seen.end(), [](bool s) { return s; }), "TurbovlaPolicy: every camera is required");

    auto* stateHost = static_cast<__half*>(mStateHost.rawPointer());
    for (int32_t i = 0; i < mStateDim; ++i)
    {
        stateHost[i] = __float2half((state[i] - mStateMean[i]) / (mStateStd[i] + 1e-6F));
    }

    TurbovlaChunk chunk;
    chunk.textCached = task == mLastTask;
    int64_t const length = mTextLength;
    if (!chunk.textCached)
    {
        TurbovlaTextInputs const in = textInputs(task);
        auto* ids = static_cast<int64_t*>(mTextHost.rawPointer());
        std::copy(in.inputIds.begin(), in.inputIds.end(), ids);
        std::copy(in.positionIds.begin(), in.positionIds.end(), ids + length);
        auto* masks = static_cast<__half*>(mTextHost.rawPointer()) + 8 * length;
        auto toHalf = [](uint8_t v) { return __float2half(static_cast<float>(v)); };
        std::transform(in.selfAttention.begin(), in.selfAttention.end(), masks, toHalf);
        std::transform(in.hiddenValid.begin(), in.hiddenValid.end(), masks + length * length, toHalf);
        std::transform(in.attention.begin(), in.attention.end(), masks + length * length + length, toHalf);
        CUDA_CHECK(
            cudaMemcpyAsync(mInputIds.rawPointer(), ids, length * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(
            mPositionIds.rawPointer(), ids + length, length * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(
            mSelfAttention.rawPointer(), masks, length * length * sizeof(__half), cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(mHiddenValid.rawPointer(), masks + length * length, length * sizeof(__half),
            cudaMemcpyHostToDevice, mStream));
        CUDA_CHECK(cudaMemcpyAsync(mAttention.rawPointer(), masks + length * length + length, length * sizeof(__half),
            cudaMemcpyHostToDevice, mStream));
    }
    CUDA_CHECK(cudaEventRecord(mEvents[0], mStream));
    if (!chunk.textCached)
    {
        ELLM_CHECK(mText.enqueue(mStream), "TurbovlaPolicy: text engine failed");
        mLastTask = task;
    }
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));

    for (auto& f : staging)
    {
        f.get();
    }
    CUDA_CHECK(cudaMemcpyAsync(mPixels.rawPointer(), mPixelsHost.rawPointer(), mPixelsHost.getMemoryCapacity(),
        cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(
        cudaMemcpyAsync(mState.rawPointer(), stateHost, mStateDim * sizeof(__half), cudaMemcpyHostToDevice, mStream));
    chunk.hostMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    ELLM_CHECK(mPolicy.enqueue(mStream), "TurbovlaPolicy: policy engine failed");
    CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));
    size_t const elems = static_cast<size_t>(mChunk) * mActionDim;
    CUDA_CHECK(cudaMemcpyAsync(
        mActionsHost.rawPointer(), mActions.rawPointer(), elems * sizeof(__half), cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.textMs, mEvents[0], mEvents[1]));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.policyMs, mEvents[1], mEvents[2]));

    auto const* out = static_cast<__half const*>(mActionsHost.rawPointer());
    chunk.normalized.resize(elems);
    std::transform(out, out + elems, chunk.normalized.begin(), [](__half h) { return __half2float(h); });
    if (blend != nullptr && !mPrevious.empty() && blend->overlapSteps > 0)
    {
        int32_t const startRow = blend->startRow >= 0 ? blend->startRow : mChunk - blend->overlapSteps;
        int32_t const overlap = std::max(0, std::min(blend->overlapSteps, mChunk - startRow));
        std::vector<float> const weights = blendWeights(overlap, blend->frozenSteps, blend->rampRate);
        for (int32_t t = 0; t < overlap; ++t)
        {
            for (int32_t d = 0; d < mActionDim; ++d)
            {
                float& value = chunk.normalized[static_cast<size_t>(t) * mActionDim + d];
                float const previous = mPrevious[static_cast<size_t>(startRow + t) * mActionDim + d];
                value = weights[t] * previous + (1.0F - weights[t]) * value;
            }
        }
    }
    mPrevious = chunk.normalized;

    chunk.actions.resize(elems);
    int32_t const gripper = mActionDim - 1;
    for (int32_t t = 0; t < mChunk; ++t)
    {
        for (int32_t d = 0; d < mActionDim; ++d)
        {
            size_t const i = static_cast<size_t>(t) * mActionDim + d;
            float const v = chunk.normalized[i];
            chunk.actions[i] = d == gripper ? (v < 0.0F ? -1.0F : 1.0F)
                                            : 0.5F * (v + 1.0F) * (mActionMax[d] - mActionMin[d]) + mActionMin[d];
        }
    }
    return chunk;
}

} // namespace turbovla
} // namespace trt_edgellm
