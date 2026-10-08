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

#include "xvlaPolicy.h"

#include "common/checkMacros.h"

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace trt_edgellm
{
namespace xvla
{

namespace
{

using Json = nlohmann::json;
using nvinfer1::DataType;

constexpr float kImagenetMean[3] = {0.485F, 0.456F, 0.406F};
constexpr float kImagenetStd[3] = {0.229F, 0.224F, 0.225F};

Json readJson(std::string const& path)
{
    std::ifstream file(path);
    ELLM_CHECK(file.good(), "XvlaPolicy: missing " + path);
    return Json::parse(file);
}

rt::Tensor makeTensor(
    std::vector<int64_t> const& shape, DataType type, char const* name, rt::DeviceType device = rt::DeviceType::kGPU)
{
    return rt::Tensor(rt::Coords(shape), device, type, name);
}

//! torch's upsample_bilinear2d source index for align_corners=False.
inline float sourceIndex(float scale, int32_t dst)
{
    return std::max(scale * (static_cast<float>(dst) + 0.5F) - 0.5F, 0.0F);
}

} // namespace

XvlaPolicy::XvlaPolicy(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    Json const config = readJson(engineDir + "/config.json");
    ELLM_CHECK(config.at("model_family").get<std::string>() == "xvla", "XvlaPolicy: not an X-VLA engine dir");
    ELLM_CHECK(config.at("state_normalization").get<std::string>() == "IDENTITY",
        "XvlaPolicy: only identity state normalization is supported");
    mCameras = config.at("cameras").get<std::vector<std::string>>();
    ELLM_CHECK(static_cast<int32_t>(mCameras.size()) == config.at("num_views").get<int32_t>(),
        "XvlaPolicy: every view needs a camera name");
    auto const size = config.at("image_size").get<std::vector<int32_t>>();
    ELLM_CHECK(size.size() == 2 && size[0] == size[1], "XvlaPolicy: only square model images are supported");
    mImageSize = size[0];
    mImageTokens = config.at("image_tokens_per_view").get<int32_t>();
    mMaxTokens = config.at("max_tokens").get<int32_t>();
    ELLM_CHECK(config.at("padding_side").get<std::string>() == "right", "XvlaPolicy: only right padding is supported");
    mPadTokenId = config.at("pad_token_id").get<int64_t>();
    mChunk = config.at("chunk_size").get<int32_t>();
    mActionDim = config.at("action_dim").get<int32_t>();
    mProprioDim = config.at("proprio_dim").get<int32_t>();
    mStateDim = config.at("state_dim").get<int32_t>();
    mNumSteps = config.at("num_steps").get<int32_t>();
    mGripper = config.at("gripper_idx").get<std::vector<int32_t>>();
    // LeRobot unnormalizes with the saved statistics and passes actions through when the checkpoint has none.
    if (config.contains("action_mean"))
    {
        mActionMean = config.at("action_mean").get<std::vector<float>>();
        mActionStd = config.at("action_std").get<std::vector<float>>();
    }

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir + "/tokenizer"), "XvlaPolicy: failed to load the tokenizer");

    mRuntime = vla::createTrtRuntime();
    mVision = vla::TrtEngine(*mRuntime, engineDir + "/vision.engine", stream);
    mEncoder = vla::TrtEngine(*mRuntime, engineDir + "/encoder.engine", stream);
    mStep = vla::TrtEngine(*mRuntime, engineDir + "/step.engine", stream);
    mContextMemory = vla::allocateSharedContextMemory({&mVision, &mEncoder, &mStep}, "xvla::contextMemory");
    mHidden = static_cast<int32_t>(mEncoder.engine().getTensorShape("vlm_features").d[2]);

    auto const views = static_cast<int64_t>(mCameras.size());
    int64_t const pixels = views * 3 * mImageSize * mImageSize;
    mPixelsHost = makeTensor({pixels}, DataType::kHALF, "xvla::pixelsHost", rt::DeviceType::kCPU);
    mPixels = makeTensor({pixels}, DataType::kHALF, "xvla::pixels");
    mFeatures = makeTensor({views, mImageTokens, mHidden}, DataType::kHALF, "xvla::features");
    mTokensHost = makeTensor({mMaxTokens}, DataType::kINT64, "xvla::tokensHost", rt::DeviceType::kCPU);
    mTokens = makeTensor({1, mMaxTokens}, DataType::kINT64, "xvla::tokens");
    mVlm = makeTensor({1, mImageTokens + mMaxTokens, mHidden}, DataType::kHALF, "xvla::vlm");
    int64_t const chunkElems = static_cast<int64_t>(mChunk) * mActionDim;
    mX1 = makeTensor({1, mChunk, mActionDim}, DataType::kHALF, "xvla::x1");
    mAction[0] = makeTensor({1, mChunk, mActionDim}, DataType::kHALF, "xvla::action0");
    mAction[1] = makeTensor({1, mChunk, mActionDim}, DataType::kHALF, "xvla::action1");
    mTimes = makeTensor({mNumSteps}, DataType::kHALF, "xvla::times");
    mProprio = makeTensor({1, mProprioDim}, DataType::kHALF, "xvla::proprio");
    mStageHost
        = makeTensor({chunkElems + mProprioDim + mNumSteps}, DataType::kHALF, "xvla::stageHost", rt::DeviceType::kCPU);
    mOutHost = makeTensor({chunkElems}, DataType::kHALF, "xvla::outHost", rt::DeviceType::kCPU);

    auto* times = static_cast<__half*>(mStageHost.rawPointer());
    for (int32_t s = 0; s < mNumSteps; ++s)
    {
        times[s] = __float2half(static_cast<float>(mNumSteps - s) / static_cast<float>(mNumSteps));
    }
    CUDA_CHECK(cudaMemcpyAsync(mTimes.rawPointer(), times, mNumSteps * sizeof(__half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (cudaEvent_t& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

XvlaPolicy::~XvlaPolicy() noexcept
{
    for (cudaEvent_t event : mEvents)
    {
        cudaEventDestroy(event);
    }
}

std::vector<float> XvlaPolicy::preprocessView(unsigned char const* rgb, int32_t height, int32_t width) const
{
    int32_t const size = mImageSize;
    auto normalized = [&](int32_t y, int32_t x, int32_t c) {
        float const v = static_cast<float>(rgb[(static_cast<size_t>(y) * width + x) * 3 + c]) / 255.0F;
        return (v - kImagenetMean[c]) / kImagenetStd[c];
    };
    std::vector<float> out(static_cast<size_t>(3) * size * size, 0.0F);
    if (height == size && width == size)
    {
        for (int32_t c = 0; c < 3; ++c)
        {
            for (int32_t y = 0; y < size; ++y)
            {
                for (int32_t x = 0; x < size; ++x)
                {
                    out[(static_cast<size_t>(c) * size + y) * size + x] = normalized(y, x, c);
                }
            }
        }
        return out;
    }
    double const ratio = std::max(static_cast<double>(width) / size, static_cast<double>(height) / size);
    auto const resizedH = static_cast<int32_t>(height / ratio);
    auto const resizedW = static_cast<int32_t>(width / ratio);
    int32_t const padH = std::max(0, size - resizedH);
    int32_t const padW = std::max(0, size - resizedW);
    float const scaleH = static_cast<float>(height) / static_cast<float>(resizedH);
    float const scaleW = static_cast<float>(width) / static_cast<float>(resizedW);
    for (int32_t y = 0; y < resizedH; ++y)
    {
        float const sy = sourceIndex(scaleH, y);
        auto const y0 = static_cast<int32_t>(sy);
        int32_t const y1 = y0 + (y0 < height - 1 ? 1 : 0);
        float const ly = sy - static_cast<float>(y0);
        for (int32_t x = 0; x < resizedW; ++x)
        {
            float const sx = sourceIndex(scaleW, x);
            auto const x0 = static_cast<int32_t>(sx);
            int32_t const x1 = x0 + (x0 < width - 1 ? 1 : 0);
            float const lx = sx - static_cast<float>(x0);
            for (int32_t c = 0; c < 3; ++c)
            {
                float const top = (1.0F - lx) * normalized(y0, x0, c) + lx * normalized(y0, x1, c);
                float const bottom = (1.0F - lx) * normalized(y1, x0, c) + lx * normalized(y1, x1, c);
                out[(static_cast<size_t>(c) * size + (y + padH)) * size + (x + padW)] = (1.0F - ly) * top + ly * bottom;
            }
        }
    }
    return out;
}

std::vector<int64_t> XvlaPolicy::tokenize(std::string const& task) const
{
    auto const ids = mTokenizer->encode(task, /*addBos=*/true, /*addEos=*/true);
    std::vector<int64_t> out(ids.begin(), ids.end());
    if (static_cast<int32_t>(out.size()) > mMaxTokens)
    {
        // HF truncation keeps the closing special token.
        int64_t const eos = out.back();
        out.resize(mMaxTokens);
        out.back() = eos;
    }
    out.resize(mMaxTokens, mPadTokenId);
    return out;
}

XvlaChunk XvlaPolicy::act(std::vector<XvlaView> const& views, std::vector<float> const& state, std::string const& task,
    std::vector<float> const& noise)
{
    XvlaChunk chunk;
    ELLM_CHECK(static_cast<int32_t>(state.size()) == mStateDim, "XvlaPolicy: state has the wrong width");
    size_t const viewSize = static_cast<size_t>(3) * mImageSize * mImageSize;
    auto* pixels = static_cast<__half*>(mPixelsHost.rawPointer());
    // Present cameras first, in config order; LeRobot then pads the missing ones as masked views.
    std::vector<int32_t> present;
    for (size_t c = 0; c < mCameras.size(); ++c)
    {
        auto const it
            = std::find_if(views.begin(), views.end(), [&](XvlaView const& v) { return v.camera == mCameras[c]; });
        if (it == views.end())
        {
            continue;
        }
        std::vector<float> const planar = preprocessView(it->rgb, it->height, it->width);
        std::transform(planar.begin(), planar.end(), pixels + present.size() * viewSize, __float2half);
        present.push_back(static_cast<int32_t>(c));
    }
    ELLM_CHECK(
        !present.empty() && present.front() == 0, "XvlaPolicy: the first camera (the encoder's view) is required");
    for (size_t i = 0; i < present.size(); ++i)
    {
        ELLM_CHECK(present[i] == static_cast<int32_t>(i), "XvlaPolicy: cameras may only be missing at the end");
    }
    auto const numPresent = static_cast<int64_t>(present.size());

    chunk.tokenIds = tokenize(task);
    std::copy(chunk.tokenIds.begin(), chunk.tokenIds.end(), mTokensHost.dataPointer<int64_t>());

    int64_t const chunkElems = static_cast<int64_t>(mChunk) * mActionDim;
    auto* stage = static_cast<__half*>(mStageHost.rawPointer()) + mNumSteps;
    if (noise.empty())
    {
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (int64_t i = 0; i < chunkElems; ++i)
        {
            stage[i] = __float2half(normal(mNoiseGen));
        }
    }
    else
    {
        ELLM_CHECK(static_cast<int64_t>(noise.size()) == chunkElems, "XvlaPolicy: noise has the wrong size");
        std::transform(noise.begin(), noise.end(), stage, __float2half);
    }
    for (int32_t d = 0; d < mProprioDim; ++d)
    {
        stage[chunkElems + d] = __float2half(d < mStateDim ? state[d] : 0.0F);
    }

    size_t const featureBytes = static_cast<size_t>(mImageTokens) * mHidden * sizeof(__half);
    CUDA_CHECK(cudaMemcpyAsync(
        mPixels.rawPointer(), pixels, numPresent * viewSize * sizeof(__half), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        mTokens.rawPointer(), mTokensHost.rawPointer(), mMaxTokens * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mX1.rawPointer(), stage, chunkElems * sizeof(__half), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        mProprio.rawPointer(), stage + chunkElems, mProprioDim * sizeof(__half), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemsetAsync(mAction[0].rawPointer(), 0, chunkElems * sizeof(__half), mStream));
    auto* features = static_cast<char*>(mFeatures.rawPointer());
    int64_t const missing = static_cast<int64_t>(mCameras.size()) - numPresent;
    if (missing > 0)
    {
        CUDA_CHECK(cudaMemsetAsync(features + numPresent * featureBytes, 0, missing * featureBytes, mStream));
    }

    CUDA_CHECK(cudaEventRecord(mEvents[0], mStream));
    mVision.setShape("images", {numPresent, 3, mImageSize, mImageSize});
    mVision.bind("images", mPixels.rawPointer());
    mVision.bind("image_features", features);
    ELLM_CHECK(mVision.enqueue(mStream), "XvlaPolicy: vision enqueue failed");
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));

    mEncoder.bind("primary_features", features);
    mEncoder.bind("token_ids", mTokens.rawPointer());
    mEncoder.bind("vlm_features", mVlm.rawPointer());
    ELLM_CHECK(mEncoder.enqueue(mStream), "XvlaPolicy: encoder enqueue failed");
    CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));

    mStep.bind("x1", mX1.rawPointer());
    mStep.bind("proprio", mProprio.rawPointer());
    mStep.bind("vlm_features", mVlm.rawPointer());
    mStep.bind("aux_visual_inputs", features + featureBytes);
    for (int32_t s = 0; s < mNumSteps; ++s)
    {
        mStep.bind("action", mAction[s % 2].rawPointer());
        mStep.bind("t", mTimes.dataPointer<__half>() + s);
        mStep.bind("action_next", mAction[(s + 1) % 2].rawPointer());
        ELLM_CHECK(mStep.enqueue(mStream), "XvlaPolicy: step enqueue failed");
    }
    CUDA_CHECK(cudaEventRecord(mEvents[3], mStream));
    CUDA_CHECK(cudaMemcpyAsync(mOutHost.rawPointer(), mAction[mNumSteps % 2].rawPointer(), chunkElems * sizeof(__half),
        cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    cudaEventElapsedTime(&chunk.visionMs, mEvents[0], mEvents[1]);
    cudaEventElapsedTime(&chunk.encoderMs, mEvents[1], mEvents[2]);
    cudaEventElapsedTime(&chunk.denoiseMs, mEvents[2], mEvents[3]);

    auto const* out = static_cast<__half const*>(mOutHost.rawPointer());
    chunk.actions.resize(chunkElems);
    for (int64_t i = 0; i < chunkElems; ++i)
    {
        chunk.actions[i] = __half2float(out[i]);
    }
    for (int32_t t = 0; t < mChunk; ++t)
    {
        for (int32_t g : mGripper)
        {
            float& v = chunk.actions[static_cast<size_t>(t) * mActionDim + g];
            v = 1.0F / (1.0F + std::exp(-v));
        }
        for (int32_t d = 0; d < mActionDim && !mActionMean.empty(); ++d)
        {
            float& v = chunk.actions[static_cast<size_t>(t) * mActionDim + d];
            v = v * mActionStd[d] + mActionMean[d];
        }
    }
    return chunk;
}

} // namespace xvla
} // namespace trt_edgellm
