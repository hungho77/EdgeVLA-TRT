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

#include "smolvlaPolicy.h"

#include "common/checkMacros.h"

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace smolvla
{
namespace
{

using Json = nlohmann::json;

Json readJson(std::string const& path)
{
    std::ifstream file(path);
    ELLM_CHECK(file.good(), "SmolvlaPolicy: missing " + path);
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

SmolvlaPolicy::SmolvlaPolicy(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    Json const config = readJson(engineDir + "/config.json");
    mImageSize = config.at("image_size").get<int32_t>();
    mImageTokens = config.at("image_tokens_per_view").get<int32_t>();
    mMaxViews = config.at("max_views").get<int32_t>();
    mMaxTokens = config.at("max_tokens").get<int32_t>();
    mMaxPrefix = config.at("max_prefix_len").get<int32_t>();
    mChunk = config.at("chunk_size").get<int32_t>();
    mMaxActionDim = config.at("max_action_dim").get<int32_t>();
    mMaxStateDim = config.at("max_state_dim").get<int32_t>();
    mStateDim = config.at("state_dim").get<int32_t>();
    mActionDim = config.at("action_dim").get<int32_t>();
    mNumSteps = config.at("num_steps").get<int32_t>();
    mKVHeads = config.at("kv_heads").get<int32_t>();
    mHeadDim = config.at("head_dim").get<int32_t>();
    mCameras = config.at("cameras").get<std::vector<std::string>>();
    mKVNames = config.at("kv_names").get<std::vector<std::string>>();
    mPromptSuffix = config.value("prompt_suffix", std::string("\n"));

    Json const norm = readJson(engineDir + "/assets/normalization.json");
    mStateMean = norm.at("state_mean").get<std::vector<float>>();
    mStateStd = norm.at("state_std").get<std::vector<float>>();
    mStateEps = norm.at("state_eps").get<float>();
    mActionMean = norm.at("action_mean").get<std::vector<float>>();
    mActionStd = norm.at("action_std").get<std::vector<float>>();

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir + "/assets"), "SmolvlaPolicy: failed to load the tokenizer");

    mRuntime = vla::createTrtRuntime();
    mVisual = vla::TrtEngine(*mRuntime, engineDir + "/visual.engine", stream);
    mPrefix = vla::TrtEngine(*mRuntime, engineDir + "/prefix.engine", stream);
    mDenoise = vla::TrtEngine(*mRuntime, engineDir + "/denoise.engine", stream);
    mContextMemory = vla::allocateSharedContextMemory({&mVisual, &mPrefix, &mDenoise}, "smolvla::contextMemory");

    int64_t const pixels = static_cast<int64_t>(mMaxViews) * 3 * mImageSize * mImageSize;
    mPixelsHost = makeTensor({pixels}, DataType::kHALF, "smolvla::pixelsHost", rt::DeviceType::kCPU);
    mPixels = makeTensor({pixels}, DataType::kHALF, "smolvla::pixels");
    mHidden = static_cast<int32_t>(mVisual.engine().getTensorShape("image_features").d[2]);
    mImageFeatures = makeTensor(
        {static_cast<int64_t>(mMaxViews) * mImageTokens * mHidden}, DataType::kHALF, "smolvla::imageFeatures");
    mTokensHost = makeTensor({mMaxTokens}, DataType::kINT64, "smolvla::tokensHost", rt::DeviceType::kCPU);
    mTokens = makeTensor({mMaxTokens}, DataType::kINT64, "smolvla::tokens");
    mStateHost = makeTensor({mMaxStateDim}, DataType::kFLOAT, "smolvla::stateHost", rt::DeviceType::kCPU);
    mState = makeTensor({mMaxStateDim}, DataType::kFLOAT, "smolvla::state");
    for (size_t i = 0; i < mKVNames.size(); ++i)
    {
        mKV.push_back(makeTensor({1, mMaxPrefix, mKVHeads, mHeadDim}, DataType::kHALF, "smolvla::kv"));
    }
    int64_t const chunkElems = static_cast<int64_t>(mChunk) * mMaxActionDim;
    mX[0] = makeTensor({1, mChunk, mMaxActionDim}, DataType::kFLOAT, "smolvla::x0");
    mX[1] = makeTensor({1, mChunk, mMaxActionDim}, DataType::kFLOAT, "smolvla::x1");
    mNoiseHost = makeTensor({chunkElems}, DataType::kFLOAT, "smolvla::noiseHost", rt::DeviceType::kCPU);
    mOutHost = makeTensor({chunkElems}, DataType::kFLOAT, "smolvla::outHost", rt::DeviceType::kCPU);
    mTimesteps = makeTensor({mNumSteps}, DataType::kFLOAT, "smolvla::timesteps");
    mDt = makeTensor({1}, DataType::kFLOAT, "smolvla::dt");

    // LeRobot's Euler schedule: t = 1 + step * dt, dt = -1 / steps.
    auto stage = makeTensor({mNumSteps + 1}, DataType::kFLOAT, "smolvla::stage", rt::DeviceType::kCPU);
    float const dt = -1.0F / static_cast<float>(mNumSteps);
    for (int32_t s = 0; s < mNumSteps; ++s)
    {
        stage.dataPointer<float>()[s] = 1.0F + static_cast<float>(s) * dt;
    }
    stage.dataPointer<float>()[mNumSteps] = dt;
    CUDA_CHECK(cudaMemcpyAsync(
        mTimesteps.rawPointer(), stage.rawPointer(), mNumSteps * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mDt.rawPointer(), stage.dataPointer<float>() + mNumSteps, sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (cudaEvent_t& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

SmolvlaPolicy::~SmolvlaPolicy() noexcept
{
    for (auto& [len, exec] : mGraphs)
    {
        cudaGraphExecDestroy(exec);
    }
    for (cudaEvent_t event : mEvents)
    {
        cudaEventDestroy(event);
    }
}

std::vector<float> SmolvlaPolicy::preprocessView(unsigned char const* rgb, int32_t height, int32_t width, int32_t size)
{
    float const ratio = std::max(static_cast<float>(width) / size, static_cast<float>(height) / size);
    auto const resizedH = static_cast<int32_t>(static_cast<float>(height) / ratio);
    auto const resizedW = static_cast<int32_t>(static_cast<float>(width) / ratio);
    int32_t const padH = std::max(0, size - resizedH);
    int32_t const padW = std::max(0, size - resizedW);
    float const scaleH = static_cast<float>(height) / static_cast<float>(resizedH);
    float const scaleW = static_cast<float>(width) / static_cast<float>(resizedW);
    bool const identity = height == size && width == size;

    // Padding is 0 in [0, 1], i.e. -1 after the [-1, 1] mapping.
    std::vector<float> out(static_cast<size_t>(3) * size * size, -1.0F);
    for (int32_t y = 0; y < resizedH; ++y)
    {
        float const sy = identity ? static_cast<float>(y) : sourceIndex(scaleH, y);
        auto const y0 = static_cast<int32_t>(sy);
        int32_t const y1 = y0 + (y0 < height - 1 ? 1 : 0);
        float const ly = sy - static_cast<float>(y0);
        for (int32_t x = 0; x < resizedW; ++x)
        {
            float const sx = identity ? static_cast<float>(x) : sourceIndex(scaleW, x);
            auto const x0 = static_cast<int32_t>(sx);
            int32_t const x1 = x0 + (x0 < width - 1 ? 1 : 0);
            float const lx = sx - static_cast<float>(x0);
            for (int32_t c = 0; c < 3; ++c)
            {
                auto const at = [&](int32_t yy, int32_t xx) {
                    return static_cast<float>(rgb[(static_cast<size_t>(yy) * width + xx) * 3 + c]) / 255.0F;
                };
                float const top = (1.0F - lx) * at(y0, x0) + lx * at(y0, x1);
                float const bottom = (1.0F - lx) * at(y1, x0) + lx * at(y1, x1);
                float const v = (1.0F - ly) * top + ly * bottom;
                out[(static_cast<size_t>(c) * size + (y + padH)) * size + (x + padW)] = v * 2.0F - 1.0F;
            }
        }
    }
    return out;
}

std::vector<int64_t> SmolvlaPolicy::tokenize(std::string const& task) const
{
    auto const ids = mTokenizer->encode(task + mPromptSuffix, /*addBos=*/false, /*addEos=*/false);
    std::vector<int64_t> out(ids.begin(), ids.end());
    // LeRobot truncates at tokenizer_max_length.
    if (static_cast<int32_t>(out.size()) > mMaxTokens)
    {
        out.resize(static_cast<size_t>(mMaxTokens));
    }
    return out;
}

std::vector<float> SmolvlaPolicy::normalizeState(std::vector<float> const& state) const
{
    ELLM_CHECK(static_cast<int32_t>(state.size()) == mStateDim, "SmolvlaPolicy: state has the wrong width");
    std::vector<float> out(static_cast<size_t>(mMaxStateDim), 0.0F);
    for (int32_t d = 0; d < mStateDim; ++d)
    {
        out[d] = (state[d] - mStateMean[d]) / (mStateStd[d] + mStateEps);
    }
    return out;
}

void SmolvlaPolicy::enqueueDenoiseLoop(int64_t prefixLen)
{
    for (size_t i = 0; i < mKVNames.size(); ++i)
    {
        mDenoise.setShape(mKVNames[i].c_str(), {1, prefixLen, mKVHeads, mHeadDim});
        mDenoise.bind(mKVNames[i].c_str(), mKV[i].rawPointer());
    }
    mDenoise.setShape("x_t", {1, mChunk, mMaxActionDim});
    mDenoise.setShape("timestep", {1});
    mDenoise.setShape("dt", {});
    mDenoise.bind("dt", mDt.rawPointer());
    for (int32_t step = 0; step < mNumSteps; ++step)
    {
        mDenoise.bind("x_t", mX[step % 2].rawPointer());
        mDenoise.bind("timestep", mTimesteps.dataPointer<float>() + step);
        mDenoise.bind("x_next", mX[(step + 1) % 2].rawPointer());
        ELLM_CHECK(mDenoise.enqueue(mStream), "SmolvlaPolicy: denoise enqueue failed");
    }
}

SmolvlaChunk SmolvlaPolicy::act(SmolvlaObservation const& observation, std::vector<float> const& noise)
{
    // Cameras in the order LeRobot feeds them; a configured camera the request lacks is skipped.
    std::vector<SmolvlaView const*> ordered;
    for (auto const& camera : mCameras)
    {
        for (auto const& view : observation.views)
        {
            if (view.camera == camera)
            {
                ordered.push_back(&view);
            }
        }
    }
    ELLM_CHECK(!ordered.empty() && ordered.size() == observation.views.size(),
        "SmolvlaPolicy: every view must name one of the configured cameras");
    auto const views = static_cast<int64_t>(ordered.size());

    SmolvlaChunk chunk;
    chunk.tokenIds = tokenize(observation.task);
    std::vector<float> const state = normalizeState(observation.state);
    auto const tokens = static_cast<int64_t>(chunk.tokenIds.size());
    int64_t const prefixLen = views * mImageTokens + tokens + 1;
    ELLM_CHECK(prefixLen <= mMaxPrefix, "SmolvlaPolicy: prefix exceeds the engines' profile");

    // Pinned staging is rewritten only after the previous call drained the stream.
    int64_t const viewElems = static_cast<int64_t>(3) * mImageSize * mImageSize;
    for (int64_t v = 0; v < views; ++v)
    {
        std::vector<float> const pixels
            = preprocessView(ordered[v]->rgb, ordered[v]->height, ordered[v]->width, mImageSize);
        half* dst = mPixelsHost.dataPointer<half>() + v * viewElems;
        for (int64_t i = 0; i < viewElems; ++i)
        {
            dst[i] = __float2half(pixels[i]);
        }
    }
    std::copy(chunk.tokenIds.begin(), chunk.tokenIds.end(), mTokensHost.dataPointer<int64_t>());
    std::copy(state.begin(), state.end(), mStateHost.dataPointer<float>());
    if (noise.empty())
    {
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::generate_n(mNoiseHost.dataPointer<float>(), mChunk * mMaxActionDim, [&] { return normal(mNoiseGen); });
    }
    else
    {
        ELLM_CHECK(static_cast<int64_t>(noise.size()) == static_cast<int64_t>(mChunk) * mMaxActionDim,
            "SmolvlaPolicy: noise must be [chunk, maxActionDim]");
        std::copy(noise.begin(), noise.end(), mNoiseHost.dataPointer<float>());
    }
    CUDA_CHECK(cudaMemcpyAsync(mPixels.rawPointer(), mPixelsHost.rawPointer(),
        static_cast<size_t>(views * viewElems) * sizeof(half), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mTokens.rawPointer(), mTokensHost.rawPointer(),
        static_cast<size_t>(tokens) * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        mState.rawPointer(), mStateHost.rawPointer(), mMaxStateDim * sizeof(float), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mX[0].rawPointer(), mNoiseHost.rawPointer(),
        static_cast<size_t>(mChunk) * mMaxActionDim * sizeof(float), cudaMemcpyHostToDevice, mStream));

    CUDA_CHECK(cudaEventRecord(mEvents[0], mStream));
    mVisual.setShape("pixel_values", {views, 3, mImageSize, mImageSize});
    mVisual.bind("pixel_values", mPixels.rawPointer());
    mVisual.bind("image_features", mImageFeatures.rawPointer());
    ELLM_CHECK(mVisual.enqueue(mStream), "SmolvlaPolicy: visual enqueue failed");
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));

    mPrefix.setShape("image_features", {1, views * mImageTokens, mHidden});
    mPrefix.setShape("token_ids", {1, tokens});
    mPrefix.setShape("state", {1, mMaxStateDim});
    mPrefix.bind("image_features", mImageFeatures.rawPointer());
    mPrefix.bind("token_ids", mTokens.rawPointer());
    mPrefix.bind("state", mState.rawPointer());
    for (size_t i = 0; i < mKVNames.size(); ++i)
    {
        mPrefix.bind(mKVNames[i].c_str(), mKV[i].rawPointer());
    }
    ELLM_CHECK(mPrefix.enqueue(mStream), "SmolvlaPolicy: prefix enqueue failed");
    CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));

    auto const cached = mGraphs.find(prefixLen);
    if (mUseCudaGraph && cached != mGraphs.end())
    {
        CUDA_CHECK(cudaGraphLaunch(cached->second, mStream));
    }
    else
    {
        // TensorRT needs one regular enqueue for these shapes before its kernels can be captured.
        enqueueDenoiseLoop(prefixLen);
        if (mUseCudaGraph)
        {
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            cudaGraph_t graph{};
            CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
            enqueueDenoiseLoop(prefixLen);
            CUDA_CHECK(cudaStreamEndCapture(mStream, &graph));
            cudaGraphExec_t exec{};
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            mGraphs.emplace(prefixLen, exec);
        }
    }
    CUDA_CHECK(cudaEventRecord(mEvents[3], mStream));
    CUDA_CHECK(cudaMemcpyAsync(mOutHost.rawPointer(), mX[mNumSteps % 2].rawPointer(),
        static_cast<size_t>(mChunk) * mMaxActionDim * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    cudaEventElapsedTime(&chunk.visualMs, mEvents[0], mEvents[1]);
    cudaEventElapsedTime(&chunk.prefixMs, mEvents[1], mEvents[2]);
    cudaEventElapsedTime(&chunk.denoiseMs, mEvents[2], mEvents[3]);

    float const* out = mOutHost.dataPointer<float>();
    chunk.normalized.assign(out, out + static_cast<int64_t>(mChunk) * mMaxActionDim);
    chunk.robot.resize(static_cast<size_t>(mChunk) * mActionDim);
    for (int32_t t = 0; t < mChunk; ++t)
    {
        for (int32_t d = 0; d < mActionDim; ++d)
        {
            chunk.robot[static_cast<size_t>(t) * mActionDim + d]
                = out[static_cast<size_t>(t) * mMaxActionDim + d] * mActionStd[d] + mActionMean[d];
        }
    }
    return chunk;
}

} // namespace smolvla
} // namespace trt_edgellm
