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

#include "evo1Policy.h"

#include "common/checkMacros.h"

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>

namespace trt_edgellm
{
namespace evo1
{
namespace
{

using Json = nlohmann::json;
using nvinfer1::DataType;

constexpr int64_t kSlotBytes = 256;
// The official normalizer's eps in (x - min) / (max - min + eps).
constexpr float kBoundsEps = 1e-8F;

rt::Tensor makeTensor(
    std::vector<int64_t> const& shape, DataType type, char const* name, rt::DeviceType device = rt::DeviceType::kGPU)
{
    return rt::Tensor(rt::Coords(shape), device, type, name);
}

std::string strip(std::string const& text)
{
    auto const first = text.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string::npos)
    {
        return {};
    }
    return text.substr(first, text.find_last_not_of(" \t\n\r\f\v") - first + 1);
}

} // namespace

Evo1Policy::Evo1Policy(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    std::ifstream file(engineDir + "/config.json");
    ELLM_CHECK(file.good(), "Evo1Policy: missing " + engineDir + "/config.json");
    Json const config = Json::parse(file);
    ELLM_CHECK(config.at("model_family").get<std::string>() == "evo1", "Evo1Policy: not an Evo-1 engine directory");
    mCameras = config.at("cameras").get<std::vector<std::string>>();
    mMaxViews = config.at("max_views").get<int32_t>();
    mImageSize = config.at("image_size").get<int32_t>();
    mImageTokens = config.at("image_tokens").get<int32_t>();
    mTextLength = config.at("text_length").get<int32_t>();
    mPrefixIds = config.at("prefix_ids").get<std::vector<int32_t>>();
    mImageContextId = config.at("image_context_id").get<int32_t>();
    mPadId = config.at("pad_id").get<int32_t>();
    mSwapRedBlue = config.value("swap_red_blue", false);
    mHorizon = config.at("chunk_size").get<int32_t>();
    mMaxDim = config.at("max_dim").get<int32_t>();
    mTimeIndices = config.at("time_indices").get<std::vector<int32_t>>();
    mSteps = static_cast<int32_t>(mTimeIndices.size());
    mStateMin = config.at("state_min").get<std::vector<float>>();
    mStateMax = config.at("state_max").get<std::vector<float>>();
    mActionMin = config.at("action_min").get<std::vector<float>>();
    mActionMax = config.at("action_max").get<std::vector<float>>();
    ELLM_CHECK(static_cast<int32_t>(mCameras.size()) <= mMaxViews, "Evo1Policy: more cameras than views");
    ELLM_CHECK(stateDim() <= mMaxDim && actionDim() <= mMaxDim, "Evo1Policy: state / action wider than max_dim");
    auto const mean = config.at("image_mean").get<std::vector<float>>();
    auto const stdev = config.at("image_std").get<std::vector<float>>();
    for (int32_t c = 0; c < 3; ++c)
    {
        for (int32_t v = 0; v < 256; ++v)
        {
            mPixelLut[c][v] = (static_cast<float>(v) / 255.0F - mean[c]) / stdev[c];
        }
    }

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir), "Evo1Policy: failed to load the tokenizer");
    mRuntime = vla::createTrtRuntime();
    mVision = vla::TrtEngine(*mRuntime, engineDir + "/vision.engine", stream);
    mPrefix = vla::TrtEngine(*mRuntime, engineDir + "/prefix.engine", stream);
    mContext = vla::TrtEngine(*mRuntime, engineDir + "/context.engine", stream);
    mStep = vla::TrtEngine(*mRuntime, engineDir + "/step.engine", stream);
    mContextMemory = vla::allocateSharedContextMemory({&mVision, &mPrefix, &mContext, &mStep}, "evo1::contextMemory");
    for (vla::TrtEngine* engine : {&mVision, &mPrefix, &mContext, &mStep})
    {
        engine->setDeviceMemory(mContextMemory);
    }
    mHidden = static_cast<int32_t>(mPrefix.engine().getTensorShape("hidden").d[2]);
    nvinfer1::Dims const kvShape = mContext.engine().getTensorShape("kv_0");
    int64_t const kvElems = 2LL * kvShape.d[2] * kvShape.d[3] * kvShape.d[4];
    mBlocks = 0;
    while (mContext.engine().getTensorShape(("kv_" + std::to_string(mBlocks)).c_str()).nbDims > 0)
    {
        ++mBlocks;
    }

    int64_t const cameras = static_cast<int64_t>(mCameras.size());
    int64_t const pixelsPerView = 3LL * mImageSize * mImageSize;
    int64_t const chunkElems = static_cast<int64_t>(mHorizon) * mMaxDim;
    mPixelsHost = makeTensor({std::max<int64_t>(cameras, 1), 3, mImageSize, mImageSize}, DataType::kHALF,
        "evo1::pixelsHost", rt::DeviceType::kCPU);
    mPixels = makeTensor({std::max<int64_t>(cameras, 1), 3, mImageSize, mImageSize}, DataType::kHALF, "evo1::pixels");
    mVisual = makeTensor({static_cast<int64_t>(mMaxViews) * mImageTokens, mHidden}, DataType::kHALF, "evo1::visual");
    mPromptHost = makeTensor(
        {static_cast<int64_t>(mTextLength) * (4 + 2)}, DataType::kUINT8, "evo1::promptHost", rt::DeviceType::kCPU);
    mInputIds = makeTensor({1, mTextLength}, DataType::kINT32, "evo1::inputIds");
    mAttention = makeTensor({1, mTextLength}, DataType::kHALF, "evo1::attention");
    mHiddenState = makeTensor({1, mTextLength, mHidden}, DataType::kHALF, "evo1::hidden");
    mStateHost = makeTensor({1, mMaxDim}, DataType::kHALF, "evo1::stateHost", rt::DeviceType::kCPU);
    mState = makeTensor({1, mMaxDim}, DataType::kHALF, "evo1::state");
    mKv = makeTensor({mBlocks * kvElems}, DataType::kHALF, "evo1::kv");
    mX[0] = makeTensor({1, chunkElems}, DataType::kFLOAT, "evo1::x0");
    mX[1] = makeTensor({1, chunkElems}, DataType::kFLOAT, "evo1::x1");
    mScalars = makeTensor({(mSteps + 1) * kSlotBytes}, DataType::kUINT8, "evo1::scalars");
    mActionMask = makeTensor({mMaxDim}, DataType::kFLOAT, "evo1::actionMask");
    mStrength = makeTensor({mHorizon, 1}, DataType::kFLOAT, "evo1::strength");
    mStageHost = makeTensor({(mSteps + 1) * kSlotBytes + (chunkElems + mHorizon + mMaxDim) * 4}, DataType::kUINT8,
        "evo1::stage", rt::DeviceType::kCPU);
    mOutHost = makeTensor({chunkElems}, DataType::kFLOAT, "evo1::out", rt::DeviceType::kCPU);

    // Time indices (INT32) and dt (FP32), each in its own 256-byte slot: TensorRT wants aligned binding addresses.
    auto* slots = static_cast<char*>(mStageHost.rawPointer());
    std::memset(slots, 0, (mSteps + 1) * kSlotBytes);
    for (int32_t k = 0; k < mSteps; ++k)
    {
        *reinterpret_cast<int32_t*>(slots + k * kSlotBytes) = mTimeIndices[k];
    }
    *reinterpret_cast<float*>(slots + mSteps * kSlotBytes) = 1.0F / static_cast<float>(mSteps);
    CUDA_CHECK(
        cudaMemcpyAsync(mScalars.rawPointer(), slots, (mSteps + 1) * kSlotBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto* maskHost = static_cast<float*>(mStageHost.rawPointer());
    for (int32_t d = 0; d < mMaxDim; ++d)
    {
        maskHost[d] = d < actionDim() ? 1.0F : 0.0F;
    }
    CUDA_CHECK(cudaMemcpyAsync(mActionMask.rawPointer(), maskHost, mMaxDim * 4, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The black views' tokens: the vision engine on one black image, copied into every missing view's rows.
    size_t const viewTokenBytes = static_cast<size_t>(mImageTokens) * mHidden * sizeof(__half);
    if (cameras < mMaxViews)
    {
        auto* black = static_cast<__half*>(mPixelsHost.rawPointer());
        for (int32_t c = 0; c < 3; ++c)
        {
            std::fill_n(black + c * mImageSize * mImageSize, mImageSize * mImageSize, __float2half(mPixelLut[c][0]));
        }
        CUDA_CHECK(cudaMemcpyAsync(
            mPixels.rawPointer(), black, pixelsPerView * sizeof(__half), cudaMemcpyHostToDevice, stream));
        auto* first = static_cast<char*>(mVisual.rawPointer()) + cameras * viewTokenBytes;
        mVision.bind("pixels", mPixels.rawPointer());
        mVision.bind("image_tokens", first);
        mVision.setShape("pixels", {1, 3, mImageSize, mImageSize});
        ELLM_CHECK(mVision.enqueue(stream), "Evo1Policy: vision engine failed");
        for (int64_t v = cameras + 1; v < mMaxViews; ++v)
        {
            CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(mVisual.rawPointer()) + v * viewTokenBytes, first,
                viewTokenBytes, cudaMemcpyDeviceToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    mVision.bind("pixels", mPixels.rawPointer());
    mVision.bind("image_tokens", mVisual.rawPointer());
    mPrefix.bind("input_ids", mInputIds.rawPointer());
    mPrefix.bind("image_tokens", mVisual.rawPointer());
    mPrefix.bind("attention", mAttention.rawPointer());
    mPrefix.bind("hidden", mHiddenState.rawPointer());
    mContext.bind("hidden", mHiddenState.rawPointer());
    mContext.bind("state", mState.rawPointer());
    for (int32_t b = 0; b < mBlocks; ++b)
    {
        mContext.bind(("kv_" + std::to_string(b)).c_str(), static_cast<__half*>(mKv.rawPointer()) + b * kvElems);
    }
    for (cudaEvent_t& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

Evo1Policy::~Evo1Policy() noexcept
{
    if (mGraph != nullptr)
    {
        cudaGraphExecDestroy(mGraph);
    }
    for (cudaEvent_t event : mEvents)
    {
        if (event != nullptr)
        {
            cudaEventDestroy(event);
        }
    }
}

std::vector<int32_t> Evo1Policy::promptIds(std::string const& task) const
{
    // The image blocks end in "\n", which the byte-level BPE never merges with the instruction's first word, so the
    // instruction is tokenized on its own.
    std::vector<int32_t> ids = mPrefixIds;
    for (auto const id : mTokenizer->encode(strip(task)))
    {
        ids.push_back(static_cast<int32_t>(id));
    }
    return ids;
}

void Evo1Policy::stageFrame(int32_t view, unsigned char const* rgb, int32_t height, int32_t width)
{
    cv::Mat resized;
    unsigned char const* src = rgb;
    if (height != mImageSize || width != mImageSize)
    {
        cv::Mat const frame(height, width, CV_8UC3, const_cast<unsigned char*>(rgb));
        cv::resize(frame, resized, cv::Size(mImageSize, mImageSize), 0.0, 0.0, cv::INTER_LINEAR);
        src = resized.ptr<unsigned char>();
    }
    int64_t const plane = static_cast<int64_t>(mImageSize) * mImageSize;
    auto* dst = static_cast<__half*>(mPixelsHost.rawPointer()) + view * 3 * plane;
    for (int32_t c = 0; c < 3; ++c)
    {
        int32_t const in = mSwapRedBlue ? 2 - c : c;
        auto const& lut = mPixelLut[c];
        __half* out = dst + c * plane;
        for (int64_t p = 0; p < plane; ++p)
        {
            out[p] = __float2half(lut[src[p * 3 + in]]);
        }
    }
}

void Evo1Policy::enqueueDenoise()
{
    auto* scalars = static_cast<char*>(mScalars.rawPointer());
    nvinfer1::Dims const kvShape = mContext.engine().getTensorShape("kv_0");
    int64_t const kvElems = 2LL * kvShape.d[2] * kvShape.d[3] * kvShape.d[4];
    for (int32_t k = 0; k < mSteps; ++k)
    {
        mStep.bind("action", mX[k % 2].rawPointer());
        mStep.bind("time_index", scalars + k * kSlotBytes);
        mStep.bind("action_mask", mActionMask.rawPointer());
        mStep.bind("strength", mStrength.rawPointer());
        mStep.bind("dt", scalars + mSteps * kSlotBytes);
        for (int32_t b = 0; b < mBlocks; ++b)
        {
            mStep.bind(("kv_" + std::to_string(b)).c_str(), static_cast<__half*>(mKv.rawPointer()) + b * kvElems);
        }
        mStep.bind("action_next", mX[(k + 1) % 2].rawPointer());
        ELLM_CHECK(mStep.enqueue(mStream), "Evo1Policy: step engine failed");
    }
}

Evo1Chunk Evo1Policy::act(std::vector<Evo1View> const& views, std::vector<float> const& state, std::string const& task,
    std::vector<float> const& noise, Evo1Rtc const* rtc)
{
    auto const start = std::chrono::steady_clock::now();
    ELLM_CHECK(static_cast<int32_t>(state.size()) == stateDim(),
        "Evo1Policy: state needs " + std::to_string(stateDim()) + " values");
    Evo1Chunk chunk;
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    std::vector<std::future<void>> staging;
    for (size_t c = 0; c < mCameras.size(); ++c)
    {
        auto const it
            = std::find_if(views.begin(), views.end(), [&](Evo1View const& v) { return v.camera == mCameras[c]; });
        ELLM_CHECK(it != views.end(), "Evo1Policy: missing camera " + mCameras[c]);
        staging.push_back(std::async(std::launch::async,
            [this, it, c] { stageFrame(static_cast<int32_t>(c), it->rgb, it->height, it->width); }));
    }
    for (auto& f : staging)
    {
        f.get();
    }
    int64_t const cameras = static_cast<int64_t>(mCameras.size());
    CUDA_CHECK(cudaMemcpyAsync(mPixels.rawPointer(), mPixelsHost.rawPointer(),
        cameras * 3LL * mImageSize * mImageSize * sizeof(__half), cudaMemcpyHostToDevice, mStream));
    float const frameMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();

    CUDA_CHECK(cudaEventRecord(mEvents[0], mStream));
    mVision.setShape("pixels", {cameras, 3, mImageSize, mImageSize});
    ELLM_CHECK(mVision.enqueue(mStream), "Evo1Policy: vision engine failed");
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));

    // The prompt, the state and the initial sample are prepared while the vision engine runs.
    auto const promptStart = std::chrono::steady_clock::now();
    std::vector<int32_t> ids = promptIds(task);
    chunk.promptTokens = static_cast<int32_t>(std::min<size_t>(ids.size(), mTextLength));
    auto* hostIds = static_cast<int32_t*>(mPromptHost.rawPointer());
    auto* hostMask = reinterpret_cast<__half*>(hostIds + mTextLength);
    int64_t imageToken = 0;
    for (int32_t p = 0; p < mTextLength; ++p)
    {
        bool const real = p < chunk.promptTokens;
        hostIds[p] = real ? ids[p] : mPadId;
        bool attend = real;
        if (real && ids[p] == mImageContextId)
        {
            attend = imageToken++ / mImageTokens < cameras;
        }
        hostMask[p] = __float2half(attend ? 1.0F : 0.0F);
    }
    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), hostIds, mTextLength * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mAttention.rawPointer(), hostMask, mTextLength * 2, cudaMemcpyHostToDevice, mStream));

    auto* stateHost = static_cast<__half*>(mStateHost.rawPointer());
    for (int32_t d = 0; d < mMaxDim; ++d)
    {
        float value = 0.0F;
        if (d < stateDim())
        {
            value = std::clamp(
                2.0F * (state[d] - mStateMin[d]) / (mStateMax[d] - mStateMin[d] + kBoundsEps) - 1.0F, -1.0F, 1.0F);
        }
        stateHost[d] = __float2half(value);
    }
    CUDA_CHECK(cudaMemcpyAsync(mState.rawPointer(), stateHost, mMaxDim * 2, cudaMemcpyHostToDevice, mStream));

    int64_t const chunkElems = static_cast<int64_t>(mHorizon) * mMaxDim;
    auto* x0 = reinterpret_cast<float*>(static_cast<char*>(mStageHost.rawPointer()) + (mSteps + 1) * kSlotBytes);
    auto* strength = x0 + chunkElems;
    if (noise.empty())
    {
        std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
        std::generate(x0, x0 + chunkElems, [&] { return uniform(mNoiseGen); });
    }
    else
    {
        ELLM_CHECK(static_cast<int64_t>(noise.size()) == chunkElems, "Evo1Policy: noise must be [horizon, max dim]");
        std::copy(noise.begin(), noise.end(), x0);
    }
    std::fill_n(strength, mHorizon, 1.0F);
    if (rtc != nullptr && rtc->overlapSteps > 0 && !mPrevious.empty())
    {
        int32_t const startRow = rtc->startRow >= 0 ? rtc->startRow : mHorizon - rtc->overlapSteps;
        ELLM_CHECK(rtc->overlapSteps <= mHorizon && startRow >= 0 && startRow <= mHorizon && rtc->frozenSteps >= 0,
            "Evo1Policy: RTC overlap and start row must lie inside the chunk");
        int32_t const overlap = std::min(rtc->overlapSteps, mHorizon - startRow);
        int32_t const frozen = std::min(rtc->frozenSteps, overlap);
        std::copy_n(
            mPrevious.begin() + static_cast<int64_t>(startRow) * mMaxDim, static_cast<int64_t>(overlap) * mMaxDim, x0);
        double const last = std::max(1.0 - std::exp(-static_cast<double>(rtc->rampRate)), 1e-8);
        for (int32_t row = 0; row < overlap; ++row)
        {
            double const t = static_cast<double>(row - frozen + 1) / (overlap - frozen + 1);
            strength[row] = row < frozen ? 0.0F : static_cast<float>((1.0 - std::exp(-rtc->rampRate * t)) / last);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mX[0].rawPointer(), x0, chunkElems * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mStrength.rawPointer(), strength, mHorizon * 4, cudaMemcpyHostToDevice, mStream));
    chunk.hostMs
        = frameMs + std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - promptStart).count();

    ELLM_CHECK(mPrefix.enqueue(mStream), "Evo1Policy: prefix engine failed");
    CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));
    ELLM_CHECK(mContext.enqueue(mStream), "Evo1Policy: context engine failed");
    if (mUseCudaGraph && mGraph != nullptr)
    {
        CUDA_CHECK(cudaGraphLaunch(mGraph, mStream));
    }
    else
    {
        // TensorRT needs one regular enqueue before its kernels can be captured.
        enqueueDenoise();
        if (mUseCudaGraph)
        {
            CUDA_CHECK(cudaMemcpyAsync(mX[0].rawPointer(), x0, chunkElems * 4, cudaMemcpyHostToDevice, mStream));
            cudaGraph_t captured{};
            CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
            try
            {
                enqueueDenoise();
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
            CUDA_CHECK(cudaGraphInstantiate(&mGraph, captured, 0));
            CUDA_CHECK(cudaGraphDestroy(captured));
            CUDA_CHECK(cudaGraphLaunch(mGraph, mStream));
        }
    }
    CUDA_CHECK(cudaEventRecord(mEvents[3], mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        mOutHost.rawPointer(), mX[mSteps % 2].rawPointer(), chunkElems * 4, cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.visionMs, mEvents[0], mEvents[1]));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.prefixMs, mEvents[1], mEvents[2]));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.actionMs, mEvents[2], mEvents[3]));

    auto const* out = static_cast<float const*>(mOutHost.rawPointer());
    chunk.normalized.assign(out, out + chunkElems);
    for (int32_t t = 0; t < mHorizon; ++t)
    {
        std::fill(chunk.normalized.begin() + static_cast<int64_t>(t) * mMaxDim + actionDim(),
            chunk.normalized.begin() + static_cast<int64_t>(t + 1) * mMaxDim, 0.0F);
    }
    mPrevious = chunk.normalized;
    chunk.actions.resize(static_cast<size_t>(mHorizon) * actionDim());
    for (int32_t t = 0; t < mHorizon; ++t)
    {
        for (int32_t d = 0; d < actionDim(); ++d)
        {
            float const v = chunk.normalized[static_cast<size_t>(t) * mMaxDim + d];
            chunk.actions[static_cast<size_t>(t) * actionDim() + d]
                = (v + 1.0F) / 2.0F * (mActionMax[d] - mActionMin[d] + kBoundsEps) + mActionMin[d];
        }
    }
    return chunk;
}

} // namespace evo1
} // namespace trt_edgellm
