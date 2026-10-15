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

#include "xr1Policy.h"

#include "common/checkMacros.h"
#include "vlaImage.h"

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
namespace xr1
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

} // namespace

Xr1Policy::Xr1Policy(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    std::ifstream file(engineDir + "/config.json");
    ELLM_CHECK(file.good(), "Xr1Policy: missing " + engineDir + "/config.json");
    Json const config = Json::parse(file);
    ELLM_CHECK(config.at("model_family").get<std::string>() == "xiaomi_robotics_1",
        "Xr1Policy: not a Xiaomi-Robotics-1 engine directory");
    mCameras = config.at("cameras").get<std::vector<std::string>>();
    mImageSize = config.at("image_size").get<int32_t>();
    mCropRatio = config.at("crop_ratio").get<float>();
    mPatch = config.at("patch_size").get<int32_t>();
    mTemporal = config.at("temporal_patch_size").get<int32_t>();
    mMerge = config.at("merge_size").get<int32_t>();
    auto const mean = config.at("image_mean").get<std::vector<float>>();
    auto const stdev = config.at("image_std").get<std::vector<float>>();
    std::copy_n(mean.begin(), 3, mImageMean);
    std::copy_n(stdev.begin(), 3, mImageStd);
    mImageTokenId = config.at("image_token_id").get<int32_t>();
    mPromptHead = config.at("prompt_head_ids").get<std::vector<int32_t>>();
    mPromptTail = config.at("prompt_tail_ids").get<std::vector<int32_t>>();
    mLayers = config.at("layers").get<int32_t>();
    mKvHeads = config.at("kv_heads").get<int32_t>();
    mHeadDim = config.at("head_dim").get<int32_t>();
    mHorizon = config.at("chunk_size").get<int32_t>();
    mMaxDim = config.at("max_dim").get<int32_t>();
    mStateWidth = config.at("state_dim").get<int32_t>();
    mSteps = config.at("steps").get<int32_t>();
    mNoise = config.at("noise").get<std::vector<float>>();
    mActionMean = config.at("action_mean").get<std::vector<float>>();
    mActionStd = config.at("action_std").get<std::vector<float>>();
    mActionMask = config.at("action_mask").get<std::vector<float>>();
    int64_t const chunkElems = static_cast<int64_t>(mHorizon) * mMaxDim;
    ELLM_CHECK(
        static_cast<int64_t>(mNoise.size()) == chunkElems && static_cast<int64_t>(mActionMask.size()) == chunkElems,
        "Xr1Policy: noise and action mask must be [horizon, max dim]");
    mActionDim = static_cast<int32_t>(
        std::count_if(mActionMask.begin(), mActionMask.begin() + mMaxDim, [](float v) { return v > 0.5F; }));
    mStateDim = mStateWidth;

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir), "Xr1Policy: failed to load the tokenizer");
    mRuntime = vla::createTrtRuntime();
    mVision = vla::TrtEngine(*mRuntime, engineDir + "/vision.engine", stream);
    mPrefixA = vla::TrtEngine(*mRuntime, engineDir + "/prefix_a.engine", stream);
    mPrefixB = vla::TrtEngine(*mRuntime, engineDir + "/prefix_b.engine", stream);
    mStep = vla::TrtEngine(*mRuntime, engineDir + "/step.engine", stream);
    mContextMemory = vla::allocateSharedContextMemory({&mVision, &mPrefixA, &mPrefixB, &mStep}, "xr1::contextMemory");
    for (vla::TrtEngine* engine : {&mVision, &mPrefixA, &mPrefixB, &mStep})
    {
        engine->setDeviceMemory(mContextMemory);
    }
    mLayersA = static_cast<int32_t>(mPrefixA.engine().getTensorShape("keys").d[0]);
    mMaxPrompt = mPrefixA.engine().getProfileShape("input_ids", 0, nvinfer1::OptProfileSelector::kMAX).d[1];

    auto const shape = [](vla::TrtEngine const& e, char const* name) {
        nvinfer1::Dims const d = e.engine().getTensorShape(name);
        return std::vector<int64_t>(d.d, d.d + d.nbDims);
    };
    std::vector<int64_t> const pixels = shape(mVision, "pixels");
    int64_t const hidden = mPrefixA.engine().getTensorShape("hidden_out").d[2];
    int64_t const L = mMaxPrompt;
    mPixelsHost = makeTensor(pixels, DataType::kHALF, "xr1::pixelsHost", rt::DeviceType::kCPU);
    mPixels = makeTensor(pixels, DataType::kHALF, "xr1::pixels");
    mImageTokens = makeTensor(shape(mVision, "image_tokens"), DataType::kHALF, "xr1::imageTokens");
    mDeepstack = makeTensor(shape(mVision, "deepstack"), DataType::kHALF, "xr1::deepstack");
    mPromptHost = makeTensor(
        {L * 4 + 3 * L * 4 + mStateWidth * 2 + L * 2 + 64}, DataType::kUINT8, "xr1::promptHost", rt::DeviceType::kCPU);
    mInputIds = makeTensor({1, L}, DataType::kINT32, "xr1::inputIds");
    mPositionIds = makeTensor({3, L}, DataType::kINT32, "xr1::positionIds");
    mState = makeTensor({1, 1, mStateWidth}, DataType::kHALF, "xr1::state");
    mPrefixMask = makeTensor({1, L}, DataType::kHALF, "xr1::prefixMask");
    mHidden = makeTensor({1, L, hidden}, DataType::kHALF, "xr1::hidden");
    mHiddenOut = makeTensor({1, L, hidden}, DataType::kHALF, "xr1::hiddenOut");
    mKeys = makeTensor({mLayers, 1, L, mKvHeads, mHeadDim}, DataType::kHALF, "xr1::keys");
    mValues = makeTensor({mLayers, 1, L, mKvHeads, mHeadDim}, DataType::kHALF, "xr1::values");
    for (auto& x : mX)
    {
        x = makeTensor({1, mHorizon, mMaxDim}, DataType::kFLOAT, "xr1::x");
    }
    mNoiseDevice = makeTensor({1, mHorizon, mMaxDim}, DataType::kFLOAT, "xr1::noise");
    mSeed = makeTensor({1, mHorizon, mMaxDim}, DataType::kFLOAT, "xr1::seed");
    mWeight = makeTensor({mHorizon, 1}, DataType::kFLOAT, "xr1::weight");
    mActionMaskDevice = makeTensor({1, mHorizon, mMaxDim}, DataType::kFLOAT, "xr1::actionMask");
    mScalars = makeTensor({(mSteps + 2) * kSlotBytes}, DataType::kUINT8, "xr1::scalars");
    mStageHost = makeTensor({(mSteps + 2) * kSlotBytes + (3 * chunkElems + mHorizon) * 4}, DataType::kUINT8,
        "xr1::stage", rt::DeviceType::kCPU);
    mOutHost = makeTensor({chunkElems}, DataType::kFLOAT, "xr1::out", rt::DeviceType::kCPU);

    // Step indices (INT32), the suffix start (INT32, per call) and dt (FP32), one aligned slot each.
    auto* slots = static_cast<char*>(mStageHost.rawPointer());
    std::memset(slots, 0, (mSteps + 2) * kSlotBytes);
    for (int32_t k = 0; k < mSteps; ++k)
    {
        *reinterpret_cast<int32_t*>(slots + k * kSlotBytes) = k;
    }
    *reinterpret_cast<float*>(slots + (mSteps + 1) * kSlotBytes) = 1.0F / static_cast<float>(mSteps);
    CUDA_CHECK(
        cudaMemcpyAsync(mScalars.rawPointer(), slots, (mSteps + 2) * kSlotBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mActionMaskDevice.rawPointer(), mActionMask.data(), chunkElems * 4, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    mVision.bind("pixels", mPixels.rawPointer());
    mVision.bind("image_tokens", mImageTokens.rawPointer());
    mVision.bind("deepstack", mDeepstack.rawPointer());
    mPrefixA.bind("input_ids", mInputIds.rawPointer());
    mPrefixA.bind("position_ids", mPositionIds.rawPointer());
    mPrefixA.bind("image_tokens", mImageTokens.rawPointer());
    mPrefixA.bind("deepstack", mDeepstack.rawPointer());
    mPrefixA.bind("hidden_out", mHidden.rawPointer());
    mPrefixA.bind("keys", mKeys.rawPointer());
    mPrefixA.bind("values", mValues.rawPointer());
    mPrefixB.bind("hidden", mHidden.rawPointer());
    mPrefixB.bind("position_ids", mPositionIds.rawPointer());
    mPrefixB.bind("hidden_out", mHiddenOut.rawPointer());
    for (cudaEvent_t& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

Xr1Policy::~Xr1Policy() noexcept
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

std::vector<int32_t> Xr1Policy::promptIds(std::string const& task) const
{
    // The chat template's ids around the instruction were checked at export to tokenize the same joined or split.
    std::vector<int32_t> ids = mPromptHead;
    for (auto const id : mTokenizer->encode(task))
    {
        ids.push_back(static_cast<int32_t>(id));
    }
    ids.insert(ids.end(), mPromptTail.begin(), mPromptTail.end());
    return ids;
}

std::vector<int32_t> Xr1Policy::positionIds(std::vector<int32_t> const& ids) const
{
    // Qwen3-VL get_rope_index for single-frame images: text tokens take consecutive positions on all three axes;
    // an image's merged grid sits at (p, p + row, p + col) from the next free position p, after which positions
    // continue past its largest index.
    int32_t const side = mImageSize / mPatch / mMerge;
    size_t const n = ids.size();
    std::vector<int32_t> out(3 * n);
    int32_t next = 0;
    size_t i = 0;
    while (i < n)
    {
        if (ids[i] == mImageTokenId)
        {
            int32_t const offset = next;
            for (int32_t r = 0; r < side; ++r)
            {
                for (int32_t c = 0; c < side; ++c, ++i)
                {
                    out[i] = offset;
                    out[n + i] = offset + r;
                    out[2 * n + i] = offset + c;
                }
            }
            next = offset + side;
            continue;
        }
        out[i] = out[n + i] = out[2 * n + i] = next++;
        ++i;
    }
    return out;
}

std::vector<float> Xr1Policy::patches(unsigned char const* rgb, int32_t height, int32_t width) const
{
    int32_t const S = mImageSize;
    std::vector<unsigned char> image;
    if (height != S || width != S)
    {
        // Frames off the training size go through the slow processor's own resample first.
        image = vla::resizeBicubicPil(rgb, height, width, S, S);
        rgb = image.data();
    }
    // eval_robocasa's center_crop_pil: an integer crop, then PIL's bilinear resize back to the frame size.
    auto const cropSize = static_cast<int32_t>(static_cast<double>(S) * mCropRatio);
    std::vector<unsigned char> view(rgb, rgb + static_cast<size_t>(S) * S * 3);
    if (cropSize < S)
    {
        int32_t const offset = (S - cropSize) / 2;
        std::vector<unsigned char> crop(static_cast<size_t>(cropSize) * cropSize * 3);
        for (int32_t y = 0; y < cropSize; ++y)
        {
            std::memcpy(crop.data() + static_cast<size_t>(y) * cropSize * 3,
                rgb + (static_cast<size_t>(y + offset) * S + offset) * 3, static_cast<size_t>(cropSize) * 3);
        }
        view = vla::resizeBilinearPil(crop.data(), cropSize, cropSize, S, S);
    }
    int32_t const grid = S / mPatch;
    int32_t const blocks = grid / mMerge;
    int32_t const patchDim = 3 * mTemporal * mPatch * mPatch;
    std::vector<float> out(static_cast<size_t>(grid) * grid * patchDim);
    size_t row = 0;
    for (int32_t bh = 0; bh < blocks; ++bh)
    {
        for (int32_t bw = 0; bw < blocks; ++bw)
        {
            for (int32_t mh = 0; mh < mMerge; ++mh)
            {
                for (int32_t mw = 0; mw < mMerge; ++mw, ++row)
                {
                    int32_t const y0 = (bh * mMerge + mh) * mPatch;
                    int32_t const x0 = (bw * mMerge + mw) * mPatch;
                    float* dst = out.data() + row * patchDim;
                    for (int32_t c = 0; c < 3; ++c)
                    {
                        for (int32_t t = 0; t < mTemporal; ++t)
                        {
                            for (int32_t py = 0; py < mPatch; ++py)
                            {
                                for (int32_t px = 0; px < mPatch; ++px)
                                {
                                    // The slow processor: rescale in float64, cast to float32, normalize in float32.
                                    auto const v = static_cast<float>(
                                        view[(static_cast<size_t>(y0 + py) * S + x0 + px) * 3 + c] * (1.0 / 255.0));
                                    *dst++ = (v - mImageMean[c]) / mImageStd[c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

void Xr1Policy::enqueueDenoise(int64_t length)
{
    auto* scalars = static_cast<char*>(mScalars.rawPointer());
    for (int32_t k = 0; k < mSteps; ++k)
    {
        mStep.bind("x", mX[k % 2].rawPointer());
        mStep.bind("step", scalars + k * kSlotBytes);
        mStep.bind("state", mState.rawPointer());
        mStep.bind("prefix_keys", mKeys.rawPointer());
        mStep.bind("prefix_values", mValues.rawPointer());
        mStep.bind("prefix_mask", mPrefixMask.rawPointer());
        mStep.bind("suffix_start", scalars + mSteps * kSlotBytes);
        mStep.bind("action_mask", mActionMaskDevice.rawPointer());
        mStep.bind("dt", scalars + (mSteps + 1) * kSlotBytes);
        mStep.bind("noise", mNoiseDevice.rawPointer());
        mStep.bind("seed", mSeed.rawPointer());
        mStep.bind("weight", mWeight.rawPointer());
        mStep.bind("x_next", mX[(k + 1) % 2].rawPointer());
        mStep.setShape("prefix_keys", {mLayers, 1, length, mKvHeads, mHeadDim});
        mStep.setShape("prefix_values", {mLayers, 1, length, mKvHeads, mHeadDim});
        mStep.setShape("prefix_mask", {1, length});
        ELLM_CHECK(mStep.enqueue(mStream), "Xr1Policy: step engine failed");
    }
}

Xr1Chunk Xr1Policy::act(std::vector<Xr1View> const& views, std::vector<float> const& state, std::string const& task,
    std::vector<float> const& noise, Xr1Rtc const* rtc)
{
    auto const start = std::chrono::steady_clock::now();
    ELLM_CHECK(static_cast<int32_t>(state.size()) <= mStateWidth,
        "Xr1Policy: state wider than " + std::to_string(mStateWidth));
    Xr1Chunk chunk;
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    size_t const viewElems = mPixelsHost.getShape().volume() / mCameras.size();
    std::vector<std::future<void>> staging;
    for (size_t c = 0; c < mCameras.size(); ++c)
    {
        auto const it
            = std::find_if(views.begin(), views.end(), [&](Xr1View const& v) { return v.camera == mCameras[c]; });
        ELLM_CHECK(it != views.end(), "Xr1Policy: missing camera " + mCameras[c]);
        auto* dst = static_cast<__half*>(mPixelsHost.rawPointer()) + c * viewElems;
        staging.push_back(std::async(std::launch::async, [this, it, dst, viewElems] {
            std::vector<float> const p = patches(it->rgb, it->height, it->width);
            ELLM_CHECK(p.size() == viewElems, "Xr1Policy: the vision engine's patch count differs");
            std::transform(p.begin(), p.end(), dst, [](float v) { return __float2half(v); });
        }));
    }
    for (auto& f : staging)
    {
        f.get();
    }
    CUDA_CHECK(cudaMemcpyAsync(mPixels.rawPointer(), mPixelsHost.rawPointer(), mPixelsHost.getMemoryCapacity(),
        cudaMemcpyHostToDevice, mStream));
    float const frameMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    CUDA_CHECK(cudaEventRecord(mEvents[0], mStream));
    ELLM_CHECK(mVision.enqueue(mStream), "Xr1Policy: vision engine failed");
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));

    auto const promptStart = std::chrono::steady_clock::now();
    std::vector<int32_t> const ids = promptIds(task);
    auto const length = static_cast<int64_t>(ids.size());
    ELLM_CHECK(length <= mMaxPrompt, "Xr1Policy: the prompt exceeds the engines' profile");
    chunk.promptTokens = static_cast<int32_t>(length);
    std::vector<int32_t> const positions = positionIds(ids);
    auto* hostIds = static_cast<int32_t*>(mPromptHost.rawPointer());
    auto* hostPositions = hostIds + mMaxPrompt;
    auto* hostState = reinterpret_cast<__half*>(hostPositions + 3 * mMaxPrompt);
    auto* hostMask = hostState + mStateWidth;
    std::copy(ids.begin(), ids.end(), hostIds);
    std::copy(positions.begin(), positions.end(), hostPositions);
    for (int32_t d = 0; d < mStateWidth; ++d)
    {
        hostState[d] = __float2half(d < static_cast<int32_t>(state.size()) ? state[d] : 0.0F);
    }
    std::fill_n(hostMask, length, __float2half(1.0F));
    int32_t const suffixStart = *std::max_element(positions.begin(), positions.end()) + 1;
    auto* slots = static_cast<char*>(mStageHost.rawPointer());
    *reinterpret_cast<int32_t*>(slots + mSteps * kSlotBytes) = suffixStart;
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(mScalars.rawPointer()) + mSteps * kSlotBytes,
        slots + mSteps * kSlotBytes, 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), hostIds, length * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpy2DAsync(mPositionIds.rawPointer(), length * 4, hostPositions, length * 4, length * 4, 3,
        cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mState.rawPointer(), hostState, mStateWidth * 2, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mPrefixMask.rawPointer(), hostMask, length * 2, cudaMemcpyHostToDevice, mStream));

    int64_t const chunkElems = static_cast<int64_t>(mHorizon) * mMaxDim;
    auto* x0 = reinterpret_cast<float*>(slots + (mSteps + 2) * kSlotBytes);
    auto* seed = x0 + chunkElems;
    auto* weight = seed + chunkElems;
    if (noise.empty())
    {
        std::copy(mNoise.begin(), mNoise.end(), x0);
    }
    else
    {
        ELLM_CHECK(static_cast<int64_t>(noise.size()) == chunkElems, "Xr1Policy: noise must be [horizon, max dim]");
        std::copy(noise.begin(), noise.end(), x0);
    }
    std::fill_n(seed, chunkElems, 0.0F);
    std::fill_n(weight, mHorizon, 0.0F);
    if (rtc != nullptr && rtc->overlapSteps > 0 && !mPrevious.empty())
    {
        int32_t const startRow = rtc->startRow >= 0 ? rtc->startRow : mHorizon - rtc->overlapSteps;
        ELLM_CHECK(rtc->overlapSteps <= mHorizon && startRow >= 0 && startRow <= mHorizon && rtc->frozenSteps >= 0,
            "Xr1Policy: RTC overlap and start row must lie inside the chunk");
        int32_t const overlap = std::min(rtc->overlapSteps, mHorizon - startRow);
        int32_t const frozen = std::min(rtc->frozenSteps, overlap);
        std::copy_n(mPrevious.begin() + static_cast<int64_t>(startRow) * mMaxDim,
            static_cast<int64_t>(overlap) * mMaxDim, seed);
        // weight = 1 - ramp, with GR00T's ramp 1 - exp(-rate * u), normalized, over the rows after the frozen ones.
        double const last = std::max(1.0 - std::exp(-static_cast<double>(rtc->rampRate)), 1e-8);
        for (int32_t row = 0; row < overlap; ++row)
        {
            double ramp = 0.0;
            if (row >= frozen)
            {
                double const u = static_cast<double>(row - frozen + 1) / (overlap - frozen + 1);
                ramp = (1.0 - std::exp(-rtc->rampRate * u)) / last;
            }
            weight[row] = static_cast<float>(1.0 - ramp);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mX[0].rawPointer(), x0, chunkElems * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mNoiseDevice.rawPointer(), x0, chunkElems * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mSeed.rawPointer(), seed, chunkElems * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mWeight.rawPointer(), weight, mHorizon * 4, cudaMemcpyHostToDevice, mStream));
    chunk.hostMs
        = frameMs + std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - promptStart).count();

    size_t const halfKv = static_cast<size_t>(mLayersA) * length * mKvHeads * mHeadDim;
    mPrefixB.bind("keys", static_cast<__half*>(mKeys.rawPointer()) + halfKv);
    mPrefixB.bind("values", static_cast<__half*>(mValues.rawPointer()) + halfKv);
    mPrefixA.setShape("input_ids", {1, length});
    mPrefixA.setShape("position_ids", {3, length});
    ELLM_CHECK(mPrefixA.enqueue(mStream), "Xr1Policy: prefix_a engine failed");
    mPrefixB.setShape("hidden", {1, length, mHidden.getShape()[2]});
    mPrefixB.setShape("position_ids", {3, length});
    ELLM_CHECK(mPrefixB.enqueue(mStream), "Xr1Policy: prefix_b engine failed");
    CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));

    auto const graph = mGraphs.find(length);
    if (mUseCudaGraph && graph != mGraphs.end())
    {
        CUDA_CHECK(cudaGraphLaunch(graph->second, mStream));
    }
    else
    {
        // TensorRT needs one regular enqueue for these shapes before its kernels can be captured.
        enqueueDenoise(length);
        if (mUseCudaGraph)
        {
            CUDA_CHECK(cudaMemcpyAsync(mX[0].rawPointer(), x0, chunkElems * 4, cudaMemcpyHostToDevice, mStream));
            cudaGraph_t captured{};
            CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
            try
            {
                enqueueDenoise(length);
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
            mGraphs.emplace(length, exec);
            CUDA_CHECK(cudaGraphLaunch(exec, mStream));
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
    mPrevious = chunk.normalized;
    chunk.actions.resize(static_cast<size_t>(mHorizon) * mActionDim);
    for (int32_t t = 0; t < mHorizon; ++t)
    {
        for (int32_t d = 0; d < mActionDim; ++d)
        {
            size_t const i = static_cast<size_t>(t) * mMaxDim + d;
            chunk.actions[static_cast<size_t>(t) * mActionDim + d] = out[i] * mActionStd[i] + mActionMean[i];
        }
    }
    return chunk;
}

} // namespace xr1
} // namespace trt_edgellm
