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

#include "lingbotPolicy.h"

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
namespace lingbot
{
namespace
{

using Json = nlohmann::json;
using nvinfer1::DataType;

constexpr int64_t kSlotBytes = 256;
// The bounds normalizer's eps in (x - low) / (high - low + eps).
constexpr float kBoundsEps = 1e-6F;

rt::Tensor makeTensor(
    std::vector<int64_t> const& shape, DataType type, char const* name, rt::DeviceType device = rt::DeviceType::kGPU)
{
    return rt::Tensor(rt::Coords(shape), device, type, name);
}

std::vector<LingbotFeature> readLayout(Json const& entries)
{
    std::vector<LingbotFeature> layout;
    for (auto const& e : entries)
    {
        LingbotFeature f;
        f.start = e.at("start").get<int32_t>();
        f.count = e.at("count").get<int32_t>();
        f.slot = e.at("slot").get<int32_t>();
        f.low = e.at("low").get<std::vector<float>>();
        f.high = e.at("high").get<std::vector<float>>();
        f.clip = e.value("clip", false);
        layout.push_back(std::move(f));
    }
    return layout;
}

int32_t robotWidth(std::vector<LingbotFeature> const& layout)
{
    int32_t width = 0;
    for (auto const& f : layout)
    {
        width = std::max(width, f.start + f.count);
    }
    return width;
}

} // namespace

LingbotPolicy::LingbotPolicy(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    std::ifstream file(engineDir + "/config.json");
    ELLM_CHECK(file.good(), "LingbotPolicy: missing " + engineDir + "/config.json");
    Json const config = Json::parse(file);
    ELLM_CHECK(config.at("model_family").get<std::string>() == "lingbot_vla_v2",
        "LingbotPolicy: not a LingBot-VLA 2.0 engine directory");
    mCameras = config.at("cameras").get<std::vector<std::string>>();
    mImageSize = config.at("image_size").get<int32_t>();
    mPatch = config.at("patch_size").get<int32_t>();
    mTemporal = config.at("temporal_patch_size").get<int32_t>();
    mMerge = config.at("merge_size").get<int32_t>();
    auto const mean = config.at("image_mean").get<std::vector<float>>();
    auto const stdev = config.at("image_std").get<std::vector<float>>();
    for (int32_t c = 0; c < 3; ++c)
    {
        // The fast processor folds the 1 / 255 rescale into the mean and std: (x - 255 * mean) / (255 * std).
        mImageMean[c] = mean[c] * 255.0F;
        mImageStd[c] = stdev[c] * 255.0F;
    }
    mPrefixLength = config.at("prefix_length").get<int32_t>();
    mTextLength = config.at("text_length").get<int32_t>();
    mAlignTokens = config.at("align_tokens").get<int32_t>();
    mPadId = config.at("pad_id").get<int32_t>();
    mPromptPrefix = config.at("prompt_prefix_ids").get<std::vector<int32_t>>();
    mPromptSuffix = config.at("prompt_suffix_ids").get<std::vector<int32_t>>();
    mLayers = config.at("layers").get<int32_t>();
    mKvHeads = config.at("kv_heads").get<int32_t>();
    mHeadDim = config.at("head_dim").get<int32_t>();
    mHorizon = config.at("chunk_size").get<int32_t>();
    mMaxDim = config.at("max_dim").get<int32_t>();
    mSteps = config.at("steps").get<int32_t>();
    mDt = config.at("dt").get<float>();
    mStateLayout = readLayout(config.at("state"));
    mActionLayout = readLayout(config.at("actions"));
    mStateDim = robotWidth(mStateLayout);
    mActionDim = robotWidth(mActionLayout);

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir), "LingbotPolicy: failed to load the tokenizer");
    mRuntime = vla::createTrtRuntime();
    mVision = vla::TrtEngine(*mRuntime, engineDir + "/vision.engine", stream);
    mPrefixA = vla::TrtEngine(*mRuntime, engineDir + "/prefix_a.engine", stream);
    mPrefixB = vla::TrtEngine(*mRuntime, engineDir + "/prefix_b.engine", stream);
    mStep = vla::TrtEngine(*mRuntime, engineDir + "/step.engine", stream);
    mContextMemory
        = vla::allocateSharedContextMemory({&mVision, &mPrefixA, &mPrefixB, &mStep}, "lingbot::contextMemory");
    for (vla::TrtEngine* engine : {&mVision, &mPrefixA, &mPrefixB, &mStep})
    {
        engine->setDeviceMemory(mContextMemory);
    }
    mLayersA = static_cast<int32_t>(mPrefixA.engine().getTensorShape("keys").d[0]);

    auto const shape = [](vla::TrtEngine const& e, char const* name) {
        nvinfer1::Dims const d = e.engine().getTensorShape(name);
        return std::vector<int64_t>(d.d, d.d + d.nbDims);
    };
    std::vector<int64_t> const pixels = shape(mVision, "pixels");
    std::vector<int64_t> const hidden = shape(mPrefixA, "hidden_out");
    int64_t const chunkElems = static_cast<int64_t>(mHorizon) * mMaxDim;
    int64_t const P = mPrefixLength;
    mPixelsHost = makeTensor(pixels, DataType::kHALF, "lingbot::pixelsHost", rt::DeviceType::kCPU);
    mPixels = makeTensor(pixels, DataType::kHALF, "lingbot::pixels");
    mImageTokens = makeTensor(shape(mVision, "image_tokens"), DataType::kHALF, "lingbot::imageTokens");
    mDeepstack = makeTensor(shape(mVision, "deepstack"), DataType::kHALF, "lingbot::deepstack");
    mPromptHost = makeTensor({mTextLength * 4 + 3 * P * 4 + mTextLength * 2 + P * 2 + mMaxDim * 2 + 64},
        DataType::kUINT8, "lingbot::promptHost", rt::DeviceType::kCPU);
    mInputIds = makeTensor({1, mTextLength}, DataType::kINT32, "lingbot::inputIds");
    mLangMask = makeTensor({1, mTextLength}, DataType::kHALF, "lingbot::langMask");
    mPositionIds = makeTensor({3, P}, DataType::kINT32, "lingbot::positionIds");
    mPrefixMask = makeTensor({1, P}, DataType::kHALF, "lingbot::prefixMask");
    mState = makeTensor({1, mMaxDim}, DataType::kHALF, "lingbot::state");
    mHidden = makeTensor(hidden, DataType::kHALF, "lingbot::hidden");
    mHiddenOut = makeTensor(hidden, DataType::kHALF, "lingbot::hiddenOut");
    mKeys = makeTensor({mLayers, 1, P, mKvHeads, mHeadDim}, DataType::kHALF, "lingbot::keys");
    mValues = makeTensor({mLayers, 1, P, mKvHeads, mHeadDim}, DataType::kHALF, "lingbot::values");
    mX[0] = makeTensor({1, mHorizon, mMaxDim}, DataType::kFLOAT, "lingbot::x0");
    mX[1] = makeTensor({1, mHorizon, mMaxDim}, DataType::kFLOAT, "lingbot::x1");
    mNoise = makeTensor({1, mHorizon, mMaxDim}, DataType::kFLOAT, "lingbot::noise");
    mSeed = makeTensor({1, mHorizon, mMaxDim}, DataType::kFLOAT, "lingbot::seed");
    mWeight = makeTensor({mHorizon, 1}, DataType::kFLOAT, "lingbot::weight");
    mScalars = makeTensor({(mSteps + 2) * kSlotBytes}, DataType::kUINT8, "lingbot::scalars");
    mStageHost = makeTensor({(mSteps + 2) * kSlotBytes + (3 * chunkElems + mHorizon) * 4}, DataType::kUINT8,
        "lingbot::stage", rt::DeviceType::kCPU);
    mOutHost = makeTensor({chunkElems}, DataType::kFLOAT, "lingbot::out", rt::DeviceType::kCPU);

    // Step indices (INT32), then the suffix start (INT32, written per call) and dt (FP32), one aligned slot each.
    auto* slots = static_cast<char*>(mStageHost.rawPointer());
    std::memset(slots, 0, (mSteps + 2) * kSlotBytes);
    for (int32_t k = 0; k < mSteps; ++k)
    {
        *reinterpret_cast<int32_t*>(slots + k * kSlotBytes) = k;
    }
    *reinterpret_cast<float*>(slots + (mSteps + 1) * kSlotBytes) = mDt;
    CUDA_CHECK(
        cudaMemcpyAsync(mScalars.rawPointer(), slots, (mSteps + 2) * kSlotBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t const halfKv = static_cast<size_t>(mLayersA) * P * mKvHeads * mHeadDim;
    mVision.bind("pixels", mPixels.rawPointer());
    mVision.bind("image_tokens", mImageTokens.rawPointer());
    mVision.bind("deepstack", mDeepstack.rawPointer());
    mPrefixA.bind("input_ids", mInputIds.rawPointer());
    mPrefixA.bind("lang_mask", mLangMask.rawPointer());
    mPrefixA.bind("position_ids", mPositionIds.rawPointer());
    mPrefixA.bind("image_tokens", mImageTokens.rawPointer());
    mPrefixA.bind("deepstack", mDeepstack.rawPointer());
    mPrefixA.bind("hidden_out", mHidden.rawPointer());
    mPrefixA.bind("keys", mKeys.rawPointer());
    mPrefixA.bind("values", mValues.rawPointer());
    mPrefixB.bind("hidden", mHidden.rawPointer());
    mPrefixB.bind("lang_mask", mLangMask.rawPointer());
    mPrefixB.bind("position_ids", mPositionIds.rawPointer());
    mPrefixB.bind("hidden_out", mHiddenOut.rawPointer());
    mPrefixB.bind("keys", static_cast<__half*>(mKeys.rawPointer()) + halfKv);
    mPrefixB.bind("values", static_cast<__half*>(mValues.rawPointer()) + halfKv);
    for (cudaEvent_t& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

LingbotPolicy::~LingbotPolicy() noexcept
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

std::vector<int32_t> LingbotPolicy::promptIds(std::string const& task) const
{
    // The template's ids around the instruction were checked at export to tokenize the same joined or split.
    std::vector<int32_t> ids = mPromptPrefix;
    for (auto const id : mTokenizer->encode(task))
    {
        ids.push_back(static_cast<int32_t>(id));
    }
    ids.insert(ids.end(), mPromptSuffix.begin(), mPromptSuffix.end());
    return ids;
}

std::vector<int32_t> LingbotPolicy::positionIds(int32_t validText) const
{
    // Qwen3-VL get_rope_index over the unmasked tokens: a vision-start token, the image's merged grid at
    // (p + 1, p + 1 + row, p + 1 + col), the vision-end token after the grid's largest index; then the valid
    // instruction tokens and the alignment queries consecutively. Padding keeps position 1.
    int32_t const P = mPrefixLength;
    std::vector<int32_t> ids(static_cast<size_t>(3) * P, 1);
    auto set = [&](int32_t index, int32_t t, int32_t h, int32_t w) {
        ids[index] = t;
        ids[P + index] = h;
        ids[2 * P + index] = w;
    };
    int32_t const side = mImageSize / mPatch / mMerge;
    int32_t index = 0;
    int32_t next = 0;
    for (size_t c = 0; c < mCameras.size(); ++c)
    {
        set(index++, next, next, next);
        int32_t const offset = next + 1;
        for (int32_t r = 0; r < side; ++r)
        {
            for (int32_t col = 0; col < side; ++col)
            {
                set(index++, offset, offset + r, offset + col);
            }
        }
        next = offset + side;
        set(index++, next, next, next);
        ++next;
    }
    for (int32_t t = 0; t < mTextLength; ++t, ++index)
    {
        if (t < validText)
        {
            set(index, next, next, next);
            ++next;
        }
    }
    for (int32_t a = 0; a < mAlignTokens; ++a, ++index, ++next)
    {
        set(index, next, next, next);
    }
    return ids;
}

std::vector<float> LingbotPolicy::patches(unsigned char const* rgb, int32_t height, int32_t width) const
{
    int32_t const S = mImageSize;
    std::vector<float> image = vla::resizeBilinearAntialias(rgb, height, width, S, S);
    // FeatureTransform treats a float frame peaking at 2 or less as [0, 1] and brings it back to 8 bits.
    float const peak = *std::max_element(image.begin(), image.end());
    if (peak <= 2.0F)
    {
        for (float& v : image)
        {
            v = std::clamp(std::nearbyint(v * 255.0F), 0.0F, 255.0F);
        }
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
                                    float const v = image[(static_cast<size_t>(y0 + py) * S + x0 + px) * 3 + c];
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

void LingbotPolicy::enqueueDenoise()
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
        mStep.bind("dt", scalars + (mSteps + 1) * kSlotBytes);
        mStep.bind("noise", mNoise.rawPointer());
        mStep.bind("seed", mSeed.rawPointer());
        mStep.bind("weight", mWeight.rawPointer());
        mStep.bind("x_next", mX[(k + 1) % 2].rawPointer());
        ELLM_CHECK(mStep.enqueue(mStream), "LingbotPolicy: step engine failed");
    }
}

LingbotChunk LingbotPolicy::act(std::vector<LingbotView> const& views, std::vector<float> const& state,
    std::string const& task, std::vector<float> const& noise, LingbotRtc const* rtc)
{
    auto const start = std::chrono::steady_clock::now();
    ELLM_CHECK(static_cast<int32_t>(state.size()) >= mStateDim,
        "LingbotPolicy: state needs " + std::to_string(mStateDim) + " values");
    LingbotChunk chunk;
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    size_t const cameraElems = mPixelsHost.getShape().volume() / mCameras.size();
    std::vector<std::future<void>> staging;
    for (size_t c = 0; c < mCameras.size(); ++c)
    {
        auto const it
            = std::find_if(views.begin(), views.end(), [&](LingbotView const& v) { return v.camera == mCameras[c]; });
        ELLM_CHECK(it != views.end(), "LingbotPolicy: missing camera " + mCameras[c]);
        auto* dst = static_cast<__half*>(mPixelsHost.rawPointer()) + c * cameraElems;
        staging.push_back(std::async(std::launch::async, [this, it, dst, cameraElems] {
            std::vector<float> const p = patches(it->rgb, it->height, it->width);
            ELLM_CHECK(p.size() == cameraElems, "LingbotPolicy: the vision engine's patch count differs");
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
    ELLM_CHECK(mVision.enqueue(mStream), "LingbotPolicy: vision engine failed");
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));

    // The prompt, positions, state and initial sample are prepared while the vision engine runs.
    auto const promptStart = std::chrono::steady_clock::now();
    std::vector<int32_t> ids = promptIds(task);
    int32_t const valid = std::min<int32_t>(static_cast<int32_t>(ids.size()), mTextLength);
    chunk.promptTokens = valid;
    int32_t const P = mPrefixLength;
    auto* hostIds = static_cast<int32_t*>(mPromptHost.rawPointer());
    auto* hostPositions = hostIds + mTextLength;
    auto* hostLangMask = reinterpret_cast<__half*>(hostPositions + 3 * P);
    auto* hostPrefixMask = hostLangMask + mTextLength;
    auto* hostState = hostPrefixMask + P;
    for (int32_t t = 0; t < mTextLength; ++t)
    {
        hostIds[t] = t < valid ? ids[t] : mPadId;
        hostLangMask[t] = __float2half(t < valid ? 1.0F : 0.0F);
    }
    std::vector<int32_t> const positions = positionIds(valid);
    std::copy(positions.begin(), positions.end(), hostPositions);
    int32_t const images = P - mTextLength - mAlignTokens;
    for (int32_t p = 0; p < P; ++p)
    {
        bool const text = p >= images && p < images + mTextLength;
        hostPrefixMask[p] = __float2half(!text || p - images < valid ? 1.0F : 0.0F);
    }
    std::fill_n(hostState, mMaxDim, __float2half(0.0F));
    for (auto const& f : mStateLayout)
    {
        for (int32_t i = 0; i < f.count; ++i)
        {
            float v = (state[f.start + i] - f.low[i]) / (f.high[i] - f.low[i] + kBoundsEps) * 2.0F - 1.0F;
            if (f.clip)
            {
                v = std::clamp(v, -1.0F, 1.0F);
            }
            hostState[f.slot + i] = __float2half(v);
        }
    }
    int32_t suffixStart = 0;
    for (int32_t p = 0; p < P; ++p)
    {
        if (__half2float(hostPrefixMask[p]) > 0.5F)
        {
            for (int32_t a = 0; a < 3; ++a)
            {
                suffixStart = std::max(suffixStart, positions[a * P + p] + 1);
            }
        }
    }
    auto* slots = static_cast<char*>(mStageHost.rawPointer());
    *reinterpret_cast<int32_t*>(slots + mSteps * kSlotBytes) = suffixStart;
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(mScalars.rawPointer()) + mSteps * kSlotBytes,
        slots + mSteps * kSlotBytes, 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), hostIds, mTextLength * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mPositionIds.rawPointer(), hostPositions, 3 * P * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mLangMask.rawPointer(), hostLangMask, mTextLength * 2, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mPrefixMask.rawPointer(), hostPrefixMask, P * 2, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mState.rawPointer(), hostState, mMaxDim * 2, cudaMemcpyHostToDevice, mStream));

    int64_t const chunkElems = static_cast<int64_t>(mHorizon) * mMaxDim;
    auto* x0 = reinterpret_cast<float*>(slots + (mSteps + 2) * kSlotBytes);
    auto* seed = x0 + chunkElems;
    auto* weight = seed + chunkElems;
    if (noise.empty())
    {
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::generate(x0, x0 + chunkElems, [&] { return normal(mNoiseGen); });
    }
    else
    {
        ELLM_CHECK(static_cast<int64_t>(noise.size()) == chunkElems, "LingbotPolicy: noise must be [horizon, max dim]");
        std::copy(noise.begin(), noise.end(), x0);
    }
    std::fill_n(seed, chunkElems, 0.0F);
    std::fill_n(weight, mHorizon, 0.0F);
    if (rtc != nullptr && rtc->overlapSteps > 0 && !mPrevious.empty())
    {
        int32_t const startRow = rtc->startRow >= 0 ? rtc->startRow : mHorizon - rtc->overlapSteps;
        ELLM_CHECK(rtc->overlapSteps <= mHorizon && startRow >= 0 && startRow <= mHorizon && rtc->frozenSteps >= 0,
            "LingbotPolicy: RTC overlap and start row must lie inside the chunk");
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
    CUDA_CHECK(cudaMemcpyAsync(mNoise.rawPointer(), x0, chunkElems * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mSeed.rawPointer(), seed, chunkElems * 4, cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mWeight.rawPointer(), weight, mHorizon * 4, cudaMemcpyHostToDevice, mStream));
    chunk.hostMs
        = frameMs + std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - promptStart).count();

    ELLM_CHECK(mPrefixA.enqueue(mStream), "LingbotPolicy: prefix_a engine failed");
    ELLM_CHECK(mPrefixB.enqueue(mStream), "LingbotPolicy: prefix_b engine failed");
    CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));
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
    mPrevious = chunk.normalized;
    chunk.actions.assign(static_cast<size_t>(mHorizon) * mActionDim, 0.0F);
    for (int32_t t = 0; t < mHorizon; ++t)
    {
        for (auto const& f : mActionLayout)
        {
            for (int32_t i = 0; i < f.count; ++i)
            {
                float const v = chunk.normalized[static_cast<size_t>(t) * mMaxDim + f.slot + i];
                chunk.actions[static_cast<size_t>(t) * mActionDim + f.start + i]
                    = (v + 1.0F) / 2.0F * (f.high[i] - f.low[i] + kBoundsEps) + f.low[i];
            }
        }
    }
    return chunk;
}

} // namespace lingbot
} // namespace trt_edgellm
