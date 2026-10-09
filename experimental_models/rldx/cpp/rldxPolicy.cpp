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

#include "rldxPolicy.h"

#include "common/checkMacros.h"
#include "multimodal/common/multimodalRunner.h"
#include "rldxText.h"
#include "runtime/llmRuntimeUtils.h"

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>

namespace trt_edgellm
{
namespace rldx
{
namespace
{

using Json = nlohmann::json;
using nvinfer1::DataType;

constexpr int64_t kScalarSlot = 128; //!< halves per 256 bytes

rt::Tensor makeTensor(
    std::vector<int64_t> const& shape, DataType type, char const* name, rt::DeviceType device = rt::DeviceType::kGPU)
{
    return rt::Tensor(rt::Coords(shape), device, type, name);
}

void appendRanges(Json const& groups, std::vector<float>& low, std::vector<float>& high)
{
    for (auto const& group : groups)
    {
        auto const lo = group.at("q01").get<std::vector<float>>();
        auto const hi = group.at("q99").get<std::vector<float>>();
        low.insert(low.end(), lo.begin(), lo.end());
        high.insert(high.end(), hi.begin(), hi.end());
    }
}

int64_t profileMax(vla::TrtEngine const& engine, char const* name, int32_t dim)
{
    return engine.engine().getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX).d[dim];
}

} // namespace

RldxPolicy::RldxPolicy(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    std::ifstream file(engineDir + "/config.json");
    ELLM_CHECK(file.good(), "RldxPolicy: missing " + engineDir + "/config.json");
    Json const config = Json::parse(file);
    ELLM_CHECK(config.at("model_family").get<std::string>() == "rldx", "RldxPolicy: not an RLDX engine dir");
    mCameras = config.at("cameras").get<std::vector<std::string>>();
    mHistory = config.at("frame_history").get<std::vector<int32_t>>();
    std::sort(mHistory.begin(), mHistory.end());
    ELLM_CHECK(mHistory.back() == 0, "RldxPolicy: the frame history must end at the current frame");
    mImageSize = config.at("image_size").get<int32_t>();
    auto const grid = config.at("grid").get<std::vector<int32_t>>();
    mGrid[0] = grid[0];
    mGrid[1] = grid[1];
    mCognition = config.at("cognition_tokens").get<int32_t>();
    mHidden = config.at("hidden_size").get<int32_t>();
    mHorizon = config.at("action_horizon").get<int32_t>();
    mMaxActionDim = config.at("max_action_dim").get<int32_t>();
    mMaxStateDim = config.at("max_state_dim").get<int32_t>();
    mSteps = config.at("denoising_steps").get<int32_t>();
    mGeometry.headDim = config.at("head_dim").get<int32_t>();
    mGeometry.ropeTheta = config.at("rope_theta").get<float>();
    auto const section = config.at("mrope_section").get<std::vector<int32_t>>();
    std::copy(section.begin(), section.end(), mGeometry.mropeSection);
    appendRanges(config.at("state"), mStateLow, mStateHigh);
    appendRanges(config.at("action"), mActionLow, mActionHigh);

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir), "RldxPolicy: failed to load the tokenizer");
    mVision = rt::MultimodalRunner::create(engineDir + "/visual", 1, 4096, stream);
    mRuntime = vla::createTrtRuntime();
    mLlmA = vla::TrtEngine(*mRuntime, engineDir + "/llm_a.engine", stream);
    mLlmB = vla::TrtEngine(*mRuntime, engineDir + "/llm_b.engine", stream);
    mAction = vla::TrtEngine(*mRuntime, engineDir + "/action.engine", stream);
    int64_t const contextBytes = std::max({mLlmA.deviceMemorySize(), mLlmB.deviceMemorySize(),
        mAction.deviceMemorySize(), mVision->getRequiredContextMemorySize()});
    mContextMemory = makeTensor({contextBytes}, DataType::kUINT8, "rldx::contextMemory");
    mLlmA.setDeviceMemory(mContextMemory);
    mLlmB.setDeviceMemory(mContextMemory);
    mAction.setDeviceMemory(mContextMemory);
    ELLM_CHECK(mVision->setContextMemory(mContextMemory), "RldxPolicy: visual context memory");

    int64_t const promptMax = profileMax(mLlmA, "input_ids", 0);
    int64_t const sequenceMax = profileMax(mLlmA, "cos", 0);
    int64_t const compressedMax = profileMax(mLlmB, "keep_index", 0);
    int64_t const visualTokens
        = static_cast<int64_t>(mCameras.size()) * static_cast<int64_t>(mHistory.size()) * mGrid[0] * mGrid[1];
    int64_t const head = mGeometry.headDim;
    mInputIds = makeTensor({promptMax}, DataType::kINT64, "rldx::inputIds");
    mVisualIndex = makeTensor({promptMax}, DataType::kINT64, "rldx::visualIndex");
    mCos = makeTensor({sequenceMax, head}, DataType::kFLOAT, "rldx::cos");
    mSin = makeTensor({sequenceMax, head}, DataType::kFLOAT, "rldx::sin");
    mPool = makeTensor({sequenceMax}, DataType::kFLOAT, "rldx::pool");
    mKeepIndex = makeTensor({compressedMax}, DataType::kINT64, "rldx::keepIndex");
    mCosB = makeTensor({compressedMax, head}, DataType::kFLOAT, "rldx::cosB");
    mSinB = makeTensor({compressedMax, head}, DataType::kFLOAT, "rldx::sinB");
    mPromptHost = makeTensor(
        {(2 * promptMax + compressedMax) * 8 + (2 * sequenceMax * head + sequenceMax + 2 * compressedMax * head) * 4},
        DataType::kUINT8, "rldx::promptHost", rt::DeviceType::kCPU);
    mDeepstack = makeTensor({3, visualTokens, mHidden}, DataType::kHALF, "rldx::deepstack");
    mHiddenA = makeTensor({1, sequenceMax, mHidden}, DataType::kHALF, "rldx::hiddenA");
    mCognitionFeatures = makeTensor({1, mCognition, mHidden}, DataType::kHALF, "rldx::cognition");
    int64_t const chunkElems = static_cast<int64_t>(mHorizon) * mMaxActionDim;
    mX[0] = makeTensor({1, mHorizon, mMaxActionDim}, DataType::kHALF, "rldx::x0");
    mX[1] = makeTensor({1, mHorizon, mMaxActionDim}, DataType::kHALF, "rldx::x1");
    mVelocity = makeTensor({1, mHorizon, mMaxActionDim}, DataType::kHALF, "rldx::velocity");
    // TensorRT wants 256-byte aligned binding addresses: one slot per scalar.
    mTimes = makeTensor({(mSteps + 1) * kScalarSlot}, DataType::kHALF, "rldx::times");
    mState = makeTensor({1, 1, mMaxStateDim}, DataType::kHALF, "rldx::state");
    mStrength = makeTensor({1, mHorizon, 1}, DataType::kHALF, "rldx::strength");
    mStageHost = makeTensor({chunkElems + mMaxStateDim + mHorizon + (mSteps + 1) * kScalarSlot}, DataType::kHALF,
        "rldx::stage", rt::DeviceType::kCPU);
    mOutHost = makeTensor({chunkElems}, DataType::kHALF, "rldx::out", rt::DeviceType::kCPU);

    // Flow time t = k / steps per step, then dt.
    auto* times = static_cast<__half*>(mStageHost.rawPointer()) + chunkElems + mMaxStateDim + mHorizon;
    std::fill_n(times, (mSteps + 1) * kScalarSlot, __float2half(0.0F));
    for (int32_t k = 0; k < mSteps; ++k)
    {
        times[k * kScalarSlot] = __float2half(static_cast<float>(k) / static_cast<float>(mSteps));
    }
    times[mSteps * kScalarSlot] = __float2half(1.0F / static_cast<float>(mSteps));
    CUDA_CHECK(cudaMemcpyAsync(
        mTimes.rawPointer(), times, (mSteps + 1) * kScalarSlot * sizeof(__half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    mLlmA.bind("input_ids", mInputIds.rawPointer());
    mLlmA.bind("deepstack", mDeepstack.rawPointer());
    mLlmA.bind("visual_index", mVisualIndex.rawPointer());
    mLlmA.bind("cos", mCos.rawPointer());
    mLlmA.bind("sin", mSin.rawPointer());
    mLlmA.bind("hidden", mHiddenA.rawPointer());
    mLlmB.bind("hidden", mHiddenA.rawPointer());
    mLlmB.bind("pool", mPool.rawPointer());
    mLlmB.bind("keep_index", mKeepIndex.rawPointer());
    mLlmB.bind("cos", mCosB.rawPointer());
    mLlmB.bind("sin", mSinB.rawPointer());
    mLlmB.bind("cognition", mCognitionFeatures.rawPointer());
    for (cudaEvent_t& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

RldxPolicy::~RldxPolicy() noexcept
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

rt::imageUtils::ImageData RldxPolicy::preprocessFrame(unsigned char const* rgb, int32_t height, int32_t width) const
{
    // resize_preserve_aspect_area_then_crop: never upscale; the short side becomes the largest multiple of 32
    // within the area budget, the long side follows the aspect ratio and is centre-cropped to a multiple of 32.
    int32_t const m = 32;
    double const maxArea = static_cast<double>(mImageSize) * mImageSize;
    double const scale = std::min(1.0, std::sqrt(maxArea / (static_cast<double>(height) * width)));
    bool const tall = height > width;
    int32_t const shortSide = tall ? width : height;
    int32_t const longSide = tall ? height : width;
    int32_t const shortResized = std::max(m, static_cast<int32_t>(std::floor(shortSide * scale / m)) * m);
    int32_t const longResized = static_cast<int32_t>(longSide * (static_cast<double>(shortResized) / shortSide));
    int32_t const resizedH = tall ? longResized : shortResized;
    int32_t const resizedW = tall ? shortResized : longResized;
    int32_t const cropH = resizedH - resizedH % m;
    int32_t const cropW = resizedW - resizedW % m;

    cv::Mat const source(height, width, CV_8UC3, const_cast<unsigned char*>(rgb));
    cv::Mat resized = source;
    if (resizedH != height || resizedW != width)
    {
        cv::resize(source, resized, cv::Size(resizedW, resizedH), 0.0, 0.0, cv::INTER_AREA);
    }
    cv::Mat const cropped = resized(cv::Rect((resizedW - cropW) / 2, (resizedH - cropH) / 2, cropW, cropH));
    rt::Tensor pixels({1, cropH, cropW, 3}, rt::DeviceType::kCPU, DataType::kUINT8, "rldx::frame");
    auto* out = pixels.dataPointer<unsigned char>();
    for (int32_t y = 0; y < cropH; ++y)
    {
        std::memcpy(
            out + static_cast<size_t>(y) * cropW * 3, cropped.ptr<unsigned char>(y), static_cast<size_t>(cropW) * 3);
    }
    return rt::imageUtils::ImageData(std::move(pixels));
}

RldxPrompt const& RldxPolicy::prompt(std::string const& task, std::vector<std::pair<int32_t, int32_t>> const& grids)
{
    if (task == mLastTask && grids == mLastGrids)
    {
        return mPrompt;
    }
    auto const ids = mTokenizer->encode(formalizeLanguage(task));
    RldxPrompt built = buildPrompt(std::vector<int32_t>(ids.begin(), ids.end()), grids,
        static_cast<int32_t>(mCameras.size()), mCognition, mGeometry);
    int64_t const s0 = static_cast<int64_t>(built.inputIds.size());
    ELLM_CHECK(s0 <= mInputIds.getShape()[0] && built.sequence() <= mCos.getShape()[0]
            && built.compressed() <= mKeepIndex.getShape()[0],
        "RldxPolicy: the instruction is longer than the engines' profile");
    mLastTask.clear();
    mPrompt = std::move(built);

    // One pinned staging buffer, uploaded once per task; the stream is synchronized before it is rewritten.
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    auto* stage = static_cast<char*>(mPromptHost.rawPointer());
    auto upload = [&](rt::Tensor& dst, void const* src, size_t bytes) {
        std::memcpy(stage, src, bytes);
        CUDA_CHECK(cudaMemcpyAsync(dst.rawPointer(), stage, bytes, cudaMemcpyHostToDevice, mStream));
        stage += bytes;
    };
    upload(mInputIds, mPrompt.inputIds.data(), s0 * sizeof(int64_t));
    upload(mVisualIndex, mPrompt.visualIndex.data(), s0 * sizeof(int64_t));
    upload(mCos, mPrompt.cos.data(), mPrompt.cos.size() * sizeof(float));
    upload(mSin, mPrompt.sin.data(), mPrompt.sin.size() * sizeof(float));
    upload(mPool, mPrompt.pool.data(), mPrompt.pool.size() * sizeof(float));
    upload(mKeepIndex, mPrompt.keepIndex.data(), mPrompt.keepIndex.size() * sizeof(int64_t));
    upload(mCosB, mPrompt.cosCompressed.data(), mPrompt.cosCompressed.size() * sizeof(float));
    upload(mSinB, mPrompt.sinCompressed.data(), mPrompt.sinCompressed.size() * sizeof(float));
    mLastTask = task;
    mLastGrids = grids;
    return mPrompt;
}

void RldxPolicy::enqueueDenoise()
{
    auto* times = static_cast<__half*>(mTimes.rawPointer());
    for (int32_t k = 0; k < mSteps; ++k)
    {
        mAction.bind("x", mX[k % 2].rawPointer());
        mAction.bind("t", times + k * kScalarSlot);
        mAction.bind("dt", times + mSteps * kScalarSlot);
        mAction.bind("cognition", mCognitionFeatures.rawPointer());
        mAction.bind("state", mState.rawPointer());
        mAction.bind("strength", mStrength.rawPointer());
        mAction.bind("x_next", mX[(k + 1) % 2].rawPointer());
        mAction.bind("velocity", mVelocity.rawPointer());
        ELLM_CHECK(mAction.enqueue(mStream), "RldxPolicy: action engine failed");
    }
}

RldxChunk RldxPolicy::act(std::vector<RldxFrame> const& frames, std::vector<float> const& state,
    std::string const& task, std::vector<float> const& noise, RldxRtc const* rtc)
{
    auto const start = std::chrono::steady_clock::now();
    ELLM_CHECK(static_cast<int32_t>(state.size()) == stateDim(),
        "RldxPolicy: state needs " + std::to_string(stateDim()) + " values");
    RldxChunk chunk;

    // Frame-major, view-minor, as the processor orders them; a missing history offset takes the camera's oldest
    // frame, as the official evaluation buffer is filled at reset.
    rt::LLMGenerationRequest request;
    request.requests.resize(1);
    std::vector<std::pair<int32_t, int32_t>> grids;
    for (int32_t offset : mHistory)
    {
        for (auto const& camera : mCameras)
        {
            RldxFrame const* best = nullptr;
            for (auto const& frame : frames)
            {
                if (frame.camera == camera && frame.offset >= offset
                    && (best == nullptr || frame.offset < best->offset))
                {
                    best = &frame;
                }
            }
            ELLM_CHECK(best != nullptr, "RldxPolicy: no current frame for camera " + camera);
            chunk.historyFilled += best->offset != offset ? 1 : 0;
            rt::imageUtils::ImageData image = preprocessFrame(best->rgb, best->height, best->width);
            // Qwen3-VL: 16-pixel patches merged 2 x 2.
            grids.emplace_back(static_cast<int32_t>(image.height) / 32, static_cast<int32_t>(image.width) / 32);
            request.requests[0].imageBuffers.push_back(std::move(image));
        }
    }
    RldxPrompt const& tables = prompt(task, grids);
    int64_t const s0 = static_cast<int64_t>(tables.inputIds.size());

    int32_t const stateDims = stateDim();
    int64_t const chunkElems = static_cast<int64_t>(mHorizon) * mMaxActionDim;
    auto* stage = static_cast<__half*>(mStageHost.rawPointer());
    std::vector<float> x0(static_cast<size_t>(chunkElems));
    if (noise.empty())
    {
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::generate(x0.begin(), x0.end(), [&] { return normal(mNoiseGen); });
    }
    else
    {
        ELLM_CHECK(
            static_cast<int64_t>(noise.size()) == chunkElems, "RldxPolicy: noise must be [horizon, max action dim]");
        x0 = noise;
    }
    std::vector<float> strength(mHorizon, 1.0F);
    if (rtc != nullptr && rtc->overlapSteps > 0 && !mPrevious.empty())
    {
        int32_t const startRow = rtc->startRow >= 0 ? rtc->startRow : mHorizon - rtc->overlapSteps;
        ELLM_CHECK(rtc->overlapSteps <= mHorizon && startRow >= 0 && startRow <= mHorizon && rtc->frozenSteps >= 0,
            "RldxPolicy: RTC overlap and start row must lie inside the chunk");
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
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    std::transform(x0.begin(), x0.end(), stage, [](float v) { return __float2half(v); });
    for (int32_t i = 0; i < mMaxStateDim; ++i)
    {
        float value = 0.0F;
        if (i < stateDims)
        {
            float const span = mStateHigh[i] - mStateLow[i];
            // np.isclose(q99, q01): the official normalization maps a degenerate range to 0.
            value = std::fabs(span) <= 1e-8F + 1e-5F * std::fabs(mStateLow[i])
                ? 0.0F
                : std::clamp(2.0F * (state[i] - mStateLow[i]) / span - 1.0F, -1.0F, 1.0F);
        }
        stage[chunkElems + i] = __float2half(value);
    }
    std::transform(
        strength.begin(), strength.end(), stage + chunkElems + mMaxStateDim, [](float v) { return __float2half(v); });
    CUDA_CHECK(
        cudaMemcpyAsync(mX[0].rawPointer(), stage, chunkElems * sizeof(__half), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        mState.rawPointer(), stage + chunkElems, mMaxStateDim * sizeof(__half), cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(mStrength.rawPointer(), stage + chunkElems + mMaxStateDim, mHorizon * sizeof(__half),
        cudaMemcpyHostToDevice, mStream));

    CUDA_CHECK(cudaEventRecord(mEvents[0], mStream));
    std::vector<std::vector<int32_t>> ids;
    ELLM_CHECK(mVision->preprocess(request, ids, nullptr, std::nullopt, mStream, /*imageOnly=*/true),
        "RldxPolicy: visual preprocessing failed");
    ELLM_CHECK(mVision->infer(mStream), "RldxPolicy: visual engine failed");
    rt::Tensor& visual = mVision->getOutputEmbedding();
    int64_t const visualTokens = visual.getShape()[0];
    auto const deepstack = mVision->getDeepstackFeatures();
    int64_t const expectedTokens = std::count(tables.inputIds.begin(), tables.inputIds.end(), int64_t{151655});
    ELLM_CHECK(deepstack.size() == 3 && visualTokens == expectedTokens && visualTokens <= mDeepstack.getShape()[1],
        "RldxPolicy: unexpected visual engine outputs");
    size_t const layerBytes = static_cast<size_t>(visualTokens) * mHidden * sizeof(__half);
    for (size_t d = 0; d < deepstack.size(); ++d)
    {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(mDeepstack.rawPointer()) + d * layerBytes,
            deepstack[d].get().rawPointer(), layerBytes, cudaMemcpyDeviceToDevice, mStream));
    }
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));

    int64_t const sequence = tables.sequence();
    int64_t const compressed = tables.compressed();
    int64_t const head = mGeometry.headDim;
    mLlmA.bind("visual", visual.rawPointer());
    mLlmA.setShape("input_ids", {s0});
    mLlmA.setShape("visual", {visualTokens, mHidden});
    mLlmA.setShape("deepstack", {3, visualTokens, mHidden});
    mLlmA.setShape("visual_index", {s0});
    mLlmA.setShape("cos", {sequence, head});
    mLlmA.setShape("sin", {sequence, head});
    ELLM_CHECK(mLlmA.enqueue(mStream), "RldxPolicy: llm_a engine failed");
    mLlmB.setShape("hidden", {1, sequence, mHidden});
    mLlmB.setShape("pool", {sequence});
    mLlmB.setShape("keep_index", {compressed});
    mLlmB.setShape("cos", {compressed, head});
    mLlmB.setShape("sin", {compressed, head});
    ELLM_CHECK(mLlmB.enqueue(mStream), "RldxPolicy: llm_b engine failed");
    CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));
    chunk.hostMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();

    if (mGraph != nullptr)
    {
        CUDA_CHECK(cudaGraphLaunch(mGraph, mStream));
    }
    else
    {
        // TensorRT needs one regular enqueue before its kernels can be captured.
        enqueueDenoise();
        if (mUseCudaGraph)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                mX[0].rawPointer(), stage, chunkElems * sizeof(__half), cudaMemcpyHostToDevice, mStream));
            cudaGraph_t graph{};
            CUDA_CHECK(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeThreadLocal));
            try
            {
                enqueueDenoise();
            }
            catch (...)
            {
                cudaStreamEndCapture(mStream, &graph);
                if (graph != nullptr)
                {
                    cudaGraphDestroy(graph);
                }
                throw;
            }
            CUDA_CHECK(cudaStreamEndCapture(mStream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&mGraph, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            CUDA_CHECK(cudaGraphLaunch(mGraph, mStream));
        }
    }
    CUDA_CHECK(cudaEventRecord(mEvents[3], mStream));
    CUDA_CHECK(cudaMemcpyAsync(mOutHost.rawPointer(), mX[mSteps % 2].rawPointer(), chunkElems * sizeof(__half),
        cudaMemcpyDeviceToHost, mStream));
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.visionMs, mEvents[0], mEvents[1]));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.llmMs, mEvents[1], mEvents[2]));
    CUDA_CHECK(cudaEventElapsedTime(&chunk.actionMs, mEvents[2], mEvents[3]));

    auto const* out = static_cast<__half const*>(mOutHost.rawPointer());
    chunk.normalized.resize(static_cast<size_t>(chunkElems));
    std::transform(out, out + chunkElems, chunk.normalized.begin(), [](__half h) { return __half2float(h); });
    mPrevious = chunk.normalized;
    int32_t const actionDims = actionDim();
    chunk.actions.resize(static_cast<size_t>(mHorizon) * actionDims);
    for (int32_t t = 0; t < mHorizon; ++t)
    {
        for (int32_t d = 0; d < actionDims; ++d)
        {
            float const v = std::clamp(chunk.normalized[static_cast<size_t>(t) * mMaxActionDim + d], -1.0F, 1.0F);
            chunk.actions[static_cast<size_t>(t) * actionDims + d]
                = (v + 1.0F) / 2.0F * (mActionHigh[d] - mActionLow[d]) + mActionLow[d];
        }
    }
    return chunk;
}

} // namespace rldx
} // namespace trt_edgellm
