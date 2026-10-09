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

#include "gr00tEagleBackbone.h"

#include "common/checkMacros.h"
#include "vlaImage.h"

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <unordered_map>

namespace trt_edgellm
{
namespace gr00t
{

namespace
{

using Json = nlohmann::json;
using nvinfer1::DataType;

Json readJson(std::string const& path)
{
    std::ifstream file(path);
    ELLM_CHECK(file.good(), "Gr00tEagleBackbone: missing " + path);
    return Json::parse(file);
}

rt::Tensor makeTensor(
    std::vector<int64_t> const& shape, DataType type, char const* name, rt::DeviceType device = rt::DeviceType::kGPU)
{
    return rt::Tensor(rt::Coords(shape), device, type, name);
}

//! Python's round(): half to even (the default floating-point rounding mode).
int32_t pyRound(double value)
{
    return static_cast<int32_t>(std::nearbyint(value));
}

//! albumentations SmallestMaxSize: scale so the shorter side is \p edge, skipped at scale 1.
cv::Mat smallestMaxSize(cv::Mat const& image, int32_t edge)
{
    double const scale = static_cast<double>(edge) / static_cast<double>(std::min(image.rows, image.cols));
    if (scale == 1.0)
    {
        return image;
    }
    cv::Mat resized;
    cv::resize(
        image, resized, cv::Size(pyRound(image.cols * scale), pyRound(image.rows * scale)), 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

//! Eagle's smart_resize: each side rounded to a multiple of 28, kept within its pixel budget.
std::pair<int32_t, int32_t> smartResize(int32_t height, int32_t width)
{
    constexpr int32_t kFactor = 28;
    constexpr int64_t kMinPixels = 4 * 28 * 28;
    constexpr int64_t kMaxPixels = 4096 * 28 * 28;
    constexpr int32_t kMaxSide = 500 * 14;
    int32_t h = std::min(std::max(kFactor, pyRound(static_cast<double>(height) / kFactor) * kFactor), kMaxSide);
    int32_t w = std::min(std::max(kFactor, pyRound(static_cast<double>(width) / kFactor) * kFactor), kMaxSide);
    if (static_cast<int64_t>(h) * w > kMaxPixels)
    {
        double const beta = std::sqrt(static_cast<double>(h) * w / kMaxPixels);
        h = static_cast<int32_t>(std::floor(h / beta / kFactor)) * kFactor;
        w = static_cast<int32_t>(std::floor(w / beta / kFactor)) * kFactor;
    }
    else if (static_cast<int64_t>(h) * w < kMinPixels)
    {
        double const beta = std::sqrt(static_cast<double>(kMinPixels) / (static_cast<double>(height) * width));
        h = static_cast<int32_t>(std::ceil(height * beta / kFactor)) * kFactor;
        w = static_cast<int32_t>(std::ceil(width * beta / kFactor)) * kFactor;
    }
    return {h, w};
}

//! torch's antialiased bilinear coefficients for one axis (F.interpolate(mode="bilinear", antialias=True,
//! align_corners=False) on FP32), with its float / double promotions.
struct AxisWeights
{
    int32_t maxSize{};
    std::vector<int32_t> start;
    std::vector<int32_t> count;
    std::vector<float> weights; //!< [out, maxSize]
};

AxisWeights torchAntialiasWeights(int32_t inSize, int32_t outSize)
{
    float const scale = static_cast<float>(inSize) / static_cast<float>(outSize);
    float const support = scale >= 1.0F ? scale : 1.0F;
    float const invScale = scale >= 1.0F ? 1.0F / scale : 1.0F;
    AxisWeights a;
    a.maxSize = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    a.start.resize(outSize);
    a.count.resize(outSize);
    a.weights.assign(static_cast<size_t>(outSize) * a.maxSize, 0.0F);
    for (int32_t i = 0; i < outSize; ++i)
    {
        auto const center = static_cast<float>(scale * (i + 0.5));
        auto const xmin = std::max<int64_t>(static_cast<int64_t>(static_cast<double>(center - support) + 0.5), 0);
        int64_t const xsize = std::clamp<int64_t>(
            std::min<int64_t>(static_cast<int64_t>(static_cast<double>(center + support) + 0.5), inSize) - xmin, 0,
            a.maxSize);
        float* w = &a.weights[static_cast<size_t>(i) * a.maxSize];
        float total = 0.0F;
        for (int64_t j = 0; j < xsize; ++j)
        {
            auto const x
                = static_cast<float>((static_cast<double>(static_cast<float>(j + xmin) - center) + 0.5) * invScale);
            w[j] = std::abs(x) < 1.0F ? 1.0F - std::abs(x) : 0.0F;
            total += w[j];
        }
        if (total != 0.0F)
        {
            for (int64_t j = 0; j < xsize; ++j)
            {
                w[j] /= total;
            }
        }
        a.start[i] = static_cast<int32_t>(xmin);
        a.count[i] = static_cast<int32_t>(xsize);
    }
    return a;
}

//! torch's antialiased bilinear resize of planar FP32 [channels, height, width]: the width pass, then the height
//! pass, each a product then fused multiply-adds in FP32, as torch's aarch64 build accumulates. This order and the
//! FMAs matter: N1.5 truncates the result to 8 bits, and plain multiply-adds flip ~1% of the pixels by one step.
std::vector<float> resizeBilinearAntialiasTorch(
    std::vector<float> const& src, int32_t channels, int32_t height, int32_t width, int32_t outHeight, int32_t outWidth)
{
    AxisWeights const wx = torchAntialiasWeights(width, outWidth);
    std::vector<float> horizontal(static_cast<size_t>(channels) * height * outWidth);
    for (int64_t row = 0; row < static_cast<int64_t>(channels) * height; ++row)
    {
        for (int32_t x = 0; x < outWidth; ++x)
        {
            float const* w = &wx.weights[static_cast<size_t>(x) * wx.maxSize];
            float const* in = &src[row * width + wx.start[x]];
            float out = in[0] * w[0];
            for (int32_t j = 1; j < wx.count[x]; ++j)
            {
                out = std::fmaf(in[j], w[j], out);
            }
            horizontal[row * outWidth + x] = out;
        }
    }
    AxisWeights const wy = torchAntialiasWeights(height, outHeight);
    std::vector<float> out(static_cast<size_t>(channels) * outHeight * outWidth);
    for (int32_t c = 0; c < channels; ++c)
    {
        for (int32_t y = 0; y < outHeight; ++y)
        {
            float const* w = &wy.weights[static_cast<size_t>(y) * wy.maxSize];
            for (int32_t x = 0; x < outWidth; ++x)
            {
                float const* in = &horizontal[(static_cast<size_t>(c) * height + wy.start[y]) * outWidth + x];
                float value = in[0] * w[0];
                for (int32_t j = 1; j < wy.count[y]; ++j)
                {
                    value = std::fmaf(in[static_cast<size_t>(j) * outWidth], w[j], value);
                }
                out[(static_cast<size_t>(c) * outHeight + y) * outWidth + x] = value;
            }
        }
    }
    return out;
}

} // namespace

namespace
{
//! The hidden-state capture slot of an engine exported with emit_hidden_states.
constexpr int32_t kHiddenCaptureSlot = 1;
} // namespace

Gr00tEagleBackbone::Gr00tEagleBackbone(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    Json const config = readJson(engineDir + "/config.json");
    ELLM_CHECK(config.at("model_family").get<std::string>() == "gr00t_eagle",
        "Gr00tEagleBackbone: " + engineDir + " is not a GR00T Eagle backbone");
    std::string const pipeline = config.at("image_pipeline").get<std::string>();
    ELLM_CHECK(
        pipeline == "gr00t_n16" || pipeline == "gr00t_n15", "Gr00tEagleBackbone: unknown image pipeline " + pipeline);
    mN15Pipeline = pipeline == "gr00t_n15";
    if (mN15Pipeline)
    {
        mCropFraction = config.at("crop_scale").get<double>();
    }
    else
    {
        mShortestEdge = config.at("shortest_image_edge").get<int32_t>();
        mCropFraction = config.at("crop_fraction").get<double>();
    }
    mTextAfterImages = config.at("text_after_images").get<bool>();
    mImageHeight = config.at("image_height").get<int32_t>();
    mImageWidth = config.at("image_width").get<int32_t>();
    mImageTokens = config.at("image_tokens_per_view").get<int32_t>();
    mMaxViews = config.at("max_views").get<int32_t>();
    mMaxTokens = config.at("max_tokens").get<int32_t>();
    mHidden = config.at("hidden_size").get<int32_t>();
    mImageTokenId = config.at("image_token_id").get<int64_t>();
    mFormalize = config.at("formalize_language").get<bool>();
    mPromptPrefix = config.at("prompt_prefix").get<std::string>();
    mImagePrefix = config.at("image_prefix").get<std::string>();
    mImageContext = config.at("image_context").get<std::string>();
    mImageSuffix = config.at("image_suffix").get<std::string>();
    mPromptSuffix = config.at("prompt_suffix").get<std::string>();

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir), "Gr00tEagleBackbone: failed to load the tokenizer");

    mRuntime = vla::createTrtRuntime();
    mVisual = vla::TrtEngine(*mRuntime, engineDir + "/visual.engine", stream);
    if (std::ifstream(engineDir + "/llm/config.json").good())
    {
        mPluginHandle = loadEdgellmPluginLib();
        std::unordered_map<std::string, std::string> const noLora;
        mLlm = std::make_unique<rt::LLMInferenceRuntime>(engineDir + "/llm", "", noLora, stream);
        mContextMemory = vla::allocateSharedContextMemory({&mVisual}, "gr00t::eagleContextMemory");
    }
    else
    {
        mPrefix = vla::TrtEngine(*mRuntime, engineDir + "/prefix.engine", stream);
        mContextMemory = vla::allocateSharedContextMemory({&mVisual, &mPrefix}, "gr00t::eagleContextMemory");
        mMaxTokens = std::min<int32_t>(mMaxTokens,
            static_cast<int32_t>(
                mPrefix.engine().getProfileShape("token_ids", 0, nvinfer1::OptProfileSelector::kMAX).d[1]));
    }

    int64_t const pixels = static_cast<int64_t>(mMaxViews) * 3 * mImageHeight * mImageWidth;
    mPixelsHost = makeTensor({pixels}, DataType::kHALF, "gr00t::pixelsHost", rt::DeviceType::kCPU);
    mPixels = makeTensor({pixels}, DataType::kHALF, "gr00t::pixels");
    mImageFeatures = makeTensor(
        {static_cast<int64_t>(mMaxViews) * mImageTokens, mHidden}, DataType::kHALF, "gr00t::imageFeatures");
    mTokensHost = makeTensor({mMaxTokens}, DataType::kINT64, "gr00t::tokensHost", rt::DeviceType::kCPU);
    mTokens = makeTensor({mMaxTokens}, DataType::kINT64, "gr00t::tokens");
    mFeatures = makeTensor({1, mMaxTokens, mHidden}, DataType::kHALF, "gr00t::backboneFeatures");
    for (cudaEvent_t& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

Gr00tEagleBackbone::~Gr00tEagleBackbone() noexcept
{
    for (cudaEvent_t event : mEvents)
    {
        cudaEventDestroy(event);
    }
}

std::string Gr00tEagleBackbone::formalize(std::string const& task)
{
    std::string out;
    out.reserve(task.size());
    for (unsigned char const c : task)
    {
        if (c >= 0x80 || std::isalnum(c) || c == '_' || std::isspace(c))
        {
            out.push_back(static_cast<char>(c < 0x80 ? std::tolower(c) : c));
        }
    }
    return out;
}

std::string Gr00tEagleBackbone::prompt(std::string const& task, int32_t numViews) const
{
    std::string const instruction = mFormalize ? formalize(task) : task;
    std::string text = mPromptPrefix + (mTextAfterImages ? "" : instruction);
    std::string context;
    for (int32_t i = 0; i < mImageTokens; ++i)
    {
        context += mImageContext;
    }
    for (int32_t v = 0; v < numViews; ++v)
    {
        std::string prefix = mImagePrefix;
        prefix.replace(prefix.find("{index}"), 7, std::to_string(v + 1));
        text += prefix + context + mImageSuffix;
    }
    return text + (mTextAfterImages ? instruction : "") + mPromptSuffix;
}

std::vector<float> Gr00tEagleBackbone::preprocessViewN15(Gr00tView const& view) const
{
    // VideoToTensor ([0, 1] floats), VideoCrop (eval: centre crop of int(side * scale)), VideoResize (bilinear,
    // antialiased) and VideoToNumpy (truncation to 8 bits); Eagle 2.5 then sees a single tile.
    int32_t const cropH = static_cast<int32_t>(view.height * mCropFraction);
    int32_t const cropW = static_cast<int32_t>(view.width * mCropFraction);
    int32_t const top = pyRound((view.height - cropH) / 2.0);
    int32_t const left = pyRound((view.width - cropW) / 2.0);
    std::vector<float> crop(static_cast<size_t>(3) * cropH * cropW);
    for (int32_t c = 0; c < 3; ++c)
    {
        for (int32_t y = 0; y < cropH; ++y)
        {
            for (int32_t x = 0; x < cropW; ++x)
            {
                crop[(static_cast<size_t>(c) * cropH + y) * cropW + x]
                    = static_cast<float>(view.rgb[(static_cast<size_t>(top + y) * view.width + left + x) * 3 + c])
                    / 255.0F;
            }
        }
    }
    std::vector<float> planar = resizeBilinearAntialiasTorch(crop, 3, cropH, cropW, mImageHeight, mImageWidth);
    for (float& value : planar)
    {
        float const pixel = static_cast<float>(static_cast<unsigned char>(value * 255.0F)) * (1.0F / 255.0F);
        value = (pixel - 0.5F) / 0.5F;
    }
    return planar;
}

std::vector<float> Gr00tEagleBackbone::preprocessView(Gr00tView const& view) const
{
    if (mN15Pipeline)
    {
        return preprocessViewN15(view);
    }
    cv::Mat const frame(view.height, view.width, CV_8UC3, const_cast<unsigned char*>(view.rgb));
    cv::Mat image = smallestMaxSize(frame, mShortestEdge);
    int32_t const cropH = std::max(1, static_cast<int32_t>(image.rows * mCropFraction));
    int32_t const cropW = std::max(1, static_cast<int32_t>(image.cols * mCropFraction));
    image = image(cv::Rect((image.cols - cropW) / 2, (image.rows - cropH) / 2, cropW, cropH)).clone();
    image = smallestMaxSize(image, mShortestEdge);
    if (!image.isContinuous())
    {
        image = image.clone();
    }
    auto const [height, width] = smartResize(image.rows, image.cols);
    ELLM_CHECK(height == mImageHeight && width == mImageWidth,
        "Gr00tEagleBackbone: this camera resolution gives " + std::to_string(height) + "x" + std::to_string(width)
            + " images, the engines were built for " + std::to_string(mImageHeight) + "x"
            + std::to_string(mImageWidth));
    std::vector<unsigned char> const resized = vla::resizeBicubicPil(image.data, image.rows, image.cols, height, width);

    std::vector<float> planar(static_cast<size_t>(3) * height * width);
    for (int32_t y = 0; y < height; ++y)
    {
        for (int32_t x = 0; x < width; ++x)
        {
            for (int32_t c = 0; c < 3; ++c)
            {
                float const value
                    = static_cast<float>(resized[(static_cast<size_t>(y) * width + x) * 3 + c]) * (1.0F / 255.0F);
                planar[(static_cast<size_t>(c) * height + y) * width + x] = (value - 0.5F) / 0.5F;
            }
        }
    }
    return planar;
}

rt::Tensor const& Gr00tEagleBackbone::encode(std::vector<Gr00tView> const& views, std::string const& task)
{
    auto const hostStart = std::chrono::steady_clock::now();
    auto const numViews = static_cast<int32_t>(views.size());
    ELLM_CHECK(numViews >= 1 && numViews <= mMaxViews, "Gr00tEagleBackbone: unsupported number of views");
    size_t const viewSize = static_cast<size_t>(3) * mImageHeight * mImageWidth;
    auto* pixels = static_cast<__half*>(mPixelsHost.rawPointer());
    // Views are independent and preprocessView is pure, so they run concurrently into disjoint slices.
    auto stage = [&](int32_t v) {
        std::vector<float> const planar = preprocessView(views[v]);
        for (size_t i = 0; i < viewSize; ++i)
        {
            pixels[v * viewSize + i] = __float2half(planar[i]);
        }
    };
    std::vector<std::future<void>> pending;
    for (int32_t v = 1; v < numViews; ++v)
    {
        pending.push_back(std::async(std::launch::async, stage, v));
    }
    stage(0);
    for (auto& view : pending)
    {
        view.get();
    }

    // The prompt depends only on the task and the view count, which stay fixed within an episode.
    if (task != mLastTask || numViews != mLastViews)
    {
        auto const ids = mTokenizer->encode(prompt(task, numViews), /*addBos=*/false, /*addEos=*/false);
        mTokenIds.assign(ids.begin(), ids.end());
        mLastTask = task;
        mLastViews = numViews;
    }
    auto const tokens = static_cast<int64_t>(mTokenIds.size());
    ELLM_CHECK(tokens >= 2 && tokens <= mMaxTokens, "Gr00tEagleBackbone: prompt length is outside the engine's range");
    mImageMask.assign(mTokenIds.size(), 0);
    int64_t imageTokens = 0;
    for (size_t i = 0; i < mTokenIds.size(); ++i)
    {
        mImageMask[i] = mTokenIds[i] == mImageTokenId ? 1 : 0;
        imageTokens += mImageMask[i];
    }
    ELLM_CHECK(imageTokens == static_cast<int64_t>(numViews) * mImageTokens,
        "Gr00tEagleBackbone: the tokenizer did not produce one image-context token per image feature");
    std::copy(mTokenIds.begin(), mTokenIds.end(), mTokensHost.dataPointer<int64_t>());
    mHostMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - hostStart).count();

    CUDA_CHECK(cudaMemcpyAsync(mPixels.rawPointer(), mPixelsHost.rawPointer(), numViews * viewSize * sizeof(__half),
        cudaMemcpyHostToDevice, mStream));
    CUDA_CHECK(cudaMemcpyAsync(
        mTokens.rawPointer(), mTokensHost.rawPointer(), tokens * sizeof(int64_t), cudaMemcpyHostToDevice, mStream));

    CUDA_CHECK(cudaEventRecord(mEvents[0], mStream));
    mVisual.setShape("pixel_values", {numViews, 3, mImageHeight, mImageWidth});
    mVisual.bind("pixel_values", mPixels.rawPointer());
    mVisual.bind("image_features", mImageFeatures.rawPointer());
    ELLM_CHECK(mVisual.enqueue(mStream), "Gr00tEagleBackbone: visual enqueue failed");
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));

    if (mLlm)
    {
        ELLM_CHECK(mImageFeatures.reshape(rt::Coords({imageTokens, mHidden})),
            "Gr00tEagleBackbone: image feature reshape failed");
        rt::LLMGenerationRequest request;
        request.requests.resize(1);
        rt::Message message;
        message.role = "user";
        message.contents.push_back({"text", ""});
        request.requests[0].messages.push_back(std::move(message));
        request.preTokenizedInputIds = {std::vector<int32_t>(mTokenIds.begin(), mTokenIds.end())};
        request.precomputedImageEmbeddings = &mImageFeatures;
        request.applyChatTemplate = false;
        request.maxGenerateLength = 1;
        request.acceptHiddenLayer = kHiddenCaptureSlot;
        request.temperature = 1.0F;
        request.topP = 1.0F;
        request.topK = 1;
        rt::LLMGenerationResponse response;
        ELLM_CHECK(mLlm->handleRequest(request, response, mStream, /*outputThinkerEmbeddings=*/true),
            "Gr00tEagleBackbone: LLM prefix request failed");
        rt::Tensor const* hidden = mLlm->getBaseModelHiddenStates(kHiddenCaptureSlot);
        ELLM_CHECK(hidden != nullptr && !hidden->isEmpty(), "Gr00tEagleBackbone: the LLM returned no hidden states");
        CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));
        return *hidden;
    }
    ELLM_CHECK(mFeatures.reshape(rt::Coords({1, tokens, mHidden})), "Gr00tEagleBackbone: feature reshape failed");
    mPrefix.setShape("token_ids", {1, tokens});
    mPrefix.setShape("image_features", {1, imageTokens, mHidden});
    mPrefix.bind("token_ids", mTokens.rawPointer());
    mPrefix.bind("image_features", mImageFeatures.rawPointer());
    mPrefix.bind("backbone_features", mFeatures.rawPointer());
    ELLM_CHECK(mPrefix.enqueue(mStream), "Gr00tEagleBackbone: prefix enqueue failed");
    CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));
    return mFeatures;
}

float Gr00tEagleBackbone::visualMs() const
{
    float ms = 0.0F;
    CUDA_CHECK(cudaEventSynchronize(mEvents[1]));
    CUDA_CHECK(cudaEventElapsedTime(&ms, mEvents[0], mEvents[1]));
    return ms;
}

float Gr00tEagleBackbone::prefixMs() const
{
    float ms = 0.0F;
    CUDA_CHECK(cudaEventSynchronize(mEvents[2]));
    CUDA_CHECK(cudaEventElapsedTime(&ms, mEvents[1], mEvents[2]));
    return ms;
}

} // namespace gr00t
} // namespace trt_edgellm
