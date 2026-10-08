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

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

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

constexpr int32_t kPrecisionBits = 32 - 8 - 2;

double bicubicFilter(double x)
{
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
    {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0)
    {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    }
    return 0.0;
}

//! Pillow's precompute_coeffs + normalize_coeffs_8bpc for one axis.
struct Coefficients
{
    int32_t ksize{};
    std::vector<int32_t> bounds;  //!< [out, 2]: first source index, count
    std::vector<int32_t> weights; //!< [out, ksize], fixed point
};

Coefficients pilCoefficients(int32_t inSize, int32_t outSize)
{
    double const scale = static_cast<double>(inSize) / outSize;
    double const filterScale = std::max(scale, 1.0);
    double const support = 2.0 * filterScale;
    Coefficients c;
    c.ksize = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
    c.bounds.resize(static_cast<size_t>(outSize) * 2);
    c.weights.assign(static_cast<size_t>(outSize) * c.ksize, 0);
    std::vector<double> k(c.ksize);
    for (int32_t xx = 0; xx < outSize; ++xx)
    {
        double const center = (xx + 0.5) * scale;
        int32_t const xmin = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        int32_t const xmax = std::min(static_cast<int32_t>(center + support + 0.5), inSize) - xmin;
        double sum = 0.0;
        for (int32_t x = 0; x < xmax; ++x)
        {
            k[x] = bicubicFilter((x + xmin - center + 0.5) / filterScale);
            sum += k[x];
        }
        for (int32_t x = 0; x < xmax; ++x)
        {
            double const w = sum != 0.0 ? k[x] / sum : k[x];
            c.weights[static_cast<size_t>(xx) * c.ksize + x]
                = static_cast<int32_t>(w < 0 ? -0.5 + w * (1 << kPrecisionBits) : 0.5 + w * (1 << kPrecisionBits));
        }
        c.bounds[static_cast<size_t>(xx) * 2] = xmin;
        c.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    return c;
}

inline unsigned char clip8(int32_t value)
{
    return static_cast<unsigned char>(std::clamp(value >> kPrecisionBits, 0, 255));
}

} // namespace

std::vector<unsigned char> Gr00tEagleBackbone::resizeBicubicPil(
    unsigned char const* rgb, int32_t height, int32_t width, int32_t outHeight, int32_t outWidth)
{
    if (height == outHeight && width == outWidth)
    {
        return std::vector<unsigned char>(rgb, rgb + static_cast<size_t>(height) * width * 3);
    }
    // Pillow resamples horizontally, then vertically, rounding to 8 bits in between; it skips a pass whose
    // axis keeps its size.
    std::vector<unsigned char> horizontal;
    unsigned char const* source = rgb;
    if (width != outWidth)
    {
        Coefficients const c = pilCoefficients(width, outWidth);
        horizontal.resize(static_cast<size_t>(height) * outWidth * 3);
        for (int32_t y = 0; y < height; ++y)
        {
            for (int32_t xx = 0; xx < outWidth; ++xx)
            {
                int32_t const xmin = c.bounds[xx * 2];
                int32_t const count = c.bounds[xx * 2 + 1];
                int32_t const* k = &c.weights[static_cast<size_t>(xx) * c.ksize];
                for (int32_t ch = 0; ch < 3; ++ch)
                {
                    int32_t sum = 1 << (kPrecisionBits - 1);
                    for (int32_t x = 0; x < count; ++x)
                    {
                        sum += rgb[(static_cast<size_t>(y) * width + xmin + x) * 3 + ch] * k[x];
                    }
                    horizontal[(static_cast<size_t>(y) * outWidth + xx) * 3 + ch] = clip8(sum);
                }
            }
        }
        source = horizontal.data();
    }
    if (height == outHeight)
    {
        return horizontal;
    }
    Coefficients const c = pilCoefficients(height, outHeight);
    std::vector<unsigned char> out(static_cast<size_t>(outHeight) * outWidth * 3);
    for (int32_t yy = 0; yy < outHeight; ++yy)
    {
        int32_t const ymin = c.bounds[yy * 2];
        int32_t const count = c.bounds[yy * 2 + 1];
        int32_t const* k = &c.weights[static_cast<size_t>(yy) * c.ksize];
        for (int32_t x = 0; x < outWidth * 3; ++x)
        {
            int32_t sum = 1 << (kPrecisionBits - 1);
            for (int32_t y = 0; y < count; ++y)
            {
                sum += source[static_cast<size_t>(ymin + y) * outWidth * 3 + x] * k[y];
            }
            out[static_cast<size_t>(yy) * outWidth * 3 + x] = clip8(sum);
        }
    }
    return out;
}

Gr00tEagleBackbone::Gr00tEagleBackbone(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
{
    Json const config = readJson(engineDir + "/config.json");
    ELLM_CHECK(config.at("model_family").get<std::string>() == "gr00t_eagle",
        "Gr00tEagleBackbone: " + engineDir + " is not a GR00T Eagle backbone");
    mShortestEdge = config.at("shortest_image_edge").get<int32_t>();
    mCropFraction = config.at("crop_fraction").get<double>();
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
    mPrefix = vla::TrtEngine(*mRuntime, engineDir + "/prefix.engine", stream);
    mContextMemory = vla::allocateSharedContextMemory({&mVisual, &mPrefix}, "gr00t::eagleContextMemory");

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
    std::string text = mPromptPrefix + (mFormalize ? formalize(task) : task);
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
    return text + mPromptSuffix;
}

std::vector<float> Gr00tEagleBackbone::preprocessView(Gr00tView const& view) const
{
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
    std::vector<unsigned char> const resized = resizeBicubicPil(image.data, image.rows, image.cols, height, width);

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
    auto const numViews = static_cast<int32_t>(views.size());
    ELLM_CHECK(numViews >= 1 && numViews <= mMaxViews, "Gr00tEagleBackbone: unsupported number of views");
    size_t const viewSize = static_cast<size_t>(3) * mImageHeight * mImageWidth;
    auto* pixels = static_cast<__half*>(mPixelsHost.rawPointer());
    for (int32_t v = 0; v < numViews; ++v)
    {
        std::vector<float> const planar = preprocessView(views[v]);
        for (size_t i = 0; i < viewSize; ++i)
        {
            pixels[v * viewSize + i] = __float2half(planar[i]);
        }
    }

    auto const ids = mTokenizer->encode(prompt(task, numViews), /*addBos=*/false, /*addEos=*/false);
    mTokenIds.assign(ids.begin(), ids.end());
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
