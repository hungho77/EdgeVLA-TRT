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

#pragma once

#include "common/tensor.h"
#include "tokenizer/tokenizer.h"
#include "vlaEngine.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace gr00t
{

//! One camera frame, tightly packed row-major [height, width, 3] 8-bit RGB, borrowed for the call.
struct Gr00tView
{
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

//! GR00T's Eagle backbone (N1.5 or N1.6) on the visual / prefix engines written by
//! ``tensorrt_edgellm.models.eagle.export``, with the official preprocessing. N1.6: albumentations
//! SmallestMaxSize (OpenCV INTER_AREA), FractionalCenterCrop and SmallestMaxSize again, Eagle's
//! smart_resize with PIL's bicubic resize, and the lower-cased, punctuation-free task before the images.
//! N1.5: centre crop and torch's antialiased bilinear resize to 224x224 on [0, 1] floats, truncated to 8 bits,
//! and the task after the images. Both feed [-1, 1] pixels and Eagle's chat template.
//!
//! NOT thread-safe: one instance, one stream; every buffer is allocated at construction.
class Gr00tEagleBackbone
{
public:
    Gr00tEagleBackbone(std::string const& engineDir, cudaStream_t stream);
    ~Gr00tEagleBackbone() noexcept;

    Gr00tEagleBackbone(Gr00tEagleBackbone const&) = delete;
    Gr00tEagleBackbone& operator=(Gr00tEagleBackbone const&) = delete;

    //! \p views in the embodiment's camera order. Enqueues on the constructor's stream and returns the
    //! backbone features, [1, tokens, hidden] FP16 on the GPU, valid until the next call.
    rt::Tensor const& encode(std::vector<Gr00tView> const& views, std::string const& task);

    //! Of the last encode(): 1 at the image-context positions, one entry per token.
    std::vector<uint8_t> const& imageMask() const noexcept
    {
        return mImageMask;
    }
    std::vector<int64_t> const& tokenIds() const noexcept
    {
        return mTokenIds;
    }
    //! Engine time of the last encode() (synchronizes on its events).
    float visualMs() const;
    float prefixMs() const;

    //! One view as SigLIP2 sees it: planar [3, imageHeight, imageWidth] in [-1, 1].
    std::vector<float> preprocessView(Gr00tView const& view) const;
    std::string prompt(std::string const& task, int32_t numViews) const;
    //! GR00T's formalize_language: lower case, then drop every character that is neither a word character
    //! nor whitespace (bytes of multi-byte UTF-8 characters are kept).
    static std::string formalize(std::string const& task);

private:
    cudaStream_t mStream;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    vla::TrtEngine mVisual;
    vla::TrtEngine mPrefix;
    rt::Tensor mContextMemory;
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;

    std::vector<float> preprocessViewN15(Gr00tView const& view) const;

    bool mN15Pipeline{false};
    bool mTextAfterImages{false};
    int32_t mShortestEdge{256};
    double mCropFraction{0.95}; //!< N1.6's crop_fraction or N1.5's crop scale; double: int(side * fraction) in Python
    int32_t mImageHeight{0};
    int32_t mImageWidth{0};
    int32_t mImageTokens{0};
    int32_t mMaxViews{0};
    int32_t mMaxTokens{0};
    int32_t mHidden{0};
    int64_t mImageTokenId{0};
    bool mFormalize{true};
    std::string mPromptPrefix, mImagePrefix, mImageContext, mImageSuffix, mPromptSuffix;

    rt::Tensor mPixelsHost; //!< pinned FP16 [maxViews, 3, H, W]
    rt::Tensor mPixels;
    rt::Tensor mImageFeatures; //!< [maxViews * imageTokens, hidden] FP16
    rt::Tensor mTokensHost;    //!< pinned INT64 [maxTokens]
    rt::Tensor mTokens;
    rt::Tensor mFeatures; //!< [1, maxTokens, hidden] FP16, reshaped per call
    std::vector<uint8_t> mImageMask;
    std::vector<int64_t> mTokenIds;
    cudaEvent_t mEvents[3]{};
};

} // namespace gr00t
} // namespace trt_edgellm
