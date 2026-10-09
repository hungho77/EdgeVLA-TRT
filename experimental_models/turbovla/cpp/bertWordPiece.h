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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace turbovla
{

//! An uncased BERT tokenizer (HF tokenizers' BertNormalizer with lowercase and accent stripping, BertPreTokenizer,
//! WordPiece, "[CLS] $A [SEP]"), loaded from a tokenizer.json. Accents are stripped through the codepoint-to-base
//! NFD table the BPE tokenizers use, which covers composed Latin, Greek and Cyrillic letters.
class BertWordPiece
{
public:
    explicit BertWordPiece(std::string const& tokenizerJson);

    //! [CLS] tokens [SEP], truncated to \p maxLength and padded with [PAD] to it; \p attentionMask is 1 on the
    //! [CLS] .. [SEP] span.
    std::vector<int64_t> encode(std::string const& text, int32_t maxLength, std::vector<uint8_t>* attentionMask) const;

    int64_t tokenId(std::string const& token) const;

private:
    void wordPiece(std::string const& word, std::vector<int64_t>& out) const;

    std::unordered_map<std::string, int64_t> mVocab;
    int64_t mCls{101};
    int64_t mSep{102};
    int64_t mPad{0};
    int64_t mUnk{100};
    int32_t mMaxCharsPerWord{100};
};

} // namespace turbovla
} // namespace trt_edgellm
