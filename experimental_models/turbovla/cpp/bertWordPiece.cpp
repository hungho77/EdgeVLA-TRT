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

#include "bertWordPiece.h"

#include "common/checkMacros.h"
#include "tokenizer/tokenizerUtils.h"
#include "tokenizer/unicodeData.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

namespace trt_edgellm
{
namespace turbovla
{
namespace
{

using tokenizer::codepointFlags;

bool isChinese(uint32_t cp)
{
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF)
        || (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF)
        || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

//! BERT treats every non-alphanumeric printable ASCII character as punctuation, besides Unicode's P category.
bool isPunctuation(uint32_t cp, codepointFlags flags)
{
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)
        || flags.isPunctuation;
}

uint32_t baseLetter(uint32_t cp)
{
    auto const& ranges = tokenizer::unicodeRangesNfd;
    auto const it = std::upper_bound(ranges.begin(), ranges.end(), cp,
        [](uint32_t value, tokenizer::rangeNfd const& range) { return value < range.first; });
    if (it == ranges.begin())
    {
        return cp;
    }
    auto const& range = *(it - 1);
    return cp <= range.last ? range.nfd : cp;
}

} // namespace

BertWordPiece::BertWordPiece(std::string const& tokenizerJson)
{
    std::ifstream file(tokenizerJson);
    ELLM_CHECK(file.good(), "BertWordPiece: cannot open " + tokenizerJson);
    nlohmann::json const config = nlohmann::json::parse(file);
    auto const& model = config.at("model");
    // Older tokenizer.json files (bert-base-uncased's among them) carry no model type.
    ELLM_CHECK(model.value("type", "WordPiece") == "WordPiece" && model.value("continuing_subword_prefix", "") == "##",
        "BertWordPiece: " + tokenizerJson + " is not a WordPiece model");
    auto const& normalizer = config.at("normalizer");
    ELLM_CHECK(normalizer.value("type", "") == "BertNormalizer" && normalizer.value("lowercase", false),
        "BertWordPiece: only uncased BERT normalization is supported");
    for (auto const& [token, id] : model.at("vocab").items())
    {
        mVocab.emplace(token, id.get<int64_t>());
    }
    mMaxCharsPerWord = model.value("max_input_chars_per_word", 100);
    mCls = tokenId("[CLS]");
    mSep = tokenId("[SEP]");
    mPad = tokenId("[PAD]");
    mUnk = tokenId(model.value("unk_token", std::string("[UNK]")));
}

int64_t BertWordPiece::tokenId(std::string const& token) const
{
    auto const it = mVocab.find(token);
    ELLM_CHECK(it != mVocab.end(), "BertWordPiece: no token " + token);
    return it->second;
}

void BertWordPiece::wordPiece(std::string const& word, std::vector<int64_t>& out) const
{
    std::vector<uint32_t> const cpts = tokenizer::unicodeCptsFromUtf8(word);
    if (static_cast<int32_t>(cpts.size()) > mMaxCharsPerWord)
    {
        out.push_back(mUnk);
        return;
    }
    size_t const mark = out.size();
    size_t start = 0;
    while (start < cpts.size())
    {
        size_t end = cpts.size();
        int64_t match = -1;
        while (start < end)
        {
            std::string piece = start > 0 ? "##" : "";
            for (size_t i = start; i < end; ++i)
            {
                piece += tokenizer::unicodeCptToUtf8(cpts[i]);
            }
            auto const it = mVocab.find(piece);
            if (it != mVocab.end())
            {
                match = it->second;
                break;
            }
            --end;
        }
        if (match < 0)
        {
            out.resize(mark);
            out.push_back(mUnk);
            return;
        }
        out.push_back(match);
        start = end;
    }
}

std::vector<int64_t> BertWordPiece::encode(
    std::string const& text, int32_t maxLength, std::vector<uint8_t>* attentionMask) const
{
    std::vector<std::string> words;
    std::string word;
    auto flush = [&] {
        if (!word.empty())
        {
            words.push_back(std::move(word));
            word.clear();
        }
    };
    for (uint32_t cp : tokenizer::unicodeCptsFromUtf8(text))
    {
        codepointFlags flags = tokenizer::unicodeCptFlags(cp);
        if (cp == 0 || cp == 0xFFFD || (flags.isControl && cp != '\t' && cp != '\n' && cp != '\r'))
        {
            continue;
        }
        if (flags.isWhitespace)
        {
            flush();
            continue;
        }
        cp = baseLetter(cp);
        flags = tokenizer::unicodeCptFlags(cp);
        if (flags.isAccentMark)
        {
            continue;
        }
        auto const lower = tokenizer::unicodeMapLowercase.find(cp);
        if (lower != tokenizer::unicodeMapLowercase.end())
        {
            cp = lower->second;
        }
        if (isChinese(cp) || isPunctuation(cp, flags))
        {
            flush();
            words.push_back(tokenizer::unicodeCptToUtf8(cp));
            continue;
        }
        word += tokenizer::unicodeCptToUtf8(cp);
    }
    flush();

    std::vector<int64_t> pieces;
    for (auto const& w : words)
    {
        wordPiece(w, pieces);
    }
    ELLM_CHECK(maxLength >= 2, "BertWordPiece: maxLength must fit [CLS] and [SEP]");
    pieces.resize(std::min(pieces.size(), static_cast<size_t>(maxLength - 2)));

    std::vector<int64_t> ids(static_cast<size_t>(maxLength), mPad);
    ids[0] = mCls;
    std::copy(pieces.begin(), pieces.end(), ids.begin() + 1);
    ids[pieces.size() + 1] = mSep;
    if (attentionMask != nullptr)
    {
        attentionMask->assign(static_cast<size_t>(maxLength), 0);
        std::fill(attentionMask->begin(), attentionMask->begin() + static_cast<int64_t>(pieces.size()) + 2, 1);
    }
    return ids;
}

} // namespace turbovla
} // namespace trt_edgellm
