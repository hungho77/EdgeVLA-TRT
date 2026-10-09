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

#include "molmoact2Text.h"

#include "tokenizer/tokenizerUtils.h"
#include "tokenizer/unicodeData.h"

#include <algorithm>
#include <cmath>
#include <regex>

namespace trt_edgellm
{
namespace molmoact2
{
namespace
{

using Codepoints = std::vector<uint32_t>;

bool isSpace(uint32_t cp)
{
    return tokenizer::unicodeCptFlags(cp).isWhitespace;
}

bool contains(char32_t const* set, uint32_t cp)
{
    for (; *set != 0; ++set)
    {
        if (static_cast<uint32_t>(*set) == cp)
        {
            return true;
        }
    }
    return false;
}

void stripSet(Codepoints& s, char32_t const* set, bool left)
{
    auto pred = [&](uint32_t cp) { return set == nullptr ? isSpace(cp) : contains(set, cp); };
    while (!s.empty() && pred(s.back()))
    {
        s.pop_back();
    }
    if (left)
    {
        auto const first = std::find_if_not(s.begin(), s.end(), pred);
        s.erase(s.begin(), first);
    }
}

std::string toUtf8(Codepoints const& s)
{
    std::string out;
    for (uint32_t cp : s)
    {
        out += tokenizer::unicodeCptToUtf8(cp);
    }
    return out;
}

constexpr char32_t kTrailingPunctuation[] = U".,!?;:…";
constexpr char32_t kTrailingClosers[] = U"\"'”’)]}";
constexpr char32_t kSurrounding[] = U"\"'`“”‘’[](){}";

} // namespace

std::string normalizeQuestion(std::string const& text)
{
    // re.sub(r"\s+", " ", text).strip()
    Codepoints s;
    for (uint32_t cp : tokenizer::unicodeCptsFromUtf8(text))
    {
        if (isSpace(cp))
        {
            if (!s.empty() && s.back() != ' ')
            {
                s.push_back(' ');
            }
            continue;
        }
        s.push_back(cp);
    }
    stripSet(s, nullptr, true);
    if (s.empty())
    {
        return "";
    }
    static std::regex const prefixes[] = {
        std::regex(R"(^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*)", std::regex::icase),
        std::regex(R"(^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+)", std::regex::icase),
    };
    Codepoints previous;
    while (!s.empty() && s != previous)
    {
        previous = s;
        stripSet(s, nullptr, true);
        stripSet(s, kSurrounding, true);
        stripSet(s, nullptr, true);
        for (auto const& pattern : prefixes)
        {
            std::string const utf8 = toUtf8(s);
            std::smatch match;
            if (std::regex_search(utf8, match, pattern))
            {
                s = tokenizer::unicodeCptsFromUtf8(utf8.substr(match.length(0)));
            }
            stripSet(s, nullptr, true);
        }
        stripSet(s, kTrailingPunctuation, false);
        stripSet(s, nullptr, false);
        stripSet(s, kTrailingClosers, false);
        stripSet(s, nullptr, false);
        stripSet(s, kTrailingPunctuation, false);
        stripSet(s, nullptr, false);
    }
    // re.split(r"[.!?]+"), stripped non-empty chunks joined with "; " when there are several.
    std::vector<Codepoints> chunks(1);
    for (uint32_t cp : s)
    {
        if (cp == '.' || cp == '!' || cp == '?')
        {
            if (!chunks.back().empty())
            {
                chunks.emplace_back();
            }
            continue;
        }
        chunks.back().push_back(cp);
    }
    Codepoints joined;
    int32_t kept = 0;
    for (auto& chunk : chunks)
    {
        stripSet(chunk, nullptr, true);
        if (chunk.empty())
        {
            continue;
        }
        if (kept++ > 0)
        {
            joined.push_back(';');
            joined.push_back(' ');
        }
        joined.insert(joined.end(), chunk.begin(), chunk.end());
    }
    Codepoints& result = kept > 1 ? joined : s;
    for (uint32_t& cp : result)
    {
        auto const lower = tokenizer::unicodeMapLowercase.find(cp);
        if (lower != tokenizer::unicodeMapLowercase.end())
        {
            cp = lower->second;
        }
    }
    return toUtf8(result);
}

std::string stateTokens(std::vector<float> const& normalizedState, int32_t bins)
{
    std::string out = "<state_start>";
    for (float value : normalizedState)
    {
        if (std::isnan(value))
        {
            value = 0.0F;
        }
        value = std::clamp(value, -1.0F, 1.0F);
        float const scaled = (value + 1.0F) / 2.0F * static_cast<float>(bins - 1);
        auto const bin = std::clamp(static_cast<int64_t>(std::nearbyint(scaled)), int64_t{0}, int64_t{bins - 1});
        out += "<state_" + std::to_string(bin) + ">";
    }
    return out + "<state_end>";
}

std::string robotPrompt(std::string const& task, std::string const& state, std::string const& setup,
    std::string const& controlMode, int32_t images, std::string const& imageTokens)
{
    std::string prefix;
    if (images == 1)
    {
        prefix = imageTokens;
    }
    for (int32_t i = 0; images > 1 && i < images; ++i)
    {
        prefix += "Image " + std::to_string(i + 1) + imageTokens;
    }
    std::string const stateClause = state.empty() ? "" : " The current state of the robot is " + state + ".";
    return prefix + "<|im_start|>user\n" + "The task is to " + task + ". The setup is <setup_start>" + setup
        + "<setup_end>." + stateClause + " The expected control mode is <control_start>" + controlMode
        + "<control_end>. Given these, what action should the robot take to complete the task?<|im_end|>\n"
        + "<|im_start|>assistant\n<action_output>";
}

} // namespace molmoact2
} // namespace trt_edgellm
