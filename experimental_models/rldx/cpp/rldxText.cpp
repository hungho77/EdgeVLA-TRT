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

#include "rldxText.h"

#include "tokenizer/tokenizerUtils.h"
#include "tokenizer/unicodeData.h"

namespace trt_edgellm
{
namespace rldx
{

std::string formalizeLanguage(std::string const& text)
{
    std::string out;
    for (uint32_t cp : tokenizer::unicodeCptsFromUtf8(text))
    {
        auto const lower = tokenizer::unicodeMapLowercase.find(cp);
        if (lower != tokenizer::unicodeMapLowercase.end())
        {
            cp = lower->second;
        }
        tokenizer::codepointFlags const flags = tokenizer::unicodeCptFlags(cp);
        if (flags.isLetter || flags.isNumber || flags.isWhitespace || cp == '_')
        {
            out += tokenizer::unicodeCptToUtf8(cp);
        }
    }
    return out;
}

} // namespace rldx
} // namespace trt_edgellm
