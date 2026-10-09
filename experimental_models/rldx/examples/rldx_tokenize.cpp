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

//! Formalize and tokenize each stdin line as RLDX's processor does, printing one JSON array of ids per line, for
//! checking against the HF tokenizer:  rldx_tokenize TOKENIZER_DIR < instructions.txt

#include "rldxText.h"
#include "tokenizer/tokenizer.h"

#include <cstdio>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s TOKENIZER_DIR < lines\n", argv[0]);
        return 2;
    }
    trt_edgellm::tokenizer::Tokenizer tokenizer;
    if (!tokenizer.loadFromHF(argv[1]))
    {
        std::fprintf(stderr, "cannot load the tokenizer from %s\n", argv[1]);
        return 1;
    }
    std::string line;
    while (std::getline(std::cin, line))
    {
        auto const ids = tokenizer.encode(trt_edgellm::rldx::formalizeLanguage(line));
        std::printf("%s\n", nlohmann::json(ids).dump().c_str());
    }
    return 0;
}
