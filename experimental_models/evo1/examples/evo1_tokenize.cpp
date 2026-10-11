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
//! Tokenize each stdin line as Evo1Policy tokenizes an instruction (stripped, the tokenizer.json of an engine
//! directory) and print one JSON array of ids per line, for checking it against the HF tokenizer:
//!   evo1_tokenize ENGINE_DIR < instructions.txt

#include "tokenizer/tokenizer.h"

#include <cstdio>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s ENGINE_DIR < lines\n", argv[0]);
        return 2;
    }
    trt_edgellm::tokenizer::Tokenizer tokenizer;
    if (!tokenizer.loadFromHF(argv[1]))
    {
        std::fprintf(stderr, "cannot load %s/tokenizer.json\n", argv[1]);
        return 1;
    }
    std::string line;
    while (std::getline(std::cin, line))
    {
        auto const first = line.find_first_not_of(" \t\n\r\f\v");
        std::string const text
            = first == std::string::npos ? "" : line.substr(first, line.find_last_not_of(" \t\n\r\f\v") - first + 1);
        std::printf("%s\n", nlohmann::json(tokenizer.encode(text)).dump().c_str());
    }
    return 0;
}
