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

//! Tokenize each stdin line with BertWordPiece and print one JSON array of ids per line, for checking it against the
//! HF tokenizer:  turbovla_tokenize tokenizer.json 64 < instructions.txt

#include "bertWordPiece.h"

#include <cstdio>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s tokenizer.json MAX_LENGTH < lines\n", argv[0]);
        return 2;
    }
    trt_edgellm::turbovla::BertWordPiece const tokenizer(argv[1]);
    int32_t const maxLength = std::stoi(argv[2]);
    std::string line;
    while (std::getline(std::cin, line))
    {
        std::vector<uint8_t> mask;
        std::vector<int64_t> ids = tokenizer.encode(line, maxLength, &mask);
        size_t used = 0;
        while (used < mask.size() && mask[used] != 0)
        {
            ++used;
        }
        ids.resize(used);
        std::printf("%s\n", nlohmann::json(ids).dump().c_str());
    }
    return 0;
}
