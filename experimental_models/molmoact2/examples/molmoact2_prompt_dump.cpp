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

//! Build MolmoAct2's LIBERO prompt for each stdin line "task<TAB>s0 s1 ... s7" (normalized state) and print its token
//! ids as JSON, for checking against the official processor:  molmoact2_prompt_dump TOKENIZER_DIR < lines

#include "molmoact2Text.h"
#include "tokenizer/tokenizer.h"

#include <cstdio>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    using namespace trt_edgellm;
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s TOKENIZER_DIR < lines\n", argv[0]);
        return 2;
    }
    tokenizer::Tokenizer tok;
    if (!tok.loadFromHF(argv[1]))
    {
        return 1;
    }
    std::string imageTokens = "<im_start>";
    for (int32_t i = 0; i < 196; ++i)
    {
        imageTokens += "<im_patch>";
    }
    imageTokens += "<im_end>";
    std::string line;
    while (std::getline(std::cin, line))
    {
        auto const tab = line.find('\t');
        std::istringstream values(line.substr(tab + 1));
        std::vector<float> state;
        for (float v; values >> v;)
        {
            state.push_back(v);
        }
        std::string const prompt = molmoact2::robotPrompt(molmoact2::normalizeQuestion(line.substr(0, tab)),
            molmoact2::stateTokens(state, 256), "single franka robotic arm in libero", "delta end-effector pose", 2,
            imageTokens);
        auto ids = tok.encode(prompt);
        ids.insert(ids.begin(), 151645);
        std::printf("%s\n", nlohmann::json(ids).dump().c_str());
    }
    return 0;
}
