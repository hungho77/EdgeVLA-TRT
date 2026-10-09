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

//! Build RLDX's prompt tables for a task (formalized, tokenized, 4 frames x 2 views of 8 x 8 merged tokens) and write
//! them as raw files for checking against the official processor and get_rope_index:
//!   rldx_prompt_dump TOKENIZER_DIR OUT_PREFIX "task"

#include "rldxPrompt.h"
#include "rldxText.h"
#include "tokenizer/tokenizer.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace
{

template <typename T>
void write(std::vector<T> const& values, std::string const& path)
{
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<char const*>(values.data()), values.size() * sizeof(T));
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: %s TOKENIZER_DIR OUT_PREFIX TASK\n", argv[0]);
        return 2;
    }
    trt_edgellm::tokenizer::Tokenizer tokenizer;
    if (!tokenizer.loadFromHF(argv[1]))
    {
        return 1;
    }
    auto const tokens = tokenizer.encode(trt_edgellm::rldx::formalizeLanguage(argv[3]));
    auto const prompt = trt_edgellm::rldx::buildPrompt(std::vector<int32_t>(tokens.begin(), tokens.end()),
        std::vector<std::pair<int32_t, int32_t>>(8, {8, 8}), 2, 64, trt_edgellm::rldx::RldxTextGeometry{});
    std::string const prefix = argv[2];
    write(prompt.inputIds, prefix + "_ids.bin");
    write(prompt.visualIndex, prefix + "_visual_index.bin");
    write(prompt.cos, prefix + "_cos.bin");
    write(prompt.sin, prefix + "_sin.bin");
    write(prompt.pool, prefix + "_pool.bin");
    write(prompt.keepIndex, prefix + "_keep.bin");
    write(prompt.cosCompressed, prefix + "_cos_b.bin");
    write(prompt.sinCompressed, prefix + "_sin_b.bin");
    std::printf("S0 %zu S %ld S' %ld\n", prompt.inputIds.size(), prompt.sequence(), prompt.compressed());
    return 0;
}
