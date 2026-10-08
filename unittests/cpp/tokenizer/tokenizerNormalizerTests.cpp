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

//! A Prepend normalizer (Llama-2 style SentencePiece tokenizer.json, which prepends "\u2581") marks the start of the
//! text, so the first word encodes like every other word. "_" stands in for "\u2581" to keep the toy vocab
//! byte-complete.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "tokenizer/tokenizer.h"

using namespace trt_edgellm;

namespace
{

std::filesystem::path writePrependTokenizer()
{
    auto const dir = std::filesystem::temp_directory_path() / "edgellm_tok_prepend_normalizer";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "tokenizer.json") << R"JSON({
  "model": {"type": "BPE", "vocab": {"_": 0, "a": 1, "b": 2, "_a": 3, "_b": 4, "<x>": 5}, "merges": ["_ a", "_ b"]},
  "added_tokens": [{"id": 5, "content": "<x>", "special": true}],
  "normalizer": {"type": "Sequence", "normalizers": [
    {"type": "Prepend", "prepend": "_"},
    {"type": "Replace", "pattern": {"String": " "}, "content": "_"}
  ]},
  "pre_tokenizer": null
})JSON";
    std::ofstream(dir / "tokenizer_config.json") << "{}";
    return dir;
}

} // namespace

TEST(TokenizerNormalizerTest, PrependMarksTheFirstWord)
{
    tokenizer::Tokenizer tok;
    ASSERT_TRUE(tok.loadFromHF(writePrependTokenizer()));
    EXPECT_EQ(tok.encode("a b"), (std::vector<tokenizer::Rank>{3, 4}));
    EXPECT_TRUE(tok.encode("").empty());
    // Hugging Face tokenizers (0.19+) normalize each segment between added tokens: these are its ids.
    EXPECT_EQ(tok.encode("a<x>b"), (std::vector<tokenizer::Rank>{3, 5, 4}));
    EXPECT_EQ(tok.encode("<x>a"), (std::vector<tokenizer::Rank>{5, 3}));
    EXPECT_EQ(tok.encode("a<x>"), (std::vector<tokenizer::Rank>{3, 5}));
}
