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

//! SmolVLA policy call: request JSON in ({"task", "state", "cameras": {name: image path}}), action chunk out.

#include "smolvlaPolicy.h"

#include "runtime/imageUtils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

std::string argOf(int argc, char** argv, char const* flag, std::string const& fallback = "")
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::strcmp(argv[i], flag) == 0)
        {
            return argv[i + 1];
        }
    }
    return fallback;
}

std::vector<float> readFloats(std::string const& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<float> data(static_cast<size_t>(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    return data;
}

float median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char** argv)
{
    std::string const engineDir = argOf(argc, argv, "--engineDir");
    std::string const inputFile = argOf(argc, argv, "--inputFile");
    if (engineDir.empty() || inputFile.empty())
    {
        std::fprintf(stderr,
            "usage: %s --engineDir DIR --inputFile request.json [--noise x0.f32] [--output out.json] [--iters N]\n"
            "          [--cudaGraph 1]\n",
            argv[0]);
        return 2;
    }
    nlohmann::json const request = nlohmann::json::parse(std::ifstream(inputFile));
    std::vector<rt::imageUtils::ImageData> images;
    smolvla::SmolvlaObservation observation;
    for (auto const& [camera, path] : request.at("cameras").items())
    {
        images.push_back(rt::imageUtils::loadRgbImageFromFile(path.get<std::string>()));
    }
    size_t index = 0;
    for (auto const& [camera, path] : request.at("cameras").items())
    {
        auto const& image = images[index++];
        observation.views.push_back(smolvla::SmolvlaView{
            camera, image.data(), static_cast<int32_t>(image.height), static_cast<int32_t>(image.width)});
    }
    observation.state = request.at("state").get<std::vector<float>>();
    observation.task = request.at("task").get<std::string>();
    std::vector<float> const noise
        = argOf(argc, argv, "--noise").empty() ? std::vector<float>{} : readFloats(argOf(argc, argv, "--noise"));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    smolvla::SmolvlaPolicy policy(engineDir, stream);
    policy.setUseCudaGraph(argOf(argc, argv, "--cudaGraph", "1") != "0");
    smolvla::SmolvlaChunk chunk = policy.act(observation, noise);

    int32_t const iters = std::stoi(argOf(argc, argv, "--iters", "0"));
    if (iters > 0)
    {
        std::vector<float> visual, prefix, denoise;
        for (int32_t i = 0; i < iters; ++i)
        {
            smolvla::SmolvlaChunk const timed = policy.act(observation, noise);
            visual.push_back(timed.visualMs);
            prefix.push_back(timed.prefixMs);
            denoise.push_back(timed.denoiseMs);
        }
        std::printf("median ms over %d calls: visual %.1f, prefix %.1f, denoise %.1f, engines %.1f\n", iters,
            median(visual), median(prefix), median(denoise), median(visual) + median(prefix) + median(denoise));
    }

    std::string const output = argOf(argc, argv, "--output");
    if (!output.empty())
    {
        nlohmann::json out;
        out["normalized"] = chunk.normalized;
        out["robot_actions"] = chunk.robot;
        out["token_ids"] = chunk.tokenIds;
        out["chunk_size"] = policy.chunkSize();
        out["action_dim"] = policy.actionDim();
        std::ofstream(output) << out.dump();
    }
    std::printf("chunk %d x %d, first action:", policy.chunkSize(), policy.actionDim());
    for (int32_t d = 0; d < policy.actionDim(); ++d)
    {
        std::printf(" %.3f", chunk.robot[d]);
    }
    std::printf("\n");
    cudaStreamDestroy(stream);
    return 0;
}
