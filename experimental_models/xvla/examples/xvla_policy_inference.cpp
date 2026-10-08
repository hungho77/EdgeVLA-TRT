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

//! X-VLA policy call: request JSON in ({"task", "state", "cameras": {name: image path}}), action chunk out.

#include "xvlaPolicy.h"

#include "runtime/imageUtils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
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
            "usage: %s --engineDir DIR --inputFile request.json [--noise x1.f32] [--output out.json] [--iters N] "
            "[--cudaGraph 1]\n",
            argv[0]);
        return 2;
    }
    nlohmann::json const request = nlohmann::json::parse(std::ifstream(inputFile));
    std::vector<rt::imageUtils::ImageData> images;
    std::vector<std::string> names;
    for (auto const& [camera, path] : request.at("cameras").items())
    {
        images.push_back(rt::imageUtils::loadRgbImageFromFile(path.get<std::string>()));
        names.push_back(camera);
    }
    std::vector<xvla::XvlaView> views;
    for (size_t i = 0; i < images.size(); ++i)
    {
        views.push_back(xvla::XvlaView{
            names[i], images[i].data(), static_cast<int32_t>(images[i].height), static_cast<int32_t>(images[i].width)});
    }
    std::vector<float> const state = request.at("state").get<std::vector<float>>();
    std::string const task = request.at("task").get<std::string>();
    std::vector<float> const noise
        = argOf(argc, argv, "--noise").empty() ? std::vector<float>{} : readFloats(argOf(argc, argv, "--noise"));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    {
        xvla::XvlaPolicy policy(engineDir, stream);
        policy.setUseCudaGraph(argOf(argc, argv, "--cudaGraph", "1") != "0");
        xvla::XvlaChunk const chunk = policy.act(views, state, task, noise);
        int32_t const iters = std::stoi(argOf(argc, argv, "--iters", "0"));
        if (iters > 0)
        {
            std::vector<float> vision, encoder, denoise;
            for (int32_t i = 0; i < iters; ++i)
            {
                xvla::XvlaChunk const timed = policy.act(views, state, task, noise);
                vision.push_back(timed.visionMs);
                encoder.push_back(timed.encoderMs);
                denoise.push_back(timed.denoiseMs);
            }
            std::printf("median ms over %d calls: vision %.1f, encoder %.1f, denoise %.1f, engines %.1f\n", iters,
                median(vision), median(encoder), median(denoise), median(vision) + median(encoder) + median(denoise));
        }
        std::string const output = argOf(argc, argv, "--output");
        if (!output.empty())
        {
            nlohmann::json out;
            out["actions"] = chunk.actions;
            out["token_ids"] = chunk.tokenIds;
            nlohmann::json pixels = nlohmann::json::array();
            for (auto const& view : views)
            {
                pixels.push_back(policy.preprocessView(view.rgb, view.height, view.width));
            }
            out["pixel_values"] = pixels;
            std::ofstream(output) << out.dump();
        }
        std::printf("chunk %d x %d, first action:", policy.chunkSize(), policy.actionDim());
        for (int32_t d = 0; d < policy.actionDim(); ++d)
        {
            std::printf(" %.3f", chunk.actions[d]);
        }
        std::printf("\n");
    }
    cudaStreamDestroy(stream);
    return 0;
}
