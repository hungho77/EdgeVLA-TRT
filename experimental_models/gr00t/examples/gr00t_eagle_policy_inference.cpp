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

//! GR00T N1.5 / N1.6 policy call on a raw observation: Eagle backbone -> action head -> absolute joint targets.
//! Request JSON in ({"task", "state", "cameras": {video key: image path}}), action chunk out; cameras are fed in
//! processing.json's video_keys order.

#include "gr00tEagleBackbone.h"
#include "gr00tN17Policy.h"

#include "runtime/imageUtils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
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
    std::string const backboneDir = argOf(argc, argv, "--backboneDir");
    std::string const actionDir = argOf(argc, argv, "--actionDir");
    std::string const inputFile = argOf(argc, argv, "--inputFile");
    if (backboneDir.empty() || actionDir.empty() || inputFile.empty())
    {
        std::fprintf(stderr,
            "usage: %s --backboneDir DIR --actionDir DIR --inputFile request.json [--noise x0.f32] [--features f.f32] "
            "[--output "
            "out.json]\n"
            "          [--iters N] [--cudaGraph 1]\n",
            argv[0]);
        return 2;
    }
    nlohmann::json const request = nlohmann::json::parse(std::ifstream(inputFile));
    nlohmann::json const processing = nlohmann::json::parse(std::ifstream(actionDir + "/processing.json"));
    std::vector<rt::imageUtils::ImageData> images;
    std::vector<gr00t::Gr00tView> views;
    for (auto const& key : processing.at("video_keys"))
    {
        images.push_back(rt::imageUtils::loadRgbImageFromFile(request.at("cameras").at(key.get<std::string>())));
    }
    for (auto const& image : images)
    {
        views.push_back(
            gr00t::Gr00tView{image.data(), static_cast<int32_t>(image.height), static_cast<int32_t>(image.width)});
    }
    std::vector<float> const state = request.at("state").get<std::vector<float>>();
    std::string const task = request.at("task").get<std::string>();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    gr00t::Gr00tEagleBackbone backbone(backboneDir, stream);
    gr00t::Gr00tN17Policy policy(actionDir, stream);
    policy.runner().setUseCudaGraph(argOf(argc, argv, "--cudaGraph", "1") != "0");
    auto const& config = policy.runner().config();
    int64_t const noiseSize = static_cast<int64_t>(config.actionHorizon) * config.actionDim;
    std::vector<float> noiseHost = argOf(argc, argv, "--noise").empty() ? std::vector<float>(noiseSize, 0.0F)
                                                                        : readFloats(argOf(argc, argv, "--noise"));
    if (argOf(argc, argv, "--noise").empty())
    {
        std::mt19937_64 generator(0);
        std::normal_distribution<float> normal;
        std::generate(noiseHost.begin(), noiseHost.end(), [&] { return normal(generator); });
    }
    if (static_cast<int64_t>(noiseHost.size()) != noiseSize)
    {
        std::fprintf(
            stderr, "noise holds %zu values, expected %lld\n", noiseHost.size(), static_cast<long long>(noiseSize));
        return 2;
    }
    rt::Tensor noise(rt::Coords({config.actionHorizon, config.actionDim}), rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "gr00t::noise");
    cudaMemcpyAsync(noise.rawPointer(), noiseHost.data(), noiseSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    rt::Tensor const* features = nullptr;
    auto step = [&]() {
        features = &backbone.encode(views, task);
        return policy.act(*features, backbone.imageMask(), state, noise, stream);
    };
    std::vector<float> const actions = step();
    std::string const featuresFile = argOf(argc, argv, "--features");
    if (!featuresFile.empty())
    {
        // Backbone features as FP32 [tokens, hidden], for comparing prefix paths and references.
        std::vector<__half> half(static_cast<size_t>(features->getShape().volume()));
        cudaMemcpy(half.data(), features->rawPointer(), half.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        std::vector<float> values(half.size());
        std::transform(half.begin(), half.end(), values.begin(), [](__half v) { return __half2float(v); });
        std::ofstream(featuresFile, std::ios::binary)
            .write(reinterpret_cast<char const*>(values.data()), static_cast<std::streamsize>(values.size() * 4));
    }

    int32_t const iters = std::stoi(argOf(argc, argv, "--iters", "0"));
    if (iters > 0)
    {
        std::vector<float> visual, prefix, host, total;
        for (int32_t i = 0; i < iters; ++i)
        {
            auto const start = std::chrono::steady_clock::now();
            step();
            total.push_back(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count());
            visual.push_back(backbone.visualMs());
            prefix.push_back(backbone.prefixMs());
            host.push_back(backbone.hostMs());
        }
        std::printf(
            "median ms over %d calls: host %.1f, visual %.1f, prefix %.1f, policy step (preprocessing to actions) "
            "%.1f\n",
            iters, median(host), median(visual), median(prefix), median(total));
    }

    std::string const output = argOf(argc, argv, "--output");
    if (!output.empty())
    {
        nlohmann::json out;
        out["robot_actions"] = actions;
        out["action_dim"] = static_cast<int32_t>(actions.size()) / policy.processing().actionHorizon();
        out["token_ids"] = backbone.tokenIds();
        float const* model = policy.lastModelActions();
        out["model_actions"] = std::vector<float>(model, model + noiseSize);
        nlohmann::json pixels = nlohmann::json::array();
        for (auto const& view : views)
        {
            pixels.push_back(backbone.preprocessView(view));
        }
        out["pixel_values"] = pixels;
        std::ofstream(output) << out.dump();
    }
    int32_t const dim = static_cast<int32_t>(actions.size()) / policy.processing().actionHorizon();
    std::printf("chunk %d x %d, first action:", policy.processing().actionHorizon(), dim);
    for (int32_t d = 0; d < dim; ++d)
    {
        std::printf(" %.3f", actions[d]);
    }
    std::printf("\n");
    cudaStreamDestroy(stream);
    return 0;
}
