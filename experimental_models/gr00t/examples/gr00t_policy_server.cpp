/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! GR00T policy server for a robot or a simulator: one request in, one action chunk out, over stdin / stdout or
//! TCP (--port, see vlaServer.h for the framing and the inline-frame protocol). The backbone is N1.7's Edge-LLM VLM
//! (--llmEngineDir, --multimodalEngineDir) or N1.5 / N1.6's Eagle (--eagleBackboneDir).
//!
//! Request:  {"frames": [{"name": "top", "height": H, "width": W, "bytes": N}, ...] + raw RGB bytes,
//!                                                   raw camera frames in processing.json's video_keys order; the
//!                                                   server applies the official image and language preprocessing
//!              or "images": ["top.png", ...],       file paths. N1.7: frames after GR00T's eval image transform and
//!                                                   an instruction after its language formalization (what
//!                                                   gr00t_policy_client.py sends); Eagle: raw frames
//!            "state": [...],                        raw state, groups concatenated in modality order
//!            "task": "pick the cube",               ("instruction" is accepted too)
//!            "rtc": {"overlap": 8, "frozen": 2,     optional, chunk inpainting from the previous reply;
//!                    "ramp_rate": 6.0,              start_row: rows of it executed when this chunk starts
//!                    "start_row": 8},               (default action_horizon - overlap)
//!            "seed": 0, "noise_file": "noise.f32",  optional; noise_file holds [max_horizon, max_action_dim]
//!            "reset": true,                         optional, start of an episode
//!            "debug": true}                         optional, also return the normalized model actions
//! Reply:    {"actions": [[...], ...],               absolute raw actions [action_horizon][raw action dim]
//!            "timing_ms": {...}}  or {"error": "..."}

#include "gr00tN17Policy.h"
#ifdef GR00T_EAGLE_BACKBONE
#include "gr00tEagleBackbone.h"
#endif

#include "common/tensor.h"
#include "common/trtUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "vlaBackbone.h"
#include "vlaServer.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{

constexpr int32_t kCaptureSlot = 1;
constexpr char const* kImagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>";

//! GR00T N1.7's prompt: images first, then the instruction, in one user turn without a system prompt.
std::string buildPrompt(std::string const& instruction, size_t numImages)
{
    std::string text = "<|im_start|>user\n";
    for (size_t i = 0; i < numImages; ++i)
    {
        text += kImagePlaceholder;
    }
    return text + instruction + "<|im_end|>\n";
}

//! GR00T's formalize_language: lower case without punctuation (re.sub(r"[^\w\s]", "", text.lower())).
std::string formalizeLanguage(std::string const& text)
{
    std::string out;
    for (unsigned char const c : text)
    {
        if (c >= 0x80 || std::isalnum(c) || c == '_' || std::isspace(c))
        {
            out.push_back(static_cast<char>(c < 0x80 ? std::tolower(c) : c));
        }
    }
    return out;
}

//! The GPU time of one of the core runtime's profiled stages since the last gTimer.reset(); 0 when it did not run
//! (the vision encoder is skipped for images the runtime has cached).
double stageMs(std::string const& stage)
{
    auto const data = gTimer.getTimingData(stage);
    return data ? data->getTotalGpuTimeMs() : 0.0;
}

double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, char** argv)
{
    std::string const llmDir = vla::argOf(argc, argv, "--llmEngineDir");
    std::string const visDir = vla::argOf(argc, argv, "--multimodalEngineDir");
    std::string const actionDir = vla::argOf(argc, argv, "--actionEngineDir");
    std::string const eagleDir = vla::argOf(argc, argv, "--eagleBackboneDir");
    bool const eagle = !eagleDir.empty();
    if (actionDir.empty() || (!eagle && (llmDir.empty() || visDir.empty())))
    {
        std::fprintf(stderr,
            "usage: %s (--llmEngineDir DIR --multimodalEngineDir DIR | --eagleBackboneDir DIR) --actionEngineDir DIR\n"
            "          [--cudaGraph 1] [--port N [--host H]]\n"
            "  actionEngineDir holds the action engines, config.json and processing.json.\n",
            argv[0]);
        return 2;
    }
#ifndef GR00T_EAGLE_BACKBONE
    if (eagle)
    {
        std::fprintf(stderr, "this build has no Eagle backbone (it needs OpenCV)\n");
        return 2;
    }
#endif

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::unique_ptr<void, DlDeleter> pluginHandle;
    std::unique_ptr<rt::LLMInferenceRuntime> backbone;
    if (!eagle)
    {
        pluginHandle = loadEdgellmPluginLib();
        std::unordered_map<std::string, std::string> const noLora;
        backbone = std::make_unique<rt::LLMInferenceRuntime>(llmDir, visDir, noLora, stream);
        // The core runtime times its vision encoder and LLM prefill (CUDA events) only with profiling on.
        setProfilingEnabled(true);
    }
#ifdef GR00T_EAGLE_BACKBONE
    std::unique_ptr<gr00t::Gr00tEagleBackbone> eagleBackbone;
    if (eagle)
    {
        eagleBackbone = std::make_unique<gr00t::Gr00tEagleBackbone>(eagleDir, stream);
    }
#endif
    gr00t::Gr00tN17Policy policy(actionDir, stream);
    policy.runner().setUseCudaGraph(vla::argOf(argc, argv, "--cudaGraph", "1") != "0");
    gr00t::Gr00tProcessing const& processing = policy.processing();
    auto const& cfg = policy.runner().config();
    int32_t const imageTokenId = std::stoi(vla::argOf(argc, argv, "--imageTokenId", "151655"));

    size_t const noiseCount = static_cast<size_t>(cfg.actionHorizon) * cfg.actionDim;
    rt::Tensor noiseHost(rt::Coords(std::vector<int64_t>{cfg.actionHorizon, cfg.actionDim}), rt::DeviceType::kCPU,
        nvinfer1::DataType::kFLOAT, "gr00t::noiseHost");
    rt::Tensor noise(rt::Coords(std::vector<int64_t>{cfg.actionHorizon, cfg.actionDim}), rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "gr00t::noise");
    std::mt19937_64 generator(0);

    Json const processingJson = Json::parse(std::ifstream(actionDir + "/processing.json"));
    Json const ready = {{"ready", true}, {"family", eagle ? "gr00t_eagle" : "gr00t_n17"},
        {"cameras", processingJson.value("video_keys", Json::array())}, {"state_dim", processing.rawStateDim()},
        {"action_dim", processing.rawActionDim()}, {"chunk", processing.actionHorizon()}, {"rtc", "overlap_frozen"},
        {"raw_state_dim", processing.rawStateDim()}, {"raw_action_dim", processing.rawActionDim()},
        {"action_horizon", processing.actionHorizon()}};

    vla::PolicyServer server(argc, argv);
    server.run(
        ready,
        [&](vla::ServerRequest& request) {
            Json const& in = request.header;
            auto const t0 = std::chrono::steady_clock::now();
            if (in.value("reset", false))
            {
                policy.resetEpisode();
            }
            bool const inline_ = !request.frames.empty();
            std::vector<vla::NamedFrame> frames = vla::collectFrames(request, "images");
            std::vector<float> const rawState = in.at("state").get<std::vector<float>>();
            std::string instruction
                = in.contains("task") ? in.at("task").get<std::string>() : in.at("instruction").get<std::string>();

            // Noise: from a file (reproducing a reference), else N(0, 1) from the request seed or the stream.
            if (in.contains("noise_file"))
            {
                std::ifstream f(in.at("noise_file").get<std::string>(), std::ios::binary);
                f.read(static_cast<char*>(noiseHost.rawPointer()),
                    static_cast<std::streamsize>(noiseCount * sizeof(float)));
                if (!f)
                {
                    throw std::runtime_error("noise_file is shorter than [max_horizon, max_action_dim] floats");
                }
            }
            else
            {
                if (in.contains("seed"))
                {
                    generator.seed(in.at("seed").get<uint64_t>());
                }
                std::normal_distribution<float> normal(0.0F, 1.0F);
                float* values = noiseHost.dataPointer<float>();
                for (size_t i = 0; i < noiseCount; ++i)
                {
                    values[i] = normal(generator);
                }
            }
            cudaMemcpyAsync(
                noise.rawPointer(), noiseHost.rawPointer(), noiseCount * sizeof(float), cudaMemcpyHostToDevice, stream);

            rt::Tensor const* features = nullptr;
            std::vector<uint8_t> imageMask;
            Json timing;
#ifdef GR00T_EAGLE_BACKBONE
            if (eagle)
            {
                std::vector<gr00t::Gr00tView> views;
                for (auto const& frame : frames)
                {
                    views.push_back(gr00t::Gr00tView{frame.image.data(), static_cast<int32_t>(frame.image.height),
                        static_cast<int32_t>(frame.image.width)});
                }
                features = &eagleBackbone->encode(views, instruction);
                imageMask = eagleBackbone->imageMask();
                timing["vision"] = eagleBackbone->visualMs();
                timing["llm"] = eagleBackbone->prefixMs();
                timing["host"] = eagleBackbone->hostMs();
            }
#endif
            if (!eagle)
            {
                std::vector<rt::imageUtils::ImageData> images;
                if (inline_)
                {
                // Raw camera frames: the official eval transform and language formalization run here.
#ifdef GR00T_EAGLE_BACKBONE
                    for (auto const& frame : frames)
                    {
                        images.push_back(gr00t::gr00tN17EvalImage(frame.image.data(),
                            static_cast<int32_t>(frame.image.height), static_cast<int32_t>(frame.image.width)));
                    }
                    instruction = formalizeLanguage(instruction);
#else
                    throw std::runtime_error(
                        "raw frames for GR00T N1.7 need the OpenCV build (preprocessed paths work)");
#endif
                }
                else
                {
                    for (auto& frame : frames)
                    {
                        images.push_back(std::move(frame.image));
                    }
                }
                rt::LLMGenerationRequest const llmRequest = vla::makeBackboneRequest(
                    buildPrompt(instruction, images.size()), std::move(images), kCaptureSlot);
                features = vla::runBackbone(*backbone, llmRequest, stream);
                if (features == nullptr)
                {
                    throw std::runtime_error("backbone request failed");
                }
                imageMask = vla::tokenMask(*backbone, imageTokenId);
                cudaStreamSynchronize(stream);
                timing["vision"] = stageMs(metrics::StageNames::kVISION_ENCODER);
                timing["llm"] = stageMs(metrics::StageNames::kLLM_PREFILL);
                gTimer.reset();
            }
            cudaStreamSynchronize(stream);
            double const backboneMs = msSince(t0);

            auto const t1 = std::chrono::steady_clock::now();
            gr00t::Gr00tN17Policy::Rtc rtc;
            bool const useRtc = in.contains("rtc");
            if (useRtc)
            {
                auto const& r = in.at("rtc");
                rtc.overlapSteps = r.at("overlap").get<int32_t>();
                rtc.frozenSteps = r.value("frozen", 0);
                rtc.rampRate = r.value("ramp_rate", 6.0F);
                rtc.startRow = r.value("start_row", -1);
            }
            std::vector<float> const absolute
                = policy.act(*features, imageMask, rawState, noise, stream, useRtc ? &rtc : nullptr);
            double const actionMs = msSince(t1);

            Json reply;
            Json rows = Json::array();
            for (int32_t t = 0; t < processing.actionHorizon(); ++t)
            {
                rows.push_back(
                    std::vector<float>(absolute.begin() + static_cast<int64_t>(t) * processing.rawActionDim(),
                        absolute.begin() + static_cast<int64_t>(t + 1) * processing.rawActionDim()));
            }
            reply["actions"] = rows;
            if (in.value("debug", false))
            {
                // Normalized model output rows [0, action_horizon), every padded column.
                float const* raw = policy.lastModelActions();
                reply["model_actions"]
                    = std::vector<float>(raw, raw + static_cast<int64_t>(processing.actionHorizon()) * cfg.actionDim);
            }
            timing["backbone"] = backboneMs;
            timing["action"] = actionMs;
            timing["total"] = msSince(t0);
            reply["timing_ms"] = timing;
            reply["backbone_tokens"] = features->getShape()[1];
            return reply;
        },
        [&] { policy.resetEpisode(); });
    cudaStreamDestroy(stream);
    return 0;
}
