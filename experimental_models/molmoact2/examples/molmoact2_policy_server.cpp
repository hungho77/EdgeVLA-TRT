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

//! MolmoAct2 policy server for a robot or a simulator: one request in, one action chunk out, over stdin / stdout or
//! TCP (--port, see vlaServer.h for the framing and the inline-frame protocol).
//!
//! Request:  {"frames": [{"name": "image", ...}, {"name": "wrist_image", ...}] + raw RGB bytes,
//!              or "cameras": {"image": "agentview.png", "wrist_image": "wrist.png"},
//!              the ready line's cameras, as the checkpoint's dataset recorded them
//!            "state": [...],                           the checkpoint's state layout
//!            "task": "pick the cube",
//!            "rtc": {"overlap": 5, "frozen": 2,        optional, real-time chunking from the previous reply
//!                    "ramp_rate": 6.0, "start_row": 5},
//!            "noise": [[...] x 32] x 10,               optional, the initial flow sample (else the seeded generator)
//!            "seed": 0,                                optional, reseeds the generator
//!            "normalized": true,                       optional, also return the normalized chunk
//!            "reset": true}                            optional, start of an episode
//! Reply:    {"actions": [[...], ...], "prompt_tokens": n, "timing_ms": {...}}  or {"error": "..."}
//! Actions are the checkpoint's, unnormalized (the gripper dim as the model returns it, in [-1, 1]).

#include "molmoact2Policy.h"

#include "vlaServer.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

namespace
{

Json rowsOf(std::vector<float> const& values, int32_t rows, int32_t dim)
{
    Json out = Json::array();
    for (int32_t t = 0; t < rows; ++t)
    {
        out.push_back(std::vector<float>(
            values.begin() + static_cast<int64_t>(t) * dim, values.begin() + static_cast<int64_t>(t + 1) * dim));
    }
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    std::string const engineDir = vla::argOf(argc, argv, "--engineDir");
    if (engineDir.empty())
    {
        std::fprintf(stderr, "usage: %s --engineDir DIR [--seed N] [--cudaGraph 1] [--port N [--host H]]\n", argv[0]);
        return 2;
    }

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    molmoact2::MolmoAct2Policy policy(engineDir, stream);
    policy.setNoiseSeed(std::stoull(vla::argOf(argc, argv, "--seed", "0")));
    policy.setUseCudaGraph(vla::argOf(argc, argv, "--cudaGraph", "1") != "0");

    Json const ready
        = {{"ready", true}, {"family", "molmoact2"}, {"cameras", policy.cameras()}, {"state_dim", policy.stateDim()},
            {"action_dim", policy.actionDim()}, {"chunk", policy.chunkSize()}, {"rtc", "overlap_frozen"}};
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
            if (in.contains("seed"))
            {
                policy.setNoiseSeed(in.at("seed").get<uint64_t>());
            }
            std::vector<vla::NamedFrame> const frames = vla::collectFrames(request, "cameras");
            std::vector<molmoact2::MolmoAct2View> views;
            for (auto const& frame : frames)
            {
                views.push_back({frame.name, frame.image.data(), static_cast<int32_t>(frame.image.height),
                    static_cast<int32_t>(frame.image.width)});
            }
            std::vector<float> noise;
            if (in.contains("noise"))
            {
                for (auto const& row : in.at("noise"))
                {
                    auto const values = row.get<std::vector<float>>();
                    noise.insert(noise.end(), values.begin(), values.end());
                }
            }
            molmoact2::MolmoAct2Rtc rtc;
            bool const useRtc = in.contains("rtc");
            if (useRtc)
            {
                auto const& r = in.at("rtc");
                rtc.overlapSteps = r.at("overlap").get<int32_t>();
                rtc.frozenSteps = r.value("frozen", 0);
                rtc.rampRate = r.value("ramp_rate", 6.0F);
                rtc.startRow = r.value("start_row", -1);
            }
            std::string const task
                = in.contains("task") ? in.at("task").get<std::string>() : in.at("instruction").get<std::string>();
            molmoact2::MolmoAct2Chunk const chunk
                = policy.act(views, in.at("state").get<std::vector<float>>(), task, noise, useRtc ? &rtc : nullptr);

            Json reply;
            reply["actions"] = rowsOf(chunk.actions, policy.chunkSize(), policy.actionDim());
            if (in.value("normalized", false))
            {
                reply["normalized"] = rowsOf(chunk.normalized, policy.chunkSize(),
                    static_cast<int32_t>(chunk.normalized.size()) / policy.chunkSize());
            }
            reply["prompt_tokens"] = chunk.promptTokens;
            reply["timing_ms"] = {{"host", chunk.hostMs}, {"vision", chunk.visionMs}, {"llm", chunk.prefixMs},
                {"action", chunk.actionMs},
                {"total", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()}};
            return reply;
        },
        [&] { policy.resetEpisode(); });
    cudaStreamDestroy(stream);
    return 0;
}
