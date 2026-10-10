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

//! SmolVLA policy server for a robot or a simulator: one request in, one action chunk out, over stdin / stdout or
//! TCP (--port, see vlaServer.h for the framing and the inline-frame protocol).
//!
//! Request:  {"frames": [{"name": "top", "height": H, "width": W, "bytes": N}, ...] + raw RGB bytes,
//!              or "cameras": {"top": "top.png", ...},   frames keyed by the dataset's camera names
//!            "state": [...],                           robot units, the dataset's state width
//!            "task": "pick the cube",
//!            "rtc": {"delay": 4, "horizon": 20,        optional, real-time chunking from the previous reply
//!                    "start_row": 10},
//!            "seed": 0,                                optional, reseeds x_0
//!            "reset": true}                            optional, start of an episode
//! Reply:    {"actions": [[...], ...], "timing_ms": {...}}  or {"error": "..."}
//! Actions are the dataset's own, in robot units (mean/std unnormalized), as LeRobot returns them.

#include "smolvlaPolicy.h"

#include "vlaServer.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

int main(int argc, char** argv)
{
    std::string const engineDir = vla::argOf(argc, argv, "--engineDir");
    if (engineDir.empty())
    {
        std::fprintf(stderr, "usage: %s --engineDir DIR [--cudaGraph 1] [--port N [--host H]]\n", argv[0]);
        return 2;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    smolvla::SmolvlaPolicy policy(engineDir, stream);
    policy.setUseCudaGraph(vla::argOf(argc, argv, "--cudaGraph", "1") != "0");

    Json const ready
        = {{"ready", true}, {"family", "smolvla"}, {"cameras", policy.cameras()}, {"state_dim", policy.stateDim()},
            {"action_dim", policy.actionDim()}, {"chunk", policy.chunkSize()}, {"rtc", "delay_horizon"}};
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
            smolvla::SmolvlaObservation observation;
            for (auto const& frame : frames)
            {
                observation.views.push_back(smolvla::SmolvlaView{frame.name, frame.image.data(),
                    static_cast<int32_t>(frame.image.height), static_cast<int32_t>(frame.image.width)});
            }
            observation.state = in.at("state").get<std::vector<float>>();
            observation.task = in.at("task").get<std::string>();
            smolvla::SmolvlaRtc rtc;
            bool const useRtc = in.contains("rtc");
            if (useRtc)
            {
                auto const& r = in.at("rtc");
                rtc.inferenceDelay = r.at("delay").get<int32_t>();
                rtc.executionHorizon = r.at("horizon").get<int32_t>();
                rtc.startRow = r.value("start_row", -1);
            }
            smolvla::SmolvlaChunk const chunk = policy.act(observation, {}, useRtc ? &rtc : nullptr);

            Json rows = Json::array();
            for (int32_t t = 0; t < policy.chunkSize(); ++t)
            {
                rows.push_back(std::vector<float>(chunk.robot.begin() + static_cast<int64_t>(t) * policy.actionDim(),
                    chunk.robot.begin() + static_cast<int64_t>(t + 1) * policy.actionDim()));
            }
            Json reply;
            reply["actions"] = rows;
            reply["timing_ms"] = {{"vision", chunk.visualMs}, {"llm", chunk.prefixMs}, {"action", chunk.denoiseMs},
                {"total", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()}};
            return reply;
        },
        [&] { policy.resetEpisode(); });
    cudaStreamDestroy(stream);
    return 0;
}
