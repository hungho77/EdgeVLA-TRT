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

//! pi0.5 policy server for a robot or a simulator: one request in, one action chunk out, over stdin / stdout or
//! TCP (--port, see vlaServer.h for the framing and the inline-frame protocol).
//!
//! Request:  {"frames": [{"name": "observation/image", "height": H, "width": W, "bytes": N}, ...] + raw RGB,
//!              or "cameras": {"observation/image": "top.png", ...},  frames keyed by the contract's camera slots
//!            "state": [...],                           robot units, the embodiment's width
//!            "task": "pick the cube",
//!            "rtc": {"overlap": 5, "frozen": 2,        optional, real-time chunking from the previous reply
//!                    "ramp_rate": 6.0, "start_row": 5},
//!            "seed": 0,                                optional, reseeds x_0
//!            "reset": true}                            optional, start of an episode
//! Reply:    {"actions": [[...], ...], "timing_ms": {...}}  or {"error": "..."}
//! Actions are robot units for the contract's embodiment (its openpi output transform applied).

#include "common/trtUtils.h"
#include "runtime/pi05Policy.h"
#include "vlaServer.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

int main(int argc, char** argv)
{
    std::string const engineDir = vla::argOf(argc, argv, "--engineDir");
    if (engineDir.empty())
    {
        std::fprintf(stderr, "usage: %s --engineDir DIR [--steps N] [--seed N] [--cudaGraph 1] [--port N [--host H]]\n",
            argv[0]);
        return 2;
    }

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // Only the action engine carries a plugin node.
    auto const pluginHandles = loadEdgellmPluginLib();
    pi05::Pi05Policy policy(engineDir, stream);
    pi05::Pi05Runtime& runtime = policy.runtime();
    runtime.setNoiseSeed(std::stoull(vla::argOf(argc, argv, "--seed", "0")));
    runtime.setUseCudaGraph(vla::argOf(argc, argv, "--cudaGraph", "1") != "0");
    std::string const steps = vla::argOf(argc, argv, "--steps");
    if (!steps.empty())
    {
        runtime.setNumDenoiseSteps(std::stoi(steps));
    }

    std::vector<std::string> cameras;
    for (auto const& slot : policy.cameras())
    {
        cameras.push_back(slot.name);
    }
    pi05::Pi05Contract const& contract = policy.contract();
    Json const ready = {{"ready", true}, {"family", "pi05"}, {"policy_config", contract.policyConfig},
        {"cameras", cameras}, {"state_dim", contract.stateDim}, {"action_dim", policy.robotActionDim()},
        {"chunk", contract.actionHorizon}, {"rtc", "overlap_frozen"}};
    vla::PolicyServer server(argc, argv);
    server.run(
        ready,
        [&](vla::ServerRequest& request) {
            Json const& in = request.header;
            if (in.value("reset", false))
            {
                policy.resetEpisode();
            }
            if (in.contains("seed"))
            {
                runtime.setNoiseSeed(in.at("seed").get<uint64_t>());
            }
            std::vector<vla::NamedFrame> const frames = vla::collectFrames(request, "cameras");
            pi05::Pi05Observation observation;
            for (auto const& frame : frames)
            {
                pi05::Pi05CameraView view;
                view.name = frame.name;
                view.rgb = frame.image.data();
                view.height = static_cast<int32_t>(frame.image.height);
                view.width = static_cast<int32_t>(frame.image.width);
                observation.cameras.push_back(view);
            }
            observation.state = in.at("state").get<std::vector<float>>();
            observation.task = in.at("task").get<std::string>();

            pi05::Pi05Rtc rtc;
            bool const useRtc = in.contains("rtc");
            if (useRtc)
            {
                auto const& r = in.at("rtc");
                rtc.overlapSteps = r.at("overlap").get<int32_t>();
                rtc.frozenSteps = r.value("frozen", 0);
                rtc.rampRate = r.value("ramp_rate", 6.0F);
                rtc.startRow = r.value("start_row", -1);
            }
            pi05::Pi05ActionChunk const chunk = policy.infer(observation, useRtc ? &rtc : nullptr);

            int32_t const dim = policy.robotActionDim();
            Json rows = Json::array();
            for (int32_t t = 0; t < chunk.horizon; ++t)
            {
                rows.push_back(std::vector<float>(chunk.robotActions.begin() + static_cast<int64_t>(t) * dim,
                    chunk.robotActions.begin() + static_cast<int64_t>(t + 1) * dim));
            }
            Json reply;
            reply["actions"] = rows;
            auto const& stages = chunk.timings.stages;
            reply["timing_ms"] = {{"host", chunk.timings.observation.totalMs}, {"vision", stages.visualMs},
                {"llm", stages.assembleMs + stages.prefixMs}, {"action", stages.actionMs},
                {"engines", chunk.timings.engineMs}, {"total", chunk.timings.policyMs}};
            return reply;
        },
        [&] { policy.resetEpisode(); });
    cudaStreamDestroy(stream);
    return 0;
}
