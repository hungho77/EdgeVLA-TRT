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

//! TurboVLA policy server for a robot or a simulator: one request in, one action chunk out, over stdin / stdout or
//! TCP (--port, see vlaServer.h for the framing and the inline-frame protocol).
//!
//! Request:  {"frames": [{"name": "primary", "height": H, "width": W, "bytes": N}, ...] + raw RGB bytes,
//!              or "cameras": {"primary": "agentview.png", "wrist": "wrist.png"},
//!            "state": [...],                           the checkpoint's state layout
//!            "task": "pick the cube",
//!            "rtc": {"overlap": 6, "frozen": 3,        optional, blend with the previous reply (see TurbovlaBlend)
//!                    "ramp_rate": 6.0, "start_row": 6},
//!            "normalized": true,                       optional, also return the decoder's normalized chunk
//!            "reset": true}                            optional, start of an episode
//! Reply:    {"actions": [[...], ...], "timing_ms": {...}}  or {"error": "..."}
//! Actions are the environment's: arm deltas unnormalized with the suite statistics, the gripper +1 / -1.

#include "turbovlaPolicy.h"

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
        std::fprintf(stderr, "usage: %s --engineDir DIR [--port N [--host H]]\n", argv[0]);
        return 2;
    }

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    turbovla::TurbovlaPolicy policy(engineDir, stream);

    Json const ready = {{"ready", true}, {"family", "turbovla"}, {"cameras", policy.cameras()},
        {"state_dim", policy.stateDim()}, {"action_dim", policy.actionDim()}, {"chunk", policy.chunkSize()},
        {"rtc", "overlap_frozen"}, {"rtc_method", "blend"}};
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
            std::vector<vla::NamedFrame> const frames = vla::collectFrames(request, "cameras");
            std::vector<turbovla::TurbovlaView> views;
            for (auto const& frame : frames)
            {
                views.push_back(turbovla::TurbovlaView{frame.name, frame.image.data(),
                    static_cast<int32_t>(frame.image.height), static_cast<int32_t>(frame.image.width)});
            }
            turbovla::TurbovlaBlend blend;
            bool const useBlend = in.contains("rtc");
            if (useBlend)
            {
                auto const& r = in.at("rtc");
                blend.overlapSteps = r.at("overlap").get<int32_t>();
                blend.frozenSteps = r.value("frozen", 0);
                blend.rampRate = r.value("ramp_rate", 6.0F);
                blend.startRow = r.value("start_row", -1);
            }
            turbovla::TurbovlaChunk const chunk = policy.act(views, in.at("state").get<std::vector<float>>(),
                in.at("task").get<std::string>(), useBlend ? &blend : nullptr);

            Json reply;
            reply["actions"] = rowsOf(chunk.actions, policy.chunkSize(), policy.actionDim());
            if (in.value("normalized", false))
            {
                reply["normalized"] = rowsOf(chunk.normalized, policy.chunkSize(), policy.actionDim());
            }
            reply["timing_ms"] = {{"host", chunk.hostMs}, {"text", chunk.textMs}, {"policy", chunk.policyMs},
                {"total", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()}};
            return reply;
        },
        [&] { policy.resetEpisode(); });
    cudaStreamDestroy(stream);
    return 0;
}
