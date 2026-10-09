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

//! OpenVLA policy server for a robot or a simulator: one request in, one action out, over stdin / stdout or TCP
//! (--port, see vlaServer.h for the framing and the inline-frame protocol).
//!
//! Request:  {"frames": [{"name": "image", "height": H, "width": W, "bytes": N}] + raw RGB bytes,
//!              or "image": "frame.png",               the third-person camera frame
//!            "task": "pick up the banana",            ("instruction" is accepted too)
//!            "unnorm_key": "bridge_orig"}             optional, default --unnormKey
//! Reply:    {"actions": [[...]], "action_tokens": [...], "timing_ms": {...}}  or {"error": "..."}
//! The action is the dataset's unnormalized end-effector delta and gripper (OpenVLA's predict_action), one per call.

#include "openvlaPolicy.h"

#include "common/trtUtils.h"
#include "vlaServer.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

int main(int argc, char** argv)
{
    std::string const visionDir = vla::argOf(argc, argv, "--visionDir");
    std::string const llmDir = vla::argOf(argc, argv, "--llmEngineDir");
    if (visionDir.empty() || llmDir.empty())
    {
        std::fprintf(
            stderr, "usage: %s --visionDir DIR --llmEngineDir DIR [--unnormKey KEY] [--port N [--host H]]\n", argv[0]);
        return 2;
    }

    auto const pluginHandle = loadEdgellmPluginLib();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    openvla::OpenvlaPolicy policy(visionDir, llmDir, stream);
    std::vector<std::string> const keys = policy.unnormKeys();
    std::string const defaultKey = vla::argOf(argc, argv, "--unnormKey", keys.size() == 1 ? keys.front() : "");

    Json ready = {{"ready", true}, {"family", "openvla"}, {"cameras", {"image"}}, {"state_dim", 0}, {"chunk", 1},
        {"rtc", nullptr}, {"unnorm_keys", keys}, {"unnorm_key", defaultKey}};
    if (!defaultKey.empty())
    {
        ready["action_dim"] = policy.actionDim(defaultKey);
    }
    vla::PolicyServer server(argc, argv);
    server.run(ready, [&](vla::ServerRequest& request) {
        Json const& in = request.header;
        rt::imageUtils::ImageData const image = request.frames.empty()
            ? rt::imageUtils::loadRgbImageFromFile(in.at("image").get<std::string>())
            : std::move(request.frames.front().image);
        std::string const task
            = in.contains("task") ? in.at("task").get<std::string>() : in.at("instruction").get<std::string>();
        std::string const key = in.value("unnorm_key", defaultKey);
        openvla::OpenvlaStep const step = policy.act(
            image.data(), static_cast<int32_t>(image.height), static_cast<int32_t>(image.width), task, key);
        Json reply;
        reply["actions"] = Json::array({step.actions});
        reply["action_tokens"] = step.actionIds;
        reply["timing_ms"] = {{"vision", step.visionMs}, {"llm", step.llmMs}};
        return reply;
    });
    cudaStreamDestroy(stream);
    return 0;
}
