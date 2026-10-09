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

#pragma once

#include "runtime/imageUtils.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace vla
{

//! One camera frame of a request.
struct NamedFrame
{
    std::string name;
    rt::imageUtils::ImageData image; //!< host [1, H, W, 3] uint8 RGB
};

//! One request: its JSON header and the frames that came inline behind it.
struct ServerRequest
{
    nlohmann::json header;
    std::vector<NamedFrame> frames; //!< in the order of the header's "frames"
};

//! The transport every VLA policy server shares, stdin / stdout or one TCP client at a time.
//!
//! A request is one JSON line, optionally followed by raw camera frames:
//!   {"frames": [{"name": "top", "height": 480, "width": 640, "bytes": 921600}, ...], ...}
//!   + the frames' bytes in that order, each tightly packed row-major [height, width, 3] uint8 **RGB** (not
//!     OpenCV's default BGR; nothing can detect a swapped order).
//! Requests that name image files instead keep working; collectFrames() resolves both. The reply is one JSON line.
//! A blank line or EOF ends a stdin session; in TCP mode the server then waits for the next client.
//!
//! Options: --port N serves TCP instead of stdin (127.0.0.1 unless --host is given, e.g. 0.0.0.0 for the LAN).
//! In TCP mode only protocol bytes go on the socket; runtime logs stay on the process stdout.
class PolicyServer
{
public:
    using Handler = std::function<nlohmann::json(ServerRequest&)>;

    PolicyServer(int argc, char** argv);
    ~PolicyServer() noexcept;
    PolicyServer(PolicyServer const&) = delete;
    PolicyServer& operator=(PolicyServer const&) = delete;

    //! Serve until stdin closes (stdio) or forever (TCP). \p ready is sent when a session starts, after
    //! \p onSessionStart (e.g. a policy's episode reset). A handler exception becomes an {"error"} reply.
    void run(nlohmann::json const& ready, Handler const& handle, std::function<void()> const& onSessionStart = {});

    bool tcp() const noexcept
    {
        return mPort > 0;
    }

private:
    class Stream;
    void serveSession(Stream& stream, nlohmann::json const& ready, Handler const& handle);

    int32_t mPort{0};
    std::string mHost{"127.0.0.1"};
};

//! The request's frames by name: the inline frames, or else the image files a path field names. \p pathsKey is
//! a JSON object {name: path} (cameras) or array [path, ...] (named by position, "0", "1", ...).
std::vector<NamedFrame> collectFrames(ServerRequest& request, char const* pathsKey);

//! The standard argument lookup the servers use: the value after \p flag, or \p fallback.
std::string argOf(int argc, char** argv, char const* flag, std::string const& fallback = "");

} // namespace vla
} // namespace trt_edgellm
