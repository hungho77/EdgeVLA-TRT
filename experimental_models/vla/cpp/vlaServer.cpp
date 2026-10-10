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

#include "vlaServer.h"

#include "common/checkMacros.h"
#include "common/tensor.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trt_edgellm
{
namespace vla
{

using Json = nlohmann::json;

//! Line and exact-length reads, and whole-buffer writes, over stdin / stdout or a socket.
class PolicyServer::Stream
{
public:
    explicit Stream(int fd = -1)
        : mFd(fd)
    {
    }

    //! False at EOF.
    bool readLine(std::string& line)
    {
        line.clear();
        if (mFd < 0)
        {
            return static_cast<bool>(std::getline(std::cin, line));
        }
        while (true)
        {
            size_t const newline = mBuffer.find('\n', mStart);
            if (newline != std::string::npos)
            {
                line.assign(mBuffer, mStart, newline - mStart);
                mStart = newline + 1;
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }
                return true;
            }
            if (!fill())
            {
                return false;
            }
        }
    }

    void readExact(void* dst, size_t size)
    {
        auto* out = static_cast<char*>(dst);
        if (mFd < 0)
        {
            ELLM_CHECK(static_cast<bool>(std::cin.read(out, static_cast<std::streamsize>(size))),
                "vla server: stdin ended inside a frame");
            return;
        }
        while (size > 0)
        {
            if (mStart == mBuffer.size() && !fill())
            {
                throw std::runtime_error("vla server: client closed inside a frame");
            }
            size_t const n = std::min(size, mBuffer.size() - mStart);
            std::memcpy(out, mBuffer.data() + mStart, n);
            mStart += n;
            out += n;
            size -= n;
        }
    }

    void writeLine(std::string const& text)
    {
        if (mFd < 0)
        {
            std::printf("%s\n", text.c_str());
            std::fflush(stdout);
            return;
        }
        std::string const line = text + "\n";
        size_t sent = 0;
        while (sent < line.size())
        {
            ssize_t const n = ::send(mFd, line.data() + sent, line.size() - sent, MSG_NOSIGNAL);
            if (n <= 0)
            {
                throw std::runtime_error("vla server: client went away");
            }
            sent += static_cast<size_t>(n);
        }
    }

private:
    bool fill()
    {
        if (mStart > 0)
        {
            mBuffer.erase(0, mStart);
            mStart = 0;
        }
        char chunk[1 << 16];
        ssize_t const n = ::recv(mFd, chunk, sizeof(chunk), 0);
        if (n <= 0)
        {
            return false;
        }
        mBuffer.append(chunk, static_cast<size_t>(n));
        return true;
    }

    int mFd;
    std::string mBuffer;
    size_t mStart{0};
};

std::string argOf(int argc, char** argv, char const* flag, std::string const& fallback)
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

PolicyServer::PolicyServer(int argc, char** argv)
    : mPort(std::stoi(argOf(argc, argv, "--port", "0")))
    , mHost(argOf(argc, argv, "--host", "127.0.0.1"))
    , mTimingEvery(std::stoi(argOf(argc, argv, "--timingEvery", "20")))
{
}

PolicyServer::~PolicyServer() noexcept = default;

void PolicyServer::serveSession(Stream& stream, Json const& ready, Handler const& handle)
{
    stream.writeLine(ready.dump());
    std::string line;
    while (stream.readLine(line))
    {
        if (line.empty())
        {
            return;
        }
        auto const start = std::chrono::steady_clock::now();
        auto const msSince = [](std::chrono::steady_clock::time_point t) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        };
        double receiveMs = 0.0;
        Json reply;
        try
        {
            ServerRequest request;
            request.header = Json::parse(line);
            // Frames are read before anything can fail on the header's content, so the stream stays in sync.
            for (auto const& frame : request.header.value("frames", Json::array()))
            {
                int64_t const height = frame.at("height").get<int64_t>();
                int64_t const width = frame.at("width").get<int64_t>();
                int64_t const bytes = frame.at("bytes").get<int64_t>();
                ELLM_CHECK(height > 0 && width > 0 && bytes == height * width * 3,
                    "vla server: a frame must carry height * width * 3 RGB bytes");
                rt::Tensor pixels(
                    {1, height, width, 3}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "vla::serverFrame");
                stream.readExact(pixels.dataPointer<unsigned char>(), static_cast<size_t>(bytes));
                request.frames.push_back(NamedFrame{frame.value("name", std::to_string(request.frames.size())),
                    rt::imageUtils::ImageData(std::move(pixels))});
            }
            receiveMs = msSince(start);
            reply = handle(request);
        }
        catch (std::exception const& e)
        {
            reply = {{"error", e.what()}};
        }
        if (reply.contains("timing_ms") && reply["timing_ms"].is_object())
        {
            reply["timing_ms"]["receive"] = receiveMs;
            reply["timing_ms"]["server"] = msSince(start);
            recordTiming(reply["timing_ms"]);
        }
        stream.writeLine(reply.dump());
    }
}

void PolicyServer::recordTiming(Json const& timing)
{
    if (mTimingEvery <= 0)
    {
        return;
    }
    for (auto const& [key, value] : timing.items())
    {
        if (value.is_number())
        {
            mTimings[key].push_back(value.get<double>());
        }
    }
    auto const count = mTimings.count("server") ? mTimings["server"].size() : 0;
    if (count < static_cast<size_t>(mTimingEvery))
    {
        return;
    }
    // The pipeline's stages first, then whatever else a family reports.
    std::vector<std::string> order{"server", "receive", "total", "host", "vision", "llm", "action"};
    for (auto const& [key, values] : mTimings)
    {
        if (std::find(order.begin(), order.end(), key) == order.end())
        {
            order.push_back(key);
        }
    }
    std::string line = "vla server: timing over " + std::to_string(count) + " requests, ms p50 (p95):";
    for (auto const& key : order)
    {
        auto it = mTimings.find(key);
        if (it == mTimings.end() || it->second.empty())
        {
            continue;
        }
        std::vector<double>& v = it->second;
        std::sort(v.begin(), v.end());
        auto const at = [&v](double q) { return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))]; };
        char buffer[96];
        std::snprintf(buffer, sizeof(buffer), " %s %.1f (%.1f)", key.c_str(), at(0.5), at(0.95));
        line += buffer;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
    mTimings.clear();
}

void PolicyServer::run(Json const& ready, Handler const& handle, std::function<void()> const& onSessionStart)
{
    if (!tcp())
    {
        if (onSessionStart)
        {
            onSessionStart();
        }
        Stream stdio;
        serveSession(stdio, ready, handle);
        return;
    }

    int const listener = ::socket(AF_INET, SOCK_STREAM, 0);
    ELLM_CHECK(listener >= 0, "vla server: socket() failed");
    int const one = 1;
    ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(mPort));
    ELLM_CHECK(::inet_pton(AF_INET, mHost.c_str(), &address.sin_addr) == 1, "vla server: bad --host " + mHost);
    ELLM_CHECK(::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
        "vla server: cannot bind " + mHost + ":" + std::to_string(mPort) + ": " + std::strerror(errno));
    ELLM_CHECK(::listen(listener, 1) == 0, "vla server: listen() failed");
    std::printf("vla server: listening on %s:%d\n", mHost.c_str(), mPort);
    std::fflush(stdout);
    while (true)
    {
        int const client = ::accept(listener, nullptr, nullptr);
        if (client < 0)
        {
            continue;
        }
        ::setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        if (onSessionStart)
        {
            onSessionStart();
        }
        try
        {
            Stream stream(client);
            serveSession(stream, ready, handle);
        }
        catch (std::exception const& e)
        {
            std::printf("vla server: session ended: %s\n", e.what());
            std::fflush(stdout);
        }
        ::close(client);
    }
}

std::vector<NamedFrame> collectFrames(ServerRequest& request, char const* pathsKey)
{
    if (!request.frames.empty())
    {
        return std::move(request.frames);
    }
    std::vector<NamedFrame> frames;
    Json const& paths = request.header.at(pathsKey);
    if (paths.is_object())
    {
        for (auto const& [name, path] : paths.items())
        {
            frames.push_back(NamedFrame{name, rt::imageUtils::loadRgbImageFromFile(path.get<std::string>())});
        }
    }
    else
    {
        for (auto const& path : paths)
        {
            frames.push_back(NamedFrame{
                std::to_string(frames.size()), rt::imageUtils::loadRgbImageFromFile(path.get<std::string>())});
        }
    }
    return frames;
}

} // namespace vla
} // namespace trt_edgellm
