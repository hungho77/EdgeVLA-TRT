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

#pragma once

#include "common/tensor.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace vla
{

std::unique_ptr<nvinfer1::IRuntime> createTrtRuntime();

//! A TensorRT engine with one USER_MANAGED execution context on optimization profile 0. The owner supplies the
//! context's scratch memory (see allocateSharedContextMemory) before the first enqueue.
class TrtEngine
{
public:
    TrtEngine() = default;
    TrtEngine(nvinfer1::IRuntime& runtime, std::string const& path, cudaStream_t stream);

    int64_t deviceMemorySize() const;
    void setDeviceMemory(rt::Tensor& memory);

    void bind(char const* name, void const* address);
    void setShape(char const* name, std::vector<int64_t> const& shape);
    bool enqueue(cudaStream_t stream);

    nvinfer1::ICudaEngine& engine() const noexcept
    {
        return *mEngine;
    }
    nvinfer1::IExecutionContext& context() const noexcept
    {
        return *mContext;
    }

private:
    std::string mPath;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
};

//! One scratch allocation for engines that only ever run one after another on one stream, sized to the largest.
//! Engines that may overlap (e.g. a control loop next to a planner) need separate allocations.
rt::Tensor allocateSharedContextMemory(std::vector<TrtEngine*> const& engines, std::string const& name);

} // namespace vla
} // namespace trt_edgellm
