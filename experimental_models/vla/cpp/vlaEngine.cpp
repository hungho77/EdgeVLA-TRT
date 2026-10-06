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

#include "vlaEngine.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"

#include <algorithm>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace vla
{

std::unique_ptr<IRuntime> createTrtRuntime()
{
    std::unique_ptr<IRuntime> runtime(createInferRuntime(gLogger));
    ELLM_CHECK(runtime, "vla: failed to create the TensorRT runtime");
    return runtime;
}

TrtEngine::TrtEngine(IRuntime& runtime, std::string const& path, cudaStream_t stream)
    : mPath(path)
{
    mEngine = deserializeCudaEngineFromFile(runtime, path);
    ELLM_CHECK(mEngine, "vla: failed to load " + path);
    mContext = std::unique_ptr<IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mContext, "vla: failed to create a context for " + path);
    ELLM_CHECK(
        mContext->setOptimizationProfileAsync(0, stream), "vla: failed to set the optimization profile for " + path);
}

int64_t TrtEngine::deviceMemorySize() const
{
    return mEngine->getDeviceMemorySizeV2();
}

void TrtEngine::setDeviceMemory(rt::Tensor& memory)
{
    mContext->setDeviceMemoryV2(memory.rawPointer(), memory.getShape()[0]);
}

void TrtEngine::bind(char const* name, void const* address)
{
    ELLM_CHECK(mContext->setTensorAddress(name, const_cast<void*>(address)),
        "vla: failed to bind " + std::string(name) + " of " + mPath);
}

void TrtEngine::setShape(char const* name, std::vector<int64_t> const& shape)
{
    Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    std::copy(shape.begin(), shape.end(), dims.d);
    ELLM_CHECK(
        mContext->setInputShape(name, dims), "vla: failed to set the shape of " + std::string(name) + " of " + mPath);
}

bool TrtEngine::enqueue(cudaStream_t stream)
{
    return mContext->enqueueV3(stream);
}

rt::Tensor allocateSharedContextMemory(std::vector<TrtEngine*> const& engines, std::string const& name)
{
    int64_t bytes = 0;
    for (TrtEngine const* engine : engines)
    {
        bytes = std::max(bytes, engine->deviceMemorySize());
    }
    rt::Tensor memory(rt::Coords(std::vector<int64_t>{bytes}), rt::DeviceType::kGPU, DataType::kUINT8, name);
    for (TrtEngine* engine : engines)
    {
        engine->setDeviceMemory(memory);
    }
    return memory;
}

} // namespace vla
} // namespace trt_edgellm
