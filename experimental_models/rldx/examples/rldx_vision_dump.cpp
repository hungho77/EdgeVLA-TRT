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

//! Run the RLDX visual engine (Edge-LLM's Qwen3-VL ViT runner) on images given in the processor's order (frame-major,
//! view-minor) and write the merged visual embeddings and the deepstack features as raw FP32, for checking against
//! the official policy:  rldx_vision_dump ENGINE_DIR OUT_PREFIX img0.png ... img7.png

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "multimodal/common/multimodalRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "vlaBackbone.h"

#include <cstdio>
#include <cuda_fp16.h>
#include <fstream>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

void writeFp32(rt::Tensor const& tensor, std::string const& path, cudaStream_t stream)
{
    int64_t const count = tensor.getShape().volume();
    std::vector<__half> host(static_cast<size_t>(count));
    CUDA_CHECK(
        cudaMemcpyAsync(host.data(), tensor.rawPointer(), count * sizeof(__half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> values(host.size());
    for (size_t i = 0; i < host.size(); ++i)
    {
        values[i] = __half2float(host[i]);
    }
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<char const*>(values.data()), values.size() * sizeof(float));
    std::printf("%s: %s\n", path.c_str(), tensor.getShape().formatString().c_str());
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: %s ENGINE_DIR OUT_PREFIX IMAGE...\n", argv[0]);
        return 2;
    }
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto const pluginHandles = loadEdgellmPluginLib();
    auto runner = rt::MultimodalRunner::create(std::string(argv[1]) + "/visual", 1, 4096, stream);
    rt::Tensor contextMemory(
        {runner->getRequiredContextMemorySize()}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "rldx::ctx");
    ELLM_CHECK(runner->setContextMemory(contextMemory), "setContextMemory failed");

    rt::LLMGenerationRequest request;
    request.requests.resize(1);
    request.requests[0].imageBuffers = vla::loadImages(std::vector<std::string>(argv + 3, argv + argc));
    std::vector<std::vector<int32_t>> ids;
    ELLM_CHECK(
        runner->preprocess(request, ids, nullptr, std::nullopt, stream, /*imageOnly=*/true), "preprocess failed");
    ELLM_CHECK(runner->infer(stream), "infer failed");
    std::string const prefix = argv[2];
    writeFp32(runner->getOutputEmbedding(), prefix + "_visual.bin", stream);
    auto const deepstack = runner->getDeepstackFeatures();
    for (size_t i = 0; i < deepstack.size(); ++i)
    {
        writeFp32(deepstack[i].get(), prefix + "_deepstack" + std::to_string(i) + ".bin", stream);
    }
    cudaStreamDestroy(stream);
    return 0;
}
