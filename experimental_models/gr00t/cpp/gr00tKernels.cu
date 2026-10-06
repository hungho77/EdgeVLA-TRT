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

#include "gr00tKernels.h"

#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace gr00t
{
namespace
{

__global__ void halfToFloatKernel(__half const* in, float* out, int64_t count)
{
    int64_t const i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
    {
        out[i] = __half2float(in[i]);
    }
}

} // namespace

void launchHalfToFloat(void const* in, float* out, int64_t count, cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    int64_t const blocks = (count + kThreads - 1) / kThreads;
    halfToFloatKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        static_cast<__half const*>(in), out, count);
}

} // namespace gr00t
} // namespace trt_edgellm
