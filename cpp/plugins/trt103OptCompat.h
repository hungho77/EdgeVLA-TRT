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

#include <NvInferRuntime.h>

#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! JetPack 6 test harness: TensorRT 10.3 leaves DynamicPluginTensorDesc::opt empty in configurePlugin.
//! Substitute max so profile validation sees a concrete shape.
inline std::vector<nvinfer1::DynamicPluginTensorDesc> fillMissingOptProfile(
    nvinfer1::DynamicPluginTensorDesc const* desc, int32_t count)
{
    std::vector<nvinfer1::DynamicPluginTensorDesc> result;
    if (desc == nullptr || count <= 0)
    {
        return result;
    }
    result.assign(desc, desc + count);
    for (auto& d : result)
    {
        if (d.opt.nbDims == 0 && d.max.nbDims > 0)
        {
            d.opt = d.max;
        }
    }
    return result;
}

} // namespace plugins
} // namespace trt_edgellm
