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

#include "vlaDualRate.h"

#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace internvla_n1
{

//! System 2's output for System 1 (the reference agent runs it on a background thread behind a `should_infer`
//! flag). System 1 owns its context pool and stream so the two overlap; see InternVLAN1System1Runner.
struct InternVLAN1Plan
{
    std::vector<float> conditioning; //!< [2, condLen, latentDim], null row first.
    int64_t condLen{0};
    int64_t latentDim{0};
    int64_t observationIndex{-1};
};

using InternVLAN1DualSystemState = vla::DualRateState<InternVLAN1Plan>;
using InternVLAN1DualSystemDriver = vla::DualRateDriver<InternVLAN1Plan>;

} // namespace internvla_n1
} // namespace trt_edgellm
