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

#include "gr00tN17Policy.h"

#include "common/checkMacros.h"

namespace trt_edgellm
{
namespace gr00t
{

Gr00tN17Policy::Gr00tN17Policy(std::string const& engineDir, cudaStream_t stream)
    : mRunner(engineDir, stream)
    , mProcessing(engineDir + "/processing.json")
    , mModelActionsHost(rt::Coords(std::vector<int64_t>{mRunner.config().actionHorizon, mRunner.config().actionDim}),
          rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "gr00t::modelActionsHost")
{
    ELLM_CHECK(mProcessing.maxStateDim() == mRunner.config().stateDim
            && mProcessing.maxActionDim() == mRunner.config().actionDim
            && mProcessing.actionHorizon() <= mRunner.config().actionHorizon,
        "Gr00tN17Policy: processing.json does not match the action engines");
}

std::vector<float> Gr00tN17Policy::act(rt::Tensor const& backboneFeatures, std::vector<uint8_t> const& imageMask,
    std::vector<float> const& rawState, rt::Tensor const& noise, cudaStream_t stream, Rtc const* rtc)
{
    mRunner.prepare(backboneFeatures, imageMask, stream);
    mRunner.encodeState(mProcessing.normalizeState(rawState), stream);

    // The previous chunk is kept in absolute joint space: relative groups are re-expressed against the current
    // state and each row gets the bounds of its new index, so the frozen rows reproduce the committed actions
    // even though the arm moved between calls.
    std::vector<float> seed;
    Gr00tN17ActionRunner::RtcOptions options;
    int32_t const horizon = mProcessing.actionHorizon();
    if (rtc != nullptr && !mPrevious.empty() && rtc->overlapSteps > 0)
    {
        ELLM_CHECK(rtc->overlapSteps <= horizon, "Gr00tN17Policy: RTC overlap exceeds the action horizon");
        int64_t const start = static_cast<int64_t>(horizon - rtc->overlapSteps) * mProcessing.rawActionDim();
        seed = mProcessing.encodeActions(mPrevious.data() + start, rtc->overlapSteps, rawState);
        options.overlapSteps = rtc->overlapSteps;
        options.frozenSteps = rtc->frozenSteps;
        options.rampRate = rtc->rampRate;
        options.seed = seed.data();
    }
    rt::Tensor const& actions = mRunner.sample(noise, stream, options.seed != nullptr ? &options : nullptr);
    CUDA_CHECK(cudaMemcpyAsync(mModelActionsHost.rawPointer(), actions.rawPointer(),
        static_cast<size_t>(mRunner.config().actionHorizon) * mRunner.config().actionDim * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    mPrevious = mProcessing.decodeActions(mModelActionsHost.dataPointer<float>(), rawState);
    return mPrevious;
}

} // namespace gr00t
} // namespace trt_edgellm
