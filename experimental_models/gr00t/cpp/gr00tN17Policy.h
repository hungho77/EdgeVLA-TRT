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

#include "gr00tN17ActionRunner.h"
#include "gr00tProcessing.h"

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace gr00t
{

//! GR00T N1.7 action head plus one embodiment's processing: backbone features and raw robot state in,
//! absolute raw actions out.
class Gr00tN17Policy
{
public:
    //! Real-time chunking. The new chunk's first rows continue the previous chunk from row startRow, i.e. the
    //! robot has executed startRow actions of the previous chunk when the new one starts.
    struct Rtc
    {
        int32_t overlapSteps{}; //!< rows seeded from the previous chunk (fewer if it ends sooner)
        int32_t frozenSteps{};  //!< rows reproduced exactly, covering the policy latency
        float rampRate{6.0F};
        int32_t startRow{-1}; //!< -1: horizon - overlapSteps, the official convention
    };

    //! \p engineDir holds the action engines, config.json and processing.json.
    Gr00tN17Policy(std::string const& engineDir, cudaStream_t stream);

    Gr00tN17ActionRunner& runner() noexcept
    {
        return mRunner;
    }
    Gr00tProcessing const& processing() const noexcept
    {
        return mProcessing;
    }

    //! Forget the previous chunk, e.g. at the start of an episode.
    void resetEpisode() noexcept
    {
        mPrevious.clear();
    }

    //! \p backboneFeatures and \p imageMask as for Gr00tN17ActionRunner::prepare; \p noise is
    //! [maxHorizon, maxActionDim] FP32 on the GPU. Returns absolute raw actions [actionHorizon, rawActionDim].
    //! With \p rtc and a previous chunk in this episode, the chunk is inpainted from it.
    std::vector<float> act(rt::Tensor const& backboneFeatures, std::vector<uint8_t> const& imageMask,
        std::vector<float> const& rawState, rt::Tensor const& noise, cudaStream_t stream, Rtc const* rtc = nullptr);

    //! Normalized model output of the last act(), [maxHorizon, maxActionDim].
    float const* lastModelActions() const noexcept
    {
        return mModelActionsHost.dataPointer<float>();
    }

private:
    Gr00tN17ActionRunner mRunner;
    Gr00tProcessing mProcessing;
    rt::Tensor mModelActionsHost; //!< pinned
    std::vector<float> mPrevious; //!< absolute raw actions of the last chunk, empty when none
};

} // namespace gr00t
} // namespace trt_edgellm
