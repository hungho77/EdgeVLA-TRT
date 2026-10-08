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

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace gr00t
{

//! One embodiment's state/action processing, from processing.json (export_gr00t_processing.py).
//! Raw state and actions are the groups concatenated in their modality order.
class Gr00tProcessing
{
public:
    explicit Gr00tProcessing(std::string const& path);

    int32_t rawStateDim() const noexcept
    {
        return mRawStateDim;
    }
    int32_t rawActionDim() const noexcept
    {
        return mRawActionDim;
    }
    int32_t actionHorizon() const noexcept
    {
        return mActionHorizon;
    }
    int32_t maxStateDim() const noexcept
    {
        return mMaxStateDim;
    }
    int32_t maxActionDim() const noexcept
    {
        return mMaxActionDim;
    }

    //! Raw state (rawStateDim values) -> model state (maxStateDim values, zero-padded).
    std::vector<float> normalizeState(std::vector<float> const& rawState) const;

    //! Model output ([maxHorizon, maxActionDim], row-major) -> absolute raw actions [actionHorizon, rawActionDim].
    std::vector<float> decodeActions(float const* modelActions, std::vector<float> const& rawState) const;

    //! Inverse of decodeActions for chunk rows [0, numRows): absolute raw actions [numRows, rawActionDim] ->
    //! model actions [numRows, maxActionDim], relative to \p rawState, clipped to [-1, 1], zero-padded.
    std::vector<float> encodeActions(
        float const* absoluteActions, int32_t numRows, std::vector<float> const& rawState) const;

private:
    struct Group
    {
        std::string name;
        int32_t dim{};
        int32_t offset{};        //!< offset in the concatenated raw vector
        std::vector<double> min; //!< dim values, or steps * dim for per-step bounds
        std::vector<double> max;
        int32_t steps{1}; //!< 1 for shared bounds, otherwise one row per action step
        bool relative{false};
        int32_t referenceOffset{-1}; //!< raw-state offset of the reference group for relative actions
    };

    std::vector<Group> mState;
    std::vector<Group> mAction;
    int32_t mRawStateDim{0};
    int32_t mRawActionDim{0};
    int32_t mActionHorizon{0};
    int32_t mMaxStateDim{0};
    int32_t mMaxActionDim{0};
    bool mClipState{true};
};

} // namespace gr00t
} // namespace trt_edgellm
