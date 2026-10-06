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

#include "gr00tProcessing.h"

#include "common/checkMacros.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>

namespace trt_edgellm
{
namespace gr00t
{
namespace
{

using Json = nlohmann::json;

//! Flattens a bounds entry that is either [dim] or [steps][dim]; returns the step count.
int32_t readBounds(Json const& value, std::vector<double>& out)
{
    if (!value.empty() && value.front().is_array())
    {
        for (auto const& row : value)
        {
            for (auto const& v : row)
            {
                out.push_back(v.get<double>());
            }
        }
        return static_cast<int32_t>(value.size());
    }
    for (auto const& v : value)
    {
        out.push_back(v.get<double>());
    }
    return 1;
}

//! GR00T's np.isclose(max, min) with default tolerances.
bool degenerate(double lo, double hi)
{
    return std::fabs(hi - lo) <= 1e-8 + 1e-5 * std::fabs(lo);
}

} // namespace

Gr00tProcessing::Gr00tProcessing(std::string const& path)
{
    std::ifstream file(path);
    ELLM_CHECK(file.good(), "Gr00tProcessing: missing " + path);
    Json const json = Json::parse(file);
    mMaxStateDim = json.at("max_state_dim").get<int32_t>();
    mMaxActionDim = json.at("max_action_dim").get<int32_t>();
    mActionHorizon = json.at("action_horizon").get<int32_t>();
    mClipState = json.value("clip_state", true);

    std::map<std::string, int32_t> stateOffsets;
    for (auto const& g : json.at("state"))
    {
        Group group;
        group.name = g.at("name").get<std::string>();
        group.dim = g.at("dim").get<int32_t>();
        group.offset = mRawStateDim;
        readBounds(g.at("min"), group.min);
        group.steps = readBounds(g.at("max"), group.max);
        ELLM_CHECK(group.steps == 1, "Gr00tProcessing: per-step state bounds are not supported");
        stateOffsets[group.name] = group.offset;
        mRawStateDim += group.dim;
        mState.push_back(std::move(group));
    }
    for (auto const& g : json.at("action"))
    {
        Group group;
        group.name = g.at("name").get<std::string>();
        group.dim = g.at("dim").get<int32_t>();
        group.offset = mRawActionDim;
        readBounds(g.at("min"), group.min);
        group.steps = readBounds(g.at("max"), group.max);
        ELLM_CHECK(group.steps == 1 || group.steps == mActionHorizon,
            "Gr00tProcessing: per-step action bounds must cover the action horizon");
        group.relative = g.value("relative", false);
        if (group.relative)
        {
            auto const ref = stateOffsets.find(g.at("reference_state").get<std::string>());
            ELLM_CHECK(ref != stateOffsets.end(), "Gr00tProcessing: unknown reference state for " + group.name);
            group.referenceOffset = ref->second;
        }
        mRawActionDim += group.dim;
        mAction.push_back(std::move(group));
    }
    ELLM_CHECK(mRawStateDim <= mMaxStateDim && mRawActionDim <= mMaxActionDim,
        "Gr00tProcessing: groups exceed the model's padded dimensions");
}

std::vector<float> Gr00tProcessing::normalizeState(std::vector<float> const& rawState) const
{
    ELLM_CHECK(static_cast<int32_t>(rawState.size()) == mRawStateDim, "Gr00tProcessing: raw state has the wrong width");
    std::vector<float> out(static_cast<size_t>(mMaxStateDim), 0.0F);
    for (auto const& g : mState)
    {
        for (int32_t d = 0; d < g.dim; ++d)
        {
            double const lo = g.min[d];
            double const hi = g.max[d];
            double v = 0.0;
            if (!degenerate(lo, hi))
            {
                v = 2.0 * (rawState[g.offset + d] - lo) / (hi - lo) - 1.0;
                if (mClipState)
                {
                    v = std::clamp(v, -1.0, 1.0);
                }
            }
            out[g.offset + d] = static_cast<float>(v);
        }
    }
    return out;
}

std::vector<float> Gr00tProcessing::decodeActions(float const* modelActions, std::vector<float> const& rawState) const
{
    ELLM_CHECK(static_cast<int32_t>(rawState.size()) == mRawStateDim, "Gr00tProcessing: raw state has the wrong width");
    std::vector<float> out(static_cast<size_t>(mActionHorizon) * mRawActionDim);
    for (int32_t t = 0; t < mActionHorizon; ++t)
    {
        for (auto const& g : mAction)
        {
            size_t const row = g.steps == 1 ? 0 : static_cast<size_t>(t) * g.dim;
            for (int32_t d = 0; d < g.dim; ++d)
            {
                double const lo = g.min[row + d];
                double const hi = g.max[row + d];
                double const a = std::clamp(
                    static_cast<double>(modelActions[static_cast<size_t>(t) * mMaxActionDim + g.offset + d]), -1.0,
                    1.0);
                double v = (a + 1.0) / 2.0 * (hi - lo) + lo;
                if (g.relative)
                {
                    v += rawState[g.referenceOffset + d];
                }
                out[static_cast<size_t>(t) * mRawActionDim + g.offset + d] = static_cast<float>(v);
            }
        }
    }
    return out;
}

std::vector<float> Gr00tProcessing::encodeActions(
    float const* absoluteActions, int32_t numRows, std::vector<float> const& rawState) const
{
    ELLM_CHECK(static_cast<int32_t>(rawState.size()) == mRawStateDim, "Gr00tProcessing: raw state has the wrong width");
    ELLM_CHECK(numRows >= 0 && numRows <= mActionHorizon, "Gr00tProcessing: more rows than the action horizon");
    std::vector<float> out(static_cast<size_t>(numRows) * mMaxActionDim, 0.0F);
    for (int32_t t = 0; t < numRows; ++t)
    {
        for (auto const& g : mAction)
        {
            size_t const row = g.steps == 1 ? 0 : static_cast<size_t>(t) * g.dim;
            for (int32_t d = 0; d < g.dim; ++d)
            {
                double const lo = g.min[row + d];
                double const hi = g.max[row + d];
                double v = absoluteActions[static_cast<size_t>(t) * mRawActionDim + g.offset + d];
                if (g.relative)
                {
                    v -= rawState[g.referenceOffset + d];
                }
                double const a = degenerate(lo, hi) ? 0.0 : std::clamp(2.0 * (v - lo) / (hi - lo) - 1.0, -1.0, 1.0);
                out[static_cast<size_t>(t) * mMaxActionDim + g.offset + d] = static_cast<float>(a);
            }
        }
    }
    return out;
}

} // namespace gr00t
} // namespace trt_edgellm
