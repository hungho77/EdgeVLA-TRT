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

#include "vlaDualRate.h"

#include "common/checkMacros.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace vla
{

//! Asynchronous action chunking: the control loop executes one chunk row per tick while the next chunk is
//! computed on a planner thread.
//!
//! Chunk row r of a chunk planned at tick t is the action for tick t + r. Once replanAfter rows of the current
//! chunk have been issued, step() snapshots the observation (on the control thread) and requests the next chunk
//! for the current tick; with real-time chunking the planner seeds it from the current chunk starting at row
//! tick - currentChunkTick, so set replanAfter = horizon - overlap and cover the planner latency (in ticks) with
//! frozen rows. When the new chunk lands the loop continues in it at row tick - plannedTick, skipping the rows
//! the latency consumed.
//!
//! \tparam Observation what the planner needs from the robot (frames, state), copied at the request tick.
template <typename Observation>
class AsyncChunker
{
public:
    struct Chunk
    {
        std::vector<float> actions; //!< [horizon, actionDim], row-major
        int64_t observationIndex{-1};
    };
    //! Captures the robot's observation for \p tick. Runs on the control thread inside step().
    using Snapshot = std::function<Observation(int64_t tick)>;
    //! Plans the chunk whose row 0 is the action for \p tick from that tick's snapshot. Runs on the planner thread.
    using Planner = std::function<std::vector<float>(int64_t tick, Observation const& observation)>;

    AsyncChunker(int32_t horizon, int32_t actionDim, int32_t replanAfter, Snapshot snapshot, Planner planner)
        : mHorizon(horizon)
        , mActionDim(actionDim)
        , mReplanAfter(replanAfter)
        , mSnapshot(std::move(snapshot))
    {
        ELLM_CHECK(horizon > 0 && actionDim > 0 && replanAfter >= 0 && replanAfter < horizon,
            "AsyncChunker: replanAfter must lie inside the chunk");
        // mObservation is written by step() only while no request is in flight (see step()), and the driver's
        // mutex orders that write before the planner thread reads it.
        mDriver = std::make_unique<DualRateDriver<Chunk>>(mState, [this, planner = std::move(planner)](int64_t tick) {
            Chunk chunk;
            chunk.actions = planner(tick, mObservation);
            ELLM_CHECK(static_cast<int64_t>(chunk.actions.size()) == static_cast<int64_t>(mHorizon) * mActionDim,
                "AsyncChunker: the planner returned a chunk of the wrong size");
            return chunk;
        });
    }

    //! The action for \p tick (actionDim values, valid until the next call), or nullptr when no chunk covers it:
    //! before the first chunk lands, or when the planner fell behind past the end of the current chunk.
    float const* step(int64_t tick)
    {
        int64_t const staleness = mState.stalenessAt(tick);
        if (staleness >= 0 && tick - staleness != mCurrent.observationIndex)
        {
            mState.latest(mCurrent);
            mLastAdoptionLag = tick - mCurrent.observationIndex;
        }

        int64_t const row = mCurrent.observationIndex < 0 ? -1 : tick - mCurrent.observationIndex;
        // One request in flight at a time: the next is only made once the chunk it continues has landed, i.e. after
        // the planner has finished with the previous snapshot.
        if ((row < 0 || row >= mReplanAfter) && mRequestedFor <= mCurrent.observationIndex)
        {
            mObservation = mSnapshot(tick);
            mDriver->requestReplan(tick);
            mRequestedFor = tick;
        }
        if (row < 0 || row >= mHorizon)
        {
            return nullptr;
        }
        return mCurrent.actions.data() + row * mActionDim;
    }

    //! The chunk step() reads from; observationIndex is -1 before the first one lands.
    Chunk const& current() const noexcept
    {
        return mCurrent;
    }
    //! Ticks the current chunk had already advanced into when it was adopted (planner latency in ticks).
    int64_t lastAdoptionLag() const noexcept
    {
        return mLastAdoptionLag;
    }
    int64_t chunksCompleted() const
    {
        return mDriver->plansCompleted();
    }

    void stop() noexcept
    {
        mDriver->stop();
    }

private:
    int32_t mHorizon;
    int32_t mActionDim;
    int32_t mReplanAfter;
    Snapshot mSnapshot;
    Observation mObservation{};
    DualRateState<Chunk> mState;
    std::unique_ptr<DualRateDriver<Chunk>> mDriver;
    Chunk mCurrent;
    int64_t mRequestedFor{-1};
    int64_t mLastAdoptionLag{0};
};

} // namespace vla
} // namespace trt_edgellm
