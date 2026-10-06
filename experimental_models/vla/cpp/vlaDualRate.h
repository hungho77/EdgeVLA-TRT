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

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace trt_edgellm
{
namespace vla
{

//! \brief The handoff between a slow planner and a fast control loop.
//!
//! The planner (a VLM replan, or a whole action chunk) runs at a low rate; the control loop consumes **the
//! latest plan available** rather than waiting for a fresh one. The contract:
//!
//! * The control loop never blocks on the planner. If no plan has landed yet, `latest()` says so.
//! * A plan is published atomically; half of one plan and half of another is worse than a stale plan.
//! * Staleness is observable, so the caller can enforce a bound or skip the actions the robot already passed.
//!
//! \tparam PlanT copyable, with an `int64_t observationIndex` member (the observation it was computed from).
template <typename PlanT>
class DualRateState
{
public:
    using Plan = PlanT;

    enum class Mode : int32_t
    {
        kSync = 0,         //!< replan for every observation; deterministic, for offline evaluation
        kPartialAsync = 1, //!< replan on a cadence and run on the newest plan; what the robot wants
    };

    explicit DualRateState(Mode mode = Mode::kPartialAsync)
        : mMode(mode)
    {
    }

    Mode mode() const noexcept
    {
        return mMode;
    }

    //! Replaces any previous plan; plans are not queued, a backlog would steer on superseded plans.
    void publish(Plan plan)
    {
        std::lock_guard<std::mutex> const guard(mMutex);
        mPlan = std::move(plan);
        mHasPlan = true;
    }

    //! \return false when nothing has been published yet, leaving \p out untouched.
    bool latest(Plan& out) const
    {
        std::lock_guard<std::mutex> const guard(mMutex);
        if (!mHasPlan)
        {
            return false;
        }
        out = mPlan;
        return true;
    }

    //! Observations since the newest plan's own observation, or -1 if none published.
    int64_t stalenessAt(int64_t currentObservationIndex) const
    {
        std::lock_guard<std::mutex> const guard(mMutex);
        return mHasPlan ? currentObservationIndex - mPlan.observationIndex : -1;
    }

    //! True on the cadence, in kSync mode, and always when \p forced.
    bool shouldReplan(int64_t observationIndex, int64_t cadence, bool forced) const
    {
        return forced || mMode == Mode::kSync || cadence <= 0 || observationIndex % cadence == 0;
    }

private:
    Mode mMode;
    mutable std::mutex mMutex;
    Plan mPlan{};
    bool mHasPlan{false};
};

//! \brief Runs the planner on its own thread and publishes into a DualRateState.
//!
//! The planner is injected: what a plan is (which frames, prompt, engines) belongs to the model. Requests
//! coalesce: one arriving while a plan is in flight replaces the pending one instead of queueing behind it.
template <typename PlanT>
class DualRateDriver
{
public:
    using State = DualRateState<PlanT>;
    //! Computes a plan for an observation. Runs on the planner thread, never on the caller's.
    using Planner = std::function<PlanT(int64_t observationIndex)>;

    DualRateDriver(State& state, Planner planner)
        : mState(state)
        , mPlanner(std::move(planner))
    {
        mThread = std::thread(&DualRateDriver::run, this);
    }

    ~DualRateDriver() noexcept
    {
        stop();
    }

    DualRateDriver(DualRateDriver const&) = delete;
    DualRateDriver& operator=(DualRateDriver const&) = delete;

    //! Ask for a plan at \p observationIndex. Returns immediately.
    void requestReplan(int64_t observationIndex)
    {
        {
            std::lock_guard<std::mutex> const guard(mMutex);
            mPending = observationIndex;
        }
        mWakeCv.notify_one();
    }

    //! Block until no plan is in flight. For tests and shutdown, not the hot loop.
    void waitIdle()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        // mStop is part of the predicate: once the planner thread has exited nothing notifies mIdleCv again.
        mIdleCv.wait(lock, [this] { return mStop || (!mBusy && mPending < 0); });
    }

    //! Stop the planner thread. Idempotent; also called by the destructor.
    void stop() noexcept
    {
        {
            std::lock_guard<std::mutex> const guard(mMutex);
            if (mStop)
            {
                return;
            }
            mStop = true;
        }
        mWakeCv.notify_all();
        mIdleCv.notify_all();
        if (mThread.joinable())
        {
            mThread.join();
        }
    }

    int64_t plansCompleted() const
    {
        std::lock_guard<std::mutex> const guard(mMutex);
        return mCompleted;
    }

private:
    void run()
    {
        while (true)
        {
            int64_t observationIndex = -1;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mWakeCv.wait(lock, [this] { return mStop || mPending >= 0; });
                if (mStop)
                {
                    return;
                }
                observationIndex = mPending;
                mPending = -1;
                mBusy = true;
            }

            // Unlocked, so a slow plan cannot block requestReplan.
            PlanT plan = mPlanner(observationIndex);
            plan.observationIndex = observationIndex;
            mState.publish(std::move(plan));

            {
                std::lock_guard<std::mutex> const guard(mMutex);
                mBusy = false;
                ++mCompleted;
            }
            mIdleCv.notify_all();
        }
    }

    State& mState;
    Planner mPlanner;
    std::thread mThread;
    mutable std::mutex mMutex;
    std::condition_variable mWakeCv;
    std::condition_variable mIdleCv;
    int64_t mPending{-1};
    bool mBusy{false};
    bool mStop{false};
    int64_t mCompleted{0};
};

} // namespace vla
} // namespace trt_edgellm
