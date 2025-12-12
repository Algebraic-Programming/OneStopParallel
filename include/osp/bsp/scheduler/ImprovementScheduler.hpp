/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once
#include "Scheduler.hpp"

namespace osp {

/**
 * @class ImprovementScheduler
 * @brief Abstract base class for improvement scheduling scheduler.
 *
 * The ImprovementScheduler class provides a common interface for improvement scheduling scheduler.
 * Subclasses of this class can implement specific improvement scheduler by overriding the virtual methods.
 */
template <typename GraphT>
class ImprovementScheduler {
  protected:
    unsigned timeLimitSeconds_; /**< The time limit in seconds for the improvement algorithm. */

  public:
    /**
     * @brief Constructor for ImprovementScheduler.
     * @param timelimit The time limit in seconds for the improvement algorithm. Default is 3600 seconds (1 hour).
     */
    ImprovementScheduler(unsigned timelimit = 3600) : timeLimitSeconds_(timelimit) {}

    /**
     * @brief Destructor for ImprovementScheduler.
     */
    virtual ~ImprovementScheduler() = default;

    /**
     * @brief Set the time limit in seconds for the improvement algorithm.
     * @param limit The time limit in seconds.
     */
    virtual void SetTimeLimitSeconds(unsigned int limit) { timeLimitSeconds_ = limit; }

    /**
     * @brief Set the time limit in hours for the improvement algorithm.
     * @param limit The time limit in hours.
     */
    virtual void SetTimeLimitHours(unsigned int limit) { timeLimitSeconds_ = limit * 3600; }

    /**
     * @brief Get the time limit in seconds for the improvement algorithm.
     * @return The time limit in seconds.
     */
    inline unsigned int GetTimeLimitSeconds() const { return timeLimitSeconds_; }

    /**
     * @brief Get the time limit in hours for the improvement algorithm.
     * @return The time limit in hours.
     */
    inline unsigned int GetTimeLimitHours() const { return timeLimitSeconds_ / 3600; }

    /**
     * @brief Get the name of the improvement scheduling algorithm.
     * @return The name of the algorithm as a string.
     */
    virtual std::string GetScheduleName() const = 0;

    /**
     * @brief Improve the given BspSchedule.
     * @param schedule The BspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT> &schedule) = 0;

    /**
     * @brief Improve the given BspSchedule within the time limit.
     * @param schedule The BspSchedule to be improved.
     * @return The status of the improvement operation.
     */
    virtual ReturnStatus ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &schedule) = 0;
};

template <typename GraphT>
class ComboScheduler : public Scheduler<GraphT> {
  private:
    Scheduler<GraphT> &baseScheduler_;
    ImprovementScheduler<GraphT> &improvementScheduler_;

  public:
    ComboScheduler(Scheduler<GraphT> &base, ImprovementScheduler<GraphT> &improvement)
        : Scheduler<GraphT>(), baseScheduler_(base), improvementScheduler_(improvement) {}

    virtual ~ComboScheduler() = default;

    virtual std::string getScheduleName() const override {
        return baseScheduler_.getScheduleName() + "+" + improvementScheduler_.getScheduleName();
    }

    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        ReturnStatus status = baseScheduler_.ComputeSchedule(schedule);
        if (status != ReturnStatus::OSP_SUCCESS and status != ReturnStatus::BEST_FOUND) {
            return status;
        }

        return improvementScheduler_.improveSchedule(schedule);
    }
};

}    // namespace osp
