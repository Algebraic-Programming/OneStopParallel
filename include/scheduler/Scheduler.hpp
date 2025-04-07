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

#include <chrono>
#include <future>
#include <iostream>
#include <thread>

#include "model/bsp/BspInstance.hpp"
#include "model/bsp/BspSchedule.hpp"

namespace osp {

enum RETURN_STATUS { SUCCESS, BEST_FOUND, TIMEOUT, ERROR };

inline std::string to_string(const RETURN_STATUS status) {
    switch (status) {
    case SUCCESS:
        return "SUCCESS";
    case BEST_FOUND:
        return "BEST FOUND";
    case TIMEOUT:
        return "TIMEOUT";
    case ERROR:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

/**
 * @class Scheduler
 * @brief Abstract base class for scheduling scheduler.
 *
 * The Scheduler class provides a common interface for scheduling scheduler in the BSP scheduling system.
 * It defines methods for setting and getting the time limit, as well as computing schedules.
 */
template<typename Graph_t>
class Scheduler {

  protected:
    unsigned int timeLimitSeconds; /**< The time limit in seconds for computing a schedule. */

  public:
    /**
     * @brief Constructor for the Scheduler class.
     * @param timelimit The time limit in seconds for computing a schedule. Default is 3600 seconds (1 hour).
     */
    Scheduler(unsigned timelimit = 3600) : timeLimitSeconds(timelimit) {}

    /**
     * @brief Destructor for the Scheduler class.
     */
    virtual ~Scheduler() = default;

    /**
     * @brief Set the time limit in seconds for computing a schedule.
     * @param limit The time limit in seconds.
     */
    virtual void setTimeLimitSeconds(unsigned int limit) { timeLimitSeconds = limit; }

    /**
     * @brief Set the time limit in hours for computing a schedule.
     * @param limit The time limit in hours.
     */
    virtual void setTimeLimitHours(unsigned int limit) { timeLimitSeconds = limit * 3600; }

    /**
     * @brief Get the time limit in seconds for computing a schedule.
     * @return The time limit in seconds.
     */
    inline unsigned int getTimeLimitSeconds() const { return timeLimitSeconds; }

    /**
     * @brief Get the time limit in hours for computing a schedule.
     * @return The time limit in hours.
     */
    inline unsigned int getTimeLimitHours() const { return timeLimitSeconds / 3600; }

    /**
     * @brief Get the name of the scheduling algorithm.
     * @return The name of the scheduling algorithm.
     */
    virtual std::string getScheduleName() const = 0;

    /**
     * @brief Compute a BSP schedule for the given BSP instance.
     * @param instance The BSP instance for which to compute the schedule.
     * @return A pair containing the return status and the computed schedule.
     */

    virtual std::pair<RETURN_STATUS, BspSchedule<Graph_t>> computeSchedule(const BspInstance<Graph_t> &instance) = 0;

    /**
     * @brief Compute a schedule for the given BSP instance within the time limit.
     * @param instance The BSP instance for which to compute the schedule.
     * @return A pair containing the return status and the computed schedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule<Graph_t>>
    computeScheduleWithTimeLimit(const BspInstance<Graph_t> &instance);

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) { throw std::runtime_error("Not implemented"); }
};

} // namespace osp