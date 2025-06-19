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

#include "bsp/model/BspInstance.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "bsp/model/BspScheduleCS.hpp"
#include "concepts/computational_dag_concept.hpp"

namespace osp {



/**
 * @class Scheduler
 * @brief Abstract base class for scheduling scheduler.
 *
 * The Scheduler class provides a common interface for scheduling scheduler in the BSP scheduling system.
 * It defines methods for setting and getting the time limit, as well as computing schedules.
 */
template<typename Graph_t>
class Scheduler {

    static_assert(is_computational_dag_v<Graph_t>, "BspSchedule can only be used with computational DAGs.");

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
    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) = 0;

    virtual RETURN_STATUS computeScheduleCS(BspScheduleCS<Graph_t> &schedule) {

        auto result = computeSchedule(schedule);
        if (result == SUCCESS || result == BEST_FOUND) {
            schedule.setAutoCommunicationSchedule();
            return result;
        } else {
            return ERROR;
        }
    }

    // /**
    //  * @brief Compute a schedule for the given BSP instance within the time limit.
    //  * @param instance The BSP instance for which to compute the schedule.
    //  * @return A pair containing the return status and the computed schedule.
    //  */
    // virtual std::pair<RETURN_STATUS, BspSchedule<Graph_t>>
    // computeScheduleWithTimeLimit(const BspInstance<Graph_t> &instance) {

    //     std::packaged_task<std::pair<RETURN_STATUS, BspSchedule<Graph_t>>(const BspInstance<Graph_t> &)> task(
    //         [this](const BspInstance<Graph_t> &instance) -> std::pair<RETURN_STATUS, BspSchedule<Graph_t>> {
    //             return computeSchedule(instance);
    //         });
    //     auto future = task.get_future();
    //     std::thread thr(std::move(task), std::ref(instance));
    //     if (future.wait_for(std::chrono::seconds(getTimeLimitSeconds())) == std::future_status::timeout) {
    //         thr.detach(); // we leave the thread still running
    //         std::cerr << "Timeout reached, execution of computeSchedule() aborted" << std::endl;
    //         return std::make_pair(TIMEOUT, BspSchedule<Graph_t>());
    //     }
    //     thr.join();
    //     try {
    //         const auto result = future.get();
    //         return result;
    //     } catch (const std::exception &e) {
    //         std::cerr << "Exception caught in computeScheduleWithTimeLimit(): " << e.what() << std::endl;
    //         return std::make_pair(ERROR, BspSchedule<Graph_t>());
    //     }
    // }
};

} // namespace osp