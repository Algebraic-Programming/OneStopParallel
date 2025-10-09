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

@author Toni Boehnlein, Christos Matzoros, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include "osp/bsp/model/MaxBspSchedule.hpp"
#include "osp/bsp/model/MaxBspScheduleCS.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"

namespace osp {

/**
 * @class Scheduler
 * @brief Abstract base class for scheduling scheduler.
 *
 * The Scheduler class provides a common interface for scheduling scheduler in the BSP scheduling system.
 * It defines methods for setting and getting the time limit, as well as computing schedules.
 */
template<typename Graph_t>
class MaxBspScheduler : public Scheduler<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "BspSchedule can only be used with computational DAGs.");

    /**
     * @brief Get the name of the scheduling algorithm.
     * @return The name of the scheduling algorithm.
     */
    virtual std::string getScheduleName() const override = 0;

    /**
     * @brief Compute a BSP schedule for the given BSP instance.
     * @param instance The BSP instance for which to compute the schedule.
     * @return A pair containing the return status and the computed schedule.
     */
    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {
        MaxBspSchedule<Graph_t> tmpSched(schedule.getInstance());
        RETURN_STATUS status = computeSchedule(tmpSched);
        schedule = tmpSched;
        return status;
    }

    virtual RETURN_STATUS computeScheduleCS(BspScheduleCS<Graph_t> &schedule) {

        auto result = computeSchedule(schedule);
        if (result == RETURN_STATUS::OSP_SUCCESS || result == RETURN_STATUS::BEST_FOUND) {
            schedule.setAutoCommunicationSchedule();
            return result;
        } else {
            return RETURN_STATUS::ERROR;
        }
    }

    /**
     * @brief Compute a BSP schedule for the given BSP instance.
     * @param instance The BSP instance for which to compute the schedule.
     * @return A pair containing the return status and the computed schedule.
     */
    virtual RETURN_STATUS computeSchedule(MaxBspSchedule<Graph_t> &schedule) = 0;

    virtual RETURN_STATUS computeScheduleCS(MaxBspScheduleCS<Graph_t> &schedule) {
// Fix me todo
        auto result = computeSchedule(schedule);
        if (result == RETURN_STATUS::OSP_SUCCESS || result == RETURN_STATUS::BEST_FOUND) {
            // schedule.setAutoCommunicationSchedule();
            return result;
        } else {
            return RETURN_STATUS::ERROR;
        }
    }
};

} // namespace osp