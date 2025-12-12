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

#include <string>

#include "osp/auxiliary/return_status.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/BspScheduleCS.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

/**
 * @class Scheduler
 * @brief Interface for BSP schedulers.
 *
 * The Scheduler class defines the common interface for all scheduling algorithms computing BSP schedules.
 * It specifies the contract for computing standard BSP schedules (BspSchedule) and communication-aware schedules
 * (BspScheduleCS).
 */
template <typename GraphT>
class Scheduler {
    static_assert(IsComputationalDagV<GraphT>, "Scheduler can only be used with computational DAGs.");

  public:
    /**
     * @brief Constructor for the Scheduler class.
     */
    Scheduler() = default;

    /**
     * @brief Destructor for the Scheduler class.
     */
    virtual ~Scheduler() = default;

    /**
     * @brief Get the name of the scheduling algorithm.
     * @return The name of the scheduling algorithm.
     */
    virtual std::string GetScheduleName() const = 0;

    /**
     * @brief Computes a BSP schedule for the given BSP instance.
     *
     * This pure virtual function must be implemented by derived classes to provide
     * the specific scheduling logic. It modifies the passed BspSchedule object.
     *
     * @param schedule The BspSchedule object to be computed. It contains the BspInstance.
     * @return ReturnStatus::OSP_SUCCESS if a schedule was successfully computed,
     *         ReturnStatus::ERROR if an error occurred, or other status codes as appropriate.
     */
    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) = 0;

    /**
     * @brief Computes a BSP schedule with communication schedule (CS).
     *
     * This method provides a default implementation that first computes the basic BSP schedule using computeSchedule().
     * If successful, it then calls setAutoCommunicationSchedule() on the schedule to set a communication schedule.
     *
     * @param schedule The BspScheduleCS object to be computed. It contains the BspInstance.
     * @return ReturnStatus::OSP_SUCCESS or ReturnStatus::BEST_FOUND if a schedule was successfully computed,
     *         ReturnStatus::ERROR if an error occurred, or other status codes as appropriate.
     */
    virtual ReturnStatus ComputeScheduleCs(BspScheduleCS<GraphT> &schedule) {
        auto result = computeSchedule(schedule);
        if (result == ReturnStatus::OSP_SUCCESS || result == ReturnStatus::BEST_FOUND) {
            schedule.setAutoCommunicationSchedule();
            return result;
        } else {
            return ReturnStatus::ERROR;
        }
    }
};

}    // namespace osp
