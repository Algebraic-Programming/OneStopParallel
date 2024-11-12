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

#include "model/BspInstance.hpp"
#include "model/BspSchedule.hpp"
#include "model/SspSchedule.hpp"
#include "scheduler/Scheduler.hpp"

/**
 * @class SSPScheduler
 * @brief Abstract base class for scheduling SSP.
 * 
 * The Scheduler class provides a common interface for scheduling scheduler in the SSP scheduling system.
 * It defines methods for setting and getting the time limit, as well as computing schedules.
 */
class SSPScheduler : public Scheduler {

    public:
        /**
         * @brief Constructor for the Scheduler class.
         * @param timelimit The time limit in seconds for computing a schedule. Default is 3600 seconds (1 hour).
         */
        SSPScheduler(unsigned timelimit = 3600) : Scheduler(timelimit) {}

        /**
         * @brief Destructor for the Scheduler class.
         */
        virtual ~SSPScheduler() = default;

                /**
         * @brief Set the time limit in seconds for computing a schedule.
         * @param limit The time limit in seconds.
         */
        virtual void setTimeLimitSeconds(unsigned int limit) override { timeLimitSeconds = limit; }

        /**
         * @brief Set the time limit in hours for computing a schedule.
         * @param limit The time limit in hours.
         */
        virtual void setTimeLimitHours(unsigned int limit) override { timeLimitSeconds = limit * 3600; }

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
        virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override { return computeSspSchedule(instance, 1); };

        /**
         * @brief Compute a SSP schedule for the given BSP instance.
         * @param instance The BSP instance for which to compute the schedule.
         * @param stale staleness of the schedule
         * @return A pair containing the return status and the computed schedule.
         */
        virtual std::pair<RETURN_STATUS, SspSchedule> computeSspSchedule(const BspInstance &instance, unsigned stale) = 0;

        /**
         * @brief Compute a SSP schedule for the given BSP instance within the time limit.
         * @param instance The BSP instance for which to compute the schedule.
         * @return A pair containing the return status and the computed schedule.
         */
        virtual std::pair<RETURN_STATUS, SspSchedule> computeSspScheduleWithTimeLimit(const BspInstance &instance, unsigned stale);

        virtual void setUseMemoryConstraint(bool use_memory_constraint_) override { throw std::runtime_error("Not implemented");}
};

