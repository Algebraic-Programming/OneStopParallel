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
#include "model/DAGPartition.hpp"
#include "scheduler/Scheduler.hpp"

/**
 * @class Partitioner
 * @brief Abstract base class for partitioning scheduler.
 * 
 * The Partitioner class provides a common interface for partitioning scheduler.
 * It defines methods for setting and getting the time limit, as well as computing partitions.
 */
class Partitioner {

    protected:
        unsigned int timeLimitSeconds; /**< The time limit in seconds for computing a partition. */
        bool use_memory_constraint; /**< Whether or not the partitioner should consider memory constraints. */

    public:
        /**
         * @brief Constructor for the Partitioner class.
         * @param timelimit The time limit in seconds for computing a partition. Default is 3600 seconds (1 hour).
         */
        Partitioner(unsigned timelimit = 3600, bool use_memory_constraint_ = false) : timeLimitSeconds(timelimit), use_memory_constraint(use_memory_constraint_) {}

        /**
         * @brief Destructor for the Partitioner class.
         */
        virtual ~Partitioner() = default;

        /**
         * @brief Set the time limit in seconds for computing a partition.
         * @param limit The time limit in seconds.
         */
        virtual void setTimeLimitSeconds(unsigned int limit) { timeLimitSeconds = limit; }

        /**
         * @brief Set the time limit in hours for computing a partition.
         * @param limit The time limit in hours.
         */
        virtual void setTimeLimitHours(unsigned int limit) { timeLimitSeconds = limit * 3600; }

        /**
         * @brief Get the time limit in seconds for computing a partition.
         * @return The time limit in seconds.
         */
        inline unsigned int getTimeLimitSeconds() const { return timeLimitSeconds; }

        /**
         * @brief Get the time limit in hours for computing a partition.
         * @return The time limit in hours.
         */
        inline unsigned int getTimeLimitHours() const { return timeLimitSeconds / 3600; }

        /**
         * @brief Get the name of the partitioning algorithm.
         * @return The name of the partitioning algorithm.
         */
        virtual std::string getPartitionerName() const = 0;

        /**
         * @brief Compute a partition for the given BSP instance.
         * @param instance The BSP instance for which to compute the partition.
         * @return A pair containing the return status and the computed partition.
         */
        virtual std::pair<RETURN_STATUS, DAGPartition> computePartition(const BspInstance &instance) = 0;

        /**
         * @brief Compute a partition for the given BSP instance within the time limit.
         * @param instance The BSP instance for which to compute the partition.
         * @return A pair containing the return status and the computed partition.
         */
        virtual std::pair<RETURN_STATUS, DAGPartition> computePartitionWithTimeLimit(const BspInstance &instance);


        virtual void setUseMemoryConstraint(bool use_memory_constraint_) { use_memory_constraint = use_memory_constraint_;}
};

