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

#include "model/BspSchedule.hpp"

/**
 * @class CSchedule
 * @brief Represents a classic schedule for scheduling scheduler.
 * 
 * This class stores the processor and time information for a schedule.
 */
class CSchedule {

  public:
    std::vector<int> proc; /**< The processor assigned to each task. */
    std::vector<int> time; /**< The time at which each task starts. */

    /**
     * @brief Constructs a CSchedule object with the given size.
     * @param size The size of the schedule.
     */
    CSchedule(unsigned size) : proc(std::vector<int>(size, -1)), time(std::vector<int>(size,0)) {}

    /**
     * @brief Converts the CSchedule object to a BspSchedule object.
     * @param instance The BspInstance object representing the BSP instance.
     * @param local_greedyProcLists The local greedy processor lists.
     * @return The converted BspSchedule object.
     */
    BspSchedule convertToBspSchedule(const BspInstance &instance,
                                     const std::vector<std::deque<unsigned>> &local_greedyProcLists);
};