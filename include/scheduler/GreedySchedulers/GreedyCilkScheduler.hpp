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
#include <climits>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ClassicSchedule.hpp"
#include "scheduler/Scheduler.hpp"
#include "auxiliary/auxiliary.hpp"
#include "model/BspSchedule.hpp"

enum CilkMode { CILK, SJF, RANDOM };


/**
 * @class GreedyCilkScheduler
 * @brief A class that represents a greedy scheduler for Cilk-based BSP scheduling scheduler.
 * 
 * The GreedyCilkScheduler class is a concrete implementation of the Scheduler base class. It provides
 * a greedy scheduling algorithm for Cilk-based BSP (Bulk Synchronous Parallel) systems. The scheduler
 * selects the next node and processor to execute a task based on a greedy strategy.
 */
class GreedyCilkScheduler : public Scheduler {

  private:
    CilkMode mode; /**< The mode of the Cilk scheduler. */


    void Choose(const BspInstance &instance, std::vector<std::deque<unsigned>> &procQueue,
                const std::set<unsigned> &readyNodes, const std::vector<bool> &procFree, unsigned &node, unsigned &p);

  public:
    /**
     * @brief Constructs a GreedyCilkScheduler object with the specified Cilk mode.
     * 
     * This constructor initializes a GreedyCilkScheduler object with the specified Cilk mode.
     * 
     * @param mode_ The Cilk mode for the scheduler.
     */
    GreedyCilkScheduler(CilkMode mode_ = CILK) : Scheduler(), mode(mode_) {}

    /**
     * @brief Destroys the GreedyCilkScheduler object.
     * 
     * This destructor destroys the GreedyCilkScheduler object.
     */
    virtual ~GreedyCilkScheduler() = default;

    /**
     * @brief Computes the schedule for the given BSP instance using the greedy scheduling algorithm.
     * 
     * This member function computes the schedule for the given BSP instance using the greedy scheduling algorithm.
     * It overrides the computeSchedule() function of the base Scheduler class.
     * 
     * @param instance The BSP instance to compute the schedule for.
     * @return A pair containing the return status and the computed BSP schedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    /**
     * @brief Sets the Cilk mode for the scheduler.
     * 
     * This member function sets the Cilk mode for the scheduler.
     * 
     * @param mode_ The Cilk mode to set.
     */
    inline void setMode(CilkMode mode_) { mode = mode_; }

    /**
     * @brief Gets the Cilk mode of the scheduler.
     * 
     * This member function gets the Cilk mode of the scheduler.
     * 
     * @return The Cilk mode of the scheduler.
     */
    inline CilkMode getMode() const { return mode; }

    /**
     * @brief Gets the name of the schedule.
     * 
     * This member function gets the name of the schedule based on the Cilk mode.
     * 
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override {

        switch (mode) {
        case CILK:
            return "CilkGreedy";
            break;

        case SJF:
            return "SJFGreedy";

        case RANDOM:
            return "RandomGreedy";

        default:
            return "UnknownModeGreedy";

        }
    }
};
