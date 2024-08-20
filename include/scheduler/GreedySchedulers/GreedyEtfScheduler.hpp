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

enum EtfMode { ETF, BL_EST };

/**
 * @brief The GreedyEtfScheduler class is a subclass of the Scheduler class and implements a greedy scheduling algorithm.
 * 
 * This class provides methods to compute a schedule using the greedy ETF (Earliest Task First) algorithm.
 * It calculates the bottom level of each task and uses it to determine the earliest start time (EST) for each task on each processor.
 * The algorithm selects the task with the earliest EST and assigns it to the processor with the earliest available start time.
 * The process is repeated until all tasks are scheduled.
 */
class GreedyEtfScheduler : public Scheduler {

  private:
    EtfMode mode; // The mode of the scheduler (ETF or BL_EST)
    bool use_numa; // Flag indicating whether to use NUMA-aware scheduling

    /**
     * @brief Computes the bottom level of each task.
     * 
     * @param instance The BspInstance object representing the BSP instance.
     * @param avg_ The average execution time of the tasks.
     * @return A vector containing the bottom level of each task.
     */
    std::vector<int> ComputeBottomLevel(const BspInstance &instance, unsigned avg_) const;

    /**
     * @brief Calculates the earliest start time (EST) for a task on a processor.
     * 
     * @param instance The BspInstance object representing the BSP instance.
     * @param schedule The current schedule.
     * @param node The node (processor) on which the task is to be scheduled.
     * @param proc The processor index.
     * @param procAvailableFrom The earliest available start time for each processor.
     * @param send The send buffer sizes for each node.
     * @param rec The receive buffer sizes for each node.
     * @param avg_ The average execution time of the tasks.
     * @return The earliest start time (EST) for the task on the processor.
     */
    int GetESTforProc(const BspInstance &instance, CSchedule &schedule, int node, int proc, const int procAvailableFrom,
                      std::vector<int> &send, std::vector<int> &rec, unsigned avg_) const;

    /**
     * @brief Finds the best EST for a set of nodes.
     * 
     * @param instance The BspInstance object representing the BSP instance.
     * @param schedule The current schedule.
     * @param nodeList The list of nodes to consider.
     * @param procAvailableFrom The earliest available start time for each processor.
     * @param send The send buffer sizes for each node.
     * @param rec The receive buffer sizes for each node.
     * @param avg_ The average execution time of the tasks.
     * @return A triple containing the best EST, the node index, and the processor index.
     */
    intTriple GetBestESTforNodes(const BspInstance &instance, CSchedule &schedule, const std::vector<int> &nodeList,
                                 const std::vector<int> &procAvailableFrom, std::vector<int> &send,
                                 std::vector<int> &rec, unsigned avg_) const;

  public:
    /**
     * @brief Constructs a GreedyEtfScheduler object with the specified mode.
     * 
     * @param mode_ The mode of the scheduler (ETF or BL_EST).
     */
    GreedyEtfScheduler(EtfMode mode_ = ETF) : Scheduler(), mode(mode_), use_numa(false) {}

    /**
     * @brief Default destructor for the GreedyEtfScheduler class.
     */
    virtual ~GreedyEtfScheduler() = default;

    /**
     * @brief Computes a schedule for the given BSP instance using the greedy ETF algorithm.
     * 
     * @param instance The BspInstance object representing the BSP instance.
     * @return A pair containing the return status and the computed BspSchedule object.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    /**
     * @brief Sets the mode of the scheduler.
     * 
     * @param mode_ The mode of the scheduler (ETF or BL_EST).
     */
    inline void setMode(EtfMode mode_) { mode = mode_; }

    /**
     * @brief Gets the mode of the scheduler.
     * 
     * @return The mode of the scheduler (ETF or BL_EST).
     */
    inline EtfMode getMode() const { return mode; }

    /**
     * @brief Sets whether to use NUMA-aware scheduling.
     * 
     * @param numa Flag indicating whether to use NUMA-aware scheduling.
     */
    inline void setUseNuma(bool numa) { use_numa = numa; }

    /**
     * @brief Checks if NUMA-aware scheduling is enabled.
     * 
     * @return True if NUMA-aware scheduling is enabled, false otherwise.
     */
    inline bool useNuma() const { return use_numa; }

    /**
     * @brief Gets the name of the schedule.
     * 
     * @return The name of the schedule based on the mode.
     */
    virtual std::string getScheduleName() const override {
        switch (mode) {
        case ETF:
            return "ETFGreedy";

        case BL_EST:
            return "BL-ESTGreedy";
        
        default:
            return "UnknownModeGreedy";
        }
    }
};
