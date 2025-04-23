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

#ifdef EIGEN_FOUND

#include <climits>
#include <set>
#include <string>
#include <vector>

#include <omp.h>
#include <stdlib.h>

#include "scheduler/Scheduler.hpp"
#include "auxiliary/auxiliary.hpp"
#include "model/SmSchedule.hpp"
#include "model/SmInstance.hpp"

struct SMFunGrowlv2_Params {
  // Funnel
  double maxWeightAverageMultiplier = 5.0;

  // Grow local
  unsigned minSuperstepSize = 20;
  unsigned syncCostMultiplierMinSuperstepWeight = 1;
  unsigned syncCostMultiplierParallelCheck = 4;

  unsigned numThreads = 0; // 0 for auto
  unsigned maxNumThreads = UINT_MAX; // used when auto num threads
};


/**
 * @brief The GreedyBspScheduler class represents a scheduler that uses a greedy algorithm to compute schedules for
 * SmInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given SmInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "BspGreedy" in this case.
 */
class SMFunGrowlv2 {

  private:

  SMFunGrowlv2_Params params;

  public:
    /**
     * @brief Default constructor for GreedyBspScheduler.
     */
    SMFunGrowlv2(SMFunGrowlv2_Params params_ = SMFunGrowlv2_Params()) : params(params_) {}

    /**
     * @brief Default destructor for GreedyBspScheduler.
     */
    virtual ~SMFunGrowlv2() = default;

    /**
     * @brief Compute a schedule for the given SmInstance.
     *
     * This method computes a schedule for the given SmInstance using a greedy algorithm.
     *
     * @param instance The SmInstance object representing the instance to compute the schedule for.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, SmSchedule> computeSmSchedule(const SmInstance &instance) const;

    void computeTransitiveReduction(const SmInstance &instance, bool *const transEdgeMask, const unsigned int startNode, const unsigned int endNode) const;

    std::pair<RETURN_STATUS, SmSchedule> computeScheduleParallel(const SmInstance &instance, unsigned int numThreads) const;
    void computePartialSchedule(const SmInstance &instance, const bool *const transEdgeMask, unsigned *const node_to_proc, unsigned *const node_to_supstep,  const unsigned int startNode, const unsigned int endNode, unsigned &supstep) const;    
    void incrementScheduleSupersteps(unsigned *const node_to_supstep,  const unsigned startNode, const unsigned endNode, const unsigned incr) const;
    


    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "SMFunGrowl" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const {

        return "SMFunGrowlv2";
    }

};

#endif