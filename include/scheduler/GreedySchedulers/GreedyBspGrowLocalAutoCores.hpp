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


#include "scheduler/Scheduler.hpp"
#include "auxiliary/auxiliary.hpp"
#include "model/BspSchedule.hpp"

struct GreedyBspGrowLocalAutoCores_Params {
  unsigned minSuperstepSize = 20;
  unsigned syncCostMultiplierMinSuperstepWeight = 1;
  unsigned syncCostMultiplierParallelCheck = 4;
};


/**
 * @brief The GreedyBspGrowLocalAutoCores class represents a scheduler that uses a greedy algorithm to compute schedules for
 * BspInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "GreedyBspGrowLocalAutoCores" in this case.
 */
class GreedyBspGrowLocalAutoCores : public Scheduler {

  private:

    GreedyBspGrowLocalAutoCores_Params params;


  public:
    /**
     * @brief Default constructor for GreedyBspGrowLocalAutoCores.
     */
    GreedyBspGrowLocalAutoCores(GreedyBspGrowLocalAutoCores_Params params_ = GreedyBspGrowLocalAutoCores_Params()): Scheduler(), params(params_) {}

    /**
     * @brief Default destructor for GreedyBspGrowLocalAutoCores.
     */
    virtual ~GreedyBspGrowLocalAutoCores() = default;

    /**
     * @brief Compute a schedule for the given BspInstance.
     *
     * This method computes a schedule for the given BspInstance using a greedy algorithm.
     *
     * @param instance The BspInstance object representing the instance to compute the schedule for.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;


    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "GreedyBspGrowLocalAutoCores" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override {

        return "GreedyBspGrowLocalAutoCores";
    }

};
