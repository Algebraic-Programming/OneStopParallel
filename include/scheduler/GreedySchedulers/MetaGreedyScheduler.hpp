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

#include "GreedyBspScheduler.hpp"
#include "GreedyCilkScheduler.hpp"
#include "GreedyEtfScheduler.hpp"
#include "GreedyLayers.hpp"
#include "scheduler/GreedySchedulers/GreedyBspLocking.hpp"
#include "scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceFillupScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyBspFillupScheduler.hpp"
#include "scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "scheduler/Scheduler.hpp"

enum GREEDY_COST_FUNCTION { BSP, SUPERSTEPS };

inline std::string to_string(const GREEDY_COST_FUNCTION status) {
    switch (status) {
    case BSP:
        return "BSP";
    case SUPERSTEPS:
        return "SUPERSTEPS";
    default:
        return "DEFAULT";
    }
}


/**
 * @class MetaGreedyScheduler
 * @brief A class that represents a meta greedy scheduler.
 *
 * The MetaGreedyScheduler class is a subclass of the Scheduler class. It implements the computeSchedule,
 * runGreedyMode, and getScheduleName methods. It provides a way to compute a schedule using a meta greedy
 * algorithm and retrieve the name of the schedule.
 */
class MetaGreedyScheduler : public Scheduler {

  private:
    GREEDY_COST_FUNCTION cost_function = SUPERSTEPS;

  public:
    /**
     * @brief Constructs a MetaGreedyScheduler object.
     */
    MetaGreedyScheduler() : Scheduler() {}

    /**
     * @brief Destroys the MetaGreedyScheduler object.
     */
    virtual ~MetaGreedyScheduler() = default;

    /**
     * @brief Computes a schedule using a meta greedy algorithm.
     * @param instance The BspInstance object representing the instance.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    /**
     * @brief Runs the greedy mode algorithm.
     * @param instance The BspInstance object representing the instance.
     * @param mode The mode to run the algorithm in.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> runGreedyMode(const BspInstance &instance, const std::string &mode);

    virtual void set_cost_function(GREEDY_COST_FUNCTION cost_function) { this->cost_function = cost_function; }

    /**
     * @brief Gets the name of the schedule.
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "BestGreedy_" + cost_function; }
};
