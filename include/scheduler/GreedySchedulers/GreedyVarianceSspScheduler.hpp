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
#include <algorithm>
#include <stdexcept>
#include <queue>

#include "scheduler/SSPScheduler.hpp"
#include "auxiliary/auxiliary.hpp"
#include "model/BspSchedule.hpp"

/**
 * @brief The GreedyVarianceSspScheduler class represents a scheduler that uses a greedy algorithm to compute schedules for BspInstance.
 * 
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "BspGreedy" in this case.
 */
class GreedyVarianceSspScheduler : public SSPScheduler {

  private:

    float max_percent_idle_processors;
    bool increase_parallelism_in_new_superstep;
    bool use_memory_constraint = false;
    std::vector<int> current_proc_persistent_memory;
    std::vector<int> current_proc_transient_memory;

    std::vector<double> compute_work_variance(const ComputationalDag& graph) const;

    struct VarianceCompare
    {
        bool operator()(const std::pair<VertexType, double>& lhs, const std::pair<VertexType, double>& rhs) const {
            return ((lhs.second > rhs.second) || ((lhs.second == rhs.second) && (lhs.first < rhs.first)));
        }
    };

   bool check_mem_feasibility(const BspInstance &instance, const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
                                               const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady) const;

    void Choose(const BspInstance &instance, const std::vector<double> &work_variance,
                std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady, std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
                const std::vector<bool> &procFree, VertexType &node, unsigned &p,
                const bool endSupStep, const size_t remaining_time) const;


    bool CanChooseNode(const BspInstance &instance, const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
                       const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady, const std::vector<bool> &procFree) const;

    unsigned get_nr_parallelizable_nodes(const BspInstance &instance,
                                         const unsigned &stale,
                                         const std::vector<unsigned>& nr_old_ready_nodes_per_type,
                                         const std::vector<unsigned>& nr_ready_nodes_per_type,
                                         const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
                                         const std::vector<unsigned>& nr_procs_per_type) const;

  public:
    /**
     * @brief Default constructor for GreedyVarianceSspScheduler.
     */
    GreedyVarianceSspScheduler(float max_percent_idle_processors_ = 0.2, bool increase_parallelism_in_new_superstep_ = true) : SSPScheduler(), max_percent_idle_processors(max_percent_idle_processors_), increase_parallelism_in_new_superstep(increase_parallelism_in_new_superstep_) {}

    /**
     * @brief Default destructor for GreedyVarianceSspScheduler.
     */
    virtual ~GreedyVarianceSspScheduler() = default;

    /**
     * @brief Compute a SSP schedule for the given BspInstance.
     * 
     * This method computes a schedule for the given BspInstance using a greedy algorithm.
     * 
     * @param instance The BspInstance object representing the instance to compute the schedule for.
     * @return A pair containing the return status and the computed SspSchedule.
     */
    virtual std::pair<RETURN_STATUS, SspSchedule> computeSspSchedule(const BspInstance &instance, unsigned stale) override;

    /**
     * @brief Get the name of the schedule.
     * 
     * This method returns the name of the schedule, which is "BspGreedy" in this case.
     * 
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override {

        if (use_memory_constraint) {
            return "VarianceGreedySspMemory";
        } else {
            return "VarianceGreedySsp";
        }
    }

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override { use_memory_constraint = use_memory_constraint_; }


};
