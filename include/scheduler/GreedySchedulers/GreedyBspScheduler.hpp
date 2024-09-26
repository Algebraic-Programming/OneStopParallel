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

#include <boost/heap/fibonacci_heap.hpp>

#include "scheduler/Scheduler.hpp"
#include "auxiliary/auxiliary.hpp"
#include "model/BspSchedule.hpp"




/**
 * @brief The GreedyBspScheduler class represents a scheduler that uses a greedy algorithm to compute schedules for
 * BspInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "BspGreedy" in this case.
 */
class GreedyBspScheduler : public Scheduler {

  private:
    struct heap_node {

        VertexType node;

        double score;

        heap_node() : node(0), score(0) {}
        heap_node(VertexType node, double score) : node(node), score(score) {}

        bool operator<(heap_node const &rhs) const {
            return (score < rhs.score) || (score == rhs.score and node < rhs.node);
        }
    };

    std::vector<boost::heap::fibonacci_heap<heap_node>> max_proc_score_heap;
    std::vector<boost::heap::fibonacci_heap<heap_node>> max_all_proc_score_heap;

    using heap_handle = typename boost::heap::fibonacci_heap<heap_node>::handle_type;

    std::vector<std::unordered_map<VertexType, heap_handle>> node_proc_heap_handles;
    std::vector<std::unordered_map<VertexType, heap_handle>> node_all_proc_heap_handles;

    float max_percent_idle_processors;
    bool increase_parallelism_in_new_superstep;
    bool use_memory_constraint = false;

    std::vector<int> current_proc_persistent_memory;
    std::vector<int> current_proc_transient_memory;



    double computeScore(VertexType node, unsigned proc, const std::vector<std::vector<bool>> &procInHyperedge,
                        const BspInstance &instance);


    void Choose(const BspInstance &instance, const std::vector<std::vector<bool>> &procInHyperedge,
                    const std::set<VertexType> &allReady, const std::vector<std::set<VertexType>> &procReady,
                    const std::vector<bool> &procFree, VertexType &node, unsigned &p) const;

    bool CanChooseNode(const BspInstance &instance, const std::set<VertexType> &allReady,
                           const std::vector<std::set<VertexType>> &procReady, const std::vector<bool> &procFree) const;

    bool check_mem_feasibility(const BspInstance &instance, const std::set<VertexType> &allReady,
                                               const std::vector<std::set<VertexType>> &procReady) const;


  public:
    /**
     * @brief Default constructor for GreedyBspScheduler.
     */
    GreedyBspScheduler(float max_percent_idle_processors_ = 0.2, bool increase_parallelism_in_new_superstep_ = true)
        : Scheduler(), max_percent_idle_processors(max_percent_idle_processors_), increase_parallelism_in_new_superstep(increase_parallelism_in_new_superstep_) {}

    /**
     * @brief Default destructor for GreedyBspScheduler.
     */
    virtual ~GreedyBspScheduler() = default;

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
     * This method returns the name of the schedule, which is "BspGreedy" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override {

        if (use_memory_constraint) {
            return "BspGreedyMemory";
        } else {
            return "BspGreedy";
        }
    }

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override {
        use_memory_constraint = use_memory_constraint_;
    }
};
