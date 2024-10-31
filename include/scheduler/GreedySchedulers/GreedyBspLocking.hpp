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
#include <cmath>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/heap/fibonacci_heap.hpp>

#include "auxiliary/auxiliary.hpp"
#include "model/BspSchedule.hpp"
#include "scheduler/Scheduler.hpp"

/**
 * @brief The GreedyBspLocking class represents a scheduler that uses a greedy algorithm to compute schedules for
 * BspInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "BspGreedy" in this case.
 */

class GreedyBspLocking : public Scheduler {

  private:
    struct heap_node {

        VertexType node;

        int score;
        int secondary_score;

        heap_node() : node(0), score(0), secondary_score(0) {}
        heap_node(VertexType node, int score, int secondary_score)
            : node(node), score(score), secondary_score(secondary_score) {}

        bool operator<(heap_node const &rhs) const {
            return (score < rhs.score) || (score == rhs.score and secondary_score < rhs.secondary_score) ||
                   (score == rhs.score and secondary_score == rhs.secondary_score and node > rhs.node);
        }
    };

    std::vector<int> get_longest_path(const ComputationalDag &graph) const;

    std::vector<boost::heap::fibonacci_heap<heap_node>> max_proc_score_heap;
    std::vector<boost::heap::fibonacci_heap<heap_node>> max_all_proc_score_heap;

    using heap_handle = typename boost::heap::fibonacci_heap<heap_node>::handle_type;

    std::vector<std::unordered_map<VertexType, heap_handle>> node_proc_heap_handles;
    std::vector<std::unordered_map<VertexType, heap_handle>> node_all_proc_heap_handles;

    std::deque<int> locked_set;
    std::vector<int> locked;
    int lock_penalty = 1;
    std::vector<int> ready_phase;

    std::vector<int> default_value;

    float max_percent_idle_processors;
    bool increase_parallelism_in_new_superstep;
    bool use_memory_constraint = false;
    std::vector<int> current_proc_persistent_memory;
    std::vector<int> current_proc_transient_memory;

    std::pair<int, double> computeScore(VertexType node, unsigned proc,
                                        const BspInstance &instance);

    bool check_mem_feasibility(const BspInstance &instance, const std::set<VertexType> &allReady,
                               const std::vector<std::set<VertexType>> &procReady) const;

    bool Choose(const BspInstance &instance, std::set<VertexType> &allReady, std::vector<std::set<VertexType>> &procReady,
                const std::vector<bool> &procFree, VertexType &node, unsigned &p, const bool endSupStep,
                const size_t remaining_time);

    bool CanChooseNode(const BspInstance &instance, const std::set<VertexType> &allReady,
                       const std::vector<std::set<VertexType>> &procReady, const std::vector<bool> &procFree) const;

  public:
    /**
     * @brief Default constructor for GreedyBspLocking.
     */
    GreedyBspLocking(float max_percent_idle_processors_ = 0.4, bool increase_parallelism_in_new_superstep_ = true)
        : Scheduler(), max_percent_idle_processors(max_percent_idle_processors_),
          increase_parallelism_in_new_superstep(increase_parallelism_in_new_superstep_) {}

    /**
     * @brief Default destructor for GreedyBspLocking.
     */
    virtual ~GreedyBspLocking() = default;

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
            return "BspGreedyLockingMemory";
        } else {
            return "BspGreedyLocking_" + std::to_string(max_percent_idle_processors);
        }
    }

    void set_max_percent_idle_processors(float max_percent_idle_processors_) {
        max_percent_idle_processors = max_percent_idle_processors_;
    }

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override {
        use_memory_constraint = use_memory_constraint_;
    }
};