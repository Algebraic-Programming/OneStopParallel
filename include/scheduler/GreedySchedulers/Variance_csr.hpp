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

#include <algorithm>
#include <chrono>
#include <climits>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/topological_sort.hpp>

#include <boost/graph/filtered_graph.hpp>

#include "scheduler/Scheduler.hpp"
#include "auxiliary/auxiliary.hpp"
#include "model/BspSchedule.hpp"
#include "model/BspInstance_csr.hpp"

#include "boost_extensions/transitive_edge_reduction.hpp"

// #include <boost/heap/fibonacci_heap.hpp>
// #include <boost/heap/binomial_heap.hpp>
// #include <boost/heap/d_ary_heap.hpp>
// #include <boost/heap/fibonacci_heap.hpp>
// #include <boost/heap/heap_concepts.hpp>
// #include <boost/heap/heap_merge.hpp>
// #include <boost/heap/pairing_heap.hpp>
// #include <boost/heap/policies.hpp>
// #include <boost/heap/priority_queue.hpp>
// #include <boost/heap/skew_heap.hpp>

/**
 * @brief The GreedyVarianceFillupScheduler class represents a scheduler that uses a greedy algorithm to compute
 * schedules for BspInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "BspGreedy" in this case.
 */
class Variance_csr : public Scheduler {

  private:


    // class var_dfs_visitor : public boost::default_dfs_visitor {

    //   public:
    //     std::vector<double> work_var;
        
    //     void discover_vertex(VertexType v, const GraphType &g) const {
    //         double temp = 0;
    //         double max_priority = 0;

    //         for (const auto &child : boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, g))) {
    //             max_priority = std::max(work_var[child], max_priority);
    //         }

    //         for (const auto &child : boost::extensions::make_source_iterator_range(boost::adjacent_vertices(v, g))) {
    //             temp += std::exp(2 * (work_var[child] - max_priority));
    //         }

    //         temp = std::log(temp) / 2 + max_priority;

    //         double node_weight = std::log((double)g[v].workWeight);
    //         double larger_val = node_weight > temp ? node_weight : temp;

    //         work_var[v] = std::log(std::exp(node_weight - larger_val) + std::exp(temp - larger_val)) + larger_val;
    //         return;
    //     }
    // };

    struct heap_node {

        VertexType node;

        double score;

        heap_node() : node(0), score(0) {}
        heap_node(VertexType node, double score) : node(node), score(score) {}

        bool operator<(heap_node const &rhs) const {
            return (score < rhs.score) || (score == rhs.score and node > rhs.node);
        }
    };

    // using heap = typename boost::heap::priority_queue<heap_node>;

    std::vector<std::vector<heap_node>> procReady;
    std::vector<heap_node> allReady;

    // std::vector<boost::heap::fibonacci_heap<heap_node>> procReady;
    // boost::heap::fibonacci_heap<heap_node> allReady;

    std::unordered_set<VertexType> ready;

    float max_percent_idle_processors;

    std::vector<double> compute_work_variance(const ComputationalDag &graph) const;
    std::vector<double> compute_work_variance_csr(const csr_graph &graph) const;

    void Choose(const BspInstance &instance, const std::vector<double> &work_variance,
                const std::vector<bool> &procFree, VertexType &node, unsigned &p, const bool endSupStep,
                const size_t remaining_time);

void Choose_csr(unsigned num_p, const csr_graph& G, const std::vector<double> &work_variance,
                          const std::vector<bool> &procFree, VertexType &node, unsigned &p, const bool endSupStep,
                          const size_t remaining_time);

    inline bool CanChooseNode(unsigned p, const std::vector<bool> &procFree) {
        if (!allReady.empty())
            for (unsigned i = 0; i < p; ++i)
                if (procFree[i])
                    return true;

        for (unsigned i = 0; i < p; ++i)
            if (procFree[i] && !procReady[i].empty())
                return true;

        return false;
    }

  public:
    /**
     * @brief Default constructor for GreedyVarianceFillupScheduler.
     */
    Variance_csr(float max_percent_idle_processors_ = 0.2)
        : Scheduler(), max_percent_idle_processors(max_percent_idle_processors_) {}

    /**
     * @brief Default destructor for GreedyVarianceFillupScheduler.
     */
    virtual ~Variance_csr() = default;

    /**
     * @brief Compute a schedule for the given BspInstance.
     *
     * This method computes a schedule for the given BspInstance using a greedy algorithm.
     *
     * @param instance The BspInstance object representing the instance to compute the schedule for.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    virtual std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    //std::pair<RETURN_STATUS, BspSchedule> 
    void computeSchedule(const BspInstance_csr &instance);

    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "BspGreedy" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "Variance_csr"; }
};
