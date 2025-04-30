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
#include <vector>

#include "MemoryConstraintModules.hpp"
#include "auxiliary/misc.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "graph_algorithms/directed_graph_top_sort.hpp"
#include "model/BspSchedule.hpp"

namespace osp {

/**
 * @brief The VarianceFillup class represents a scheduler that uses a greedy algorithm to compute
 * schedules for BspInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "BspGreedy" in this case.
 */
template<typename Graph_t, typename MemoryConstraint_t = no_memory_constraint>
class VarianceFillup : public Scheduler<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "VarianceFillup can only be used with computational DAGs.");

  private:
    using VertexType = vertex_idx_t<Graph_t>;

    constexpr static bool use_memory_constraint =
        is_memory_constraint_v<MemoryConstraint_t> or is_memory_constraint_schedule_v<MemoryConstraint_t>;

    static_assert(not use_memory_constraint or std::is_same_v<Graph_t, typename MemoryConstraint_t::Graph_impl_t>,
                  "Graph_t must be the same as MemoryConstraint_t::Graph_impl_t.");

    MemoryConstraint_t memory_constraint;

    float max_percent_idle_processors;
    bool increase_parallelism_in_new_superstep;

    std::vector<double> compute_work_variance(const ComputationalDag &graph) const {

        std::vector<double> work_variance(graph.numberOfVertices(), 0.0);

        const std::vector<VertexType> top_order = GetTopOrder(AS_IT_COMES, graph);

        for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
            double temp = 0;
            double max_priority = 0;
            for (const auto &child : graph.children(*r_iter)) {
                max_priority = std::max(work_variance[child], max_priority);
            }
            for (const auto &child : graph.children(*r_iter)) {
                temp += std::exp(2 * (work_variance[child] - max_priority));
            }
            temp = std::log(temp) / 2 + max_priority;

            double node_weight = std::log((double)std::max(graph.vertex_work_weight(*r_iter), 1));
            double larger_val = node_weight > temp ? node_weight : temp;

            work_variance[*r_iter] =
                std::log(std::exp(node_weight - larger_val) + std::exp(temp - larger_val)) + larger_val;
        }

        return work_variance;
    }

    std::vector<std::vector<std::vector<unsigned>>>
    procTypesCompatibleWithNodeType_omit_procType(const BspInstance &instance) const;

    struct VarianceCompare {
        bool operator()(const std::pair<VertexType, double> &lhs, const std::pair<VertexType, double> &rhs) const {
            return ((lhs.second > rhs.second) || ((lhs.second == rhs.second) && (lhs.first < rhs.first)));
        }
    };

    bool
    check_mem_feasibility(const BspInstance &instance,
                          const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
                          const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady) const;

    void
    Choose(const BspInstance &instance, const std::vector<double> &work_variance,
           std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
           std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
           const std::vector<bool> &procFree, VertexType &node, unsigned &p, const bool endSupStep,
           const size_t remaining_time, const std::vector<double> &work_variances,
           const std::vector<std::vector<std::vector<unsigned>>> &procTypesCompatibleWithNodeType_skip_proctype) const {

        double maxScore = -1;
        bool found_allocation = false;
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (procFree[i] && !procReady[i].empty()) {
                // select node
                for (auto node_pair_it = procReady[i].begin(); node_pair_it != procReady[i].end();) {
                    if (endSupStep &&
                        (remaining_time < instance.getComputationalDag().nodeWorkWeight(node_pair_it->first))) {
                        node_pair_it = procReady[i].erase(node_pair_it);
                        continue;
                    }

                    const double &score = node_pair_it->second;

                    if (score > maxScore) {
                        maxScore = score;
                        node = node_pair_it->first;
                        p = i;

                        procReady[i].erase(node_pair_it);
                        return;
                    }
                    node_pair_it++;
                }
            }
        }

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (procFree[i] && !allReady[instance.getArchitecture().processorType(i)].empty()) {
                // select node
                for (auto it = allReady[instance.getArchitecture().processorType(i)].begin();
                     it != allReady[instance.getArchitecture().processorType(i)].end();) {
                    if (endSupStep && (remaining_time < instance.getComputationalDag().nodeWorkWeight(it->first))) {
                        it = allReady[instance.getArchitecture().processorType(i)].erase(it);
                        continue;
                    }

                    const double &score = it->second;

                    if (score > maxScore) {
                        if (use_memory_constraint) {

                            if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                                if (current_proc_persistent_memory[i] +
                                        instance.getComputationalDag().nodeMemoryWeight(it->first) <=
                                    instance.getArchitecture().memoryBound(i)) {

                                    node = it->first;
                                    p = i;

                                    allReady[instance.getArchitecture().processorType(i)].erase(it);
                                    for (unsigned procType : procTypesCompatibleWithNodeType_skip_proctype
                                             [instance.getArchitecture().processorType(i)]
                                             [instance.getComputationalDag().nodeType(node)]) {
                                        allReady[procType].erase(std::make_pair(node, work_variances[node]));
                                    }
                                    return;
                                }

                            } else if (instance.getArchitecture().getMemoryConstraintType() ==
                                       PERSISTENT_AND_TRANSIENT) {
                                if (current_proc_persistent_memory[i] +
                                        instance.getComputationalDag().nodeMemoryWeight(it->first) +
                                        std::max(current_proc_transient_memory[i],
                                                 instance.getComputationalDag().nodeCommunicationWeight(it->first)) <=
                                    instance.getArchitecture().memoryBound(i)) {

                                    node = it->first;
                                    p = i;

                                    allReady[instance.getArchitecture().processorType(i)].erase(it);
                                    for (unsigned procType : procTypesCompatibleWithNodeType_skip_proctype
                                             [instance.getArchitecture().processorType(i)]
                                             [instance.getComputationalDag().nodeType(node)]) {
                                        allReady[procType].erase(std::make_pair(node, work_variances[node]));
                                    }
                                    return;
                                }
                            }

                        } else {
                            node = it->first;
                            p = i;

                            allReady[instance.getArchitecture().processorType(i)].erase(it);
                            for (unsigned procType :
                                 procTypesCompatibleWithNodeType_skip_proctype[instance.getArchitecture().processorType(
                                     i)][instance.getComputationalDag().nodeType(node)]) {
                                allReady[procType].erase(std::make_pair(node, work_variances[node]));
                            }
                            return;
                        }
                    }
                    it++;
                }
            }
        }
    }

    bool CanChooseNode(const BspInstance &instance,
                       const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
                       const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
                       const std::vector<bool> &procFree) const {
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
            if (procFree[i] && !procReady[i].empty())
                return true;

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
            if (procFree[i] && !allReady[instance.getArchitecture().processorType(i)].empty())
                return true;

        return false;
    };

    unsigned get_nr_parallelizable_nodes(const BspInstance &instance,
                                         const std::vector<unsigned> &nr_ready_nodes_per_type,
                                         const std::vector<unsigned> &nr_procs_per_type) const;

  public:
    /**
     * @brief Default constructor for VarianceFillup.
     */
    VarianceFillup(float max_percent_idle_processors_ = 0.2f, bool increase_parallelism_in_new_superstep_ = true)
        : Scheduler(), max_percent_idle_processors(max_percent_idle_processors_),
          increase_parallelism_in_new_superstep(increase_parallelism_in_new_superstep_) {}

    /**
     * @brief Default destructor for VarianceFillup.
     */
    virtual ~VarianceFillup() = default;

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
            return "VarianceGreedyFillupMemory";
        } else {
            return "VarianceGreedyFillup";
        }
    }

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override {
        use_memory_constraint = use_memory_constraint_;
    }
};

} // namespace osp