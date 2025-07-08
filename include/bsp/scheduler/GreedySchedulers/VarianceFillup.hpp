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
#include "bsp/model/BspSchedule.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "graph_algorithms/directed_graph_top_sort.hpp"

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

    double max_percent_idle_processors;
    bool increase_parallelism_in_new_superstep;

    std::vector<double> compute_work_variance(const Graph_t &graph) const {

        std::vector<double> work_variance(graph.num_vertices(), 0.0);

        const std::vector<VertexType> top_order = GetTopOrder(graph);

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

            double node_weight = std::log((double)std::max(graph.vertex_work_weight(*r_iter), static_cast<v_workw_t<Graph_t>>(1)));
            double larger_val = node_weight > temp ? node_weight : temp;

            work_variance[*r_iter] =
                std::log(std::exp(node_weight - larger_val) + std::exp(temp - larger_val)) + larger_val;
        }

        return work_variance;
    }

    std::vector<std::vector<std::vector<unsigned>>>
    procTypesCompatibleWithNodeType_omit_procType(const BspInstance<Graph_t> &instance) const {

        const std::vector<std::vector<unsigned>> procTypesCompatibleWithNodeType =
            instance.getProcTypesCompatibleWithNodeType();

        std::vector<std::vector<std::vector<unsigned>>> procTypesCompatibleWithNodeType_skip(
            instance.getArchitecture().getNumberOfProcessorTypes(),
            std::vector<std::vector<unsigned>>(instance.getComputationalDag().num_vertex_types()));
        for (unsigned procType = 0; procType < instance.getArchitecture().getNumberOfProcessorTypes(); procType++) {
            for (unsigned nodeType = 0; nodeType < instance.getComputationalDag().num_vertex_types(); nodeType++) {
                for (unsigned otherProcType : procTypesCompatibleWithNodeType[nodeType]) {
                    if (procType == otherProcType)
                        continue;
                    procTypesCompatibleWithNodeType_skip[procType][nodeType].emplace_back(otherProcType);
                }
            }
        }

        return procTypesCompatibleWithNodeType_skip;
    }

    struct VarianceCompare {
        bool operator()(const std::pair<VertexType, double> &lhs, const std::pair<VertexType, double> &rhs) const {
            return ((lhs.second > rhs.second) || ((lhs.second == rhs.second) && (lhs.first < rhs.first)));
        }
    };

    bool check_mem_feasibility(
        const BspInstance<Graph_t> &instance,
        const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
        const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady) const {

        if constexpr (use_memory_constraint) {
            if (instance.getArchitecture().getMemoryConstraintType() == MEMORY_CONSTRAINT_TYPE::PERSISTENT_AND_TRANSIENT) {

                for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
                    if (!procReady[i].empty()) {

                        const std::pair<VertexType, double> &node_pair = *procReady[i].begin();
                        VertexType top_node = node_pair.first;

                        if (memory_constraint.can_add(top_node, i)) {
                            return true;
                        }
                    }
                }

                for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {

                    if (allReady[instance.getArchitecture().processorType(i)].empty())
                        continue;

                    const std::pair<VertexType, double> &node_pair =
                        *allReady[instance.getArchitecture().processorType(i)].begin();
                    VertexType top_node = node_pair.first;

                    if (memory_constraint.can_add(top_node, i)) {
                        return true;
                    }
                }

                return false;
            }
        }

        return true;
    };

    void
    Choose(const BspInstance<Graph_t> &instance, const std::vector<double> &work_variance,
           std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
           std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
           const std::vector<bool> &procFree, VertexType &node, unsigned &p, const bool endSupStep,
           const v_workw_t<Graph_t> remaining_time,
           const std::vector<std::vector<std::vector<unsigned>>> &procTypesCompatibleWithNodeType_skip_proctype) const {

        double maxScore = -1;
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (procFree[i] && !procReady[i].empty()) {
                // select node
                for (auto node_pair_it = procReady[i].begin(); node_pair_it != procReady[i].end();) {
                    if (endSupStep &&
                        (remaining_time < instance.getComputationalDag().vertex_work_weight(node_pair_it->first))) {
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
                    if (endSupStep && (remaining_time < instance.getComputationalDag().vertex_work_weight(it->first))) {
                        it = allReady[instance.getArchitecture().processorType(i)].erase(it);
                        continue;
                    }

                    const double &score = it->second;

                    if (score > maxScore) {

                        if constexpr (use_memory_constraint) {

                            if (memory_constraint.can_add(it->first, i)) {

                                node = it->first;
                                p = i;

                                allReady[instance.getArchitecture().processorType(i)].erase(it);
                                for (unsigned procType : procTypesCompatibleWithNodeType_skip_proctype
                                         [instance.getArchitecture().processorType(i)]
                                         [instance.getComputationalDag().vertex_type(node)]) {
                                    allReady[procType].erase(std::make_pair(node, work_variance[node]));
                                }
                                return;
                            }
                        } else {

                            node = it->first;
                            p = i;

                            allReady[instance.getArchitecture().processorType(i)].erase(it);
                            for (unsigned procType :
                                 procTypesCompatibleWithNodeType_skip_proctype[instance.getArchitecture().processorType(
                                     i)][instance.getComputationalDag().vertex_type(node)]) {
                                allReady[procType].erase(std::make_pair(node, work_variance[node]));
                            }
                            return;
                        }
                    }
                    it++;
                }
            }
        }
    }

    bool CanChooseNode(const BspInstance<Graph_t> &instance,
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
    }

    unsigned get_nr_parallelizable_nodes(const BspInstance<Graph_t> &instance,
                                         const std::vector<unsigned> &nr_ready_nodes_per_type,
                                         const std::vector<unsigned> &nr_procs_per_type) const {
        unsigned nr_nodes = 0;

        std::vector<unsigned> ready_nodes_per_type = nr_ready_nodes_per_type;
        std::vector<unsigned> procs_per_type = nr_procs_per_type;
        for (unsigned proc_type = 0; proc_type < instance.getArchitecture().getNumberOfProcessorTypes(); ++proc_type)
            for (unsigned node_type = 0; node_type < instance.getComputationalDag().num_vertex_types(); ++node_type)
                if (instance.isCompatibleType(node_type, proc_type)) {
                    unsigned matched = std::min(ready_nodes_per_type[node_type], procs_per_type[proc_type]);
                    nr_nodes += matched;
                    ready_nodes_per_type[node_type] -= matched;
                    procs_per_type[proc_type] -= matched;
                }

        return nr_nodes;
    }

  public:
    /**
     * @brief Default constructor for VarianceFillup.
     */
    VarianceFillup(float max_percent_idle_processors_ = 0.2f, bool increase_parallelism_in_new_superstep_ = true)
        : max_percent_idle_processors(max_percent_idle_processors_),
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
    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {

        const auto &instance = schedule.getInstance();

        for (const auto &v : instance.getComputationalDag().vertices()) {
            schedule.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
        }

        unsigned supstepIdx = 0;

        if constexpr (is_memory_constraint_v<MemoryConstraint_t>) {
            memory_constraint.initialize(instance);
        } else if constexpr (is_memory_constraint_schedule_v<MemoryConstraint_t>) {
            memory_constraint.initialize(schedule, supstepIdx);
        }

        const auto &N = instance.numberOfVertices();
        const unsigned &params_p = instance.numberOfProcessors();
        const auto &G = instance.getComputationalDag();

        const std::vector<double> work_variances = compute_work_variance(G);

        std::set<std::pair<VertexType, double>, VarianceCompare> ready;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReady(params_p);
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> allReady(
            instance.getArchitecture().getNumberOfProcessorTypes());

        const std::vector<std::vector<unsigned>> procTypesCompatibleWithNodeType =
            instance.getProcTypesCompatibleWithNodeType();
        const std::vector<std::vector<std::vector<unsigned>>> procTypesCompatibleWithNodeType_skip_proctype =
            procTypesCompatibleWithNodeType_omit_procType(instance);

        std::vector<unsigned> nr_ready_nodes_per_type(G.num_vertex_types(), 0);
        std::vector<unsigned> nr_procs_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
        for (unsigned proc = 0; proc < params_p; ++proc)
            ++nr_procs_per_type[instance.getArchitecture().processorType(proc)];

        std::vector<VertexType> nrPredecRemain(N);
        for (VertexType node = 0; node < N; node++) {
            const auto num_parents = G.in_degree(node);
            nrPredecRemain[node] = num_parents;
            if (num_parents == 0) {
                ready.insert(std::make_pair(node, work_variances[node]));
                ++nr_ready_nodes_per_type[G.vertex_type(node)];
                for (unsigned procType : procTypesCompatibleWithNodeType[G.vertex_type(node)])
                    allReady[procType].insert(std::make_pair(node, work_variances[node]));
            }
        }

        std::vector<bool> procFree(params_p, true);
        unsigned free = params_p;

        std::set<std::pair<v_workw_t<Graph_t>, VertexType>> finishTimes;
        finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

        bool endSupStep = false;
        while (!ready.empty() || !finishTimes.empty()) {
            if (finishTimes.empty() && endSupStep) {
                for (unsigned i = 0; i < params_p; ++i) {
                    procReady[i].clear();

                    if constexpr (use_memory_constraint) {
                        memory_constraint.reset(i);
                    }
                }

                for (unsigned procType = 0; procType < instance.getArchitecture().getNumberOfProcessorTypes();
                     ++procType)
                    allReady[procType].clear();

                for (const auto &nodeAndValuePair : ready) {
                    const auto node = nodeAndValuePair.first;
                    for (unsigned procType : procTypesCompatibleWithNodeType[G.vertex_type(node)])
                        allReady[procType].insert(allReady[procType].end(), nodeAndValuePair);
                }

                ++supstepIdx;
                endSupStep = false;

                finishTimes.emplace(0, std::numeric_limits<VertexType>::max());
            }

            const v_workw_t<Graph_t> time = finishTimes.begin()->first;
            const v_workw_t<Graph_t> max_finish_time = finishTimes.rbegin()->first;

            // Find new ready jobs
            while (!finishTimes.empty() && finishTimes.begin()->first == time) {
                const VertexType node = finishTimes.begin()->second;
                finishTimes.erase(finishTimes.begin());
                if (node != std::numeric_limits<VertexType>::max()) {
                    for (const auto &succ : G.children(node)) {
                        nrPredecRemain[succ]--;
                        if (nrPredecRemain[succ] == 0) {
                            ready.emplace(succ, work_variances[succ]);
                            ++nr_ready_nodes_per_type[G.vertex_type(succ)];

                            bool canAdd = true;
                            for (const auto &pred : G.parents(succ)) {
                                if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) &&
                                    schedule.assignedSuperstep(pred) == supstepIdx)
                                    canAdd = false;
                            }

                            if constexpr (use_memory_constraint) {

                                if (canAdd) {
                                    if (not memory_constraint.can_add(succ, schedule.assignedProcessor(node)))
                                        canAdd = false;
                                }
                            }

                            if (!instance.isCompatible(succ, schedule.assignedProcessor(node)))
                                canAdd = false;

                            if (canAdd) {
                                procReady[schedule.assignedProcessor(node)].emplace(succ, work_variances[succ]);
                            }
                        }
                    }
                    procFree[schedule.assignedProcessor(node)] = true;
                    ++free;
                }
            }

            // Assign new jobs to processors
            if (!CanChooseNode(instance, allReady, procReady, procFree)) {
                endSupStep = true;
            }
            while (CanChooseNode(instance, allReady, procReady, procFree)) {

                VertexType nextNode = std::numeric_limits<VertexType>::max();
                unsigned nextProc = params_p;
                Choose(instance, work_variances, allReady, procReady, procFree, nextNode, nextProc, endSupStep,
                       max_finish_time - time, procTypesCompatibleWithNodeType_skip_proctype);

                if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == params_p) {
                    endSupStep = true;
                    break;
                }

                ready.erase(std::make_pair(nextNode, work_variances[nextNode]));
                --nr_ready_nodes_per_type[G.vertex_type(nextNode)];
                schedule.setAssignedProcessor(nextNode, nextProc);
                schedule.setAssignedSuperstep(nextNode, supstepIdx);

                if constexpr (use_memory_constraint) {
                    memory_constraint.add(nextNode, nextProc);

                    std::vector<std::pair<VertexType, double>> toErase;

                    for (const auto &node_pair : procReady[nextProc]) {
                        if (not memory_constraint.can_add(node_pair.first, nextProc)) {
                            toErase.push_back(node_pair);
                        }
                    }

                    for (const auto &node : toErase) {
                        procReady[nextProc].erase(node);
                    }
                }

                finishTimes.emplace(time + G.vertex_work_weight(nextNode), nextNode);
                procFree[nextProc] = false;
                --free;
            }

            if constexpr (use_memory_constraint) {

                if (not check_mem_feasibility(instance, allReady, procReady)) {

                    return RETURN_STATUS::ERROR;
                }
            }

            if (free > params_p * max_percent_idle_processors &&
                ((!increase_parallelism_in_new_superstep) ||
                 get_nr_parallelizable_nodes(instance, nr_ready_nodes_per_type, nr_procs_per_type) >=
                     std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
                              params_p - free + ((unsigned)(0.5 * free)))))
                endSupStep = true;
        }

        assert(schedule.satisfiesPrecedenceConstraints());

        return RETURN_STATUS::OSP_SUCCESS;
    }

    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "BspGreedy" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override {

        if constexpr (use_memory_constraint) {
            return "VarianceGreedyFillupMemory";
        } else {
            return "VarianceGreedyFillup";
        }
    }
};

} // namespace osp