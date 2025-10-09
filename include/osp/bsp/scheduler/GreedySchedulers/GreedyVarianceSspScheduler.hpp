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

@author Toni Boehnlein, Christos Matzoros, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
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
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/MaxBspSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

/**
 * @brief The GreedyVarianceSspScheduler class represents a scheduler that uses a greedy algorithm
 * with stale synchronous parallel (SSP) execution model.
 *
 * It computes schedules for BspInstance using variance-based priorities.
 */
template<typename Graph_t, typename MemoryConstraint_t = no_memory_constraint>
class GreedyVarianceSspScheduler : public Scheduler<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "GreedyVarianceSspScheduler can only be used with computational DAGs.");

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

    void Choose(const BspInstance<Graph_t> &instance,
            const std::vector<double> &work_variance,
            std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
            std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
            const std::vector<bool> &procFree,
            VertexType &node, unsigned &p,
            const bool endSupStep,
            const v_workw_t<Graph_t> remaining_time,
            const std::vector<std::vector<std::vector<unsigned>>> &procTypesCompatibleWithNodeType_skip_proctype) const 
    {
        double maxScore = -1;
        bool found_allocation = false;

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (!procFree[i] || procReady[i].empty())
                continue;

            auto it = procReady[i].begin();
            while (it != procReady[i].end()) {
                if (endSupStep &&
                    (remaining_time < instance.getComputationalDag().vertex_work_weight(it->first))) {
                    it = procReady[i].erase(it);
                    continue;
                }

                const double &score = it->second;

                if (score > maxScore) {
                    const unsigned procType = instance.getArchitecture().processorType(i);

                    if constexpr (use_memory_constraint) {
                        if (memory_constraint.can_add(it->first, i)) {
                            node = it->first;
                            p = i;
                            found_allocation = true;

                            procReady[i].erase(it);

                            if (procType < procTypesCompatibleWithNodeType_skip_proctype.size()) {
                                const auto &compatibleTypes =
                                    procTypesCompatibleWithNodeType_skip_proctype[procType]
                                        [instance.getComputationalDag().vertex_type(node)];

                                for (unsigned otherType : compatibleTypes) {
                                    for (unsigned j = 0; j < instance.numberOfProcessors(); ++j) {
                                        if (j != i &&
                                            instance.getArchitecture().processorType(j) == otherType &&
                                            j < procReady.size()) {
                                            procReady[j].erase(std::make_pair(node, work_variance[node]));
                                        }
                                    }
                                }
                            }

                            return;
                        }
                    } else {
                        node = it->first;
                        p = i;
                        found_allocation = true;

                        procReady[i].erase(it);

                        if (procType < procTypesCompatibleWithNodeType_skip_proctype.size()) {
                            const auto &compatibleTypes =
                                procTypesCompatibleWithNodeType_skip_proctype[procType]
                                    [instance.getComputationalDag().vertex_type(node)];

                            for (unsigned otherType : compatibleTypes) {
                                for (unsigned j = 0; j < instance.numberOfProcessors(); ++j) {
                                    if (j != i &&
                                        instance.getArchitecture().processorType(j) == otherType &&
                                        j < procReady.size()) {
                                        procReady[j].erase(std::make_pair(node, work_variance[node]));
                                    }
                                }
                            }
                        }

                        return;
                    }
                }

                ++it;
            }
        }

        if (found_allocation)
            return;

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            const unsigned procType = instance.getArchitecture().processorType(i);
            if (!procFree[i] || procType >= allReady.size() || allReady[procType].empty())
                continue;

            auto &readyList = allReady[procType];
            auto it = readyList.begin();

            while (it != readyList.end()) {
                if (endSupStep &&
                    (remaining_time < instance.getComputationalDag().vertex_work_weight(it->first))) {
                    it = readyList.erase(it);
                    continue;
                }

                const double &score = it->second;

                if (score > maxScore) {
                    if constexpr (use_memory_constraint) {
                        if (memory_constraint.can_add(it->first, i)) {
                            node = it->first;
                            p = i;

                            readyList.erase(it);

                            const auto &compatibleTypes =
                                procTypesCompatibleWithNodeType_skip_proctype[procType]
                                    [instance.getComputationalDag().vertex_type(node)];

                            for (unsigned otherType : compatibleTypes) {
                                if (otherType < allReady.size())
                                    allReady[otherType].erase(std::make_pair(node, work_variance[node]));
                            }

                            return;
                        }
                    } else {
                        node = it->first;
                        p = i;

                        readyList.erase(it);

                        const auto &compatibleTypes =
                            procTypesCompatibleWithNodeType_skip_proctype[procType]
                                [instance.getComputationalDag().vertex_type(node)];

                        for (unsigned otherType : compatibleTypes) {
                            if (otherType < allReady.size())
                                allReady[otherType].erase(std::make_pair(node, work_variance[node]));
                        }

                        return;
                    }
                }

                ++it;
            }
        }
    };


    bool check_mem_feasibility(
        const BspInstance<Graph_t> &instance,
        const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
        const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady) const 
    {
        if constexpr (use_memory_constraint) {
            if (instance.getArchitecture().getMemoryConstraintType() == MEMORY_CONSTRAINT_TYPE::PERSISTENT_AND_TRANSIENT) 
            {
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
    }

    unsigned get_nr_parallelizable_nodes(
        const BspInstance<Graph_t> &instance,
        const unsigned &stale,
        const std::vector<unsigned> &nr_old_ready_nodes_per_type,
        const std::vector<unsigned> &nr_ready_nodes_per_type,
        const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
        const std::vector<unsigned> &nr_procs_per_type) const 
    {
        unsigned nr_nodes = 0;
        unsigned num_proc_types = instance.getArchitecture().getNumberOfProcessorTypes();

        std::vector<unsigned> procs_per_type = nr_procs_per_type;

        if (stale > 1) {
            for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                if (!procReady[proc].empty()) {
                    procs_per_type[instance.getArchitecture().processorType(proc)]--;
                    nr_nodes++;
                }
            }
        }

        std::vector<unsigned> ready_nodes_per_type = nr_ready_nodes_per_type;
        for (unsigned node_type = 0; node_type < ready_nodes_per_type.size(); node_type++) {
            ready_nodes_per_type[node_type] += nr_old_ready_nodes_per_type[node_type];
        }

        for (unsigned proc_type = 0; proc_type < num_proc_types; ++proc_type) {
            for (unsigned node_type = 0; node_type < instance.getComputationalDag().num_vertex_types(); ++node_type) {
                if (instance.isCompatibleType(node_type, proc_type)) {
                    unsigned matched = std::min(ready_nodes_per_type[node_type],
                                                procs_per_type[proc_type]);
                    nr_nodes += matched;
                    ready_nodes_per_type[node_type] -= matched;
                    procs_per_type[proc_type] -= matched;
                }
            }
        }

        return nr_nodes;
    }


    public:

    RETURN_STATUS computeSspSchedule(BspSchedule<Graph_t> &schedule, unsigned stale) {

        const auto &instance = schedule.getInstance();
        const auto &G = instance.getComputationalDag();
        const auto &N = instance.numberOfVertices();
        const unsigned &P = instance.numberOfProcessors();

        for (auto v : G.vertices()) {
            schedule.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
        }
        
        unsigned supstepIdx = 0;

        if constexpr (is_memory_constraint_v<MemoryConstraint_t>) {
            memory_constraint.initialize(instance);
        } else if constexpr (is_memory_constraint_schedule_v<MemoryConstraint_t>) {
            memory_constraint.initialize(schedule, supstepIdx);
        }

        const std::vector<double> work_variances = compute_work_variance(G);

        std::set<std::pair<VertexType, double>, VarianceCompare> old_ready;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> ready(stale);
        std::vector<std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>>> procReady(
            stale, std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>>(P));
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> allReady(
            instance.getArchitecture().getNumberOfProcessorTypes());

        const auto procTypesCompatibleWithNodeType = instance.getProcTypesCompatibleWithNodeType();
        const std::vector<std::vector<std::vector<unsigned>>> procTypesCompatibleWithNodeType_skip_proctype =
            procTypesCompatibleWithNodeType_omit_procType(instance);

        std::vector<unsigned> nr_old_ready_nodes_per_type(G.num_vertex_types(), 0);
        std::vector<std::vector<unsigned>> nr_ready_stale_nodes_per_type(
            stale, std::vector<unsigned>(G.num_vertex_types(), 0));
        std::vector<unsigned> nr_procs_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
        for (auto proc = 0u; proc < P; ++proc) {
            ++nr_procs_per_type[instance.getArchitecture().processorType(proc)];
        }

        std::vector<unsigned> nrPredecRemain(N);
        for (VertexType node = 0; node < static_cast<VertexType>(N); ++node) {
            const auto num_parents = G.in_degree(node);
            nrPredecRemain[node] = static_cast<unsigned>(num_parents);
            if (num_parents == 0) {
                ready[0].insert(std::make_pair(node, work_variances[node]));
                nr_ready_stale_nodes_per_type[0][G.vertex_type(node)]++;
            }
        }

        std::vector<bool> procFree(P, true);
        unsigned free = P;

        std::set<std::pair<v_workw_t<Graph_t>, VertexType>> finishTimes;
        finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

        std::vector<unsigned> number_of_allocated_allReady_tasks_in_superstep(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
        std::vector<unsigned> limit_of_number_of_allocated_allReady_tasks_in_superstep(instance.getArchitecture().getNumberOfProcessorTypes(), 0);

        


        bool endSupStep = true;
        bool begin_outer_while = true;
        bool able_to_schedule_in_step = false;
        unsigned successive_empty_supersteps = 0u;

        auto nonempty_ready = [&]() {
            return std::any_of(ready.cbegin(), ready.cend(),
                             [](const std::set<std::pair<VertexType, double>, VarianceCompare>& ready_set) { return !ready_set.empty(); });
        };

        while (!old_ready.empty() || nonempty_ready() || !finishTimes.empty()) {
            if (finishTimes.empty() && endSupStep) {
                able_to_schedule_in_step = false;
                number_of_allocated_allReady_tasks_in_superstep = std::vector<unsigned>(instance.getArchitecture().getNumberOfProcessorTypes(), 0);

                for (unsigned i = 0; i < P; ++i)
                    procReady[supstepIdx % stale][i].clear();

                if (!begin_outer_while) {
                    supstepIdx++;
                } else {
                    begin_outer_while = false;
                }

                 for (unsigned procType = 0; procType < instance.getArchitecture().getNumberOfProcessorTypes(); ++procType)
                    allReady[procType].clear();

                old_ready.insert(ready[supstepIdx % stale].begin(), ready[supstepIdx % stale].end());
                ready[supstepIdx % stale].clear();
                for (unsigned node_type = 0; node_type < G.num_vertex_types(); ++node_type) {
                    nr_old_ready_nodes_per_type[node_type] += nr_ready_stale_nodes_per_type[supstepIdx % stale][node_type];
                    nr_ready_stale_nodes_per_type[supstepIdx % stale][node_type] = 0;
                }

                for (const auto &nodeAndValuePair : old_ready) {
                    VertexType node = nodeAndValuePair.first;
                    for (unsigned procType : procTypesCompatibleWithNodeType[G.vertex_type(node)]) {
                        allReady[procType].insert(allReady[procType].end(), nodeAndValuePair);
                    }
                }

                if constexpr (use_memory_constraint) {
                    if (instance.getArchitecture().getMemoryConstraintType() == MEMORY_CONSTRAINT_TYPE::LOCAL) {
                        for (unsigned proc = 0; proc < P; proc++)
                            memory_constraint.reset(proc);
                    }
                }

                for (unsigned procType = 0; procType < instance.getArchitecture().getNumberOfProcessorTypes(); procType++) {
                    unsigned equal_split = (static_cast<unsigned int>(allReady[procType].size()) + stale - 1) / stale;
                    unsigned at_least_for_long_step = 3 * nr_procs_per_type[procType];

                    limit_of_number_of_allocated_allReady_tasks_in_superstep[procType] = std::max(at_least_for_long_step, equal_split);
                }

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
                    const unsigned proc_of_node = schedule.assignedProcessor(node);

                    for (const auto& succ : G.children(node)) {
                        nrPredecRemain[succ]--;
                        if (nrPredecRemain[succ] == 0) {
                            ready[supstepIdx % stale].emplace(succ, work_variances[succ]);
                            nr_ready_stale_nodes_per_type[supstepIdx % stale][G.vertex_type(succ)]++;

                            unsigned earliest_add = supstepIdx;
                            for (const auto& pred : G.parents(succ)) {
                                if (schedule.assignedProcessor(pred) != proc_of_node) {
                                    earliest_add = std::max(earliest_add,
                                                            stale + schedule.assignedSuperstep(pred));
                                }
                            }

                            if (instance.isCompatible(succ, proc_of_node)) {
                                bool memory_ok = true;

                                if constexpr (use_memory_constraint) {
                                    if (earliest_add == supstepIdx) {
                                        memory_ok = memory_constraint.can_add(succ, proc_of_node);
                                    }
                                }
                                for (unsigned step_to_add = earliest_add;
                                    step_to_add < supstepIdx + stale; ++step_to_add) {
                                    if ((step_to_add == supstepIdx) && !memory_ok) {
                                        continue; 
                                    }
                                    procReady[step_to_add % stale][proc_of_node].emplace(
                                        succ, work_variances[succ]);
                                }
                            }
                        }
                    }

                    procFree[proc_of_node] = true;
                    ++free;
                }
            }

            // Assign new jobs
            if (!CanChooseNode(instance, allReady, procReady[supstepIdx % stale], procFree)) {
                endSupStep = true;
            }
            while (CanChooseNode(instance, allReady, procReady[supstepIdx % stale], procFree)) {
                VertexType nextNode = std::numeric_limits<VertexType>::max();
                unsigned nextProc = P;

                Choose( instance, work_variances, allReady, 
                        procReady[supstepIdx % stale], procFree, 
                        nextNode, nextProc, endSupStep, max_finish_time - time, procTypesCompatibleWithNodeType_skip_proctype);

                if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == P) {
                    endSupStep = true;
                    break;
                }

                if (procReady[supstepIdx % stale][nextProc].find(std::make_pair(nextNode, work_variances[nextNode])) !=
                    procReady[supstepIdx % stale][nextProc].end()) {
                    for (size_t i = 0; i < stale; i++) {
                        procReady[i][nextProc].erase(std::make_pair(nextNode, work_variances[nextNode]));
                    }
                } else {
                    for(unsigned procType : procTypesCompatibleWithNodeType[G.vertex_type(nextNode)]) {
                        allReady[procType].erase(std::make_pair(nextNode, work_variances[nextNode]));
                    }
                    nr_old_ready_nodes_per_type[G.vertex_type(nextNode)]--;
                    const unsigned nextProcType = instance.getArchitecture().processorType(nextProc);
                    number_of_allocated_allReady_tasks_in_superstep[nextProcType]++;
                    if (number_of_allocated_allReady_tasks_in_superstep[nextProcType] >= limit_of_number_of_allocated_allReady_tasks_in_superstep[nextProcType]) {
                        allReady[nextProcType].clear();
                    }
                }

                for (size_t i = 0; i < stale; i++) {
                    ready[i].erase(std::make_pair(nextNode, work_variances[nextNode]));
                }
                old_ready.erase(std::make_pair(nextNode, work_variances[nextNode]));

                schedule.setAssignedProcessor(nextNode, nextProc);
                schedule.setAssignedSuperstep(nextNode, supstepIdx);
                able_to_schedule_in_step = true;

                if constexpr (use_memory_constraint) {
                    memory_constraint.add(nextNode, nextProc);

                    std::vector<std::pair<VertexType, double>> toErase;
                    for (const auto &node_pair : procReady[supstepIdx % stale][nextProc]) {
                        if (!memory_constraint.can_add(node_pair.first, nextProc)) {
                            toErase.push_back(node_pair);
                        }
                    }
                    for (const auto &n : toErase) {
                        procReady[supstepIdx % stale][nextProc].erase(n);
                    }
                }

                finishTimes.emplace(time + G.vertex_work_weight(nextNode), nextNode);
                procFree[nextProc] = false;
                --free;
            }

            if (able_to_schedule_in_step)
                successive_empty_supersteps = 0;
            else if (++successive_empty_supersteps > 100 + stale)
                return RETURN_STATUS::ERROR;

            if (free > static_cast<decltype(free)>(P * max_percent_idle_processors) &&
                ((!increase_parallelism_in_new_superstep) ||
                get_nr_parallelizable_nodes(
                    instance, stale, nr_old_ready_nodes_per_type,
                    nr_ready_stale_nodes_per_type[(supstepIdx + 1) % stale],
                    procReady[(supstepIdx + 1) % stale],
                    nr_procs_per_type) >= std::min(
                                            std::min(P, static_cast<unsigned>(1.2 * (P - free))),
                                            P - free + static_cast<unsigned>(0.5 * free)))) 
            {
                endSupStep = true;
            }
        }

        assert(schedule.satisfiesPrecedenceConstraints());
        //schedule.setAutoCommunicationSchedule();

        return RETURN_STATUS::OSP_SUCCESS;
    }

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {
        return computeSspSchedule(schedule, 1U);
    }

    RETURN_STATUS computeSchedule(MaxBspSchedule<Graph_t> &schedule) {
        return computeSspSchedule(schedule, 2U);
    }

    std::string getScheduleName() const override {
        if constexpr (use_memory_constraint) {
            return "GreedyVarianceSspMemory";
        } else {
            return "GreedyVarianceSsp";
        }
    }

};

} // namespace osp
