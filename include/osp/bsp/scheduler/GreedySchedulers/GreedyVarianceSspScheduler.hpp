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
#include "osp/bsp/scheduler/MaxBspScheduler.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

/**
 * @brief The GreedyVarianceSspScheduler class represents a scheduler that uses a greedy algorithm
 * with stale synchronous parallel (SSP) execution model.
 *
 * It computes schedules for BspInstance using variance-based priorities.
 */
template <typename GraphT, typename MemoryConstraintT = no_memory_constraint>
class GreedyVarianceSspScheduler : public MaxBspScheduler<GraphT> {
    static_assert(IsComputationalDagV<Graph_t>, "GreedyVarianceSspScheduler can only be used with computational DAGs.");

  private:
    using VertexType = vertex_idx_t<Graph_t>;

    constexpr static bool useMemoryConstraint_ = is_memory_constraint_v<MemoryConstraint_t>
                                                 or is_memory_constraint_schedule_v<MemoryConstraint_t>;

    static_assert(not useMemoryConstraint_ or std::is_same_v<GraphT, typename MemoryConstraintT::Graph_impl_t>,
                  "Graph_t must be the same as MemoryConstraint_t::Graph_impl_t.");

    MemoryConstraintT memoryConstraint_;
    double maxPercentIdleProcessors_;
    bool increaseParallelismInNewSuperstep_;

    std::vector<double> ComputeWorkVariance(const GraphT &graph) const {
        std::vector<double> workVariance(graph.NumVertices(), 0.0);
        const std::vector<VertexType> topOrder = GetTopOrder(graph);

        for (auto rIter = top_order.rbegin(); rIter != top_order.crend(); r_iter++) {
            double temp = 0;
            double maxPriority = 0;
            for (const auto &child : graph.Children(*r_iter)) {
                max_priority = std::max(work_variance[child], max_priority);
            }
            for (const auto &child : graph.Children(*r_iter)) {
                temp += std::exp(2 * (work_variance[child] - max_priority));
            }
            temp = std::log(temp) / 2 + maxPriority;

            double nodeWeight
                = std::log(static_cast<double>(std::max(graph.VertexWorkWeight(*r_iter), static_cast<v_workw_t<Graph_t>>(1))));
            double largerVal = nodeWeight > temp ? nodeWeight : temp;

            workVariance[*r_iter] = std::log(std::exp(nodeWeight - largerVal) + std::exp(temp - largerVal)) + largerVal;
        }

        return workVariance;
    }

    std::vector<std::vector<std::vector<unsigned>>> ProcTypesCompatibleWithNodeTypeOmitProcType(
        const BspInstance<GraphT> &instance) const {
        const std::vector<std::vector<unsigned>> procTypesCompatibleWithNodeType = instance.getProcTypesCompatibleWithNodeType();

        std::vector<std::vector<std::vector<unsigned>>> procTypesCompatibleWithNodeTypeSkip(
            instance.GetArchitecture().getNumberOfProcessorTypes(),
            std::vector<std::vector<unsigned>>(instance.GetComputationalDag().NumVertexTypes()));
        for (unsigned procType = 0; procType < instance.GetArchitecture().getNumberOfProcessorTypes(); procType++) {
            for (unsigned nodeType = 0; nodeType < instance.GetComputationalDag().NumVertexTypes(); nodeType++) {
                for (unsigned otherProcType : procTypesCompatibleWithNodeType[nodeType]) {
                    if (procType == otherProcType) {
                        continue;
                    }
                    procTypesCompatibleWithNodeTypeSkip[procType][nodeType].emplace_back(otherProcType);
                }
            }
        }

        return procTypesCompatibleWithNodeTypeSkip;
    }

    struct VarianceCompare {
        bool operator()(const std::pair<VertexType, double> &lhs, const std::pair<VertexType, double> &rhs) const {
            return ((lhs.second > rhs.second) || ((lhs.second >= rhs.second) && (lhs.first < rhs.first)));
        }
    };

    bool CanChooseNode(const BspInstance<GraphT> &instance,
                       const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
                       const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
                       const std::vector<bool> &procFree) const {
        for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
            if (procFree[i] && !procReady[i].empty()) {
                return true;
            }
        }

        for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
            if (procFree[i] && !allReady[instance.GetArchitecture().processorType(i)].empty()) {
                return true;
            }
        }

        return false;
    }

    void Choose(const BspInstance<GraphT> &instance,
                const std::vector<double> &workVariance,
                std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
                std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
                const std::vector<bool> &procFree,
                VertexType &node,
                unsigned &p,
                const bool endSupStep,
                const v_workw_t<Graph_t> remainingTime,
                const std::vector<std::vector<std::vector<unsigned>>> &procTypesCompatibleWithNodeTypeSkipProctype) const {
        double maxScore = -1;
        bool foundAllocation = false;

        for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
            if (!procFree[i] || procReady[i].empty()) {
                continue;
            }

            auto it = procReady[i].begin();
            while (it != procReady[i].end()) {
                if (endSupStep && (remaining_time < instance.GetComputationalDag().VertexWorkWeight(it->first))) {
                    it = procReady[i].erase(it);
                    continue;
                }

                const double &score = it->second;

                if (score > maxScore) {
                    const unsigned procType = instance.GetArchitecture().processorType(i);

                    if constexpr (useMemoryConstraint_) {
                        if (memoryConstraint_.can_add(it->first, i)) {
                            node = it->first;
                            p = i;
                            foundAllocation = true;

                            if (procType < procTypesCompatibleWithNodeTypeSkipProctype.size()) {
                                const auto &compatibleTypes
                                    = procTypesCompatibleWithNodeTypeSkipProctype[procType]
                                                                                 [instance.GetComputationalDag().VertexType(node)];

                                for (unsigned otherType : compatibleTypes) {
                                    for (unsigned j = 0; j < instance.NumberOfProcessors(); ++j) {
                                        if (j != i && instance.GetArchitecture().processorType(j) == otherType
                                            && j < procReady.size()) {
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
                        foundAllocation = true;

                        if (procType < procTypesCompatibleWithNodeTypeSkipProctype.size()) {
                            const auto &compatibleTypes
                                = procTypesCompatibleWithNodeTypeSkipProctype[procType]
                                                                             [instance.GetComputationalDag().VertexType(node)];

                            for (unsigned otherType : compatibleTypes) {
                                for (unsigned j = 0; j < instance.NumberOfProcessors(); ++j) {
                                    if (j != i && instance.GetArchitecture().processorType(j) == otherType && j < procReady.size()) {
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

        if (foundAllocation) {
            return;
        }

        for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
            const unsigned procType = instance.GetArchitecture().processorType(i);
            if (!procFree[i] || procType >= allReady.size() || allReady[procType].empty()) {
                continue;
            }

            auto &readyList = allReady[procType];
            auto it = readyList.begin();

            while (it != readyList.end()) {
                if (endSupStep && (remaining_time < instance.GetComputationalDag().VertexWorkWeight(it->first))) {
                    it = readyList.erase(it);
                    continue;
                }

                const double &score = it->second;

                if (score > maxScore) {
                    if constexpr (useMemoryConstraint_) {
                        if (memoryConstraint_.can_add(it->first, i)) {
                            node = it->first;
                            p = i;

                            const auto &compatibleTypes
                                = procTypesCompatibleWithNodeTypeSkipProctype[procType]
                                                                             [instance.GetComputationalDag().VertexType(node)];

                            for (unsigned otherType : compatibleTypes) {
                                if (otherType < allReady.size()) {
                                    allReady[otherType].erase(std::make_pair(node, work_variance[node]));
                                }
                            }

                            return;
                        }
                    } else {
                        node = it->first;
                        p = i;

                        const auto &compatibleTypes
                            = procTypesCompatibleWithNodeTypeSkipProctype[procType][instance.GetComputationalDag().VertexType(node)];

                        for (unsigned otherType : compatibleTypes) {
                            if (otherType < allReady.size()) {
                                allReady[otherType].erase(std::make_pair(node, work_variance[node]));
                            }
                        }

                        return;
                    }
                }
                ++it;
            }
        }
    };

    bool CheckMemFeasibility(const BspInstance<GraphT> &instance,
                             const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &allReady,
                             const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady) const {
        if constexpr (useMemoryConstraint_) {
            if (instance.GetArchitecture().getMemoryConstraintType() == MEMORY_CONSTRAINT_TYPE::PERSISTENT_AND_TRANSIENT) {
                for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                    if (!procReady[i].empty()) {
                        const std::pair<VertexType, double> &nodePair = *procReady[i].begin();
                        VertexType topNode = node_pair.first;

                        if (memoryConstraint_.can_add(top_node, i)) {
                            return true;
                        }
                    }
                }

                for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                    if (allReady[instance.GetArchitecture().processorType(i)].empty()) {
                        continue;
                    }

                    const std::pair<VertexType, double> &nodePair = *allReady[instance.GetArchitecture().processorType(i)].begin();
                    VertexType topNode = node_pair.first;

                    if (memoryConstraint_.can_add(top_node, i)) {
                        return true;
                    }
                }

                return false;
            }
        }

        return true;
    }

    unsigned GetNrParallelizableNodes(const BspInstance<GraphT> &instance,
                                      const unsigned &stale,
                                      const std::vector<unsigned> &nrOldReadyNodesPerType,
                                      const std::vector<unsigned> &nrReadyNodesPerType,
                                      const std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> &procReady,
                                      const std::vector<unsigned> &nrProcsPerType) const {
        unsigned nrNodes = 0;
        unsigned numProcTypes = instance.GetArchitecture().getNumberOfProcessorTypes();

        std::vector<unsigned> procsPerType = nrProcsPerType;

        if (stale > 1) {
            for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                if (!procReady[proc].empty()) {
                    procsPerType[instance.GetArchitecture().processorType(proc)]--;
                    nrNodes++;
                }
            }
        }

        std::vector<unsigned> readyNodesPerType = nrReadyNodesPerType;
        for (unsigned nodeType = 0; nodeType < readyNodesPerType.size(); nodeType++) {
            readyNodesPerType[nodeType] += nrOldReadyNodesPerType[nodeType];
        }

        for (unsigned procType = 0; procType < numProcTypes; ++procType) {
            for (unsigned nodeType = 0; nodeType < instance.GetComputationalDag().NumVertexTypes(); ++nodeType) {
                if (instance.isCompatibleType(nodeType, procType)) {
                    unsigned matched = std::min(readyNodesPerType[nodeType], procsPerType[procType]);
                    nrNodes += matched;
                    readyNodesPerType[nodeType] -= matched;
                    procsPerType[procType] -= matched;
                }
            }
        }

        return nrNodes;
    }

  public:
    /**
     * @brief Default constructor for GreedyVarianceSspScheduler.
     */
    GreedyVarianceSspScheduler(float maxPercentIdleProcessors = 0.2f, bool increaseParallelismInNewSuperstep = true)
        : maxPercentIdleProcessors_(maxPercentIdleProcessors),
          increaseParallelismInNewSuperstep_(increaseParallelismInNewSuperstep) {}

    /**
     * @brief Default destructor for GreedyVarianceSspScheduler.
     */
    virtual ~GreedyVarianceSspScheduler() = default;

    RETURN_STATUS ComputeSspSchedule(BspSchedule<GraphT> &schedule, unsigned stale) {
        const auto &instance = schedule.GetInstance();
        const auto &g = instance.GetComputationalDag();
        const VertexType &n = instance.NumberOfVertices();
        const unsigned &p = instance.NumberOfProcessors();

        unsigned supstepIdx = 0;

        if constexpr (is_memory_constraint_v<MemoryConstraint_t>) {
            memoryConstraint_.initialize(instance);
        } else if constexpr (is_memory_constraint_schedule_v<MemoryConstraint_t>) {
            memoryConstraint_.initialize(schedule, supstepIdx);
        }

        const std::vector<double> workVariances = ComputeWorkVariance(g);

        std::set<std::pair<VertexType, double>, VarianceCompare> oldReady;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> ready(stale);
        std::vector<std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>>> procReady(
            stale, std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>>(P));
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> allReady(
            instance.GetArchitecture().getNumberOfProcessorTypes());

        const auto procTypesCompatibleWithNodeType = instance.getProcTypesCompatibleWithNodeType();
        const std::vector<std::vector<std::vector<unsigned>>> procTypesCompatibleWithNodeTypeSkipProctype
            = ProcTypesCompatibleWithNodeTypeOmitProcType(instance);

        std::vector<unsigned> nrOldReadyNodesPerType(g.NumVertexTypes(), 0);
        std::vector<std::vector<unsigned>> nrReadyStaleNodesPerType(stale, std::vector<unsigned>(g.NumVertexTypes(), 0));
        std::vector<unsigned> nrProcsPerType(instance.GetArchitecture().getNumberOfProcessorTypes(), 0);
        for (auto proc = 0u; proc < p; ++proc) {
            ++nrProcsPerType[instance.GetArchitecture().processorType(proc)];
        }

        std::vector<VertexType> nrPredecRemain(n);

        for (VertexType node = 0; node < N; ++node) {
            const auto numParents = g.in_degree(node);

            nrPredecRemain[node] = num_parents;

            if (numParents == 0) {
                ready[0].insert(std::make_pair(node, workVariances[node]));
                nrReadyStaleNodesPerType[0][g.VertexType(node)]++;
            }
        }

        std::vector<bool> procFree(p, true);
        unsigned free = p;

        std::set<std::pair<v_workw_t<Graph_t>, VertexType>> finishTimes;
        finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

        std::vector<unsigned> numberOfAllocatedAllReadyTasksInSuperstep(instance.GetArchitecture().getNumberOfProcessorTypes(), 0);
        std::vector<unsigned> limitOfNumberOfAllocatedAllReadyTasksInSuperstep(
            instance.GetArchitecture().getNumberOfProcessorTypes(), 0);

        bool endSupStep = true;
        bool beginOuterWhile = true;
        bool ableToScheduleInStep = false;
        unsigned successiveEmptySupersteps = 0u;

        auto nonemptyReady = [&]() {
            return std::any_of(
                ready.cbegin(), ready.cend(), [](const std::set<std::pair<VertexType, double>, VarianceCompare> &readySet) {
                    return !ready_set.empty();
                });
        };

        while (!old_ready.empty() || nonemptyReady() || !finishTimes.empty()) {
            if (finishTimes.empty() && endSupStep) {
                ableToScheduleInStep = false;
                numberOfAllocatedAllReadyTasksInSuperstep
                    = std::vector<unsigned>(instance.GetArchitecture().getNumberOfProcessorTypes(), 0);

                for (unsigned i = 0; i < p; ++i) {
                    procReady[supstepIdx % stale][i].clear();
                }

                if (!beginOuterWhile) {
                    supstepIdx++;
                } else {
                    beginOuterWhile = false;
                }

                for (unsigned procType = 0; procType < instance.GetArchitecture().getNumberOfProcessorTypes(); ++procType) {
                    allReady[procType].clear();
                }

                oldReady.insert(ready[supstepIdx % stale].begin(), ready[supstepIdx % stale].end());
                ready[supstepIdx % stale].clear();
                for (unsigned nodeType = 0; nodeType < g.NumVertexTypes(); ++nodeType) {
                    nrOldReadyNodesPerType[nodeType] += nrReadyStaleNodesPerType[supstepIdx % stale][nodeType];
                    nrReadyStaleNodesPerType[supstepIdx % stale][nodeType] = 0;
                }

                for (const auto &nodeAndValuePair : old_ready) {
                    VertexType node = nodeAndValuePair.first;
                    for (unsigned procType : procTypesCompatibleWithNodeType[G.VertexType(node)]) {
                        allReady[procType].insert(allReady[procType].end(), nodeAndValuePair);
                    }
                }

                if constexpr (useMemoryConstraint_) {
                    if (instance.GetArchitecture().getMemoryConstraintType() == MEMORY_CONSTRAINT_TYPE::LOCAL) {
                        for (unsigned proc = 0; proc < p; proc++) {
                            memoryConstraint_.reset(proc);
                        }
                    }
                }

                for (unsigned procType = 0; procType < instance.GetArchitecture().getNumberOfProcessorTypes(); procType++) {
                    unsigned equalSplit = (static_cast<unsigned>(allReady[procType].size()) + stale - 1) / stale;
                    unsigned atLeastForLongStep = 3 * nrProcsPerType[procType];
                    limitOfNumberOfAllocatedAllReadyTasksInSuperstep[procType] = std::max(atLeastForLongStep, equalSplit);
                }

                endSupStep = false;
                finishTimes.emplace(0, std::numeric_limits<VertexType>::max());
            }

            const v_workw_t<Graph_t> time = finishTimes.begin()->first;
            const v_workw_t<Graph_t> maxFinishTime = finishTimes.rbegin()->first;

            // Find new ready jobs
            while (!finishTimes.empty() && finishTimes.begin()->first == time) {
                const VertexType node = finishTimes.begin()->second;
                finishTimes.erase(finishTimes.begin());

                if (node != std::numeric_limits<VertexType>::max()) {
                    const unsigned procOfNode = schedule.assignedProcessor(node);

                    for (const auto &succ : G.Children(node)) {
                        nrPredecRemain[succ]--;
                        if (nrPredecRemain[succ] == 0) {
                            ready[supstepIdx % stale].emplace(succ, work_variances[succ]);
                            nr_ready_stale_nodes_per_type[supstepIdx % stale][G.VertexType(succ)]++;

                            unsigned earliest_add = supstepIdx;
                            for (const auto &pred : G.Parents(succ)) {
                                if (schedule.assignedProcessor(pred) != proc_of_node) {
                                    earliest_add = std::max(earliest_add, stale + schedule.assignedSuperstep(pred));
                                }
                            }

                            if (instance.isCompatible(succ, proc_of_node)) {
                                bool memory_ok = true;

                                if constexpr (use_memory_constraint) {
                                    if (earliest_add == supstepIdx) {
                                        memory_ok = memory_constraint.can_add(succ, proc_of_node);
                                    }
                                }
                                for (unsigned step_to_add = earliest_add; step_to_add < supstepIdx + stale; ++step_to_add) {
                                    if ((step_to_add == supstepIdx) && !memory_ok) {
                                        continue;
                                    }
                                    procReady[step_to_add % stale][proc_of_node].emplace(succ, work_variances[succ]);
                                }
                            }
                        }
                    }

                    procFree[procOfNode] = true;
                    ++free;
                }
            }

            // Assign new jobs
            if (!CanChooseNode(instance, allReady, procReady[supstepIdx % stale], procFree)) {
                endSupStep = true;
            }

            while (CanChooseNode(instance, allReady, procReady[supstepIdx % stale], procFree)) {
                VertexType nextNode = std::numeric_limits<VertexType>::max();
                unsigned nextProc = p;

                Choose(instance,
                       work_variances,
                       allReady,
                       procReady[supstepIdx % stale],
                       procFree,
                       nextNode,
                       nextProc,
                       endSupStep,
                       max_finish_time - time,
                       procTypesCompatibleWithNodeType_skip_proctype);

                if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == P) {
                    endSupStep = true;
                    break;
                }

                if (procReady[supstepIdx % stale][nextProc].find(std::make_pair(nextNode, workVariances[nextNode]))
                    != procReady[supstepIdx % stale][nextProc].end()) {
                    for (size_t i = 0; i < stale; i++) {
                        procReady[i][nextProc].erase(std::make_pair(nextNode, workVariances[nextNode]));
                    }
                } else {
                    for (unsigned procType : procTypesCompatibleWithNodeType[G.VertexType(nextNode)]) {
                        allReady[procType].erase(std::make_pair(nextNode, work_variances[nextNode]));
                    }
                    nrOldReadyNodesPerType[g.VertexType(nextNode)]--;
                    const unsigned nextProcType = instance.GetArchitecture().processorType(nextProc);
                    numberOfAllocatedAllReadyTasksInSuperstep[nextProcType]++;

                    if (numberOfAllocatedAllReadyTasksInSuperstep[nextProcType]
                        >= limitOfNumberOfAllocatedAllReadyTasksInSuperstep[nextProcType]) {
                        allReady[nextProcType].clear();
                    }
                }

                for (size_t i = 0; i < stale; i++) {
                    ready[i].erase(std::make_pair(nextNode, workVariances[nextNode]));
                }

                oldReady.erase(std::make_pair(nextNode, workVariances[nextNode]));

                schedule.setAssignedProcessor(nextNode, nextProc);
                schedule.setAssignedSuperstep(nextNode, supstepIdx);
                ableToScheduleInStep = true;

                if constexpr (useMemoryConstraint_) {
                    memoryConstraint_.add(nextNode, nextProc);

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

                finishTimes.emplace(time + g.VertexWorkWeight(nextNode), nextNode);
                procFree[nextProc] = false;
                --free;
            }

            if (ableToScheduleInStep) {
                successiveEmptySupersteps = 0;
            } else if (++successiveEmptySupersteps > 100 + stale) {
                return RETURN_STATUS::ERROR;
            }

            if (free > (P * max_percent_idle_processors)
                && ((!increase_parallelism_in_new_superstep)
                    || get_nr_parallelizable_nodes(instance,
                                                   stale,
                                                   nr_old_ready_nodes_per_type,
                                                   nr_ready_stale_nodes_per_type[(supstepIdx + 1) % stale],
                                                   procReady[(supstepIdx + 1) % stale],
                                                   nr_procs_per_type)
                           >= std::min(std::min(P, static_cast<unsigned>(1.2 * (P - free))),
                                       P - free + static_cast<unsigned>(0.5 * free)))) {
                endSupStep = true;
            }
        }

        assert(schedule.satisfiesPrecedenceConstraints());
        // schedule.setAutoCommunicationSchedule();

        return RETURN_STATUS::OSP_SUCCESS;
    }

    RETURN_STATUS computeSchedule(BspSchedule<GraphT> &schedule) override { return computeSspSchedule(schedule, 1U); }

    RETURN_STATUS computeSchedule(MaxBspSchedule<GraphT> &schedule) override { return computeSspSchedule(schedule, 2U); }

    std::string getScheduleName() const override {
        if constexpr (useMemoryConstraint_) {
            return "GreedyVarianceSspMemory";
        } else {
            return "GreedyVarianceSsp";
        }
    }
};

}    // namespace osp
