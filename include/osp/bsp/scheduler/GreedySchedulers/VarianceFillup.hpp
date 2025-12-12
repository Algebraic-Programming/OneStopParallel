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
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

/**
 * @brief The VarianceFillup class represents a scheduler that uses a greedy algorithm to compute
 * schedules for BspInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "BspGreedy" in this case.
 */
template <typename GraphT, typename MemoryConstraintT = NoMemoryConstraint>
class VarianceFillup : public Scheduler<GraphT> {
    static_assert(IsComputationalDagV<Graph_t>, "VarianceFillup can only be used with computational DAGs.");

  private:
    using VertexType = vertex_idx_t<Graph_t>;

    constexpr static bool useMemoryConstraint_ = is_memory_constraint_v<MemoryConstraintT>
                                                 or is_memory_constraint_schedule_v<MemoryConstraintT>;

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
            for (const auto &child : graph.children(*r_iter)) {
                max_priority = std::max(work_variance[child], max_priority);
            }
            for (const auto &child : graph.children(*r_iter)) {
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
            std::vector<std::vector<unsigned>>(instance.getComputationalDag().NumVertexTypes()));
        for (unsigned procType = 0; procType < instance.GetArchitecture().getNumberOfProcessorTypes(); procType++) {
            for (unsigned nodeType = 0; nodeType < instance.getComputationalDag().NumVertexTypes(); nodeType++) {
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
    };

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
        for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
            if (procFree[i] && !procReady[i].empty()) {
                // select node
                for (auto nodePairIt = procReady[i].begin(); nodePairIt != procReady[i].end();) {
                    if (endSupStep && (remaining_time < instance.getComputationalDag().VertexWorkWeight(node_pair_it->first))) {
                        nodePairIt = procReady[i].erase(node_pair_it);
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
                    nodePairIt++;
                }
            }
        }

        for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
            if (procFree[i] && !allReady[instance.GetArchitecture().processorType(i)].empty()) {
                // select node
                for (auto it = allReady[instance.GetArchitecture().processorType(i)].begin();
                     it != allReady[instance.GetArchitecture().processorType(i)].end();) {
                    if (endSupStep && (remaining_time < instance.getComputationalDag().VertexWorkWeight(it->first))) {
                        it = allReady[instance.GetArchitecture().processorType(i)].erase(it);
                        continue;
                    }

                    const double &score = it->second;

                    if (score > maxScore) {
                        if constexpr (useMemoryConstraint_) {
                            if (memoryConstraint_.can_add(it->first, i)) {
                                node = it->first;
                                p = i;

                                allReady[instance.GetArchitecture().processorType(i)].erase(it);
                                for (unsigned procType :
                                     procTypesCompatibleWithNodeType_skip_proctype[instance.GetArchitecture().processorType(
                                         i)][instance.getComputationalDag().VertexType(node)]) {
                                    allReady[procType].erase(std::make_pair(node, work_variance[node]));
                                }
                                return;
                            }
                        } else {
                            node = it->first;
                            p = i;

                            allReady[instance.GetArchitecture().processorType(i)].erase(it);
                            for (unsigned procType :
                                 procTypesCompatibleWithNodeType_skip_proctype[instance.GetArchitecture().processorType(i)]
                                                                              [instance.getComputationalDag().VertexType(node)]) {
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

    unsigned GetNrParallelizableNodes(const BspInstance<GraphT> &instance,
                                      const std::vector<unsigned> &nrReadyNodesPerType,
                                      const std::vector<unsigned> &nrProcsPerType) const {
        unsigned nrNodes = 0;

        std::vector<unsigned> readyNodesPerType = nrReadyNodesPerType;
        std::vector<unsigned> procsPerType = nrProcsPerType;
        for (unsigned procType = 0; procType < instance.GetArchitecture().getNumberOfProcessorTypes(); ++procType) {
            for (unsigned nodeType = 0; nodeType < instance.getComputationalDag().NumVertexTypes(); ++nodeType) {
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
     * @brief Default constructor for VarianceFillup.
     */
    VarianceFillup(float maxPercentIdleProcessors = 0.2f, bool increaseParallelismInNewSuperstep = true)
        : maxPercentIdleProcessors_(maxPercentIdleProcessors),
          increaseParallelismInNewSuperstep_(increaseParallelismInNewSuperstep) {}

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
    virtual RETURN_STATUS computeSchedule(BspSchedule<GraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();

        for (const auto &v : instance.getComputationalDag().vertices()) {
            schedule.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
        }

        unsigned supstepIdx = 0;

        if constexpr (is_memory_constraint_v<MemoryConstraintT>) {
            memoryConstraint_.initialize(instance);
        } else if constexpr (is_memory_constraint_schedule_v<MemoryConstraintT>) {
            memoryConstraint_.initialize(schedule, supstepIdx);
        }

        const auto &n = instance.numberOfVertices();
        const unsigned &paramsP = instance.NumberOfProcessors();
        const auto &g = instance.getComputationalDag();

        const std::vector<double> workVariances = ComputeWorkVariance(g);

        std::set<std::pair<VertexType, double>, VarianceCompare> ready;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> procReady(paramsP);
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> allReady(
            instance.GetArchitecture().getNumberOfProcessorTypes());

        const std::vector<std::vector<unsigned>> procTypesCompatibleWithNodeType = instance.getProcTypesCompatibleWithNodeType();
        const std::vector<std::vector<std::vector<unsigned>>> procTypesCompatibleWithNodeTypeSkipProctype
            = ProcTypesCompatibleWithNodeTypeOmitProcType(instance);

        std::vector<unsigned> nrReadyNodesPerType(g.NumVertexTypes(), 0);
        std::vector<unsigned> nrProcsPerType(instance.GetArchitecture().getNumberOfProcessorTypes(), 0);
        for (unsigned proc = 0; proc < paramsP; ++proc) {
            ++nrProcsPerType[instance.GetArchitecture().processorType(proc)];
        }

        std::vector<VertexType> nrPredecRemain(n);
        for (VertexType node = 0; node < n; node++) {
            const auto numParents = g.in_degree(node);
            nrPredecRemain[node] = num_parents;
            if (numParents == 0) {
                ready.insert(std::make_pair(node, workVariances[node]));
                ++nrReadyNodesPerType[g.VertexType(node)];
                for (unsigned procType : procTypesCompatibleWithNodeType[G.VertexType(node)]) {
                    allReady[procType].insert(std::make_pair(node, work_variances[node]));
                }
            }
        }

        std::vector<bool> procFree(paramsP, true);
        unsigned free = paramsP;

        std::set<std::pair<v_workw_t<Graph_t>, VertexType>> finishTimes;
        finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

        bool endSupStep = false;
        while (!ready.empty() || !finishTimes.empty()) {
            if (finishTimes.empty() && endSupStep) {
                for (unsigned i = 0; i < paramsP; ++i) {
                    procReady[i].clear();

                    if constexpr (useMemoryConstraint_) {
                        memoryConstraint_.reset(i);
                    }
                }

                for (unsigned procType = 0; procType < instance.GetArchitecture().getNumberOfProcessorTypes(); ++procType) {
                    allReady[procType].clear();
                }

                for (const auto &nodeAndValuePair : ready) {
                    const auto node = nodeAndValuePair.first;
                    for (unsigned procType : procTypesCompatibleWithNodeType[G.VertexType(node)]) {
                        allReady[procType].insert(allReady[procType].end(), nodeAndValuePair);
                    }
                }

                ++supstepIdx;
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
                    for (const auto &succ : G.children(node)) {
                        nrPredecRemain[succ]--;
                        if (nrPredecRemain[succ] == 0) {
                            ready.emplace(succ, work_variances[succ]);
                            ++nr_ready_nodes_per_type[G.VertexType(succ)];

                            bool canAdd = true;
                            for (const auto &pred : G.parents(succ)) {
                                if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node)
                                    && schedule.assignedSuperstep(pred) == supstepIdx) {
                                    canAdd = false;
                                }
                            }

                            if constexpr (use_memory_constraint) {
                                if (canAdd) {
                                    if (not memory_constraint.can_add(succ, schedule.assignedProcessor(node))) {
                                        canAdd = false;
                                    }
                                }
                            }

                            if (!instance.isCompatible(succ, schedule.assignedProcessor(node))) {
                                canAdd = false;
                            }

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
                unsigned nextProc = paramsP;
                Choose(instance,
                       work_variances,
                       allReady,
                       procReady,
                       procFree,
                       nextNode,
                       nextProc,
                       endSupStep,
                       max_finish_time - time,
                       procTypesCompatibleWithNodeType_skip_proctype);

                if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == params_p) {
                    endSupStep = true;
                    break;
                }

                ready.erase(std::make_pair(nextNode, workVariances[nextNode]));
                --nrReadyNodesPerType[g.VertexType(nextNode)];
                schedule.setAssignedProcessor(nextNode, nextProc);
                schedule.setAssignedSuperstep(nextNode, supstepIdx);

                if constexpr (useMemoryConstraint_) {
                    memoryConstraint_.add(nextNode, nextProc);

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

                finishTimes.emplace(time + g.VertexWorkWeight(nextNode), nextNode);
                procFree[nextProc] = false;
                --free;
            }

            if constexpr (useMemoryConstraint_) {
                if (not check_mem_feasibility(instance, allReady, procReady)) {
                    return RETURN_STATUS::ERROR;
                }
            }

            if (free > paramsP * maxPercentIdleProcessors_
                && ((!increaseParallelismInNewSuperstep_)
                    || GetNrParallelizableNodes(instance, nrReadyNodesPerType, nrProcsPerType)
                           >= std::min(std::min(paramsP, static_cast<unsigned>(1.2 * (paramsP - free))),
                                       paramsP - free + (static_cast<unsigned>(0.5 * free))))) {
                endSupStep = true;
            }
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
        if constexpr (useMemoryConstraint_) {
            return "VarianceGreedyFillupMemory";
        } else {
            return "VarianceGreedyFillup";
        }
    }
};

}    // namespace osp
