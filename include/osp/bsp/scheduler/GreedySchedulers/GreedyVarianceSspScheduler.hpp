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
template <typename GraphT, typename MemoryConstraintT = NoMemoryConstraint>
class GreedyVarianceSspScheduler : public MaxBspScheduler<GraphT> {
    static_assert(isComputationalDagV<GraphT>, "GreedyVarianceSspScheduler can only be used with computational DAGs.");

  private:
    using VertexType = VertexIdxT<GraphT>;

    constexpr static bool useMemoryConstraint_ = isMemoryConstraintV<MemoryConstraintT>
                                                 or isMemoryConstraintScheduleV<MemoryConstraintT>;

    static_assert(not useMemoryConstraint_ or std::is_same_v<GraphT, typename MemoryConstraintT::GraphImplT>,
                  "GraphT must be the same as MemoryConstraintT::GraphImplT.");

    MemoryConstraintT memoryConstraint_;
    double maxPercentIdleProcessors_;
    bool increaseParallelismInNewSuperstep_;

    std::vector<double> ComputeWorkVariance(const GraphT &graph) const {
        std::vector<double> workVariance(graph.NumVertices(), 0.0);
        const std::vector<VertexType> topOrder = GetTopOrder(graph);

        for (auto rIter = topOrder.rbegin(); rIter != topOrder.crend(); rIter++) {
            double temp = 0;
            double maxPriority = 0;
            for (const auto &child : graph.Children(*rIter)) {
                maxPriority = std::max(workVariance[child], maxPriority);
            }
            for (const auto &child : graph.Children(*rIter)) {
                temp += std::exp(2 * (workVariance[child] - maxPriority));
            }
            temp = std::log(temp) / 2 + maxPriority;

            double nodeWeight
                = std::log(static_cast<double>(std::max(graph.VertexWorkWeight(*rIter), static_cast<VWorkwT<GraphT>>(1))));
            double largerVal = nodeWeight > temp ? nodeWeight : temp;

            workVariance[*rIter] = std::log(std::exp(nodeWeight - largerVal) + std::exp(temp - largerVal)) + largerVal;
        }

        return workVariance;
    }

    std::vector<std::vector<std::vector<unsigned>>> ProcTypesCompatibleWithNodeTypeOmitProcType(
        const BspInstance<GraphT> &instance) const {
        const std::vector<std::vector<unsigned>> procTypesCompatibleWithNodeType = instance.GetProcTypesCompatibleWithNodeType();

        std::vector<std::vector<std::vector<unsigned>>> procTypesCompatibleWithNodeTypeSkip(
            instance.GetArchitecture().GetNumberOfProcessorTypes(),
            std::vector<std::vector<unsigned>>(instance.GetComputationalDag().NumVertexTypes()));
        for (unsigned procType = 0; procType < instance.GetArchitecture().GetNumberOfProcessorTypes(); procType++) {
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
            if (procFree[i] && !allReady[instance.GetArchitecture().ProcessorType(i)].empty()) {
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
                const VWorkwT<GraphT> remainingTime,
                const std::vector<std::vector<std::vector<unsigned>>> &procTypesCompatibleWithNodeTypeSkipProctype) const {
        double maxScore = -1;
        bool foundAllocation = false;

        for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
            if (!procFree[i] || procReady[i].empty()) {
                continue;
            }

            auto it = procReady[i].begin();
            while (it != procReady[i].end()) {
                if (endSupStep && (remainingTime < instance.GetComputationalDag().VertexWorkWeight(it->first))) {
                    it = procReady[i].erase(it);
                    continue;
                }

                const double &score = it->second;

                if (score > maxScore) {
                    const unsigned procType = instance.GetArchitecture().ProcessorType(i);

                    if constexpr (useMemoryConstraint_) {
                        if (memoryConstraint_.CanAdd(it->first, i)) {
                            node = it->first;
                            p = i;
                            foundAllocation = true;

                            if (procType < procTypesCompatibleWithNodeTypeSkipProctype.size()) {
                                const auto &compatibleTypes
                                    = procTypesCompatibleWithNodeTypeSkipProctype[procType]
                                                                                 [instance.GetComputationalDag().VertexType(node)];

                                for (unsigned otherType : compatibleTypes) {
                                    for (unsigned j = 0; j < instance.NumberOfProcessors(); ++j) {
                                        if (j != i && instance.GetArchitecture().ProcessorType(j) == otherType
                                            && j < procReady.size()) {
                                            procReady[j].erase(std::make_pair(node, workVariance[node]));
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
                                    if (j != i && instance.GetArchitecture().ProcessorType(j) == otherType && j < procReady.size()) {
                                        procReady[j].erase(std::make_pair(node, workVariance[node]));
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
            const unsigned procType = instance.GetArchitecture().ProcessorType(i);
            if (!procFree[i] || procType >= allReady.size() || allReady[procType].empty()) {
                continue;
            }

            auto &readyList = allReady[procType];
            auto it = readyList.begin();

            while (it != readyList.end()) {
                if (endSupStep && (remainingTime < instance.GetComputationalDag().VertexWorkWeight(it->first))) {
                    it = readyList.erase(it);
                    continue;
                }

                const double &score = it->second;

                if (score > maxScore) {
                    if constexpr (useMemoryConstraint_) {
                        if (memoryConstraint_.CanAdd(it->first, i)) {
                            node = it->first;
                            p = i;

                            const auto &compatibleTypes
                                = procTypesCompatibleWithNodeTypeSkipProctype[procType]
                                                                             [instance.GetComputationalDag().VertexType(node)];

                            for (unsigned otherType : compatibleTypes) {
                                if (otherType < allReady.size()) {
                                    allReady[otherType].erase(std::make_pair(node, workVariance[node]));
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
                                allReady[otherType].erase(std::make_pair(node, workVariance[node]));
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
            if (instance.GetArchitecture().GetMemoryConstraintType() == MemoryConstraintType::PERSISTENT_AND_TRANSIENT) {
                for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                    if (!procReady[i].empty()) {
                        const std::pair<VertexType, double> &nodePair = *procReady[i].begin();
                        VertexType topNode = nodePair.first;

                        if (memoryConstraint_.CanAdd(topNode, i)) {
                            return true;
                        }
                    }
                }

                for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                    if (allReady[instance.GetArchitecture().ProcessorType(i)].empty()) {
                        continue;
                    }

                    const std::pair<VertexType, double> &nodePair = *allReady[instance.GetArchitecture().ProcessorType(i)].begin();
                    VertexType topNode = nodePair.first;

                    if (memoryConstraint_.CanAdd(topNode, i)) {
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
        unsigned numProcTypes = instance.GetArchitecture().GetNumberOfProcessorTypes();

        std::vector<unsigned> procsPerType = nrProcsPerType;

        if (stale > 1) {
            for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                if (!procReady[proc].empty()) {
                    procsPerType[instance.GetArchitecture().ProcessorType(proc)]--;
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
                if (instance.IsCompatibleType(nodeType, procType)) {
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

    ReturnStatus ComputeSspSchedule(BspSchedule<GraphT> &schedule, unsigned stale) {
        const auto &instance = schedule.GetInstance();
        const auto &g = instance.GetComputationalDag();
        const VertexType &n = instance.NumberOfVertices();
        const unsigned &p = instance.NumberOfProcessors();

        unsigned supstepIdx = 0;

        if constexpr (isMemoryConstraintV<MemoryConstraintT>) {
            memoryConstraint_.Initialize(instance);
        } else if constexpr (isMemoryConstraintScheduleV<MemoryConstraintT>) {
            memoryConstraint_.Initialize(schedule, supstepIdx);
        }

        const std::vector<double> workVariances = ComputeWorkVariance(g);

        std::set<std::pair<VertexType, double>, VarianceCompare> oldReady;
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> ready(stale);
        std::vector<std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>>> procReady(
            stale, std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>>(p));
        std::vector<std::set<std::pair<VertexType, double>, VarianceCompare>> allReady(
            instance.GetArchitecture().GetNumberOfProcessorTypes());

        const auto procTypesCompatibleWithNodeType = instance.GetProcTypesCompatibleWithNodeType();
        const std::vector<std::vector<std::vector<unsigned>>> procTypesCompatibleWithNodeTypeSkipProctype
            = ProcTypesCompatibleWithNodeTypeOmitProcType(instance);

        std::vector<unsigned> nrOldReadyNodesPerType(g.NumVertexTypes(), 0);
        std::vector<std::vector<unsigned>> nrReadyStaleNodesPerType(stale, std::vector<unsigned>(g.NumVertexTypes(), 0));
        std::vector<unsigned> nrProcsPerType(instance.GetArchitecture().GetNumberOfProcessorTypes(), 0);
        for (auto proc = 0u; proc < p; ++proc) {
            ++nrProcsPerType[instance.GetArchitecture().ProcessorType(proc)];
        }

        std::vector<VertexType> nrPredecRemain(n);

        for (VertexType node = 0; node < n; ++node) {
            const auto numParents = g.InDegree(node);

            nrPredecRemain[node] = numParents;

            if (numParents == 0) {
                ready[0].insert(std::make_pair(node, workVariances[node]));
                nrReadyStaleNodesPerType[0][g.VertexType(node)]++;
            }
        }

        std::vector<bool> procFree(p, true);
        unsigned free = p;

        std::set<std::pair<VWorkwT<GraphT>, VertexType>> finishTimes;
        finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

        std::vector<unsigned> numberOfAllocatedAllReadyTasksInSuperstep(instance.GetArchitecture().GetNumberOfProcessorTypes(), 0);
        std::vector<unsigned> limitOfNumberOfAllocatedAllReadyTasksInSuperstep(
            instance.GetArchitecture().GetNumberOfProcessorTypes(), 0);

        bool endSupStep = true;
        bool beginOuterWhile = true;
        bool ableToScheduleInStep = false;
        unsigned successiveEmptySupersteps = 0u;

        auto nonemptyReady = [&]() {
            return std::any_of(
                ready.cbegin(), ready.cend(), [](const std::set<std::pair<VertexType, double>, VarianceCompare> &readySet) {
                    return !readySet.empty();
                });
        };

        while (!oldReady.empty() || nonemptyReady() || !finishTimes.empty()) {
            if (finishTimes.empty() && endSupStep) {
                ableToScheduleInStep = false;
                numberOfAllocatedAllReadyTasksInSuperstep
                    = std::vector<unsigned>(instance.GetArchitecture().GetNumberOfProcessorTypes(), 0);

                for (unsigned i = 0; i < p; ++i) {
                    procReady[supstepIdx % stale][i].clear();
                }

                if (!beginOuterWhile) {
                    supstepIdx++;
                } else {
                    beginOuterWhile = false;
                }

                for (unsigned procType = 0; procType < instance.GetArchitecture().GetNumberOfProcessorTypes(); ++procType) {
                    allReady[procType].clear();
                }

                oldReady.insert(ready[supstepIdx % stale].begin(), ready[supstepIdx % stale].end());
                ready[supstepIdx % stale].clear();
                for (unsigned nodeType = 0; nodeType < g.NumVertexTypes(); ++nodeType) {
                    nrOldReadyNodesPerType[nodeType] += nrReadyStaleNodesPerType[supstepIdx % stale][nodeType];
                    nrReadyStaleNodesPerType[supstepIdx % stale][nodeType] = 0;
                }

                for (const auto &nodeAndValuePair : oldReady) {
                    VertexType node = nodeAndValuePair.first;
                    for (unsigned procType : procTypesCompatibleWithNodeType[g.VertexType(node)]) {
                        allReady[procType].insert(allReady[procType].end(), nodeAndValuePair);
                    }
                }

                if constexpr (useMemoryConstraint_) {
                    if (instance.GetArchitecture().GetMemoryConstraintType() == MemoryConstraintType::LOCAL) {
                        for (unsigned proc = 0; proc < p; proc++) {
                            memoryConstraint_.Reset(proc);
                        }
                    }
                }

                for (unsigned procType = 0; procType < instance.GetArchitecture().GetNumberOfProcessorTypes(); procType++) {
                    unsigned equalSplit = (static_cast<unsigned>(allReady[procType].size()) + stale - 1) / stale;
                    unsigned atLeastForLongStep = 3 * nrProcsPerType[procType];
                    limitOfNumberOfAllocatedAllReadyTasksInSuperstep[procType] = std::max(atLeastForLongStep, equalSplit);
                }

                endSupStep = false;
                finishTimes.emplace(0, std::numeric_limits<VertexType>::max());
            }

            const VWorkwT<GraphT> time = finishTimes.begin()->first;
            const VWorkwT<GraphT> maxFinishTime = finishTimes.rbegin()->first;

            // Find new ready jobs
            while (!finishTimes.empty() && finishTimes.begin()->first == time) {
                const VertexType node = finishTimes.begin()->second;
                finishTimes.erase(finishTimes.begin());

                if (node != std::numeric_limits<VertexType>::max()) {
                    const unsigned procOfNode = schedule.AssignedProcessor(node);

                    for (const auto &succ : g.Children(node)) {
                        nrPredecRemain[succ]--;
                        if (nrPredecRemain[succ] == 0) {
                            ready[supstepIdx % stale].emplace(succ, workVariances[succ]);
                            nrReadyStaleNodesPerType[supstepIdx % stale][g.VertexType(succ)]++;

                            unsigned earliestAdd = supstepIdx;
                            for (const auto &pred : g.Parents(succ)) {
                                if (schedule.AssignedProcessor(pred) != procOfNode) {
                                    earliestAdd = std::max(earliestAdd, stale + schedule.AssignedSuperstep(pred));
                                }
                            }

                            if (instance.IsCompatible(succ, procOfNode)) {
                                bool memoryOk = true;

                                if constexpr (useMemoryConstraint_) {
                                    if (earliestAdd == supstepIdx) {
                                        memoryOk = memoryConstraint_.CanAdd(succ, procOfNode);
                                    }
                                }
                                for (unsigned stepToAdd = earliestAdd; stepToAdd < supstepIdx + stale; ++stepToAdd) {
                                    if ((stepToAdd == supstepIdx) && !memoryOk) {
                                        continue;
                                    }
                                    procReady[stepToAdd % stale][procOfNode].emplace(succ, workVariances[succ]);
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
                       workVariances,
                       allReady,
                       procReady[supstepIdx % stale],
                       procFree,
                       nextNode,
                       nextProc,
                       endSupStep,
                       maxFinishTime - time,
                       procTypesCompatibleWithNodeTypeSkipProctype);

                if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == p) {
                    endSupStep = true;
                    break;
                }

                if (procReady[supstepIdx % stale][nextProc].find(std::make_pair(nextNode, workVariances[nextNode]))
                    != procReady[supstepIdx % stale][nextProc].end()) {
                    for (size_t i = 0; i < stale; i++) {
                        procReady[i][nextProc].erase(std::make_pair(nextNode, workVariances[nextNode]));
                    }
                } else {
                    for (unsigned procType : procTypesCompatibleWithNodeType[g.VertexType(nextNode)]) {
                        allReady[procType].erase(std::make_pair(nextNode, workVariances[nextNode]));
                    }
                    nrOldReadyNodesPerType[g.VertexType(nextNode)]--;
                    const unsigned nextProcType = instance.GetArchitecture().ProcessorType(nextProc);
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

                schedule.SetAssignedProcessor(nextNode, nextProc);
                schedule.SetAssignedSuperstep(nextNode, supstepIdx);
                ableToScheduleInStep = true;

                if constexpr (useMemoryConstraint_) {
                    memoryConstraint_.Add(nextNode, nextProc);

                    std::vector<std::pair<VertexType, double>> toErase;
                    for (const auto &nodePair : procReady[supstepIdx % stale][nextProc]) {
                        if (!memoryConstraint_.CanAdd(nodePair.first, nextProc)) {
                            toErase.push_back(nodePair);
                        }
                    }
                    for (const auto &vert : toErase) {
                        procReady[supstepIdx % stale][nextProc].erase(vert);
                    }
                }

                finishTimes.emplace(time + g.VertexWorkWeight(nextNode), nextNode);
                procFree[nextProc] = false;
                --free;
            }

            if (ableToScheduleInStep) {
                successiveEmptySupersteps = 0;
            } else if (++successiveEmptySupersteps > 100 + stale) {
                return ReturnStatus::ERROR;
            }

            if (free > (p * maxPercentIdleProcessors_)
                && ((!increaseParallelismInNewSuperstep_)
                    || GetNrParallelizableNodes(instance,
                                                stale,
                                                nrOldReadyNodesPerType,
                                                nrReadyStaleNodesPerType[(supstepIdx + 1) % stale],
                                                procReady[(supstepIdx + 1) % stale],
                                                nrProcsPerType)
                           >= std::min(std::min(p, static_cast<unsigned>(1.2 * (p - free))),
                                       p - free + static_cast<unsigned>(0.5 * free)))) {
                endSupStep = true;
            }
        }

        assert(schedule.SatisfiesPrecedenceConstraints());
        // schedule.SetAutoCommunicationSchedule();

        return ReturnStatus::OSP_SUCCESS;
    }

    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override { return ComputeSspSchedule(schedule, 1U); }

    ReturnStatus ComputeSchedule(MaxBspSchedule<GraphT> &schedule) override { return ComputeSspSchedule(schedule, 2U); }

    std::string GetScheduleName() const override {
        if constexpr (useMemoryConstraint_) {
            return "GreedyVarianceSspMemory";
        } else {
            return "GreedyVarianceSsp";
        }
    }
};

}    // namespace osp
