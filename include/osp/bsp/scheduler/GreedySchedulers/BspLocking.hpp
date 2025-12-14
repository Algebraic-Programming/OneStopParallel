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

#include "MemoryConstraintModules.hpp"
#include "osp/auxiliary/datastructures/heaps/PairingHeap.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

/**
 * @brief The GreedyBspLocking class represents a scheduler that uses a greedy algorithm to compute schedules for
 * BspInstance.
 *
 * This class inherits from the Scheduler class and implements the ComputeSchedule() and getScheduleName() methods.
 * The ComputeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "BspGreedy" in this case.
 */

template <typename GraphT, typename MemoryConstraintT = NoMemoryConstraint>
class BspLocking : public Scheduler<GraphT> {
    static_assert(IsComputationalDagV<GraphT>, "BspLocking can only be used with computational DAGs.");

  private:
    using VertexType = VertexIdxT<GraphT>;

    constexpr static bool useMemoryConstraint_ = IsMemoryConstraintV<MemoryConstraintT>
                                                 or IsMemoryConstraintScheduleV<MemoryConstraintT>;

    static_assert(not useMemoryConstraint_ or std::is_same_v<GraphT, typename MemoryConstraintT::Graph_impl_t>,
                  "GraphT must be the same as MemoryConstraintT::Graph_impl_t.");

    MemoryConstraintT memoryConstraint_;

    using Priority = std::tuple<int, unsigned, VertexType>;

    struct PriorityCompare {
        bool operator()(const Priority &a, const Priority &b) const {
            if (std::get<0>(a) != std::get<0>(b)) {
                return std::get<0>(a) > std::get<0>(b);    // Higher score is better
            }
            if (std::get<1>(a) != std::get<1>(b)) {
                return std::get<1>(a) > std::get<1>(b);    // Higher secondary_score is better
            }
            return std::get<2>(a) < std::get<2>(b);    // Smaller node index is better for tie-breaking
        }
    };

    using MaxHeap = PairingHeap<VertexType, Priority, PriorityCompare>;

    std::vector<MaxHeap> maxProcScoreHeap_;
    std::vector<MaxHeap> maxAllProcScoreHeap_;

    static std::vector<VWorkwT<GraphT>> GetLongestPath(const GraphT &graph) {
        std::vector<VWorkwT<GraphT>> longestPath(graph.NumVertices(), 0);

        const std::vector<VertexType> topOrder = GetTopOrder(graph);

        for (auto rIter = topOrder.rbegin(); rIter != topOrder.crend(); rIter++) {
            longestPath[*rIter] = graph.VertexWorkWeight(*rIter);
            if (graph.OutDegree(*rIter) > 0) {
                VWorkwT<GraphT> max = 0;
                for (const auto &child : graph.Children(*rIter)) {
                    if (max <= longestPath[child]) {
                        max = longestPath[child];
                    }
                }
                longestPath[*rIter] += max;
            }
        }

        return longestPath;
    }

    std::deque<VertexType> lockedSet_;
    std::vector<unsigned> locked_;
    int lockPenalty_ = 1;
    std::vector<unsigned> readyPhase_;

    std::vector<int> defaultValue_;

    double maxPercentIdleProcessors_;
    bool increaseParallelismInNewSuperstep_;

    int ComputeScore(VertexType node, unsigned proc, const BspInstance<GraphT> &instance) {
        int score = 0;
        for (const auto &succ : instance.GetComputationalDag().Children(node)) {
            if (locked_[succ] < instance.NumberOfProcessors() && locked_[succ] != proc) {
                score -= lockPenalty_;
            }
        }

        return score + defaultValue_[node];
    };

    bool CheckMemFeasibility(const BspInstance<GraphT> &instance,
                             const std::set<VertexType> &allReady,
                             const std::vector<std::set<VertexType>> &procReady) const {
        if constexpr (useMemoryConstraint_) {
            if (instance.GetArchitecture().GetMemoryConstraintType() == MemoryConstraintType::PERSISTENT_AND_TRANSIENT) {
                for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                    if (!procReady[i].empty()) {
                        VertexType topNode = maxProcScoreHeap_[i].top();

                        if (memoryConstraint_.can_add(topNode, i)) {
                            return true;
                        }
                    }
                }

                if (!allReady.empty()) {
                    for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                        VertexType topNode = maxAllProcScoreHeap_[i].top();

                        if (memoryConstraint_.can_add(topNode, i)) {
                            return true;
                        }
                    }
                }

                return false;
            }
        }

        return true;
    }

    bool Choose(const BspInstance<GraphT> &instance,
                std::set<VertexType> &allReady,
                std::vector<std::set<VertexType>> &procReady,
                const std::vector<bool> &procFree,
                VertexType &node,
                unsigned &p,
                const bool endSupStep,
                const VWorkwT<GraphT> remainingTime) {
        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
            if (procFree[proc] && !procReady[proc].empty()) {
                // select node
                VertexType topNode = maxProcScoreHeap_[proc].top();

                // filling up
                bool procReadyEmpty = false;
                while (endSupStep && (remainingTime < instance.GetComputationalDag().VertexWorkWeight(topNode))) {
                    procReady[proc].erase(topNode);
                    readyPhase_[topNode] = std::numeric_limits<unsigned>::max();
                    maxProcScoreHeap_[proc].pop();
                    if (!procReady[proc].empty()) {
                        topNode = maxProcScoreHeap_[proc].top();
                    } else {
                        procReadyEmpty = true;
                        break;
                    }
                }
                if (procReadyEmpty) {
                    continue;
                }

                node = topNode;
                p = proc;
            }
        }

        if (p < instance.NumberOfProcessors()) {
            return true;
        }

        Priority bestPriority = {std::numeric_limits<int>::min(), 0, 0};
        bool foundNode = false;

        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
            if (!procFree[proc] or maxAllProcScoreHeap_[proc].is_empty()) {
                continue;
            }

            VertexType topNode = maxAllProcScoreHeap_[proc].top();

            // filling up
            bool allProcreadyEmpty = false;
            while (endSupStep && (remainingTime < instance.GetComputationalDag().VertexWorkWeight(topNode))) {
                allReady.erase(topNode);
                for (unsigned procDel = 0; procDel < instance.NumberOfProcessors(); procDel++) {
                    if (procDel == proc || !instance.IsCompatible(topNode, procDel)) {
                        continue;
                    }
                    maxAllProcScoreHeap_[procDel].erase(topNode);
                }
                maxAllProcScoreHeap_[proc].pop();
                readyPhase_[topNode] = std::numeric_limits<unsigned>::max();
                if (!maxAllProcScoreHeap_[proc].is_empty()) {
                    topNode = maxAllProcScoreHeap_[proc].top();
                } else {
                    allProcreadyEmpty = true;
                    break;
                }
            }
            if (allProcreadyEmpty) {
                continue;
            }

            Priority topPriority = maxAllProcScoreHeap_[proc].get_value(topNode);
            if (!foundNode || PriorityCompare{}(topPriority, bestPriority)) {
                if constexpr (useMemoryConstraint_) {
                    if (memoryConstraint_.can_add(topNode, proc)) {
                        bestPriority = topPriority;
                        node = topNode;
                        p = proc;
                        foundNode = true;
                    }

                } else {
                    bestPriority = topPriority;
                    node = topNode;
                    p = proc;
                    foundNode = true;
                }
            }
        }
        return (foundNode && std::get<0>(bestPriority) > -3);
    }

    bool CanChooseNode(const BspInstance<GraphT> &instance,
                       const std::vector<std::set<VertexType>> &procReady,
                       const std::vector<bool> &procFree) const {
        for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
            if (procFree[i] && !procReady[i].empty()) {
                return true;
            }
        }

        for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
            if (procFree[i] && !maxAllProcScoreHeap_[i].is_empty()) {
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
        for (unsigned procType = 0; procType < instance.GetArchitecture().GetNumberOfProcessorTypes(); ++procType) {
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
     * @brief Default constructor for GreedyBspLocking.
     */
    BspLocking(float maxPercentIdleProcessors = 0.4f, bool increaseParallelismInNewSuperstep = true)
        : maxPercentIdleProcessors_(maxPercentIdleProcessors),
          increaseParallelismInNewSuperstep_(increaseParallelismInNewSuperstep) {}

    /**
     * @brief Default destructor for GreedyBspLocking.
     */
    virtual ~BspLocking() = default;

    /**
     * @brief Compute a schedule for the given BspInstance.
     *
     * This method computes a schedule for the given BspInstance using a greedy algorithm.
     *
     * @param instance The BspInstance object representing the instance to compute the schedule for.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();

        for (const auto &v : instance.GetComputationalDag().Vertices()) {
            schedule.SetAssignedProcessor(v, std::numeric_limits<unsigned>::max());
        }

        unsigned supstepIdx = 0;

        if constexpr (IsMemoryConstraintV<MemoryConstraintT>) {
            memoryConstraint_.initialize(instance);
        } else if constexpr (IsMemoryConstraintScheduleV<MemoryConstraintT>) {
            memoryConstraint_.initialize(schedule, supstepIdx);
        }

        const auto &n = instance.NumberOfVertices();
        const unsigned &paramsP = instance.NumberOfProcessors();
        const auto &g = instance.GetComputationalDag();

        const std::vector<VWorkwT<GraphT>> pathLength = GetLongestPath(g);
        VWorkwT<GraphT> maxPath = 1;
        for (const auto &i : instance.Vertices()) {
            if (pathLength[i] > maxPath) {
                maxPath = pathLength[i];
            }
        }

        defaultValue_.clear();
        defaultValue_.resize(n, 0);
        for (const auto &i : instance.Vertices()) {
            defaultValue_[i] = static_cast<int>(pathLength[i] * static_cast<VWorkwT<GraphT>>(20) / maxPath);
        }

        maxProcScoreHeap_.clear();
        maxProcScoreHeap_.resize(paramsP);
        maxAllProcScoreHeap_.clear();
        maxAllProcScoreHeap_.resize(paramsP);

        lockedSet_.clear();
        locked_.clear();
        locked_.resize(n, std::numeric_limits<unsigned>::max());

        std::set<VertexType> ready;
        readyPhase_.clear();
        readyPhase_.resize(n, std::numeric_limits<unsigned>::max());

        std::vector<std::set<VertexType>> procReady(paramsP);
        std::set<VertexType> allReady;

        std::vector<VertexType> nrPredecDone(n, 0);
        std::vector<bool> procFree(paramsP, true);
        unsigned free = paramsP;

        std::vector<unsigned> nrReadyNodesPerType(g.NumVertexTypes(), 0);
        std::vector<unsigned> nrProcsPerType(instance.GetArchitecture().GetNumberOfProcessorTypes(), 0);
        for (unsigned proc = 0; proc < paramsP; ++proc) {
            ++nrProcsPerType[instance.GetArchitecture().ProcessorType(proc)];
        }

        std::set<std::pair<VWorkwT<GraphT>, VertexType>> finishTimes;
        finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

        for (const auto &v : SourceVerticesView(g)) {
            ready.insert(v);
            allReady.insert(v);
            ++nrReadyNodesPerType[g.VertexType(v)];
            readyPhase_[v] = paramsP;

            for (unsigned proc = 0; proc < paramsP; ++proc) {
                if (instance.IsCompatible(v, proc)) {
                    Priority priority = {defaultValue_[v], static_cast<unsigned>(g.OutDegree(v)), v};
                    maxAllProcScoreHeap_[proc].push(v, priority);
                }
            }
        }

        bool endSupStep = false;

        while (!ready.empty() || !finishTimes.empty()) {
            if (finishTimes.empty() && endSupStep) {
                for (unsigned proc = 0; proc < paramsP; ++proc) {
                    procReady[proc].clear();
                    maxProcScoreHeap_[proc].clear();

                    if constexpr (useMemoryConstraint_) {
                        memoryConstraint_.reset(proc);
                    }
                }

                allReady = ready;

                for (const auto &node : lockedSet_) {
                    locked_[node] = std::numeric_limits<unsigned>::max();
                }
                lockedSet_.clear();

                for (unsigned proc = 0; proc < paramsP; ++proc) {
                    maxAllProcScoreHeap_[proc].clear();
                }

                for (const auto &v : ready) {
                    readyPhase_[v] = paramsP;
                    for (unsigned proc = 0; proc < paramsP; ++proc) {
                        if (!instance.IsCompatible(v, proc)) {
                            continue;
                        }

                        int score = ComputeScore(v, proc, instance);
                        Priority priority = {score, static_cast<unsigned>(g.OutDegree(v)), v};
                        maxAllProcScoreHeap_[proc].push(v, priority);
                    }
                }

                ++supstepIdx;

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
                    for (const auto &succ : g.Children(node)) {
                        ++nrPredecDone[succ];
                        if (nrPredecDone[succ] == g.InDegree(succ)) {
                            ready.insert(succ);
                            ++nrReadyNodesPerType[g.VertexType(succ)];

                            bool canAdd = true;
                            for (const auto &pred : g.Parents(succ)) {
                                if (schedule.AssignedProcessor(pred) != schedule.AssignedProcessor(node)
                                    && schedule.AssignedSuperstep(pred) == supstepIdx) {
                                    canAdd = false;
                                    break;
                                }
                            }

                            if constexpr (useMemoryConstraint_) {
                                if (canAdd) {
                                    if (not memoryConstraint_.CanAdd(succ, schedule.AssignedProcessor(node))) {
                                        canAdd = false;
                                    }
                                }
                            }

                            if (!instance.IsCompatible(succ, schedule.AssignedProcessor(node))) {
                                canAdd = false;
                            }

                            if (canAdd) {
                                procReady[schedule.AssignedProcessor(node)].insert(succ);
                                readyPhase_[succ] = schedule.AssignedProcessor(node);

                                int score = ComputeScore(succ, schedule.AssignedProcessor(node), instance);
                                Priority priority = {score, static_cast<unsigned>(g.OutDegree(succ)), succ};

                                maxProcScoreHeap_[schedule.AssignedProcessor(node)].push(succ, priority);
                            }
                        }
                    }
                    procFree[schedule.AssignedProcessor(node)] = true;
                    ++free;
                }
            }

            // Assign new jobs to processors
            if (!CanChooseNode(instance, procReady, procFree)) {
                endSupStep = true;
            }

            while (CanChooseNode(instance, procReady, procFree)) {
                VertexType nextNode = std::numeric_limits<VertexType>::max();
                unsigned nextProc = instance.NumberOfProcessors();
                Choose(instance, allReady, procReady, procFree, nextNode, nextProc, endSupStep, maxFinishTime - time);

                if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == instance.NumberOfProcessors()) {
                    endSupStep = true;
                    break;
                }

                if (readyPhase_[nextNode] < paramsP) {
                    procReady[nextProc].erase(nextNode);

                    maxProcScoreHeap_[nextProc].erase(nextNode);

                } else {
                    allReady.erase(nextNode);

                    for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
                        if (instance.IsCompatible(nextNode, proc) && maxAllProcScoreHeap_[proc].contains(nextNode)) {
                            maxAllProcScoreHeap_[proc].erase(nextNode);
                        }
                    }
                }

                ready.erase(nextNode);
                --nrReadyNodesPerType[g.VertexType(nextNode)];
                schedule.SetAssignedProcessor(nextNode, nextProc);
                schedule.SetAssignedSuperstep(nextNode, supstepIdx);

                readyPhase_[nextNode] = std::numeric_limits<unsigned>::max();

                if constexpr (useMemoryConstraint_) {
                    memoryConstraint_.Add(nextNode, nextProc);

                    std::vector<VertexType> toErase;
                    for (const auto &node : procReady[nextProc]) {
                        if (not memoryConstraint_.CanAdd(node, nextProc)) {
                            toErase.push_back(node);
                        }
                    }

                    for (const auto &node : toErase) {
                        procReady[nextProc].erase(node);
                        maxProcScoreHeap_[nextProc].erase(node);
                        readyPhase_[node] = std::numeric_limits<unsigned>::max();
                    }
                }

                finishTimes.emplace(time + g.VertexWorkWeight(nextNode), nextNode);
                procFree[nextProc] = false;
                --free;

                // update auxiliary structures

                for (const auto &succ : g.Children(nextNode)) {
                    if (locked_[succ] < paramsP && locked_[succ] != nextProc) {
                        for (const auto &parent : g.Parents(succ)) {
                            if (readyPhase_[parent] < std::numeric_limits<unsigned>::max() && readyPhase_[parent] < paramsP
                                && readyPhase_[parent] != locked_[succ]) {
                                Priority p = maxProcScoreHeap_[readyPhase_[parent]].get_value(parent);
                                std::get<0>(p) += lockPenalty_;
                                maxProcScoreHeap_[readyPhase_[parent]].update(parent, p);
                            }
                            if (readyPhase_[parent] == paramsP) {
                                for (unsigned proc = 0; proc < paramsP; ++proc) {
                                    if (proc == locked_[succ] || !instance.IsCompatible(parent, proc)) {
                                        continue;
                                    }

                                    if (maxAllProcScoreHeap_[proc].contains(parent)) {
                                        Priority p = maxAllProcScoreHeap_[proc].get_value(parent);
                                        std::get<0>(p) += lockPenalty_;
                                        maxAllProcScoreHeap_[proc].update(parent, p);
                                    }
                                }
                            }
                        }
                        locked_[succ] = paramsP;
                    } else if (locked_[succ] == std::numeric_limits<unsigned>::max()) {
                        lockedSet_.push_back(succ);
                        locked_[succ] = nextProc;

                        for (const auto &parent : g.Parents(succ)) {
                            if (readyPhase_[parent] < std::numeric_limits<unsigned>::max() && readyPhase_[parent] < paramsP
                                && readyPhase_[parent] != nextProc) {
                                Priority p = maxProcScoreHeap_[readyPhase_[parent]].get_value(parent);
                                std::get<0>(p) -= lockPenalty_;
                                maxProcScoreHeap_[readyPhase_[parent]].update(parent, p);
                            }
                            if (readyPhase_[parent] == paramsP) {
                                for (unsigned proc = 0; proc < paramsP; ++proc) {
                                    if (proc == nextProc || !instance.IsCompatible(parent, proc)) {
                                        continue;
                                    }

                                    if (maxAllProcScoreHeap_[proc].contains(parent)) {
                                        Priority p = maxAllProcScoreHeap_[proc].get_value(parent);
                                        std::get<0>(p) -= lockPenalty_;
                                        maxAllProcScoreHeap_[proc].update(parent, p);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if constexpr (useMemoryConstraint_) {
                if (not CheckMemFeasibility(instance, allReady, procReady)) {
                    return ReturnStatus::ERROR;
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

        assert(schedule.SatisfiesPrecedenceConstraints());

        return ReturnStatus::OSP_SUCCESS;
    }

    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "BspGreedy" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string GetScheduleName() const override {
        if (useMemoryConstraint_) {
            return "BspGreedyLockingMemory";
        } else {
            return "BspGreedyLocking";
        }
    }

    void SetMaxPercentIdleProcessors(float maxPercentIdleProcessors) { maxPercentIdleProcessors_ = maxPercentIdleProcessors; }
};

}    // namespace osp
