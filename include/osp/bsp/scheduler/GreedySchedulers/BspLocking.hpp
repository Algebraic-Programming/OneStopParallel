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
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
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
                  "Graph_t must be the same as MemoryConstraint_t::Graph_impl_t.");

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

        for (auto rIter = top_order.rbegin(); rIter != top_order.crend(); r_iter++) {
            longestPath[*r_iter] = graph.VertexWorkWeight(*r_iter);
            if (graph.OutDegree(*r_iter) > 0) {
                VWorkwT<GraphT> max = 0;
                for (const auto &child : graph.Children(*r_iter)) {
                    if (max <= longest_path[child]) {
                        max = longest_path[child];
                    }
                }
                longestPath[*r_iter] += max;
            }
        }

        return longest_path;
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
            if (locked[succ] < instance.NumberOfProcessors() && locked[succ] != proc) {
                score -= lock_penalty;
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
                        VertexType topNode = max_proc_score_heap[i].top();

                        if (memoryConstraint_.can_add(top_node, i)) {
                            return true;
                        }
                    }
                }

                if (!allReady.empty()) {
                    for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                        VertexType topNode = max_all_proc_score_heap[i].top();

                        if (memoryConstraint_.can_add(top_node, i)) {
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
                VertexType topNode = max_proc_score_heap[proc].top();

                // filling up
                bool procreadyEmpty = false;
                while (endSupStep && (remaining_time < instance.GetComputationalDag().VertexWorkWeight(top_node))) {
                    procReady[proc].erase(top_node);
                    readyPhase_[top_node] = std::numeric_limits<unsigned>::max();
                    max_proc_score_heap[proc].pop();
                    if (!procReady[proc].empty()) {
                        top_node = max_proc_score_heap[proc].top();
                    } else {
                        procreadyEmpty = true;
                        break;
                    }
                }
                if (procreadyEmpty) {
                    continue;
                }

                node = top_node;
                p = proc;
            }
        }

        if (p < instance.NumberOfProcessors()) {
            return true;
        }

        Priority bestPriority = {std::numeric_limits<int>::min(), 0, 0};
        bool foundNode = false;

        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
            if (!procFree[proc] or max_all_proc_score_heap[proc].is_empty()) {
                continue;
            }

            VertexType topNode = max_all_proc_score_heap[proc].top();

            // filling up
            bool allProcreadyEmpty = false;
            while (endSupStep && (remaining_time < instance.GetComputationalDag().VertexWorkWeight(top_node))) {
                allReady.erase(top_node);
                for (unsigned procDel = 0; procDel < instance.NumberOfProcessors(); procDel++) {
                    if (procDel == proc || !instance.isCompatible(top_node, procDel)) {
                        continue;
                    }
                    max_all_proc_score_heap[proc_del].erase(top_node);
                }
                max_all_proc_score_heap[proc].pop();
                readyPhase_[top_node] = std::numeric_limits<unsigned>::max();
                if (!max_all_proc_score_heap[proc].is_empty()) {
                    top_node = max_all_proc_score_heap[proc].top();
                } else {
                    allProcreadyEmpty = true;
                    break;
                }
            }
            if (allProcreadyEmpty) {
                continue;
            }

            Priority topPriority = max_all_proc_score_heap[proc].get_value(top_node);
            if (!foundNode || PriorityCompare{}(top_priority, best_priority)) {
                if constexpr (useMemoryConstraint_) {
                    if (memoryConstraint_.can_add(top_node, proc)) {
                        bestPriority = top_priority;
                        node = top_node;
                        p = proc;
                        foundNode = true;
                    }

                } else {
                    bestPriority = top_priority;
                    node = top_node;
                    p = proc;
                    foundNode = true;
                }
            }
        }
        return (foundNode && std::get<0>(best_priority) > -3);
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
            if (procFree[i] && !max_all_proc_score_heap[i].is_empty()) {
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
    virtual ReturnStatus computeSchedule(BspSchedule<GraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();

        for (const auto &v : instance.GetComputationalDag().vertices()) {
            schedule.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
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

        const std::vector<VWorkwT<GraphT>> pathLength = get_longest_path(g);
        VWorkwT<GraphT> maxPath = 1;
        for (const auto &i : instance.vertices()) {
            if (pathLength[i] > max_path) {
                maxPath = path_length[i];
            }
        }

        defaultValue_.clear();
        defaultValue_.resize(n, 0);
        for (const auto &i : instance.vertices()) {
            // assert(path_length[i] * 20 / max_path <= std::numeric_limits<int>::max());
            defaultValue_[i] = static_cast<int>(path_length[i] * static_cast<VWorkwT<GraphT>>(20) / max_path);
        }

        max_proc_score_heap = std::vector<MaxHeap>(params_p);
        max_all_proc_score_heap = std::vector<MaxHeap>(params_p);

        locked_set.clear();
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
        std::vector<unsigned> nrProcsPerType(instance.GetArchitecture().getNumberOfProcessorTypes(), 0);
        for (unsigned proc = 0; proc < paramsP; ++proc) {
            ++nrProcsPerType[instance.GetArchitecture().processorType(proc)];
        }

        std::set<std::pair<VWorkwT<GraphT>, VertexType>> finishTimes;
        finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

        for (const auto &v : source_vertices_view(g)) {
            ready.insert(v);
            allReady.insert(v);
            ++nrReadyNodesPerType[g.VertexType(v)];
            readyPhase_[v] = paramsP;

            for (unsigned proc = 0; proc < paramsP; ++proc) {
                if (instance.isCompatible(v, proc)) {
                    Priority priority = {defaultValue_[v], static_cast<unsigned>(g.OutDegree(v)), v};
                    max_all_proc_score_heap[proc].push(v, priority);
                }
            }
        }

        bool endSupStep = false;

        while (!ready.empty() || !finishTimes.empty()) {
            if (finishTimes.empty() && endSupStep) {
                for (unsigned proc = 0; proc < paramsP; ++proc) {
                    procReady[proc].clear();
                    max_proc_score_heap[proc].clear();

                    if constexpr (useMemoryConstraint_) {
                        memoryConstraint_.reset(proc);
                    }
                }

                allReady = ready;

                for (const auto &node : locked_set) {
                    locked[node] = std::numeric_limits<unsigned>::max();
                }
                locked_set.clear();

                for (unsigned proc = 0; proc < paramsP; ++proc) {
                    max_all_proc_score_heap[proc].clear();
                }

                for (const auto &v : ready) {
                    ready_phase[v] = params_p;
                    for (unsigned proc = 0; proc < params_p; ++proc) {
                        if (!instance.isCompatible(v, proc)) {
                            continue;
                        }

                        int score = computeScore(v, proc, instance);
                        Priority priority = {score, static_cast<unsigned>(G.OutDegree(v)), v};
                        max_all_proc_score_heap[proc].push(v, priority);
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
                    for (const auto &succ : G.Children(node)) {
                        ++nrPredecDone[succ];
                        if (nrPredecDone[succ] == G.in_degree(succ)) {
                            ready.insert(succ);
                            ++nr_ready_nodes_per_type[G.VertexType(succ)];

                            bool canAdd = true;
                            for (const auto &pred : G.Parents(succ)) {
                                if (schedule.AssignedProcessor(pred) != schedule.AssignedProcessor(node)
                                    && schedule.AssignedSuperstep(pred) == supstepIdx) {
                                    canAdd = false;
                                    break;
                                }
                            }

                            if constexpr (use_memory_constraint) {
                                if (canAdd) {
                                    if (not memory_constraint.can_add(succ, schedule.AssignedProcessor(node))) {
                                        canAdd = false;
                                    }
                                }
                            }

                            if (!instance.isCompatible(succ, schedule.AssignedProcessor(node))) {
                                canAdd = false;
                            }

                            if (canAdd) {
                                procReady[schedule.AssignedProcessor(node)].insert(succ);
                                ready_phase[succ] = schedule.AssignedProcessor(node);

                                int score = computeScore(succ, schedule.AssignedProcessor(node), instance);
                                Priority priority = {score, static_cast<unsigned>(G.OutDegree(succ)), succ};

                                max_proc_score_heap[schedule.AssignedProcessor(node)].push(succ, priority);
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
                Choose(instance, allReady, procReady, procFree, nextNode, nextProc, endSupStep, max_finish_time - time);

                if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == instance.NumberOfProcessors()) {
                    endSupStep = true;
                    break;
                }

                if (readyPhase_[nextNode] < paramsP) {
                    procReady[nextProc].erase(nextNode);

                    max_proc_score_heap[nextProc].erase(nextNode);

                } else {
                    allReady.erase(nextNode);

                    for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
                        if (instance.isCompatible(nextNode, proc) && max_all_proc_score_heap[proc].contains(nextNode)) {
                            max_all_proc_score_heap[proc].erase(nextNode);
                        }
                    }
                }

                ready.erase(nextNode);
                --nrReadyNodesPerType[g.VertexType(nextNode)];
                schedule.setAssignedProcessor(nextNode, nextProc);
                schedule.setAssignedSuperstep(nextNode, supstepIdx);

                readyPhase_[nextNode] = std::numeric_limits<unsigned>::max();

                if constexpr (useMemoryConstraint_) {
                    memoryConstraint_.add(nextNode, nextProc);

                    std::vector<VertexType> toErase;
                    for (const auto &node : procReady[nextProc]) {
                        if (not memory_constraint.can_add(node, nextProc)) {
                            toErase.push_back(node);
                        }
                    }

                    for (const auto &node : toErase) {
                        procReady[nextProc].erase(node);
                        max_proc_score_heap[nextProc].erase(node);
                        ready_phase[node] = std::numeric_limits<unsigned>::max();
                    }
                }

                finishTimes.emplace(time + g.VertexWorkWeight(nextNode), nextNode);
                procFree[nextProc] = false;
                --free;

                // update auxiliary structures

                for (const auto &succ : G.Children(nextNode)) {
                    if (locked[succ] < params_p && locked[succ] != nextProc) {
                        for (const auto &parent : G.Parents(succ)) {
                            if (ready_phase[parent] < std::numeric_limits<unsigned>::max() && ready_phase[parent] < params_p
                                && ready_phase[parent] != locked[succ]) {
                                Priority p = max_proc_score_heap[ready_phase[parent]].get_value(parent);
                                std::get<0>(p) += lock_penalty;
                                max_proc_score_heap[ready_phase[parent]].update(parent, p);
                            }
                            if (ready_phase[parent] == params_p) {
                                for (unsigned proc = 0; proc < params_p; ++proc) {
                                    if (proc == locked[succ] || !instance.isCompatible(parent, proc)) {
                                        continue;
                                    }

                                    if (max_all_proc_score_heap[proc].contains(parent)) {
                                        Priority p = max_all_proc_score_heap[proc].get_value(parent);
                                        std::get<0>(p) += lock_penalty;
                                        max_all_proc_score_heap[proc].update(parent, p);
                                    }
                                }
                            }
                        }
                        locked[succ] = params_p;
                    } else if (locked[succ] == std::numeric_limits<unsigned>::max()) {
                        locked_set.push_back(succ);
                        locked[succ] = nextProc;

                        for (const auto &parent : G.Parents(succ)) {
                            if (ready_phase[parent] < std::numeric_limits<unsigned>::max() && ready_phase[parent] < params_p
                                && ready_phase[parent] != nextProc) {
                                Priority p = max_proc_score_heap[ready_phase[parent]].get_value(parent);
                                std::get<0>(p) -= lock_penalty;
                                max_proc_score_heap[ready_phase[parent]].update(parent, p);
                            }
                            if (ready_phase[parent] == params_p) {
                                for (unsigned proc = 0; proc < params_p; ++proc) {
                                    if (proc == nextProc || !instance.isCompatible(parent, proc)) {
                                        continue;
                                    }

                                    if (max_all_proc_score_heap[proc].contains(parent)) {
                                        Priority p = max_all_proc_score_heap[proc].get_value(parent);
                                        std::get<0>(p) -= lock_penalty;
                                        max_all_proc_score_heap[proc].update(parent, p);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if constexpr (useMemoryConstraint_) {
                if (not check_mem_feasibility(instance, allReady, procReady)) {
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

        assert(schedule.satisfiesPrecedenceConstraints());

        return ReturnStatus::OSP_SUCCESS;
    }

    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "BspGreedy" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override {
        if (useMemoryConstraint_) {
            return "BspGreedyLockingMemory";
        } else {
            return "BspGreedyLocking";
        }
    }

    void SetMaxPercentIdleProcessors(float maxPercentIdleProcessors) { maxPercentIdleProcessors_ = maxPercentIdleProcessors; }
};

}    // namespace osp
