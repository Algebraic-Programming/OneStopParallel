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

#include <boost/heap/fibonacci_heap.hpp>
#include <climits>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "MemoryConstraintModules.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

/**
 * @brief The GreedyBspScheduler class represents a scheduler that uses a greedy algorithm to compute schedules for
 * BspInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "BspGreedy" in this case.
 */
template <typename GraphT, typename MemoryConstraintT = NoMemoryConstraint>
class GreedyBspScheduler : public Scheduler<GraphT> {
    static_assert(IsComputationalDagV<GraphT>, "GreedyBspScheduler can only be used with computational DAGs.");

  private:
    using VertexType = VertexIdxT<GraphT>;

    constexpr static bool useMemoryConstraint_ = IsMemoryConstraintV<MemoryConstraintT>
                                                 or IsMemoryConstraintScheduleV<MemoryConstraintT>;

    static_assert(not useMemoryConstraint_ or std::is_same_v<GraphT, typename MemoryConstraintT::Graph_impl_t>,
                  "Graph_t must be the same as MemoryConstraint_t::Graph_impl_t.");

    MemoryConstraintT memoryConstraint_;

    struct HeapNode {
        VertexType node_;

        double score_;

        HeapNode() : node(0), score_(0) {}

        HeapNode(VertexType nodeArg, double scoreArg) : node(node_arg), score_(scoreArg) {}

        bool operator<(HeapNode const &rhs) const { return (score < rhs.score) || (score <= rhs.score and node < rhs.node); }
    };

    std::vector<boost::heap::fibonacci_heap<HeapNode>> maxProcScoreHeap_;
    std::vector<boost::heap::fibonacci_heap<HeapNode>> maxAllProcScoreHeap_;

    using HeapHandle = typename boost::heap::fibonacci_heap<HeapNode>::handle_type;

    std::vector<std::unordered_map<VertexType, heap_handle>> nodeProcHeapHandles_;
    std::vector<std::unordered_map<VertexType, heap_handle>> nodeAllProcHeapHandles_;

    float maxPercentIdleProcessors_;
    bool increaseParallelismInNewSuperstep_;

    double ComputeScore(VertexType node,
                        unsigned proc,
                        const std::vector<std::vector<bool>> &procInHyperedge,
                        const BspInstance<GraphT> &instance) const {
        double score = 0;
        for (const auto &pred : instance.GetComputationalDag().Parents(node)) {
            if (procInHyperedge[pred][proc]) {
                score += static_cast<double>(instance.GetComputationalDag().VertexCommWeight(pred))
                         / static_cast<double>(instance.GetComputationalDag().OutDegree(pred));
            }
        }
        return score;
    }

    void Choose(const BspInstance<GraphT> &instance,
                const std::vector<std::set<VertexType>> &procReady,
                const std::vector<bool> &procFree,
                VertexType &node,
                unsigned &p) const {
        double maxScore = -1.0;

        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
            if (procFree[proc] && !procReady[proc].empty()) {
                // select node
                HeapNode topNode = maxProcScoreHeap_[proc].top();

                if (topNode.score_ > maxScore) {
                    maxScore = topNode.score_;
                    node = topNode.node_;
                    p = proc;
                    return;
                }
            }
        }

        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
            if (!procFree[proc] or maxAllProcScoreHeap_[proc].empty()) {
                continue;
            }

            HeapNode topNode = maxAllProcScoreHeap_[proc].top();

            if (topNode.score_ > maxScore) {
                if constexpr (useMemoryConstraint_) {
                    if (memoryConstraint_.can_add(topNode.node_, proc)) {
                        maxScore = topNode.score_;
                        node = topNode.node_;
                        p = proc;
                    }

                } else {
                    maxScore = topNode.score_;
                    node = topNode.node_;
                    p = proc;
                }
            }
        }
    };

    bool CanChooseNode(const BspInstance<GraphT> &instance,
                       const std::set<VertexType> &allReady,
                       const std::vector<std::set<VertexType>> &procReady,
                       const std::vector<bool> &procFree) const {
        for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
            if (procFree[i] && !procReady[i].empty()) {
                return true;
            }
        }

        if (!allReady.empty()) {
            for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                if (procFree[i]) {
                    return true;
                }
            }
        }

        return false;
    };

    bool CheckMemFeasibility(const BspInstance<GraphT> &instance,
                             const std::set<VertexType> &allReady,
                             const std::vector<std::set<VertexType>> &procReady) const {
        if constexpr (useMemoryConstraint_) {
            if (instance.GetArchitecture().GetMemoryConstraintType() == MemoryConstraintType::PERSISTENT_AND_TRANSIENT) {
                unsigned numEmptyProc = 0;

                for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                    if (!procReady[i].empty()) {
                        const HeapNode &topNode = maxProcScoreHeap_[i].top();

                        // todo check if this is correct
                        if (memoryConstraint_.can_add(topNode.node_, i)) {
                            return true;
                        }
                    } else {
                        ++numEmptyProc;
                    }
                }

                if (numEmptyProc == instance.NumberOfProcessors() && allReady.empty()) {
                    return true;
                }

                if (!allReady.empty()) {
                    for (unsigned i = 0; i < instance.NumberOfProcessors(); ++i) {
                        const HeapNode &topNode = maxAllProcScoreHeap_[i].top();

                        // todo check if this is correct
                        if (memoryConstraint_.can_add(topNode.node_, i)) {
                            return true;
                        }
                    }
                }

                return false;
            }
        }
        return true;
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
     * @brief Default constructor for GreedyBspScheduler.
     */
    GreedyBspScheduler(float maxPercentIdleProcessors = 0.2f, bool increaseParallelismInNewSuperstep = true)
        : maxPercentIdleProcessors_(maxPercentIdleProcessors),
          increaseParallelismInNewSuperstep_(increaseParallelismInNewSuperstep) {}

    /**
     * @brief Default destructor for GreedyBspScheduler.
     */
    virtual ~GreedyBspScheduler() = default;

    /**
     * @brief Compute a schedule for the given BspInstance.
     *
     * This method computes a schedule for the given BspInstance using a greedy algorithm.
     *
     * @param instance The BspInstance object representing the instance to compute the schedule for.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    ReturnStatus computeSchedule(BspSchedule<GraphT> &schedule) override {
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

        const std::size_t &n = instance.NumberOfVertices();
        const unsigned &paramsP = instance.NumberOfProcessors();
        const auto &g = instance.GetComputationalDag();

        maxProcScoreHeap_ = std::vector<boost::heap::fibonacci_heap<HeapNode>>(paramsP);
        maxAllProcScoreHeap_ = std::vector<boost::heap::fibonacci_heap<HeapNode>>(paramsP);

        node_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);
        node_all_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);

        std::set<VertexType> ready;

        std::vector<std::vector<bool>> procInHyperedge = std::vector<std::vector<bool>>(n, std::vector<bool>(paramsP, false));

        std::vector<std::set<VertexType>> procReady(paramsP);
        std::set<VertexType> allReady;

        std::vector<unsigned> nrPredecDone(n, 0);
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

            for (unsigned proc = 0; proc < paramsP; ++proc) {
                if (instance.isCompatible(v, proc)) {
                    HeapNode newNode(v, 0.0);
                    node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
                }
            }
        }

        bool endSupStep = false;
        while (!ready.empty() || !finishTimes.empty()) {
            if (finishTimes.empty() && endSupStep) {
                for (unsigned proc = 0; proc < paramsP; ++proc) {
                    procReady[proc].clear();
                    maxProcScoreHeap_[proc].clear();
                    node_proc_heap_handles[proc].clear();

                    if constexpr (useMemoryConstraint_) {
                        memoryConstraint_.reset(proc);
                    }
                }

                allReady = ready;

                for (unsigned proc = 0; proc < paramsP; ++proc) {
                    maxAllProcScoreHeap_[proc].clear();
                    node_all_proc_heap_handles[proc].clear();
                }

                for (const auto &v : ready) {
                    for (unsigned proc = 0; proc < params_p; ++proc) {
                        if (!instance.isCompatible(v, proc)) {
                            continue;
                        }

                        double score = computeScore(v, proc, procInHyperedge, instance);
                        heap_node new_node(v, score);
                        node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
                    }
                }

                ++supstepIdx;

                endSupStep = false;

                finishTimes.emplace(0, std::numeric_limits<VertexType>::max());
            }

            const VWorkwT<GraphT> time = finishTimes.begin()->first;

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
                                if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node)
                                    && schedule.assignedSuperstep(pred) == supstepIdx) {
                                    canAdd = false;
                                    break;
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
                                procReady[schedule.assignedProcessor(node)].insert(succ);

                                double score = computeScore(succ, schedule.assignedProcessor(node), procInHyperedge, instance);

                                heap_node new_node(succ, score);
                                node_proc_heap_handles[schedule.assignedProcessor(node)][succ]
                                    = max_proc_score_heap[schedule.assignedProcessor(node)].push(new_node);
                            }
                        }
                    }
                    procFree[schedule.assignedProcessor(node)] = true;
                    ++free;
                }
            }

            if (endSupStep) {
                continue;
            }

            // Assign new jobs to processors
            if (!CanChooseNode(instance, allReady, procReady, procFree)) {
                endSupStep = true;
            }

            while (CanChooseNode(instance, allReady, procReady, procFree)) {
                VertexType nextNode = std::numeric_limits<VertexType>::max();
                unsigned nextProc = instance.NumberOfProcessors();
                Choose(instance, procReady, procFree, nextNode, nextProc);

                if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == instance.NumberOfProcessors()) {
                    endSupStep = true;
                    break;
                }

                if (procReady[nextProc].find(nextNode) != procReady[nextProc].end()) {
                    procReady[nextProc].erase(nextNode);

                    max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][nextNode]);
                    node_proc_heap_handles[nextProc].erase(nextNode);

                } else {
                    allReady.erase(nextNode);

                    for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
                        if (instance.isCompatible(nextNode, proc)) {
                            max_all_proc_score_heap[proc].erase(node_all_proc_heap_handles[proc][nextNode]);
                            node_all_proc_heap_handles[proc].erase(nextNode);
                        }
                    }
                }

                ready.erase(nextNode);
                --nrReadyNodesPerType[g.VertexType(nextNode)];
                schedule.setAssignedProcessor(nextNode, nextProc);
                schedule.setAssignedSuperstep(nextNode, supstepIdx);

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
                        max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][node]);
                        node_proc_heap_handles[nextProc].erase(node);
                    }
                }

                finishTimes.emplace(time + g.VertexWorkWeight(nextNode), nextNode);
                procFree[nextProc] = false;
                --free;

                // update comm auxiliary structure
                procInHyperedge[nextNode][nextProc] = true;

                for (const auto &pred : G.Parents(nextNode)) {
                    if (procInHyperedge[pred][nextProc]) {
                        continue;
                    }

                    procInHyperedge[pred][nextProc] = true;

                    for (const auto &child : G.Children(pred)) {
                        if (child != nextNode && procReady[nextProc].find(child) != procReady[nextProc].end()) {
                            (*node_proc_heap_handles[nextProc][child]).score
                                += static_cast<double>(instance.GetComputationalDag().VertexCommWeight(pred))
                                   / static_cast<double>(instance.GetComputationalDag().OutDegree(pred));
                            max_proc_score_heap[nextProc].update(node_proc_heap_handles[nextProc][child]);
                        }

                        if (child != nextNode && allReady.find(child) != allReady.end() && instance.isCompatible(child, nextProc)) {
                            (*node_all_proc_heap_handles[nextProc][child]).score
                                += static_cast<double>(instance.GetComputationalDag().VertexCommWeight(pred))
                                   / static_cast<double>(instance.GetComputationalDag().OutDegree(pred));
                            max_all_proc_score_heap[nextProc].update(node_all_proc_heap_handles[nextProc][child]);
                        }
                    }
                }
            }

            if constexpr (useMemoryConstraint_) {
                if (not check_mem_feasibility(instance, allReady, procReady)) {
                    return ReturnStatus::ERROR;
                }
            }

            if (free > static_cast<unsigned>(static_cast<float>(paramsP) * maxPercentIdleProcessors_)
                && ((!increaseParallelismInNewSuperstep_)
                    || GetNrParallelizableNodes(instance, nrReadyNodesPerType, nrProcsPerType)
                           >= std::min(std::min(paramsP, static_cast<unsigned>(1.2 * (paramsP - free))),
                                       paramsP - free + (static_cast<unsigned>(0.5 * free))))) {
                endSupStep = true;
            }
        }

        assert(schedule.satisfiesPrecedenceConstraints());

        return ReturnStatus::OSP_SUCCESS;
    };

    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "BspGreedy" in this case.
     *
     * @return The name of the schedule.
     */
    std::string GetScheduleName() const override { return "BspGreedy"; }
};

}    // namespace osp
