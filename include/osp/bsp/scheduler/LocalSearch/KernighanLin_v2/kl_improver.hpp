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
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "kl_active_schedule.hpp"
#include "kl_util.hpp"
#include "osp/auxiliary/datastructures/heaps/PairingHeap.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/model/util/CompatibleProcessorRange.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

struct KlParameter {
    double timeQuality = 0.8;
    double superstepRemoveStrength = 0.5;
    unsigned numParallelLoops = 4;

    unsigned maxInnerIterationsReset = 500;
    unsigned maxNoImprovementIterations = 50;

    constexpr static unsigned abortScatterNodesViolationThreshold = 500;
    constexpr static unsigned initialViolationThreshold = 250;

    unsigned maxNoVioaltionsRemovedBacktrackReset;
    unsigned removeStepEpocs;
    unsigned nodeMaxStepSelectionEpochs;
    unsigned maxNoVioaltionsRemovedBacktrackForRemoveStepReset;
    unsigned maxOuterIterations;
    unsigned tryRemoveStepAfterNumOuterIterations;
    unsigned minInnerIterReset;

    unsigned threadMinRange = 8;
    unsigned threadRangeGap = 0;
};

template <typename VertexType>
struct KlUpdateInfo {
    VertexType node = 0;

    bool fullUpdate = false;
    bool updateFromStep = false;
    bool updateToStep = false;
    bool updateEntireToStep = false;
    bool updateEntireFromStep = false;

    KlUpdateInfo() = default;

    KlUpdateInfo(VertexType n) : node(n), fullUpdate(false), updateEntireToStep(false), updateEntireFromStep(false) {}

    KlUpdateInfo(VertexType n, bool full) : node(n), fullUpdate(full), updateEntireToStep(false), updateEntireFromStep(false) {}
};

template <typename GraphT,
          typename CommCostFunctionT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned WindowSize = 1,
          typename CostT = double>
class KlImprover : public ImprovementScheduler<GraphT> {
    static_assert(isDirectedGraphEdgeDescV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    static_assert(hasHashableEdgeDescV<GraphT>, "Graph_t must satisfy the has_hashable_edge_desc concept");
    static_assert(isComputationalDagV<GraphT>, "Graph_t must satisfy the computational_dag concept");

  protected:
    constexpr static unsigned windowRange = 2 * WindowSize + 1;
    constexpr static bool enableQuickMoves = true;
    constexpr static bool enablePreresolvingViolations = true;
    constexpr static double epsilon = 1e-9;

    using MemwT = VMemwT<GraphT>;
    using CommwT = VCommwT<GraphT>;
    using WorkWeightT = VWorkwT<GraphT>;
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    using KlMove = KlMoveStruct<CostT, VertexType>;
    using HeapDatastructure = MaxPairingHeap<VertexType, KlMove>;
    using ActiveScheduleT = KlActiveSchedule<GraphT, CostT, MemoryConstraintT>;
    using NodeSelectionContainerT = AdaptiveAffinityTable<GraphT, CostT, ActiveScheduleT, WindowSize>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;

    struct ThreadSearchContext {
        unsigned threadId = 0;
        unsigned startStep = 0;
        unsigned endStep = 0;
        unsigned originalEndStep = 0;

        VectorVertexLockManger<VertexType> lockManager;
        HeapDatastructure maxGainHeap;
        NodeSelectionContainerT affinityTable;
        std::vector<std::vector<CostT>> localAffinityTable;
        RewardPenaltyStrategy<CostT, CommCostFunctionT, ActiveScheduleT> rewardPenaltyStrat;
        VertexSelectionStrategy<GraphT, NodeSelectionContainerT, ActiveScheduleT> selectionStrategy;
        ThreadLocalActiveScheduleData<GraphT, CostT> activeScheduleData;

        double averageGain = 0.0;
        unsigned maxInnerIterations = 0;
        unsigned noImprovementIterationsReducePenalty = 0;
        unsigned minInnerIter = 0;
        unsigned noImprovementIterationsIncreaseInnerIter = 0;
        unsigned stepSelectionEpochCounter = 0;
        unsigned stepSelectionCounter = 0;
        unsigned stepToRemove = 0;
        unsigned localSearchStartStep = 0;
        unsigned unlockEdgeBacktrackCounter = 0;
        unsigned unlockEdgeBacktrackCounterReset = 0;
        unsigned maxNoVioaltionsRemovedBacktrack = 0;

        inline unsigned NumSteps() const { return endStep - startStep + 1; }

        inline unsigned StartIdx(const unsigned nodeStep) const {
            return nodeStep < startStep + WindowSize ? WindowSize - (nodeStep - startStep) : 0;
        }

        inline unsigned EndIdx(unsigned nodeStep) const {
            return nodeStep + WindowSize <= endStep ? windowRange : windowRange - (nodeStep + WindowSize - endStep);
        }
    };

    bool computeWithTimeLimit_ = false;

    BspSchedule<GraphT> *inputSchedule_;
    const GraphT *graph_;
    const BspInstance<GraphT> *instance_;

    CompatibleProcessorRange<GraphT> procRange_;

    KlParameter parameters_;
    std::mt19937 gen_;

    ActiveScheduleT activeSchedule_;
    CommCostFunctionT commCostF_;
    std::vector<ThreadSearchContext> threadDataVec_;
    std::vector<bool> threadFinishedVec_;

    inline unsigned RelStepIdx(const unsigned nodeStep, const unsigned moveStep) const {
        return (moveStep >= nodeStep) ? ((moveStep - nodeStep) + WindowSize) : (WindowSize - (nodeStep - moveStep));
    }

    inline bool IsCompatible(VertexType node, unsigned proc) const {
        return activeSchedule_.GetInstance().IsCompatible(node, proc);
    }

    void SetStartStep(const unsigned step, ThreadSearchContext &threadData) {
        threadData.startStep = step;
        threadData.stepToRemove = step;
        threadData.stepSelectionCounter = step;

        threadData.averageGain = 0.0;
        threadData.maxInnerIterations = parameters_.maxInnerIterationsReset;
        threadData.noImprovementIterationsReducePenalty = parameters_.maxNoImprovementIterations / 5;
        threadData.minInnerIter = parameters_.minInnerIterReset;
        threadData.stepSelectionEpochCounter = 0;
        threadData.noImprovementIterationsIncreaseInnerIter = 10;
        threadData.unlockEdgeBacktrackCounterReset = 0;
        threadData.unlockEdgeBacktrackCounter = threadData.unlockEdgeBacktrackCounterReset;
        threadData.maxNoVioaltionsRemovedBacktrack = parameters_.maxNoVioaltionsRemovedBacktrackReset;
    }

    KlMove GetBestMove(NodeSelectionContainerT &affinityTable,
                       VectorVertexLockManger<VertexType> &lockManager,
                       HeapDatastructure &maxGainHeap) {
        // To introduce non-determinism and help escape local optima, if there are multiple moves with the same
        // top gain, we randomly select one. We check up to `local_max` ties.
        const unsigned localMax = 50;
        std::vector<VertexType> topGainNodes = maxGainHeap.GetTopKeys(localMax);

        if (topGainNodes.empty()) {
            // This case is guarded by the caller, but for safety:
            topGainNodes.push_back(maxGainHeap.Top());
        }

        std::uniform_int_distribution<size_t> dis(0, topGainNodes.size() - 1);
        const VertexType node = topGainNodes[dis(gen_)];

        KlMove bestMove = maxGainHeap.GetValue(node);
        maxGainHeap.Erase(node);
        lockManager.Lock(node);
        affinityTable.Remove(node);

        return bestMove;
    }

    inline void ProcessOtherStepsBestMove(const unsigned idx,
                                          const unsigned nodeStep,
                                          const VertexType &node,
                                          const CostT affinityCurrentProcStep,
                                          CostT &maxGain,
                                          unsigned &maxProc,
                                          unsigned &maxStep,
                                          const std::vector<std::vector<CostT>> &affinityTableNode) const {
        for (const unsigned p : procRange_.CompatibleProcessorsVertex(node)) {
            if constexpr (ActiveScheduleT::useMemoryConstraint) {
                if (not activeSchedule_.memoryConstraint.CanMove(node, p, nodeStep + idx - WindowSize)) {
                    continue;
                }
            }

            const CostT gain = affinityCurrentProcStep - affinityTableNode[p][idx];
            if (gain > maxGain) {
                maxGain = gain;
                maxProc = p;
                maxStep = idx;
            }
        }
    }

    template <bool MoveToSameSuperStep>
    KlMove ComputeBestMove(VertexType node,
                           const std::vector<std::vector<CostT>> &affinityTableNode,
                           ThreadSearchContext &threadData) {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);

        CostT maxGain = std::numeric_limits<CostT>::lowest();

        unsigned maxProc = std::numeric_limits<unsigned>::max();
        unsigned maxStep = std::numeric_limits<unsigned>::max();

        const CostT affinityCurrentProcStep = affinityTableNode[nodeProc][WindowSize];

        unsigned idx = threadData.StartIdx(nodeStep);
        for (; idx < WindowSize; idx++) {
            ProcessOtherStepsBestMove(idx, nodeStep, node, affinityCurrentProcStep, maxGain, maxProc, maxStep, affinityTableNode);
        }

        if constexpr (MoveToSameSuperStep) {
            for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                if (proc == nodeProc) {
                    continue;
                }

                if constexpr (ActiveScheduleT::useMemoryConstraint) {
                    if (not activeSchedule_.memoryConstraint.CanMove(node, proc, nodeStep + idx - WindowSize)) {
                        continue;
                    }
                }

                const CostT gain = affinityCurrentProcStep - affinityTableNode[proc][WindowSize];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = proc;
                    maxStep = idx;
                }
            }
        }

        idx++;

        const unsigned bound = threadData.EndIdx(nodeStep);
        for (; idx < bound; idx++) {
            ProcessOtherStepsBestMove(idx, nodeStep, node, affinityCurrentProcStep, maxGain, maxProc, maxStep, affinityTableNode);
        }

        return KlMove(node, maxGain, nodeProc, nodeStep, maxProc, nodeStep + maxStep - WindowSize);
    }

    KlGainUpdateInfo UpdateNodeWorkAffinityAfterMove(VertexType node,
                                                     KlMove move,
                                                     const PreMoveWorkData<WorkWeightT> &prevWorkData,
                                                     std::vector<std::vector<CostT>> &affinityTableNode) {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const WorkWeightT vertexWeight = graph_->VertexWorkWeight(node);

        KlGainUpdateInfo updateInfo(node);

        if (move.fromStep == move.toStep) {
            const unsigned lowerBound = move.fromStep > WindowSize ? move.fromStep - WindowSize : 0;
            if (lowerBound <= nodeStep && nodeStep <= move.fromStep + WindowSize) {
                updateInfo.updateFromStep = true;
                updateInfo.updateToStep = true;

                const WorkWeightT prevMaxWork = prevWorkData.fromStepMaxWork;
                const WorkWeightT prevSecondMaxWork = prevWorkData.fromStepSecondMaxWork;

                if (nodeStep == move.fromStep) {
                    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
                    const WorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(move.fromStep);
                    const WorkWeightT newSecondMaxWeight = activeSchedule_.GetStepSecondMaxWork(move.fromStep);
                    const WorkWeightT newStepProcWork = activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);
                    const WorkWeightT prevStepProcWork
                        = (nodeProc == move.fromProc) ? newStepProcWork + graph_->VertexWorkWeight(move.node)
                          : (nodeProc == move.toProc) ? newStepProcWork - graph_->VertexWorkWeight(move.node)
                                                      : newStepProcWork;
                    const bool prevIsSoleMaxProcessor = (prevWorkData.fromStepMaxWorkProcessorCount == 1)
                                                        && (prevMaxWork == prevStepProcWork);
                    const CostT prevNodeProcAffinity
                        = prevIsSoleMaxProcessor ? std::min(vertexWeight, prevMaxWork - prevSecondMaxWork) : 0.0;
                    const bool newIsSoleMaxProcessor = (activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                                       && (newMaxWeight == newStepProcWork);
                    const CostT newNodeProcAffinity
                        = newIsSoleMaxProcessor ? std::min(vertexWeight, newMaxWeight - newSecondMaxWeight) : 0.0;

                    const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
                    if (std::abs(diff) > epsilon) {
                        updateInfo.fullUpdate = true;
                        affinityTableNode[nodeProc][WindowSize] += diff;    // Use the pre-calculated diff
                    }

                    if ((prevMaxWork != newMaxWeight) || updateInfo.fullUpdate) {
                        updateInfo.updateEntireFromStep = true;

                        for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                            if ((proc == nodeProc) || (proc == move.fromProc) || (proc == move.toProc)) {
                                continue;
                            }

                            const WorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
                            const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, newWeight, prevNodeProcAffinity);
                            const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                            affinityTableNode[proc][WindowSize] += (otherAffinity - prevOtherAffinity);
                        }
                    }

                    if (nodeProc != move.fromProc && IsCompatible(node, move.fromProc)) {
                        const WorkWeightT prevNewWeight = vertexWeight
                                                          + activeSchedule_.GetStepProcessorWork(nodeStep, move.fromProc)
                                                          + graph_->VertexWorkWeight(move.node);
                        const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, prevNewWeight, prevNodeProcAffinity);
                        const WorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, move.fromProc);
                        const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
                        affinityTableNode[move.fromProc][WindowSize] += (otherAffinity - prevOtherAffinity);
                    }

                    if (nodeProc != move.toProc && IsCompatible(node, move.toProc)) {
                        const WorkWeightT prevNewWeight = vertexWeight
                                                          + activeSchedule_.GetStepProcessorWork(nodeStep, move.toProc)
                                                          - graph_->VertexWorkWeight(move.node);
                        const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, prevNewWeight, prevNodeProcAffinity);
                        const WorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, move.toProc);
                        const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
                        affinityTableNode[move.toProc][WindowSize] += (otherAffinity - prevOtherAffinity);
                    }

                } else {
                    const WorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(move.fromStep);
                    const unsigned idx = RelStepIdx(nodeStep, move.fromStep);
                    if (prevMaxWork != newMaxWeight) {
                        updateInfo.updateEntireFromStep = true;
                        // update moving to all procs with special for move.from_proc
                        for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                            const WorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep, proc);
                            if (proc == move.fromProc) {
                                const WorkWeightT prevNewWeight = vertexWeight
                                                                  + activeSchedule_.GetStepProcessorWork(move.fromStep, proc)
                                                                  + graph_->VertexWorkWeight(move.node);
                                const CostT prevAffinity = prevMaxWork < prevNewWeight ? static_cast<CostT>(prevNewWeight)
                                                                                             - static_cast<CostT>(prevMaxWork)
                                                                                       : 0.0;
                                const CostT newAffinity = newMaxWeight < newWeight
                                                              ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                                              : 0.0;
                                affinityTableNode[proc][idx] += newAffinity - prevAffinity;
                            } else if (proc == move.toProc) {
                                const WorkWeightT prevNewWeight = vertexWeight
                                                                  + activeSchedule_.GetStepProcessorWork(move.toStep, proc)
                                                                  - graph_->VertexWorkWeight(move.node);
                                const CostT prevAffinity = prevMaxWork < prevNewWeight ? static_cast<CostT>(prevNewWeight)
                                                                                             - static_cast<CostT>(prevMaxWork)
                                                                                       : 0.0;
                                const CostT newAffinity = newMaxWeight < newWeight
                                                              ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                                              : 0.0;
                                affinityTableNode[proc][idx] += newAffinity - prevAffinity;
                            } else {
                                const CostT prevAffinity = prevMaxWork < newWeight
                                                               ? static_cast<CostT>(newWeight) - static_cast<CostT>(prevMaxWork)
                                                               : 0.0;
                                const CostT newAffinity = newMaxWeight < newWeight
                                                              ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                                              : 0.0;
                                affinityTableNode[proc][idx] += newAffinity - prevAffinity;
                            }
                        }
                    } else {
                        // update only move.from_proc and move.to_proc
                        if (IsCompatible(node, move.fromProc)) {
                            const WorkWeightT fromNewWeight
                                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep, move.fromProc);
                            const WorkWeightT fromPrevNewWeight = fromNewWeight + graph_->VertexWorkWeight(move.node);
                            const CostT fromPrevAffinity = prevMaxWork < fromPrevNewWeight ? static_cast<CostT>(fromPrevNewWeight)
                                                                                                 - static_cast<CostT>(prevMaxWork)
                                                                                           : 0.0;

                            const CostT fromNewAffinity = newMaxWeight < fromNewWeight ? static_cast<CostT>(fromNewWeight)
                                                                                             - static_cast<CostT>(newMaxWeight)
                                                                                       : 0.0;
                            affinityTableNode[move.fromProc][idx] += fromNewAffinity - fromPrevAffinity;
                        }

                        if (IsCompatible(node, move.toProc)) {
                            const WorkWeightT toNewWeight
                                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.toStep, move.toProc);
                            const WorkWeightT toPrevNewWeight = toNewWeight - graph_->VertexWorkWeight(move.node);
                            const CostT toPrevAffinity = prevMaxWork < toPrevNewWeight ? static_cast<CostT>(toPrevNewWeight)
                                                                                             - static_cast<CostT>(prevMaxWork)
                                                                                       : 0.0;

                            const CostT toNewAffinity = newMaxWeight < toNewWeight
                                                            ? static_cast<CostT>(toNewWeight) - static_cast<CostT>(newMaxWeight)
                                                            : 0.0;
                            affinityTableNode[move.toProc][idx] += toNewAffinity - toPrevAffinity;
                        }
                    }
                }
            }

        } else {
            const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
            ProcessWorkUpdateStep(node,
                                  nodeStep,
                                  nodeProc,
                                  vertexWeight,
                                  move.fromStep,
                                  move.fromProc,
                                  graph_->VertexWorkWeight(move.node),
                                  prevWorkData.fromStepMaxWork,
                                  prevWorkData.fromStepSecondMaxWork,
                                  prevWorkData.fromStepMaxWorkProcessorCount,
                                  updateInfo.updateFromStep,
                                  updateInfo.updateEntireFromStep,
                                  updateInfo.fullUpdate,
                                  affinityTableNode);
            ProcessWorkUpdateStep(node,
                                  nodeStep,
                                  nodeProc,
                                  vertexWeight,
                                  move.toStep,
                                  move.toProc,
                                  -graph_->VertexWorkWeight(move.node),
                                  prevWorkData.toStepMaxWork,
                                  prevWorkData.toStepSecondMaxWork,
                                  prevWorkData.toStepMaxWorkProcessorCount,
                                  updateInfo.updateToStep,
                                  updateInfo.updateEntireToStep,
                                  updateInfo.fullUpdate,
                                  affinityTableNode);
        }

        return updateInfo;
    }

    void ProcessWorkUpdateStep(VertexType node,
                               unsigned nodeStep,
                               unsigned nodeProc,
                               WorkWeightT vertexWeight,
                               unsigned moveStep,
                               unsigned moveProc,
                               WorkWeightT moveCorrectionNodeWeight,
                               const WorkWeightT prevMoveStepMaxWork,
                               const WorkWeightT prevMoveStepSecondMaxWork,
                               unsigned prevMoveStepMaxWorkProcessorCount,
                               bool &updateStep,
                               bool &updateEntireStep,
                               bool &fullUpdate,
                               std::vector<std::vector<CostT>> &affinityTableNode);
    void UpdateNodeWorkAffinity(NodeSelectionContainerT &nodes,
                                KlMove move,
                                const PreMoveWorkData<WorkWeightT> &prevWorkData,
                                std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain);
    void UpdateBestMove(
        VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateBestMove(VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateMaxGain(KlMove move, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain, ThreadSearchContext &threadData);
    void ComputeWorkAffinity(VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData);

    inline void RecomputeNodeMaxGain(VertexType node, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
        const auto bestMove = ComputeBestMove<true>(node, affinityTable[node], threadData);
        threadData.maxGainHeap.Update(node, bestMove);
    }

    inline CostT ComputeSameStepAffinity(const WorkWeightT &maxWorkForStep,
                                         const WorkWeightT &newWeight,
                                         const CostT &nodeProcAffinity) {
        const CostT maxWorkAfterRemoval = static_cast<CostT>(maxWorkForStep) - nodeProcAffinity;
        if (newWeight > maxWorkAfterRemoval) {
            return newWeight - maxWorkAfterRemoval;
        }
        return 0.0;
    }

    inline CostT ApplyMove(KlMove move, ThreadSearchContext &threadData) {
        activeSchedule_.ApplyMove(move, threadData.activeScheduleData);
        commCostF_.UpdateDatastructureAfterMove(move, threadData.startStep, threadData.endStep);
        CostT changeInCost = -move.gain;
        changeInCost
            += static_cast<CostT>(threadData.activeScheduleData.resolvedViolations.size()) * threadData.rewardPenaltyStrat.reward;
        changeInCost
            -= static_cast<CostT>(threadData.activeScheduleData.newViolations.size()) * threadData.rewardPenaltyStrat.penalty;

#ifdef KL_DEBUG
        std::cout << "penalty: " << thread_data.reward_penalty_strat.penalty
                  << " num violations: " << thread_data.active_schedule_data.current_violations.size()
                  << " num new violations: " << thread_data.active_schedule_data.new_violations.size()
                  << ", num resolved violations: " << thread_data.active_schedule_data.resolved_violations.size()
                  << ", reward: " << thread_data.reward_penalty_strat.reward << std::endl;
        std::cout << "apply move, previous cost: " << thread_data.active_schedule_data.cost
                  << ", new cost: " << thread_data.active_schedule_data.cost + change_in_cost << ", "
                  << (thread_data.active_schedule_data.feasible ? "feasible," : "infeasible,") << std::endl;
#endif

        threadData.activeScheduleData.UpdateCost(changeInCost);

        return changeInCost;
    }

    void RunQuickMoves(unsigned &innerIter,
                       ThreadSearchContext &threadData,
                       const CostT changeInCost,
                       const VertexType bestMoveNode) {
#ifdef KL_DEBUG
        std::cout << "Starting quick moves sequence." << std::endl;
#endif
        innerIter++;

        const size_t numAppliedMoves = threadData.activeScheduleData.appliedMoves.size() - 1;
        const CostT savedCost = threadData.activeScheduleData.cost - changeInCost;

        std::unordered_set<VertexType> localLock;
        localLock.insert(bestMoveNode);
        std::vector<VertexType> quickMovesStack;
        quickMovesStack.reserve(10 + threadData.activeScheduleData.newViolations.size() * 2);

        for (const auto &keyValuePair : threadData.activeScheduleData.newViolations) {
            const auto &key = keyValuePair.first;
            quickMovesStack.push_back(key);
        }

        while (quickMovesStack.size() > 0) {
            auto nextNodeToMove = quickMovesStack.back();
            quickMovesStack.pop_back();

            threadData.rewardPenaltyStrat.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData.currentViolations.size()) + 1.0);
            ComputeNodeAffinities(nextNodeToMove, threadData.localAffinityTable, threadData);
            KlMove bestQuickMove = ComputeBestMove<true>(nextNodeToMove, threadData.localAffinityTable, threadData);

            localLock.insert(nextNodeToMove);
            if (bestQuickMove.gain <= std::numeric_limits<CostT>::lowest()) {
                continue;
            }

#ifdef KL_DEBUG
            std::cout << " >>> move node " << best_quick_move.node << " with gain " << best_quick_move.gain
                      << ", from proc|step: " << best_quick_move.from_proc << "|" << best_quick_move.from_step
                      << " to: " << best_quick_move.to_proc << "|" << best_quick_move.to_step << std::endl;
#endif

            ApplyMove(bestQuickMove, threadData);
            innerIter++;

            if (threadData.activeScheduleData.newViolations.size() > 0) {
                bool abort = false;

                for (const auto &keyValuePair : threadData.activeScheduleData.newViolations) {
                    const auto &key = keyValuePair.first;
                    if (localLock.find(key) != localLock.end()) {
                        abort = true;
                        break;
                    }
                    quickMovesStack.push_back(key);
                }

                if (abort) {
                    break;
                }

            } else if (threadData.activeScheduleData.feasible) {
                break;
            }
        }

        if (!threadData.activeScheduleData.feasible) {
            activeSchedule_.RevertScheduleToBound(
                numAppliedMoves, savedCost, true, commCostF_, threadData.activeScheduleData, threadData.startStep, threadData.endStep);
#ifdef KL_DEBUG
            std::cout << "Ending quick moves sequence with infeasible solution." << std::endl;
#endif
        }
#ifdef KL_DEBUG
        else {
            std::cout << "Ending quick moves sequence with feasible solution." << std::endl;
        }
#endif

        threadData.affinityTable.Trim();
        threadData.maxGainHeap.Clear();
        threadData.rewardPenaltyStrat.InitRewardPenalty(1.0);
        InsertGainHeap(threadData);    // Re-initialize the heap with the current state
    }

    void ResolveViolations(ThreadSearchContext &threadData) {
        auto &currentViolations = threadData.activeScheduleData.currentViolations;
        unsigned numViolations = static_cast<unsigned>(currentViolations.size());
        if (numViolations > 0) {
#ifdef KL_DEBUG_1
            std::cout << "thread " << thread_data.thread_id << ", Starting preresolving violations with " << num_violations
                      << " initial violations" << std::endl;
#endif
            threadData.rewardPenaltyStrat.InitRewardPenalty(static_cast<double>(numViolations) + 1.0);
            std::unordered_set<VertexType> localLock;
            unsigned numIter = 0;
            const unsigned minIter = numViolations / 4;
            while (not currentViolations.empty()) {
                std::uniform_int_distribution<size_t> dis(0, currentViolations.size() - 1);
                auto it = currentViolations.begin();
                std::advance(it, dis(gen_));
                const auto &nextEdge = *it;
                const VertexType sourceV = Source(nextEdge, *graph_);
                const VertexType targetV = Target(nextEdge, *graph_);
                const bool sourceLocked = localLock.find(sourceV) != localLock.end();
                const bool targetLocked = localLock.find(targetV) != localLock.end();

                if (sourceLocked && targetLocked) {
#ifdef KL_DEBUG_1
                    std::cout << "source, target locked" << std::endl;
#endif
                    break;
                }

                KlMove bestMove;
                if (sourceLocked || targetLocked) {
                    const VertexType node = sourceLocked ? targetV : sourceV;
                    ComputeNodeAffinities(node, threadData.localAffinityTable, threadData);
                    bestMove = ComputeBestMove<true>(node, threadData.localAffinityTable, threadData);
                } else {
                    ComputeNodeAffinities(sourceV, threadData.localAffinityTable, threadData);
                    KlMove bestSourceVMove = ComputeBestMove<true>(sourceV, threadData.localAffinityTable, threadData);
                    ComputeNodeAffinities(targetV, threadData.localAffinityTable, threadData);
                    KlMove bestTargetVMove = ComputeBestMove<true>(targetV, threadData.localAffinityTable, threadData);
                    bestMove = bestTargetVMove.gain > bestSourceVMove.gain ? std::move(bestTargetVMove)
                                                                           : std::move(bestSourceVMove);
                }

                localLock.insert(bestMove.node);
                if (bestMove.gain <= std::numeric_limits<CostT>::lowest()) {
                    continue;
                }

                ApplyMove(bestMove, threadData);
                threadData.affinityTable.Insert(bestMove.node);
#ifdef KL_DEBUG_1
                std::cout << "move node " << best_move.node << " with gain " << best_move.gain
                          << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step
                          << " to: " << best_move.to_proc << "|" << best_move.to_step << std::endl;
#endif
                const unsigned newNumViolations = static_cast<unsigned>(currentViolations.size());
                if (newNumViolations == 0) {
                    break;
                }

                if (threadData.activeScheduleData.newViolations.size() > 0) {
                    for (const auto &vertexEdgePair : threadData.activeScheduleData.newViolations) {
                        const auto &vertex = vertexEdgePair.first;
                        threadData.affinityTable.Insert(vertex);
                    }
                }

                const double gain = static_cast<double>(numViolations) - static_cast<double>(newNumViolations);
                numViolations = newNumViolations;
                UpdateAvgGain(gain, numIter++, threadData.averageGain);
#ifdef KL_DEBUG_1
                std::cout << "thread " << thread_data.thread_id << ",  preresolving violations with " << num_violations
                          << " violations, " << num_iter << " #iterations, " << thread_data.average_gain << " average gain"
                          << std::endl;
#endif
                if (numIter > minIter && threadData.averageGain < 0.0) {
                    break;
                }
            }
            threadData.averageGain = 0.0;
        }
    }

    void RunLocalSearch(ThreadSearchContext &threadData) {
#ifdef KL_DEBUG_1
        std::cout << "thread " << thread_data.thread_id
                  << ", start local search, initial schedule cost: " << thread_data.active_schedule_data.cost << " with "
                  << thread_data.num_steps() << " supersteps." << std::endl;
#endif
        std::vector<VertexType> newNodes;
        std::vector<VertexType> unlockNodes;
        std::map<VertexType, KlGainUpdateInfo> recomputeMaxGain;

        const auto startTime = std::chrono::high_resolution_clock::now();

        unsigned noImprovementIterCounter = 0;
        unsigned outerIter = 0;

        for (; outerIter < parameters_.maxOuterIterations; outerIter++) {
            CostT initialInnerIterCost = threadData.activeScheduleData.cost;

            ResetInnerSearchStructures(threadData);
            SelectActiveNodes(threadData);
            threadData.rewardPenaltyStrat.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData.currentViolations.size()) + 1.0);
            InsertGainHeap(threadData);

            unsigned innerIter = 0;
            unsigned violationRemovedCount = 0;
            unsigned resetCounter = 0;
            bool iterInitalFeasible = threadData.activeScheduleData.feasible;

#ifdef KL_DEBUG
            std::cout << "------ start inner loop ------" << std::endl;
            std::cout << "initial node selection: {";
            for (size_t i = 0; i < thread_data.affinity_table.size(); ++i) {
                std::cout << thread_data.affinity_table.get_selected_nodes()[i] << ", ";
            }
            std::cout << "}" << std::endl;
#endif
#ifdef KL_DEBUG_1
            if (not iter_inital_feasible) {
                std::cout << "initial solution not feasible, num violations: "
                          << thread_data.active_schedule_data.current_violations.size()
                          << ". Penalty: " << thread_data.reward_penalty_strat.penalty
                          << ", reward: " << thread_data.reward_penalty_strat.reward << std::endl;
            }
#endif
#ifdef KL_DEBUG_COST_CHECK
            active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
            if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                          << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
            }
            if constexpr (active_schedule_t::use_memory_constraint) {
                if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                    std::cout << "memory constraint not satisfied" << std::endl;
                }
            }
#endif

            while (innerIter < threadData.maxInnerIterations && threadData.maxGainHeap.Size() > 0) {
                KlMove bestMove
                    = GetBestMove(threadData.affinityTable,
                                  threadData.lockManager,
                                  threadData.maxGainHeap);    // locks best_move.node and removes it from node_selection
                if (bestMove.gain <= std::numeric_limits<CostT>::lowest()) {
                    break;
                }
                UpdateAvgGain(bestMove.gain, innerIter, threadData.averageGain);
#ifdef KL_DEBUG
                std::cout << " >>> move node " << best_move.node << " with gain " << best_move.gain
                          << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step
                          << " to: " << best_move.to_proc << "|" << best_move.to_step << ",avg gain: " << thread_data.average_gain
                          << std::endl;
#endif
                if (innerIter > threadData.minInnerIter && threadData.averageGain < 0.0) {
#ifdef KL_DEBUG
                    std::cout << "Negative average gain: " << thread_data.average_gain << ", end local search" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                if (not active_schedule.GetInstance().isCompatible(best_move.node, best_move.to_proc)) {
                    std::cout << "move to incompatibe node" << std::endl;
                }
#endif

                const auto prevWorkData = activeSchedule_.GetPreMoveWorkData(bestMove);
                const typename CommCostFunctionT::PreMoveCommDataT prevCommData = commCostF_.GetPreMoveCommData(bestMove);
                const CostT changeInCost = ApplyMove(bestMove, threadData);
#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                              << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                        std::cout << "memory constraint not satisfied" << std::endl;
                    }
                }
#endif
                if constexpr (enableQuickMoves) {
                    if (iterInitalFeasible && threadData.activeScheduleData.newViolations.size() > 0) {
                        RunQuickMoves(innerIter, threadData, changeInCost, bestMove.node);
#ifdef KL_DEBUG_COST_CHECK
                        active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                        if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                            std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                                      << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                            std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<"
                                      << std::endl;
                        }
                        if constexpr (active_schedule_t::use_memory_constraint) {
                            if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                                std::cout << "memory constraint not satisfied" << std::endl;
                            }
                        }
#endif
                        continue;
                    }
                }

                if (threadData.activeScheduleData.currentViolations.size() > 0) {
                    if (threadData.activeScheduleData.resolvedViolations.size() > 0) {
                        violationRemovedCount = 0;
                    } else {
                        violationRemovedCount++;

                        if (violationRemovedCount > 3) {
                            if (resetCounter < threadData.maxNoVioaltionsRemovedBacktrack
                                && ((not iterInitalFeasible)
                                    || (threadData.activeScheduleData.cost < threadData.activeScheduleData.bestCost))) {
                                threadData.affinityTable.ResetNodeSelection();
                                threadData.maxGainHeap.Clear();
                                threadData.lockManager.Clear();
                                threadData.selectionStrategy.SelectNodesViolations(threadData.affinityTable,
                                                                                   threadData.activeScheduleData.currentViolations,
                                                                                   threadData.startStep,
                                                                                   threadData.endStep);
#ifdef KL_DEBUG
                                std::cout << "Infeasible, and no violations resolved for 5 iterations, reset node selection"
                                          << std::endl;
#endif
                                threadData.rewardPenaltyStrat.InitRewardPenalty(
                                    static_cast<double>(threadData.activeScheduleData.currentViolations.size()));
                                InsertGainHeap(threadData);

                                resetCounter++;
                                innerIter++;
                                continue;
                            } else {
#ifdef KL_DEBUG
                                std::cout << "Infeasible, and no violations resolved for 5 iterations, end local search"
                                          << std::endl;
#endif
                                break;
                            }
                        }
                    }
                }

                if (IsLocalSearchBlocked(threadData)) {
                    if (not BlockedEdgeStrategy(bestMove.node, unlockNodes, threadData)) {
                        break;
                    }
                }

                threadData.affinityTable.Trim();
                UpdateAffinities(bestMove, threadData, recomputeMaxGain, newNodes, prevWorkData, prevCommData);

                for (const auto v : unlockNodes) {
                    threadData.lockManager.Unlock(v);
                }
                newNodes.insert(newNodes.end(), unlockNodes.begin(), unlockNodes.end());
                unlockNodes.clear();

#ifdef KL_DEBUG
                std::cout << "recmopute max gain: {";
                for (const auto map_pair : recompute_max_gain) {
                    const auto &key = map_pair.first;
                    std::cout << key << ", ";
                }
                std::cout << "}" << std::endl;
                std::cout << "new nodes: {";
                for (const auto v : new_nodes) {
                    std::cout << v << ", ";
                }
                std::cout << "}" << std::endl;
#endif
#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                              << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                        std::cout << "memory constraint not satisfied" << std::endl;
                    }
                }
#endif
                UpdateMaxGain(bestMove, recomputeMaxGain, threadData);
                InsertNewNodesGainHeap(newNodes, threadData.affinityTable, threadData);

                recomputeMaxGain.clear();
                newNodes.clear();

                innerIter++;
            }

#ifdef KL_DEBUG
            std::cout << "--- end inner loop after " << inner_iter
                      << " inner iterations, gain heap size: " << thread_data.max_gain_heap.size() << ", outer iteraion "
                      << outer_iter << "/" << parameters.max_outer_iterations
                      << ", current cost: " << thread_data.active_schedule_data.cost << ", "
                      << (thread_data.active_schedule_data.feasible ? "feasible" : "infeasible") << std::endl;
#endif
#ifdef KL_DEBUG_1
            const unsigned num_steps_tmp = thread_data.end_step;
#endif
            activeSchedule_.RevertToBestSchedule(threadData.localSearchStartStep,
                                                 threadData.stepToRemove,
                                                 commCostF_,
                                                 threadData.activeScheduleData,
                                                 threadData.startStep,
                                                 threadData.endStep);
#ifdef KL_DEBUG_1
            if (thread_data.local_search_start_step > 0) {
                if (num_steps_tmp == thread_data.end_step) {
                    std::cout << "thread " << thread_data.thread_id << ", removing step " << thread_data.step_to_remove
                              << " succeded " << std::endl;
                } else {
                    std::cout << "thread " << thread_data.thread_id << ", removing step " << thread_data.step_to_remove
                              << " failed " << std::endl;
                }
            }
#endif

#ifdef KL_DEBUG_COST_CHECK
            active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
            if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                          << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
            }
            if constexpr (active_schedule_t::use_memory_constraint) {
                if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                    std::cout << "memory constraint not satisfied" << std::endl;
                }
            }
#endif

            if (computeWithTimeLimit_) {
                auto finishTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finishTime - startTime).count();
                if (duration > ImprovementScheduler<GraphT>::timeLimitSeconds_) {
                    break;
                }
            }

            if (OtherThreadsFinished(threadData.threadId)) {
#ifdef KL_DEBUG_1
                std::cout << "thread " << thread_data.thread_id << ", other threads finished, end local search" << std::endl;
#endif
                break;
            }

            if (initialInnerIterCost <= threadData.activeScheduleData.cost) {
                noImprovementIterCounter++;

                if (noImprovementIterCounter >= parameters_.maxNoImprovementIterations) {
#ifdef KL_DEBUG_1
                    std::cout << "thread " << thread_data.thread_id << ", no improvement for "
                              << parameters.max_no_improvement_iterations << " iterations, end local search" << std::endl;
#endif
                    break;
                }
            } else {
                noImprovementIterCounter = 0;
            }

            AdjustLocalSearchParameters(outerIter, noImprovementIterCounter, threadData);
        }

#ifdef KL_DEBUG_1
        std::cout << "thread " << thread_data.thread_id << ", local search end after " << outer_iter
                  << " outer iterations, current cost: " << thread_data.active_schedule_data.cost << " with "
                  << thread_data.num_steps() << " supersteps, vs serial cost " << active_schedule.get_total_work_weight() << "."
                  << std::endl;
#endif
        threadFinishedVec_[threadData.threadId] = true;
    }

    bool OtherThreadsFinished(const unsigned threadId) {
        const size_t numThreads = threadFinishedVec_.size();
        if (numThreads == 1) {
            return false;
        }

        for (size_t i = 0; i < numThreads; i++) {
            if (i != threadId && !threadFinishedVec_[i]) {
                return false;
            }
        }
        return true;
    }

    inline void UpdateAffinities(const KlMove &bestMove,
                                 ThreadSearchContext &threadData,
                                 std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain,
                                 std::vector<VertexType> &newNodes,
                                 const PreMoveWorkData<VWorkwT<GraphT>> &prevWorkData,
                                 const typename CommCostFunctionT::PreMoveCommDataT &prevCommData) {
        if constexpr (CommCostFunctionT::isMaxCommCostFunction) {
            commCostF_.UpdateNodeCommAffinity(
                bestMove,
                threadData,
                threadData.rewardPenaltyStrat.penalty,
                threadData.rewardPenaltyStrat.reward,
                recomputeMaxGain,
                newNodes);    // this only updated reward/penalty, collects new_nodes, and fills recompute_max_gain

            // Add nodes from affected steps to new_nodes
            // {
            //     std::unordered_set<unsigned> steps_to_check;
            //     const unsigned num_steps = active_schedule.num_steps();

            //     auto add_steps_range = [&](unsigned center_step) {
            //         unsigned start = (center_step > window_size) ? center_step - window_size : 0;
            //         unsigned end = std::min(center_step + window_size, num_steps - 1);

            //         // Constrain to thread range
            //         if (start < thread_data.start_step)
            //             start = thread_data.start_step;
            //         if (end > thread_data.end_step)
            //             end = thread_data.end_step;

            //         for (unsigned s = start; s <= end; ++s) {
            //             steps_to_check.insert(s);
            //         }
            //     };

            //     add_steps_range(best_move.from_step);
            //     add_steps_range(best_move.to_step);

            //     for (unsigned step : steps_to_check) {
            //         for (unsigned proc = 0; proc < instance->NumberOfProcessors(); ++proc) {
            //             const auto &nodes_in_step = active_schedule.getSetSchedule().step_processor_vertices[step][proc];
            //             for (const auto &node : nodes_in_step) {
            //                 if (!thread_data.affinity_table.is_selected(node) && !thread_data.lock_manager.is_locked(node)) {
            //                     new_nodes.push_back(node);
            //                 }
            //             }
            //         }
            //     }

            //     // Deduplicate new_nodes
            //     std::sort(new_nodes.begin(), new_nodes.end());
            //     new_nodes.erase(std::unique(new_nodes.begin(), new_nodes.end()), new_nodes.end());
            // }

            // Determine the steps where max/second_max/max_count for work/comm changed
            std::unordered_set<unsigned> changedSteps;

            // Check work changes for from_step
            if (bestMove.fromStep == bestMove.toStep) {
                // Same step - check if max/second_max changed
                const auto currentMax = activeSchedule_.GetStepMaxWork(bestMove.fromStep);
                const auto currentSecondMax = activeSchedule_.GetStepSecondMaxWork(bestMove.fromStep);
                const auto currentCount = activeSchedule_.GetStepMaxWorkProcessorCount()[bestMove.fromStep];
                if (currentMax != prevWorkData.fromStepMaxWork || currentSecondMax != prevWorkData.fromStepSecondMaxWork
                    || currentCount != prevWorkData.fromStepMaxWorkProcessorCount) {
                    changedSteps.insert(bestMove.fromStep);
                }
            } else {
                // Different steps - check both
                const auto currentFromMax = activeSchedule_.GetStepMaxWork(bestMove.fromStep);
                const auto currentFromSecondMax = activeSchedule_.GetStepSecondMaxWork(bestMove.fromStep);
                const auto currentFromCount = activeSchedule_.GetStepMaxWorkProcessorCount()[bestMove.fromStep];
                if (currentFromMax != prevWorkData.fromStepMaxWork || currentFromSecondMax != prevWorkData.fromStepSecondMaxWork
                    || currentFromCount != prevWorkData.fromStepMaxWorkProcessorCount) {
                    changedSteps.insert(bestMove.fromStep);
                }

                const auto currentToMax = activeSchedule_.GetStepMaxWork(bestMove.toStep);
                const auto currentToSecondMax = activeSchedule_.GetStepSecondMaxWork(bestMove.toStep);
                const auto currentToCount = activeSchedule_.GetStepMaxWorkProcessorCount()[bestMove.toStep];
                if (currentToMax != prevWorkData.toStepMaxWork || currentToSecondMax != prevWorkData.toStepSecondMaxWork
                    || currentToCount != prevWorkData.toStepMaxWorkProcessorCount) {
                    changedSteps.insert(bestMove.toStep);
                }
            }

            for (const auto &[step, step_info] : prevCommData.stepData) {
                typename CommCostFunctionT::PreMoveCommDataT::StepInfo currentInfo;
                // Query current values
                const auto currentMax = commCostF_.commDs.StepMaxComm(step);
                const auto currentSecondMax = commCostF_.commDs.StepSecondMaxComm(step);
                const auto currentCount = commCostF_.commDs.StepMaxCommCount(step);

                if (currentMax != step_info.maxComm || currentSecondMax != step_info.secondMaxComm
                    || currentCount != step_info.maxCommCount) {
                    changedSteps.insert(step);
                }
            }

            // Recompute affinities for all active nodes
            const size_t activeCount = threadData.affinityTable.Size();
            for (size_t i = 0; i < activeCount; ++i) {
                const VertexType node = threadData.affinityTable.GetSelectedNodes()[i];

                // Determine if this node needs affinity recomputation
                // A node needs recomputation if it's in or adjacent to changed steps
                const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);

                // Calculate window bounds for this node once
                const int nodeLowerBound = static_cast<int>(nodeStep) - static_cast<int>(WindowSize);
                const unsigned nodeUpperBound = nodeStep + WindowSize;

                bool needsUpdate = false;
                // Check if any changed step falls within the node's window
                for (unsigned step : changedSteps) {
                    if (static_cast<int>(step) >= nodeLowerBound && step <= nodeUpperBound) {
                        needsUpdate = true;
                        break;
                    }
                }

                if (needsUpdate) {
                    auto &affinityTableNode = threadData.affinityTable.GetAffinityTable(node);

                    // Reset affinity table entries to zero
                    const unsigned numProcs = activeSchedule_.GetInstance().NumberOfProcessors();
                    for (unsigned p = 0; p < numProcs; ++p) {
                        for (unsigned idx = 0; idx < affinityTableNode[p].size(); ++idx) {
                            affinityTableNode[p][idx] = 0;
                        }
                    }

                    ComputeNodeAffinities(node, affinityTableNode, threadData);
                    recomputeMaxGain[node] = KlGainUpdateInfo(node, true);
                }
            }
        } else {
            UpdateNodeWorkAffinity(threadData.affinityTable, bestMove, prevWorkData, recomputeMaxGain);
            commCostF_.UpdateNodeCommAffinity(bestMove,
                                              threadData,
                                              threadData.rewardPenaltyStrat.penalty,
                                              threadData.rewardPenaltyStrat.reward,
                                              recomputeMaxGain,
                                              newNodes);
        }
    }

    inline bool BlockedEdgeStrategy(VertexType node, std::vector<VertexType> &unlockNodes, ThreadSearchContext &threadData) {
        if (threadData.unlockEdgeBacktrackCounter > 1) {
            for (const auto vertexEdgePair : threadData.activeScheduleData.newViolations) {
                const auto &e = vertexEdgePair.second;
                const auto sourceV = Source(e, *graph_);
                const auto targetV = Target(e, *graph_);

                if (node == sourceV && threadData.lockManager.IsLocked(targetV)) {
                    unlockNodes.push_back(targetV);
                } else if (node == targetV && threadData.lockManager.IsLocked(sourceV)) {
                    unlockNodes.push_back(sourceV);
                }
            }
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, backtrack counter: " << thread_data.unlock_edge_backtrack_counter
                      << std::endl;
#endif
            threadData.unlockEdgeBacktrackCounter--;
            return true;
        } else {
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, end local search" << std::endl;
#endif
            return false;    // or reset local search and initalize with violating nodes
        }
    }

    inline void AdjustLocalSearchParameters(unsigned outerIter, unsigned noImpCounter, ThreadSearchContext &threadData) {
        if (noImpCounter >= threadData.noImprovementIterationsReducePenalty && threadData.rewardPenaltyStrat.initialPenalty > 1.0) {
            threadData.rewardPenaltyStrat.initialPenalty
                = static_cast<CostT>(std::floor(std::sqrt(threadData.rewardPenaltyStrat.initialPenalty)));
            threadData.unlockEdgeBacktrackCounterReset += 1;
            threadData.noImprovementIterationsReducePenalty += 15;
#ifdef KL_DEBUG_1
            std::cout << "thread " << thread_data.thread_id << ", no improvement for "
                      << thread_data.no_improvement_iterations_reduce_penalty << " iterations, reducing initial penalty to "
                      << thread_data.reward_penalty_strat.initial_penalty << std::endl;
#endif
        }

        if (parameters_.tryRemoveStepAfterNumOuterIterations > 0
            && ((outerIter + 1) % parameters_.tryRemoveStepAfterNumOuterIterations) == 0) {
            threadData.stepSelectionEpochCounter = 0;
            ;
#ifdef KL_DEBUG
            std::cout << "reset remove epoc counter after " << outer_iter << " iterations." << std::endl;
#endif
        }

        if (noImpCounter >= threadData.noImprovementIterationsIncreaseInnerIter) {
            threadData.minInnerIter = static_cast<unsigned>(std::ceil(threadData.minInnerIter * 2.2));
            threadData.noImprovementIterationsIncreaseInnerIter += 20;
#ifdef KL_DEBUG_1
            std::cout << "thread " << thread_data.thread_id << ", no improvement for "
                      << thread_data.no_improvement_iterations_increase_inner_iter << " iterations, increasing min inner iter to "
                      << thread_data.min_inner_iter << std::endl;
#endif
        }
    }

    bool IsLocalSearchBlocked(ThreadSearchContext &threadData);
    void SetParameters(VertexIdxT<GraphT> numNodes);
    void ResetInnerSearchStructures(ThreadSearchContext &threadData) const;
    void InitializeDatastructures(BspSchedule<GraphT> &schedule);
    void PrintHeap(HeapDatastructure &maxGainHeap) const;
    void CleanupDatastructures();
    void UpdateAvgGain(const CostT gain, const unsigned numIter, double &averageGain);
    void InsertGainHeap(ThreadSearchContext &threadData);
    void InsertNewNodesGainHeap(std::vector<VertexType> &newNodes, NodeSelectionContainerT &nodes, ThreadSearchContext &threadData);

    inline void ComputeNodeAffinities(VertexType node,
                                      std::vector<std::vector<CostT>> &affinityTableNode,
                                      ThreadSearchContext &threadData) {
        ComputeWorkAffinity(node, affinityTableNode, threadData);
        commCostF_.ComputeCommAffinity(node,
                                       affinityTableNode,
                                       threadData.rewardPenaltyStrat.penalty,
                                       threadData.rewardPenaltyStrat.reward,
                                       threadData.startStep,
                                       threadData.endStep);
    }

    void SelectActiveNodes(ThreadSearchContext &threadData) {
        if (SelectNodesCheckRemoveSuperstep(threadData.stepToRemove, threadData)) {
            activeSchedule_.swap_empty_step_fwd(threadData.stepToRemove, threadData.endStep);
            threadData.endStep--;
            threadData.localSearchStartStep = static_cast<unsigned>(threadData.activeScheduleData.appliedMoves.size());
            threadData.activeScheduleData.UpdateCost(static_cast<CostT>(-1.0 * instance_->SynchronisationCosts()));

            if constexpr (enablePreresolvingViolations) {
                ResolveViolations(threadData);
            }

            if (threadData.activeScheduleData.currentViolations.size() > parameters_.initialViolationThreshold) {
                activeSchedule_.RevertToBestSchedule(threadData.localSearchStartStep,
                                                     threadData.stepToRemove,
                                                     commCostF_,
                                                     threadData.activeScheduleData,
                                                     threadData.startStep,
                                                     threadData.endStep);
            } else {
                threadData.unlockEdgeBacktrackCounter
                    = static_cast<unsigned>(threadData.activeScheduleData.currentViolations.size());
                threadData.maxInnerIterations
                    = std::max(threadData.unlockEdgeBacktrackCounter * 5u, parameters_.maxInnerIterationsReset);
                threadData.maxNoVioaltionsRemovedBacktrack = parameters_.maxNoVioaltionsRemovedBacktrackForRemoveStepReset;
#ifdef KL_DEBUG_1
                std::cout << "thread " << thread_data.thread_id << ", Trying to remove step " << thread_data.step_to_remove
                          << std::endl;
#endif
                return;
            }
        }
        // thread_data.step_to_remove = thread_data.start_step;
        threadData.localSearchStartStep = 0;
        threadData.selectionStrategy.SelectActiveNodes(threadData.affinityTable, threadData.startStep, threadData.endStep);
    }

    bool CheckRemoveSuperstep(unsigned step);
    bool SelectNodesCheckRemoveSuperstep(unsigned &step, ThreadSearchContext &threadData);

    bool ScatterNodesSuperstep(unsigned step, ThreadSearchContext &threadData) {
        assert(step <= threadData.endStep && threadData.startStep <= step);
        bool abort = false;

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            const std::vector<VertexType> stepProcNodeVec(
                activeSchedule_.GetSetSchedule().stepProcessorVertices[step][proc].begin(),
                activeSchedule_.GetSetSchedule().stepProcessorVertices[step][proc].end());
            for (const auto &node : stepProcNodeVec) {
                threadData.rewardPenaltyStrat.InitRewardPenalty(
                    static_cast<double>(threadData.activeScheduleData.currentViolations.size()) + 1.0);
                ComputeNodeAffinities(node, threadData.localAffinityTable, threadData);
                KlMove bestMove = ComputeBestMove<false>(node, threadData.localAffinityTable, threadData);

                if (bestMove.gain <= std::numeric_limits<double>::lowest()) {
                    abort = true;
                    break;
                }

                ApplyMove(bestMove, threadData);
                if (threadData.activeScheduleData.currentViolations.size() > parameters_.abortScatterNodesViolationThreshold) {
                    abort = true;
                    break;
                }

                threadData.affinityTable.Insert(node);
                // thread_data.selection_strategy.add_neighbours_to_selection(node, thread_data.affinity_table,
                // thread_data.start_step, thread_data.end_step);
                if (threadData.activeScheduleData.newViolations.size() > 0) {
                    for (const auto &vertexEdgePair : threadData.activeScheduleData.newViolations) {
                        const auto &vertex = vertexEdgePair.first;
                        threadData.affinityTable.Insert(vertex);
                    }
                }

#ifdef KL_DEBUG
                std::cout << "move node " << best_move.node << " with gain " << best_move.gain
                          << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step
                          << " to: " << best_move.to_proc << "|" << best_move.to_step << std::endl;
#endif

#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test()
                              << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.satisfied_memory_constraint()) {
                        std::cout << "memory constraint not satisfied" << std::endl;
                    }
                }
#endif
            }

            if (abort) {
                break;
            }
        }

        if (abort) {
            activeSchedule_.RevertToBestSchedule(
                0, 0, commCostF_, threadData.activeScheduleData, threadData.startStep, threadData.endStep);
            threadData.affinityTable.ResetNodeSelection();
            return false;
        }
        return true;
    }

    void SynchronizeActiveSchedule(const unsigned numThreads) {
        if (numThreads == 1) {    // single thread case
            activeSchedule_.SetCost(threadDataVec_[0].activeScheduleData.cost);
            activeSchedule_.GetVectorSchedule().NumberOfSupersteps = threadDataVec_[0].NumSteps();
            return;
        }

        unsigned writeCursor = threadDataVec_[0].endStep + 1;
        for (unsigned i = 1; i < numThreads; ++i) {
            auto &thread = threadDataVec_[i];
            if (thread.startStep <= thread.endStep) {
                for (unsigned j = thread.startStep; j <= thread.endStep; ++j) {
                    if (j != writeCursor) {
                        activeSchedule_.swap_steps(j, writeCursor);
                    }
                    writeCursor++;
                }
            }
        }
        activeSchedule_.GetVectorSchedule().NumberOfSupersteps = writeCursor;
        const CostT newCost = commCostF_.ComputeScheduleCost();
        activeSchedule_.SetCost(newCost);
    }

  public:
    KlImprover() : ImprovementScheduler<GraphT>() {
        std::random_device rd;
        gen_ = std::mt19937(rd());
    }

    explicit KlImprover(unsigned seed) : ImprovementScheduler<GraphT>() { gen_ = std::mt19937(seed); }

    virtual ~KlImprover() = default;

    virtual RETURN_STATUS ImproveSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.GetInstance().NumberOfProcessors() < 2) {
            return RETURN_STATUS::BEST_FOUND;
        }

        const unsigned numThreads = 1;

        threadDataVec_.resize(numThreads);
        threadFinishedVec_.assign(numThreads, true);

        SetParameters(schedule.GetInstance().NumberOfVertices());
        InitializeDatastructures(schedule);
        const CostT initialCost = activeSchedule_.GetCost();
        const unsigned numSteps = schedule.NumberOfSupersteps();

        SetStartStep(0, threadDataVec_[0]);
        threadDataVec_[0].endStep = (numSteps > 0) ? numSteps - 1 : 0;

        auto &threadData = this->threadDataVec_[0];
        threadData.activeScheduleData.InitializeCost(activeSchedule_.GetCost());
        threadData.selectionStrategy.Setup(threadData.startStep, threadData.endStep);
        RunLocalSearch(threadData);

        SynchronizeActiveSchedule(numThreads);

        if (initialCost > activeSchedule_.GetCost()) {
            activeSchedule_.WriteSchedule(schedule);
            CleanupDatastructures();
            return RETURN_STATUS::OSP_SUCCESS;
        } else {
            CleanupDatastructures();
            return RETURN_STATUS::BEST_FOUND;
        }
    }

    virtual RETURN_STATUS ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &schedule) override {
        computeWithTimeLimit_ = true;
        return ImproveSchedule(schedule);
    }

    virtual void SetTimeQualityParameter(const double timeQuality) { this->parameters_.timeQuality = timeQuality; }

    virtual void SetSuperstepRemoveStrengthParameter(const double superstepRemoveStrength) {
        this->parameters_.superstepRemoveStrength = superstepRemoveStrength;
    }

    virtual std::string GetScheduleName() const { return "kl_improver_" + commCostF_.Name(); }
};

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::SetParameters(VertexIdxT<GraphT> numNodes) {
    const unsigned logNumNodes = (numNodes > 1) ? static_cast<unsigned>(std::log(numNodes)) : 1;

    // Total number of outer iterations. Proportional to sqrt N.
    parameters_.maxOuterIterations
        = static_cast<unsigned>(std::sqrt(numNodes) * (parameters_.timeQuality * 10.0) / parameters_.numParallelLoops);

    // Number of times to reset the search for violations before giving up.
    parameters_.maxNoVioaltionsRemovedBacktrackReset = parameters_.timeQuality < 0.75 ? 1 : parameters_.timeQuality < 1.0 ? 2 : 3;

    // Parameters for the superstep removal heuristic.
    parameters_.maxNoVioaltionsRemovedBacktrackForRemoveStepReset
        = 3 + static_cast<unsigned>(parameters_.superstepRemoveStrength * 7);
    parameters_.nodeMaxStepSelectionEpochs = parameters_.superstepRemoveStrength < 0.75  ? 1
                                             : parameters_.superstepRemoveStrength < 1.0 ? 2
                                                                                         : 3;
    parameters_.removeStepEpocs = static_cast<unsigned>(parameters_.superstepRemoveStrength * 4.0);

    parameters_.minInnerIterReset = static_cast<unsigned>(logNumNodes + logNumNodes * (1.0 + parameters_.timeQuality));

    if (parameters_.removeStepEpocs > 0) {
        parameters_.tryRemoveStepAfterNumOuterIterations = parameters_.maxOuterIterations / parameters_.removeStepEpocs;
    } else {
        // Effectively disable superstep removal if remove_step_epocs is 0.
        parameters_.tryRemoveStepAfterNumOuterIterations = parameters_.maxOuterIterations + 1;
    }

    unsigned i = 0;
    for (auto &thread : threadDataVec_) {
        thread.threadId = i++;
        // The number of nodes to consider in each inner iteration. Proportional to log(N).
        thread.selectionStrategy.selectionThreshold
            = static_cast<std::size_t>(std::ceil(parameters_.timeQuality * 10 * logNumNodes + logNumNodes));
    }

#ifdef KL_DEBUG_1
    std::cout << "kl set parameter, number of nodes: " << num_nodes << std::endl;
    std::cout << "max outer iterations: " << parameters.max_outer_iterations << std::endl;
    std::cout << "max inner iterations: " << parameters.max_inner_iterations_reset << std::endl;
    std::cout << "no improvement iterations reduce penalty: " << thread_data_vec[0].no_improvement_iterations_reduce_penalty
              << std::endl;
    std::cout << "selction threshold: " << thread_data_vec[0].selection_strategy.selection_threshold << std::endl;
    std::cout << "remove step epocs: " << parameters.remove_step_epocs << std::endl;
    std::cout << "try remove step after num outer iterations: " << parameters.try_remove_step_after_num_outer_iterations
              << std::endl;
    std::cout << "number of parallel loops: " << parameters.num_parallel_loops << std::endl;
#endif
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::UpdateNodeWorkAffinity(
    NodeSelectionContainerT &nodes,
    KlMove move,
    const PreMoveWorkData<WorkWeightT> &prevWorkData,
    std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain) {
    const size_t activeCount = nodes.Size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = nodes.GetSelectedNodes()[i];

        KlGainUpdateInfo updateInfo = UpdateNodeWorkAffinityAfterMove(node, move, prevWorkData, nodes.At(node));
        if (updateInfo.updateFromStep || updateInfo.updateToStep) {
            recomputeMaxGain[node] = updateInfo;
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::UpdateMaxGain(
    KlMove move, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain, ThreadSearchContext &threadData) {
    for (auto &pair : recomputeMaxGain) {
        if (pair.second.fullUpdate) {
            RecomputeNodeMaxGain(pair.first, threadData.affinityTable, threadData);
        } else {
            if (pair.second.updateEntireFromStep) {
                UpdateBestMove(pair.first, move.fromStep, threadData.affinityTable, threadData);
            } else if (pair.second.updateFromStep && IsCompatible(pair.first, move.fromProc)) {
                UpdateBestMove(pair.first, move.fromStep, move.fromProc, threadData.affinityTable, threadData);
            }

            if (move.fromStep != move.toStep || not pair.second.updateEntireFromStep) {
                if (pair.second.updateEntireToStep) {
                    UpdateBestMove(pair.first, move.toStep, threadData.affinityTable, threadData);
                } else if (pair.second.updateToStep && IsCompatible(pair.first, move.toProc)) {
                    UpdateBestMove(pair.first, move.toStep, move.toProc, threadData.affinityTable, threadData);
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::ComputeWorkAffinity(
    VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData) {
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
    const WorkWeightT vertexWeight = graph_->VertexWorkWeight(node);

    unsigned step = (nodeStep > WindowSize) ? (nodeStep - WindowSize) : 0;
    for (unsigned idx = threadData.StartIdx(nodeStep); idx < threadData.EndIdx(nodeStep); ++idx, ++step) {
        if (idx == WindowSize) {
            continue;
        }

        const CostT maxWorkForStep = static_cast<CostT>(activeSchedule_.GetStepMaxWork(step));

        for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
            const WorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(step, proc);
            const CostT workDiff = static_cast<CostT>(newWeight) - maxWorkForStep;
            affinityTableNode[proc][idx] = std::max(0.0, workDiff);
        }
    }

    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const WorkWeightT maxWorkForStep = activeSchedule_.GetStepMaxWork(nodeStep);
    const bool isSoleMaxProcessor = (activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                    && (maxWorkForStep == activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc));

    const CostT nodeProcAffinity
        = isSoleMaxProcessor ? std::min(vertexWeight, maxWorkForStep - activeSchedule_.GetStepSecondMaxWork(nodeStep)) : 0.0;
    affinityTableNode[nodeProc][WindowSize] = nodeProcAffinity;

    for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
        if (proc == nodeProc) {
            continue;
        }

        const WorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
        affinityTableNode[proc][WindowSize] = ComputeSameStepAffinity(maxWorkForStep, newWeight, nodeProcAffinity);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::ProcessWorkUpdateStep(
    VertexType node,
    unsigned nodeStep,
    unsigned nodeProc,
    WorkWeightT vertexWeight,
    unsigned moveStep,
    unsigned moveProc,
    WorkWeightT moveCorrectionNodeWeight,
    const WorkWeightT prevMoveStepMaxWork,
    const WorkWeightT prevMoveStepSecondMaxWork,
    unsigned prevMoveStepMaxWorkProcessorCount,
    bool &updateStep,
    bool &updateEntireStep,
    bool &fullUpdate,
    std::vector<std::vector<CostT>> &affinityTableNode) {
    const unsigned lowerBound = moveStep > WindowSize ? moveStep - WindowSize : 0;
    if (lowerBound <= nodeStep && nodeStep <= moveStep + WindowSize) {
        updateStep = true;
        if (nodeStep == moveStep) {
            const WorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(moveStep);
            const WorkWeightT newSecondMaxWeight = activeSchedule_.GetStepSecondMaxWork(moveStep);
            const WorkWeightT newStepProcWork = activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);

            const WorkWeightT prevStepProcWork = (nodeProc == moveProc) ? newStepProcWork + moveCorrectionNodeWeight
                                                                        : newStepProcWork;
            const bool prevIsSoleMaxProcessor = (prevMoveStepMaxWorkProcessorCount == 1)
                                                && (prevMoveStepMaxWork == prevStepProcWork);
            const CostT prevNodeProcAffinity
                = prevIsSoleMaxProcessor ? std::min(vertexWeight, prevMoveStepMaxWork - prevMoveStepSecondMaxWork) : 0.0;

            const bool newIsSoleMaxProcessor = (activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                               && (newMaxWeight == newStepProcWork);
            const CostT newNodeProcAffinity = newIsSoleMaxProcessor ? std::min(vertexWeight, newMaxWeight - newSecondMaxWeight)
                                                                    : 0.0;

            const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
            const bool updateNodeProcAffinity = std::abs(diff) > epsilon;
            if (updateNodeProcAffinity) {
                fullUpdate = true;
                affinityTableNode[nodeProc][WindowSize] += diff;
            }

            if ((prevMoveStepMaxWork != newMaxWeight) || updateNodeProcAffinity) {
                updateEntireStep = true;

                for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                    if ((proc == nodeProc) || (proc == moveProc)) {
                        continue;
                    }

                    const WorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
                    const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMoveStepMaxWork, newWeight, prevNodeProcAffinity);
                    const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                    affinityTableNode[proc][WindowSize] += (otherAffinity - prevOtherAffinity);
                }
            }

            if (nodeProc != moveProc && IsCompatible(node, moveProc)) {
                const WorkWeightT prevNewWeight
                    = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc) + moveCorrectionNodeWeight;
                const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMoveStepMaxWork, prevNewWeight, prevNodeProcAffinity);
                const WorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc);
                const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                affinityTableNode[moveProc][WindowSize] += (otherAffinity - prevOtherAffinity);
            }

        } else {
            const WorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(moveStep);
            const unsigned idx = RelStepIdx(nodeStep, moveStep);
            if (prevMoveStepMaxWork != newMaxWeight) {
                updateEntireStep = true;

                // update moving to all procs with special for move_proc
                for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                    const WorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(moveStep, proc);
                    if (proc != moveProc) {
                        const CostT prevAffinity = prevMoveStepMaxWork < newWeight
                                                       ? static_cast<CostT>(newWeight) - static_cast<CostT>(prevMoveStepMaxWork)
                                                       : 0.0;
                        const CostT newAffinity
                            = newMaxWeight < newWeight ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight) : 0.0;
                        affinityTableNode[proc][idx] += newAffinity - prevAffinity;

                    } else {
                        const WorkWeightT prevNewWeight
                            = vertexWeight + activeSchedule_.GetStepProcessorWork(moveStep, proc) + moveCorrectionNodeWeight;
                        const CostT prevAffinity = prevMoveStepMaxWork < prevNewWeight
                                                       ? static_cast<CostT>(prevNewWeight) - static_cast<CostT>(prevMoveStepMaxWork)
                                                       : 0.0;

                        const CostT newAffinity
                            = newMaxWeight < newWeight ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight) : 0.0;
                        affinityTableNode[proc][idx] += newAffinity - prevAffinity;
                    }
                }
            } else {
                // update only move_proc
                if (IsCompatible(node, moveProc)) {
                    const WorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(moveStep, moveProc);
                    const WorkWeightT prevNewWeight = newWeight + moveCorrectionNodeWeight;
                    const CostT prevAffinity = prevMoveStepMaxWork < prevNewWeight
                                                   ? static_cast<CostT>(prevNewWeight) - static_cast<CostT>(prevMoveStepMaxWork)
                                                   : 0.0;

                    const CostT newAffinity
                        = newMaxWeight < newWeight ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight) : 0.0;
                    affinityTableNode[moveProc][idx] += newAffinity - prevAffinity;
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::SelectNodesCheckRemoveSuperstep(
    unsigned &stepToRemove, ThreadSearchContext &threadData) {
    if (threadData.stepSelectionEpochCounter >= parameters_.nodeMaxStepSelectionEpochs || threadData.NumSteps() < 3) {
        return false;
    }

    for (stepToRemove = threadData.stepSelectionCounter; stepToRemove <= threadData.endStep; stepToRemove++) {
        assert(stepToRemove >= threadData.startStep && stepToRemove <= threadData.endStep);
#ifdef KL_DEBUG
        std::cout << "Checking to remove step " << step_to_remove << "/" << thread_data.end_step << std::endl;
#endif
        if (CheckRemoveSuperstep(stepToRemove)) {
#ifdef KL_DEBUG
            std::cout << "Checking to scatter step " << step_to_remove << "/" << thread_data.end_step << std::endl;
#endif
            assert(stepToRemove >= threadData.startStep && stepToRemove <= threadData.endStep);
            if (ScatterNodesSuperstep(stepToRemove, threadData)) {
                threadData.stepSelectionCounter = stepToRemove + 1;

                if (threadData.stepSelectionCounter > threadData.endStep) {
                    threadData.stepSelectionCounter = threadData.startStep;
                    threadData.stepSelectionEpochCounter++;
                }
                return true;
            }
        }
    }

    threadData.stepSelectionEpochCounter++;
    threadData.stepSelectionCounter = threadData.startStep;
    return false;
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::CheckRemoveSuperstep(unsigned step) {
    if (activeSchedule_.NumSteps() < 2) {
        return false;
    }

    if (activeSchedule_.GetStepMaxWork(step) < instance_->SynchronisationCosts()) {
        return true;
    }

    return false;
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::ResetInnerSearchStructures(
    ThreadSearchContext &threadData) const {
    threadData.unlockEdgeBacktrackCounter = threadData.unlockEdgeBacktrackCounterReset;
    threadData.maxInnerIterations = parameters_.maxInnerIterationsReset;
    threadData.maxNoVioaltionsRemovedBacktrack = parameters_.maxNoVioaltionsRemovedBacktrackReset;
    threadData.averageGain = 0.0;
    threadData.affinityTable.ResetNodeSelection();
    threadData.maxGainHeap.Clear();
    threadData.lockManager.Clear();
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::IsLocalSearchBlocked(
    ThreadSearchContext &threadData) {
    for (const auto &pair : threadData.activeScheduleData.newViolations) {
        if (threadData.lockManager.IsLocked(pair.first)) {
            return true;
        }
    }
    return false;
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::InitializeDatastructures(
    BspSchedule<GraphT> &schedule) {
    inputSchedule_ = &schedule;
    instance_ = &schedule.GetInstance();
    graph_ = &instance_->GetComputationalDag();

    activeSchedule_.Initialize(schedule);

    procRange_.Initialize(*instance_);
    commCostF_.Initialize(activeSchedule_, procRange_);
    const CostT initialCost = commCostF_.ComputeScheduleCost();
    activeSchedule_.SetCost(initialCost);

    for (auto &tData : threadDataVec_) {
        tData.affinityTable.Initialize(activeSchedule_, tData.selectionStrategy.selectionThreshold);
        tData.lockManager.Initialize(graph_->NumVertices());
        tData.rewardPenaltyStrat.Initialize(
            activeSchedule_, commCostF_.GetMaxCommWeightMultiplied(), activeSchedule_.GetMaxWorkWeight());
        tData.selectionStrategy.Initialize(activeSchedule_, gen_, tData.startStep, tData.endStep);

        tData.localAffinityTable.resize(instance_->NumberOfProcessors());
        for (unsigned i = 0; i < instance_->NumberOfProcessors(); ++i) {
            tData.localAffinityTable[i].resize(windowRange);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::UpdateAvgGain(const CostT gain,
                                                                                                const unsigned numIter,
                                                                                                double &averageGain) {
    averageGain = static_cast<double>((averageGain * numIter + gain)) / (numIter + 1.0);
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::InsertGainHeap(ThreadSearchContext &threadData) {
    const size_t activeCount = threadData.affinityTable.Size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = threadData.affinityTable.GetSelectedNodes()[i];
        ComputeNodeAffinities(node, threadData.affinityTable.At(node), threadData);
        const auto bestMove = ComputeBestMove<true>(node, threadData.affinityTable[node], threadData);
        threadData.maxGainHeap.Push(node, bestMove);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::InsertNewNodesGainHeap(
    std::vector<VertexType> &newNodes, NodeSelectionContainerT &nodes, ThreadSearchContext &threadData) {
    for (const auto &node : newNodes) {
        nodes.Insert(node);
        ComputeNodeAffinities(node, threadData.affinityTable.At(node), threadData);
        const auto bestMove = ComputeBestMove<true>(node, threadData.affinityTable[node], threadData);
        threadData.maxGainHeap.Push(node, bestMove);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::CleanupDatastructures() {
    threadDataVec_.clear();
    activeSchedule_.clear();
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::PrintHeap(HeapDatastructure &maxGainHeap) const {
    if (maxGainHeap.IsEmpty()) {
        std::cout << "heap is empty" << std::endl;
        return;
    }
    HeapDatastructure tempHeap = maxGainHeap;    // requires copy constructor

    std::cout << "heap current size: " << tempHeap.Size() << std::endl;
    const auto &topVal = tempHeap.GetValue(tempHeap.Top());
    std::cout << "heap top node " << topVal.node << " gain " << topVal.gain << std::endl;

    unsigned count = 0;
    while (!tempHeap.IsEmpty() && count++ < 15) {
        const auto &val = tempHeap.GetValue(tempHeap.Top());
        std::cout << "node " << val.node << " gain " << val.gain << " to proc " << val.toProc << " to step " << val.toStep
                  << std::endl;
        tempHeap.Pop();
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);

    if ((nodeProc == proc) && (nodeStep == step)) {
        return;
    }

    KlMove nodeMove = threadData.maxGainHeap.GetValue(node);
    CostT maxGain = nodeMove.gain;

    unsigned maxProc = nodeMove.toProc;
    unsigned maxStep = nodeMove.toStep;

    if ((maxStep == step) && (maxProc == proc)) {
        RecomputeNodeMaxGain(node, affinityTable, threadData);
    } else {
        if constexpr (ActiveScheduleT::useMemoryConstraint) {
            if (not activeSchedule_.memoryConstraint.CanMove(node, proc, step)) {
                return;
            }
        }
        const unsigned idx = RelStepIdx(nodeStep, step);
        const CostT gain = affinityTable[node][nodeProc][WindowSize] - affinityTable[node][proc][idx];
        if (gain > maxGain) {
            maxGain = gain;
            maxProc = proc;
            maxStep = step;
        }

        const CostT diff = maxGain - nodeMove.gain;
        if ((std::abs(diff) > epsilon) || (maxProc != nodeMove.toProc) || (maxStep != nodeMove.toStep)) {
            nodeMove.gain = maxGain;
            nodeMove.toProc = maxProc;
            nodeMove.toStep = maxStep;
            threadData.maxGainHeap.Update(node, nodeMove);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned WindowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);

    KlMove nodeMove = threadData.maxGainHeap.GetValue(node);
    CostT maxGain = nodeMove.gain;

    unsigned maxProc = nodeMove.toProc;
    unsigned maxStep = nodeMove.toStep;

    if (maxStep == step) {
        RecomputeNodeMaxGain(node, affinityTable, threadData);
    } else {
        if (nodeStep != step) {
            const unsigned idx = RelStepIdx(nodeStep, step);
            for (const unsigned p : procRange_.CompatibleProcessorsVertex(node)) {
                if constexpr (ActiveScheduleT::useMemoryConstraint) {
                    if (not activeSchedule_.memoryConstraint.CanMove(node, p, step)) {
                        continue;
                    }
                }
                const CostT gain = affinityTable[node][nodeProc][WindowSize] - affinityTable[node][p][idx];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = p;
                    maxStep = step;
                }
            }
        } else {
            for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                if (proc == nodeProc) {
                    continue;
                }
                if constexpr (ActiveScheduleT::useMemoryConstraint) {
                    if (not activeSchedule_.memoryConstraint.CanMove(node, proc, step)) {
                        continue;
                    }
                }
                const CostT gain = affinityTable[node][nodeProc][WindowSize] - affinityTable[node][proc][WindowSize];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = proc;
                    maxStep = step;
                }
            }
        }

        const CostT diff = maxGain - nodeMove.gain;
        if ((std::abs(diff) > epsilon) || (maxProc != nodeMove.toProc) || (maxStep != nodeMove.toStep)) {
            nodeMove.gain = maxGain;
            nodeMove.toProc = maxProc;
            nodeMove.toStep = maxStep;
            threadData.maxGainHeap.Update(node, nodeMove);
        }
    }
}

}    // namespace osp
