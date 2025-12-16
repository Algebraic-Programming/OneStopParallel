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
    double timeQuality_ = 0.8;
    double superstepRemoveStrength_ = 0.5;
    unsigned numParallelLoops_ = 4;

    unsigned maxInnerIterationsReset_ = 500;
    unsigned maxNoImprovementIterations_ = 50;

    constexpr static unsigned abortScatterNodesViolationThreshold_ = 500;
    constexpr static unsigned initialViolationThreshold_ = 250;

    unsigned maxNoVioaltionsRemovedBacktrackReset_;
    unsigned removeStepEpocs_;
    unsigned nodeMaxStepSelectionEpochs_;
    unsigned maxNoVioaltionsRemovedBacktrackForRemoveStepReset_;
    unsigned maxOuterIterations_;
    unsigned tryRemoveStepAfterNumOuterIterations_;
    unsigned minInnerIterReset_;

    unsigned threadMinRange_ = 8;
    unsigned threadRangeGap_ = 0;
};

template <typename VertexType>
struct KlUpdateInfo {
    VertexType node_ = 0;

    bool fullUpdate_ = false;
    bool updateFromStep_ = false;
    bool updateToStep_ = false;
    bool updateEntireToStep_ = false;
    bool updateEntireFromStep_ = false;

    KlUpdateInfo() = default;

    KlUpdateInfo(VertexType n) : node_(n), fullUpdate_(false), updateEntireToStep_(false), updateEntireFromStep_(false) {}

    KlUpdateInfo(VertexType n, bool full)
        : node_(n), fullUpdate_(full), updateEntireToStep_(false), updateEntireFromStep_(false) {}
};

template <typename GraphT,
          typename CommCostFunctionT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          typename CostT = double>
class KlImprover : public ImprovementScheduler<GraphT> {
    static_assert(isDirectedGraphEdgeDescV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    static_assert(hasHashableEdgeDescV<GraphT>, "Graph_t must satisfy the HasHashableEdgeDesc concept");
    static_assert(isComputationalDagV<GraphT>, "Graph_t must satisfy the computational_dag concept");

  protected:
    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool enableQuickMoves_ = true;
    constexpr static bool enablePreresolvingViolations_ = true;
    constexpr static double epsilon_ = 1e-9;

    using VertexMemWeightT = osp::VMemwT<GraphT>;
    using VertexCommWeightT = osp::VCommwT<GraphT>;
    using VertexWorkWeightT = osp::VWorkwT<GraphT>;
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    using KlMove = KlMoveStruct<CostT, VertexType>;
    using HeapDatastructure = MaxPairingHeap<VertexType, KlMove>;
    using ActiveScheduleT = KlActiveSchedule<GraphT, CostT, MemoryConstraintT>;
    using NodeSelectionContainerT = AdaptiveAffinityTable<GraphT, CostT, ActiveScheduleT, windowSize>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;

    struct ThreadSearchContext {
        unsigned threadId_ = 0;
        unsigned startStep_ = 0;
        unsigned endStep_ = 0;
        unsigned originalEndStep_ = 0;

        VectorVertexLockManager<VertexType> lockManager_;
        HeapDatastructure maxGainHeap_;
        NodeSelectionContainerT affinityTable_;
        std::vector<std::vector<CostT>> localAffinityTable_;
        RewardPenaltyStrategy<CostT, CommCostFunctionT, ActiveScheduleT> rewardPenaltyStrat_;
        VertexSelectionStrategy<GraphT, NodeSelectionContainerT, ActiveScheduleT> selectionStrategy_;
        ThreadLocalActiveScheduleData<GraphT, CostT> activeScheduleData_;

        double averageGain_ = 0.0;
        unsigned maxInnerIterations_ = 0;
        unsigned noImprovementIterationsReducePenalty_ = 0;
        unsigned minInnerIter_ = 0;
        unsigned noImprovementIterationsIncreaseInnerIter_ = 0;
        unsigned stepSelectionEpochCounter_ = 0;
        unsigned stepSelectionCounter_ = 0;
        unsigned stepToRemove_ = 0;
        unsigned localSearchStartStep_ = 0;
        unsigned unlockEdgeBacktrackCounter_ = 0;
        unsigned unlockEdgeBacktrackCounterReset_ = 0;
        unsigned maxNoVioaltionsRemovedBacktrack_ = 0;

        inline unsigned NumSteps() const { return endStep_ - startStep_ + 1; }

        inline unsigned StartIdx(const unsigned nodeStep) const {
            return nodeStep < startStep_ + windowSize ? windowSize - (nodeStep - startStep_) : 0;
        }

        inline unsigned EndIdx(unsigned nodeStep) const {
            return nodeStep + windowSize <= endStep_ ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep_);
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
        return (moveStep >= nodeStep) ? ((moveStep - nodeStep) + windowSize) : (windowSize - (nodeStep - moveStep));
    }

    inline bool IsCompatible(VertexType node, unsigned proc) const {
        return activeSchedule_.GetInstance().IsCompatible(node, proc);
    }

    void SetStartStep(const unsigned step, ThreadSearchContext &threadData) {
        threadData.startStep_ = step;
        threadData.stepToRemove_ = step;
        threadData.stepSelectionCounter_ = step;

        threadData.averageGain_ = 0.0;
        threadData.maxInnerIterations_ = parameters_.maxInnerIterationsReset_;
        threadData.noImprovementIterationsReducePenalty_ = parameters_.maxNoImprovementIterations_ / 5;
        threadData.minInnerIter_ = parameters_.minInnerIterReset_;
        threadData.stepSelectionEpochCounter_ = 0;
        threadData.noImprovementIterationsIncreaseInnerIter_ = 10;
        threadData.unlockEdgeBacktrackCounterReset_ = 0;
        threadData.unlockEdgeBacktrackCounter_ = threadData.unlockEdgeBacktrackCounterReset_;
        threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackReset_;
    }

    KlMove GetBestMove(NodeSelectionContainerT &affinityTable,
                       VectorVertexLockManager<VertexType> &lockManager,
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
            if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                if (not activeSchedule_.memoryConstraint_.CanMove(node, p, nodeStep + idx - windowSize)) {
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

    template <bool moveToSameSuperStep>
    KlMove ComputeBestMove(VertexType node,
                           const std::vector<std::vector<CostT>> &affinityTableNode,
                           ThreadSearchContext &threadData) {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);

        CostT maxGain = std::numeric_limits<CostT>::lowest();

        unsigned maxProc = std::numeric_limits<unsigned>::max();
        unsigned maxStep = std::numeric_limits<unsigned>::max();

        const CostT affinityCurrentProcStep = affinityTableNode[nodeProc][windowSize];

        unsigned idx = threadData.StartIdx(nodeStep);
        for (; idx < windowSize; idx++) {
            ProcessOtherStepsBestMove(idx, nodeStep, node, affinityCurrentProcStep, maxGain, maxProc, maxStep, affinityTableNode);
        }

        if constexpr (moveToSameSuperStep) {
            for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                if (proc == nodeProc) {
                    continue;
                }

                if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                    if (not activeSchedule_.memoryConstraint_.CanMove(node, proc, nodeStep + idx - windowSize)) {
                        continue;
                    }
                }

                const CostT gain = affinityCurrentProcStep - affinityTableNode[proc][windowSize];
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

        return KlMove(node, maxGain, nodeProc, nodeStep, maxProc, nodeStep + maxStep - windowSize);
    }

    KlGainUpdateInfo UpdateNodeWorkAffinityAfterMove(VertexType node,
                                                     KlMove move,
                                                     const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                                                     std::vector<std::vector<CostT>> &affinityTableNode) {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);

        KlGainUpdateInfo updateInfo(node);

        if (move.fromStep_ == move.toStep_) {
            const unsigned lowerBound = move.fromStep_ > windowSize ? move.fromStep_ - windowSize : 0;
            if (lowerBound <= nodeStep && nodeStep <= move.fromStep_ + windowSize) {
                updateInfo.updateFromStep_ = true;
                updateInfo.updateToStep_ = true;

                const VertexWorkWeightT prevMaxWork = prevWorkData.fromStepMaxWork_;
                const VertexWorkWeightT prevSecondMaxWork = prevWorkData.fromStepSecondMaxWork_;

                if (nodeStep == move.fromStep_) {
                    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
                    const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(move.fromStep_);
                    const VertexWorkWeightT newSecondMaxWeight = activeSchedule_.GetStepSecondMaxWork(move.fromStep_);
                    const VertexWorkWeightT newStepProcWork = activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);
                    const VertexWorkWeightT prevStepProcWork
                        = (nodeProc == move.fromProc_) ? newStepProcWork + graph_->VertexWorkWeight(move.node_)
                          : (nodeProc == move.toProc_) ? newStepProcWork - graph_->VertexWorkWeight(move.node_)
                                                       : newStepProcWork;
                    const bool prevIsSoleMaxProcessor = (prevWorkData.fromStepMaxWorkProcessorCount_ == 1)
                                                        && (prevMaxWork == prevStepProcWork);
                    const CostT prevNodeProcAffinity
                        = prevIsSoleMaxProcessor ? std::min(vertexWeight, prevMaxWork - prevSecondMaxWork) : 0.0;
                    const bool newIsSoleMaxProcessor = (activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                                       && (newMaxWeight == newStepProcWork);
                    const CostT newNodeProcAffinity
                        = newIsSoleMaxProcessor ? std::min(vertexWeight, newMaxWeight - newSecondMaxWeight) : 0.0;

                    const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
                    if (std::abs(diff) > epsilon_) {
                        updateInfo.fullUpdate_ = true;
                        affinityTableNode[nodeProc][windowSize] += diff;    // Use the pre-calculated diff
                    }

                    if ((prevMaxWork != newMaxWeight) || updateInfo.fullUpdate_) {
                        updateInfo.updateEntireFromStep_ = true;

                        for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                            if ((proc == nodeProc) || (proc == move.fromProc_) || (proc == move.toProc_)) {
                                continue;
                            }

                            const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
                            const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, newWeight, prevNodeProcAffinity);
                            const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                            affinityTableNode[proc][windowSize] += (otherAffinity - prevOtherAffinity);
                        }
                    }

                    if (nodeProc != move.fromProc_ && IsCompatible(node, move.fromProc_)) {
                        const VertexWorkWeightT prevNewWeight = vertexWeight
                                                                + activeSchedule_.GetStepProcessorWork(nodeStep, move.fromProc_)
                                                                + graph_->VertexWorkWeight(move.node_);
                        const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, prevNewWeight, prevNodeProcAffinity);
                        const VertexWorkWeightT newWeight
                            = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, move.fromProc_);
                        const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
                        affinityTableNode[move.fromProc_][windowSize] += (otherAffinity - prevOtherAffinity);
                    }

                    if (nodeProc != move.toProc_ && IsCompatible(node, move.toProc_)) {
                        const VertexWorkWeightT prevNewWeight = vertexWeight
                                                                + activeSchedule_.GetStepProcessorWork(nodeStep, move.toProc_)
                                                                - graph_->VertexWorkWeight(move.node_);
                        const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, prevNewWeight, prevNodeProcAffinity);
                        const VertexWorkWeightT newWeight
                            = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, move.toProc_);
                        const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
                        affinityTableNode[move.toProc_][windowSize] += (otherAffinity - prevOtherAffinity);
                    }

                } else {
                    const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(move.fromStep_);
                    const unsigned idx = RelStepIdx(nodeStep, move.fromStep_);
                    if (prevMaxWork != newMaxWeight) {
                        updateInfo.updateEntireFromStep_ = true;
                        // update moving to all procs with special for move.fromProc_
                        for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                            const VertexWorkWeightT newWeight
                                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, proc);
                            if (proc == move.fromProc_) {
                                const VertexWorkWeightT prevNewWeight
                                    = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, proc)
                                      + graph_->VertexWorkWeight(move.node_);
                                const CostT prevAffinity = prevMaxWork < prevNewWeight ? static_cast<CostT>(prevNewWeight)
                                                                                             - static_cast<CostT>(prevMaxWork)
                                                                                       : 0.0;
                                const CostT newAffinity = newMaxWeight < newWeight
                                                              ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                                              : 0.0;
                                affinityTableNode[proc][idx] += newAffinity - prevAffinity;
                            } else if (proc == move.toProc_) {
                                const VertexWorkWeightT prevNewWeight = vertexWeight
                                                                        + activeSchedule_.GetStepProcessorWork(move.toStep_, proc)
                                                                        - graph_->VertexWorkWeight(move.node_);
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
                        // update only move.fromProc_ and move.toProc_
                        if (IsCompatible(node, move.fromProc_)) {
                            const VertexWorkWeightT fromNewWeight
                                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, move.fromProc_);
                            const VertexWorkWeightT fromPrevNewWeight = fromNewWeight + graph_->VertexWorkWeight(move.node_);
                            const CostT fromPrevAffinity = prevMaxWork < fromPrevNewWeight ? static_cast<CostT>(fromPrevNewWeight)
                                                                                                 - static_cast<CostT>(prevMaxWork)
                                                                                           : 0.0;

                            const CostT fromNewAffinity = newMaxWeight < fromNewWeight ? static_cast<CostT>(fromNewWeight)
                                                                                             - static_cast<CostT>(newMaxWeight)
                                                                                       : 0.0;
                            affinityTableNode[move.fromProc_][idx] += fromNewAffinity - fromPrevAffinity;
                        }

                        if (IsCompatible(node, move.toProc_)) {
                            const VertexWorkWeightT toNewWeight
                                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.toStep_, move.toProc_);
                            const VertexWorkWeightT toPrevNewWeight = toNewWeight - graph_->VertexWorkWeight(move.node_);
                            const CostT toPrevAffinity = prevMaxWork < toPrevNewWeight ? static_cast<CostT>(toPrevNewWeight)
                                                                                             - static_cast<CostT>(prevMaxWork)
                                                                                       : 0.0;

                            const CostT toNewAffinity = newMaxWeight < toNewWeight
                                                            ? static_cast<CostT>(toNewWeight) - static_cast<CostT>(newMaxWeight)
                                                            : 0.0;
                            affinityTableNode[move.toProc_][idx] += toNewAffinity - toPrevAffinity;
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
                                  move.fromStep_,
                                  move.fromProc_,
                                  graph_->VertexWorkWeight(move.node_),
                                  prevWorkData.fromStepMaxWork_,
                                  prevWorkData.fromStepSecondMaxWork_,
                                  prevWorkData.fromStepMaxWorkProcessorCount_,
                                  updateInfo.updateFromStep_,
                                  updateInfo.updateEntireFromStep_,
                                  updateInfo.fullUpdate_,
                                  affinityTableNode);
            ProcessWorkUpdateStep(node,
                                  nodeStep,
                                  nodeProc,
                                  vertexWeight,
                                  move.toStep_,
                                  move.toProc_,
                                  -graph_->VertexWorkWeight(move.node_),
                                  prevWorkData.toStepMaxWork_,
                                  prevWorkData.toStepSecondMaxWork_,
                                  prevWorkData.toStepMaxWorkProcessorCount_,
                                  updateInfo.updateToStep_,
                                  updateInfo.updateEntireToStep_,
                                  updateInfo.fullUpdate_,
                                  affinityTableNode);
        }

        return updateInfo;
    }

    void ProcessWorkUpdateStep(VertexType node,
                               unsigned nodeStep,
                               unsigned nodeProc,
                               VertexWorkWeightT vertexWeight,
                               unsigned moveStep,
                               unsigned moveProc,
                               VertexWorkWeightT moveCorrectionNodeWeight,
                               const VertexWorkWeightT prevMoveStepMaxWork,
                               const VertexWorkWeightT prevMoveStepSecondMaxWork,
                               unsigned prevMoveStepMaxWorkProcessorCount,
                               bool &updateStep,
                               bool &updateEntireStep,
                               bool &fullUpdate,
                               std::vector<std::vector<CostT>> &affinityTableNode);
    void UpdateNodeWorkAffinity(NodeSelectionContainerT &nodes,
                                KlMove move,
                                const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                                std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain);
    void UpdateBestMove(
        VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateBestMove(VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateMaxGain(KlMove move, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain, ThreadSearchContext &threadData);
    void ComputeWorkAffinity(VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData);

    inline void RecomputeNodeMaxGain(VertexType node, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
        const auto bestMove = ComputeBestMove<true>(node, affinityTable[node], threadData);
        threadData.maxGainHeap_.Update(node, bestMove);
    }

    inline CostT ComputeSameStepAffinity(const VertexWorkWeightT &maxWorkForStep,
                                         const VertexWorkWeightT &newWeight,
                                         const CostT &nodeProcAffinity) {
        const CostT maxWorkAfterRemoval = static_cast<CostT>(maxWorkForStep) - nodeProcAffinity;
        if (newWeight > maxWorkAfterRemoval) {
            return newWeight - maxWorkAfterRemoval;
        }
        return 0.0;
    }

    inline CostT ApplyMove(KlMove move, ThreadSearchContext &threadData) {
        activeSchedule_.ApplyMove(move, threadData.activeScheduleData_);
        commCostF_.UpdateDatastructureAfterMove(move, threadData.startStep_, threadData.endStep_);
        CostT changeInCost = -move.gain_;
        changeInCost += static_cast<CostT>(threadData.activeScheduleData_.resolvedViolations_.size())
                        * threadData.rewardPenaltyStrat_.reward_;
        changeInCost
            -= static_cast<CostT>(threadData.activeScheduleData_.newViolations_.size()) * threadData.rewardPenaltyStrat_.penalty_;

#ifdef KL_DEBUG
        std::cout << "penalty: " << threadData.rewardPenaltyStrat_.penalty_
                  << " num violations: " << threadData.activeScheduleData_.currentViolations_.size()
                  << " num new violations: " << threadData.activeScheduleData_.newViolations_.size()
                  << ", num resolved violations: " << threadData.activeScheduleData_.resolvedViolations_.size()
                  << ", reward: " << threadData.rewardPenaltyStrat_.reward_ << std::endl;
        std::cout << "apply move, previous cost: " << threadData.activeScheduleData_.cost_
                  << ", new cost: " << threadData.activeScheduleData_.cost_ + changeInCost << ", "
                  << (threadData.activeScheduleData_.feasible_ ? "feasible," : "infeasible,") << std::endl;
#endif

        threadData.activeScheduleData_.UpdateCost(changeInCost);

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

        const size_t numAppliedMoves = threadData.activeScheduleData_.appliedMoves_.size() - 1;
        const CostT savedCost = threadData.activeScheduleData_.cost_ - changeInCost;

        std::unordered_set<VertexType> localLock;
        localLock.insert(bestMoveNode);
        std::vector<VertexType> quickMovesStack;
        quickMovesStack.reserve(10 + threadData.activeScheduleData_.newViolations_.size() * 2);

        for (const auto &keyValuePair : threadData.activeScheduleData_.newViolations_) {
            const auto &key = keyValuePair.first;
            quickMovesStack.push_back(key);
        }

        while (quickMovesStack.size() > 0) {
            auto nextNodeToMove = quickMovesStack.back();
            quickMovesStack.pop_back();

            threadData.rewardPenaltyStrat_.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
            ComputeNodeAffinities(nextNodeToMove, threadData.localAffinityTable_, threadData);
            KlMove bestQuickMove = ComputeBestMove<true>(nextNodeToMove, threadData.localAffinityTable_, threadData);

            localLock.insert(nextNodeToMove);
            if (bestQuickMove.gain_ <= std::numeric_limits<CostT>::lowest()) {
                continue;
            }

#ifdef KL_DEBUG
            std::cout << " >>> move node " << bestQuickMove.node_ << " with gain " << bestQuickMove.gain_
                      << ", from proc|step: " << bestQuickMove.fromProc_ << "|" << bestQuickMove.fromStep_
                      << " to: " << bestQuickMove.toProc_ << "|" << bestQuickMove.toStep_ << std::endl;
#endif

            ApplyMove(bestQuickMove, threadData);
            innerIter++;

            if (threadData.activeScheduleData_.newViolations_.size() > 0) {
                bool abort = false;

                for (const auto &keyValuePair : threadData.activeScheduleData_.newViolations_) {
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

            } else if (threadData.activeScheduleData_.feasible_) {
                break;
            }
        }

        if (!threadData.activeScheduleData_.feasible_) {
            activeSchedule_.RevertScheduleToBound(numAppliedMoves,
                                                  savedCost,
                                                  true,
                                                  commCostF_,
                                                  threadData.activeScheduleData_,
                                                  threadData.startStep_,
                                                  threadData.endStep_);
#ifdef KL_DEBUG
            std::cout << "Ending quick moves sequence with infeasible solution." << std::endl;
#endif
        }
#ifdef KL_DEBUG
        else {
            std::cout << "Ending quick moves sequence with feasible solution." << std::endl;
        }
#endif

        threadData.affinityTable_.Trim();
        threadData.maxGainHeap_.Clear();
        threadData.rewardPenaltyStrat_.InitRewardPenalty(1.0);
        InsertGainHeap(threadData);    // Re-initialize the heap with the current state
    }

    void ResolveViolations(ThreadSearchContext &threadData) {
        auto &currentViolations = threadData.activeScheduleData_.currentViolations_;
        unsigned numViolations = static_cast<unsigned>(currentViolations.size());
        if (numViolations > 0) {
#ifdef KL_DEBUG_1
            std::cout << "thread " << threadData.threadId_ << ", Starting preresolving violations with " << numViolations
                      << " initial violations" << std::endl;
#endif
            threadData.rewardPenaltyStrat_.InitRewardPenalty(static_cast<double>(numViolations) + 1.0);
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
                    ComputeNodeAffinities(node, threadData.localAffinityTable_, threadData);
                    bestMove = ComputeBestMove<true>(node, threadData.localAffinityTable_, threadData);
                } else {
                    ComputeNodeAffinities(sourceV, threadData.localAffinityTable_, threadData);
                    KlMove bestSourceVMove = ComputeBestMove<true>(sourceV, threadData.localAffinityTable_, threadData);
                    ComputeNodeAffinities(targetV, threadData.localAffinityTable_, threadData);
                    KlMove bestTargetVMove = ComputeBestMove<true>(targetV, threadData.localAffinityTable_, threadData);
                    bestMove = bestTargetVMove.gain_ > bestSourceVMove.gain_ ? std::move(bestTargetVMove)
                                                                             : std::move(bestSourceVMove);
                }

                localLock.insert(bestMove.node_);
                if (bestMove.gain_ <= std::numeric_limits<CostT>::lowest()) {
                    continue;
                }

                ApplyMove(bestMove, threadData);
                threadData.affinityTable_.Insert(bestMove.node_);
#ifdef KL_DEBUG_1
                std::cout << "move node " << bestMove.node_ << " with gain " << bestMove.gain_
                          << ", from proc|step: " << bestMove.fromProc_ << "|" << bestMove.fromStep_
                          << " to: " << bestMove.toProc_ << "|" << bestMove.toStep_ << std::endl;
#endif
                const unsigned newNumViolations = static_cast<unsigned>(currentViolations.size());
                if (newNumViolations == 0) {
                    break;
                }

                if (threadData.activeScheduleData_.newViolations_.size() > 0) {
                    for (const auto &vertexEdgePair : threadData.activeScheduleData_.newViolations_) {
                        const auto &vertex = vertexEdgePair.first;
                        threadData.affinityTable_.Insert(vertex);
                    }
                }

                const double gain = static_cast<double>(numViolations) - static_cast<double>(newNumViolations);
                numViolations = newNumViolations;
                UpdateAvgGain(gain, numIter++, threadData.averageGain_);
#ifdef KL_DEBUG_1
                std::cout << "thread " << threadData.threadId_ << ",  preresolving violations with " << numViolations
                          << " violations, " << numIter << " #iterations, " << threadData.averageGain_ << " average gain"
                          << std::endl;
#endif
                if (numIter > minIter && threadData.averageGain_ < 0.0) {
                    break;
                }
            }
            threadData.averageGain_ = 0.0;
        }
    }

    void RunLocalSearch(ThreadSearchContext &threadData) {
#ifdef KL_DEBUG_1
        std::cout << "thread " << threadData.threadId_
                  << ", start local search, initial schedule cost: " << threadData.activeScheduleData_.cost_ << " with "
                  << threadData.NumSteps() << " supersteps." << std::endl;
#endif
        std::vector<VertexType> newNodes;
        std::vector<VertexType> unlockNodes;
        std::map<VertexType, KlGainUpdateInfo> recomputeMaxGain;

        const auto startTime = std::chrono::high_resolution_clock::now();

        unsigned noImprovementIterCounter = 0;
        unsigned outerIter = 0;

        for (; outerIter < parameters_.maxOuterIterations_; outerIter++) {
            CostT initialInnerIterCost = threadData.activeScheduleData_.cost_;

            ResetInnerSearchStructures(threadData);
            SelectActiveNodes(threadData);
            threadData.rewardPenaltyStrat_.InitRewardPenalty(
                static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
            InsertGainHeap(threadData);

            unsigned innerIter = 0;
            unsigned violationRemovedCount = 0;
            unsigned resetCounter = 0;
            bool iterInitalFeasible = threadData.activeScheduleData_.feasible_;

#ifdef KL_DEBUG
            std::cout << "------ start inner loop ------" << std::endl;
            std::cout << "initial node selection: {";
            for (size_t i = 0; i < threadData.affinityTable_.size(); ++i) {
                std::cout << threadData.affinityTable_.GetSelectedNodes()[i] << ", ";
            }
            std::cout << "}" << std::endl;
#endif
#ifdef KL_DEBUG_1
            if (not iterInitalFeasible) {
                std::cout << "initial solution not feasible, num violations: "
                          << threadData.activeScheduleData_.currentViolations_.size()
                          << ". Penalty: " << threadData.rewardPenaltyStrat_.penalty_
                          << ", reward: " << threadData.rewardPenaltyStrat_.reward_ << std::endl;
            }
#endif
#ifdef KL_DEBUG_COST_CHECK
            activeSchedule_.GetVectorSchedule().numberOfSupersteps = threadDataVec_[0].NumSteps();
            if (std::abs(commCostF_.ComputeScheduleCostTest() - threadData.activeScheduleData_.cost_) > 0.00001) {
                std::cout << "computed cost: " << commCostF_.ComputeScheduleCostTest()
                          << ", current cost: " << threadData.activeScheduleData_.cost_ << std::endl;
                std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
            }
            if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                if (not activeSchedule_.memoryConstraint_.SatisfiedMemoryConstraint()) {
                    std::cout << "memory constraint not satisfied" << std::endl;
                }
            }
#endif

            while (innerIter < threadData.maxInnerIterations_ && threadData.maxGainHeap_.size() > 0) {
                KlMove bestMove
                    = GetBestMove(threadData.affinityTable_,
                                  threadData.lockManager_,
                                  threadData.maxGainHeap_);    // locks bestMove.node and removes it from node_selection
                if (bestMove.gain_ <= std::numeric_limits<CostT>::lowest()) {
                    break;
                }
                UpdateAvgGain(bestMove.gain_, innerIter, threadData.averageGain_);
#ifdef KL_DEBUG
                std::cout << " >>> move node " << bestMove.node_ << " with gain " << bestMove.gain_
                          << ", from proc|step: " << bestMove.fromProc_ << "|" << bestMove.fromStep_ << " to: " << bestMove.toProc_
                          << "|" << bestMove.toStep_ << ",avg gain: " << threadData.averageGain_ << std::endl;
#endif
                if (innerIter > threadData.minInnerIter_ && threadData.averageGain_ < 0.0) {
#ifdef KL_DEBUG
                    std::cout << "Negative average gain: " << threadData.averageGain_ << ", end local search" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                if (not activeSchedule_.GetInstance().IsCompatible(bestMove.node_, bestMove.toProc_)) {
                    std::cout << "move to incompatibe node" << std::endl;
                }
#endif

                const auto prevWorkData = activeSchedule_.GetPreMoveWorkData(bestMove);
                const typename CommCostFunctionT::PreMoveCommDataT prevCommData = commCostF_.GetPreMoveCommData(bestMove);
                const CostT changeInCost = ApplyMove(bestMove, threadData);
#ifdef KL_DEBUG_COST_CHECK
                activeSchedule_.GetVectorSchedule().numberOfSupersteps = threadDataVec_[0].NumSteps();
                if (std::abs(commCostF_.ComputeScheduleCostTest() - threadData.activeScheduleData_.cost_) > 0.00001) {
                    std::cout << "computed cost: " << commCostF_.ComputeScheduleCostTest()
                              << ", current cost: " << threadData.activeScheduleData_.cost_ << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                    if (not activeSchedule_.memoryConstraint_.SatisfiedMemoryConstraint()) {
                        std::cout << "memory constraint not satisfied" << std::endl;
                    }
                }
#endif
                if constexpr (enableQuickMoves_) {
                    if (iterInitalFeasible && threadData.activeScheduleData_.newViolations_.size() > 0) {
                        RunQuickMoves(innerIter, threadData, changeInCost, bestMove.node_);
#ifdef KL_DEBUG_COST_CHECK
                        activeSchedule_.GetVectorSchedule().numberOfSupersteps = threadDataVec_[0].NumSteps();
                        if (std::abs(commCostF_.ComputeScheduleCostTest() - threadData.activeScheduleData_.cost_) > 0.00001) {
                            std::cout << "computed cost: " << commCostF_.ComputeScheduleCostTest()
                                      << ", current cost: " << threadData.activeScheduleData_.cost_ << std::endl;
                            std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<"
                                      << std::endl;
                        }
                        if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                            if (not activeSchedule_.memoryConstraint_.SatisfiedMemoryConstraint()) {
                                std::cout << "memory constraint not satisfied" << std::endl;
                            }
                        }
#endif
                        continue;
                    }
                }

                if (threadData.activeScheduleData_.currentViolations_.size() > 0) {
                    if (threadData.activeScheduleData_.resolvedViolations_.size() > 0) {
                        violationRemovedCount = 0;
                    } else {
                        violationRemovedCount++;

                        if (violationRemovedCount > 3) {
                            if (resetCounter < threadData.maxNoVioaltionsRemovedBacktrack_
                                && ((not iterInitalFeasible)
                                    || (threadData.activeScheduleData_.cost_ < threadData.activeScheduleData_.bestCost_))) {
                                threadData.affinityTable_.ResetNodeSelection();
                                threadData.maxGainHeap_.Clear();
                                threadData.lockManager_.Clear();
                                threadData.selectionStrategy_.SelectNodesViolations(
                                    threadData.affinityTable_,
                                    threadData.activeScheduleData_.currentViolations_,
                                    threadData.startStep_,
                                    threadData.endStep_);
#ifdef KL_DEBUG
                                std::cout << "Infeasible, and no violations resolved for 5 iterations, reset node selection"
                                          << std::endl;
#endif
                                threadData.rewardPenaltyStrat_.InitRewardPenalty(
                                    static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()));
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
                    if (not BlockedEdgeStrategy(bestMove.node_, unlockNodes, threadData)) {
                        break;
                    }
                }

                threadData.affinityTable_.Trim();
                UpdateAffinities(bestMove, threadData, recomputeMaxGain, newNodes, prevWorkData, prevCommData);

                for (const auto v : unlockNodes) {
                    threadData.lockManager_.Unlock(v);
                }
                newNodes.insert(newNodes.end(), unlockNodes.begin(), unlockNodes.end());
                unlockNodes.clear();

#ifdef KL_DEBUG
                std::cout << "recmopute max gain: {";
                for (const auto mapPair : recomputeMaxGain) {
                    const auto &key = mapPair.first;
                    std::cout << key << ", ";
                }
                std::cout << "}" << std::endl;
                std::cout << "new nodes: {";
                for (const auto v : newNodes) {
                    std::cout << v << ", ";
                }
                std::cout << "}" << std::endl;
#endif
#ifdef KL_DEBUG_COST_CHECK
                activeSchedule_.GetVectorSchedule().numberOfSupersteps = threadDataVec_[0].NumSteps();
                if (std::abs(commCostF_.ComputeScheduleCostTest() - threadData.activeScheduleData_.cost_) > 0.00001) {
                    std::cout << "computed cost: " << commCostF_.ComputeScheduleCostTest()
                              << ", current cost: " << threadData.activeScheduleData_.cost_ << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                    if (not activeSchedule_.memoryConstraint_.SatisfiedMemoryConstraint()) {
                        std::cout << "memory constraint not satisfied" << std::endl;
                    }
                }
#endif
                UpdateMaxGain(bestMove, recomputeMaxGain, threadData);
                InsertNewNodesGainHeap(newNodes, threadData.affinityTable_, threadData);

                recomputeMaxGain.clear();
                newNodes.clear();

                innerIter++;
            }

#ifdef KL_DEBUG
            std::cout << "--- end inner loop after " << innerIter
                      << " inner iterations, gain heap size: " << threadData.maxGainHeap_.size() << ", outer iteraion "
                      << outerIter << "/" << parameters_.maxOuterIterations_
                      << ", current cost: " << threadData.activeScheduleData_.cost_ << ", "
                      << (threadData.activeScheduleData_.feasible_ ? "feasible" : "infeasible") << std::endl;
#endif
#ifdef KL_DEBUG_1
            const unsigned numStepsTmp = threadData.endStep_;
#endif
            activeSchedule_.RevertToBestSchedule(threadData.localSearchStartStep_,
                                                 threadData.stepToRemove_,
                                                 commCostF_,
                                                 threadData.activeScheduleData_,
                                                 threadData.startStep_,
                                                 threadData.endStep_);
#ifdef KL_DEBUG_1
            if (threadData.localSearchStartStep_ > 0) {
                if (numStepsTmp == threadData.endStep_) {
                    std::cout << "thread " << threadData.threadId_ << ", removing step " << threadData.stepToRemove_
                              << " succeded " << std::endl;
                } else {
                    std::cout << "thread " << threadData.threadId_ << ", removing step " << threadData.stepToRemove_ << " failed "
                              << std::endl;
                }
            }
#endif

#ifdef KL_DEBUG_COST_CHECK
            activeSchedule_.GetVectorSchedule().numberOfSupersteps = threadDataVec_[0].NumSteps();
            if (std::abs(commCostF_.ComputeScheduleCostTest() - threadData.activeScheduleData_.cost_) > 0.00001) {
                std::cout << "computed cost: " << commCostF_.ComputeScheduleCostTest()
                          << ", current cost: " << threadData.activeScheduleData_.cost_ << std::endl;
                std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
            }
            if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                if (not activeSchedule_.memoryConstraint_.SatisfiedMemoryConstraint()) {
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

            if (OtherThreadsFinished(threadData.threadId_)) {
#ifdef KL_DEBUG_1
                std::cout << "thread " << threadData.threadId_ << ", other threads finished, end local search" << std::endl;
#endif
                break;
            }

            if (initialInnerIterCost <= threadData.activeScheduleData_.cost_) {
                noImprovementIterCounter++;

                if (noImprovementIterCounter >= parameters_.maxNoImprovementIterations_) {
#ifdef KL_DEBUG_1
                    std::cout << "thread " << threadData.threadId_ << ", no improvement for "
                              << parameters_.maxNoImprovementIterations_ << " iterations, end local search" << std::endl;
#endif
                    break;
                }
            } else {
                noImprovementIterCounter = 0;
            }

            AdjustLocalSearchParameters(outerIter, noImprovementIterCounter, threadData);
        }

#ifdef KL_DEBUG_1
        std::cout << "thread " << threadData.threadId_ << ", local search end after " << outerIter
                  << " outer iterations, current cost: " << threadData.activeScheduleData_.cost_ << " with "
                  << threadData.NumSteps() << " supersteps, vs serial cost " << activeSchedule_.GetTotalWorkWeight() << "."
                  << std::endl;
#endif
        threadFinishedVec_[threadData.threadId_] = true;
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
                                 const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                                 const typename CommCostFunctionT::PreMoveCommDataT &prevCommData) {
        if constexpr (CommCostFunctionT::isMaxCommCostFunction_) {
            commCostF_.UpdateNodeCommAffinity(
                bestMove,
                threadData,
                threadData.rewardPenaltyStrat_.penalty_,
                threadData.rewardPenaltyStrat_.reward_,
                recomputeMaxGain,
                newNodes);    // this only updated reward/penalty, collects newNodes, and fills recomputeMaxGain

            // Determine the steps where max/second_max/max_count for work/comm changed
            std::unordered_set<unsigned> changedSteps;

            // Check work changes for fromStep
            if (bestMove.fromStep_ == bestMove.toStep_) {
                // Same step - check if max/second_max changed
                const auto currentMax = activeSchedule_.GetStepMaxWork(bestMove.fromStep_);
                const auto currentSecondMax = activeSchedule_.GetStepSecondMaxWork(bestMove.fromStep_);
                const auto currentCount = activeSchedule_.GetStepMaxWorkProcessorCount()[bestMove.fromStep_];
                if (currentMax != prevWorkData.fromStepMaxWork_ || currentSecondMax != prevWorkData.fromStepSecondMaxWork_
                    || currentCount != prevWorkData.fromStepMaxWorkProcessorCount_) {
                    changedSteps.insert(bestMove.fromStep_);
                }
            } else {
                // Different steps - check both
                const auto currentFromMax = activeSchedule_.GetStepMaxWork(bestMove.fromStep_);
                const auto currentFromSecondMax = activeSchedule_.GetStepSecondMaxWork(bestMove.fromStep_);
                const auto currentFromCount = activeSchedule_.GetStepMaxWorkProcessorCount()[bestMove.fromStep_];
                if (currentFromMax != prevWorkData.fromStepMaxWork_ || currentFromSecondMax != prevWorkData.fromStepSecondMaxWork_
                    || currentFromCount != prevWorkData.fromStepMaxWorkProcessorCount_) {
                    changedSteps.insert(bestMove.fromStep_);
                }

                const auto currentToMax = activeSchedule_.GetStepMaxWork(bestMove.toStep_);
                const auto currentToSecondMax = activeSchedule_.GetStepSecondMaxWork(bestMove.toStep_);
                const auto currentToCount = activeSchedule_.GetStepMaxWorkProcessorCount()[bestMove.toStep_];
                if (currentToMax != prevWorkData.toStepMaxWork_ || currentToSecondMax != prevWorkData.toStepSecondMaxWork_
                    || currentToCount != prevWorkData.toStepMaxWorkProcessorCount_) {
                    changedSteps.insert(bestMove.toStep_);
                }
            }

            for (const auto &[step, stepInfo] : prevCommData.stepData_) {
                // typename CommCostFunctionT::PreMoveCommDataT::StepInfo currentInfo;
                // Query current values
                const auto currentMax = commCostF_.commDs_.StepMaxComm(step);
                const auto currentSecondMax = commCostF_.commDs_.StepSecondMaxComm(step);
                const auto currentCount = commCostF_.commDs_.StepMaxCommCount(step);

                if (currentMax != stepInfo.maxComm_ || currentSecondMax != stepInfo.secondMaxComm_
                    || currentCount != stepInfo.maxCommCount_) {
                    changedSteps.insert(step);
                }
            }

            // Recompute affinities for all active nodes
            const size_t activeCount = threadData.affinityTable_.size();
            for (size_t i = 0; i < activeCount; ++i) {
                const VertexType node = threadData.affinityTable_.GetSelectedNodes()[i];

                // Determine if this node needs affinity recomputation
                // A node needs recomputation if it's in or adjacent to changed steps
                const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);

                // Calculate window bounds for this node once
                const int nodeLowerBound = static_cast<int>(nodeStep) - static_cast<int>(windowSize);
                const unsigned nodeUpperBound = nodeStep + windowSize;

                bool needsUpdate = false;
                // Check if any changed step falls within the node's window
                for (unsigned step : changedSteps) {
                    if (static_cast<int>(step) >= nodeLowerBound && step <= nodeUpperBound) {
                        needsUpdate = true;
                        break;
                    }
                }

                if (needsUpdate) {
                    auto &affinityTableNode = threadData.affinityTable_.GetAffinityTable(node);

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
            UpdateNodeWorkAffinity(threadData.affinityTable_, bestMove, prevWorkData, recomputeMaxGain);
            commCostF_.UpdateNodeCommAffinity(bestMove,
                                              threadData,
                                              threadData.rewardPenaltyStrat_.penalty_,
                                              threadData.rewardPenaltyStrat_.reward_,
                                              recomputeMaxGain,
                                              newNodes);
        }
    }

    inline bool BlockedEdgeStrategy(VertexType node, std::vector<VertexType> &unlockNodes, ThreadSearchContext &threadData) {
        if (threadData.unlockEdgeBacktrackCounter_ > 1) {
            for (const auto vertexEdgePair : threadData.activeScheduleData_.newViolations_) {
                const auto &e = vertexEdgePair.second;
                const auto sourceV = Source(e, *graph_);
                const auto targetV = Target(e, *graph_);

                if (node == sourceV && threadData.lockManager_.IsLocked(targetV)) {
                    unlockNodes.push_back(targetV);
                } else if (node == targetV && threadData.lockManager_.IsLocked(sourceV)) {
                    unlockNodes.push_back(sourceV);
                }
            }
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, backtrack counter: " << threadData.unlockEdgeBacktrackCounter_
                      << std::endl;
#endif
            threadData.unlockEdgeBacktrackCounter_--;
            return true;
        } else {
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, end local search" << std::endl;
#endif
            return false;    // or reset local search and initalize with violating nodes
        }
    }

    inline void AdjustLocalSearchParameters(unsigned outerIter, unsigned noImpCounter, ThreadSearchContext &threadData) {
        if (noImpCounter >= threadData.noImprovementIterationsReducePenalty_
            && threadData.rewardPenaltyStrat_.initialPenalty_ > 1.0) {
            threadData.rewardPenaltyStrat_.initialPenalty_
                = static_cast<CostT>(std::floor(std::sqrt(threadData.rewardPenaltyStrat_.initialPenalty_)));
            threadData.unlockEdgeBacktrackCounterReset_ += 1;
            threadData.noImprovementIterationsReducePenalty_ += 15;
#ifdef KL_DEBUG_1
            std::cout << "thread " << threadData.threadId_ << ", no improvement for "
                      << threadData.noImprovementIterationsReducePenalty_ << " iterations, reducing initial penalty to "
                      << threadData.rewardPenaltyStrat_.initialPenalty_ << std::endl;
#endif
        }

        if (parameters_.tryRemoveStepAfterNumOuterIterations_ > 0
            && ((outerIter + 1) % parameters_.tryRemoveStepAfterNumOuterIterations_) == 0) {
            threadData.stepSelectionEpochCounter_ = 0;
            ;
#ifdef KL_DEBUG
            std::cout << "reset remove epoc counter after " << outerIter << " iterations." << std::endl;
#endif
        }

        if (noImpCounter >= threadData.noImprovementIterationsIncreaseInnerIter_) {
            threadData.minInnerIter_ = static_cast<unsigned>(std::ceil(threadData.minInnerIter_ * 2.2));
            threadData.noImprovementIterationsIncreaseInnerIter_ += 20;
#ifdef KL_DEBUG_1
            std::cout << "thread " << threadData.threadId_ << ", no improvement for "
                      << threadData.noImprovementIterationsIncreaseInnerIter_ << " iterations, increasing min inner iter to "
                      << threadData.minInnerIter_ << std::endl;
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
                                       threadData.rewardPenaltyStrat_.penalty_,
                                       threadData.rewardPenaltyStrat_.reward_,
                                       threadData.startStep_,
                                       threadData.endStep_);
    }

    void SelectActiveNodes(ThreadSearchContext &threadData) {
        if (SelectNodesCheckRemoveSuperstep(threadData.stepToRemove_, threadData)) {
            activeSchedule_.SwapEmptyStepFwd(threadData.stepToRemove_, threadData.endStep_);
            threadData.endStep_--;
            threadData.localSearchStartStep_ = static_cast<unsigned>(threadData.activeScheduleData_.appliedMoves_.size());
            threadData.activeScheduleData_.UpdateCost(static_cast<CostT>(-1.0 * instance_->SynchronisationCosts()));

            if constexpr (enablePreresolvingViolations_) {
                ResolveViolations(threadData);
            }

            if (threadData.activeScheduleData_.currentViolations_.size() > parameters_.initialViolationThreshold_) {
                activeSchedule_.RevertToBestSchedule(threadData.localSearchStartStep_,
                                                     threadData.stepToRemove_,
                                                     commCostF_,
                                                     threadData.activeScheduleData_,
                                                     threadData.startStep_,
                                                     threadData.endStep_);
            } else {
                threadData.unlockEdgeBacktrackCounter_
                    = static_cast<unsigned>(threadData.activeScheduleData_.currentViolations_.size());
                threadData.maxInnerIterations_
                    = std::max(threadData.unlockEdgeBacktrackCounter_ * 5u, parameters_.maxInnerIterationsReset_);
                threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackForRemoveStepReset_;
#ifdef KL_DEBUG_1
                std::cout << "thread " << threadData.threadId_ << ", Trying to remove step " << threadData.stepToRemove_
                          << std::endl;
#endif
                return;
            }
        }
        // threadData.stepToRemove_ = threadData.startStep_;
        threadData.localSearchStartStep_ = 0;
        threadData.selectionStrategy_.SelectActiveNodes(threadData.affinityTable_, threadData.startStep_, threadData.endStep_);
    }

    bool CheckRemoveSuperstep(unsigned step);
    bool SelectNodesCheckRemoveSuperstep(unsigned &step, ThreadSearchContext &threadData);

    bool ScatterNodesSuperstep(unsigned step, ThreadSearchContext &threadData) {
        assert(step <= threadData.endStep_ && threadData.startStep_ <= step);
        bool abort = false;

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            const std::vector<VertexType> stepProcNodeVec(
                activeSchedule_.GetSetSchedule().stepProcessorVertices_[step][proc].begin(),
                activeSchedule_.GetSetSchedule().stepProcessorVertices_[step][proc].end());
            for (const auto &node : stepProcNodeVec) {
                threadData.rewardPenaltyStrat_.InitRewardPenalty(
                    static_cast<double>(threadData.activeScheduleData_.currentViolations_.size()) + 1.0);
                ComputeNodeAffinities(node, threadData.localAffinityTable_, threadData);
                KlMove bestMove = ComputeBestMove<false>(node, threadData.localAffinityTable_, threadData);

                if (bestMove.gain_ <= std::numeric_limits<double>::lowest()) {
                    abort = true;
                    break;
                }

                ApplyMove(bestMove, threadData);
                if (threadData.activeScheduleData_.currentViolations_.size() > parameters_.abortScatterNodesViolationThreshold_) {
                    abort = true;
                    break;
                }

                threadData.affinityTable_.Insert(node);
                // threadData.selectionStrategy_.AddNeighboursToSelection(node, threadData.affinityTable_,
                // threadData.startStep_, threadData.endStep_);
                if (threadData.activeScheduleData_.newViolations_.size() > 0) {
                    for (const auto &vertexEdgePair : threadData.activeScheduleData_.newViolations_) {
                        const auto &vertex = vertexEdgePair.first;
                        threadData.affinityTable_.Insert(vertex);
                    }
                }

#ifdef KL_DEBUG
                std::cout << "move node " << bestMove.node_ << " with gain " << bestMove.gain_
                          << ", from proc|step: " << bestMove.fromProc_ << "|" << bestMove.fromStep_
                          << " to: " << bestMove.toProc_ << "|" << bestMove.toStep_ << std::endl;
#endif

#ifdef KL_DEBUG_COST_CHECK
                activeSchedule_.GetVectorSchedule().numberOfSupersteps = threadDataVec_[0].NumSteps();
                if (std::abs(commCostF_.ComputeScheduleCostTest() - threadData.activeScheduleData_.cost_) > 0.00001) {
                    std::cout << "computed cost: " << commCostF_.ComputeScheduleCostTest()
                              << ", current cost: " << threadData.activeScheduleData_.cost_ << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                    if (not activeSchedule_.memoryConstraint_.SatisfiedMemoryConstraint()) {
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
                0, 0, commCostF_, threadData.activeScheduleData_, threadData.startStep_, threadData.endStep_);
            threadData.affinityTable_.ResetNodeSelection();
            return false;
        }
        return true;
    }

    void SynchronizeActiveSchedule(const unsigned numThreads) {
        if (numThreads == 1) {    // single thread case
            activeSchedule_.SetCost(threadDataVec_[0].activeScheduleData_.cost_);
            activeSchedule_.GetVectorSchedule().numberOfSupersteps_ = threadDataVec_[0].NumSteps();
            return;
        }

        unsigned writeCursor = threadDataVec_[0].endStep_ + 1;
        for (unsigned i = 1; i < numThreads; ++i) {
            auto &thread = threadDataVec_[i];
            if (thread.startStep_ <= thread.endStep_) {
                for (unsigned j = thread.startStep_; j <= thread.endStep_; ++j) {
                    if (j != writeCursor) {
                        activeSchedule_.SwapSteps(j, writeCursor);
                    }
                    writeCursor++;
                }
            }
        }
        activeSchedule_.GetVectorSchedule().numberOfSupersteps_ = writeCursor;
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

    virtual ReturnStatus ImproveSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.GetInstance().NumberOfProcessors() < 2) {
            return ReturnStatus::BEST_FOUND;
        }

        const unsigned numThreads = 1;

        threadDataVec_.resize(numThreads);
        threadFinishedVec_.assign(numThreads, true);

        SetParameters(schedule.GetInstance().NumberOfVertices());
        InitializeDatastructures(schedule);
        const CostT initialCost = activeSchedule_.GetCost();
        const unsigned numSteps = schedule.NumberOfSupersteps();

        SetStartStep(0, threadDataVec_[0]);
        threadDataVec_[0].endStep_ = (numSteps > 0) ? numSteps - 1 : 0;

        auto &threadData = this->threadDataVec_[0];
        threadData.activeScheduleData_.InitializeCost(activeSchedule_.GetCost());
        threadData.selectionStrategy_.Setup(threadData.startStep_, threadData.endStep_);
        RunLocalSearch(threadData);

        SynchronizeActiveSchedule(numThreads);

        if (initialCost > activeSchedule_.GetCost()) {
            activeSchedule_.WriteSchedule(schedule);
            CleanupDatastructures();
            return ReturnStatus::OSP_SUCCESS;
        } else {
            CleanupDatastructures();
            return ReturnStatus::BEST_FOUND;
        }
    }

    virtual ReturnStatus ImproveScheduleWithTimeLimit(BspSchedule<GraphT> &schedule) override {
        computeWithTimeLimit_ = true;
        return ImproveSchedule(schedule);
    }

    virtual void SetTimeQualityParameter(const double timeQuality) { this->parameters_.timeQuality_ = timeQuality; }

    virtual void SetSuperstepRemoveStrengthParameter(const double superstepRemoveStrength) {
        this->parameters_.superstepRemoveStrength_ = superstepRemoveStrength;
    }

    virtual std::string GetScheduleName() const { return "kl_improver_" + commCostF_.Name(); }
};

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::SetParameters(VertexIdxT<GraphT> numNodes) {
    const unsigned logNumNodes = (numNodes > 1) ? static_cast<unsigned>(std::log(numNodes)) : 1;

    // Total number of outer iterations. Proportional to sqrt N.
    parameters_.maxOuterIterations_
        = static_cast<unsigned>(std::sqrt(numNodes) * (parameters_.timeQuality_ * 10.0) / parameters_.numParallelLoops_);

    // Number of times to reset the search for violations before giving up.
    parameters_.maxNoVioaltionsRemovedBacktrackReset_ = parameters_.timeQuality_ < 0.75  ? 1
                                                        : parameters_.timeQuality_ < 1.0 ? 2
                                                                                         : 3;

    // Parameters for the superstep removal heuristic.
    parameters_.maxNoVioaltionsRemovedBacktrackForRemoveStepReset_
        = 3 + static_cast<unsigned>(parameters_.superstepRemoveStrength_ * 7);
    parameters_.nodeMaxStepSelectionEpochs_ = parameters_.superstepRemoveStrength_ < 0.75  ? 1
                                              : parameters_.superstepRemoveStrength_ < 1.0 ? 2
                                                                                           : 3;
    parameters_.removeStepEpocs_ = static_cast<unsigned>(parameters_.superstepRemoveStrength_ * 4.0);

    parameters_.minInnerIterReset_ = static_cast<unsigned>(logNumNodes + logNumNodes * (1.0 + parameters_.timeQuality_));

    if (parameters_.removeStepEpocs_ > 0) {
        parameters_.tryRemoveStepAfterNumOuterIterations_ = parameters_.maxOuterIterations_ / parameters_.removeStepEpocs_;
    } else {
        // Effectively disable superstep removal if remove_step_epocs is 0.
        parameters_.tryRemoveStepAfterNumOuterIterations_ = parameters_.maxOuterIterations_ + 1;
    }

    unsigned i = 0;
    for (auto &thread : threadDataVec_) {
        thread.threadId_ = i++;
        // The number of nodes to consider in each inner iteration. Proportional to log(N).
        thread.selectionStrategy_.selectionThreshold_
            = static_cast<std::size_t>(std::ceil(parameters_.timeQuality_ * 10 * logNumNodes + logNumNodes));
    }

#ifdef KL_DEBUG_1
    std::cout << "kl set parameter, number of nodes: " << numNodes << std::endl;
    std::cout << "max outer iterations: " << parameters_.maxOuterIterations_ << std::endl;
    std::cout << "max inner iterations: " << parameters_.maxInnerIterationsReset_ << std::endl;
    std::cout << "no improvement iterations reduce penalty: " << threadDataVec_[0].noImprovementIterationsReducePenalty_
              << std::endl;
    std::cout << "selction threshold: " << threadDataVec_[0].selectionStrategy_.selectionThreshold_ << std::endl;
    std::cout << "remove step epocs: " << parameters_.removeStepEpocs_ << std::endl;
    std::cout << "try remove step after num outer iterations: " << parameters_.tryRemoveStepAfterNumOuterIterations_ << std::endl;
    std::cout << "number of parallel loops: " << parameters_.numParallelLoops_ << std::endl;
#endif
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateNodeWorkAffinity(
    NodeSelectionContainerT &nodes,
    KlMove move,
    const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
    std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain) {
    const size_t activeCount = nodes.size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = nodes.GetSelectedNodes()[i];

        KlGainUpdateInfo updateInfo = UpdateNodeWorkAffinityAfterMove(node, move, prevWorkData, nodes.At(node));
        if (updateInfo.updateFromStep_ || updateInfo.updateToStep_) {
            recomputeMaxGain[node] = updateInfo;
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateMaxGain(
    KlMove move, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain, ThreadSearchContext &threadData) {
    for (auto &pair : recomputeMaxGain) {
        if (pair.second.fullUpdate_) {
            RecomputeNodeMaxGain(pair.first, threadData.affinityTable_, threadData);
        } else {
            if (pair.second.updateEntireFromStep_) {
                UpdateBestMove(pair.first, move.fromStep_, threadData.affinityTable_, threadData);
            } else if (pair.second.updateFromStep_ && IsCompatible(pair.first, move.fromProc_)) {
                UpdateBestMove(pair.first, move.fromStep_, move.fromProc_, threadData.affinityTable_, threadData);
            }

            if (move.fromStep_ != move.toStep_ || not pair.second.updateEntireFromStep_) {
                if (pair.second.updateEntireToStep_) {
                    UpdateBestMove(pair.first, move.toStep_, threadData.affinityTable_, threadData);
                } else if (pair.second.updateToStep_ && IsCompatible(pair.first, move.toProc_)) {
                    UpdateBestMove(pair.first, move.toStep_, move.toProc_, threadData.affinityTable_, threadData);
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ComputeWorkAffinity(
    VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData) {
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
    const VertexWorkWeightT vertexWeight = graph_->VertexWorkWeight(node);

    unsigned step = (nodeStep > windowSize) ? (nodeStep - windowSize) : 0;
    for (unsigned idx = threadData.StartIdx(nodeStep); idx < threadData.EndIdx(nodeStep); ++idx, ++step) {
        if (idx == windowSize) {
            continue;
        }

        const CostT maxWorkForStep = static_cast<CostT>(activeSchedule_.GetStepMaxWork(step));

        for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
            const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(step, proc);
            const CostT workDiff = static_cast<CostT>(newWeight) - maxWorkForStep;
            affinityTableNode[proc][idx] = std::max(0.0, workDiff);
        }
    }

    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const VertexWorkWeightT maxWorkForStep = activeSchedule_.GetStepMaxWork(nodeStep);
    const bool isSoleMaxProcessor = (activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                    && (maxWorkForStep == activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc));

    const CostT nodeProcAffinity
        = isSoleMaxProcessor ? std::min(vertexWeight, maxWorkForStep - activeSchedule_.GetStepSecondMaxWork(nodeStep)) : 0.0;
    affinityTableNode[nodeProc][windowSize] = nodeProcAffinity;

    for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
        if (proc == nodeProc) {
            continue;
        }

        const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
        affinityTableNode[proc][windowSize] = ComputeSameStepAffinity(maxWorkForStep, newWeight, nodeProcAffinity);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ProcessWorkUpdateStep(
    VertexType node,
    unsigned nodeStep,
    unsigned nodeProc,
    VertexWorkWeightT vertexWeight,
    unsigned moveStep,
    unsigned moveProc,
    VertexWorkWeightT moveCorrectionNodeWeight,
    const VertexWorkWeightT prevMoveStepMaxWork,
    const VertexWorkWeightT prevMoveStepSecondMaxWork,
    unsigned prevMoveStepMaxWorkProcessorCount,
    bool &updateStep,
    bool &updateEntireStep,
    bool &fullUpdate,
    std::vector<std::vector<CostT>> &affinityTableNode) {
    const unsigned lowerBound = moveStep > windowSize ? moveStep - windowSize : 0;
    if (lowerBound <= nodeStep && nodeStep <= moveStep + windowSize) {
        updateStep = true;
        if (nodeStep == moveStep) {
            const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(moveStep);
            const VertexWorkWeightT newSecondMaxWeight = activeSchedule_.GetStepSecondMaxWork(moveStep);
            const VertexWorkWeightT newStepProcWork = activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);

            const VertexWorkWeightT prevStepProcWork = (nodeProc == moveProc) ? newStepProcWork + moveCorrectionNodeWeight
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
            const bool updateNodeProcAffinity = std::abs(diff) > epsilon_;
            if (updateNodeProcAffinity) {
                fullUpdate = true;
                affinityTableNode[nodeProc][windowSize] += diff;
            }

            if ((prevMoveStepMaxWork != newMaxWeight) || updateNodeProcAffinity) {
                updateEntireStep = true;

                for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                    if ((proc == nodeProc) || (proc == moveProc)) {
                        continue;
                    }

                    const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
                    const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMoveStepMaxWork, newWeight, prevNodeProcAffinity);
                    const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                    affinityTableNode[proc][windowSize] += (otherAffinity - prevOtherAffinity);
                }
            }

            if (nodeProc != moveProc && IsCompatible(node, moveProc)) {
                const VertexWorkWeightT prevNewWeight
                    = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc) + moveCorrectionNodeWeight;
                const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMoveStepMaxWork, prevNewWeight, prevNodeProcAffinity);
                const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, moveProc);
                const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                affinityTableNode[moveProc][windowSize] += (otherAffinity - prevOtherAffinity);
            }

        } else {
            const VertexWorkWeightT newMaxWeight = activeSchedule_.GetStepMaxWork(moveStep);
            const unsigned idx = RelStepIdx(nodeStep, moveStep);
            if (prevMoveStepMaxWork != newMaxWeight) {
                updateEntireStep = true;

                // update moving to all procs with special for moveProc
                for (const unsigned proc : procRange_.CompatibleProcessorsVertex(node)) {
                    const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(moveStep, proc);
                    if (proc != moveProc) {
                        const CostT prevAffinity = prevMoveStepMaxWork < newWeight
                                                       ? static_cast<CostT>(newWeight) - static_cast<CostT>(prevMoveStepMaxWork)
                                                       : 0.0;
                        const CostT newAffinity
                            = newMaxWeight < newWeight ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight) : 0.0;
                        affinityTableNode[proc][idx] += newAffinity - prevAffinity;

                    } else {
                        const VertexWorkWeightT prevNewWeight
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
                // update only moveProc
                if (IsCompatible(node, moveProc)) {
                    const VertexWorkWeightT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(moveStep, moveProc);
                    const VertexWorkWeightT prevNewWeight = newWeight + moveCorrectionNodeWeight;
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

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::SelectNodesCheckRemoveSuperstep(
    unsigned &stepToRemove, ThreadSearchContext &threadData) {
    if (threadData.stepSelectionEpochCounter_ >= parameters_.nodeMaxStepSelectionEpochs_ || threadData.NumSteps() < 3) {
        return false;
    }

    for (stepToRemove = threadData.stepSelectionCounter_; stepToRemove <= threadData.endStep_; stepToRemove++) {
        assert(stepToRemove >= threadData.startStep_ && stepToRemove <= threadData.endStep_);
#ifdef KL_DEBUG
        std::cout << "Checking to remove step " << stepToRemove << "/" << threadData.endStep_ << std::endl;
#endif
        if (CheckRemoveSuperstep(stepToRemove)) {
#ifdef KL_DEBUG
            std::cout << "Checking to scatter step " << stepToRemove << "/" << threadData.endStep_ << std::endl;
#endif
            assert(stepToRemove >= threadData.startStep_ && stepToRemove <= threadData.endStep_);
            if (ScatterNodesSuperstep(stepToRemove, threadData)) {
                threadData.stepSelectionCounter_ = stepToRemove + 1;

                if (threadData.stepSelectionCounter_ > threadData.endStep_) {
                    threadData.stepSelectionCounter_ = threadData.startStep_;
                    threadData.stepSelectionEpochCounter_++;
                }
                return true;
            }
        }
    }

    threadData.stepSelectionEpochCounter_++;
    threadData.stepSelectionCounter_ = threadData.startStep_;
    return false;
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::CheckRemoveSuperstep(unsigned step) {
    if (activeSchedule_.NumSteps() < 2) {
        return false;
    }

    if (activeSchedule_.GetStepMaxWork(step) < instance_->SynchronisationCosts()) {
        return true;
    }

    return false;
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ResetInnerSearchStructures(
    ThreadSearchContext &threadData) const {
    threadData.unlockEdgeBacktrackCounter_ = threadData.unlockEdgeBacktrackCounterReset_;
    threadData.maxInnerIterations_ = parameters_.maxInnerIterationsReset_;
    threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackReset_;
    threadData.averageGain_ = 0.0;
    threadData.affinityTable_.ResetNodeSelection();
    threadData.maxGainHeap_.Clear();
    threadData.lockManager_.Clear();
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::IsLocalSearchBlocked(
    ThreadSearchContext &threadData) {
    for (const auto &pair : threadData.activeScheduleData_.newViolations_) {
        if (threadData.lockManager_.IsLocked(pair.first)) {
            return true;
        }
    }
    return false;
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::InitializeDatastructures(
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
        tData.affinityTable_.Initialize(activeSchedule_, tData.selectionStrategy_.selectionThreshold_);
        tData.lockManager_.Initialize(graph_->NumVertices());
        tData.rewardPenaltyStrat_.Initialize(
            activeSchedule_, commCostF_.GetMaxCommWeightMultiplied(), activeSchedule_.GetMaxWorkWeight());
        tData.selectionStrategy_.Initialize(activeSchedule_, gen_, tData.startStep_, tData.endStep_);

        tData.localAffinityTable_.resize(instance_->NumberOfProcessors());
        for (unsigned i = 0; i < instance_->NumberOfProcessors(); ++i) {
            tData.localAffinityTable_[i].resize(windowRange_);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateAvgGain(const CostT gain,
                                                                                                const unsigned numIter,
                                                                                                double &averageGain) {
    averageGain = static_cast<double>((averageGain * numIter + gain)) / (numIter + 1.0);
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::InsertGainHeap(ThreadSearchContext &threadData) {
    const size_t activeCount = threadData.affinityTable_.size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = threadData.affinityTable_.GetSelectedNodes()[i];
        ComputeNodeAffinities(node, threadData.affinityTable_.At(node), threadData);
        const auto bestMove = ComputeBestMove<true>(node, threadData.affinityTable_[node], threadData);
        threadData.maxGainHeap_.Push(node, bestMove);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::InsertNewNodesGainHeap(
    std::vector<VertexType> &newNodes, NodeSelectionContainerT &nodes, ThreadSearchContext &threadData) {
    for (const auto &node : newNodes) {
        nodes.Insert(node);
        ComputeNodeAffinities(node, threadData.affinityTable_.At(node), threadData);
        const auto bestMove = ComputeBestMove<true>(node, threadData.affinityTable_[node], threadData);
        threadData.maxGainHeap_.Push(node, bestMove);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::CleanupDatastructures() {
    threadDataVec_.clear();
    activeSchedule_.Clear();
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::PrintHeap(HeapDatastructure &maxGainHeap) const {
    if (maxGainHeap.IsEmpty()) {
        std::cout << "heap is empty" << std::endl;
        return;
    }
    HeapDatastructure tempHeap = maxGainHeap;    // requires copy constructor

    std::cout << "heap current size: " << tempHeap.size() << std::endl;
    const auto &topVal = tempHeap.GetValue(tempHeap.Top());
    std::cout << "heap top node " << topVal.node_ << " gain " << topVal.gain_ << std::endl;

    unsigned count = 0;
    while (!tempHeap.IsEmpty() && count++ < 15) {
        const auto &val = tempHeap.GetValue(tempHeap.Top());
        std::cout << "node " << val.node_ << " gain " << val.gain_ << " to proc " << val.toProc_ << " to step " << val.toStep_
                  << std::endl;
        tempHeap.Pop();
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);

    if ((nodeProc == proc) && (nodeStep == step)) {
        return;
    }

    KlMove nodeMove = threadData.maxGainHeap_.GetValue(node);
    CostT maxGain = nodeMove.gain_;

    unsigned maxProc = nodeMove.toProc_;
    unsigned maxStep = nodeMove.toStep_;

    if ((maxStep == step) && (maxProc == proc)) {
        RecomputeNodeMaxGain(node, affinityTable, threadData);
    } else {
        if constexpr (ActiveScheduleT::useMemoryConstraint_) {
            if (not activeSchedule_.memoryConstraint_.CanMove(node, proc, step)) {
                return;
            }
        }
        const unsigned idx = RelStepIdx(nodeStep, step);
        const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][proc][idx];
        if (gain > maxGain) {
            maxGain = gain;
            maxProc = proc;
            maxStep = step;
        }

        const CostT diff = maxGain - nodeMove.gain_;
        if ((std::abs(diff) > epsilon_) || (maxProc != nodeMove.toProc_) || (maxStep != nodeMove.toStep_)) {
            nodeMove.gain_ = maxGain;
            nodeMove.toProc_ = maxProc;
            nodeMove.toStep_ = maxStep;
            threadData.maxGainHeap_.Update(node, nodeMove);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
    const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);

    KlMove nodeMove = threadData.maxGainHeap_.GetValue(node);
    CostT maxGain = nodeMove.gain_;

    unsigned maxProc = nodeMove.toProc_;
    unsigned maxStep = nodeMove.toStep_;

    if (maxStep == step) {
        RecomputeNodeMaxGain(node, affinityTable, threadData);
    } else {
        if (nodeStep != step) {
            const unsigned idx = RelStepIdx(nodeStep, step);
            for (const unsigned p : procRange_.CompatibleProcessorsVertex(node)) {
                if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                    if (not activeSchedule_.memoryConstraint_.CanMove(node, p, step)) {
                        continue;
                    }
                }
                const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][p][idx];
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
                if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                    if (not activeSchedule_.memoryConstraint_.CanMove(node, proc, step)) {
                        continue;
                    }
                }
                const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][proc][windowSize];
                if (gain > maxGain) {
                    maxGain = gain;
                    maxProc = proc;
                    maxStep = step;
                }
            }
        }

        const CostT diff = maxGain - nodeMove.gain_;
        if ((std::abs(diff) > epsilon_) || (maxProc != nodeMove.toProc_) || (maxStep != nodeMove.toStep_)) {
            nodeMove.gain_ = maxGain;
            nodeMove.toProc_ = maxProc;
            nodeMove.toStep_ = maxStep;
            threadData.maxGainHeap_.Update(node, nodeMove);
        }
    }
}

}    // namespace osp
