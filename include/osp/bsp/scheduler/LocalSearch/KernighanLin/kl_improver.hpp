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
    static_assert(IsDirectedGraphEdgeDescV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    static_assert(HasHashableEdgeDescV<GraphT>, "Graph_t must satisfy the has_hashable_edge_desc concept");
    static_assert(IsComputationalDagV<GraphT>, "Graph_t must satisfy the computational_dag concept");

  protected:
    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool enableQuickMoves_ = true;
    constexpr static bool enablePreresolvingViolations_ = true;
    constexpr static double epsilon_ = 1e-9;

    using VMemwT = VMemwT<GraphT>;
    using VCommwT = VCommwT<GraphT>;
    using VWorkwT = VWorkwT<GraphT>;
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

        VectorVertexLockManger<VertexType> lockManager_;
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
        return activeSchedule_.GetInstance().isCompatible(node, proc);
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
                       VectorVertexLockManger<VertexType> &lockManager,
                       HeapDatastructure &maxGainHeap) {
        // To introduce non-determinism and help escape local optima, if there are multiple moves with the same
        // top gain, we randomly select one. We check up to `local_max` ties.
        const unsigned localMax = 50;
        std::vector<VertexType> topGainNodes = maxGainHeap.get_top_keys(localMax);

        if (topGainNodes.empty()) {
            // This case is guarded by the caller, but for safety:
            topGainNodes.push_back(maxGainHeap.top());
        }

        std::uniform_int_distribution<size_t> dis(0, topGainNodes.size() - 1);
        const VertexType node = topGainNodes[dis(gen_)];

        KlMove bestMove = maxGainHeap.get_value(node);
        maxGainHeap.erase(node);
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
        for (const unsigned p : procRange_.compatible_processors_vertex(node)) {
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
            for (const unsigned proc : procRange_.compatible_processors_vertex(node)) {
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
                                                     const PreMoveWorkData<VWorkwT> &prevWorkData,
                                                     std::vector<std::vector<CostT>> &affinityTableNode) {
        const unsigned nodeStep = activeSchedule_.AssignedSuperstep(node);
        const VWorkwT vertexWeight = graph_->VertexWorkWeight(node);

        KlGainUpdateInfo updateInfo(node);

        if (move.fromStep_ == move.toStep_) {
            const unsigned lowerBound = move.fromStep_ > windowSize ? move.fromStep_ - windowSize : 0;
            if (lowerBound <= nodeStep && nodeStep <= move.fromStep_ + windowSize) {
                updateInfo.updateFromStep_ = true;
                updateInfo.updateToStep_ = true;

                const VWorkwT prevMaxWork = prevWorkData.fromStepMaxWork_;
                const VWorkwT prevSecondMaxWork = prevWorkData.fromStepSecondMaxWork_;

                if (nodeStep == move.fromStep_) {
                    const unsigned nodeProc = activeSchedule_.AssignedProcessor(node);
                    const VWorkwT newMaxWeight = activeSchedule_.GetStepMaxWork(move.fromStep_);
                    const VWorkwT newSecondMaxWeight = activeSchedule_.GetStepSecondMaxWork(move.fromStep_);
                    const VWorkwT newStepProcWork = activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);
                    const VWorkwT prevStepProcWork
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

                        for (const unsigned proc : procRange_.compatible_processors_vertex(node)) {
                            if ((proc == nodeProc) || (proc == move.fromProc_) || (proc == move.toProc_)) {
                                continue;
                            }

                            const VWorkwT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, proc);
                            const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, newWeight, prevNodeProcAffinity);
                            const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

                            affinityTableNode[proc][windowSize] += (otherAffinity - prevOtherAffinity);
                        }
                    }

                    if (nodeProc != move.fromProc_ && IsCompatible(node, move.fromProc_)) {
                        const VWorkwT prevNewWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, move.fromProc_)
                                                      + graph_->VertexWorkWeight(move.node_);
                        const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, prevNewWeight, prevNodeProcAffinity);
                        const VWorkwT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, move.fromProc_);
                        const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
                        affinityTableNode[move.fromProc_][windowSize] += (otherAffinity - prevOtherAffinity);
                    }

                    if (nodeProc != move.toProc_ && IsCompatible(node, move.toProc_)) {
                        const VWorkwT prevNewWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, move.toProc_)
                                                      - graph_->VertexWorkWeight(move.node_);
                        const CostT prevOtherAffinity = ComputeSameStepAffinity(prevMaxWork, prevNewWeight, prevNodeProcAffinity);
                        const VWorkwT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(nodeStep, move.toProc_);
                        const CostT otherAffinity = ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
                        affinityTableNode[move.toProc_][windowSize] += (otherAffinity - prevOtherAffinity);
                    }

                } else {
                    const VWorkwT newMaxWeight = activeSchedule_.GetStepMaxWork(move.fromStep_);
                    const unsigned idx = RelStepIdx(nodeStep, move.fromStep_);
                    if (prevMaxWork != newMaxWeight) {
                        updateInfo.updateEntireFromStep_ = true;
                        // update moving to all procs with special for move.fromProc_
                        for (const unsigned proc : procRange_.compatible_processors_vertex(node)) {
                            const VWorkwT newWeight = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, proc);
                            if (proc == move.fromProc_) {
                                const VWorkwT prevNewWeight = vertexWeight
                                                              + activeSchedule_.GetStepProcessorWork(move.fromStep_, proc)
                                                              + graph_->VertexWorkWeight(move.node_);
                                const CostT prevAffinity = prevMaxWork < prevNewWeight ? static_cast<CostT>(prevNewWeight)
                                                                                             - static_cast<CostT>(prevMaxWork)
                                                                                       : 0.0;
                                const CostT newAffinity = newMaxWeight < newWeight
                                                              ? static_cast<CostT>(newWeight) - static_cast<CostT>(newMaxWeight)
                                                              : 0.0;
                                affinityTableNode[proc][idx] += newAffinity - prevAffinity;
                            } else if (proc == move.toProc_) {
                                const VWorkwT prevNewWeight = vertexWeight
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
                            const VWorkwT fromNewWeight
                                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.fromStep_, move.fromProc_);
                            const VWorkwT fromPrevNewWeight = fromNewWeight + graph_->VertexWorkWeight(move.node_);
                            const CostT fromPrevAffinity = prevMaxWork < fromPrevNewWeight ? static_cast<CostT>(fromPrevNewWeight)
                                                                                                 - static_cast<CostT>(prevMaxWork)
                                                                                           : 0.0;

                            const CostT fromNewAffinity = newMaxWeight < fromNewWeight ? static_cast<CostT>(fromNewWeight)
                                                                                             - static_cast<CostT>(newMaxWeight)
                                                                                       : 0.0;
                            affinityTableNode[move.fromProc_][idx] += fromNewAffinity - fromPrevAffinity;
                        }

                        if (IsCompatible(node, move.toProc_)) {
                            const VWorkwT toNewWeight
                                = vertexWeight + activeSchedule_.GetStepProcessorWork(move.toStep_, move.toProc_);
                            const VWorkwT toPrevNewWeight = toNewWeight - graph_->VertexWorkWeight(move.node_);
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
                               VWorkwT vertexWeight,
                               unsigned moveStep,
                               unsigned moveProc,
                               VWorkwT moveCorrectionNodeWeight,
                               const VWorkwT prevMoveStepMaxWork,
                               const VWorkwT prevMoveStepSecondMaxWork,
                               unsigned prevMoveStepMaxWorkProcessorCount,
                               bool &updateStep,
                               bool &updateEntireStep,
                               bool &fullUpdate,
                               std::vector<std::vector<CostT>> &affinityTableNode);
    void UpdateNodeWorkAffinity(NodeSelectionContainerT &nodes,
                                KlMove move,
                                const PreMoveWorkData<VWorkwT> &prevWorkData,
                                std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain);
    void UpdateBestMove(
        VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateBestMove(VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateMaxGain(KlMove move, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain, ThreadSearchContext &threadData);
    void ComputeWorkAffinity(VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData);

    inline void RecomputeNodeMaxGain(VertexType node, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
        const auto bestMove = ComputeBestMove<true>(node, affinityTable[node], threadData);
        threadData.maxGainHeap_.update(node, bestMove);
    }

    inline CostT ComputeSameStepAffinity(const VWorkwT &maxWorkForStep, const VWorkwT &newWeight, const CostT &nodeProcAffinity) {
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
        threadData.maxGainHeap_.clear();
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
                const VertexType targetV = Traget(nextEdge, *graph_);
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
                if (not activeSchedule_.GetInstance().isCompatible(bestMove.node_, bestMove.toProc_)) {
                    std::cout << "move to incompatibe node" << std::endl;
                }
#endif

                const auto prevWorkData = activeSchedule_.GetPreMoveWorkData(bestMove);
                const typename CommCostFunctionT::PreMoveCommData prevCommData = commCostF_.GetPreMoveCommData(bestMove);
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
                                threadData.maxGainHeap_.clear();
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
                if (duration > ImprovementScheduler<GraphT>::timeLimitSeconds) {
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

    inline void UpdateAffinities(const kl_move &bestMove,
                                 ThreadSearchContext &threadData,
                                 std::map<VertexType, kl_gain_update_info> &recomputeMaxGain,
                                 std::vector<VertexType> &newNodes,
                                 const pre_move_work_data<VWorkwT<GraphT>> &prevWorkData,
                                 const typename CommCostFunctionT::pre_move_comm_data_t &prevCommData) {
        if constexpr (CommCostFunctionT::is_max_comm_cost_function) {
            commCostF_.update_node_comm_affinity(
                best_move,
                threadData,
                threadData.rewardPenaltyStrat_.penalty,
                threadData.rewardPenaltyStrat_.reward,
                recompute_max_gain,
                new_nodes);    // this only updated reward/penalty, collects new_nodes, and fills recompute_max_gain

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
            if (bestMove.from_step == best_move.to_step) {
                // Same step - check if max/second_max changed
                const auto currentMax = activeSchedule_.get_step_max_work(best_move.from_step);
                const auto currentSecondMax = activeSchedule_.get_step_second_max_work(best_move.from_step);
                const auto currentCount = activeSchedule_.get_step_max_work_processor_count()[best_move.from_step];
                if (currentMax != prev_work_data.from_step_max_work
                    || current_second_max != prev_work_data.from_step_second_max_work
                    || current_count != prev_work_data.from_step_max_work_processor_count) {
                    changedSteps.insert(best_move.from_step);
                }
            } else {
                // Different steps - check both
                const auto currentFromMax = activeSchedule_.get_step_max_work(best_move.from_step);
                const auto currentFromSecondMax = activeSchedule_.get_step_second_max_work(best_move.from_step);
                const auto currentFromCount = activeSchedule_.get_step_max_work_processor_count()[best_move.from_step];
                if (currentFromMax != prev_work_data.from_step_max_work
                    || current_from_second_max != prev_work_data.from_step_second_max_work
                    || current_from_count != prev_work_data.from_step_max_work_processor_count) {
                    changedSteps.insert(best_move.from_step);
                }

                const auto currentToMax = activeSchedule_.get_step_max_work(best_move.to_step);
                const auto currentToSecondMax = activeSchedule_.get_step_second_max_work(best_move.to_step);
                const auto currentToCount = activeSchedule_.get_step_max_work_processor_count()[best_move.to_step];
                if (currentToMax != prev_work_data.to_step_max_work
                    || current_to_second_max != prev_work_data.to_step_second_max_work
                    || current_to_count != prev_work_data.to_step_max_work_processor_count) {
                    changedSteps.insert(best_move.to_step);
                }
            }

            for (const auto &[step, step_info] : prevCommData.step_data) {
                typename CommCostFunctionT::pre_move_comm_data_t::step_info currentInfo;
                // Query current values
                const auto currentMax = commCostF_.comm_ds.step_max_comm(step);
                const auto currentSecondMax = commCostF_.comm_ds.step_second_max_comm(step);
                const auto currentCount = commCostF_.comm_ds.step_max_comm_count(step);

                if (currentMax != step_info.max_comm || currentSecondMax != step_info.second_max_comm
                    || currentCount != step_info.max_comm_count) {
                    changedSteps.insert(step);
                }
            }

            // Recompute affinities for all active nodes
            const size_t activeCount = threadData.affinityTable_.size();
            for (size_t i = 0; i < activeCount; ++i) {
                const VertexType node = threadData.affinityTable_.get_selected_nodes()[i];

                // Determine if this node needs affinity recomputation
                // A node needs recomputation if it's in or adjacent to changed steps
                const unsigned nodeStep = activeSchedule_.assigned_superstep(node);

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
                    auto &affinityTableNode = threadData.affinityTable_.get_affinity_table(node);

                    // Reset affinity table entries to zero
                    const unsigned numProcs = activeSchedule_.GetInstance().NumberOfProcessors();
                    for (unsigned p = 0; p < numProcs; ++p) {
                        for (unsigned idx = 0; idx < affinityTableNode[p].size(); ++idx) {
                            affinityTableNode[p][idx] = 0;
                        }
                    }

                    compute_node_affinities(node, affinity_table_node, thread_data);
                    recomputeMaxGain[node] = kl_gain_update_info(node, true);
                }
            }
        } else {
            update_node_work_affinity(thread_data.affinity_table, best_move, prev_work_data, recompute_max_gain);
            commCostF_.update_node_comm_affinity(best_move,
                                                 threadData,
                                                 threadData.rewardPenaltyStrat_.penalty,
                                                 threadData.rewardPenaltyStrat_.reward,
                                                 recompute_max_gain,
                                                 new_nodes);
        }
    }

    inline bool BlockedEdgeStrategy(VertexType node, std::vector<VertexType> &unlockNodes, ThreadSearchContext &threadData) {
        if (threadData.unlockEdgeBacktrackCounter_ > 1) {
            for (const auto vertexEdgePair : threadData.activeScheduleData_.new_violations) {
                const auto &e = vertexEdgePair.second;
                const auto sourceV = Source(e, *graph_);
                const auto targetV = Traget(e, *graph_);

                if (node == sourceV && threadData.lockManager_.is_locked(targetV)) {
                    unlockNodes.push_back(targetV);
                } else if (node == targetV && threadData.lockManager_.is_locked(sourceV)) {
                    unlockNodes.push_back(sourceV);
                }
            }
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, backtrack counter: " << thread_data.unlock_edge_backtrack_counter
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
            && threadData.rewardPenaltyStrat_.initial_penalty > 1.0) {
            threadData.rewardPenaltyStrat_.initial_penalty
                = static_cast<CostT>(std::floor(std::sqrt(threadData.rewardPenaltyStrat_.initial_penalty)));
            threadData.unlockEdgeBacktrackCounterReset_ += 1;
            threadData.noImprovementIterationsReducePenalty_ += 15;
#ifdef KL_DEBUG_1
            std::cout << "thread " << thread_data.thread_id << ", no improvement for "
                      << thread_data.no_improvement_iterations_reduce_penalty << " iterations, reducing initial penalty to "
                      << thread_data.reward_penalty_strat.initial_penalty << std::endl;
#endif
        }

        if (parameters_.tryRemoveStepAfterNumOuterIterations_ > 0
            && ((outerIter + 1) % parameters_.tryRemoveStepAfterNumOuterIterations_) == 0) {
            threadData.stepSelectionEpochCounter_ = 0;
            ;
#ifdef KL_DEBUG
            std::cout << "reset remove epoc counter after " << outer_iter << " iterations." << std::endl;
#endif
        }

        if (noImpCounter >= threadData.noImprovementIterationsIncreaseInnerIter_) {
            threadData.minInnerIter_ = static_cast<unsigned>(std::ceil(threadData.minInnerIter_ * 2.2));
            threadData.noImprovementIterationsIncreaseInnerIter_ += 20;
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
    void PrintHeap(heap_datastructure &maxGainHeap) const;
    void CleanupDatastructures();
    void UpdateAvgGain(const CostT gain, const unsigned numIter, double &averageGain);
    void InsertGainHeap(ThreadSearchContext &threadData);
    void InsertNewNodesGainHeap(std::vector<VertexType> &newNodes, NodeSelectionContainerT &nodes, ThreadSearchContext &threadData);

    inline void ComputeNodeAffinities(VertexType node,
                                      std::vector<std::vector<CostT>> &affinityTableNode,
                                      ThreadSearchContext &threadData) {
        compute_work_affinity(node, affinity_table_node, thread_data);
        commCostF_.compute_comm_affinity(node,
                                         affinityTableNode,
                                         threadData.rewardPenaltyStrat_.penalty,
                                         threadData.rewardPenaltyStrat_.reward,
                                         threadData.startStep_,
                                         threadData.endStep_);
    }

    void SelectActiveNodes(ThreadSearchContext &threadData) {
        if (SelectNodesCheckRemoveSuperstep(threadData.stepToRemove_, threadData)) {
            activeSchedule_.swap_empty_step_fwd(threadData.stepToRemove_, threadData.endStep_);
            threadData.endStep_--;
            threadData.localSearchStartStep_ = static_cast<unsigned>(threadData.activeScheduleData_.applied_moves.size());
            threadData.activeScheduleData_.update_cost(static_cast<CostT>(-1.0 * instance_->SynchronisationCosts()));

            if constexpr (enablePreresolvingViolations_) {
                ResolveViolations(threadData);
            }

            if (threadData.activeScheduleData_.current_violations.size() > parameters_.initialViolationThreshold_) {
                activeSchedule_.revert_to_best_schedule(threadData.localSearchStartStep_,
                                                        threadData.stepToRemove_,
                                                        commCostF_,
                                                        threadData.activeScheduleData_,
                                                        threadData.startStep_,
                                                        threadData.endStep_);
            } else {
                threadData.unlockEdgeBacktrackCounter_
                    = static_cast<unsigned>(threadData.activeScheduleData_.current_violations.size());
                threadData.maxInnerIterations_
                    = std::max(threadData.unlockEdgeBacktrackCounter_ * 5u, parameters_.maxInnerIterationsReset_);
                threadData.maxNoVioaltionsRemovedBacktrack_ = parameters_.maxNoVioaltionsRemovedBacktrackForRemoveStepReset_;
#ifdef KL_DEBUG_1
                std::cout << "thread " << thread_data.thread_id << ", Trying to remove step " << thread_data.step_to_remove
                          << std::endl;
#endif
                return;
            }
        }
        // thread_data.step_to_remove = thread_data.start_step;
        threadData.localSearchStartStep_ = 0;
        threadData.selectionStrategy_.select_active_nodes(threadData.affinityTable_, threadData.startStep_, threadData.endStep_);
    }

    bool CheckRemoveSuperstep(unsigned step);
    bool SelectNodesCheckRemoveSuperstep(unsigned &step, ThreadSearchContext &threadData);

    bool ScatterNodesSuperstep(unsigned step, ThreadSearchContext &threadData) {
        assert(step <= threadData.endStep_ && threadData.startStep_ <= step);
        bool abort = false;

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            const std::vector<VertexType> stepProcNodeVec(
                activeSchedule_.getSetSchedule().step_processor_vertices[step][proc].begin(),
                activeSchedule_.getSetSchedule().step_processor_vertices[step][proc].end());
            for (const auto &node : step_proc_node_vec) {
                thread_data.reward_penalty_strat.init_reward_penalty(
                    static_cast<double>(thread_data.active_schedule_data.current_violations.size()) + 1.0);
                compute_node_affinities(node, thread_data.local_affinity_table, thread_data);
                kl_move best_move = compute_best_move<false>(node, thread_data.local_affinity_table, thread_data);

                if (best_move.gain <= std::numeric_limits<double>::lowest()) {
                    abort = true;
                    break;
                }

                apply_move(best_move, thread_data);
                if (thread_data.active_schedule_data.current_violations.size()
                    > parameters.abort_scatter_nodes_violation_threshold) {
                    abort = true;
                    break;
                }

                thread_data.affinity_table.insert(node);
                // thread_data.selection_strategy.add_neighbours_to_selection(node, thread_data.affinity_table,
                // thread_data.start_step, thread_data.end_step);
                if (thread_data.active_schedule_data.new_violations.size() > 0) {
                    for (const auto &vertex_edge_pair : thread_data.active_schedule_data.new_violations) {
                        const auto &vertex = vertex_edge_pair.first;
                        thread_data.affinity_table.insert(vertex);
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
            activeSchedule_.revert_to_best_schedule(
                0, 0, commCostF_, threadData.activeScheduleData_, threadData.startStep_, threadData.endStep_);
            threadData.affinityTable_.reset_node_selection();
            return false;
        }
        return true;
    }

    void SynchronizeActiveSchedule(const unsigned numThreads) {
        if (numThreads == 1) {    // single thread case
            activeSchedule_.set_cost(threadDataVec_[0].active_schedule_data.cost);
            activeSchedule_.getVectorSchedule().number_of_supersteps = threadDataVec_[0].num_steps();
            return;
        }

        unsigned writeCursor = threadDataVec_[0].end_step + 1;
        for (unsigned i = 1; i < numThreads; ++i) {
            auto &thread = threadDataVec_[i];
            if (thread.start_step <= thread.end_step) {
                for (unsigned j = thread.start_step; j <= thread.end_step; ++j) {
                    if (j != writeCursor) {
                        activeSchedule_.swap_steps(j, writeCursor);
                    }
                    writeCursor++;
                }
            }
        }
        activeSchedule_.getVectorSchedule().number_of_supersteps = writeCursor;
        const CostT newCost = commCostF_.compute_schedule_cost();
        activeSchedule_.set_cost(newCost);
    }

  public:
    KlImprover() : ImprovementScheduler<GraphT>() {
        std::random_device rd;
        gen_ = std::mt19937(rd());
    }

    explicit KlImprover(unsigned seed) : ImprovementScheduler<GraphT>() { gen_ = std::mt19937(seed); }

    virtual ~KlImprover() = default;

    virtual ReturnStatus improveSchedule(BspSchedule<GraphT> &schedule) override {
        if (schedule.GetInstance().NumberOfProcessors() < 2) {
            return ReturnStatus::BEST_FOUND;
        }

        const unsigned numThreads = 1;

        threadDataVec_.resize(numThreads);
        threadFinishedVec_.assign(numThreads, true);

        set_parameters(schedule.GetInstance().NumberOfVertices());
        InitializeDatastructures(schedule);
        const CostT initialCost = activeSchedule_.get_cost();
        const unsigned numSteps = schedule.NumberOfSupersteps();

        SetStartStep(0, threadDataVec_[0]);
        threadDataVec_[0].end_step = (numSteps > 0) ? numSteps - 1 : 0;

        auto &threadData = this->threadDataVec_[0];
        threadData.active_schedule_data.initialize_cost(activeSchedule_.get_cost());
        threadData.selection_strategy.setup(threadData.start_step, threadData.end_step);
        RunLocalSearch(threadData);

        SynchronizeActiveSchedule(numThreads);

        if (initialCost > activeSchedule_.get_cost()) {
            activeSchedule_.write_schedule(schedule);
            CleanupDatastructures();
            return ReturnStatus::OSP_SUCCESS;
        } else {
            CleanupDatastructures();
            return ReturnStatus::BEST_FOUND;
        }
    }

    virtual ReturnStatus improveScheduleWithTimeLimit(BspSchedule<GraphT> &schedule) override {
        computeWithTimeLimit_ = true;
        return improveSchedule(schedule);
    }

    virtual void SetTimeQualityParameter(const double timeQuality) { this->parameters_.timeQuality_ = timeQuality; }

    virtual void SetSuperstepRemoveStrengthParameter(const double superstepRemoveStrength) {
        this->parameters_.superstepRemoveStrength_ = superstepRemoveStrength;
    }

    virtual std::string GetScheduleName() const { return "kl_improver_" + commCostF_.name(); }
};

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::SetParameters(VertexIdxT<GraphT> numNodes) {
    const unsigned logNumNodes = (numNodes > 1) ? static_cast<unsigned>(std::log(num_nodes)) : 1;

    // Total number of outer iterations. Proportional to sqrt N.
    parameters_.maxOuterIterations_
        = static_cast<unsigned>(std::sqrt(num_nodes) * (parameters_.timeQuality_ * 10.0) / parameters_.numParallelLoops_);

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
        thread.thread_id = i++;
        // The number of nodes to consider in each inner iteration. Proportional to log(N).
        thread.selection_strategy.selection_threshold
            = static_cast<std::size_t>(std::ceil(parameters_.timeQuality_ * 10 * logNumNodes + logNumNodes));
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

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateNodeWorkAffinity(
    NodeSelectionContainerT &nodes,
    kl_move move,
    const pre_move_work_data<work_weight_t> &prevWorkData,
    std::map<VertexType, kl_gain_update_info> &recomputeMaxGain) {
    const size_t activeCount = nodes.size();

    for (size_t i = 0; i < activeCount; ++i) {
        const VertexType node = nodes.get_selected_nodes()[i];

        kl_gain_update_info updateInfo = update_node_work_affinity_after_move(node, move, prev_work_data, nodes.at(node));
        if (updateInfo.update_from_step || update_info.update_to_step) {
            recomputeMaxGain[node] = update_info;
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateMaxGain(
    kl_move move, std::map<VertexType, kl_gain_update_info> &recomputeMaxGain, ThreadSearchContext &threadData) {
    for (auto &pair : recompute_max_gain) {
        if (pair.second.full_update) {
            recompute_node_max_gain(pair.first, thread_data.affinity_table, thread_data);
        } else {
            if (pair.second.update_entire_from_step) {
                update_best_move(pair.first, move.from_step, thread_data.affinity_table, thread_data);
            } else if (pair.second.update_from_step && is_compatible(pair.first, move.from_proc)) {
                update_best_move(pair.first, move.from_step, move.from_proc, thread_data.affinity_table, thread_data);
            }

            if (move.from_step != move.to_step || not pair.second.update_entire_from_step) {
                if (pair.second.update_entire_to_step) {
                    update_best_move(pair.first, move.to_step, thread_data.affinity_table, thread_data);
                } else if (pair.second.update_to_step && is_compatible(pair.first, move.to_proc)) {
                    update_best_move(pair.first, move.to_step, move.to_proc, thread_data.affinity_table, thread_data);
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ComputeWorkAffinity(
    VertexType node, std::vector<std::vector<CostT>> &affinityTableNode, ThreadSearchContext &threadData) {
    const unsigned nodeStep = activeSchedule_.assigned_superstep(node);
    const work_weight_t vertexWeight = graph_->VertexWorkWeight(node);

    unsigned step = (nodeStep > windowSize) ? (nodeStep - windowSize) : 0;
    for (unsigned idx = threadData.StartIdx(nodeStep); idx < threadData.EndIdx(nodeStep); ++idx, ++step) {
        if (idx == windowSize) {
            continue;
        }

        const CostT maxWorkForStep = static_cast<CostT>(activeSchedule_.get_step_max_work(step));

        for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
            const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(step, proc);
            const cost_t work_diff = static_cast<cost_t>(new_weight) - max_work_for_step;
            affinity_table_node[proc][idx] = std::max(0.0, work_diff);
        }
    }

    const unsigned nodeProc = activeSchedule_.assigned_processor(node);
    const work_weight_t maxWorkForStep = activeSchedule_.get_step_max_work(nodeStep);
    const bool isSoleMaxProcessor = (activeSchedule_.get_step_max_work_processor_count()[nodeStep] == 1)
                                    && (maxWorkForStep == activeSchedule_.get_step_processor_work(nodeStep, nodeProc));

    const CostT nodeProcAffinity
        = isSoleMaxProcessor ? std::min(vertex_weight, max_work_for_step - activeSchedule_.get_step_second_max_work(nodeStep))
                             : 0.0;
    affinityTableNode[nodeProc][windowSize] = nodeProcAffinity;

    for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
        if (proc == node_proc) {
            continue;
        }

        const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, proc);
        affinity_table_node[proc][window_size] = compute_same_step_affinity(max_work_for_step, new_weight, node_proc_affinity);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ProcessWorkUpdateStep(
    VertexType node,
    unsigned nodeStep,
    unsigned nodeProc,
    work_weight_t vertexWeight,
    unsigned moveStep,
    unsigned moveProc,
    work_weight_t moveCorrectionNodeWeight,
    const work_weight_t prevMoveStepMaxWork,
    const work_weight_t prevMoveStepSecondMaxWork,
    unsigned prevMoveStepMaxWorkProcessorCount,
    bool &updateStep,
    bool &updateEntireStep,
    bool &fullUpdate,
    std::vector<std::vector<CostT>> &affinityTableNode) {
    const unsigned lowerBound = moveStep > windowSize ? moveStep - windowSize : 0;
    if (lowerBound <= nodeStep && nodeStep <= moveStep + windowSize) {
        updateStep = true;
        if (nodeStep == moveStep) {
            const work_weight_t newMaxWeight = activeSchedule_.get_step_max_work(moveStep);
            const work_weight_t newSecondMaxWeight = activeSchedule_.get_step_second_max_work(moveStep);
            const work_weight_t newStepProcWork = activeSchedule_.get_step_processor_work(nodeStep, nodeProc);

            const work_weight_t prevStepProcWork = (nodeProc == moveProc) ? new_step_proc_work + move_correction_node_weight
                                                                          : new_step_proc_work;
            const bool prevIsSoleMaxProcessor = (prevMoveStepMaxWorkProcessorCount == 1)
                                                && (prevMoveStepMaxWork == prev_step_proc_work);
            const CostT prevNodeProcAffinity
                = prevIsSoleMaxProcessor ? std::min(vertex_weight, prev_move_step_max_work - prev_move_step_second_max_work) : 0.0;

            const bool newIsSoleMaxProcessor = (activeSchedule_.get_step_max_work_processor_count()[nodeStep] == 1)
                                               && (newMaxWeight == new_step_proc_work);
            const CostT newNodeProcAffinity
                = newIsSoleMaxProcessor ? std::min(vertex_weight, new_max_weight - new_second_max_weight) : 0.0;

            const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
            const bool updateNodeProcAffinity = std::abs(diff) > epsilon_;
            if (updateNodeProcAffinity) {
                fullUpdate = true;
                affinityTableNode[nodeProc][windowSize] += diff;
            }

            if ((prevMoveStepMaxWork != new_max_weight) || updateNodeProcAffinity) {
                updateEntireStep = true;

                for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                    if ((proc == node_proc) || (proc == move_proc)) {
                        continue;
                    }

                    const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, proc);
                    const cost_t prev_other_affinity
                        = compute_same_step_affinity(prev_move_step_max_work, new_weight, prev_node_proc_affinity);
                    const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);

                    affinity_table_node[proc][window_size] += (other_affinity - prev_other_affinity);
                }
            }

            if (node_proc != move_proc && is_compatible(node, move_proc)) {
                const work_weight_t prevNewWeight
                    = vertex_weight + activeSchedule_.get_step_processor_work(nodeStep, moveProc) + move_correction_node_weight;
                const CostT prevOtherAffinity
                    = compute_same_step_affinity(prev_move_step_max_work, prev_new_weight, prev_node_proc_affinity);
                const work_weight_t newWeight = vertex_weight + activeSchedule_.get_step_processor_work(nodeStep, moveProc);
                const CostT otherAffinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);

                affinityTableNode[moveProc][windowSize] += (otherAffinity - prevOtherAffinity);
            }

        } else {
            const work_weight_t newMaxWeight = activeSchedule_.get_step_max_work(moveStep);
            const unsigned idx = RelStepIdx(nodeStep, moveStep);
            if (prevMoveStepMaxWork != new_max_weight) {
                updateEntireStep = true;

                // update moving to all procs with special for move_proc
                for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                    const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(move_step, proc);
                    if (proc != move_proc) {
                        const cost_t prev_affinity
                            = prev_move_step_max_work < new_weight
                                  ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(prev_move_step_max_work)
                                  : 0.0;
                        const cost_t new_affinity = new_max_weight < new_weight
                                                        ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight)
                                                        : 0.0;
                        affinity_table_node[proc][idx] += new_affinity - prev_affinity;

                    } else {
                        const work_weight_t prev_new_weight = vertex_weight
                                                              + active_schedule.get_step_processor_work(move_step, proc)
                                                              + move_correction_node_weight;
                        const cost_t prev_affinity
                            = prev_move_step_max_work < prev_new_weight
                                  ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_move_step_max_work)
                                  : 0.0;

                        const cost_t new_affinity = new_max_weight < new_weight
                                                        ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight)
                                                        : 0.0;
                        affinity_table_node[proc][idx] += new_affinity - prev_affinity;
                    }
                }
            } else {
                // update only move_proc
                if (is_compatible(node, move_proc)) {
                    const work_weight_t newWeight = vertex_weight + activeSchedule_.get_step_processor_work(moveStep, moveProc);
                    const work_weight_t prevNewWeight = new_weight + move_correction_node_weight;
                    const CostT prevAffinity
                        = prev_move_step_max_work < prev_new_weight
                              ? static_cast<CostT>(prev_new_weight) - static_cast<CostT>(prev_move_step_max_work)
                              : 0.0;

                    const CostT newAffinity
                        = new_max_weight < new_weight ? static_cast<CostT>(new_weight) - static_cast<CostT>(new_max_weight) : 0.0;
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
        std::cout << "Checking to remove step " << step_to_remove << "/" << thread_data.end_step << std::endl;
#endif
        if (CheckRemoveSuperstep(stepToRemove)) {
#ifdef KL_DEBUG
            std::cout << "Checking to scatter step " << step_to_remove << "/" << thread_data.end_step << std::endl;
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
    if (activeSchedule_.num_steps() < 2) {
        return false;
    }

    if (activeSchedule_.get_step_max_work(step) < instance_->SynchronisationCosts()) {
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
    threadData.affinityTable_.reset_node_selection();
    threadData.maxGainHeap_.clear();
    threadData.lockManager_.clear();
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::IsLocalSearchBlocked(
    ThreadSearchContext &threadData) {
    for (const auto &pair : threadData.activeScheduleData_.new_violations) {
        if (threadData.lockManager_.is_locked(pair.first)) {
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

    activeSchedule_.initialize(schedule);

    procRange_.initialize(*instance_);
    commCostF_.initialize(activeSchedule_, procRange_);
    const CostT initialCost = commCostF_.compute_schedule_cost();
    activeSchedule_.set_cost(initialCost);

    for (auto &tData : threadDataVec_) {
        tData.affinity_table.initialize(activeSchedule_, tData.selection_strategy.selection_threshold);
        tData.lock_manager.initialize(graph_->NumVertices());
        tData.reward_penalty_strat.initialize(
            activeSchedule_, commCostF_.get_max_comm_weight_multiplied(), activeSchedule_.get_max_work_weight());
        tData.selection_strategy.initialize(activeSchedule_, gen_, tData.start_step, tData.end_step);

        tData.local_affinity_table.resize(instance_->NumberOfProcessors());
        for (unsigned i = 0; i < instance_->NumberOfProcessors(); ++i) {
            tData.local_affinity_table[i].resize(windowRange_);
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
        const VertexType node = threadData.affinityTable_.get_selected_nodes()[i];
        compute_node_affinities(node, thread_data.affinity_table.at(node), thread_data);
        const auto bestMove = compute_best_move<true>(node, threadData.affinityTable_[node], threadData);
        threadData.maxGainHeap_.push(node, best_move);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::InsertNewNodesGainHeap(
    std::vector<VertexType> &newNodes, NodeSelectionContainerT &nodes, ThreadSearchContext &threadData) {
    for (const auto &node : new_nodes) {
        nodes.insert(node);
        compute_node_affinities(node, thread_data.affinity_table.at(node), thread_data);
        const auto best_move = compute_best_move<true>(node, thread_data.affinity_table[node], thread_data);
        thread_data.max_gain_heap.push(node, best_move);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::CleanupDatastructures() {
    threadDataVec_.clear();
    activeSchedule_.clear();
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::PrintHeap(heap_datastructure &maxGainHeap) const {
    if (maxGainHeap.is_empty()) {
        std::cout << "heap is empty" << std::endl;
        return;
    }
    heap_datastructure tempHeap = max_gain_heap;    // requires copy constructor

    std::cout << "heap current size: " << temp_heap.size() << std::endl;
    const auto &topVal = temp_heap.get_value(temp_heap.top());
    std::cout << "heap top node " << top_val.node << " gain " << top_val.gain << std::endl;

    unsigned count = 0;
    while (!temp_heap.is_empty() && count++ < 15) {
        const auto &val = temp_heap.get_value(temp_heap.top());
        std::cout << "node " << val.node << " gain " << val.gain << " to proc " << val.to_proc << " to step " << val.to_step
                  << std::endl;
        tempHeap.pop();
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = activeSchedule_.assigned_processor(node);
    const unsigned nodeStep = activeSchedule_.assigned_superstep(node);

    if ((nodeProc == proc) && (nodeStep == step)) {
        return;
    }

    kl_move nodeMove = threadData.maxGainHeap_.get_value(node);
    CostT maxGain = node_move.gain;

    unsigned maxProc = node_move.to_proc;
    unsigned maxStep = node_move.to_step;

    if ((maxStep == step) && (maxProc == proc)) {
        recompute_node_max_gain(node, affinity_table, thread_data);
    } else {
        if constexpr (ActiveScheduleT::use_memory_constraint) {
            if (not activeSchedule_.memory_constraint.can_move(node, proc, step)) {
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

        const CostT diff = maxGain - node_move.gain;
        if ((std::abs(diff) > epsilon_) || (maxProc != node_move.to_proc) || (maxStep != node_move.to_step)) {
            nodeMove.gain = maxGain;
            nodeMove.to_proc = maxProc;
            nodeMove.to_step = maxStep;
            threadData.maxGainHeap_.update(node, node_move);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = activeSchedule_.assigned_processor(node);
    const unsigned nodeStep = activeSchedule_.assigned_superstep(node);

    kl_move nodeMove = threadData.maxGainHeap_.get_value(node);
    CostT maxGain = node_move.gain;

    unsigned maxProc = node_move.to_proc;
    unsigned maxStep = node_move.to_step;

    if (maxStep == step) {
        recompute_node_max_gain(node, affinity_table, thread_data);
    } else {
        if (nodeStep != step) {
            const unsigned idx = RelStepIdx(nodeStep, step);
            for (const unsigned p : proc_range.compatible_processors_vertex(node)) {
                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.can_move(node, p, step)) {
                        continue;
                    }
                }
                const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][p][idx];
                if (gain > max_gain) {
                    max_gain = gain;
                    max_proc = p;
                    max_step = step;
                }
            }
        } else {
            for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                if (proc == node_proc) {
                    continue;
                }
                if constexpr (active_schedule_t::use_memory_constraint) {
                    if (not active_schedule.memory_constraint.can_move(node, proc, step)) {
                        continue;
                    }
                }
                const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][proc][window_size];
                if (gain > max_gain) {
                    max_gain = gain;
                    max_proc = proc;
                    max_step = step;
                }
            }
        }

        const CostT diff = maxGain - node_move.gain;
        if ((std::abs(diff) > epsilon_) || (maxProc != node_move.to_proc) || (maxStep != node_move.to_step)) {
            nodeMove.gain = maxGain;
            nodeMove.to_proc = maxProc;
            nodeMove.to_step = maxStep;
            threadData.maxGainHeap_.update(node, node_move);
        }
    }
}

}    // namespace osp
