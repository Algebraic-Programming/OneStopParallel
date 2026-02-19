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

#include "kl_improver_base.hpp"

namespace osp {

// =============================================================================
// HEAP VARIANT — for total / totalLambda cost functions
// =============================================================================
template <typename GraphT,
          typename CommCostFunctionT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          typename CostT = double>
class KlImproverHeap : public KlImproverBase<KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>,
                                             GraphT,
                                             CommCostFunctionT,
                                             MemoryConstraintT,
                                             windowSize,
                                             CostT> {
    using Base = KlImproverBase<KlImproverHeap, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>;
    friend Base;

  protected:
    using typename Base::ActiveScheduleT;
    using typename Base::HeapDatastructure;
    using typename Base::KlGainUpdateInfo;
    using typename Base::KlMove;
    using typename Base::NodeSelectionContainerT;
    using typename Base::ThreadSearchContext;
    using typename Base::VertexType;
    using typename Base::VertexWorkWeightT;

    // --- Per-thread heap data ---

    struct HeapThreadData {
        HeapDatastructure maxGainHeap_;
    };

    std::vector<HeapThreadData> heapData_;

    HeapThreadData &HD(ThreadSearchContext &td) { return heapData_[td.threadId_]; }

    // --- Heap-specific helpers ---

    inline void RecomputeNodeMaxGain(VertexType node, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
        const auto bestMove = this->template ComputeBestMove<true>(node, affinityTable[node], threadData);
        HD(threadData).maxGainHeap_.Update(node, bestMove);
    }

    // --- Incremental work affinity update methods ---

    void UpdateWorkAffinitySameStepOnMoveStep(VertexType node,
                                              const KlMove &move,
                                              const VertexWorkWeightT vertexWeight,
                                              const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                                              KlGainUpdateInfo &updateInfo,
                                              std::vector<std::vector<CostT>> &affinityTableNode);

    void UpdateWorkAffinitySameStepAdjacentToMove(VertexType node,
                                                  const KlMove &move,
                                                  unsigned nodeStep,
                                                  const VertexWorkWeightT vertexWeight,
                                                  const VertexWorkWeightT prevMaxWork,
                                                  KlGainUpdateInfo &updateInfo,
                                                  std::vector<std::vector<CostT>> &affinityTableNode);

    KlGainUpdateInfo UpdateNodeWorkAffinityAfterMove(VertexType node,
                                                     KlMove move,
                                                     const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                                                     std::vector<std::vector<CostT>> &affinityTableNode);

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

    void ProcessWorkUpdateNodeOnMoveStep(VertexType node,
                                         unsigned nodeStep,
                                         unsigned nodeProc,
                                         VertexWorkWeightT vertexWeight,
                                         unsigned moveProc,
                                         VertexWorkWeightT moveCorrectionNodeWeight,
                                         const VertexWorkWeightT prevMoveStepMaxWork,
                                         const VertexWorkWeightT prevMoveStepSecondMaxWork,
                                         unsigned prevMoveStepMaxWorkProcessorCount,
                                         bool &updateEntireStep,
                                         bool &fullUpdate,
                                         std::vector<std::vector<CostT>> &affinityTableNode);

    void ProcessWorkUpdateNodeAdjacentToMove(VertexType node,
                                             unsigned nodeStep,
                                             VertexWorkWeightT vertexWeight,
                                             unsigned moveStep,
                                             unsigned moveProc,
                                             VertexWorkWeightT moveCorrectionNodeWeight,
                                             const VertexWorkWeightT prevMoveStepMaxWork,
                                             bool &updateEntireStep,
                                             std::vector<std::vector<CostT>> &affinityTableNode);

    void UpdateNodeWorkAffinity(NodeSelectionContainerT &nodes,
                                KlMove move,
                                const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
                                std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain);

    void UpdateBestMove(
        VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);
    void UpdateBestMove(VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData);

    void UpdateMaxGain(KlMove move, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain, ThreadSearchContext &threadData);

    void PrintHeap(HeapDatastructure &maxGainHeap) const;

    // --- DISPATCH IMPLEMENTATIONS ---

    void ReinitializeMoveFinding(ThreadSearchContext &threadData) {
        auto &hd = HD(threadData);
        hd.maxGainHeap_.Clear();

        const size_t activeCount = threadData.affinityTable_.size();
        for (size_t i = 0; i < activeCount; ++i) {
            const VertexType node = threadData.affinityTable_.GetSelectedNodes()[i];
            this->ComputeNodeAffinities(node, threadData.affinityTable_.At(node), threadData);
            const auto bestMove = this->template ComputeBestMove<true>(node, threadData.affinityTable_[node], threadData);
            hd.maxGainHeap_.Push(node, bestMove);
        }
    }

    KlMove GetBestMove(ThreadSearchContext &threadData) {
        auto &hd = HD(threadData);
        if (hd.maxGainHeap_.size() == 0) {
            KlMove invalid;
            invalid.gain_ = std::numeric_limits<CostT>::lowest();
            return invalid;
        }

        // Tie-breaking: random among top equal-gain nodes
        const unsigned localMax = 50;
        std::vector<VertexType> topGainNodes = hd.maxGainHeap_.GetTopKeys(localMax);

        if (topGainNodes.empty()) {
            topGainNodes.push_back(hd.maxGainHeap_.Top());
        }

        std::uniform_int_distribution<size_t> dis(0, topGainNodes.size() - 1);
        const VertexType node = topGainNodes[dis(this->gen_)];

        KlMove bestMove = hd.maxGainHeap_.GetValue(node);
        hd.maxGainHeap_.Erase(node);
        threadData.lockManager_.Lock(node);
        threadData.affinityTable_.Remove(node);

        return bestMove;
    }

    void PostMoveUpdate(const KlMove &bestMove,
                        ThreadSearchContext &threadData,
                        std::vector<VertexType> &newNodes,
                        std::vector<VertexType> &unlockNodes,
                        const PreMoveWorkData<VertexWorkWeightT> &prevWorkData) {
        std::map<VertexType, KlGainUpdateInfo> recomputeMaxGain;

        // Incremental affinity updates
        // Note: unlockNodes are still LOCKED here, so UpdateNodeCommAffinity skips them.
        UpdateNodeWorkAffinity(threadData.affinityTable_, bestMove, prevWorkData, recomputeMaxGain);
        this->commCostF_.UpdateNodeCommAffinity(bestMove,
                                                threadData,
                                                threadData.rewardPenaltyStrat_.penalty_,
                                                threadData.rewardPenaltyStrat_.reward_,
                                                recomputeMaxGain,
                                                newNodes);

        this->DebugCostCheck(threadData);

        // Heap updates for recomputed existing nodes
        UpdateMaxGain(bestMove, recomputeMaxGain, threadData);

        // Now unlock and merge — after UpdateNodeCommAffinity has run
        for (const auto v : unlockNodes) {
            threadData.lockManager_.Unlock(v);
        }
        newNodes.insert(newNodes.end(), unlockNodes.begin(), unlockNodes.end());

        // Insert all new nodes into heap
        auto &hd = HD(threadData);
        for (const auto &node : newNodes) {
            threadData.affinityTable_.Insert(node);
            this->ComputeNodeAffinities(node, threadData.affinityTable_.At(node), threadData);
            const auto move = this->template ComputeBestMove<true>(node, threadData.affinityTable_[node], threadData);
            hd.maxGainHeap_.Push(node, move);
        }
    }

  public:
    using Base::Base;    // inherit constructors

    void InitializeVariantData() { heapData_.resize(this->threadDataVec_.size()); }
};

// =============================================================================
// OUT-OF-LINE DEFINITIONS — Heap variant
// =============================================================================

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateNodeWorkAffinity(
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
void KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateMaxGain(
    KlMove move, std::map<VertexType, KlGainUpdateInfo> &recomputeMaxGain, ThreadSearchContext &threadData) {
    auto &hd = HD(threadData);
    for (auto &pair : recomputeMaxGain) {
        if (pair.second.fullUpdate_) {
            RecomputeNodeMaxGain(pair.first, threadData.affinityTable_, threadData);
        } else {
            if (pair.second.updateEntireFromStep_) {
                UpdateBestMove(pair.first, move.fromStep_, threadData.affinityTable_, threadData);
            } else if (pair.second.updateFromStep_ && this->IsCompatible(pair.first, move.fromProc_)) {
                UpdateBestMove(pair.first, move.fromStep_, move.fromProc_, threadData.affinityTable_, threadData);
            }

            if (move.fromStep_ != move.toStep_ || not pair.second.updateEntireFromStep_) {
                if (pair.second.updateEntireToStep_) {
                    UpdateBestMove(pair.first, move.toStep_, threadData.affinityTable_, threadData);
                } else if (pair.second.updateToStep_ && this->IsCompatible(pair.first, move.toProc_)) {
                    UpdateBestMove(pair.first, move.toStep_, move.toProc_, threadData.affinityTable_, threadData);
                }
            }
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, unsigned proc, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = this->activeSchedule_.AssignedProcessor(node);
    const unsigned nodeStep = this->activeSchedule_.AssignedSuperstep(node);

    if ((nodeProc == proc) && (nodeStep == step)) {
        return;
    }

    auto &hd = HD(threadData);
    KlMove nodeMove = hd.maxGainHeap_.GetValue(node);
    CostT maxGain = nodeMove.gain_;

    unsigned maxProc = nodeMove.toProc_;
    unsigned maxStep = nodeMove.toStep_;

    if ((maxStep == step) && (maxProc == proc)) {
        RecomputeNodeMaxGain(node, affinityTable, threadData);
    } else {
        if constexpr (ActiveScheduleT::useMemoryConstraint_) {
            if (not this->activeSchedule_.memoryConstraint_.CanMove(node, proc, step)) {
                return;
            }
        }
        const unsigned idx = this->RelStepIdx(nodeStep, step);
        const CostT gain = affinityTable[node][nodeProc][windowSize] - affinityTable[node][proc][idx];
        if (gain > maxGain) {
            maxGain = gain;
            maxProc = proc;
            maxStep = step;
        }

        const CostT diff = maxGain - nodeMove.gain_;
        if ((std::abs(diff) > Base::epsilon_) || (maxProc != nodeMove.toProc_) || (maxStep != nodeMove.toStep_)) {
            nodeMove.gain_ = maxGain;
            nodeMove.toProc_ = maxProc;
            nodeMove.toStep_ = maxStep;
            hd.maxGainHeap_.Update(node, nodeMove);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateBestMove(
    VertexType node, unsigned step, NodeSelectionContainerT &affinityTable, ThreadSearchContext &threadData) {
    const unsigned nodeProc = this->activeSchedule_.AssignedProcessor(node);
    const unsigned nodeStep = this->activeSchedule_.AssignedSuperstep(node);

    auto &hd = HD(threadData);
    KlMove nodeMove = hd.maxGainHeap_.GetValue(node);
    CostT maxGain = nodeMove.gain_;

    unsigned maxProc = nodeMove.toProc_;
    unsigned maxStep = nodeMove.toStep_;

    if (maxStep == step) {
        RecomputeNodeMaxGain(node, affinityTable, threadData);
    } else {
        if (nodeStep != step) {
            const unsigned idx = this->RelStepIdx(nodeStep, step);
            for (const unsigned p : this->procRange_.CompatibleProcessorsVertex(node)) {
                if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                    if (not this->activeSchedule_.memoryConstraint_.CanMove(node, p, step)) {
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
            for (const unsigned proc : this->procRange_.CompatibleProcessorsVertex(node)) {
                if (proc == nodeProc) {
                    continue;
                }
                if constexpr (ActiveScheduleT::useMemoryConstraint_) {
                    if (not this->activeSchedule_.memoryConstraint_.CanMove(node, proc, step)) {
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
        if ((std::abs(diff) > Base::epsilon_) || (maxProc != nodeMove.toProc_) || (maxStep != nodeMove.toStep_)) {
            nodeMove.gain_ = maxGain;
            nodeMove.toProc_ = maxProc;
            nodeMove.toStep_ = maxStep;
            hd.maxGainHeap_.Update(node, nodeMove);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
typename KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::KlGainUpdateInfo
KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateNodeWorkAffinityAfterMove(
    VertexType node,
    KlMove move,
    const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
    std::vector<std::vector<CostT>> &affinityTableNode) {
    const unsigned nodeStep = this->activeSchedule_.AssignedSuperstep(node);
    const VertexWorkWeightT vertexWeight = this->graph_->VertexWorkWeight(node);

    KlGainUpdateInfo updateInfo(node);

    if (move.fromStep_ == move.toStep_) {
        const unsigned lowerBound = move.fromStep_ > windowSize ? move.fromStep_ - windowSize : 0;
        if (lowerBound <= nodeStep && nodeStep <= move.fromStep_ + windowSize) {
            updateInfo.updateFromStep_ = true;
            updateInfo.updateToStep_ = true;

            if (nodeStep == move.fromStep_) {
                UpdateWorkAffinitySameStepOnMoveStep(node, move, vertexWeight, prevWorkData, updateInfo, affinityTableNode);
            } else {
                UpdateWorkAffinitySameStepAdjacentToMove(
                    node, move, nodeStep, vertexWeight, prevWorkData.fromStepMaxWork_, updateInfo, affinityTableNode);
            }
        }
    } else {
        const unsigned nodeProc = this->activeSchedule_.AssignedProcessor(node);
        ProcessWorkUpdateStep(node,
                              nodeStep,
                              nodeProc,
                              vertexWeight,
                              move.fromStep_,
                              move.fromProc_,
                              this->graph_->VertexWorkWeight(move.node_),
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
                              -this->graph_->VertexWorkWeight(move.node_),
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

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateWorkAffinitySameStepOnMoveStep(
    VertexType node,
    const KlMove &move,
    const VertexWorkWeightT vertexWeight,
    const PreMoveWorkData<VertexWorkWeightT> &prevWorkData,
    KlGainUpdateInfo &updateInfo,
    std::vector<std::vector<CostT>> &affinityTableNode) {
    const unsigned nodeStep = move.fromStep_;
    const unsigned nodeProc = this->activeSchedule_.AssignedProcessor(node);
    const VertexWorkWeightT prevMaxWork = prevWorkData.fromStepMaxWork_;
    const VertexWorkWeightT newMaxWeight = this->activeSchedule_.GetStepMaxWork(nodeStep);
    const VertexWorkWeightT newSecondMaxWeight = this->activeSchedule_.GetStepSecondMaxWork(nodeStep);
    const VertexWorkWeightT newStepProcWork = this->activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);

    const VertexWorkWeightT prevStepProcWork
        = (nodeProc == move.fromProc_) ? newStepProcWork + this->graph_->VertexWorkWeight(move.node_)
          : (nodeProc == move.toProc_) ? newStepProcWork - this->graph_->VertexWorkWeight(move.node_)
                                       : newStepProcWork;
    const bool prevIsSoleMaxProcessor = (prevWorkData.fromStepMaxWorkProcessorCount_ == 1) && (prevMaxWork == prevStepProcWork);
    const CostT prevNodeProcAffinity
        = prevIsSoleMaxProcessor ? std::min(vertexWeight, prevMaxWork - prevWorkData.fromStepSecondMaxWork_) : 0.0;
    const bool newIsSoleMaxProcessor = (this->activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                       && (newMaxWeight == newStepProcWork);
    const CostT newNodeProcAffinity = newIsSoleMaxProcessor ? std::min(vertexWeight, newMaxWeight - newSecondMaxWeight) : 0.0;

    const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
    if (std::abs(diff) > Base::epsilon_) {
        updateInfo.fullUpdate_ = true;
        affinityTableNode[nodeProc][windowSize] += diff;
    }

    if ((prevMaxWork != newMaxWeight) || updateInfo.fullUpdate_) {
        updateInfo.updateEntireFromStep_ = true;

        for (const unsigned proc : this->procRange_.CompatibleProcessorsVertex(node)) {
            if ((proc == nodeProc) || (proc == move.fromProc_) || (proc == move.toProc_)) {
                continue;
            }

            const VertexWorkWeightT newWeight = vertexWeight + this->activeSchedule_.GetStepProcessorWork(nodeStep, proc);
            const CostT prevOtherAffinity = this->ComputeSameStepAffinity(prevMaxWork, newWeight, prevNodeProcAffinity);
            const CostT otherAffinity = this->ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

            affinityTableNode[proc][windowSize] += (otherAffinity - prevOtherAffinity);
        }
    }

    if (nodeProc != move.fromProc_ && this->IsCompatible(node, move.fromProc_)) {
        const VertexWorkWeightT prevNewWeight = vertexWeight + this->activeSchedule_.GetStepProcessorWork(nodeStep, move.fromProc_)
                                                + this->graph_->VertexWorkWeight(move.node_);
        const CostT prevOtherAffinity = this->ComputeSameStepAffinity(prevMaxWork, prevNewWeight, prevNodeProcAffinity);
        const VertexWorkWeightT newWeight = vertexWeight + this->activeSchedule_.GetStepProcessorWork(nodeStep, move.fromProc_);
        const CostT otherAffinity = this->ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
        affinityTableNode[move.fromProc_][windowSize] += (otherAffinity - prevOtherAffinity);
    }

    if (nodeProc != move.toProc_ && this->IsCompatible(node, move.toProc_)) {
        const VertexWorkWeightT prevNewWeight = vertexWeight + this->activeSchedule_.GetStepProcessorWork(nodeStep, move.toProc_)
                                                - this->graph_->VertexWorkWeight(move.node_);
        const CostT prevOtherAffinity = this->ComputeSameStepAffinity(prevMaxWork, prevNewWeight, prevNodeProcAffinity);
        const VertexWorkWeightT newWeight = vertexWeight + this->activeSchedule_.GetStepProcessorWork(nodeStep, move.toProc_);
        const CostT otherAffinity = this->ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);
        affinityTableNode[move.toProc_][windowSize] += (otherAffinity - prevOtherAffinity);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::UpdateWorkAffinitySameStepAdjacentToMove(
    VertexType node,
    const KlMove &move,
    unsigned nodeStep,
    const VertexWorkWeightT vertexWeight,
    const VertexWorkWeightT prevMaxWork,
    KlGainUpdateInfo &updateInfo,
    std::vector<std::vector<CostT>> &affinityTableNode) {
    const VertexWorkWeightT newMaxWeight = this->activeSchedule_.GetStepMaxWork(move.fromStep_);
    const unsigned idx = this->RelStepIdx(nodeStep, move.fromStep_);
    if (prevMaxWork != newMaxWeight) {
        updateInfo.updateEntireFromStep_ = true;
        for (const unsigned proc : this->procRange_.CompatibleProcessorsVertex(node)) {
            const VertexWorkWeightT newWeight = vertexWeight + this->activeSchedule_.GetStepProcessorWork(move.fromStep_, proc);
            if (proc == move.fromProc_) {
                const VertexWorkWeightT prevNewWeight = newWeight + this->graph_->VertexWorkWeight(move.node_);
                affinityTableNode[proc][idx] += this->ComputeDiffStepAffinity(newMaxWeight, newWeight)
                                                - this->ComputeDiffStepAffinity(prevMaxWork, prevNewWeight);
            } else if (proc == move.toProc_) {
                const VertexWorkWeightT prevNewWeight = vertexWeight
                                                        + this->activeSchedule_.GetStepProcessorWork(move.toStep_, proc)
                                                        - this->graph_->VertexWorkWeight(move.node_);
                affinityTableNode[proc][idx] += this->ComputeDiffStepAffinity(newMaxWeight, newWeight)
                                                - this->ComputeDiffStepAffinity(prevMaxWork, prevNewWeight);
            } else {
                affinityTableNode[proc][idx] += this->ComputeDiffStepAffinity(newMaxWeight, newWeight)
                                                - this->ComputeDiffStepAffinity(prevMaxWork, newWeight);
            }
        }
    } else {
        if (this->IsCompatible(node, move.fromProc_)) {
            const VertexWorkWeightT fromNewWeight
                = vertexWeight + this->activeSchedule_.GetStepProcessorWork(move.fromStep_, move.fromProc_);
            const VertexWorkWeightT fromPrevNewWeight = fromNewWeight + this->graph_->VertexWorkWeight(move.node_);
            affinityTableNode[move.fromProc_][idx] += this->ComputeDiffStepAffinity(newMaxWeight, fromNewWeight)
                                                      - this->ComputeDiffStepAffinity(prevMaxWork, fromPrevNewWeight);
        }

        if (this->IsCompatible(node, move.toProc_)) {
            const VertexWorkWeightT toNewWeight
                = vertexWeight + this->activeSchedule_.GetStepProcessorWork(move.toStep_, move.toProc_);
            const VertexWorkWeightT toPrevNewWeight = toNewWeight - this->graph_->VertexWorkWeight(move.node_);
            affinityTableNode[move.toProc_][idx] += this->ComputeDiffStepAffinity(newMaxWeight, toNewWeight)
                                                    - this->ComputeDiffStepAffinity(prevMaxWork, toPrevNewWeight);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ProcessWorkUpdateStep(
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
    if (!(lowerBound <= nodeStep && nodeStep <= moveStep + windowSize)) {
        return;
    }

    updateStep = true;
    if (nodeStep == moveStep) {
        ProcessWorkUpdateNodeOnMoveStep(node,
                                        nodeStep,
                                        nodeProc,
                                        vertexWeight,
                                        moveProc,
                                        moveCorrectionNodeWeight,
                                        prevMoveStepMaxWork,
                                        prevMoveStepSecondMaxWork,
                                        prevMoveStepMaxWorkProcessorCount,
                                        updateEntireStep,
                                        fullUpdate,
                                        affinityTableNode);
    } else {
        ProcessWorkUpdateNodeAdjacentToMove(node,
                                            nodeStep,
                                            vertexWeight,
                                            moveStep,
                                            moveProc,
                                            moveCorrectionNodeWeight,
                                            prevMoveStepMaxWork,
                                            updateEntireStep,
                                            affinityTableNode);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ProcessWorkUpdateNodeOnMoveStep(
    VertexType node,
    unsigned nodeStep,
    unsigned nodeProc,
    VertexWorkWeightT vertexWeight,
    unsigned moveProc,
    VertexWorkWeightT moveCorrectionNodeWeight,
    const VertexWorkWeightT prevMoveStepMaxWork,
    const VertexWorkWeightT prevMoveStepSecondMaxWork,
    unsigned prevMoveStepMaxWorkProcessorCount,
    bool &updateEntireStep,
    bool &fullUpdate,
    std::vector<std::vector<CostT>> &affinityTableNode) {
    const VertexWorkWeightT newMaxWeight = this->activeSchedule_.GetStepMaxWork(nodeStep);
    const VertexWorkWeightT newSecondMaxWeight = this->activeSchedule_.GetStepSecondMaxWork(nodeStep);
    const VertexWorkWeightT newStepProcWork = this->activeSchedule_.GetStepProcessorWork(nodeStep, nodeProc);

    const VertexWorkWeightT prevStepProcWork = (nodeProc == moveProc) ? newStepProcWork + moveCorrectionNodeWeight
                                                                      : newStepProcWork;
    const bool prevIsSoleMaxProcessor = (prevMoveStepMaxWorkProcessorCount == 1) && (prevMoveStepMaxWork == prevStepProcWork);
    const CostT prevNodeProcAffinity
        = prevIsSoleMaxProcessor ? std::min(vertexWeight, prevMoveStepMaxWork - prevMoveStepSecondMaxWork) : 0.0;

    const bool newIsSoleMaxProcessor = (this->activeSchedule_.GetStepMaxWorkProcessorCount()[nodeStep] == 1)
                                       && (newMaxWeight == newStepProcWork);
    const CostT newNodeProcAffinity = newIsSoleMaxProcessor ? std::min(vertexWeight, newMaxWeight - newSecondMaxWeight) : 0.0;

    const CostT diff = newNodeProcAffinity - prevNodeProcAffinity;
    const bool updateNodeProcAffinity = std::abs(diff) > Base::epsilon_;
    if (updateNodeProcAffinity) {
        fullUpdate = true;
        affinityTableNode[nodeProc][windowSize] += diff;
    }

    if ((prevMoveStepMaxWork != newMaxWeight) || updateNodeProcAffinity) {
        updateEntireStep = true;

        for (const unsigned proc : this->procRange_.CompatibleProcessorsVertex(node)) {
            if ((proc == nodeProc) || (proc == moveProc)) {
                continue;
            }

            const VertexWorkWeightT newWeight = vertexWeight + this->activeSchedule_.GetStepProcessorWork(nodeStep, proc);
            const CostT prevOtherAffinity = this->ComputeSameStepAffinity(prevMoveStepMaxWork, newWeight, prevNodeProcAffinity);
            const CostT otherAffinity = this->ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

            affinityTableNode[proc][windowSize] += (otherAffinity - prevOtherAffinity);
        }
    }

    if (nodeProc != moveProc && this->IsCompatible(node, moveProc)) {
        const VertexWorkWeightT prevNewWeight
            = vertexWeight + this->activeSchedule_.GetStepProcessorWork(nodeStep, moveProc) + moveCorrectionNodeWeight;
        const CostT prevOtherAffinity = this->ComputeSameStepAffinity(prevMoveStepMaxWork, prevNewWeight, prevNodeProcAffinity);
        const VertexWorkWeightT newWeight = vertexWeight + this->activeSchedule_.GetStepProcessorWork(nodeStep, moveProc);
        const CostT otherAffinity = this->ComputeSameStepAffinity(newMaxWeight, newWeight, newNodeProcAffinity);

        affinityTableNode[moveProc][windowSize] += (otherAffinity - prevOtherAffinity);
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::ProcessWorkUpdateNodeAdjacentToMove(
    VertexType node,
    unsigned nodeStep,
    VertexWorkWeightT vertexWeight,
    unsigned moveStep,
    unsigned moveProc,
    VertexWorkWeightT moveCorrectionNodeWeight,
    const VertexWorkWeightT prevMoveStepMaxWork,
    bool &updateEntireStep,
    std::vector<std::vector<CostT>> &affinityTableNode) {
    const VertexWorkWeightT newMaxWeight = this->activeSchedule_.GetStepMaxWork(moveStep);
    const unsigned idx = this->RelStepIdx(nodeStep, moveStep);
    if (prevMoveStepMaxWork != newMaxWeight) {
        updateEntireStep = true;

        for (const unsigned proc : this->procRange_.CompatibleProcessorsVertex(node)) {
            const VertexWorkWeightT newWeight = vertexWeight + this->activeSchedule_.GetStepProcessorWork(moveStep, proc);
            if (proc != moveProc) {
                affinityTableNode[proc][idx] += this->ComputeDiffStepAffinity(newMaxWeight, newWeight)
                                                - this->ComputeDiffStepAffinity(prevMoveStepMaxWork, newWeight);
            } else {
                const VertexWorkWeightT prevNewWeight = newWeight + moveCorrectionNodeWeight;
                affinityTableNode[proc][idx] += this->ComputeDiffStepAffinity(newMaxWeight, newWeight)
                                                - this->ComputeDiffStepAffinity(prevMoveStepMaxWork, prevNewWeight);
            }
        }
    } else {
        if (this->IsCompatible(node, moveProc)) {
            const VertexWorkWeightT newWeight = vertexWeight + this->activeSchedule_.GetStepProcessorWork(moveStep, moveProc);
            const VertexWorkWeightT prevNewWeight = newWeight + moveCorrectionNodeWeight;
            affinityTableNode[moveProc][idx] += this->ComputeDiffStepAffinity(newMaxWeight, newWeight)
                                                - this->ComputeDiffStepAffinity(prevMoveStepMaxWork, prevNewWeight);
        }
    }
}

template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
void KlImproverHeap<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>::PrintHeap(HeapDatastructure &maxGainHeap) const {
    if (maxGainHeap.IsEmpty()) {
        std::cout << "heap is empty" << std::endl;
        return;
    }
    HeapDatastructure tempHeap = maxGainHeap;

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

}    // namespace osp
