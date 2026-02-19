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
// SCAN VARIANT — for BSP max-comm cost functions
// =============================================================================
template <typename GraphT,
          typename CommCostFunctionT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned windowSize = 1,
          typename CostT = double>
class KlImproverScan : public KlImproverBase<KlImproverScan<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>,
                                             GraphT,
                                             CommCostFunctionT,
                                             MemoryConstraintT,
                                             windowSize,
                                             CostT> {
    using Base = KlImproverBase<KlImproverScan, GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>;
    friend Base;

  protected:
    using typename Base::KlMove;
    using typename Base::NodeSelectionContainerT;
    using typename Base::ThreadSearchContext;
    using typename Base::VertexType;
    using typename Base::VertexWorkWeightT;

    // --- Per-thread scan data ---

    struct ScanThreadData {
        KlMove currentBest_;
    };

    std::vector<ScanThreadData> scanData_;

    ScanThreadData &SD(ThreadSearchContext &td) { return scanData_[td.threadId_]; }

    // --- Core: recompute all unlocked active nodes, find global best ---

    KlMove ComputeAllAffinitiesAndFindBest(ThreadSearchContext &threadData) {
        KlMove globalBest;
        globalBest.gain_ = std::numeric_limits<CostT>::lowest();

        const unsigned numProcs = this->activeSchedule_.GetInstance().NumberOfProcessors();
        const size_t activeCount = threadData.affinityTable_.size();

        for (size_t i = 0; i < activeCount; ++i) {
            const VertexType node = threadData.affinityTable_.GetSelectedNodes()[i];
            if (threadData.lockManager_.IsLocked(node)) {
                continue;
            }

            auto &atn = threadData.affinityTable_.At(node);
            for (unsigned p = 0; p < numProcs; ++p) {
                for (unsigned idx = 0; idx < atn[p].size(); ++idx) {
                    atn[p][idx] = 0;
                }
            }

            this->ComputeNodeAffinities(node, atn, threadData);

            const auto bestMove = this->template ComputeBestMove<true>(node, atn, threadData);

            if (bestMove.gain_ > globalBest.gain_) {
                globalBest = bestMove;
            }
        }
        return globalBest;
    }

    // --- DISPATCH IMPLEMENTATIONS ---

    void ReinitializeMoveFinding(ThreadSearchContext &threadData) {
        SD(threadData).currentBest_ = ComputeAllAffinitiesAndFindBest(threadData);
    }

    KlMove GetBestMove(ThreadSearchContext &threadData) {
        auto &sd = SD(threadData);
        KlMove move = sd.currentBest_;
        if (move.gain_ > std::numeric_limits<CostT>::lowest()) {
            threadData.lockManager_.Lock(move.node_);
            threadData.affinityTable_.Remove(move.node_);
        }
        return move;
    }

    void PostMoveUpdate(const KlMove &bestMove,
                        ThreadSearchContext &threadData,
                        std::vector<VertexType> &newNodes,
                        std::vector<VertexType> &unlockNodes,
                        [[maybe_unused]] const PreMoveWorkData<VertexWorkWeightT> &prevWorkData) {
        // Collect new neighbor nodes → add to active set
        this->CollectNewNodes(bestMove, threadData, newNodes);

        // Combine neighbor nodes + unlocked nodes (already unlocked by caller)
        newNodes.insert(newNodes.end(), unlockNodes.begin(), unlockNodes.end());

        for (const auto &node : newNodes) {
            threadData.affinityTable_.Insert(node);
        }

        // Recompute ALL and find next best — correct by construction
        SD(threadData).currentBest_ = ComputeAllAffinitiesAndFindBest(threadData);
    }

  public:
    using Base::Base;    // inherit constructors

    void InitializeVariantData() { scanData_.resize(this->threadDataVec_.size()); }
};

}    // namespace osp
