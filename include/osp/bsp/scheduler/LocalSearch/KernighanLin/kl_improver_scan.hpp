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

    static constexpr size_t kMaxTieBreakCandidates = 50;

    struct ScanThreadData {
        std::vector<KlMove> topMoves_;    // up to kMaxTieBreakCandidates equal-gain moves
    };

    std::vector<ScanThreadData> scanData_;

    ScanThreadData &SD(ThreadSearchContext &td) { return scanData_[td.threadId_]; }

    // --- Core: recompute all unlocked active nodes, collect top moves ---

    void ComputeAllAffinitiesAndFindBest(ThreadSearchContext &threadData) {
        auto &sd = SD(threadData);
        sd.topMoves_.clear();

        CostT bestGain = std::numeric_limits<CostT>::lowest();

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

            if (bestMove.gain_ > bestGain) {
                bestGain = bestMove.gain_;
                sd.topMoves_.clear();
                sd.topMoves_.push_back(bestMove);
            } else if (bestMove.gain_ == bestGain && sd.topMoves_.size() < kMaxTieBreakCandidates) {
                sd.topMoves_.push_back(bestMove);
            }
        }
    }

    // --- DISPATCH IMPLEMENTATIONS ---

    void ReinitializeMoveFinding(ThreadSearchContext &threadData) { ComputeAllAffinitiesAndFindBest(threadData); }

    KlMove GetBestMove(ThreadSearchContext &threadData) {
        auto &sd = SD(threadData);

        if (sd.topMoves_.empty()) {
            KlMove invalid;
            invalid.gain_ = std::numeric_limits<CostT>::lowest();
            return invalid;
        }

        // Uniform random selection among equal-gain candidates
        std::uniform_int_distribution<size_t> dis(0, sd.topMoves_.size() - 1);
        KlMove move = sd.topMoves_[dis(this->gen_)];

        threadData.lockManager_.Lock(move.node_);
        threadData.affinityTable_.Remove(move.node_);

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
        ComputeAllAffinitiesAndFindBest(threadData);
    }

  public:
    using Base::Base;    // inherit constructors

    void InitializeVariantData() { scanData_.resize(this->threadDataVec_.size()); }
};

}    // namespace osp
