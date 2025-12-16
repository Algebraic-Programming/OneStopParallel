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

#include "kl_improver.hpp"

namespace osp {

template <typename GraphT,
          typename CommCostFunctionT,
          typename MemoryConstraintT = NoLocalSearchMemoryConstraint,
          unsigned WindowSize = 1,
          typename CostT = double>
class KlImproverTest : public KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT> {
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using HeapDatastructure = MaxPairingHeap<VertexType, KlMove>;
    using ActiveSchedule = KlActiveSchedule<GraphT, CostT, MemoryConstraintT>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;
    using NodeSelectionContainer = AdaptiveAffinityTable<GraphT, CostT, ActiveSchedule, WindowSize>;

  public:
    KlImproverTest() : KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>() {
        this->threadDataVec_.resize(1);
        this->threadFinishedVec_.assign(1, true);
    }

    virtual ~KlImproverTest() = default;

    ActiveSchedule &GetActiveSchedule() { return this->activeSchedule_; }

    auto &GetAffinityTable() { return this->threadDataVec_[0].affinityTable_; }

    auto &GetCommCostF() { return this->commCostF_; }

    void SetupSchedule(BspSchedule<GraphT> &schedule) {
        this->threadDataVec_.resize(1);
        this->SetParameters(schedule.GetInstance().GetComputationalDag().NumVertices());
        this->threadDataVec_[0].endStep_ = schedule.NumberOfSupersteps() > 0 ? schedule.NumberOfSupersteps() - 1 : 0;
        this->InitializeDatastructures(schedule);
        this->threadDataVec_[0].activeScheduleData_.InitializeCost(this->activeSchedule_.GetCost());
    }

    void ApplyMoveTest(KlMove move) { this->ApplyMove(move, this->threadDataVec_[0]); }

    auto &GetMaxGainHeap() { return this->threadDataVec_[0].maxGainHeap_; }

    auto GetCurrentCost() { return this->threadDataVec_[0].activeScheduleData_.cost_; }

    bool IsFeasible() { return this->threadDataVec_[0].activeScheduleData_.feasible; }

    void ComputeViolationsTest() { this->activeSchedule_.ComputeViolations(this->threadDataVec_[0].activeScheduleData_); }

    NodeSelectionContainer &InsertGainHeapTest(const std::vector<VertexType> &n) {
        this->threadDataVec_[0].rewardPenaltyStrat_.penalty = 0.0;
        this->threadDataVec_[0].rewardPenaltyStrat_.reward_ = 0.0;

        this->threadDataVec_[0].affinityTable_.Initialize(this->activeSchedule_, n.size());
        for (const auto &node : n) {
            this->threadDataVec_[0].affinityTable_.Insert(node);
        }

        this->InsertGainHeap(this->threadDataVec_[0]);

        return this->threadDataVec_[0].affinityTable_;
    }

    NodeSelectionContainer &InsertGainHeapTestPenalty(const std::vector<VertexType> &n) {
        this->threadDataVec_[0].affinityTable_.Initialize(this->activeSchedule_, n.size());
        for (const auto &node : n) {
            this->threadDataVec_[0].affinityTable_.Insert(node);
        }
        this->threadDataVec_[0].rewardPenaltyStrat_.penalty_ = 5.5;
        this->threadDataVec_[0].rewardPenaltyStrat_.reward_ = 0.0;

        this->InsertGainHeap(this->threadDataVec_[0]);

        return this->threadDataVec_[0].affinityTable_;
    }

    NodeSelectionContainer &InsertGainHeapTestPenaltyReward(const std::vector<VertexType> &n) {
        this->threadDataVec_[0].affinityTable_.Initialize(this->activeSchedule_, n.size());
        for (const auto &node : n) {
            this->threadDataVec_[0].affinityTable_.Insert(node);
        }

        this->threadDataVec_[0].rewardPenaltyStrat_.InitRewardPenalty();
        this->threadDataVec_[0].rewardPenaltyStrat_.reward_ = 15.0;

        this->InsertGainHeap(this->threadDataVec_[0]);

        return this->threadDataVec_[0].affinityTable_;
    }

    void UpdateAffinityTableTest(KlMove bestMove, NodeSelectionContainer &nodeSelection) {
        std::map<VertexType, KlGainUpdateInfo> recomputeMaxGain;
        std::vector<VertexType> newNodes;

        const auto prevWorkData = this->activeSchedule_.GetPreMoveWorkData(bestMove);
        const auto prevCommData = this->commCostF_.GetPreMoveCommData(bestMove);
        this->ApplyMove(bestMove, this->threadDataVec_[0]);

        this->threadDataVec_[0].affinityTable_.Trim();
        this->UpdateAffinities(bestMove, this->threadDataVec_[0], recomputeMaxGain, newNodes, prevWorkData, prevCommData);
    }

    auto RunInnerIterationTest() {
        std::map<VertexType, KlGainUpdateInfo> recomputeMaxGain;
        std::vector<VertexType> newNodes;

        this->PrintHeap(this->threadDataVec_[0].maxGainHeap_);

        KlMove bestMove = this->GetBestMove(
            this->threadDataVec_[0].affinityTable_,
            this->threadDataVec_[0].lockManager_,
            this->threadDataVec_[0].maxGainHeap_);    // locks best_move.node and removes it from node_selection

#ifdef KL_DEBUG
        std::cout << "Best move: " << bestMove.node << " gain: " << bestMove.gain << ", from: " << bestMove.from_step << "|"
                  << bestMove.from_proc << " to: " << bestMove.to_step << "|" << bestMove.toProc << std::endl;
#endif

        const auto prevWorkData = this->activeSchedule_.GetPreMoveWorkData(bestMove);
        const auto prevCommData = this->commCostF_.GetPreMoveCommData(bestMove);
        this->ApplyMove(bestMove, this->threadDataVec_[0]);

        this->threadDataVec_[0].affinityTable_.Trim();
        this->UpdateAffinities(bestMove, this->threadDataVec_[0], recomputeMaxGain, newNodes, prevWorkData, prevCommData);

#ifdef KL_DEBUG
        std::cout << "New nodes: { ";
        for (const auto v : newNodes) {
            std::cout << v << " ";
        }
        std::cout << "}" << std::endl;
#endif

        this->UpdateMaxGain(bestMove, recomputeMaxGain, this->threadDataVec_[0]);
        this->InsertNewNodesGainHeap(newNodes, this->threadDataVec_[0].affinityTable_, this->threadDataVec_[0]);

        return recomputeMaxGain;
    }

    bool IsNodeLocked(VertexType node) const { return this->threadDataVec_[0].lockManager_.IsLocked(node); }

    void GetActiveScheduleTest(BspSchedule<GraphT> &schedule) { this->activeSchedule_.WriteSchedule(schedule); }
};

}    // namespace osp
