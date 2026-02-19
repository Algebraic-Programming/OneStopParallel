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
          unsigned windowSize = 1,
          typename CostT = double>
class KlImproverTest : public KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT> {
    using Parent = KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT>;
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using HeapDatastructure = MaxPairingHeap<VertexType, KlMove>;
    using ActiveSchedule = KlActiveSchedule<GraphT, CostT, MemoryConstraintT>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;
    using NodeSelectionContainer = AdaptiveAffinityTable<GraphT, CostT, ActiveSchedule, windowSize>;
    using VertexWorkWeightT = osp::VWorkwT<GraphT>;

  public:
    KlImproverTest() : Parent() {
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

    /// Access the max-gain heap (only available for heap variant / total cost functions)
    auto &GetMaxGainHeap() {
        static_assert(!CommCostFunctionT::isMaxCommCostFunction_,
                      "GetMaxGainHeap() is only available for heap-based (total/totalLambda) cost functions");
        return this->HD(this->threadDataVec_[0]).maxGainHeap_;
    }

    auto GetCurrentCost() { return this->threadDataVec_[0].activeScheduleData_.cost_; }

    bool IsFeasible() { return this->threadDataVec_[0].activeScheduleData_.feasible_; }

    void ComputeViolationsTest() { this->activeSchedule_.ComputeViolations(this->threadDataVec_[0].activeScheduleData_); }

    /// Initialize affinities and move-finding for the given nodes.
    /// For heap variant: computes affinities and populates the heap.
    /// For scan variant: computes affinities and stores the best move.
    NodeSelectionContainer &InitMoveFindingTest(const std::vector<VertexType> &n) {
        this->threadDataVec_[0].rewardPenaltyStrat_.penalty_ = 0.0;
        this->threadDataVec_[0].rewardPenaltyStrat_.reward_ = 0.0;

        this->threadDataVec_[0].affinityTable_.Initialize(this->activeSchedule_, n.size());
        for (const auto &node : n) {
            this->threadDataVec_[0].affinityTable_.Insert(node);
        }

        this->ReinitializeMoveFinding(this->threadDataVec_[0]);

        return this->threadDataVec_[0].affinityTable_;
    }

    NodeSelectionContainer &InitMoveFindingTestPenalty(const std::vector<VertexType> &n) {
        this->threadDataVec_[0].affinityTable_.Initialize(this->activeSchedule_, n.size());
        for (const auto &node : n) {
            this->threadDataVec_[0].affinityTable_.Insert(node);
        }
        this->threadDataVec_[0].rewardPenaltyStrat_.penalty_ = 5.5;
        this->threadDataVec_[0].rewardPenaltyStrat_.reward_ = 0.0;

        this->ReinitializeMoveFinding(this->threadDataVec_[0]);

        return this->threadDataVec_[0].affinityTable_;
    }

    NodeSelectionContainer &InitMoveFindingTestPenaltyReward(const std::vector<VertexType> &n) {
        this->threadDataVec_[0].affinityTable_.Initialize(this->activeSchedule_, n.size());
        for (const auto &node : n) {
            this->threadDataVec_[0].affinityTable_.Insert(node);
        }

        this->threadDataVec_[0].rewardPenaltyStrat_.InitRewardPenalty();
        this->threadDataVec_[0].rewardPenaltyStrat_.reward_ = 15.0;

        this->ReinitializeMoveFinding(this->threadDataVec_[0]);

        return this->threadDataVec_[0].affinityTable_;
    }

    // Backward-compatible aliases for existing test code
    NodeSelectionContainer &InsertGainHeapTest(const std::vector<VertexType> &n) { return InitMoveFindingTest(n); }

    NodeSelectionContainer &InsertGainHeapTestPenalty(const std::vector<VertexType> &n) { return InitMoveFindingTestPenalty(n); }

    NodeSelectionContainer &InsertGainHeapTestPenaltyReward(const std::vector<VertexType> &n) {
        return InitMoveFindingTestPenaltyReward(n);
    }

    void UpdateAffinityTableTest(KlMove bestMove, NodeSelectionContainer &nodeSelection) {
        std::vector<VertexType> newNodes;
        std::vector<VertexType> unlockNodes;

        const auto prevWorkData = this->activeSchedule_.GetPreMoveWorkData(bestMove);
        this->ApplyMove(bestMove, this->threadDataVec_[0]);

        this->threadDataVec_[0].affinityTable_.Trim();
        this->PostMoveUpdate(bestMove, this->threadDataVec_[0], newNodes, unlockNodes, prevWorkData);
    }

    auto RunInnerIterationTest() {
        auto &td = this->threadDataVec_[0];

        // Get best move via dispatch
        KlMove bestMove = this->GetBestMove(td);

        const auto prevWorkData = this->activeSchedule_.GetPreMoveWorkData(bestMove);
        this->ApplyMove(bestMove, td);

        std::vector<VertexType> newNodes;
        std::vector<VertexType> unlockNodes;

        td.affinityTable_.Trim();

        // Unlock first (matches base loop ordering)
        for (const auto v : unlockNodes) {
            td.lockManager_.Unlock(v);
        }

        this->PostMoveUpdate(bestMove, td, newNodes, unlockNodes, prevWorkData);

        return bestMove;
    }

    bool IsNodeLocked(VertexType node) const { return this->threadDataVec_[0].lockManager_.IsLocked(node); }

    void GetActiveScheduleTest(BspSchedule<GraphT> &schedule) { this->activeSchedule_.WriteSchedule(schedule); }

    // Step removal/rollback testing

    bool CheckRemoveSuperstepTest(unsigned step) { return this->CheckRemoveSuperstep(step); }

    bool ScatterNodesSuperstepTest(unsigned step) { return this->ScatterNodesSuperstep(step, this->threadDataVec_[0]); }

    /// Apply a move to the schedule and update cost using a fresh cost computation
    /// instead of relying on the gain_ field.
    void ApplyMoveWithFreshCost(KlMove move) {
        this->activeSchedule_.ApplyMove(move, this->threadDataVec_[0].activeScheduleData_);
        this->commCostF_.UpdateDatastructureAfterMove(move, this->threadDataVec_[0].startStep_, this->threadDataVec_[0].endStep_);
        CostT freshCost = this->commCostF_.ComputeScheduleCostTest();
        CostT changeInCost = freshCost - this->threadDataVec_[0].activeScheduleData_.cost_;
        this->threadDataVec_[0].activeScheduleData_.UpdateCost(changeInCost);
    }

    /// Bubble the empty step at position @p step forward to endStep and
    /// decrement endStep.
    void SwapEmptyStepFwdTest(unsigned step) {
        unsigned oldEndStep = this->threadDataVec_[0].endStep_;
        this->activeSchedule_.SwapEmptyStepFwd(step, oldEndStep);
        for (unsigned i = step; i < oldEndStep; i++) {
            this->commCostF_.SwapCommSteps(i, i + 1);
        }
        this->threadDataVec_[0].endStep_--;
        this->commCostF_.UpdateLambdaAfterStepRemoval(step);
        this->commCostF_.FixupSendRecvAfterStepRemoval(step, oldEndStep);
    }

    /// Push a REMOVE_STEP sentinel into appliedMoves_ after the step has
    /// been physically removed via SwapEmptyStepFwdTest.
    void PushRemoveStepSentinel(unsigned stepToRemove) {
        auto &data = this->threadDataVec_[0].activeScheduleData_;
        CostT syncCost = static_cast<CostT>(this->instance_->SynchronisationCosts());
        data.appliedMoves_.push_back(KlMove::MakeRemoveStep(stepToRemove, syncCost));
    }

    /// Record the sync-cost saving after step removal.
    void UpdateCostAfterRemoval() {
        auto &data = this->threadDataVec_[0].activeScheduleData_;
        data.UpdateCost(static_cast<CostT>(-1.0 * this->instance_->SynchronisationCosts()));
    }

    /// Revert to the best schedule found so far.
    void RevertToBestScheduleTest() {
        this->activeSchedule_.RevertToBestSchedule(this->commCostF_,
                                                   this->threadDataVec_[0].activeScheduleData_,
                                                   this->threadDataVec_[0].startStep_,
                                                   this->threadDataVec_[0].endStep_);
    }

    unsigned GetEndStep() const { return this->threadDataVec_[0].endStep_; }

    unsigned NumSteps() const { return this->threadDataVec_[0].NumSteps(); }

    unsigned GetBestScheduleIdx() const { return this->threadDataVec_[0].activeScheduleData_.bestScheduleIdx_; }

    CostT GetBestCost() const { return this->threadDataVec_[0].activeScheduleData_.bestCost_; }
};

}    // namespace osp
