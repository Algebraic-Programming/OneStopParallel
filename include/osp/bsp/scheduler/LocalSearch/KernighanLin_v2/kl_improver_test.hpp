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
    using ActiveScheduleT = KlActiveSchedule<GraphT, CostT, MemoryConstraintT>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;
    using NodeSelectionContainerT = AdaptiveAffinityTable<GraphT, CostT, ActiveScheduleT, WindowSize>;

  public:
    KlImproverTest() : KlImprover<GraphT, CommCostFunctionT, MemoryConstraintT, WindowSize, CostT>() {
        this->threadDataVec_.resize(1);
        this->threadFinishedVec_.assign(1, true);
    }

    virtual ~KlImproverTest() = default;

    ActiveScheduleT &GetActiveSchedule() { return this->activeSchedule_; }

    auto &GetAffinityTable() { return this->threadDataVec_[0].affinityTable; }

    auto &GetCommCostF() { return this->commCostF_; }

    void SetupSchedule(BspSchedule<GraphT> &schedule) {
        this->threadDataVec_.resize(1);
        this->set_parameters(schedule.getInstance().GetComputationalDag().NumVertices());
        this->threadDataVec_[0].endStep = schedule.numberOfSupersteps() > 0 ? schedule.numberOfSupersteps() - 1 : 0;
        this->initialize_datastructures(schedule);
        this->threadDataVec_[0].activeScheduleData.InitializeCost(this->activeSchedule_.GetCost());
    }

    void ApplyMoveTest(KlMove move) { this->ApplyMove(move, this->threadDataVec_[0]); }

    auto &GetMaxGainHeap() { return this->thread_data_vec[0].max_gain_heap; }

    auto GetCurrentCost() { return this->threadDataVec_[0].activeScheduleData.cost; }

    bool IsFeasible() { return this->threadDataVec_[0].activeScheduleData.feasible; }

    void ComputeViolationsTest() { this->activeSchedule_.compute_violations(this->threadDataVec_[0].activeScheduleData); }

    NodeSelectionContainerT &InsertGainHeapTest(const std::vector<VertexType> &n) {
        this->threadDataVec_[0].rewardPenaltyStrat.penalty = 0.0;
        this->threadDataVec_[0].rewardPenaltyStrat.reward = 0.0;

        this->threadDataVec_[0].affinityTable.Initialize(this->activeSchedule_, n.size());
        for (const auto &node : n) {
            this->threadDataVec_[0].affinityTable.Insert(node);
        }

        this->InsertGainHeap(this->threadDataVec_[0]);

        return this->threadDataVec_[0].affinityTable;
    }

    NodeSelectionContainerT &InsertGainHeapTestPenalty(const std::vector<VertexType> &n) {
        this->threadDataVec_[0].affinityTable.Initialize(this->activeSchedule_, n.size());
        for (const auto &node : n) {
            this->threadDataVec_[0].affinityTable.Insert(node);
        }
        this->threadDataVec_[0].rewardPenaltyStrat.penalty = 5.5;
        this->threadDataVec_[0].rewardPenaltyStrat.reward = 0.0;

        this->InsertGainHeap(this->threadDataVec_[0]);

        return this->threadDataVec_[0].affinityTable;
    }

    NodeSelectionContainerT &InsertGainHeapTestPenaltyReward(const std::vector<VertexType> &n) {
        this->thread_data_vec[0].affinity_table.initialize(this->active_schedule, n.size());
        for (const auto &node : n) {
            this->thread_data_vec[0].affinity_table.insert(node);
        }

        this->thread_data_vec[0].reward_penalty_strat.init_reward_penalty();
        this->thread_data_vec[0].reward_penalty_strat.reward = 15.0;

        this->insert_gain_heap(this->thread_data_vec[0]);

        return this->thread_data_vec[0].affinity_table;
    }

    void UpdateAffinityTableTest(KlMove bestMove, NodeSelectionContainerT &nodeSelection) {
        std::map<VertexType, KlGainUpdateInfo> recomputeMaxGain;
        std::vector<VertexType> newNodes;

        const auto prevWorkData = this->active_schedule.get_pre_move_work_data(bestMove);
        const auto prevCommData = this->comm_cost_f.get_pre_move_comm_data(bestMove);
        this->apply_move(bestMove, this->thread_data_vec[0]);

        this->thread_data_vec[0].affinity_table.trim();
        this->update_affinities(bestMove, this->thread_data_vec[0], recomputeMaxGain, newNodes, prevWorkData, prevCommData);
    }

    auto RunInnerIterationTest() {
        std::map<VertexType, KlGainUpdateInfo> recomputeMaxGain;
        std::vector<VertexType> newNodes;

        this->print_heap(this->threadDataVec_[0].maxGainHeap);

        KlMove bestMove = this->GetBestMove(
            this->threadDataVec_[0].affinityTable,
            this->threadDataVec_[0].lockManager,
            this->threadDataVec_[0].maxGainHeap);    // locks best_move.node and removes it from node_selection

#ifdef KL_DEBUG
        std::cout << "Best move: " << best_move.node << " gain: " << best_move.gain << ", from: " << best_move.from_step << "|"
                  << best_move.from_proc << " to: " << best_move.to_step << "|" << best_move.to_proc << std::endl;
#endif

        const auto prevWorkData = this->activeSchedule_.GetPreMoveWorkData(bestMove);
        const auto prevCommData = this->commCostF_.GetPreMoveCommData(bestMove);
        this->ApplyMove(bestMove, this->threadDataVec_[0]);

        this->threadDataVec_[0].affinityTable.Trim();
        this->UpdateAffinities(bestMove, this->threadDataVec_[0], recomputeMaxGain, newNodes, prevWorkData, prevCommData);

#ifdef KL_DEBUG
        std::cout << "New nodes: { ";
        for (const auto v : new_nodes) {
            std::cout << v << " ";
        }
        std::cout << "}" << std::endl;
#endif

        this->update_max_gain(bestMove, recomputeMaxGain, this->threadDataVec_[0]);
        this->insert_new_nodes_gain_heap(newNodes, this->threadDataVec_[0].affinityTable, this->threadDataVec_[0]);

        return recomputeMaxGain;
    }

    bool IsNodeLocked(VertexType node) const { return this->thread_data_vec[0].lock_manager.is_locked(node); }

    void GetActiveScheduleTest(BspSchedule<GraphT> &schedule) { this->activeSchedule_.WriteSchedule(schedule); }
};

}    // namespace osp
