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
          typename MemoryConstraintT = no_local_search_memory_constraint,
          unsigned windowSize = 1,
          typename CostT = double>
class KlImproverTest : public kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t> {
    using VertexType = vertex_idx_t<Graph_t>;
    using kl_move = kl_move_struct<cost_t, VertexType>;
    using heap_datastructure = MaxPairingHeap<VertexType, kl_move>;
    using active_schedule_t = kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>;
    using kl_gain_update_info = kl_update_info<VertexType>;
    using node_selection_container_t = adaptive_affinity_table<Graph_t, cost_t, active_schedule_t, window_size>;

  public:
    KlImproverTest() : kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>() {
        this->thread_data_vec.resize(1);
        this->thread_finished_vec.assign(1, true);
    }

    virtual ~KlImproverTest() = default;

    active_schedule_t &GetActiveSchedule() { return this->active_schedule; }

    auto &GetAffinityTable() { return this->thread_data_vec[0].affinity_table; }

    auto &GetCommCostF() { return this->comm_cost_f; }

    void SetupSchedule(BspSchedule<GraphT> &schedule) {
        this->thread_data_vec.resize(1);
        this->set_parameters(schedule.GetInstance().getComputationalDag().NumVertices());
        this->thread_data_vec[0].end_step = schedule.NumberOfSupersteps() > 0 ? schedule.NumberOfSupersteps() - 1 : 0;
        this->initialize_datastructures(schedule);
        this->thread_data_vec[0].active_schedule_data.initialize_cost(this->active_schedule.get_cost());
    }

    void ApplyMoveTest(kl_move move) { this->apply_move(move, this->thread_data_vec[0]); }

    auto &GetMaxGainHeap() { return this->thread_data_vec[0].max_gain_heap; }

    auto GetCurrentCost() { return this->thread_data_vec[0].active_schedule_data.cost; }

    bool IsFeasible() { return this->thread_data_vec[0].active_schedule_data.feasible; }

    void ComputeViolationsTest() { this->active_schedule.compute_violations(this->thread_data_vec[0].active_schedule_data); }

    node_selection_container_t &InsertGainHeapTest(const std::vector<VertexType> &n) {
        this->thread_data_vec[0].reward_penalty_strat.penalty = 0.0;
        this->thread_data_vec[0].reward_penalty_strat.reward = 0.0;

        this->thread_data_vec[0].affinity_table.initialize(this->active_schedule, n.size());
        for (const auto &node : n) {
            this->thread_data_vec[0].affinity_table.insert(node);
        }

        this->insert_gain_heap(this->thread_data_vec[0]);

        return this->thread_data_vec[0].affinity_table;
    }

    node_selection_container_t &InsertGainHeapTestPenalty(const std::vector<VertexType> &n) {
        this->thread_data_vec[0].affinity_table.initialize(this->active_schedule, n.size());
        for (const auto &node : n) {
            this->thread_data_vec[0].affinity_table.insert(node);
        }
        this->thread_data_vec[0].reward_penalty_strat.penalty = 5.5;
        this->thread_data_vec[0].reward_penalty_strat.reward = 0.0;

        this->insert_gain_heap(this->thread_data_vec[0]);

        return this->thread_data_vec[0].affinity_table;
    }

    node_selection_container_t &InsertGainHeapTestPenaltyReward(const std::vector<VertexType> &n) {
        this->thread_data_vec[0].affinity_table.initialize(this->active_schedule, n.size());
        for (const auto &node : n) {
            this->thread_data_vec[0].affinity_table.insert(node);
        }

        this->thread_data_vec[0].reward_penalty_strat.init_reward_penalty();
        this->thread_data_vec[0].reward_penalty_strat.reward = 15.0;

        this->insert_gain_heap(this->thread_data_vec[0]);

        return this->thread_data_vec[0].affinity_table;
    }

    void UpdateAffinityTableTest(kl_move bestMove, node_selection_container_t &nodeSelection) {
        std::map<VertexType, kl_gain_update_info> recomputeMaxGain;
        std::vector<VertexType> newNodes;

        const auto prevWorkData = this->active_schedule.get_pre_move_work_data(best_move);
        const auto prevCommData = this->comm_cost_f.get_pre_move_comm_data(best_move);
        this->apply_move(best_move, this->thread_data_vec[0]);

        this->thread_data_vec[0].affinity_table.trim();
        this->update_affinities(best_move, this->thread_data_vec[0], recompute_max_gain, new_nodes, prev_work_data, prev_comm_data);
    }

    auto RunInnerIterationTest() {
        std::map<VertexType, kl_gain_update_info> recomputeMaxGain;
        std::vector<VertexType> newNodes;

        this->print_heap(this->thread_data_vec[0].max_gain_heap);

        kl_move bestMove = this->get_best_move(
            this->thread_data_vec[0].affinity_table,
            this->thread_data_vec[0].lock_manager,
            this->thread_data_vec[0].max_gain_heap);    // locks best_move.node and removes it from node_selection

#ifdef KL_DEBUG
        std::cout << "Best move: " << best_move.node << " gain: " << best_move.gain << ", from: " << best_move.from_step << "|"
                  << best_move.from_proc << " to: " << best_move.to_step << "|" << best_move.to_proc << std::endl;
#endif

        const auto prevWorkData = this->active_schedule.get_pre_move_work_data(best_move);
        const auto prevCommData = this->comm_cost_f.get_pre_move_comm_data(best_move);
        this->apply_move(best_move, this->thread_data_vec[0]);

        this->thread_data_vec[0].affinity_table.trim();
        this->update_affinities(best_move, this->thread_data_vec[0], recompute_max_gain, new_nodes, prev_work_data, prev_comm_data);

#ifdef KL_DEBUG
        std::cout << "New nodes: { ";
        for (const auto v : new_nodes) {
            std::cout << v << " ";
        }
        std::cout << "}" << std::endl;
#endif

        this->update_max_gain(best_move, recompute_max_gain, this->thread_data_vec[0]);
        this->insert_new_nodes_gain_heap(new_nodes, this->thread_data_vec[0].affinity_table, this->thread_data_vec[0]);

        return recompute_max_gain;
    }

    bool IsNodeLocked(VertexType node) const { return this->thread_data_vec[0].lock_manager.is_locked(node); }

    void GetActiveScheduleTest(BspSchedule<GraphT> &schedule) { this->active_schedule.write_schedule(schedule); }
};

}    // namespace osp
