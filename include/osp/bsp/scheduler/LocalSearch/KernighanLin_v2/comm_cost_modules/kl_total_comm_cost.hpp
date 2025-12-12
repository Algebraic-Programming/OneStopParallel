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

#include "../kl_active_schedule.hpp"
#include "../kl_improver.hpp"

namespace osp {

template <typename GraphT, typename CostT, typename MemoryConstraintT, unsigned windowSize = 1, bool useNodeCommunicationCostsArg = true>
struct KlTotalCommCostFunction {
    using VertexType = vertex_idx_t<Graph_t>;
    using kl_move = kl_move_struct<cost_t, VertexType>;
    using kl_gain_update_info = kl_update_info<VertexType>;

    constexpr static bool isMaxCommCostFunction_ = false;

    constexpr static unsigned windowRange_ = 2 * windowSize + 1;
    constexpr static bool useNodeCommunicationCosts_ = use_node_communication_costs_arg || not HasEdgeWeightsV<Graph_t>;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *activeSchedule_;

    CompatibleProcessorRange<GraphT> *procRange_;

    const GraphT *graph_;
    const BspInstance<GraphT> *instance_;

    CostT commMultiplier_ = 1;
    CostT maxCommWeight_ = 0;

    inline CostT GetCommMultiplier() { return commMultiplier_; }

    inline CostT GetMaxCommWeight() { return maxCommWeight_; }

    inline CostT GetMaxCommWeightMultiplied() { return maxCommWeight_ * commMultiplier_; }

    const std::string Name() const { return "toal_comm_cost"; }

    inline bool IsCompatible(VertexType node, unsigned proc) { return activeSchedule_->getInstance().isCompatible(node, proc); }

    void Initialize(KlActiveSchedule<GraphT, CostT, MemoryConstraintT> &sched, CompatibleProcessorRange<GraphT> &pRange) {
        activeSchedule_ = &sched;
        procRange_ = &pRange;
        instance_ = &sched.getInstance();
        graph_ = &instance_->getComputationalDag();
        commMultiplier_ = 1.0 / instance_->numberOfProcessors();
    }

    struct EmptyStruct {};

    using PreMoveCommDataT = EmptyStruct;

    inline EmptyStruct GetPreMoveCommData(const kl_move &) { return EmptyStruct(); }

    CostT ComputeScheduleCostTest() { return ComputeScheduleCost(); }

    void UpdateDatastructureAfterMove(const kl_move &, const unsigned, const unsigned) {}

    CostT ComputeScheduleCost() {
        CostT workCosts = 0;
        for (unsigned step = 0; step < activeSchedule_->num_steps(); step++) {
            workCosts += activeSchedule_->get_step_max_work(step);
        }

        CostT commCosts = 0;
        for (const auto &edge : Edges(*graph_)) {
            const auto &sourceV = Source(edge, *graph_);
            const auto &targetV = Traget(edge, *graph_);

            const unsigned &sourceProc = activeSchedule_->assigned_processor(sourceV);
            const unsigned &targetProc = activeSchedule_->assigned_processor(targetV);

            if (sourceProc != targetProc) {
                if constexpr (useNodeCommunicationCosts_) {
                    const CostT sourceCommCost = graph_->VertexCommWeight(sourceV);
                    maxCommWeight_ = std::max(maxCommWeight_, sourceCommCost);
                    commCosts += sourceCommCost * instance_->communicationCosts(sourceProc, targetProc);
                } else {
                    const CostT sourceCommCost = graph_->EdgeCommWeight(edge);
                    maxCommWeight_ = std::max(maxCommWeight_, sourceCommCost);
                    commCosts += sourceCommCost * instance_->communicationCosts(sourceProc, targetProc);
                }
            }
        }

        return workCosts + commCosts * commMultiplier_
               + static_cast<v_commw_t<Graph_t>>(activeSchedule_->num_steps() - 1) * instance_->synchronisationCosts();
    }

    template <typename ThreadDataT>
    void UpdateNodeCommAffinity(const kl_move &move,
                                ThreadDataT &threadData,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, kl_gain_update_info> &maxGainRecompute,
                                std::vector<VertexType> &newNodes) {
        const unsigned &startStep = threadData.start_step;
        const unsigned &endStep = threadData.end_step;

        for (const auto &target : instance->getComputationalDag().children(move.node)) {
            const unsigned target_step = active_schedule->assigned_superstep(target);
            if (target_step < start_step || target_step > end_step) {
                continue;
            }

            if (thread_data.lock_manager.is_locked(target)) {
                continue;
            }

            if (not thread_data.affinity_table.is_selected(target)) {
                new_nodes.push_back(target);
                continue;
            }

            if (max_gain_recompute.find(target) != max_gain_recompute.end()) {
                max_gain_recompute[target].full_update = true;
            } else {
                max_gain_recompute[target] = kl_gain_update_info(target, true);
            }

            const unsigned target_proc = active_schedule->assigned_processor(target);
            const unsigned target_start_idx = start_idx(target_step, start_step);
            auto &affinity_table_target = thread_data.affinity_table.at(target);

            if (move.from_step < target_step + (move.from_proc == target_proc)) {
                const unsigned diff = target_step - move.from_step;
                const unsigned bound = window_size >= diff ? window_size - diff + 1 : 0;
                unsigned idx = target_start_idx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                        affinity_table_target[p][idx] -= penalty;
                    }
                }

                if (idx - 1 < bound && is_compatible(target, move.from_proc)) {
                    affinity_table_target[move.from_proc][idx - 1] += penalty;
                }

            } else {
                const unsigned diff = move.from_step - target_step;
                const unsigned window_bound = end_idx(target_step, end_step);
                unsigned idx = std::min(window_size + diff, window_bound);

                if (idx < window_bound && is_compatible(target, move.from_proc)) {
                    affinity_table_target[move.from_proc][idx] += reward;
                }

                idx++;

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                        affinity_table_target[p][idx] += reward;
                    }
                }
            }

            if (move.to_step < target_step + (move.to_proc == target_proc)) {
                unsigned idx = target_start_idx;
                const unsigned diff = target_step - move.to_step;
                const unsigned bound = window_size >= diff ? window_size - diff + 1 : 0;
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                        affinity_table_target[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && is_compatible(target, move.to_proc)) {
                    affinity_table_target[move.to_proc][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = move.to_step - target_step;
                const unsigned window_bound = end_idx(target_step, end_step);
                unsigned idx = std::min(window_size + diff, window_bound);

                if (idx < window_bound && is_compatible(target, move.to_proc)) {
                    affinity_table_target[move.to_proc][idx] -= reward;
                }

                idx++;

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                        affinity_table_target[p][idx] -= reward;
                    }
                }
            }

            if (move.to_proc != move.from_proc) {
                const auto from_proc_target_comm_cost = instance->communicationCosts(move.from_proc, target_proc);
                const auto to_proc_target_comm_cost = instance->communicationCosts(move.to_proc, target_proc);

                const cost_t comm_gain = graph->VertexCommWeight(move.node) * comm_multiplier;

                unsigned idx = target_start_idx;
                const unsigned window_bound = end_idx(target_step, end_step);
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                        const auto x = change_comm_cost(
                            instance->communicationCosts(p, move.to_proc), to_proc_target_comm_cost, comm_gain);
                        const auto y = change_comm_cost(
                            instance->communicationCosts(p, move.from_proc), from_proc_target_comm_cost, comm_gain);
                        affinity_table_target[p][idx] += x - y;
                    }
                }
            }
        }

        for (const auto &source : instance->getComputationalDag().parents(move.node)) {
            const unsigned source_step = active_schedule->assigned_superstep(source);
            if (source_step < start_step || source_step > end_step) {
                continue;
            }

            if (thread_data.lock_manager.is_locked(source)) {
                continue;
            }

            if (not thread_data.affinity_table.is_selected(source)) {
                new_nodes.push_back(source);
                continue;
            }

            if (max_gain_recompute.find(source) != max_gain_recompute.end()) {
                max_gain_recompute[source].full_update = true;
            } else {
                max_gain_recompute[source] = kl_gain_update_info(source, true);
            }

            const unsigned source_proc = active_schedule->assigned_processor(source);
            const unsigned window_bound = end_idx(source_step, end_step);
            auto &affinity_table_source = thread_data.affinity_table.at(source);

            if (move.from_step < source_step + (move.from_proc != source_proc)) {
                const unsigned diff = source_step - move.from_step;
                const unsigned bound = window_size > diff ? window_size - diff : 0;
                unsigned idx = start_idx(source_step, start_step);
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) {
                        affinity_table_source[p][idx] += reward;
                    }
                }

                if (window_size >= diff && is_compatible(source, move.from_proc)) {
                    affinity_table_source[move.from_proc][idx] += reward;
                }

            } else {
                const unsigned diff = move.from_step - source_step;
                unsigned idx = window_size + diff;

                if (idx < window_bound && is_compatible(source, move.from_proc)) {
                    affinity_table_source[move.from_proc][idx] += penalty;
                }

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) {
                        affinity_table_source[p][idx] -= penalty;
                    }
                }
            }

            if (move.to_step < source_step + (move.to_proc != source_proc)) {
                const unsigned diff = source_step - move.to_step;
                const unsigned bound = window_size > diff ? window_size - diff : 0;
                unsigned idx = start_idx(source_step, start_step);
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) {
                        affinity_table_source[p][idx] -= reward;
                    }
                }

                if (window_size >= diff && is_compatible(source, move.to_proc)) {
                    affinity_table_source[move.to_proc][idx] -= reward;
                }

            } else {
                const unsigned diff = move.to_step - source_step;
                unsigned idx = window_size + diff;

                if (idx < window_bound && is_compatible(source, move.to_proc)) {
                    affinity_table_source[move.to_proc][idx] -= penalty;
                }
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) {
                        affinity_table_source[p][idx] += penalty;
                    }
                }
            }

            if (move.to_proc != move.from_proc) {
                const auto from_proc_source_comm_cost = instance->communicationCosts(source_proc, move.from_proc);
                const auto to_proc_source_comm_cost = instance->communicationCosts(source_proc, move.to_proc);

                const cost_t comm_gain = graph->VertexCommWeight(source) * comm_multiplier;

                unsigned idx = start_idx(source_step, start_step);
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) {
                        const cost_t x = change_comm_cost(
                            instance->communicationCosts(p, move.to_proc), to_proc_source_comm_cost, comm_gain);
                        const cost_t y = change_comm_cost(
                            instance->communicationCosts(p, move.from_proc), from_proc_source_comm_cost, comm_gain);
                        affinity_table_source[p][idx] += x - y;
                    }
                }
            }
        }
    }

    inline unsigned StartIdx(const unsigned nodeStep, const unsigned startStep) {
        return (nodeStep < windowSize + startStep) ? windowSize - (nodeStep - startStep) : 0;
    }

    inline unsigned EndIdx(const unsigned nodeStep, const unsigned endStep) {
        return (nodeStep + windowSize <= endStep) ? windowRange_ : windowRange_ - (nodeStep + windowSize - endStep);
    }

    inline CostT ChangeCommCost(const v_commw_t<Graph_t> &pTargetCommCost,
                                const v_commw_t<Graph_t> &nodeTargetCommCost,
                                const CostT &commGain) {
        return p_target_comm_cost > node_target_comm_cost ? (pTargetCommCost - node_target_comm_cost) * commGain
                                                          : (nodeTargetCommCost - p_target_comm_cost) * commGain * -1.0;
    }

    template <typename AffinityTableT>
    void ComputeCommAffinity(VertexType node,
                             AffinityTableT &affinityTableNode,
                             const CostT &penalty,
                             const CostT &reward,
                             const unsigned startStep,
                             const unsigned endStep) {
        const unsigned nodeStep = activeSchedule_->assigned_superstep(node);
        const unsigned nodeProc = activeSchedule_->assigned_processor(node);
        const unsigned windowBound = EndIdx(nodeStep, endStep);
        const unsigned nodeStartIdx = StartIdx(nodeStep, startStep);

        for (const auto &target : instance->getComputationalDag().children(node)) {
            const unsigned target_step = active_schedule->assigned_superstep(target);
            const unsigned target_proc = active_schedule->assigned_processor(target);

            if (target_step < node_step + (target_proc != node_proc)) {
                const unsigned diff = node_step - target_step;
                const unsigned bound = window_size > diff ? window_size - diff : 0;
                unsigned idx = node_start_idx;

                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(node)) {
                        affinity_table_node[p][idx] -= reward;
                    }
                }

                if (window_size >= diff && is_compatible(node, target_proc)) {
                    affinity_table_node[target_proc][idx] -= reward;
                }

            } else {
                const unsigned diff = target_step - node_step;
                unsigned idx = window_size + diff;

                if (idx < window_bound && is_compatible(node, target_proc)) {
                    affinity_table_node[target_proc][idx] -= penalty;
                }

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(node)) {
                        affinity_table_node[p][idx] += penalty;
                    }
                }
            }

            const cost_t comm_gain = graph->VertexCommWeight(node) * comm_multiplier;
            const auto node_target_comm_cost = instance->communicationCosts(node_proc, target_proc);

            for (const unsigned p : proc_range->compatible_processors_vertex(node)) {
                const cost_t comm_cost
                    = change_comm_cost(instance->communicationCosts(p, target_proc), node_target_comm_cost, comm_gain);
                for (unsigned idx = node_start_idx; idx < window_bound; idx++) {
                    affinity_table_node[p][idx] += comm_cost;
                }
            }

        }    // traget

        for (const auto &source : instance->getComputationalDag().parents(node)) {
            const unsigned source_step = active_schedule->assigned_superstep(source);
            const unsigned source_proc = active_schedule->assigned_processor(source);

            if (source_step < node_step + (source_proc == node_proc)) {
                const unsigned diff = node_step - source_step;
                const unsigned bound = window_size >= diff ? window_size - diff + 1 : 0;
                unsigned idx = node_start_idx;

                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(node)) {
                        affinity_table_node[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && is_compatible(node, source_proc)) {
                    affinity_table_node[source_proc][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = source_step - node_step;
                unsigned idx = std::min(window_size + diff, window_bound);

                if (idx < window_bound && is_compatible(node, source_proc)) {
                    affinity_table_node[source_proc][idx] -= reward;
                }

                idx++;

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(node)) {
                        affinity_table_node[p][idx] -= reward;
                    }
                }
            }

            const cost_t comm_gain = graph->VertexCommWeight(source) * comm_multiplier;
            const auto source_node_comm_cost = instance->communicationCosts(source_proc, node_proc);

            for (const unsigned p : proc_range->compatible_processors_vertex(node)) {
                const cost_t comm_cost
                    = change_comm_cost(instance->communicationCosts(p, source_proc), source_node_comm_cost, comm_gain);
                for (unsigned idx = node_start_idx; idx < window_bound; idx++) {
                    affinity_table_node[p][idx] += comm_cost;
                }
            }
        }    // source
    }
};

}    // namespace osp
