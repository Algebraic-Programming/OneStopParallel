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

#include <array>

#include "../kl_active_schedule.hpp"
#include "../kl_improver.hpp"
#include "FastDeltaTacker.hpp"
#include "max_comm_datastructure.hpp"

namespace osp {

template <typename GraphT, typename CostT, typename MemoryConstraintT, typename CommPolicy = EagerCommCostPolicy, unsigned WindowSize = 1>
struct KlMaxBspCommCostFunction {
    using VertexType = VertexIdxT<GraphT>;
    using KlMove = KlMoveStruct<CostT, VertexType>;
    using KlGainUpdateInfo = KlUpdateInfo<VertexType>;
    using CommWeightT = VCommwT<GraphT>;

    constexpr static unsigned WindowRange = 2 * WindowSize + 1;
    constexpr static bool isMaxCommCostFunction_ = true;

    KlActiveSchedule<GraphT, CostT, MemoryConstraintT> *active_schedule;
    CompatibleProcessorRange<GraphT> *proc_range;
    const GraphT *graph;
    const BspInstance<GraphT> *instance;

    MaxCommDatastructure<GraphT, CostT, KlActiveSchedule<GraphT, CostT, MemoryConstraintT>, CommPolicy> commDs_;

    inline CostT GetCommMultiplier() { return 1; }

    inline CostT GetMaxCommWeight() { return commDs_.maxCommWeight_; }

    inline CostT GetMaxCommWeightMultiplied() { return commDs_.maxCommWeight_; }

    inline const std::string Name() const { return "max_bsp_comm"; }

    inline bool IsCompatible(VertexType node, unsigned proc) { return active_schedule->GetInstance().IsCompatible(node, proc); }

    inline unsigned StartIdx(const unsigned node_step, const unsigned start_step) {
        return (node_step < WindowSize + start_step) ? WindowSize - (node_step - start_step) : 0U;
    }

    inline unsigned EndIdx(const unsigned node_step, const unsigned end_step) {
        return (node_step + WindowSize <= end_step) ? WindowRange : WindowRange - (node_step + WindowSize - end_step);
    }

    void Initialize(KlActiveSchedule<GraphT, CostT, MemoryConstraintT> &sched, CompatibleProcessorRange<GraphT> &p_range) {
        active_schedule = &sched;
        proc_range = &p_range;
        instance = &sched.GetInstance();
        graph = &instance->GetComputationalDag();

        const unsigned num_steps = active_schedule->NumSteps();
        commDs_.Initialize(*active_schedule);
    }

    using PreMoveCommDataT = PreMoveCommData<CommWeightT>;

    inline PreMoveCommDataT GetPreMoveCommData(const KlMove &move) { return commDs_.GetPreMoveCommData(move); }

    void ComputeSendReceiveDatastructures() { commDs_.ComputeCommDatastructures(0U, active_schedule->NumSteps() - 1U); }

    template <bool compute_datastructures = true>
    CostT ComputeScheduleCost() {
        if constexpr (compute_datastructures) {
            ComputeSendReceiveDatastructures();
        }

        CostT total_cost = active_schedule->GetStepMaxWork(0);
        for (unsigned step = 1U; step < active_schedule->NumSteps(); step++) {
            total_cost += std::max(active_schedule->GetStepMaxWork(step), commDs_.StepMaxComm(step - 1U))
                          * instance->CommunicationCosts();
        }

        if (active_schedule->NumSteps() > 1U) {
            total_cost += static_cast<CostT>(active_schedule->NumSteps() - 1U) * instance->SynchronisationCosts();
        }

        return total_cost;
    }

    CostT ComputeScheduleCostTest() { return ComputeScheduleCost<false>(); }

    void UpdateDatastructureAfterMove(const KlMove &move, const unsigned start_step, const unsigned end_step) {
        commDs_.UpdateDatastructureAfterMove(move, start_step, end_step);
    }

    // Structure to hold thread-local scratchpads to avoid re-allocation.
    struct ScratchData {
        std::vector<FastDeltaTracker<CommWeightT>> send_deltas;    // Size: num_steps
        std::vector<FastDeltaTracker<CommWeightT>> recv_deltas;    // Size: num_steps

        std::vector<unsigned> active_steps;    // List of steps touched in current operation
        std::vector<bool> step_is_active;      // Fast lookup for active steps

        std::vector<std::pair<unsigned, CommWeightT>> child_cost_buffer;

        void init(unsigned n_steps, unsigned n_procs) {
            if (send_deltas.size() < n_steps) {
                send_deltas.resize(n_steps);
                recv_deltas.resize(n_steps);
                step_is_active.resize(n_steps, false);
                active_steps.reserve(n_steps);
            }

            for (auto &tracker : send_deltas) {
                tracker.Initialize(n_procs);
            }
            for (auto &tracker : recv_deltas) {
                tracker.Initialize(n_procs);
            }

            child_cost_buffer.reserve(n_procs);
        }

        void clear_all() {
            for (unsigned step : active_steps) {
                send_deltas[step].Clear();
                recv_deltas[step].Clear();
                step_is_active[step] = false;
            }
            active_steps.clear();
            child_cost_buffer.clear();
        }

        void mark_active(unsigned step) {
            if (!step_is_active[step]) {
                step_is_active[step] = true;
                active_steps.push_back(step);
            }
        }
    };

    template <typename affinity_table_t>
    void ComputeCommAffinity(VertexType node,
                             affinity_table_t &affinity_table_node,
                             const CostT &penalty,
                             const CostT &reward,
                             const unsigned start_step,
                             const unsigned end_step) {
        // Use static thread_local scratchpad to avoid allocation in hot loop
        static thread_local ScratchData scratch;
        scratch.init(active_schedule->NumSteps(), instance->NumberOfProcessors());
        scratch.clear_all();

        const unsigned node_step = active_schedule->AssignedSuperstep(node);
        const unsigned node_proc = active_schedule->AssignedProcessor(node);
        const unsigned window_bound = EndIdx(node_step, end_step);
        const unsigned node_start_idx = StartIdx(node_step, start_step);
        const unsigned staleness = active_schedule->GetStaleness();

        for (const auto &target : instance->GetComputationalDag().Children(node)) {
            const unsigned target_step = active_schedule->AssignedSuperstep(target);
            const unsigned target_proc = active_schedule->AssignedProcessor(target);

            if (target_step < node_step + (target_proc != node_proc ? staleness : 0)) {
                const unsigned diff = node_step - target_step;
                const unsigned bound = WindowSize > diff ? WindowSize - diff : 0;
                unsigned idx = node_start_idx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(node)) {
                        affinity_table_node[p][idx] -= reward;
                    }
                }
                if (WindowSize >= diff && IsCompatible(node, target_proc)) {
                    affinity_table_node[target_proc][idx] -= reward;
                }
            } else {
                const unsigned diff = target_step - node_step;
                unsigned idx = WindowSize + diff;
                if (idx < window_bound && IsCompatible(node, target_proc)) {
                    affinity_table_node[target_proc][idx] -= penalty;
                }
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(node)) {
                        affinity_table_node[p][idx] += penalty;
                    }
                }
            }
        }

        for (const auto &source : instance->GetComputationalDag().Parents(node)) {
            const unsigned source_step = active_schedule->AssignedSuperstep(source);
            const unsigned source_proc = active_schedule->AssignedProcessor(source);

            if (source_step + (source_proc != node_proc ? staleness : 0) <= node_step) {
                const unsigned diff = node_step - source_step;
                const unsigned bound = WindowSize >= diff ? WindowSize - diff + 1 : 0;
                unsigned idx = node_start_idx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(node)) {
                        affinity_table_node[p][idx] += penalty;
                    }
                }
                if (idx - 1 < bound && IsCompatible(node, source_proc)) {
                    affinity_table_node[source_proc][idx - 1] -= penalty;
                }
            } else {
                const unsigned diff = source_step - node_step;
                unsigned idx = std::min(WindowSize + diff, window_bound);
                if (idx < window_bound && IsCompatible(node, source_proc)) {
                    affinity_table_node[source_proc][idx] -= reward;
                }
                idx++;
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(node)) {
                        affinity_table_node[p][idx] -= reward;
                    }
                }
            }
        }

        const CommWeightT comm_w_node = graph->VertexCommWeight(node);
        const auto &current_vec_schedule = active_schedule->GetVectorSchedule();

        auto AddDelta = [&](bool is_recv, unsigned step, unsigned proc, CommWeightT val) {
            if (val == 0) {
                return;
            }
            if (step < active_schedule->NumSteps()) {
                scratch.mark_active(step);
                if (is_recv) {
                    scratch.recv_deltas[step].Add(proc, val);
                } else {
                    scratch.send_deltas[step].Add(proc, val);
                }
            }
        };

        // 1. Remove Node from Current State (Phase 1 - Invariant for all candidates)

        // Outgoing (Children)
        // Child stops receiving from node_proc at node_step
        auto node_lambda_entries = commDs_.nodeLambdaMap_.IterateProcEntries(node);
        CommWeightT total_send_cost_removed = 0;

        for (const auto [proc, val] : node_lambda_entries) {
            if (proc != node_proc) {
                const CommWeightT cost = comm_w_node * instance->SendCosts(node_proc, proc);
                if (cost > 0) {
                    AddDelta(true, node_step, proc, -cost);
                    total_send_cost_removed += cost;
                }
            }
        }
        if (total_send_cost_removed > 0) {
            AddDelta(false, node_step, node_proc, -total_send_cost_removed);
        }

        // Incoming (Parents)
        for (const auto &u : graph->Parents(node)) {
            const unsigned u_proc = active_schedule->AssignedProcessor(u);
            const unsigned u_step = current_vec_schedule.AssignedSuperstep(u);
            const CommWeightT comm_w_u = graph->VertexCommWeight(u);

            if (u_proc != node_proc) {
                const auto &lambda_val = commDs_.nodeLambdaMap_.GetProcEntry(u, node_proc);
                if (CommPolicy::IsSingleEntry(lambda_val)) {
                    const CommWeightT cost = comm_w_u * instance->SendCosts(u_proc, node_proc);
                    if (cost > 0) {
                        AddDelta(true, u_step, node_proc, -cost);
                        AddDelta(false, u_step, u_proc, -cost);
                    }
                }
            }
        }

        // 2. Add Node to Target (Iterate candidates)

        for (const unsigned p_to : proc_range->CompatibleProcessorsVertex(node)) {
            // --- Part A: Incoming Edges (Parents -> p_to) ---
            // These updates are specific to p_to but independent of s_to.
            // We apply them, run the s_to loop, then revert them.

            for (const auto &u : graph->Parents(node)) {
                const unsigned u_proc = active_schedule->AssignedProcessor(u);
                const unsigned u_step = current_vec_schedule.AssignedSuperstep(u);
                const CommWeightT comm_w_u = graph->VertexCommWeight(u);

                if (u_proc != p_to) {
                    const auto &val_on_p_to = commDs_.nodeLambdaMap_.GetProcEntry(u, p_to);
                    // After removing node from node_proc, does u still send to p_to?
                    bool already_sending_to_p_to;
                    if (p_to == node_proc) {
                        already_sending_to_p_to = CommPolicy::HasEntry(val_on_p_to) && !CommPolicy::IsSingleEntry(val_on_p_to);
                    } else {
                        already_sending_to_p_to = CommPolicy::HasEntry(val_on_p_to);
                    }

                    if (!already_sending_to_p_to) {
                        const CommWeightT cost = comm_w_u * instance->SendCosts(u_proc, p_to);
                        if (cost > 0) {
                            AddDelta(true, u_step, p_to, cost);
                            AddDelta(false, u_step, u_proc, cost);
                        }
                    }
                }
            }

            // --- Part B: Outgoing Edges (Node -> Children) ---
            // These depend on which processors children are on.
            scratch.child_cost_buffer.clear();
            CommWeightT total_send_cost_added = 0;

            for (const auto [v_proc, val] : commDs_.nodeLambdaMap_.IterateProcEntries(node)) {
                if (v_proc != p_to) {
                    const CommWeightT cost = comm_w_node * instance->SendCosts(p_to, v_proc);
                    if (cost > 0) {
                        scratch.child_cost_buffer.push_back({v_proc, cost});
                        total_send_cost_added += cost;
                    }
                }
            }

            // Iterate Window (s_to)
            for (unsigned s_to_idx = node_start_idx; s_to_idx < window_bound; ++s_to_idx) {
                unsigned s_to = node_step + s_to_idx - WindowSize;

                // Apply Outgoing Deltas for this specific step s_to
                for (const auto &[v_proc, cost] : scratch.child_cost_buffer) {
                    AddDelta(true, s_to, v_proc, cost);
                }

                if (total_send_cost_added > 0) {
                    AddDelta(false, s_to, p_to, total_send_cost_added);
                }

                CostT total_change = 0;

                // Only check steps that are active (modified in Phase 1, Part A, or Part B)
                for (unsigned step : scratch.active_steps) {
                    // Check if dirty_procs is empty implies no change for this step
                    // FastDeltaTracker ensures dirty_procs is empty if all deltas summed to 0
                    if (!scratch.send_deltas[step].dirtyProcs_.empty() || !scratch.recv_deltas[step].dirtyProcs_.empty()) {
                        total_change += CalculateStepCostChange(step, scratch.send_deltas[step], scratch.recv_deltas[step]);
                    }
                }

                affinity_table_node[p_to][s_to_idx] += total_change * instance->CommunicationCosts();

                // Revert Outgoing Deltas for s_to (Inverse of Apply)
                for (const auto &[v_proc, cost] : scratch.child_cost_buffer) {
                    AddDelta(true, s_to, v_proc, -cost);
                }
                if (total_send_cost_added > 0) {
                    AddDelta(false, s_to, p_to, -total_send_cost_added);
                }
            }

            // Revert Incoming Deltas (Inverse of Part A)
            for (const auto &u : graph->Parents(node)) {
                const unsigned u_proc = active_schedule->AssignedProcessor(u);
                const unsigned u_step = current_vec_schedule.AssignedSuperstep(u);
                const CommWeightT comm_w_u = graph->VertexCommWeight(u);

                if (u_proc != p_to) {
                    const auto &val_on_p_to = commDs_.nodeLambdaMap_.GetProcEntry(u, p_to);
                    bool already_sending_to_p_to;
                    if (p_to == node_proc) {
                        already_sending_to_p_to = CommPolicy::HasEntry(val_on_p_to) && !CommPolicy::IsSingleEntry(val_on_p_to);
                    } else {
                        already_sending_to_p_to = CommPolicy::HasEntry(val_on_p_to);
                    }

                    if (!already_sending_to_p_to) {
                        const CommWeightT cost = comm_w_u * instance->SendCosts(u_proc, p_to);
                        if (cost > 0) {
                            AddDelta(true, u_step, p_to, -cost);
                            AddDelta(false, u_step, u_proc, -cost);
                        }
                    }
                }
            }
        }
    }

    CommWeightT CalculateStepCostChange(unsigned step,
                                        const FastDeltaTracker<CommWeightT> &delta_send,
                                        const FastDeltaTracker<CommWeightT> &delta_recv) {
        CommWeightT old_max = commDs_.StepMaxComm(step);
        unsigned old_max_count = commDs_.StepMaxCommCount(step);

        CommWeightT new_global_max = 0;
        unsigned reduced_max_instances = 0;

        // 1. Check modified sends (Iterate sparse dirty list)
        for (unsigned proc : delta_send.dirtyProcs_) {
            CommWeightT delta = delta_send.Get(proc);
            // delta cannot be 0 here due to FastDeltaTracker invariant

            CommWeightT current_val = commDs_.StepProcSend(step, proc);
            CommWeightT new_val = current_val + delta;

            if (new_val > new_global_max) {
                new_global_max = new_val;
            }
            if (delta < 0 && current_val == old_max) {
                reduced_max_instances++;
            }
        }

        // 2. Check modified receives (Iterate sparse dirty list)
        for (unsigned proc : delta_recv.dirtyProcs_) {
            CommWeightT delta = delta_recv.Get(proc);

            CommWeightT current_val = commDs_.StepProcReceive(step, proc);
            CommWeightT new_val = current_val + delta;

            if (new_val > new_global_max) {
                new_global_max = new_val;
            }
            if (delta < 0 && current_val == old_max) {
                reduced_max_instances++;
            }
        }

        // 3. Determine result
        if (new_global_max > old_max) {
            return new_global_max - old_max;
        }
        if (reduced_max_instances < old_max_count) {
            return 0;
        }

        // All old max-holders were reduced. The second_max from the unmodified
        // state may be stale if dirty procs at second_max were also reduced
        // (happens when a node has outgoing edges to multiple procs, producing
        // 3+ dirty (proc,send/recv) pairs at the same step).
        // Scan non-dirty values for the exact new max.
        CommWeightT max_non_dirty = 0;
        const unsigned num_procs = instance->NumberOfProcessors();
        for (unsigned p = 0; p < num_procs; ++p) {
            if (!delta_send.IsDirty(p)) {
                max_non_dirty = std::max(max_non_dirty, commDs_.StepProcSend(step, p));
            }
            if (!delta_recv.IsDirty(p)) {
                max_non_dirty = std::max(max_non_dirty, commDs_.StepProcReceive(step, p));
            }
        }
        return std::max(new_global_max, max_non_dirty) - old_max;
    }

    template <typename thread_data_t>
    void UpdateNodeCommAffinity(const KlMove &move,
                                thread_data_t &thread_data,
                                const CostT &penalty,
                                const CostT &reward,
                                std::map<VertexType, KlGainUpdateInfo> &,
                                std::vector<VertexType> &new_nodes) {
        // const unsigned start_step = thread_data.start_step;
        // const unsigned end_step = thread_data.end_step;

        // for (const auto &target : instance->getComputationalDag().children(move.node)) {
        //     const unsigned target_step = active_schedule->assigned_superstep(target);
        //     if (target_step < start_step || target_step > end_step) {
        //         continue;
        //     }

        //     if (thread_data.lock_manager.is_locked(target)) {
        //         continue;
        //     }

        //     if (not thread_data.affinity_table.is_selected(target)) {
        //         new_nodes.push_back(target);
        //         continue;
        //     }

        //     const unsigned target_proc = active_schedule->assigned_processor(target);
        //     const unsigned target_start_idx = start_idx(target_step, start_step);
        //     auto &affinity_table = thread_data.affinity_table.at(target);

        //     if (move.from_step < target_step + (move.from_proc == target_proc)) {
        //         const unsigned diff = target_step - move.from_step;
        //         const unsigned bound = WindowSize >= diff ? WindowSize - diff + 1 : 0;
        //         unsigned idx = target_start_idx;
        //         for (; idx < bound; idx++) {
        //             for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
        //                 affinity_table[p][idx] -= penalty;
        //             }
        //         }

        //         if (idx - 1 < bound && is_compatible(target, move.from_proc)) {
        //             affinity_table[move.from_proc][idx - 1] += penalty;
        //         }

        //     } else {
        //         const unsigned diff = move.from_step - target_step;
        //         const unsigned window_bound = end_idx(target_step, end_step);
        //         unsigned idx = std::min(WindowSize + diff, window_bound);

        //         if (idx < window_bound && is_compatible(target, move.from_proc)) {
        //             affinity_table[move.from_proc][idx] += reward;
        //         }

        //         idx++;

        //         for (; idx < window_bound; idx++) {
        //             for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
        //                 affinity_table[p][idx] += reward;
        //             }
        //         }
        //     }

        //     if (move.to_step < target_step + (move.to_proc == target_proc)) {
        //         unsigned idx = target_start_idx;
        //         const unsigned diff = target_step - move.to_step;
        //         const unsigned bound = WindowSize >= diff ? WindowSize - diff + 1 : 0;
        //         for (; idx < bound; idx++) {
        //             for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
        //                 affinity_table[p][idx] += penalty;
        //             }
        //         }

        //         if (idx - 1 < bound && is_compatible(target, move.to_proc)) {
        //             affinity_table[move.to_proc][idx - 1] -= penalty;
        //         }

        //     } else {
        //         const unsigned diff = move.to_step - target_step;
        //         const unsigned window_bound = end_idx(target_step, end_step);
        //         unsigned idx = std::min(WindowSize + diff, window_bound);

        //         if (idx < window_bound && is_compatible(target, move.to_proc)) {
        //             affinity_table[move.to_proc][idx] -= reward;
        //         }

        //         idx++;

        //         for (; idx < window_bound; idx++) {
        //             for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
        //                 affinity_table[p][idx] -= reward;
        //             }
        //         }
        //     }
        // }

        // for (const auto &source : instance->getComputationalDag().parents(move.node)) {
        //     const unsigned source_step = active_schedule->assigned_superstep(source);
        //     if (source_step < start_step || source_step > end_step) {
        //         continue;
        //     }

        //     if (thread_data.lock_manager.is_locked(source)) {
        //         continue;
        //     }

        //     if (not thread_data.affinity_table.is_selected(source)) {
        //         new_nodes.push_back(source);
        //         continue;
        //     }

        //     const unsigned source_proc = active_schedule->assigned_processor(source);
        //     const unsigned source_start_idx = start_idx(source_step, start_step);
        //     const unsigned window_bound = end_idx(source_step, end_step);
        //     auto &affinity_table_source = thread_data.affinity_table.at(source);

        //     if (move.from_step < source_step + (move.from_proc != source_proc)) {
        //         const unsigned diff = source_step - move.from_step;
        //         const unsigned bound = WindowSize > diff ? WindowSize - diff : 0;
        //         unsigned idx = source_start_idx;
        //         for (; idx < bound; idx++) {
        //             for (const unsigned p : proc_range->compatible_processors_vertex(source)) {
        //                 affinity_table_source[p][idx] += reward;
        //             }
        //         }

        //         if (WindowSize >= diff && is_compatible(source, move.from_proc)) {
        //             affinity_table_source[move.from_proc][idx] += reward;
        //         }

        //     } else {
        //         const unsigned diff = move.from_step - source_step;
        //         unsigned idx = WindowSize + diff;

        //         if (idx < window_bound && is_compatible(source, move.from_proc)) {
        //             affinity_table_source[move.from_proc][idx] += penalty;
        //         }

        //         for (; idx < window_bound; idx++) {
        //             for (const unsigned p : proc_range->compatible_processors_vertex(source)) {
        //                 affinity_table_source[p][idx] -= penalty;
        //             }
        //         }
        //     }

        //     if (move.to_step < source_step + (move.to_proc != source_proc)) {
        //         const unsigned diff = source_step - move.to_step;
        //         const unsigned bound = WindowSize > diff ? WindowSize - diff : 0;
        //         unsigned idx = source_start_idx;
        //         for (; idx < bound; idx++) {
        //             for (const unsigned p : proc_range->compatible_processors_vertex(source)) {
        //                 affinity_table_source[p][idx] -= reward;
        //             }
        //         }

        //         if (WindowSize >= diff && is_compatible(source, move.to_proc)) {
        //             affinity_table_source[move.to_proc][idx] -= reward;
        //         }

        //     } else {
        //         const unsigned diff = move.to_step - source_step;
        //         unsigned idx = WindowSize + diff;

        //         if (idx < window_bound && is_compatible(source, move.to_proc)) {
        //             affinity_table_source[move.to_proc][idx] -= penalty;
        //         }
        //         for (; idx < window_bound; idx++) {
        //             for (const unsigned p : proc_range->compatible_processors_vertex(source)) {
        //                 affinity_table_source[p][idx] += penalty;
        //             }
        //         }
        //     }
        // }
    }
};

}    // namespace osp
