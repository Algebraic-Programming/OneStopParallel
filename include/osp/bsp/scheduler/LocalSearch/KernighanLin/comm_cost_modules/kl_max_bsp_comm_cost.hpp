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
#include <array>
#include <vector>

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

    void SwapCommSteps(unsigned step1, unsigned step2) { commDs_.SwapSteps(step1, step2); }

    /// After a step removal (bubble empty step forward from removedStep to endStep),
    /// all nodes that were at step S > removedStep are now at step S-1.
    /// Update lambda entries to match the new step numbering.
    /// Only needed for policies that store step values (Lazy, Buffered).
    void UpdateLambdaAfterStepRemoval(unsigned removedStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            for (auto &nodeEntries : nodeLambdaMap_.nodeLambdaVec_) {
                for (auto &procEntry : nodeEntries) {
                    for (auto &step : procEntry) {
                        if (step > removedStep) {
                            step--;
                        }
                    }
                }
            }
        }
    }

    /// After a step insertion (bubble empty step backward to insertedStep),
    /// all nodes that were at step S >= insertedStep are now at step S+1.
    /// Update lambda entries to match the new step numbering.
    void UpdateLambdaAfterStepInsertion(unsigned insertedStep) {
        if constexpr (std::is_same_v<typename CommPolicy::ValueType, std::vector<unsigned>>) {
            for (auto &nodeEntries : nodeLambdaMap_.nodeLambdaVec_) {
                for (auto &procEntry : nodeEntries) {
                    for (auto &step : procEntry) {
                        if (step >= insertedStep) {
                            step++;
                        }
                    }
                }
            }
        }
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

        // DeltaTracker adapter for CalculateDeltaRemove/CalculateDeltaAdd
        struct DeltaAdapterT {
            decltype(AddDelta) &fn;

            void Add(bool is_recv, unsigned step, unsigned proc, CommWeightT v) { fn(is_recv, step, proc, v); }
        };

        DeltaAdapterT deltaAdapter{AddDelta};

        // Negating adapter for reverting CalculateDeltaAdd
        struct NegDeltaAdapterT {
            decltype(AddDelta) &fn;

            void Add(bool is_recv, unsigned step, unsigned proc, CommWeightT v) { fn(is_recv, step, proc, -v); }
        };

        NegDeltaAdapterT negDeltaAdapter{AddDelta};

        // ========== Phase 1: Remove Node from Current State ==========
        // (Invariant for all candidates)

        // Phase 1 Outgoing: node stops sending to children
        // Use policy-aware step attribution.
        auto node_lambda_entries = commDs_.nodeLambdaMap_.IterateProcEntries(node);

        for (const auto [proc, val] : node_lambda_entries) {
            if (proc != node_proc && CommPolicy::HasEntry(val)) {
                const CommWeightT cost = comm_w_node * instance->SendCosts(node_proc, proc);
                if (cost > 0) {
                    int recvStep = CommPolicy::OutgoingRecvStep(node_step, val);
                    int sendStep = CommPolicy::OutgoingSendStep(node_step, val);
                    if (recvStep >= 0) {
                        AddDelta(true, static_cast<unsigned>(recvStep), proc, -cost);
                    }
                    if (sendStep >= 0) {
                        AddDelta(false, static_cast<unsigned>(sendStep), node_proc, -cost);
                    }
                }
            }
        }

        // Phase 1 Incoming: parents stop sending to node on node_proc
        // Use CalculateDeltaRemove which handles both last-child-removal
        // and min-shift (relevant for Lazy/Buffered).
        for (const auto &u : graph->Parents(node)) {
            const unsigned u_proc = active_schedule->AssignedProcessor(u);
            const unsigned u_step = current_vec_schedule.AssignedSuperstep(u);
            const CommWeightT comm_w_u = graph->VertexCommWeight(u);

            if (u_proc != node_proc) {
                const auto &lambda_val = commDs_.nodeLambdaMap_.GetProcEntry(u, node_proc);
                if (CommPolicy::HasEntry(lambda_val)) {
                    const CommWeightT cost = comm_w_u * instance->SendCosts(u_proc, node_proc);
                    if (cost > 0) {
                        CommPolicy::CalculateDeltaRemove(lambda_val, node_step, u_step, u_proc, node_proc, cost, deltaAdapter);
                    }
                }
            }
        }

        // ========== Phase 2: Add Node to Each Candidate ==========

        // Helper: compute effective val after conceptually removing one instance of node_step.
        // Used for Phase 2A when p_to == node_proc.
        auto ComputeEffectiveVal = [&](const typename CommPolicy::ValueType &val) -> typename CommPolicy::ValueType {
            if constexpr (std::is_same_v<typename CommPolicy::ValueType, unsigned>) {
                // Eager: val is count; decrement
                return val > 0 ? val - 1 : 0;
            } else {
                // Lazy/Buffered: val is vector; remove one instance of node_step
                auto result = val;
                auto it = std::find(result.begin(), result.end(), node_step);
                if (it != result.end()) {
                    result.erase(it);
                }
                return result;
            }
        };

        // Per-parent precomputed data for Phase 2A incoming additions
        struct ParentAddInfo {
            unsigned u_proc;
            unsigned u_step;
            CommWeightT cost;
            typename CommPolicy::ValueType effective_val;
        };

        static thread_local std::vector<ParentAddInfo> parent_add_infos;

        // Per-dest-proc precomputed data for Phase 2B outgoing
        struct OutgoingInfo {
            unsigned v_proc;
            CommWeightT cost;
            int recv_step;    // precomputed, used when !outgoing_recv_at_parent_step
            int send_step;    // precomputed, used when !outgoing_send_at_parent_step
        };

        static thread_local std::vector<OutgoingInfo> outgoing_infos;

        for (const unsigned p_to : proc_range->CompatibleProcessorsVertex(node)) {
            // --- Precompute Phase 2A: parent effective vals ---
            parent_add_infos.clear();
            for (const auto &u : graph->Parents(node)) {
                const unsigned u_proc = active_schedule->AssignedProcessor(u);
                if (u_proc == p_to) {
                    continue;    // same proc → no comm
                }

                const unsigned u_step = current_vec_schedule.AssignedSuperstep(u);
                const CommWeightT comm_w_u = graph->VertexCommWeight(u);
                const CommWeightT cost = comm_w_u * instance->SendCosts(u_proc, p_to);
                if (cost <= 0) {
                    continue;
                }

                const auto &val_on_p_to = commDs_.nodeLambdaMap_.GetProcEntry(u, p_to);
                typename CommPolicy::ValueType effective_val;
                if (p_to == node_proc) {
                    // After Phase 1 removal of node from node_proc
                    effective_val = ComputeEffectiveVal(val_on_p_to);
                } else {
                    effective_val = val_on_p_to;
                }
                parent_add_infos.push_back({u_proc, u_step, cost, std::move(effective_val)});
            }

            // --- Precompute Phase 2B: outgoing (node→children) ---
            outgoing_infos.clear();
            for (const auto [v_proc, val] : commDs_.nodeLambdaMap_.IterateProcEntries(node)) {
                if (v_proc != p_to && CommPolicy::HasEntry(val)) {
                    const CommWeightT cost = comm_w_node * instance->SendCosts(p_to, v_proc);
                    if (cost > 0) {
                        int recvStep = -1;
                        int sendStep = -1;
                        if constexpr (!CommPolicy::outgoing_recv_at_parent_step) {
                            recvStep = CommPolicy::OutgoingRecvStep(0, val);
                        }
                        if constexpr (!CommPolicy::outgoing_send_at_parent_step) {
                            sendStep = CommPolicy::OutgoingSendStep(0, val);
                        }
                        outgoing_infos.push_back({v_proc, cost, recvStep, sendStep});
                    }
                }
            }

            // --- Iterate Window (s_to) ---
            for (unsigned s_to_idx = node_start_idx; s_to_idx < window_bound; ++s_to_idx) {
                unsigned s_to = node_step + s_to_idx - WindowSize;

                // Apply Phase 2A: incoming deltas (policy-aware, s_to-dependent)
                for (const auto &info : parent_add_infos) {
                    CommPolicy::CalculateDeltaAdd(
                        info.effective_val, s_to, info.u_step, info.u_proc, p_to, info.cost, deltaAdapter);
                }

                // Apply Phase 2B: outgoing deltas (policy-aware)
                for (const auto &info : outgoing_infos) {
                    // Recv side
                    if constexpr (CommPolicy::outgoing_recv_at_parent_step) {
                        AddDelta(true, s_to, info.v_proc, info.cost);
                    } else {
                        if (info.recv_step >= 0) {
                            AddDelta(true, static_cast<unsigned>(info.recv_step), info.v_proc, info.cost);
                        }
                    }
                    // Send side
                    if constexpr (CommPolicy::outgoing_send_at_parent_step) {
                        AddDelta(false, s_to, p_to, info.cost);
                    } else {
                        if (info.send_step >= 0) {
                            AddDelta(false, static_cast<unsigned>(info.send_step), p_to, info.cost);
                        }
                    }
                }

                // Evaluate cost change
                CostT total_change = 0;
                for (unsigned step : scratch.active_steps) {
                    if (!scratch.send_deltas[step].dirtyProcs_.empty() || !scratch.recv_deltas[step].dirtyProcs_.empty()) {
                        total_change += CalculateStepCostChange(step, scratch.send_deltas[step], scratch.recv_deltas[step]);
                    }
                }

                affinity_table_node[p_to][s_to_idx] += total_change * instance->CommunicationCosts();

                // Revert Phase 2B: outgoing deltas
                for (const auto &info : outgoing_infos) {
                    if constexpr (CommPolicy::outgoing_recv_at_parent_step) {
                        AddDelta(true, s_to, info.v_proc, -info.cost);
                    } else {
                        if (info.recv_step >= 0) {
                            AddDelta(true, static_cast<unsigned>(info.recv_step), info.v_proc, -info.cost);
                        }
                    }
                    if constexpr (CommPolicy::outgoing_send_at_parent_step) {
                        AddDelta(false, s_to, p_to, -info.cost);
                    } else {
                        if (info.send_step >= 0) {
                            AddDelta(false, static_cast<unsigned>(info.send_step), p_to, -info.cost);
                        }
                    }
                }

                // Revert Phase 2A: incoming deltas
                for (const auto &info : parent_add_infos) {
                    CommPolicy::CalculateDeltaAdd(
                        info.effective_val, s_to, info.u_step, info.u_proc, p_to, info.cost, negDeltaAdapter);
                }
            }
        }
    }

    CommWeightT CalculateStepCostChange(unsigned step,
                                        const FastDeltaTracker<CommWeightT> &delta_send,
                                        const FastDeltaTracker<CommWeightT> &delta_recv) {
        CommWeightT old_max = commDs_.StepMaxComm(step);
        CommWeightT second_max = commDs_.StepSecondMaxComm(step);
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
        // All max-holders reduced: scan non-dirty values for true fallback
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
        const unsigned start_step = thread_data.startStep_;
        const unsigned end_step = thread_data.endStep_;

        for (const auto &target : instance->GetComputationalDag().Children(move.node_)) {
            const unsigned target_step = active_schedule->AssignedSuperstep(target);
            if (target_step < start_step || target_step > end_step) {
                continue;
            }

            if (thread_data.lockManager_.IsLocked(target)) {
                continue;
            }

            if (not thread_data.affinityTable_.IsSelected(target)) {
                new_nodes.push_back(target);
                continue;
            }

            const unsigned target_proc = active_schedule->AssignedProcessor(target);
            const unsigned target_start_idx = StartIdx(target_step, start_step);
            auto &affinity_table = thread_data.affinityTable_.At(target);

            if (move.fromStep_ < target_step + (move.fromProc_ == target_proc)) {
                const unsigned diff = target_step - move.fromStep_;
                const unsigned bound = WindowSize >= diff ? WindowSize - diff + 1 : 0;
                unsigned idx = target_start_idx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(target)) {
                        affinity_table[p][idx] -= penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.fromProc_)) {
                    affinity_table[move.fromProc_][idx - 1] += penalty;
                }

            } else {
                const unsigned diff = move.fromStep_ - target_step;
                const unsigned window_bound = EndIdx(target_step, end_step);
                unsigned idx = std::min(WindowSize + diff, window_bound);

                if (idx < window_bound && IsCompatible(target, move.fromProc_)) {
                    affinity_table[move.fromProc_][idx] += reward;
                }

                idx++;

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(target)) {
                        affinity_table[p][idx] += reward;
                    }
                }
            }

            if (move.toStep_ < target_step + (move.toProc_ == target_proc)) {
                unsigned idx = target_start_idx;
                const unsigned diff = target_step - move.toStep_;
                const unsigned bound = WindowSize >= diff ? WindowSize - diff + 1 : 0;
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(target)) {
                        affinity_table[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && IsCompatible(target, move.toProc_)) {
                    affinity_table[move.toProc_][idx - 1] -= penalty;
                }

            } else {
                const unsigned diff = move.toStep_ - target_step;
                const unsigned window_bound = EndIdx(target_step, end_step);
                unsigned idx = std::min(WindowSize + diff, window_bound);

                if (idx < window_bound && IsCompatible(target, move.toProc_)) {
                    affinity_table[move.toProc_][idx] -= reward;
                }

                idx++;

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(target)) {
                        affinity_table[p][idx] -= reward;
                    }
                }
            }
        }

        for (const auto &source : instance->GetComputationalDag().Parents(move.node_)) {
            const unsigned source_step = active_schedule->AssignedSuperstep(source);
            if (source_step < start_step || source_step > end_step) {
                continue;
            }

            if (thread_data.lockManager_.IsLocked(source)) {
                continue;
            }

            if (not thread_data.affinityTable_.IsSelected(source)) {
                new_nodes.push_back(source);
                continue;
            }

            const unsigned source_proc = active_schedule->AssignedProcessor(source);
            const unsigned source_start_idx = StartIdx(source_step, start_step);
            const unsigned window_bound = EndIdx(source_step, end_step);
            auto &affinity_table_source = thread_data.affinityTable_.At(source);

            if (move.fromStep_ < source_step + (move.fromProc_ != source_proc)) {
                const unsigned diff = source_step - move.fromStep_;
                const unsigned bound = WindowSize > diff ? WindowSize - diff : 0;
                unsigned idx = source_start_idx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(source)) {
                        affinity_table_source[p][idx] += reward;
                    }
                }

                if (WindowSize >= diff && IsCompatible(source, move.fromProc_)) {
                    affinity_table_source[move.fromProc_][idx] += reward;
                }

            } else {
                const unsigned diff = move.fromStep_ - source_step;
                unsigned idx = WindowSize + diff;

                if (idx < window_bound && IsCompatible(source, move.fromProc_)) {
                    affinity_table_source[move.fromProc_][idx] += penalty;
                }

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(source)) {
                        affinity_table_source[p][idx] -= penalty;
                    }
                }
            }

            if (move.toStep_ < source_step + (move.toProc_ != source_proc)) {
                const unsigned diff = source_step - move.toStep_;
                const unsigned bound = WindowSize > diff ? WindowSize - diff : 0;
                unsigned idx = source_start_idx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(source)) {
                        affinity_table_source[p][idx] -= reward;
                    }
                }

                if (WindowSize >= diff && IsCompatible(source, move.toProc_)) {
                    affinity_table_source[move.toProc_][idx] -= reward;
                }

            } else {
                const unsigned diff = move.toStep_ - source_step;
                unsigned idx = WindowSize + diff;

                if (idx < window_bound && IsCompatible(source, move.toProc_)) {
                    affinity_table_source[move.toProc_][idx] -= penalty;
                }
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->CompatibleProcessorsVertex(source)) {
                        affinity_table_source[p][idx] += penalty;
                    }
                }
            }
        }
    }
};

}    // namespace osp
