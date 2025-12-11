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
#include "max_comm_datastructure.hpp"

namespace osp {

// A lightweight helper to track deltas without hash maps or repeated allocations.
// Uses a dense vector for O(1) lookups and a sparse list for fast iteration/clearing.
template <typename comm_weight_t>
struct FastDeltaTracker {
    std::vector<comm_weight_t> dense_vals;     // Size: num_procs
    std::vector<unsigned> dirty_procs;         // List of modified indices
    std::vector<unsigned> proc_dirty_index;    // Map proc -> index in dirty_procs (num_procs if not dirty)
    unsigned num_procs = 0;

    void initialize(unsigned n_procs) {
        if (n_procs > num_procs) {
            num_procs = n_procs;
            dense_vals.resize(num_procs, 0);
            dirty_procs.reserve(num_procs);
            proc_dirty_index.resize(num_procs, num_procs);
        }
    }

    inline void add(unsigned proc, comm_weight_t val) {
        if (val == 0) { return; }

        // If currently 0, it is becoming dirty
        if (dense_vals[proc] == 0) {
            proc_dirty_index[proc] = static_cast<unsigned>(dirty_procs.size());
            dirty_procs.push_back(proc);
        }

        dense_vals[proc] += val;

        // If it returns to 0, remove it from dirty list (Swap and Pop for O(1))
        if (dense_vals[proc] == 0) {
            unsigned idx = proc_dirty_index[proc];
            unsigned last_proc = dirty_procs.back();

            // Move last element to the hole
            dirty_procs[idx] = last_proc;
            proc_dirty_index[last_proc] = idx;

            // Remove last
            dirty_procs.pop_back();
            proc_dirty_index[proc] = num_procs;
        }
    }

    inline comm_weight_t get(unsigned proc) const {
        if (proc < dense_vals.size()) { return dense_vals[proc]; }
        return 0;
    }

    inline void clear() {
        for (unsigned p : dirty_procs) {
            dense_vals[p] = 0;
            proc_dirty_index[p] = num_procs;
        }
        dirty_procs.clear();
    }
};

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t, unsigned window_size = 1>
struct kl_bsp_comm_cost_function {
    using VertexType = vertex_idx_t<Graph_t>;
    using kl_move = kl_move_struct<cost_t, VertexType>;
    using kl_gain_update_info = kl_update_info<VertexType>;
    using comm_weight_t = v_commw_t<Graph_t>;

    constexpr static unsigned window_range = 2 * window_size + 1;
    constexpr static bool is_max_comm_cost_function = true;

    kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t> *active_schedule;
    CompatibleProcessorRange<Graph_t> *proc_range;
    const Graph_t *graph;
    const BspInstance<Graph_t> *instance;

    max_comm_datastructure<Graph_t, cost_t, kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>> comm_ds;

    inline cost_t get_comm_multiplier() { return 1; }

    inline cost_t get_max_comm_weight() { return comm_ds.max_comm_weight; }

    inline cost_t get_max_comm_weight_multiplied() { return comm_ds.max_comm_weight; }

    inline const std::string name() const { return "bsp_comm"; }

    inline bool is_compatible(VertexType node, unsigned proc) { return active_schedule->getInstance().isCompatible(node, proc); }

    inline unsigned start_idx(const unsigned node_step, const unsigned start_step) {
        return (node_step < window_size + start_step) ? window_size - (node_step - start_step) : 0;
    }

    inline unsigned end_idx(const unsigned node_step, const unsigned end_step) {
        return (node_step + window_size <= end_step) ? window_range : window_range - (node_step + window_size - end_step);
    }

    void initialize(kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t> &sched, CompatibleProcessorRange<Graph_t> &p_range) {
        active_schedule = &sched;
        proc_range = &p_range;
        instance = &sched.getInstance();
        graph = &instance->getComputationalDag();

        const unsigned num_steps = active_schedule->num_steps();
        comm_ds.initialize(*active_schedule);
    }

    using pre_move_comm_data_t = pre_move_comm_data<comm_weight_t>;

    inline pre_move_comm_data<comm_weight_t> get_pre_move_comm_data(const kl_move &move) {
        return comm_ds.get_pre_move_comm_data(move);
    }

    void compute_send_receive_datastructures() { comm_ds.compute_comm_datastructures(0, active_schedule->num_steps() - 1); }

    template <bool compute_datastructures = true>
    cost_t compute_schedule_cost() {
        if constexpr (compute_datastructures) { compute_send_receive_datastructures(); }

        cost_t total_cost = 0;
        for (unsigned step = 0; step < active_schedule->num_steps(); step++) {
            total_cost += active_schedule->get_step_max_work(step);
            total_cost += comm_ds.step_max_comm(step) * instance->communicationCosts();
        }

        if (active_schedule->num_steps() > 1) {
            total_cost += static_cast<cost_t>(active_schedule->num_steps() - 1) * instance->synchronisationCosts();
        }

        return total_cost;
    }

    cost_t compute_schedule_cost_test() { return compute_schedule_cost<false>(); }

    void update_datastructure_after_move(const kl_move &move, const unsigned start_step, const unsigned end_step) {
        comm_ds.update_datastructure_after_move(move, start_step, end_step);
    }

    // Structure to hold thread-local scratchpads to avoid re-allocation.
    struct ScratchData {
        std::vector<FastDeltaTracker<comm_weight_t>> send_deltas;    // Size: num_steps
        std::vector<FastDeltaTracker<comm_weight_t>> recv_deltas;    // Size: num_steps

        std::vector<unsigned> active_steps;    // List of steps touched in current operation
        std::vector<bool> step_is_active;      // Fast lookup for active steps

        std::vector<std::pair<unsigned, comm_weight_t>> child_cost_buffer;

        void init(unsigned n_steps, unsigned n_procs) {
            if (send_deltas.size() < n_steps) {
                send_deltas.resize(n_steps);
                recv_deltas.resize(n_steps);
                step_is_active.resize(n_steps, false);
                active_steps.reserve(n_steps);
            }

            for (auto &tracker : send_deltas) { tracker.initialize(n_procs); }
            for (auto &tracker : recv_deltas) { tracker.initialize(n_procs); }

            child_cost_buffer.reserve(n_procs);
        }

        void clear_all() {
            for (unsigned step : active_steps) {
                send_deltas[step].clear();
                recv_deltas[step].clear();
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
    void compute_comm_affinity(VertexType node,
                               affinity_table_t &affinity_table_node,
                               const cost_t &penalty,
                               const cost_t &reward,
                               const unsigned start_step,
                               const unsigned end_step) {
        // Use static thread_local scratchpad to avoid allocation in hot loop
        static thread_local ScratchData scratch;
        scratch.init(active_schedule->num_steps(), instance->numberOfProcessors());
        scratch.clear_all();

        const unsigned node_step = active_schedule->assigned_superstep(node);
        const unsigned node_proc = active_schedule->assigned_processor(node);
        const unsigned window_bound = end_idx(node_step, end_step);
        const unsigned node_start_idx = start_idx(node_step, start_step);

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
                if (window_size >= diff && is_compatible(node, target_proc)) { affinity_table_node[target_proc][idx] -= reward; }
            } else {
                const unsigned diff = target_step - node_step;
                unsigned idx = window_size + diff;
                if (idx < window_bound && is_compatible(node, target_proc)) { affinity_table_node[target_proc][idx] -= penalty; }
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(node)) {
                        affinity_table_node[p][idx] += penalty;
                    }
                }
            }
        }

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
                if (idx - 1 < bound && is_compatible(node, source_proc)) { affinity_table_node[source_proc][idx - 1] -= penalty; }
            } else {
                const unsigned diff = source_step - node_step;
                unsigned idx = std::min(window_size + diff, window_bound);
                if (idx < window_bound && is_compatible(node, source_proc)) { affinity_table_node[source_proc][idx] -= reward; }
                idx++;
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(node)) {
                        affinity_table_node[p][idx] -= reward;
                    }
                }
            }
        }

        const comm_weight_t comm_w_node = graph->vertex_comm_weight(node);
        const auto &current_vec_schedule = active_schedule->getVectorSchedule();

        auto add_delta = [&](bool is_recv, unsigned step, unsigned proc, comm_weight_t val) {
            if (val == 0) { return; }
            if (step < active_schedule->num_steps()) {
                scratch.mark_active(step);
                if (is_recv) {
                    scratch.recv_deltas[step].add(proc, val);
                } else {
                    scratch.send_deltas[step].add(proc, val);
                }
            }
        };

        // 1. Remove Node from Current State (Phase 1 - Invariant for all candidates)

        // Outgoing (Children)
        // Child stops receiving from node_proc at node_step
        auto node_lambda_entries = comm_ds.node_lambda_map.iterate_proc_entries(node);
        comm_weight_t total_send_cost_removed = 0;

        for (const auto [proc, count] : node_lambda_entries) {
            if (proc != node_proc) {
                const comm_weight_t cost = comm_w_node * instance->sendCosts(node_proc, proc);
                if (cost > 0) {
                    add_delta(true, node_step, proc, -cost);
                    total_send_cost_removed += cost;
                }
            }
        }
        if (total_send_cost_removed > 0) { add_delta(false, node_step, node_proc, -total_send_cost_removed); }

        // Incoming (Parents)
        for (const auto &u : graph->parents(node)) {
            const unsigned u_proc = active_schedule->assigned_processor(u);
            const unsigned u_step = current_vec_schedule.assignedSuperstep(u);
            const comm_weight_t comm_w_u = graph->vertex_comm_weight(u);

            if (u_proc != node_proc) {
                if (comm_ds.node_lambda_map.get_proc_entry(u, node_proc) == 1) {
                    const comm_weight_t cost = comm_w_u * instance->sendCosts(u_proc, node_proc);
                    if (cost > 0) {
                        add_delta(true, u_step, node_proc, -cost);
                        add_delta(false, u_step, u_proc, -cost);
                    }
                }
            }
        }

        // 2. Add Node to Target (Iterate candidates)

        for (const unsigned p_to : proc_range->compatible_processors_vertex(node)) {
            // --- Part A: Incoming Edges (Parents -> p_to) ---
            // These updates are specific to p_to but independent of s_to.
            // We apply them, run the s_to loop, then revert them.

            for (const auto &u : graph->parents(node)) {
                const unsigned u_proc = active_schedule->assigned_processor(u);
                const unsigned u_step = current_vec_schedule.assignedSuperstep(u);
                const comm_weight_t comm_w_u = graph->vertex_comm_weight(u);

                if (u_proc != p_to) {
                    bool already_sending_to_p_to = false;
                    unsigned count_on_p_to = comm_ds.node_lambda_map.get_proc_entry(u, p_to);

                    if (p_to == node_proc) {
                        if (count_on_p_to > 0) { count_on_p_to--; }
                    }

                    if (count_on_p_to > 0) { already_sending_to_p_to = true; }

                    if (!already_sending_to_p_to) {
                        const comm_weight_t cost = comm_w_u * instance->sendCosts(u_proc, p_to);
                        if (cost > 0) {
                            add_delta(true, u_step, p_to, cost);
                            add_delta(false, u_step, u_proc, cost);
                        }
                    }
                }
            }

            // --- Part B: Outgoing Edges (Node -> Children) ---
            // These depend on which processors children are on.
            scratch.child_cost_buffer.clear();
            comm_weight_t total_send_cost_added = 0;

            for (const auto [v_proc, count] : comm_ds.node_lambda_map.iterate_proc_entries(node)) {
                if (v_proc != p_to) {
                    const comm_weight_t cost = comm_w_node * instance->sendCosts(p_to, v_proc);
                    if (cost > 0) {
                        scratch.child_cost_buffer.push_back({v_proc, cost});
                        total_send_cost_added += cost;
                    }
                }
            }

            // Iterate Window (s_to)
            for (unsigned s_to_idx = node_start_idx; s_to_idx < window_bound; ++s_to_idx) {
                unsigned s_to = node_step + s_to_idx - window_size;

                // Apply Outgoing Deltas for this specific step s_to
                for (const auto &[v_proc, cost] : scratch.child_cost_buffer) { add_delta(true, s_to, v_proc, cost); }

                if (total_send_cost_added > 0) { add_delta(false, s_to, p_to, total_send_cost_added); }

                cost_t total_change = 0;

                // Only check steps that are active (modified in Phase 1, Part A, or Part B)
                for (unsigned step : scratch.active_steps) {
                    // Check if dirty_procs is empty implies no change for this step
                    // FastDeltaTracker ensures dirty_procs is empty if all deltas summed to 0
                    if (!scratch.send_deltas[step].dirty_procs.empty() || !scratch.recv_deltas[step].dirty_procs.empty()) {
                        total_change += calculate_step_cost_change(step, scratch.send_deltas[step], scratch.recv_deltas[step]);
                    }
                }

                affinity_table_node[p_to][s_to_idx] += total_change * instance->communicationCosts();

                // Revert Outgoing Deltas for s_to (Inverse of Apply)
                for (const auto &[v_proc, cost] : scratch.child_cost_buffer) { add_delta(true, s_to, v_proc, -cost); }
                if (total_send_cost_added > 0) { add_delta(false, s_to, p_to, -total_send_cost_added); }
            }

            // Revert Incoming Deltas (Inverse of Part A)
            for (const auto &u : graph->parents(node)) {
                const unsigned u_proc = active_schedule->assigned_processor(u);
                const unsigned u_step = current_vec_schedule.assignedSuperstep(u);
                const comm_weight_t comm_w_u = graph->vertex_comm_weight(u);

                if (u_proc != p_to) {
                    bool already_sending_to_p_to = false;
                    unsigned count_on_p_to = comm_ds.node_lambda_map.get_proc_entry(u, p_to);
                    if (p_to == node_proc) {
                        if (count_on_p_to > 0) { count_on_p_to--; }
                    }
                    if (count_on_p_to > 0) { already_sending_to_p_to = true; }

                    if (!already_sending_to_p_to) {
                        const comm_weight_t cost = comm_w_u * instance->sendCosts(u_proc, p_to);
                        if (cost > 0) {
                            add_delta(true, u_step, p_to, -cost);
                            add_delta(false, u_step, u_proc, -cost);
                        }
                    }
                }
            }
        }
    }

    comm_weight_t calculate_step_cost_change(unsigned step,
                                             const FastDeltaTracker<comm_weight_t> &delta_send,
                                             const FastDeltaTracker<comm_weight_t> &delta_recv) {
        comm_weight_t old_max = comm_ds.step_max_comm(step);
        comm_weight_t second_max = comm_ds.step_second_max_comm(step);
        unsigned old_max_count = comm_ds.step_max_comm_count(step);

        comm_weight_t new_global_max = 0;
        unsigned reduced_max_instances = 0;

        // 1. Check modified sends (Iterate sparse dirty list)
        for (unsigned proc : delta_send.dirty_procs) {
            comm_weight_t delta = delta_send.get(proc);
            // delta cannot be 0 here due to FastDeltaTracker invariant

            comm_weight_t current_val = comm_ds.step_proc_send(step, proc);
            comm_weight_t new_val = current_val + delta;

            if (new_val > new_global_max) { new_global_max = new_val; }
            if (delta < 0 && current_val == old_max) { reduced_max_instances++; }
        }

        // 2. Check modified receives (Iterate sparse dirty list)
        for (unsigned proc : delta_recv.dirty_procs) {
            comm_weight_t delta = delta_recv.get(proc);

            comm_weight_t current_val = comm_ds.step_proc_receive(step, proc);
            comm_weight_t new_val = current_val + delta;

            if (new_val > new_global_max) { new_global_max = new_val; }
            if (delta < 0 && current_val == old_max) { reduced_max_instances++; }
        }

        // 3. Determine result
        if (new_global_max > old_max) { return new_global_max - old_max; }
        if (reduced_max_instances < old_max_count) { return 0; }
        return std::max(new_global_max, second_max) - old_max;
    }

    template <typename thread_data_t>
    void update_node_comm_affinity(const kl_move &move,
                                   thread_data_t &thread_data,
                                   const cost_t &penalty,
                                   const cost_t &reward,
                                   std::map<VertexType, kl_gain_update_info> &,
                                   std::vector<VertexType> &new_nodes) {
        const unsigned start_step = thread_data.start_step;
        const unsigned end_step = thread_data.end_step;

        for (const auto &target : instance->getComputationalDag().children(move.node)) {
            const unsigned target_step = active_schedule->assigned_superstep(target);
            if (target_step < start_step || target_step > end_step) { continue; }

            if (thread_data.lock_manager.is_locked(target)) { continue; }

            if (not thread_data.affinity_table.is_selected(target)) {
                new_nodes.push_back(target);
                continue;
            }

            const unsigned target_proc = active_schedule->assigned_processor(target);
            const unsigned target_start_idx = start_idx(target_step, start_step);
            auto &affinity_table = thread_data.affinity_table.at(target);

            if (move.from_step < target_step + (move.from_proc == target_proc)) {
                const unsigned diff = target_step - move.from_step;
                const unsigned bound = window_size >= diff ? window_size - diff + 1 : 0;
                unsigned idx = target_start_idx;
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                        affinity_table[p][idx] -= penalty;
                    }
                }

                if (idx - 1 < bound && is_compatible(target, move.from_proc)) {
                    affinity_table[move.from_proc][idx - 1] += penalty;
                }

            } else {
                const unsigned diff = move.from_step - target_step;
                const unsigned window_bound = end_idx(target_step, end_step);
                unsigned idx = std::min(window_size + diff, window_bound);

                if (idx < window_bound && is_compatible(target, move.from_proc)) {
                    affinity_table[move.from_proc][idx] += reward;
                }

                idx++;

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                        affinity_table[p][idx] += reward;
                    }
                }
            }

            if (move.to_step < target_step + (move.to_proc == target_proc)) {
                unsigned idx = target_start_idx;
                const unsigned diff = target_step - move.to_step;
                const unsigned bound = window_size >= diff ? window_size - diff + 1 : 0;
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                        affinity_table[p][idx] += penalty;
                    }
                }

                if (idx - 1 < bound && is_compatible(target, move.to_proc)) { affinity_table[move.to_proc][idx - 1] -= penalty; }

            } else {
                const unsigned diff = move.to_step - target_step;
                const unsigned window_bound = end_idx(target_step, end_step);
                unsigned idx = std::min(window_size + diff, window_bound);

                if (idx < window_bound && is_compatible(target, move.to_proc)) { affinity_table[move.to_proc][idx] -= reward; }

                idx++;

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                        affinity_table[p][idx] -= reward;
                    }
                }
            }
        }

        for (const auto &source : instance->getComputationalDag().parents(move.node)) {
            const unsigned source_step = active_schedule->assigned_superstep(source);
            if (source_step < start_step || source_step > end_step) { continue; }

            if (thread_data.lock_manager.is_locked(source)) { continue; }

            if (not thread_data.affinity_table.is_selected(source)) {
                new_nodes.push_back(source);
                continue;
            }

            const unsigned source_proc = active_schedule->assigned_processor(source);
            const unsigned source_start_idx = start_idx(source_step, start_step);
            const unsigned window_bound = end_idx(source_step, end_step);
            auto &affinity_table_source = thread_data.affinity_table.at(source);

            if (move.from_step < source_step + (move.from_proc != source_proc)) {
                const unsigned diff = source_step - move.from_step;
                const unsigned bound = window_size > diff ? window_size - diff : 0;
                unsigned idx = source_start_idx;
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
                unsigned idx = source_start_idx;
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
        }
    }
};

}    // namespace osp
