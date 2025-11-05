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
#include "lambda_container.hpp"
#include "max_comm_datastructure.hpp"

namespace osp {

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t, unsigned window_size = 1>
struct kl_bsp_comm_cost_function {
    
    using VertexType = vertex_idx_t<Graph_t>;
    using kl_move = kl_move_struct<cost_t, VertexType>;
    using kl_gain_update_info = kl_update_info<VertexType>;

    constexpr static unsigned window_range = 2 * window_size + 1;

    kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t> *active_schedule;
    compatible_processor_range<Graph_t> *proc_range;
    const Graph_t *graph;
    const BspInstance<Graph_t> *instance;

    max_comm_datastructure<Graph_t> comm_ds;

    inline cost_t get_comm_multiplier() { return 1; }
    inline cost_t get_max_comm_weight() { return comm_ds.max_comm_weight; }
    inline cost_t get_max_comm_weight_multiplied() { return comm_ds.max_comm_weight; }
    inline const std::string name() const { return "bsp_comm"; }
    inline bool is_compatible(VertexType node, unsigned proc) { return active_schedule->getInstance().isCompatible(node, proc); }
    inline unsigned start_idx(const unsigned node_step, const unsigned start_step) { return (node_step < window_size + start_step) ? window_size - (node_step - start_step) : 0; }
    inline unsigned end_idx(const unsigned node_step, const unsigned end_step) { return (node_step + window_size <= end_step) ? window_range : window_range - (node_step + window_size - end_step); }

    void initialize(kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t> &sched, compatible_processor_range<Graph_t> &p_range) {
        active_schedule = &sched;
        proc_range = &p_range;
        instance = &sched.getInstance();
        graph = &instance->getComputationalDag();

        const unsigned num_steps = active_schedule->num_steps();
        comm_ds.initialize(active_schedule->getSetSchedule(), *instance, num_steps);
        comm_ds.set_active_schedule(*active_schedule);
    }

    void compute_send_receive_datastructures() {        
        comm_ds.compute_comm_datastructures(0, active_schedule->num_steps() - 1);
    }

    template<bool compute_datastructures = true>
    cost_t compute_schedule_cost() {
        if constexpr (compute_datastructures) compute_send_receive_datastructures();

        cost_t total_cost = 0;
        for (unsigned step = 0; step < active_schedule->num_steps(); step++) {
            total_cost += active_schedule->get_step_max_work(step);
            total_cost += comm_ds.step_max_comm(step) * instance->communicationCosts();
        }
        total_cost += static_cast<cost_t>(active_schedule->num_steps() - 1) * instance->synchronisationCosts();
        return total_cost;
    }

    cost_t compute_schedule_cost_test() { return compute_schedule_cost<false>(); }

    void update_datastructure_after_move(const kl_move &move, const unsigned start_step, const unsigned end_step) {        
        comm_ds.update_datastructure_after_move(move, start_step, end_step);
    }

    template <typename affinity_table_t>
    void compute_comm_affinity(VertexType node, affinity_table_t &affinity_table_node, const cost_t &penalty,
                               const cost_t &reward, const unsigned start_step, const unsigned end_step) {

        const unsigned node_step = active_schedule->assigned_superstep(node);
        const unsigned node_proc = active_schedule->assigned_processor(node);
        const unsigned window_bound = end_idx(node_step, end_step);
        const unsigned node_start_idx = start_idx(node_step, start_step);

        const cost_t comm_w_node = graph->vertex_comm_weight(node);

        const auto &current_set_schedule = active_schedule->getSetSchedule();

        for (unsigned p_to = 0; p_to < instance->numberOfProcessors(); ++p_to) {
            if (!is_compatible(node, p_to)) continue;

            for (unsigned s_to_idx = node_start_idx; s_to_idx < window_bound; ++s_to_idx) {
                unsigned s_to = node_step + s_to_idx - window_size;
                cost_t comm_cost_change = 0;

                const auto pre_move_data_from = comm_ds.get_pre_move_comm_data_step(node_step);
                const auto pre_move_data_to = comm_ds.get_pre_move_comm_data_step(s_to);
                
                // --- Outgoing communication from `node` ---
                // From
                for (const auto [proc, count] : comm_ds.node_lambda_map.iterate_proc_entries(node)) {
                     comm_cost_change += calculate_comm_cost_change_send(node_step, node_proc, comm_w_node, -1, pre_move_data_from);
                }
                // To
                lambda_vector_container temp_lambda_map; // Use a temporary map for 'to' state
                temp_lambda_map.initialize(1, instance->numberOfProcessors());
                for (const auto &v : graph->children(node)) {
                    const unsigned v_proc = current_set_schedule.assignedProcessor(v);

                    if (p_to != v_proc) {
                        if (temp_lambda_map.increase_proc_count(0, v_proc)) {
                            comm_cost_change -= calculate_comm_cost_change_send(s_to, p_to, comm_w_node, 1, pre_move_data_to);
                            comm_cost_change -= calculate_comm_cost_change_receive(s_to, v_proc, comm_w_node, 1, pre_move_data_to);
                        }
                    }
                }

                // --- Incoming communication to `node` ---
                for (const auto &u : graph->parents(node)) {
                    const unsigned u_proc = active_schedule->assigned_processor(u);
                    const unsigned u_step = current_set_schedule.assignedSuperstep(u);
                    const cost_t comm_w_u = graph->vertex_comm_weight(u);
                    const auto pre_move_data_u = comm_ds.get_pre_move_comm_data_step(u_step);
                    
                    // From
                    if (u_proc != node_proc) {
                        // Send part (from parent u) & Receive part (at node_proc) // TODO: this is not correct, the lambda map is not updated
                        if (comm_ds.node_lambda_map.get_proc_entry(u, node_proc) == 1) { // if node is the only child on this proc
                            comm_cost_change += calculate_comm_cost_change_send(u_step, u_proc, comm_w_u, -1, pre_move_data_u);
                            comm_cost_change += calculate_comm_cost_change_receive(u_step, node_proc, comm_w_u, -1, pre_move_data_u);
                        }
                    }
                    // To
                    if (u_proc != p_to) {
                        // Send part (from parent u) & Receive part (at p_to)
                        // This logic is complex for an affinity calculation.
                        // A full recompute for neighbors is a safer bet, which is what update_node_comm_affinity does. // TODO: this is not true anymore
                        // The following is an approximation.

                        // if moving node to p_to creates a new communication link for parent u
                        bool has_other_on_p_to = false;
                        for(const auto& sibling : graph->children(u)) {
                            if (sibling != node && active_schedule->assigned_processor(sibling) == p_to) { has_other_on_p_to = true; break; }
                        }
                        if (!has_other_on_p_to) {
                             comm_cost_change -= calculate_comm_cost_change_send(u_step, u_proc, comm_w_u, 1, pre_move_data_u);
                             comm_cost_change -= calculate_comm_cost_change_receive(u_step, p_to, comm_w_u, 1, pre_move_data_u);
                        }
                    }
                }
                affinity_table_node[p_to][s_to_idx] += comm_cost_change * instance->communicationCosts();
            }
        }
    }

    cost_t calculate_comm_cost_change_send(unsigned step, unsigned p_send, cost_t comm_w, int sign, const pre_move_comm_data<cost_t>& pre_move_data) {
        cost_t old_max = pre_move_data.from_step_max_comm;

        cost_t new_send = comm_ds.step_proc_send(step, p_send) + sign * comm_w;
        cost_t new_max_send = comm_ds.step_max_send(step);
        if (new_send > new_max_send) new_max_send = new_send;
        else if (comm_ds.step_proc_send(step, p_send) == new_max_send) {
            if (sign < 0 && comm_ds.step_max_send_processor_count[step] == 1) {
                new_max_send = comm_ds.step_second_max_send(step);
            } else {
                new_max_send = new_send;
            }
        }

        return std::max(new_max_send, comm_ds.step_max_receive(step)) - old_max;
    }

    cost_t calculate_comm_cost_change_receive(unsigned step, unsigned p_receive, cost_t comm_w, int sign, const pre_move_comm_data<cost_t>& pre_move_data) {
        cost_t old_max = pre_move_data.from_step_max_comm;

        cost_t new_receive = comm_ds.step_proc_receive(step, p_receive) + sign * comm_w;

        cost_t new_max_receive = comm_ds.step_max_receive(step);
        if (new_receive > new_max_receive) new_max_receive = new_receive;
        else if (comm_ds.step_proc_receive(step, p_receive) == new_max_receive) {
            if (sign < 0 && comm_ds.step_max_receive_processor_count[step] == 1) {
                new_max_receive = comm_ds.step_second_max_receive(step);
            } else {
                new_max_receive = new_receive;
            }
        }

        return std::max(comm_ds.step_max_send(step), new_max_receive) - old_max;
    }

    cost_t calculate_comm_cost_change(unsigned step, unsigned p_send, unsigned p_receive, cost_t comm_w, int sign) {
        const auto pre_move_data = comm_ds.get_pre_move_comm_data_step(step);
        cost_t change = 0;
        change += calculate_comm_cost_change_send(step, p_send, comm_w, sign, pre_move_data);
        comm_ds.step_proc_send(step, p_send) += sign * comm_w;
        change += calculate_comm_cost_change_receive(step, p_receive, comm_w, sign, pre_move_data);
        comm_ds.step_proc_send(step, p_send) -= sign * comm_w; // revert for next calculation
        return change;
    }

    template <typename thread_data_t>
    void update_node_comm_affinity(const kl_move &move, thread_data_t &thread_data, const cost_t &penalty,
                                   const cost_t &reward, std::map<VertexType, kl_gain_update_info> &max_gain_recompute,
                                   std::vector<VertexType> &new_nodes) {
        // For simplicity and correctness, we will do a full recompute for neighbors.
        // A fully incremental update is very complex for this cost function.
        auto process_neighbor = [&](VertexType neighbor) {
            if (thread_data.lock_manager.is_locked(neighbor)) return;
            if (not thread_data.affinity_table.is_selected(neighbor)) {
                new_nodes.push_back(neighbor);
                return;
            }
            if (max_gain_recompute.find(neighbor) == max_gain_recompute.end()) {
                max_gain_recompute[neighbor] = kl_gain_update_info(neighbor, true);
            } else {
                max_gain_recompute[neighbor].full_update = true;
            }
        };

        for (const auto &target : graph->children(move.node)) {
            process_neighbor(target);
        }
        for (const auto &source : graph->parents(move.node)) {
            process_neighbor(source);
        }
    }
};

} // namespace osp