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

#include "lambda_container.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace osp {

template<typename comm_weight_t>
struct pre_move_comm_data {

    struct step_info {
        comm_weight_t max_comm;
        comm_weight_t second_max_comm;
        unsigned max_comm_count;
    };

    std::unordered_map<unsigned, step_info> step_data;

    pre_move_comm_data() = default;

    void add_step(unsigned step, comm_weight_t max, comm_weight_t second, unsigned count) {
        step_data[step] = {max, second, count};
    }

    bool get_step(unsigned step, step_info &info) const {
        auto it = step_data.find(step);
        if (it != step_data.end()) {
            info = it->second;
            return true;
        }
        return false;
    }
};

template<typename Graph_t, typename cost_t, typename kl_active_schedule_t>
struct max_comm_datastructure {

    using comm_weight_t = v_commw_t<Graph_t>;
    using VertexType = vertex_idx_t<Graph_t>;
    using kl_move = kl_move_struct<cost_t, VertexType>;

    const BspInstance<Graph_t> *instance;
    const kl_active_schedule_t *active_schedule;

    std::vector<std::vector<comm_weight_t>> step_proc_send_;
    std::vector<std::vector<comm_weight_t>> step_proc_receive_;

    // Caches for fast cost calculation (Global Max/Second Max per step)
    std::vector<comm_weight_t> step_max_comm_cache;
    std::vector<comm_weight_t> step_second_max_comm_cache;
    std::vector<unsigned> step_max_comm_count_cache;

    comm_weight_t max_comm_weight = 0;

    lambda_vector_container<VertexType> node_lambda_map;

    // Optimization: Scratchpad for update_datastructure_after_move to avoid allocations
    std::vector<unsigned> affected_steps_list;
    std::vector<bool> step_is_affected;

    inline comm_weight_t step_proc_send(unsigned step, unsigned proc) const { return step_proc_send_[step][proc]; }
    inline comm_weight_t &step_proc_send(unsigned step, unsigned proc) { return step_proc_send_[step][proc]; }
    inline comm_weight_t step_proc_receive(unsigned step, unsigned proc) const {
        return step_proc_receive_[step][proc];
    }
    inline comm_weight_t &step_proc_receive(unsigned step, unsigned proc) { return step_proc_receive_[step][proc]; }

    inline comm_weight_t step_max_comm(unsigned step) const { return step_max_comm_cache[step]; }
    inline comm_weight_t step_second_max_comm(unsigned step) const { return step_second_max_comm_cache[step]; }
    inline unsigned step_max_comm_count(unsigned step) const { return step_max_comm_count_cache[step]; }

    inline void initialize(kl_active_schedule_t &kl_sched) {
        active_schedule = &kl_sched;
        instance = &active_schedule->getInstance();
        const unsigned num_steps = active_schedule->num_steps();
        const unsigned num_procs = instance->numberOfProcessors();
        max_comm_weight = 0;

        step_proc_send_.assign(num_steps, std::vector<comm_weight_t>(num_procs, 0));
        step_proc_receive_.assign(num_steps, std::vector<comm_weight_t>(num_procs, 0));

        step_max_comm_cache.assign(num_steps, 0);
        step_second_max_comm_cache.assign(num_steps, 0);
        step_max_comm_count_cache.assign(num_steps, 0);

        node_lambda_map.initialize(instance->getComputationalDag().num_vertices(), num_procs);

        // Initialize scratchpad
        step_is_affected.assign(num_steps, false);
        affected_steps_list.reserve(num_steps);
    }

    inline void clear() {
        step_proc_send_.clear();
        step_proc_receive_.clear();
        step_max_comm_cache.clear();
        step_second_max_comm_cache.clear();
        step_max_comm_count_cache.clear();
        node_lambda_map.clear();
        affected_steps_list.clear();
        step_is_affected.clear();
    }

    inline void arrange_superstep_comm_data(const unsigned step) {
        // Linear scan O(P) to find max, second_max and count
        
        // 1. Analyze Sends
        comm_weight_t max_send = 0;
        comm_weight_t second_max_send = 0;
        unsigned max_send_count = 0;

        const auto &sends = step_proc_send_[step];
        for (const auto val : sends) {
            if (val > max_send) {
                second_max_send = max_send;
                max_send = val;
                max_send_count = 1;
            } else if (val == max_send) {
                max_send_count++;
            } else if (val > second_max_send) {
                second_max_send = val;
            }
        }

        // 2. Analyze Receives
        comm_weight_t max_receive = 0;
        comm_weight_t second_max_receive = 0;
        unsigned max_receive_count = 0;

        const auto &receives = step_proc_receive_[step];
        for (const auto val : receives) {
            if (val > max_receive) {
                second_max_receive = max_receive;
                max_receive = val;
                max_receive_count = 1;
            } else if (val == max_receive) {
                max_receive_count++;
            } else if (val > second_max_receive) {
                second_max_receive = val;
            }
        }

        // 3. Aggregate Global Stats
        const comm_weight_t global_max = std::max(max_send, max_receive);
        step_max_comm_cache[step] = global_max;

        unsigned global_count = 0;
        if (max_send == global_max)
            global_count += max_send_count;
        if (max_receive == global_max)
            global_count += max_receive_count;
        step_max_comm_count_cache[step] = global_count;

        // Determine second max
        comm_weight_t cand_send = (max_send == global_max) ? second_max_send : max_send;
        comm_weight_t cand_recv = (max_receive == global_max) ? second_max_receive : max_receive;

        step_second_max_comm_cache[step] = std::max(cand_send, cand_recv);
    }

    void recompute_max_send_receive(unsigned step) { arrange_superstep_comm_data(step); }

    inline pre_move_comm_data<comm_weight_t> get_pre_move_comm_data(const kl_move &move) {
        pre_move_comm_data<comm_weight_t> data;
        std::unordered_set<unsigned> affected_steps;

        affected_steps.insert(move.from_step);
        affected_steps.insert(move.to_step);

        const auto &graph = instance->getComputationalDag();

        for (const auto &parent : graph.parents(move.node)) {
            affected_steps.insert(active_schedule->assigned_superstep(parent));
        }

        for (unsigned step : affected_steps) {
            data.add_step(step, step_max_comm(step), step_second_max_comm(step), step_max_comm_count(step));
        }

        return data;
    }

    void update_datastructure_after_move(const kl_move &move, unsigned, unsigned) {
        const auto &graph = instance->getComputationalDag();

        // --- 0. Prepare Scratchpad (Avoids Allocations) ---
        for (unsigned step : affected_steps_list) {
            if (step < step_is_affected.size())
                step_is_affected[step] = false;
        }
        affected_steps_list.clear();

        auto mark_step = [&](unsigned step) {
            if (step < step_is_affected.size() && !step_is_affected[step]) {
                step_is_affected[step] = true;
                affected_steps_list.push_back(step);
            }
        };

        const VertexType node = move.node;
        const unsigned from_step = move.from_step;
        const unsigned to_step = move.to_step;
        const unsigned from_proc = move.from_proc;
        const unsigned to_proc = move.to_proc;
        const comm_weight_t comm_w_node = graph.vertex_comm_weight(node);

        // --- 1. Handle Node Movement (Outgoing Edges: Node -> Children) ---

        if (from_step != to_step) {
            // Case 1: Node changes Step
            // Optimization: Fuse the loop to iterate lambda map only once.
            
            for (const auto [proc, count] : node_lambda_map.iterate_proc_entries(node)) {
                // A. Remove Old (Sender: from_proc, Receiver: proc)
                if (proc != from_proc) {
                    const comm_weight_t cost = comm_w_node * instance->sendCosts(from_proc, proc);
                    // Optimization: check cost > 0 to avoid dirtying cache lines with +0 ops
                    if (cost > 0) { 
                        step_proc_receive_[from_step][proc] -= cost;
                        step_proc_send_[from_step][from_proc] -= cost;
                    }
                }

                // B. Add New (Sender: to_proc, Receiver: proc)
                if (proc != to_proc) {
                    const comm_weight_t cost = comm_w_node * instance->sendCosts(to_proc, proc);
                    if (cost > 0) {
                        step_proc_receive_[to_step][proc] += cost;
                        step_proc_send_[to_step][to_proc] += cost;
                    }
                }
            }
            mark_step(from_step);
            mark_step(to_step);

        } else if (from_proc != to_proc) {
            // Case 2: Node stays in same Step, but changes Processor

            for (const auto [proc, count] : node_lambda_map.iterate_proc_entries(node)) {
                // Remove Old (Sender: from_proc, Receiver: proc)
                if (proc != from_proc) {
                    const comm_weight_t cost = comm_w_node * instance->sendCosts(from_proc, proc);
                    if (cost > 0) {
                        step_proc_receive_[from_step][proc] -= cost;
                        step_proc_send_[from_step][from_proc] -= cost;
                    }
                }

                // Add New (Sender: to_proc, Receiver: proc)
                if (proc != to_proc) {
                    const comm_weight_t cost = comm_w_node * instance->sendCosts(to_proc, proc);
                    if (cost > 0) {
                        step_proc_receive_[from_step][proc] += cost;
                        step_proc_send_[from_step][to_proc] += cost;
                    }
                }
            }
            mark_step(from_step);
        }

        // --- 2. Update Parents' Outgoing Communication (Parents → Node) ---

        if (from_proc != to_proc) {
            for (const auto &parent : graph.parents(node)) {
                const unsigned parent_step = active_schedule->assigned_superstep(parent);
                // Fast boundary check
                if (parent_step >= step_proc_send_.size())
                    continue;

                const unsigned parent_proc = active_schedule->assigned_processor(parent);
                const comm_weight_t comm_w_parent = graph.vertex_comm_weight(parent);

                const bool removed_from_proc = node_lambda_map.decrease_proc_count(parent, from_proc);
                const bool added_to_proc = node_lambda_map.increase_proc_count(parent, to_proc);

                // 1. Handle Removal from from_proc
                if (removed_from_proc) {
                    if (from_proc != parent_proc) {
                        const comm_weight_t cost = comm_w_parent * instance->sendCosts(parent_proc, from_proc);
                        if (cost > 0) {
                            step_proc_send_[parent_step][parent_proc] -= cost;
                            step_proc_receive_[parent_step][from_proc] -= cost;
                        }
                    }
                }

                // 2. Handle Addition to to_proc
                if (added_to_proc) {
                    if (to_proc != parent_proc) {
                        const comm_weight_t cost = comm_w_parent * instance->sendCosts(parent_proc, to_proc);
                        if (cost > 0) {
                            step_proc_send_[parent_step][parent_proc] += cost;
                            step_proc_receive_[parent_step][to_proc] += cost;
                        }
                    }
                }

                mark_step(parent_step);
            }
        }

        // --- 3. Re-arrange Affected Steps ---
        for (unsigned step : affected_steps_list) {
            arrange_superstep_comm_data(step);
        }
    }

    void swap_steps(const unsigned step1, const unsigned step2) {
        std::swap(step_proc_send_[step1], step_proc_send_[step2]);
        std::swap(step_proc_receive_[step1], step_proc_receive_[step2]);
        std::swap(step_max_comm_cache[step1], step_max_comm_cache[step2]);
        std::swap(step_second_max_comm_cache[step1], step_second_max_comm_cache[step2]);
        std::swap(step_max_comm_count_cache[step1], step_max_comm_count_cache[step2]);
    }

    void reset_superstep(unsigned step) {
        std::fill(step_proc_send_[step].begin(), step_proc_send_[step].end(), 0);
        std::fill(step_proc_receive_[step].begin(), step_proc_receive_[step].end(), 0);
        arrange_superstep_comm_data(step);
    }

    void compute_comm_datastructures(unsigned start_step, unsigned end_step) {
        for (unsigned step = start_step; step <= end_step; step++) {
            std::fill(step_proc_send_[step].begin(), step_proc_send_[step].end(), 0);
            std::fill(step_proc_receive_[step].begin(), step_proc_receive_[step].end(), 0);
        }

        const auto &vec_sched = active_schedule->getVectorSchedule();
        const auto &graph = instance->getComputationalDag();

        for (const auto &u : graph.vertices()) {
            node_lambda_map.reset_node(u);
            const unsigned u_proc = vec_sched.assignedProcessor(u);
            const unsigned u_step = vec_sched.assignedSuperstep(u);
            const comm_weight_t comm_w = graph.vertex_comm_weight(u);
            max_comm_weight = std::max(max_comm_weight, comm_w);

            for (const auto &v : graph.children(u)) {
                const unsigned v_proc = vec_sched.assignedProcessor(v);
                const unsigned v_step = vec_sched.assignedSuperstep(v);                
                const comm_weight_t comm_w_send_cost = (u_proc != v_proc) ? comm_w * instance->sendCosts(u_proc, v_proc) : 0;
                
                if (node_lambda_map.increase_proc_count(u, v_proc)) {
                    if (u_proc != v_proc && comm_w_send_cost > 0) {
                        attribute_communication(comm_w_send_cost, u_step, u_proc, v_proc, v_step);
                    }
                }
            }
        }

        for (unsigned step = start_step; step <= end_step; step++) {
            arrange_superstep_comm_data(step);
        }
    }

    inline void attribute_communication(const comm_weight_t &comm_w_send_cost, const unsigned u_step, const unsigned u_proc, const unsigned v_proc,
                                        const unsigned) {
        step_proc_receive_[u_step][v_proc] += comm_w_send_cost;
        step_proc_send_[u_step][u_proc] += comm_w_send_cost;
    }
};

} // namespace osp