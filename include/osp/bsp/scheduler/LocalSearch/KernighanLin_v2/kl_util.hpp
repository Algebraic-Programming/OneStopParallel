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

#include <unordered_set>
#include "kl_active_schedule.hpp"


namespace osp {


template<typename cost_t, typename comm_cost_function_t, typename kl_active_schedule_t>
struct reward_penalty_strategy {
    
    kl_active_schedule_t *active_schedule;
    cost_t instance_comm_cost;
    cost_t max_comm_weight;

    unsigned violations_threshold = 0;
    cost_t initial_penalty = 10.0;
    cost_t penalty = 0;
    cost_t reward = 0; 

    void initalize(kl_active_schedule_t & sched, const cost_t max_comm) {
        max_comm_weight = max_comm;
        active_schedule = &sched;
        instance_comm_cost = sched.getInstance().communicationCosts();
        initial_penalty = max_comm_weight * instance_comm_cost;
    }
 
    void init_reward_penalty(double reward_multiplier = 1.0) {
        penalty = (initial_penalty + 1) * reward_multiplier;
        reward = reward_multiplier * max_comm_weight * max_comm_weight * instance_comm_cost;
    }

    void update_reward_penalty() {

        const size_t num_violations = active_schedule->get_current_violations().size();

        if (num_violations <= violations_threshold) {
            penalty = initial_penalty;
            reward = 0.0;
        } else {
            violations_threshold = 0;
            penalty = std::log((num_violations)) * max_comm_weight * instance_comm_cost;
            reward = std::sqrt((num_violations + 4)) * max_comm_weight * instance_comm_cost;
        }
    }
};


template<typename VertexType>
struct set_vertex_lock_manger {

    std::unordered_set<VertexType> locked_nodes;

    void lock(VertexType node) {
        locked_nodes.insert(node);
    }

    void unlock(VertexType node) {
        locked_nodes.erase(node);
    }

    bool is_locked(VertexType node) {
        return locked_nodes.find(node) != locked_nodes.end();
    }

    void clear() {
        locked_nodes.clear();
    }
};

template<typename Graph_t, typename container_t, typename handle_t, typename kl_active_schedule_t>
struct vertex_selection_strategy {

    const kl_active_schedule_t *active_schedule;
    const Graph_t * graph; 
    std::mt19937 * gen;
    std::size_t selection_threshold;
    unsigned strategy_counter = 0;

    std::vector<vertex_idx_t<Graph_t>> permutation;
    std::size_t permutation_idx;

    unsigned max_work_counter = 0;

    inline void initialize(const kl_active_schedule_t & sche_, std::mt19937 & gen_) {
        active_schedule = &sche_;
        graph = &(sche_.getInstance().getComputationalDag());        
        gen = &gen_;

        strategy_counter = 0;
        selection_threshold = static_cast<std::size_t>(std::ceil(5.0 * std::log(graph->num_vertices())));

        permutation = std::vector<vertex_idx_t<Graph_t>>(graph->num_vertices());
        std::iota(std::begin(permutation), std::end(permutation), 0);
        permutation_idx = 0;
        std::shuffle(permutation.begin(), permutation.end(), *gen);
    }

    void add_neighbours_to_selection(vertex_idx_t<Graph_t> node, container_t &nodes) {
        for (const auto parent : graph->parents(node))
            nodes[parent] = handle_t();

        for (const auto child : graph->children(node))
            nodes[child] = handle_t();
    }

    inline void select_active_nodes(container_t & node_selection) {        
        if (strategy_counter < 3) {
            select_nodes_permutation_threshold(selection_threshold, node_selection);    
        } else if (strategy_counter == 4) {
            select_nodes_max_work_proc(selection_threshold, node_selection);
        }

        strategy_counter++;
        strategy_counter %= 5;
    }

    void select_nodes_comm(container_t & node_selection) {
        for (const auto &node : graph->vertices()) {
            for (const auto &source : graph->parents(node)) {
                if (active_schedule->assigned_processor(node) !=
                    active_schedule->assigned_processor(source)) {

                    node_selection[node] = handle_t();
                    break;
                }
            }
        }
    }

    void select_nodes_violations(container_t & node_selection) {
        for (const auto & edge : active_schedule->get_current_violations()) {
            node_selection[source(edge, *graph)] = handle_t();
            node_selection[target(edge, *graph)] = handle_t();
        }
    }

    void select_nodes_permutation_threshold(const std::size_t & threshold, container_t & node_selection) {

        const size_t bound = std::min(threshold + permutation_idx, graph->num_vertices());
        for (std::size_t i = permutation_idx; i < bound; i++) { 
                node_selection[permutation[i]] = handle_t();
        }

        permutation_idx = bound;
        if (permutation_idx + threshold >= graph->num_vertices()) {
            permutation_idx = 0;
            std::shuffle(permutation.begin(), permutation.end(), *gen);
        }
    }

    void select_nodes_max_work_proc(const std::size_t & threshold, container_t & node_selection) {        
        while (node_selection.size() < threshold - 1) {
            select_nodes_max_work_proc_helper(threshold - node_selection.size(), max_work_counter, node_selection);
            max_work_counter++;
            if(max_work_counter >= active_schedule->num_steps()) {
                max_work_counter = 0;
                break;
            }
        }
    }

    void select_nodes_max_work_proc_helper(const std::size_t & threshold, unsigned step, container_t & node_selection) {        
        const unsigned num_max_work_proc = active_schedule->work_datastructures.step_max_work_processor_count[step];
        for (unsigned idx = 0; idx < num_max_work_proc; idx++) {
            const unsigned proc = active_schedule->work_datastructures.step_processor_work_[step][idx].proc;
            const std::unordered_set<vertex_idx_t<Graph_t>> step_proc_vert = active_schedule->getSetSchedule().step_processor_vertices[step][proc];
            const size_t num_insert = std::min(threshold - node_selection.size(), step_proc_vert.size());                 
            auto end_it = step_proc_vert.begin();
            std::advance(end_it, num_insert);
            std::for_each(step_proc_vert.begin(), end_it, [&](const auto& val) {
                node_selection[val] = handle_t();
            });    
        }        
    }
};

} // namespace osp