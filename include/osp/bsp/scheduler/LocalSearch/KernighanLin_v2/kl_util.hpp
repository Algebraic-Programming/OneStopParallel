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
    cost_t max_weight;

    unsigned violations_threshold = 0;
    cost_t initial_penalty = 10.0;
    cost_t penalty = 0;
    cost_t reward = 0; 

    void initalize(kl_active_schedule_t & sched, const cost_t max_comm, const cost_t max_work) {
        max_weight = std::max(max_work, max_comm * sched.getInstance().communicationCosts());
        active_schedule = &sched;
        initial_penalty = std::sqrt(max_weight);
    }
 
    void init_reward_penalty(double multiplier = 1.0) {
        multiplier = std::min(multiplier, 10.0); 
        penalty = initial_penalty * multiplier;
        reward = max_weight * multiplier;
    }
};

template<typename VertexType>
struct set_vertex_lock_manger {

    std::unordered_set<VertexType> locked_nodes;

    void initialize(size_t ) {}

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

template<typename VertexType>
struct vector_vertex_lock_manger {

    std::vector<bool> locked_nodes;

    void initialize(size_t num_nodes) {
        locked_nodes.resize(num_nodes);
    }

    void lock(VertexType node) {
        locked_nodes[node] = true;
    }

    void unlock(VertexType node) {
        locked_nodes[node] = false;
    }

    bool is_locked(VertexType node) {
        return locked_nodes[node];
    }

    void clear() {
        locked_nodes.assign(locked_nodes.size(), false);
    }
};

template<typename Graph_t, typename cost_t, typename handle_t, typename kl_active_schedule_t, unsigned window_size>
struct adaptive_affinity_table {
    constexpr static unsigned window_range = 2 * window_size + 1;
    using VertexType = vertex_idx_t<Graph_t>;

private:
    const kl_active_schedule_t *active_schedule;
    const Graph_t * graph; 

    std::vector<bool> node_is_selected;
    std::vector<size_t> selected_nodes_idx;

    std::vector<std::vector<std::vector<cost_t>>> affinity_table;
    std::vector<handle_t> heap_handles;
    std::vector<VertexType> selected_nodes;

    std::vector<size_t> gaps;
    size_t last_idx;

public:

    void initialize(const kl_active_schedule_t & sche_, const std::size_t initial_table_size) {
        active_schedule = &sche_;
        graph = &(sche_.getInstance().getComputationalDag());

        last_idx = 0;
 
        node_is_selected.resize(graph->num_vertices());
        selected_nodes_idx.resize(graph->num_vertices());
        heap_handles.resize(initial_table_size);
        selected_nodes.resize(initial_table_size);

        node_is_selected.assign(node_is_selected.size(), false);

        affinity_table.resize(initial_table_size);
        const unsigned num_procs = sche_.getInstance().numberOfProcessors();
        for (auto &table : affinity_table) {
            table.resize(num_procs);
            for (auto &row : table) {
                row.resize(window_range);
            }
        }    
    }

    inline std::vector<VertexType>& get_selected_nodes() {
        return selected_nodes;
    }

    inline const std::vector<VertexType>& get_selected_nodes() const {
        return selected_nodes;
    }

    inline size_t size() const {
        return last_idx - gaps.size();
    }

    inline bool is_selected(VertexType node) const {
        return node_is_selected[node];
    }

    inline const std::vector<size_t> & get_selected_nodes_indices() const {
        return selected_nodes_idx;
    }

    inline size_t get_selected_nodes_idx(VertexType node) const {
        return selected_nodes_idx[node];
    }

    inline std::vector<std::vector<cost_t>> & operator[](VertexType node) {
        assert(node_is_selected[node]);
        return affinity_table[selected_nodes_idx[node]];
    }

    inline std::vector<std::vector<cost_t>> & at(VertexType node) {
        assert(node_is_selected[node]);
        return affinity_table[selected_nodes_idx[node]];
    }

    inline const std::vector<std::vector<cost_t>> & at(VertexType node) const {
        assert(node_is_selected[node]);
        return affinity_table[selected_nodes_idx[node]];
    }

    inline std::vector<std::vector<cost_t>> & get_affinity_table(VertexType node) {
        assert(node_is_selected[node]);
        return affinity_table[selected_nodes_idx[node]];
    }

    inline handle_t & get_heap_handle(VertexType node) {
        assert(node_is_selected[node]);
        return heap_handles[selected_nodes_idx[node]];
    }

    bool insert(VertexType node) {
        if (node_is_selected[node])
            return false; // Node is already in the table.

        size_t insert_location;
        if (!gaps.empty()) {
            insert_location = gaps.back();
            gaps.pop_back();
        } else {
            insert_location = last_idx;
            
            if (insert_location >= selected_nodes.size()) {
                const size_t old_size = selected_nodes.size();
                const size_t new_size = std::min(old_size * 2, graph->num_vertices());
                
                heap_handles.resize(new_size);
                selected_nodes.resize(new_size);
                affinity_table.resize(new_size);

                const unsigned num_procs = active_schedule->getInstance().numberOfProcessors();
                for (size_t i = old_size; i < new_size; ++i) {
                    affinity_table[i].resize(num_procs);
                    for (auto &row : affinity_table[i]) {
                        row.resize(window_range);
                    }
                }
            }
            last_idx++;
        }

        node_is_selected[node] = true;
        selected_nodes_idx[node] = insert_location;
        selected_nodes[insert_location] = node;
        
        return true;
    }

    void remove(VertexType node) {
        assert(node_is_selected[node]);
        node_is_selected[node] = false;

        gaps.push_back(selected_nodes_idx[node]);
    }
    
    void reset_node_selection() {
        node_is_selected.assign(node_is_selected.size(), false);        
        gaps.clear();
        last_idx = 0;
    }
    
    void clear() {
        node_is_selected.clear();
        selected_nodes_idx.clear();
        affinity_table.clear();
        heap_handles.clear();
        selected_nodes.clear();
        gaps.clear();
        last_idx = 0;
    }

    void trim() {
        while (!gaps.empty() && last_idx > 0) {
            size_t gap_idx = gaps.back();
            gaps.pop_back();

            size_t last_element_idx = last_idx - 1;

            if (gap_idx >= last_element_idx) {
                last_idx--;
                continue;
            }

            VertexType node_to_move = selected_nodes[last_element_idx];

            std::swap(affinity_table[gap_idx], affinity_table[last_element_idx]);
            std::swap(heap_handles[gap_idx],   heap_handles[last_element_idx]);
            std::swap(selected_nodes[gap_idx], selected_nodes[last_element_idx]);

            selected_nodes_idx[node_to_move] = gap_idx;

            last_idx--;
        }
        gaps.clear();
    }
};

template<typename Graph_t, typename cost_t, typename handle_t, typename kl_active_schedule_t, unsigned window_size>
struct static_affinity_table {
    constexpr static unsigned window_range = 2 * window_size + 1;
    using VertexType = vertex_idx_t<Graph_t>;

private:
    const kl_active_schedule_t *active_schedule;
    const Graph_t * graph; 

    std::unordered_map<VertexType, handle_t> selected_nodes; 

    std::vector<std::vector<std::vector<cost_t>>> affinity_table;

public:

    void initialize(const kl_active_schedule_t & sche_, const std::size_t ) {
        active_schedule = &sche_;
        graph = &(sche_.getInstance().getComputationalDag());

        affinity_table.resize(graph->num_vertices());
        const unsigned num_procs = sche_.getInstance().numberOfProcessors();
        for (auto &table : affinity_table) {
            table.resize(num_procs);
            for (auto &row : table) {
                row.resize(window_range);
            }
        }    
    }

    inline std::vector<VertexType> get_selected_nodes() const {
        std::vector<VertexType> snode;
        snode.reserve(selected_nodes.size());
        for (const auto & [key, value] : selected_nodes) {
            snode.push_back(key);
        }
        return snode;
    }

    inline size_t size() const {
        return selected_nodes.size();
    }

    inline bool is_selected(VertexType node) const {
        return selected_nodes.find(node) != selected_nodes.end(); 
    }

    inline std::vector<std::vector<cost_t>> & operator[](VertexType node) {
        return affinity_table[node];
    }

    inline std::vector<std::vector<cost_t>> & at(VertexType node) {
        return affinity_table[node];
    }

    inline const std::vector<std::vector<cost_t>> & at(VertexType node) const {
        return affinity_table[node];
    }

    inline std::vector<std::vector<cost_t>> & get_affinity_table(VertexType node) {
        return affinity_table[node];
    }

    inline handle_t & get_heap_handle(VertexType node) {
        return selected_nodes[node];
    }

    bool insert(VertexType node) {
        selected_nodes[node] = handle_t();
        return true;
    }

    void remove(VertexType node) {
        selected_nodes.erase(node);
    }
    
    void reset_node_selection() {
        selected_nodes.clear();       
    }
    
    void clear() {
        affinity_table.clear();
        selected_nodes.clear();
    }

    void trim() {}
};

template<typename Graph_t, typename container_t, typename handle_t, typename kl_active_schedule_t>
struct vertex_selection_strategy {

    const kl_active_schedule_t *active_schedule;
    const Graph_t * graph; 
    std::mt19937 * gen;
    std::size_t selection_threshold = 0;
    unsigned strategy_counter = 0;

    std::vector<vertex_idx_t<Graph_t>> permutation;
    std::size_t permutation_idx;

    unsigned max_work_counter = 0;

    inline void initialize(const kl_active_schedule_t & sche_, std::mt19937 & gen_) {
        active_schedule = &sche_;
        graph = &(sche_.getInstance().getComputationalDag());        
        gen = &gen_;

        strategy_counter = 0; 
        permutation = std::vector<vertex_idx_t<Graph_t>>(graph->num_vertices());
        std::iota(std::begin(permutation), std::end(permutation), 0);
        permutation_idx = 0;
        std::shuffle(permutation.begin(), permutation.end(), *gen);
    }

    void add_neighbours_to_selection(vertex_idx_t<Graph_t> node, container_t &nodes) {
        for (const auto parent : graph->parents(node))
            nodes.insert(parent);

        for (const auto child : graph->children(node))
            nodes.insert(child);
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
                    node_selection.insert(node);
                    break;
                }
            }
        }
    }

    void select_nodes_violations(container_t & node_selection) {
        for (const auto & edge : active_schedule->get_current_violations()) {
            node_selection.insert(source(edge, *graph));
            node_selection.insert(target(edge, *graph));
        }
    }

    void select_nodes_permutation_threshold(const std::size_t & threshold, container_t & node_selection) {

        const size_t bound = std::min(threshold + permutation_idx, graph->num_vertices());
        for (std::size_t i = permutation_idx; i < bound; i++) { 
                node_selection.insert(permutation[i]);
        }

        permutation_idx = bound;
        if (permutation_idx + threshold >= graph->num_vertices()) {
            permutation_idx = 0;
            std::shuffle(permutation.begin(), permutation.end(), *gen);
        }
    }

    void select_nodes_max_work_proc(const std::size_t & threshold, container_t & node_selection) {        
        while (node_selection.size() < threshold - 1) {
            if (max_work_counter >= active_schedule->num_steps())
                max_work_counter = 0;
                
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
                node_selection.insert(val);
            });    
        }        
    }
};

} // namespace osp