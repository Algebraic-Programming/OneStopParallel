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

#include "kl_active_schedule.hpp"
#include "kl_improver.hpp"

#include <unordered_map>
namespace osp {


struct lambda_map_container {

    std::vector<std::unordered_map<unsigned,unsigned>> node_lambda_map;

    inline void initialize(const size_t num_vertices, const unsigned) { node_lambda_map.resize(num_vertices); }
    inline void reset_node(const size_t node) { node_lambda_map[node].clear(); }
    inline void clear() { node_lambda_map.clear(); }
    inline bool has_proc_entry(const size_t node, const unsigned proc) const { return (node_lambda_map[node].find(proc) != node_lambda_map[node].end()); }
    inline bool has_no_proc_entry(const size_t node, const unsigned proc) const { return (node_lambda_map[node].find(proc) == node_lambda_map[node].end()); }
    inline unsigned & get_proc_entry(const size_t node, const unsigned proc) { return node_lambda_map[node][proc]; }

    inline bool increase_proc_count(const size_t node, const unsigned proc) {
        if (has_proc_entry(node, proc)) {
            node_lambda_map[node][proc]++;
            return false;
        } else {
            node_lambda_map[node][proc] = 1;
            return true;
        }
    }

    inline bool decrease_proc_count(const size_t node, const unsigned proc) {
        assert(has_proc_entry(node, proc));
        if (node_lambda_map[node][proc] == 1) {
            node_lambda_map[node].erase(proc);
            return true;
        } else {
            node_lambda_map[node][proc]--;
            return false;
        }
    }

    inline const auto & iterate_proc_entries(const size_t node) {
        return node_lambda_map[node];
    }

};

struct lambda_vector_container {
   
    class lambda_vector_range {
        private:
            const std::vector<unsigned> & vec_;

        public:
        class lambda_vector_iterator {
        
            using iterator_category = std::input_iterator_tag;
            using value_type = std::pair<unsigned, unsigned>;
            using difference_type = std::ptrdiff_t;
            using pointer = value_type*;
            using reference = value_type&;
        private:
            const std::vector<unsigned>& vec_;
            size_t index_;
        public:

        lambda_vector_iterator(const std::vector<unsigned>& vec) : vec_(vec), index_(0) {
            // Advance to the first valid entry
            while (index_ < vec_.size() && vec_[index_] == 0) {
                ++index_;
            }
        }

        lambda_vector_iterator(const std::vector<unsigned>& vec, size_t index) : vec_(vec), index_(index) {}

        lambda_vector_iterator& operator++() {
                ++index_;
                while (index_ < vec_.size() && vec_[index_] == 0) {
                    ++index_;
                }
                return *this;
            }

            value_type operator*() const {
                return std::make_pair(static_cast<unsigned>(index_), vec_[index_]);
            }

            bool operator==(const lambda_vector_iterator& other) const {
                return index_ == other.index_;
            }

            bool operator!=(const lambda_vector_iterator& other) const {
                return !(*this == other);
            }
        };

        lambda_vector_range(const std::vector<unsigned>& vec) : vec_(vec) {}

        lambda_vector_iterator begin() { return lambda_vector_iterator(vec_); }
        lambda_vector_iterator end() { return lambda_vector_iterator(vec_, vec_.size()); }
    };

    std::vector<std::vector<unsigned>> node_lambda_vec;
    unsigned num_procs_ = 0;

    inline void initialize(const size_t num_vertices, const unsigned num_procs) { 
        node_lambda_vec.assign(num_vertices, {num_procs});
        num_procs_ = num_procs; 
    }


    inline void reset_node(const size_t node) { node_lambda_vec[node].assign(num_procs_, 0); }
    inline void clear() { node_lambda_vec.clear(); }
    inline bool has_proc_entry(const size_t node, const unsigned proc) const { return node_lambda_vec[node][proc] > 0; }
    inline bool has_no_proc_entry(const size_t node, const unsigned proc) const { return node_lambda_vec[node][proc] == 0; }
    inline unsigned & get_proc_entry(const size_t node, const unsigned proc) { return node_lambda_vec[node][proc]; }

    inline unsigned get_proc_entry(const size_t node, const unsigned proc) const {
        assert(has_proc_entry(node, proc));
        return node_lambda_vec[node][proc];
    }

    inline bool increase_proc_count(const size_t node, const unsigned proc) {
        node_lambda_vec[node][proc]++;
        return node_lambda_vec[node][proc] == 1;
    }

    inline bool decrease_proc_count(const size_t node, const unsigned proc) {
        assert(has_proc_entry(node, proc));
        node_lambda_vec[node][proc]--;
        return node_lambda_vec[node][proc] == 0;
    }

    inline auto iterate_proc_entries(const size_t node) {
        return lambda_vector_range(node_lambda_vec[node]);
    }

};

template<typename Graph_t, typename cost_t, typename MemoryConstraint_t, unsigned window_size = 1, bool use_node_communication_costs_arg = true> 
struct kl_hyper_total_comm_cost_function {
    
    using VertexType = vertex_idx_t<Graph_t>;
    using kl_move = kl_move_struct<cost_t, VertexType>;
    using kl_gain_update_info = kl_update_info<VertexType>;
   
    constexpr static unsigned window_range = 2 * window_size + 1;
    constexpr static bool use_node_communication_costs = use_node_communication_costs_arg || not has_edge_weights_v<Graph_t>;
  
    kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t> *active_schedule;

    compatible_processor_range<Graph_t> *proc_range;

    const Graph_t *graph;
    const BspInstance<Graph_t> *instance;

    cost_t comm_multiplier = 1; 
    cost_t max_comm_weight = 0;

    lambda_vector_container node_lambda_map;

    inline cost_t get_comm_multiplier() { return comm_multiplier; }
    inline cost_t get_max_comm_weight() { return max_comm_weight; }
    inline cost_t get_max_comm_weight_multiplied() { return max_comm_weight * comm_multiplier; }

    const std::string name() const { return "toal_comm_cost"; }

    inline bool is_compatible(VertexType node, unsigned proc) { return active_schedule->getInstance().isCompatible(node, proc); }

    void initialize(kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t> &sched, compatible_processor_range<Graph_t> &p_range) {
        active_schedule = &sched;
        proc_range = &p_range;
        instance = &sched.getInstance();
        graph = &instance->getComputationalDag();
        comm_multiplier = 1.0 / instance->numberOfProcessors();  
        node_lambda_map.initialize(graph->num_vertices(), instance->numberOfProcessors());      
    }

    cost_t compute_schedule_cost() {
        cost_t work_costs = 0;
        for (unsigned step = 0; step < active_schedule->num_steps(); step++) {
            work_costs += active_schedule->get_step_max_work(step);
        }

        cost_t comm_costs = 0;
        for(const auto vertex : graph->vertices()) {
            const unsigned vertex_proc = active_schedule->assigned_processor(vertex);
            const cost_t v_comm_cost = graph->vertex_comm_weight(vertex);
            max_comm_weight = std::max(max_comm_weight, v_comm_cost);

            node_lambda_map.reset_node(vertex);

            for (const auto &target : instance->getComputationalDag().children(vertex)) {
                const unsigned target_proc = active_schedule->assigned_processor(target);

                if (node_lambda_map.increase_proc_count(vertex, target_proc)) {
                    comm_costs += v_comm_cost * instance->communicationCosts(vertex_proc, target_proc); // is 0 if target_proc == vertex_proc
                }
            } 
        }

        return work_costs + comm_costs * comm_multiplier + static_cast<v_commw_t<Graph_t>>(active_schedule->num_steps() - 1) * instance->synchronisationCosts();
    }

    cost_t compute_schedule_cost_test() {
        cost_t work_costs = 0;
        for (unsigned step = 0; step < active_schedule->num_steps(); step++) {
            work_costs += active_schedule->get_step_max_work(step);
        }

        cost_t comm_costs = 0;
        for(const auto vertex : graph->vertices()) {
            const unsigned vertex_proc = active_schedule->assigned_processor(vertex);
            const cost_t v_comm_cost = graph->vertex_comm_weight(vertex);
            for (const auto [lambda_proc, mult] : node_lambda_map.iterate_proc_entries(vertex)) {
                comm_costs += v_comm_cost * instance->communicationCosts(vertex_proc, lambda_proc);
            } 
        }

        return work_costs + comm_costs * comm_multiplier + static_cast<v_commw_t<Graph_t>>(active_schedule->num_steps() - 1) * instance->synchronisationCosts();
    }

    inline void update_datastructure_after_move(const kl_move & move, const unsigned start_step, const unsigned end_step) {
        if (move.to_proc != move.from_proc) {  
            for (const auto &source : instance->getComputationalDag().parents(move.node)) {
                const unsigned source_step = active_schedule->assigned_superstep(source);
                if (source_step < start_step || source_step > end_step)
                    continue;
                update_source_after_move(move, source);    
            }
        }
    }

    inline void update_source_after_move(const kl_move & move, VertexType source) {
        node_lambda_map.decrease_proc_count(source, move.from_proc);
        node_lambda_map.increase_proc_count(source, move.to_proc);
    }

    template<typename thread_data_t>
    void update_node_comm_affinity(const kl_move &move, thread_data_t& thread_data, const cost_t& penalty, const cost_t& reward, std::map<VertexType, kl_gain_update_info> & max_gain_recompute, std::vector<VertexType> &new_nodes) {
                
        const unsigned start_step = thread_data.start_step;
        const unsigned end_step = thread_data.end_step;
                     
        for (const auto &target : instance->getComputationalDag().children(move.node)) {
            const unsigned target_step = active_schedule->assigned_superstep(target); 
            if (target_step < start_step || target_step > end_step)
                continue;

            if(thread_data.lock_manager.is_locked(target))
                continue;

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
            auto &affinity_table = thread_data.affinity_table.at(target);

            if (move.from_step < target_step + (move.from_proc == target_proc)) {
                const unsigned diff = target_step - move.from_step;                
                const unsigned bound = window_size >= diff ? window_size - diff + 1: 0;  
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
                const unsigned bound = window_size >= diff ? window_size - diff + 1: 0;  
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) { 
                        affinity_table[p][idx] += penalty;
                    }                                                
                } 

                if (idx - 1 < bound && is_compatible(target, move.to_proc)) {
                    affinity_table[move.to_proc][idx - 1] -= penalty;    
                }

            } else {
                const unsigned diff = move.to_step - target_step;
                const unsigned window_bound = end_idx(target_step, end_step); 
                unsigned idx = std::min(window_size + diff, window_bound);                                                     
                
                if (idx < window_bound && is_compatible(target, move.to_proc)) {
                    affinity_table[move.to_proc][idx] -= reward; 
                }

                idx++;
                                    
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) { 
                        affinity_table[p][idx] -= reward;
                    }                        
                } 
            }

            if (move.to_proc != move.from_proc) {                  
                const cost_t comm_gain = graph->vertex_comm_weight(move.node) * comm_multiplier;
                
                const unsigned window_bound = end_idx(target_step, end_step);
                for (const unsigned p : proc_range->compatible_processors_vertex(target)) { 
                    if (p == target_proc)
                        continue;
                    if (node_lambda_map.get_proc_entry(move.node, target_proc) == 1) {
                        for (unsigned idx = target_start_idx; idx < window_bound; idx++) {
                            const cost_t x = instance->communicationCosts(move.from_proc, target_proc) * comm_gain;
                            const cost_t y = instance->communicationCosts(move.to_proc, target_proc) * comm_gain;
                            affinity_table[p][idx] += x - y;
                        } 
                    }

                    if (node_lambda_map.has_no_proc_entry(move.node, p)) {
                        for (unsigned idx = target_start_idx; idx < window_bound; idx++) {
                            const cost_t x = instance->communicationCosts(move.from_proc, p) * comm_gain;
                            const cost_t y = instance->communicationCosts(move.to_proc, p) * comm_gain;
                            affinity_table[p][idx] -= x - y;                        
                        }
                    }  
                }
            } 
        }

        for (const auto &source : instance->getComputationalDag().parents(move.node)) {            

            if (move.to_proc != move.from_proc) {
                const unsigned source_proc = active_schedule->assigned_processor(source);   
                if (node_lambda_map.has_no_proc_entry(source, move.from_proc)) {                    
                    const cost_t comm_gain = graph->vertex_comm_weight(source) * comm_multiplier;

                    for (const auto &target : instance->getComputationalDag().children(source)) {
                        const unsigned target_step = active_schedule->assigned_superstep(target);
                        if ((target_step < start_step || target_step > end_step) || (target == move.node) || (not thread_data.affinity_table.is_selected(target)) || thread_data.lock_manager.is_locked(target))  
                            continue;  

                        if (source_proc != move.from_proc && is_compatible(target, move.from_proc)) { 
                            if (max_gain_recompute.find(target) != max_gain_recompute.end()) { // todo more specialized update
                                max_gain_recompute[target].full_update = true;                
                            } else {
                                max_gain_recompute[target] = kl_gain_update_info(target, true);
                            }    

                            auto & affinity_table_target_from_proc = thread_data.affinity_table.at(target)[move.from_proc];
                            const unsigned target_window_bound = end_idx(target_step, end_step);
                            const cost_t comm_aff = instance->communicationCosts(source_proc, move.from_proc) * comm_gain;
                            for (unsigned idx = start_idx(target_step, start_step); idx < target_window_bound; idx++) {
                                affinity_table_target_from_proc[idx] += comm_aff;
                            }
                        }
                    }                    
                } else if (node_lambda_map.get_proc_entry(source, move.from_proc) == 1)  {
                    const cost_t comm_gain = graph->vertex_comm_weight(source) * comm_multiplier;

                    for (const auto &target : instance->getComputationalDag().children(source)) {
                        const unsigned target_step = active_schedule->assigned_superstep(target);
                        if ((target_step < start_step || target_step > end_step) || (target == move.node) || thread_data.lock_manager.is_locked(target) || (not thread_data.affinity_table.is_selected(target)))  
                            continue;   

                        const unsigned target_proc = active_schedule->assigned_processor(target);
                        if (target_proc == move.from_proc) {      
                            if (max_gain_recompute.find(target) != max_gain_recompute.end()) { // todo more specialized update
                                max_gain_recompute[target].full_update = true;                
                            } else {
                                max_gain_recompute[target] = kl_gain_update_info(target, true);
                            } 
                            
                            const unsigned target_start_idx = start_idx(target_step, start_step);
                            const unsigned target_window_bound = end_idx(target_step, end_step);
                            auto & affinity_table_target = thread_data.affinity_table.at(target);
                            const cost_t comm_aff = instance->communicationCosts(source_proc, target_proc) * comm_gain;
                            for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                                if (p == target_proc)
                                    continue;      
                                
                                for (unsigned idx = target_start_idx; idx < target_window_bound; idx++) {
                                    affinity_table_target[p][idx] -= comm_aff;
                                } 
                            }
                            break; // since node_lambda_map[source][move.from_proc] == 1
                        }   
                    }                    
                }

                if (node_lambda_map.get_proc_entry(source, move.to_proc) == 1) {
                    const cost_t comm_gain = graph->vertex_comm_weight(source) * comm_multiplier;
                    
                    for (const auto &target : instance->getComputationalDag().children(source)) {
                        const unsigned target_step = active_schedule->assigned_superstep(target);
                        if ((target_step < start_step || target_step > end_step) || (target == move.node) || (not thread_data.affinity_table.is_selected(target)) || thread_data.lock_manager.is_locked(target))  
                            continue;   
                        
                        if (source_proc != move.to_proc && is_compatible(target, move.to_proc)) {
                            if (max_gain_recompute.find(target) != max_gain_recompute.end()) {
                                max_gain_recompute[target].full_update = true;                
                            } else {
                                max_gain_recompute[target] = kl_gain_update_info(target, true);
                            } 
                            
                            const unsigned target_window_bound = end_idx(target_step, end_step);
                            auto & affinity_table_target_to_proc = thread_data.affinity_table.at(target)[move.to_proc];
                            const cost_t comm_aff = instance->communicationCosts(source_proc, move.to_proc) * comm_gain;
                            for (unsigned idx = start_idx(target_step, start_step); idx < target_window_bound; idx++) {
                                affinity_table_target_to_proc[idx] -= comm_aff;
                            }                              
                        }
                    }
                } else if (node_lambda_map.get_proc_entry(source, move.to_proc) == 2) {  
                    for (const auto &target : instance->getComputationalDag().children(source)) {
                        const unsigned target_step = active_schedule->assigned_superstep(target);
                        if ((target_step < start_step || target_step > end_step) || (target == move.node) || (not thread_data.affinity_table.is_selected(target)) || thread_data.lock_manager.is_locked(target))  
                            continue; 
                        
                        const unsigned target_proc = active_schedule->assigned_processor(target);
                        if (target_proc == move.to_proc) {
                            if (source_proc != target_proc) {
                                if (max_gain_recompute.find(target) != max_gain_recompute.end()) {
                                    max_gain_recompute[target].full_update = true;                
                                } else {
                                    max_gain_recompute[target] = kl_gain_update_info(target, true);
                                } 
                               
                                const unsigned target_start_idx = start_idx(target_step, start_step);
                                const unsigned target_window_bound = end_idx(target_step, end_step);
                                auto & affinity_table_target = thread_data.affinity_table.at(target);
                                const cost_t comm_aff = instance->communicationCosts(source_proc, target_proc) * graph->vertex_comm_weight(source) * comm_multiplier;
                                for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                                    if (p == target_proc)
                                        continue;      
                                    
                                    for (unsigned idx = target_start_idx; idx < target_window_bound; idx++) {
                                        affinity_table_target[p][idx] += comm_aff;
                                    }                                         
                                }
                            }
                            break;
                        }   
                    }                  
                }                
            }


            const unsigned source_step = active_schedule->assigned_superstep(source); 
            if (source_step < start_step || source_step > end_step)
                continue;

            if(thread_data.lock_manager.is_locked(source)) 
                continue;            

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
            const unsigned source_start_idx = start_idx(source_step, start_step);
            const unsigned window_bound = end_idx(source_step, end_step);
            auto & affinity_table_source = thread_data.affinity_table.at(source);

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
        
            if (move.to_proc != move.from_proc) {   
                if (node_lambda_map.has_no_proc_entry(source, move.from_proc)) {                    
                    const cost_t comm_gain = graph->vertex_comm_weight(source) * comm_multiplier;

                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) {        
                        if (p == source_proc)
                            continue;

                        const cost_t comm_cost = change_comm_cost(instance->communicationCosts(p, move.from_proc), instance->communicationCosts(source_proc, move.from_proc), comm_gain);
                        for (unsigned idx = source_start_idx; idx < window_bound; idx++) {
                            affinity_table_source[p][idx] -= comm_cost;
                        }                        
                    }                  
                } 

                if (node_lambda_map.get_proc_entry(source, move.to_proc) == 1) {
                    const cost_t comm_gain = graph->vertex_comm_weight(source) * comm_multiplier;

                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) {        
                        if (p == source_proc)
                            continue;

                        const cost_t comm_cost = change_comm_cost(instance->communicationCosts(p, move.to_proc), instance->communicationCosts(source_proc, move.to_proc), comm_gain);
                        for (unsigned idx = source_start_idx; idx < window_bound; idx++) {
                            affinity_table_source[p][idx] += comm_cost;
                        }                 
                    }
                }                 
            }                
        }  
    }

    inline unsigned start_idx(const unsigned node_step, const unsigned start_step) { return node_step < window_size + start_step ? window_size - (node_step - start_step) : 0; }
    inline unsigned end_idx(const unsigned node_step, const unsigned end_step) { return node_step + window_size <= end_step ? window_range : window_range - (node_step + window_size - end_step); }
   
    inline cost_t change_comm_cost(const v_commw_t<Graph_t> &p_target_comm_cost, const v_commw_t<Graph_t> &node_target_comm_cost, const cost_t &comm_gain) { return p_target_comm_cost > node_target_comm_cost ? (p_target_comm_cost - node_target_comm_cost) * comm_gain : (node_target_comm_cost - p_target_comm_cost) * comm_gain * -1.0;}

    template<typename affinity_table_t>
    void compute_comm_affinity(VertexType node, affinity_table_t& affinity_table_node, const cost_t& penalty, const cost_t& reward, const unsigned start_step, const unsigned end_step) {
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
        } // traget

        const cost_t comm_gain = graph->vertex_comm_weight(node) * comm_multiplier;

        for (const unsigned p : proc_range->compatible_processors_vertex(node)) {        
            if (p == node_proc)
                continue;

            for (const auto [lambda_proc, mult] : node_lambda_map.iterate_proc_entries(node)) {
                const cost_t comm_cost = change_comm_cost(instance->communicationCosts(p, lambda_proc), instance->communicationCosts(node_proc, lambda_proc), comm_gain);
                for (unsigned idx = node_start_idx; idx < window_bound; idx++) {
                    affinity_table_node[p][idx] += comm_cost;
                }
            }
        }

        for (const auto &source : instance->getComputationalDag().parents(node)) {
            const unsigned source_step = active_schedule->assigned_superstep(source);
            const unsigned source_proc = active_schedule->assigned_processor(source);  

            if (source_step < node_step + (source_proc == node_proc)) {
                const unsigned diff = node_step - source_step;                
                const unsigned bound = window_size >= diff ? window_size - diff + 1: 0;  
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

            const cost_t source_comm_gain = graph->vertex_comm_weight(source) * comm_multiplier;
            for (const unsigned p : proc_range->compatible_processors_vertex(node)) { 
                if (p == node_proc)
                    continue;

                if (source_proc != node_proc && node_lambda_map.get_proc_entry(source, node_proc) == 1) {
                    for (unsigned idx = node_start_idx; idx < window_bound; idx++) {
                        affinity_table_node[p][idx] -= instance->communicationCosts(source_proc, node_proc) * source_comm_gain;
                    } 
                }

                if (source_proc != p && node_lambda_map.has_no_proc_entry(source, p)) {
                    for (unsigned idx = node_start_idx; idx < window_bound; idx++) {
                        affinity_table_node[p][idx] += instance->communicationCosts(source_proc, p) * source_comm_gain;
                    }
                }            
            }
        } // source
    }
};

} // namespace osp
