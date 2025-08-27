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

namespace osp {
template<typename Graph_t, typename cost_t, typename MemoryConstraint_t, unsigned window_size = 1, bool use_node_communication_costs_arg = true> 
struct kl_total_comm_cost_function {
    
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
    }

    cost_t compute_schedule_cost_test() {
        return compute_schedule_cost();
    }

    void update_datastructure_after_move(const kl_move& ) {}

    cost_t compute_schedule_cost() {

        cost_t work_costs = 0;
        for (unsigned step = 0; step < active_schedule->num_steps(); step++) {
            work_costs += active_schedule->get_step_max_work(step);
        }

        cost_t comm_costs = 0;
        for (const auto &edge : edges(*graph)) {

            const auto &source_v = source(edge, *graph);
            const auto &target_v = target(edge, *graph);

            const unsigned &source_proc = active_schedule->assigned_processor(source_v);
            const unsigned &target_proc = active_schedule->assigned_processor(target_v);

            if (source_proc != target_proc) {

                if constexpr (use_node_communication_costs) {
                    const cost_t source_comm_cost = graph->vertex_comm_weight(source_v); 
                    max_comm_weight = std::max(max_comm_weight, source_comm_cost);
                    comm_costs += source_comm_cost * instance->communicationCosts(source_proc, target_proc);
                } else {
                    const cost_t source_comm_cost = graph->edge_comm_weight(edge);
                    max_comm_weight = std::max(max_comm_weight, source_comm_cost);
                    comm_costs += source_comm_cost * instance->communicationCosts(source_proc, target_proc);
                }
            }
        }  

        return work_costs + comm_costs * comm_multiplier + static_cast<v_commw_t<Graph_t>>(active_schedule->num_steps() - 1) * instance->synchronisationCosts();
    }

    template<typename lock_manager_t, typename affinity_table_t>
    void update_node_comm_affinity(const kl_move &move, affinity_table_t& affinity_table, lock_manager_t &lock_manager, const cost_t& penalty, const cost_t& reward, std::map<VertexType, kl_gain_update_info> & max_gain_recompute, std::vector<VertexType> &new_nodes) {
                
        for (const auto &target : instance->getComputationalDag().children(move.node)) {

            if(lock_manager.is_locked(target))
                continue;

            if (not affinity_table.is_selected(target)) {
                new_nodes.push_back(target);  
                continue;
            }

            if (max_gain_recompute.find(target) != max_gain_recompute.end()) {
                max_gain_recompute[target].full_update = true;                
            } else {
                max_gain_recompute[target] = kl_gain_update_info(target, true);
            }           

            const unsigned target_proc = active_schedule->assigned_processor(target);
            const unsigned target_step = active_schedule->assigned_superstep(target);   
            const unsigned target_start_idx = start_idx(target_step);             
         
            if (move.from_step < target_step + (move.from_proc == target_proc)) {

                const unsigned diff = target_step - move.from_step;                
                const unsigned bound = window_size >= diff ? window_size - diff + 1: 0;  
                unsigned idx = target_start_idx; 
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) { 
                        affinity_table.at(target)[p][idx] -= penalty;
                    }                                                
                } 

                if (idx - 1 < bound && is_compatible(target, move.from_proc)) {
                    affinity_table.at(target)[move.from_proc][idx - 1] += penalty;    
                }

            } else {

                const unsigned diff = move.from_step - target_step;
                const unsigned window_bound = end_idx(target_step);  
                unsigned idx = std::min(window_size + diff, window_bound);                  
                
                if (idx < window_bound && is_compatible(target, move.from_proc)) { 
                    affinity_table.at(target)[move.from_proc][idx] += reward; 
                }

                idx++;
                
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) { 
                        affinity_table.at(target)[p][idx] += reward;
                    }                        
                } 
            }

            if (move.to_step < target_step + (move.to_proc == target_proc)) {
                unsigned idx = target_start_idx; 
                const unsigned diff = target_step - move.to_step;                
                const unsigned bound = window_size >= diff ? window_size - diff + 1: 0;  
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) { 
                        affinity_table.at(target)[p][idx] += penalty;
                    }                                                
                } 

                if (idx - 1 < bound && is_compatible(target, move.to_proc)) {
                    affinity_table.at(target)[move.to_proc][idx - 1] -= penalty;    
                }

            } else {
                const unsigned diff = move.to_step - target_step;
                const unsigned window_bound = end_idx(target_step); 
                unsigned idx = std::min(window_size + diff, window_bound);                                                     
                
                if (idx < window_bound && is_compatible(target, move.to_proc)) {
                    affinity_table.at(target)[move.to_proc][idx] -= reward; 
                }

                idx++;
                                    
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) { 
                        affinity_table.at(target)[p][idx] -= reward;
                    }                        
                } 
            }
        
            if (move.to_proc != move.from_proc) {                
                const auto from_proc_target_comm_cost = instance->communicationCosts(move.from_proc, target_proc);
                const auto to_proc_target_comm_cost = instance->communicationCosts(move.to_proc, target_proc);

                const cost_t comm_gain = graph->vertex_comm_weight(move.node) * comm_multiplier;

                unsigned idx = target_start_idx;
                const unsigned window_bound = end_idx(target_step);
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(target)) {
                        const auto x = change_comm_cost(instance->communicationCosts(p, move.to_proc), to_proc_target_comm_cost, comm_gain); 
                        const auto y = change_comm_cost(instance->communicationCosts(p, move.from_proc), from_proc_target_comm_cost, comm_gain);
                        affinity_table.at(target)[p][idx] += x - y;  
                    }
                }
            } 
        }

        for (const auto &source : instance->getComputationalDag().parents(move.node)) {

            if(lock_manager.is_locked(source))
                continue;

            if (not affinity_table.is_selected(source)) {
                new_nodes.push_back(source);
                continue;
            }

            if (max_gain_recompute.find(source) != max_gain_recompute.end()) {
                max_gain_recompute[source].full_update = true;                
            } else {
                max_gain_recompute[source] = kl_gain_update_info(source, true);
            } 

            const unsigned source_proc = active_schedule->assigned_processor(source);
            const unsigned source_step = active_schedule->assigned_superstep(source);    
            const unsigned window_bound = end_idx(source_step);

            if (move.from_step < source_step + (move.from_proc != source_proc)) {

                const unsigned diff = source_step - move.from_step; 
                const unsigned bound = window_size > diff ? window_size - diff : 0; 
                unsigned idx = start_idx(source_step);
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) {  
                        affinity_table.at(source)[p][idx] += reward;
                    } 
                }

                if (window_size >= diff && is_compatible(source, move.from_proc)) {
                    affinity_table.at(source)[move.from_proc][idx] += reward;    
                }

            } else {       

                const unsigned diff = move.from_step - source_step;
                unsigned idx = window_size + diff; 
                
                if (idx < window_bound && is_compatible(source, move.from_proc)) {
                    affinity_table.at(source)[move.from_proc][idx] += penalty;                        
                }

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) { 
                        affinity_table.at(source)[p][idx] -= penalty;
                    }                        
                }                     
            }

            if (move.to_step < source_step + (move.to_proc != source_proc)) {
                const unsigned diff = source_step - move.to_step; 
                const unsigned bound = window_size > diff ? window_size - diff : 0; 
                unsigned idx = start_idx(source_step);
                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) {  
                        affinity_table.at(source)[p][idx] -= reward;
                    } 
                }

                if (window_size >= diff && is_compatible(source, move.to_proc)) {
                    affinity_table.at(source)[move.to_proc][idx] -= reward;    
                }

            } else {  
                const unsigned diff = move.to_step - source_step;
                unsigned idx = window_size + diff; 

                if (idx < window_bound && is_compatible(source, move.to_proc)) {
                    affinity_table.at(source)[move.to_proc][idx] -= penalty;                         
                }
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) { 
                        affinity_table.at(source)[p][idx] += penalty;
                    }                        
                }                     
            }  

            if (move.to_proc != move.from_proc) {                
                const auto from_proc_source_comm_cost = instance->communicationCosts(source_proc, move.from_proc);
                const auto to_proc_source_comm_cost = instance->communicationCosts(source_proc, move.to_proc);

                const cost_t comm_gain = graph->vertex_comm_weight(source) * comm_multiplier;

                unsigned idx = start_idx(source_step);
                const unsigned window_bound = end_idx(source_step);
                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(source)) { 
                        const cost_t x = change_comm_cost(instance->communicationCosts(p, move.to_proc), to_proc_source_comm_cost, comm_gain); 
                        const cost_t y = change_comm_cost(instance->communicationCosts(p, move.from_proc), from_proc_source_comm_cost, comm_gain);
                        affinity_table.at(source)[p][idx] += x - y;  
                    }
                }
            }
        } 
    }

    inline unsigned start_idx(unsigned node_step) { return node_step < window_size ? window_size - node_step : 0; }
    inline unsigned end_idx(unsigned node_step) { return node_step + window_size < active_schedule->num_steps() ? window_range : window_range - (node_step + window_size + 1 - active_schedule->num_steps()); }

    inline cost_t change_comm_cost(const v_commw_t<Graph_t> &p_target_comm_cost, const v_commw_t<Graph_t> &node_target_comm_cost, const cost_t &comm_gain) { return p_target_comm_cost > node_target_comm_cost ? (p_target_comm_cost - node_target_comm_cost) * comm_gain : (node_target_comm_cost - p_target_comm_cost) * comm_gain * -1.0;}

    template<typename affinity_table_t>
    void compute_comm_affinity(VertexType node, affinity_table_t& affinity_table, const cost_t& penalty, const cost_t& reward) {
        const unsigned node_step = active_schedule->assigned_superstep(node);
        const unsigned node_proc = active_schedule->assigned_processor(node);
        const unsigned window_bound = end_idx(node_step);
        const unsigned node_start_idx = start_idx(node_step);

        for (const auto &target : instance->getComputationalDag().children(node)) {
            const unsigned target_step = active_schedule->assigned_superstep(target);
            const unsigned target_proc = active_schedule->assigned_processor(target); 

            if (target_step < node_step + (target_proc != node_proc)) {
                const unsigned diff = node_step - target_step; 
                const unsigned bound = window_size > diff ? window_size - diff : 0; 
                unsigned idx = node_start_idx;

                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(node)) {                        
                        affinity_table.at(node)[p][idx] -= reward;
                    } 
                }

                if (window_size >= diff && is_compatible(node, target_proc)) {
                    affinity_table.at(node)[target_proc][idx] -= reward;    
                }  

            } else {  
                const unsigned diff = target_step - node_step;
                unsigned idx = window_size + diff;

                if (idx < window_bound && is_compatible(node, target_proc)) {
                    affinity_table.at(node)[target_proc][idx] -= penalty; 
                }

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(node)) {                      
                        affinity_table.at(node)[p][idx] += penalty;
                    }                        
                }                     
            }    

            const cost_t comm_gain = graph->vertex_comm_weight(node) * comm_multiplier;
            const auto node_target_comm_cost = instance->communicationCosts(node_proc, target_proc);

            for (const unsigned p : proc_range->compatible_processors_vertex(node)) {
                const cost_t comm_cost = change_comm_cost(instance->communicationCosts(p, target_proc), node_target_comm_cost, comm_gain);
                for (unsigned idx = node_start_idx; idx < window_bound; idx++) {
                    affinity_table.at(node)[p][idx] += comm_cost;
                }
            }

        } // traget

        for (const auto &source : instance->getComputationalDag().parents(node)) {
            const unsigned source_step = active_schedule->assigned_superstep(source);
            const unsigned source_proc = active_schedule->assigned_processor(source);  

            if (source_step < node_step + (source_proc == node_proc)) {
                const unsigned diff = node_step - source_step;                
                const unsigned bound = window_size >= diff ? window_size - diff + 1: 0;  
                unsigned idx = node_start_idx;

                for (; idx < bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(node)) {                        
                        affinity_table.at(node)[p][idx] += penalty; 
                    }                                                
                }

                if (idx - 1 < bound && is_compatible(node, source_proc)) {
                    affinity_table.at(node)[source_proc][idx - 1] -= penalty;    
                }

            } else {
                const unsigned diff = source_step - node_step;
                unsigned idx = std::min(window_size + diff, window_bound);

                if (idx < window_bound && is_compatible(node, source_proc)) {
                    affinity_table.at(node)[source_proc][idx] -= reward;  
                }
                
                idx++;

                for (; idx < window_bound; idx++) {
                    for (const unsigned p : proc_range->compatible_processors_vertex(node)) {                        
                        affinity_table.at(node)[p][idx] -= reward;
                    }                        
                } 
            }

            const cost_t comm_gain = graph->vertex_comm_weight(source) * comm_multiplier;
            const auto source_node_comm_cost = instance->communicationCosts(source_proc, node_proc);

            for (const unsigned p : proc_range->compatible_processors_vertex(node)) {   
                const cost_t comm_cost = change_comm_cost(instance->communicationCosts(p, source_proc), source_node_comm_cost, comm_gain);
                for (unsigned idx = node_start_idx; idx < window_bound; idx++) {
                    affinity_table.at(node)[p][idx] += comm_cost;
                }
            }
        } // source
    }
};

} // namespace osp

