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
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/heap/fibonacci_heap.hpp>
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"

#include "kl_active_schedule.hpp"
#include "kl_util.hpp"

namespace osp {

struct kl_parameter {

    unsigned max_inner_iterations = 150;
    unsigned max_outer_iterations = 100;

    unsigned max_no_improvement_iterations = 25; 

    double gain_threshold = -10.0;
    double change_in_cost_threshold = 0.0;

    bool quick_pass = false;

    unsigned max_step_selection_epochs = 4;
    unsigned reset_epoch_counter_threshold = 10;

    unsigned min_inner_iter = 30; // log(n)
};

template<typename VertexType>
struct kl_update_info {

    VertexType node = 0;

    bool full_update = false;
    bool update_from_step = false;
    bool update_to_step = false;
    bool update_entire_to_step = false;
    bool update_entire_from_step = false;

    kl_update_info() = default;
    kl_update_info(VertexType n) : node(n), full_update(false), update_entire_to_step(false), update_entire_from_step(false) {}
    kl_update_info(VertexType n, bool full) : node(n), full_update(full), update_entire_to_step(false), update_entire_from_step(false) {}
};

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t = no_local_search_memory_constraint,
         unsigned window_size = 1, typename cost_t = double>
class kl_improver : public ImprovementScheduler<Graph_t> {

    static_assert(is_directed_graph_edge_desc_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(has_hashable_edge_desc_v<Graph_t>, "Graph_t must satisfy the has_hashable_edge_desc concept");
    static_assert(is_computational_dag_v<Graph_t>, "Graph_t must satisfy the computational_dag concept");

  protected:

    constexpr static unsigned window_range = 2 * window_size + 1;

    using memw_t = v_memw_t<Graph_t>;
    using commw_t = v_commw_t<Graph_t>;
    using work_weight_t = v_workw_t<Graph_t>;
    using VertexType = vertex_idx_t<Graph_t>;

    using kl_move = kl_move_struct<cost_t, VertexType>;
    using heap_datastructure = typename boost::heap::fibonacci_heap<kl_move>;
    using heap_handle = typename heap_datastructure::handle_type;
    using active_schedule_t = kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>;
    using node_selection_container_t = std::unordered_map<VertexType, heap_handle>;
    using kl_gain_update_info = kl_update_info<VertexType>;

    bool compute_with_time_limit = false;

    BspSchedule<Graph_t> *input_schedule;
    const Graph_t *graph;
    const BspInstance<Graph_t> *instance;

    compatible_processor_range<Graph_t> proc_range;

    kl_parameter parameters;
    std::mt19937 gen;
    
    active_schedule_t active_schedule;
    
    set_vertex_lock_manger<VertexType> lock_manager;
    vertex_selection_strategy<Graph_t, node_selection_container_t, heap_handle> selection_strategy;

    comm_cost_function_t comm_cost_f;
    reward_penalty_strategy<cost_t, comm_cost_function_t, active_schedule_t> reward_penalty_strat;

    heap_datastructure max_gain_heap;
    node_selection_container_t node_selection;

    std::vector<std::vector<std::vector<cost_t>>> affinity_table;

    double average_gain = 0.0;
    unsigned no_improvement_iterations_reduce_penalty;

    // unsigned step_selection_counter = 0;
    // unsigned step_selection_epoch_counter = 0;

    // bool auto_alternate = false;
    // bool alternate_reset_remove_superstep = false;
    // bool reset_superstep = false;

    inline unsigned start_idx(unsigned node_step) { return node_step < window_size ? window_size - node_step : 0; }
    inline unsigned rel_step_idx(const unsigned node_step, const unsigned move_step) { return (move_step >= node_step) ? ((move_step - node_step) + window_size) : (window_size - (node_step - move_step)); }
    inline unsigned end_idx(unsigned node_step) { return node_step + window_size < active_schedule.num_steps() ? window_range : window_range - (node_step + window_size + 1 - active_schedule.num_steps()); }

    kl_move get_best_move() {

        const unsigned local_max = 50;
        std::vector<VertexType> max_nodes(local_max);
        unsigned count = 0;
        for (auto iter = max_gain_heap.ordered_begin(); iter != max_gain_heap.ordered_end(); ++iter) {

            if (iter->gain == max_gain_heap.top().gain && count < local_max) {
                max_nodes[count] = (iter->node);
                count++;
            } else {
                break;
            }
        }

        std::uniform_int_distribution<unsigned> dis(0, count - 1);
        unsigned i = dis(gen);

        const VertexType node = max_nodes[i];

        kl_move best_move = (*node_selection[node]);
        max_gain_heap.erase(node_selection[node]);
        lock_manager.lock(node);
        node_selection.erase(node);

#ifdef KL_DEBUG
 //       std::cout << "Best move: " << best_move.node << " gain: " << best_move.gain << ", from step|proc: " << best_move.from_step << "|" << best_move.from_proc << " to: " << best_move.to_step << "|" << best_move.to_proc << std::endl;
#endif

        return best_move;
    }
    
    inline void process_other_steps_best_move(const unsigned idx, const VertexType& node,const unsigned& node_proc, cost_t& max_gain, unsigned& max_proc, unsigned& max_step) {
    
        for (const unsigned p : proc_range.compatible_processors_vertex(node)) {
            const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][p][idx];
            if (gain > max_gain) {
                max_gain = gain;
                max_proc = p;
                max_step = idx; 
            }
        }
    }

    template<bool change_super_step>
    kl_move compute_best_move(VertexType node) {

        const unsigned node_step = active_schedule.assigned_superstep(node);
        const unsigned node_proc = active_schedule.assigned_processor(node);

        cost_t max_gain = std::numeric_limits<cost_t>::lowest();

        unsigned max_proc = std::numeric_limits<unsigned>::max();
        unsigned max_step = std::numeric_limits<unsigned>::max();

        unsigned idx = start_idx(node_step);
        for (; idx < window_size; idx++) {
            process_other_steps_best_move(idx, node, node_proc, max_gain, max_proc, max_step);
        }

        if constexpr (change_super_step) {

            for (const unsigned proc : proc_range.compatible_processors_vertex(node)) { 

                if (proc == node_proc)
                    continue;

                const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][proc][window_size];
                if (gain > max_gain) {
                    max_gain = gain;
                    max_proc = proc;
                    max_step = idx; 
                }
            }
        }

        idx++;

        const unsigned bound = end_idx(node_step);
        for (; idx < bound; idx++) {
            process_other_steps_best_move(idx, node, node_proc, max_gain, max_proc, max_step);
        }

        return kl_move(node, max_gain, node_proc, node_step, max_proc, node_step + max_step - window_size);
    }
    
 

    kl_gain_update_info update_node_work_affinity_after_move(VertexType node, kl_move move, const pre_move_work_data<work_weight_t> & prev_work_data) {

        const unsigned node_step = active_schedule.assigned_superstep(node);
        const work_weight_t vertex_weight = graph->vertex_work_weight(node);

        kl_gain_update_info update_info(node);

        if (move.from_step == move.to_step) {
        
            const unsigned lower_bound = move.from_step > window_size ? move.from_step - window_size : 0; 
            if (lower_bound <= node_step && node_step <= move.from_step + window_size) {

                update_info.update_from_step = true;
                update_info.update_to_step = true;

                const work_weight_t prev_max_work = prev_work_data.from_step_max_work;
                const work_weight_t prev_second_max_work = prev_work_data.from_step_second_max_work;

                if (node_step == move.from_step) {

                    const unsigned node_proc = active_schedule.assigned_processor(node);
                    const work_weight_t new_max_weight = active_schedule.get_step_max_work(move.from_step);   
                    const work_weight_t new_second_max_weight = active_schedule.get_step_second_max_work(move.from_step);
                    const work_weight_t new_step_proc_work = active_schedule.get_step_processor_work(node_step, node_proc);

                    const work_weight_t prev_step_proc_work = (node_proc == move.from_proc) ? new_step_proc_work + graph->vertex_work_weight(move.node) : (node_proc == move.to_proc) ? new_step_proc_work - graph->vertex_work_weight(move.node) : new_step_proc_work;                                               
                    const bool prev_is_sole_max_processor = (prev_work_data.from_step_max_work_processor_count == 1) && (prev_max_work == prev_step_proc_work);
                    const cost_t prev_node_proc_affinity = prev_is_sole_max_processor ? std::min(vertex_weight, prev_max_work - prev_second_max_work) : 0.0;

                    const bool new_is_sole_max_processor = (active_schedule.get_step_max_work_processor_count()[node_step] == 1) && (new_max_weight == new_step_proc_work);
                    const cost_t new_node_proc_affinity = new_is_sole_max_processor ? std::min(vertex_weight, new_max_weight - new_second_max_weight) : 0.0;                        
                    
                    if (new_node_proc_affinity != prev_node_proc_affinity) {
                        update_info.full_update = true;
                        affinity_table[node][node_proc][window_size] += (new_node_proc_affinity - prev_node_proc_affinity);                    
                    }     
                    
                    if ((prev_max_work != new_max_weight) || update_info.full_update) {

                        update_info.update_entire_from_step = true;

                        for (const unsigned proc : proc_range.compatible_processors_vertex(node)) { 

                            if((proc == node_proc) || (proc == move.from_proc) || (proc == move.to_proc)) {
                                continue;
                            }

                            const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, proc);
                            const cost_t prev_other_affinity = compute_same_step_affinity(prev_max_work, new_weight, prev_node_proc_affinity);   
                            const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);             
                               
                            affinity_table[node][proc][window_size] += (other_affinity - prev_other_affinity);                            
                        }
                    }  
                    
                    if (node_proc != move.from_proc) {

                        const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move.from_proc) + graph->vertex_work_weight(move.node);
                        const cost_t prev_other_affinity = compute_same_step_affinity(prev_max_work, prev_new_weight, prev_node_proc_affinity);   
                        const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move.from_proc);
                        const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);           
          
                        affinity_table[node][move.from_proc][window_size] += (other_affinity - prev_other_affinity);  
                    } 
                    
                    if (node_proc != move.to_proc) {

                        const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move.to_proc) - graph->vertex_work_weight(move.node);
                        const cost_t prev_other_affinity = compute_same_step_affinity(prev_max_work, prev_new_weight, prev_node_proc_affinity);      
                        const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move.to_proc);
                        const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);            
            
                        affinity_table[node][move.to_proc][window_size] += (other_affinity - prev_other_affinity);  
                    }

                } else {
                    
                    const work_weight_t new_max_weight = active_schedule.get_step_max_work(move.from_step);
                    const unsigned idx = rel_step_idx(node_step, move.from_step); 
                    if (prev_max_work != new_max_weight) {

                        update_info.update_entire_from_step = true;
                        
                        // update moving to all procs with special for move.from_proc
                        for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                            
                            const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(move.from_step, proc);
                            if (proc == move.from_proc) {
                                const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(move.from_step, proc) + graph->vertex_work_weight(move.node);
                                const work_weight_t prev_affinity = prev_max_work < prev_new_weight ? prev_new_weight - prev_max_work : 0;

                                const work_weight_t new_affinity = new_max_weight < new_weight ? new_weight - new_max_weight : 0;
                                affinity_table[node][proc][idx] += new_affinity - prev_affinity;
                                                            
                            } else if (proc == move.to_proc) {
                                const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(move.to_step, proc) - graph->vertex_work_weight(move.node);
                                const work_weight_t prev_affinity = prev_max_work < prev_new_weight ? prev_new_weight - prev_max_work : 0;
                                const work_weight_t new_affinity = new_max_weight < new_weight ? new_weight - new_max_weight : 0;
                                affinity_table[node][proc][idx] += new_affinity - prev_affinity;

                            } else {

                                const work_weight_t prev_affinity = prev_max_work < new_weight ? new_weight - prev_max_work : 0;
                                const work_weight_t new_affinity = new_max_weight < new_weight ? new_weight - new_max_weight : 0;
                                affinity_table[node][proc][idx] += new_affinity - prev_affinity;  
                            }
                        }
                            
                    } else {

                        // update only move.from_proc and move.to_proc
                        const work_weight_t from_new_weight = vertex_weight + active_schedule.get_step_processor_work(move.from_step, move.from_proc);
                        const work_weight_t from_prev_new_weight = from_new_weight + graph->vertex_work_weight(move.node);
                        const work_weight_t from_prev_affinity = prev_max_work < from_prev_new_weight ? from_prev_new_weight - prev_max_work : 0;

                        const work_weight_t from_new_affinity = new_max_weight < from_new_weight ? from_new_weight - new_max_weight : 0;
                        affinity_table[node][move.from_proc][idx] += from_new_affinity - from_prev_affinity;

                        const work_weight_t to_new_weight = vertex_weight + active_schedule.get_step_processor_work(move.to_step, move.to_proc);
                        const work_weight_t to_prev_new_weight = to_new_weight - graph->vertex_work_weight(move.node);
                        const work_weight_t to_prev_affinity = prev_max_work < to_prev_new_weight ? to_prev_new_weight - prev_max_work : 0;

                        const work_weight_t to_new_affinity = new_max_weight < to_new_weight ? to_new_weight - new_max_weight : 0;
                        affinity_table[node][move.to_proc][idx] += to_new_affinity - to_prev_affinity;
                    }
                }
            }
            
        } else {
            
            const unsigned node_proc = active_schedule.assigned_processor(node);
            process_work_update_step(node, node_step, node_proc, vertex_weight, move.from_step, move.from_proc, graph->vertex_work_weight(move.node), prev_work_data.from_step_max_work, prev_work_data.from_step_second_max_work, prev_work_data.from_step_max_work_processor_count, update_info.update_from_step, update_info.update_entire_from_step, update_info.full_update);
            process_work_update_step(node, node_step, node_proc, vertex_weight, move.to_step, move.to_proc, -graph->vertex_work_weight(move.node), prev_work_data.to_step_max_work, prev_work_data.to_step_second_max_work, prev_work_data.to_step_max_work_processor_count, update_info.update_to_step, update_info.update_entire_to_step, update_info.full_update);

        }

        return update_info;
    }


    void process_work_update_step(VertexType node, unsigned node_step, unsigned node_proc, work_weight_t vertex_weight, unsigned move_step, unsigned move_proc, work_weight_t move_correction_node_weight, const work_weight_t prev_move_step_max_work, const work_weight_t prev_move_step_second_max_work, unsigned prev_move_step_max_work_processor_count, bool & update_step, bool & update_entire_step, bool & full_update) {

        const unsigned lower_bound = move_step > window_size ? move_step - window_size : 0; 
        if (lower_bound <= node_step && node_step <= move_step + window_size) {

            update_step = true;

            if (node_step == move_step) {
                
                const work_weight_t new_max_weight = active_schedule.get_step_max_work(move_step);   
                const work_weight_t new_second_max_weight = active_schedule.get_step_second_max_work(move_step);
                const work_weight_t new_step_proc_work = active_schedule.get_step_processor_work(node_step, node_proc);

                const work_weight_t prev_step_proc_work = (node_proc == move_proc) ? new_step_proc_work + move_correction_node_weight : new_step_proc_work;
                const bool prev_is_sole_max_processor = (prev_move_step_max_work_processor_count == 1) && (prev_move_step_max_work == prev_step_proc_work);
                const cost_t prev_node_proc_affinity = prev_is_sole_max_processor ? std::min(vertex_weight, prev_move_step_max_work - prev_move_step_second_max_work) : 0.0;

                const bool new_is_sole_max_processor = (active_schedule.get_step_max_work_processor_count()[node_step] == 1) && (new_max_weight == new_step_proc_work);
                const cost_t new_node_proc_affinity = new_is_sole_max_processor ? std::min(vertex_weight, new_max_weight - new_second_max_weight) : 0.0;
               
                const bool update_node_proc_affinity = new_node_proc_affinity != prev_node_proc_affinity;
                if (update_node_proc_affinity) {
                    full_update = true;
                    affinity_table[node][node_proc][window_size] += (new_node_proc_affinity - prev_node_proc_affinity);
                }
      
                if ((prev_move_step_max_work != new_max_weight) || update_node_proc_affinity) {

                    update_entire_step = true;

                    for (const unsigned proc : proc_range.compatible_processors_vertex(node)) { 

                        if((proc == node_proc) || (proc == move_proc))
                            continue;

                        const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, proc);
                        const cost_t prev_other_affinity = compute_same_step_affinity(prev_move_step_max_work, new_weight, prev_node_proc_affinity);  
                        const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);            
        
                        affinity_table[node][proc][window_size] += (other_affinity - prev_other_affinity);                             
                    }
                }
                
                if (node_proc != move_proc) {

                    const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move_proc) + move_correction_node_weight;
                    const cost_t prev_other_affinity = compute_same_step_affinity(prev_move_step_max_work, prev_new_weight, prev_node_proc_affinity);  
                    const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move_proc);
                    const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);           
        
                    affinity_table[node][move_proc][window_size] += (other_affinity - prev_other_affinity); 
                }        

            } else {

                const work_weight_t new_max_weight = active_schedule.get_step_max_work(move_step);
                const unsigned idx = rel_step_idx(node_step, move_step);
                if (prev_move_step_max_work != new_max_weight) {
                    
                    update_entire_step = true;

                    // update moving to all procs with special for move_proc
                    for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                        
                        const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(move_step, proc);
                        if (proc != move_proc) {

                            const work_weight_t prev_affinity = prev_move_step_max_work < new_weight ? new_weight - prev_move_step_max_work : 0;
                            const work_weight_t new_affinity = new_max_weight < new_weight ? new_weight - new_max_weight : 0;
                            affinity_table[node][proc][idx] += new_affinity - prev_affinity;  

                        } else {
                            const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(move_step, proc) + move_correction_node_weight;
                            const work_weight_t prev_affinity = prev_move_step_max_work < prev_new_weight ? prev_new_weight - prev_move_step_max_work : 0;

                            const work_weight_t new_affinity = new_max_weight < new_weight ? new_weight - new_max_weight : 0;
                            affinity_table[node][proc][idx] += new_affinity - prev_affinity;
                        }
                    }
                        
                } else {

                    // update only move_proc
                    const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(move_step, move_proc);
                    const work_weight_t prev_new_weight = new_weight + move_correction_node_weight;
                    const work_weight_t prev_affinity = prev_move_step_max_work < prev_new_weight ? prev_new_weight - prev_move_step_max_work : 0;

                    const work_weight_t new_affinity = new_max_weight < new_weight ? new_weight - new_max_weight : 0;
                    affinity_table[node][move_proc][idx] += new_affinity - prev_affinity;
                }
            }
        }
    }


    std::map<VertexType, kl_gain_update_info> update_node_work_affinity(const node_selection_container_t &nodes, kl_move move, const pre_move_work_data<work_weight_t> & prev_work_data) {

        std::map<VertexType, kl_gain_update_info> recompute_max_gain;

        for (const auto& pair : nodes) { 
            kl_gain_update_info update_info = update_node_work_affinity_after_move(pair.first, move, prev_work_data);
            if (update_info.update_from_step || update_info.update_to_step) {
                recompute_max_gain[pair.first] = update_info;
            }        
        }
        return recompute_max_gain;
    }

    void update_best_move(VertexType node, unsigned step, unsigned proc, node_selection_container_t &nodes) {
        
        const unsigned node_proc = active_schedule.assigned_processor(node);
        const unsigned node_step = active_schedule.assigned_superstep(node);

        if((node_proc == proc) && (node_step == step)) 
            return;

        const heap_handle & node_handle = node_selection[node];
        cost_t max_gain = (*node_handle).gain;
        
        unsigned max_proc = (*node_handle).to_proc;
        unsigned max_step = (*node_handle).to_step;

        if ((max_step == step) && (max_proc == proc)) {

            recompute_node_max_gain(node, nodes);

        } else {

            const unsigned idx = rel_step_idx(node_step, step);
            const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][proc][idx];
            if (gain > max_gain) {
                max_gain = gain;
                max_proc = proc;
                max_step = step; 
            } 
        
            if ((max_gain != (*node_handle).gain) || (max_proc != (*node_handle).to_proc) || (max_step != (*node_handle).to_step)) {
                (*node_handle).gain = max_gain;
                (*node_handle).to_proc = max_proc;
                (*node_handle).to_step = max_step;
                max_gain_heap.update(node_handle);
            }        
        }
    }
    
    void update_best_move(VertexType node, unsigned step, node_selection_container_t &nodes) {
       
        const unsigned node_proc = active_schedule.assigned_processor(node);
        const unsigned node_step = active_schedule.assigned_superstep(node);

        const heap_handle & node_handle = node_selection[node];
        cost_t max_gain = (*node_handle).gain;
        
        unsigned max_proc = (*node_handle).to_proc;
        unsigned max_step = (*node_handle).to_step;

        if (max_step == step) {

            recompute_node_max_gain(node, nodes);            

        } else {
         
            if (node_step != step) {

                const unsigned idx = rel_step_idx(node_step, step);
                for (const unsigned p : proc_range.compatible_processors_vertex(node)) {
                    
                    const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][p][idx];                    
                    if (gain > max_gain) {
                        max_gain = gain;
                        max_proc = p;
                        max_step = step; 
                    }
                }
            } else {

                for (const unsigned proc : proc_range.compatible_processors_vertex(node)) { 

                    if (proc == node_proc)
                        continue;

                    const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][proc][window_size];
                    if (gain > max_gain) {
                        max_gain = gain;
                        max_proc = proc;
                        max_step = step; 
                    }
                }
            }        

            if ((max_gain != (*node_handle).gain) || (max_proc != (*node_handle).to_proc) || (max_step != (*node_handle).to_step)) {
                (*node_handle).gain = max_gain;
                (*node_handle).to_proc = max_proc;
                (*node_handle).to_step = max_step;
                max_gain_heap.update(node_handle);
            }       
        } 
    }    

    inline void recompute_node_max_gain(VertexType node, node_selection_container_t &nodes) {

        const auto best_move = compute_best_move<true>(node);
        heap_handle & node_handle = nodes[node];
        (*node_handle).gain = best_move.gain;
        (*node_handle).to_proc = best_move.to_proc;
        (*node_handle).to_step = best_move.to_step;
        max_gain_heap.update(node_handle);
    }

    void update_max_gain(kl_move move, std::map<VertexType, kl_gain_update_info> &recompute_max_gain, node_selection_container_t &nodes) {
    
        for (auto& pair : recompute_max_gain) { 

            if (pair.second.full_update) {

                recompute_node_max_gain(pair.first, nodes); 

            } else {

                if (pair.second.update_entire_from_step) {
                    update_best_move(pair.first, move.from_step, nodes);
                } else if (pair.second.update_from_step) {
                    update_best_move(pair.first, move.from_step, move.from_proc, nodes);
                } 

                if (move.from_step != move.to_step || not pair.second.update_entire_from_step) {

                    if (pair.second.update_entire_to_step) {
                        update_best_move(pair.first, move.to_step, nodes);
                    } else if (pair.second.update_to_step) {
                        update_best_move(pair.first, move.to_step, move.to_proc, nodes);
                    }
                }
            } 
        }    
    }

    void compute_work_affinity(VertexType node) {

        const unsigned node_step = active_schedule.assigned_superstep(node);
        const work_weight_t vertex_weight = graph->vertex_work_weight(node);

        unsigned step = (node_step > window_size) ? (node_step - window_size) : 0;
        for (unsigned idx = start_idx(node_step); idx < end_idx(node_step); ++idx, ++step) {

            if (idx == window_size) {
                continue;
            }

            const work_weight_t max_work_for_step = active_schedule.get_step_max_work(step);

            for (const unsigned proc : proc_range.compatible_processors_vertex(node)) { 
                const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(step, proc);
                affinity_table[node][proc][idx] = std::max(0, new_weight - max_work_for_step);
            }
        }

        const unsigned node_proc = active_schedule.assigned_processor(node);        
        const work_weight_t max_work_for_step = active_schedule.get_step_max_work(node_step);
        const bool is_sole_max_processor = (active_schedule.get_step_max_work_processor_count()[node_step] == 1) && (max_work_for_step == active_schedule.get_step_processor_work(node_step, node_proc));

        const cost_t node_proc_affinity = is_sole_max_processor ? std::min(vertex_weight, max_work_for_step - active_schedule.get_step_second_max_work(node_step)) : 0.0;
        affinity_table[node][node_proc][window_size] = node_proc_affinity;
        
        for (const unsigned proc : proc_range.compatible_processors_vertex(node)) { 

            if(proc == node_proc)
                continue;

            const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, proc);           
            affinity_table[node][proc][window_size] = compute_same_step_affinity(max_work_for_step, new_weight, node_proc_affinity);            
        }
    }   

    inline cost_t compute_same_step_affinity(const work_weight_t &max_work_for_step, const work_weight_t &new_weight, const cost_t &node_proc_affinity) {

        const cost_t max_work_after_removal = max_work_for_step - node_proc_affinity;
        if (new_weight > max_work_after_removal) {
            return new_weight - max_work_after_removal;
        }
        return 0.0;
    }
    
    inline void apply_move(kl_move move) {

        active_schedule.apply_move(move);

        cost_t change_in_cost = -move.gain;

        change_in_cost += static_cast<cost_t>(active_schedule.resolved_violations.size()) * reward_penalty_strat.reward;
        change_in_cost -= static_cast<cost_t>(active_schedule.new_violations.size()) * reward_penalty_strat.penalty;
  
#ifdef KL_DEBUG

        // std::cout << "penalty: " << reward_penalty_strat.penalty << " num new violations: " << active_schedule.new_violations.size() << ", num resolved violations: " << active_schedule.resolved_violations.size() <<  ", reward: " << reward_penalty_strat.reward << std::endl;
        // std::cout << "apply move, previous cost: " << active_schedule.get_cost() << ", new cost: " << active_schedule.get_cost() + change_in_cost << ", " << (active_schedule.is_feasible() ? "feasible," : "infeasible,") << " computed cost: " << comm_cost_f.compute_schedule_cost() << std::endl;
        // if (std::abs(comm_cost_f.compute_schedule_cost() - (active_schedule.get_cost() + change_in_cost)) > 0.00001 ) {
        //     std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
        // }

#endif

        active_schedule.update_cost(change_in_cost);
    }    

    bool run_local_search() {

        const auto start_time = std::chrono::high_resolution_clock::now();
        cost_t initial_cost = active_schedule.get_cost();
        unsigned no_improvement_iter_counter = 0;

        for (unsigned outer_iter = 0; outer_iter < parameters.max_outer_iterations; outer_iter++) {

            reset_inner_search_structures();
            
            selection_strategy.select_active_nodes(node_selection);

            reward_penalty_strat.init_reward_penalty();

// #ifdef KL_DEBUG
//             std::cout << "------ start inner loop ------" << std::endl;
// #endif

            insert_gain_heap(node_selection);

            cost_t initial_inner_iter_cost = active_schedule.get_cost();
            unsigned inner_iter = 0;

            while (inner_iter < parameters.max_inner_iterations && max_gain_heap.size() > 0) {

                kl_move best_move = get_best_move(); // locks best_move.node and removes it from node_selection

                update_avg_gain(best_move.gain, inner_iter);
                if (inner_iter > parameters.min_inner_iter && average_gain < 0.0) {
#ifdef KL_DEBUG
                    std::cout << "Negative average gain: " << average_gain << ", end local search" << std::endl;
#endif
                    break;
                }

                const auto prev_work_data = active_schedule.get_pre_move_work_data(best_move);
                apply_move(best_move);

                if(is_local_search_blocked()) {
#ifdef KL_DEBUG
                    std::cout << "Nodes of violated edge locked, end local search" << std::endl;
#endif
                    break;  //or reset local search and initalize with violating nodes
                }

                std::map<VertexType, kl_gain_update_info> recompute_max_gain = update_node_work_affinity(node_selection, best_move, prev_work_data);
                auto new_nodes = comm_cost_f.update_node_comm_affinity(best_move, node_selection, lock_manager, reward_penalty_strat.penalty, reward_penalty_strat.reward, recompute_max_gain);

                update_max_gain(best_move, recompute_max_gain, node_selection);
                insert_new_nodes_gain_heap(new_nodes, node_selection);

                inner_iter++;
            }

#ifdef KL_DEBUG
            std::cout << "--- end inner loop " << outer_iter << "/" << parameters.max_outer_iterations << ", current cost: " << active_schedule.get_cost() << ", " << (active_schedule.is_feasible() ? "feasible" : "infeasible") << std::endl;
#endif

            active_schedule.revert_to_best_schedule();

            if (compute_with_time_limit) {
                auto finish_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();
                if (duration > ImprovementScheduler<Graph_t>::timeLimitSeconds) {
                    break;
                }
            }
  
            if (initial_inner_iter_cost <= active_schedule.get_cost()) {
                no_improvement_iter_counter++;

                if (no_improvement_iter_counter >= parameters.max_no_improvement_iterations) {
#ifdef KL_DEBUG
                    std::cout << "no improvement for " << parameters.max_no_improvement_iterations
                              << " iterations, end local search" << std::endl;
#endif
                    break;
                }

                adjust_local_search_parameters(no_improvement_iter_counter);
            
            } else {
                no_improvement_iter_counter = 0;
            }

            
        }


#ifdef KL_DEBUG
        std::cout << "local search end, current cost: " << active_schedule.get_cost() << " vs initial cost: " << initial_cost << std::endl;
#endif
        if (initial_cost > active_schedule.get_cost())
            return true;
        else
            return false;
    }

    inline void adjust_local_search_parameters(unsigned no_imp_counter) { 
        
                if (no_imp_counter >= no_improvement_iterations_reduce_penalty ) {

                    reward_penalty_strat.initial_penalty = 0;
                    //no_improvement_iterations_reduce_penalty += no_improvement_iterations_reduce_penalty + 5;
#ifdef KL_DEBUG
                    std::cout << "no improvement for " << no_improvement_iterations_reduce_penalty
                              << " iterations, reducing initial penalty to " << reward_penalty_strat.initial_penalty << std::endl;
#endif
                   
                } 

    }
    
    bool is_local_search_blocked();
    void set_parameters();
    void reset_inner_search_structures();
    void initialize_datastructures(BspSchedule<Graph_t> &schedule);
    void print_heap();
    void cleanup_datastructures();
    void update_avg_gain(cost_t gain, unsigned num_iter); 
    void insert_gain_heap(node_selection_container_t &nodes);
    void insert_new_nodes_gain_heap(std::vector<VertexType>& new_nodes, node_selection_container_t &nodes);

    // void checkMergeSupersteps();
    // void checkInsertSuperstep();
    // void insertSuperstep(unsigned step);

  public:
    kl_improver() : ImprovementScheduler<Graph_t>() {

        no_improvement_iterations_reduce_penalty = parameters.max_no_improvement_iterations / 5;

        std::random_device rd;
        gen = std::mt19937(rd());
    }

    virtual ~kl_improver() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule<Graph_t> &schedule) override {

        initialize_datastructures(schedule);
        set_parameters();
        
        bool improvement_found = run_local_search();

        if (improvement_found) {

            active_schedule.write_schedule(schedule);
            cleanup_datastructures();

            return RETURN_STATUS::OSP_SUCCESS;
        
        } else {

            cleanup_datastructures();
            return RETURN_STATUS::BEST_FOUND;
        }
    }

    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule<Graph_t> &schedule) override {
        compute_with_time_limit = true;
        return improveSchedule(schedule);
    }

    void set_compute_with_time_limit(bool compute_with_time_limit_) {
        compute_with_time_limit = compute_with_time_limit_;
    }

    virtual std::string getScheduleName() const {
        return "kl_improver_" + comm_cost_f.name();
    }

    void set_quick_pass(bool quick_pass_) { parameters.quick_pass = quick_pass_; }

    // void set_alternate_reset_remove_superstep(bool alternate_reset_remove_superstep_) {
    //     auto_alternate = false;
    //     alternate_reset_remove_superstep = alternate_reset_remove_superstep_;
    // }
};


template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::set_parameters() {

    vertex_idx_t<Graph_t> num_nodes = graph->num_vertices();

    parameters.min_inner_iter = static_cast<unsigned>(std::log(num_nodes) + 10.0);

    if (num_nodes < 250) {

        parameters.max_outer_iterations = 300;
    
        //parameters.selection_threshold = num_nodes / 3;

    } else if (num_nodes < 1000) {

        parameters.max_outer_iterations = static_cast<unsigned>(num_nodes / 2);
    
        //parameters.selection_threshold = num_nodes / 3;

    } else if (num_nodes < 5000) {

        parameters.max_outer_iterations = 4 * static_cast<unsigned>(std::sqrt(num_nodes));

        //parameters.selection_threshold = num_nodes / 3;

    } else if (num_nodes < 10000) {

        parameters.max_outer_iterations = 3 * static_cast<unsigned>(std::sqrt(num_nodes));

        //parameters.selection_threshold = num_nodes / 3;

    } else if (num_nodes < 50000) {

        parameters.max_outer_iterations = static_cast<unsigned>(std::sqrt(num_nodes));

        //parameters.selection_threshold = num_nodes / 5;

    } else if (num_nodes < 100000) {

        parameters.max_outer_iterations = 2 * static_cast<unsigned>(std::log(num_nodes));

        //parameters.selection_threshold = num_nodes / 10;

    } else {

        parameters.max_outer_iterations = static_cast<unsigned>(std::min(10000.0, std::log(num_nodes)));

        //parameters.selection_threshold = num_nodes / 10;
    }

    if (parameters.quick_pass) {
        parameters.max_outer_iterations = 50;
        parameters.max_no_improvement_iterations = 25;
    }
}


//         if (auto_alternate && current_schedule.instance->getArchitecture().synchronisationCosts() > 10000.0) {
// #ifdef KL_DEBUG
//             std::cout << "KLBase set parameters, large synchchost: only remove supersets" << std::endl;
// #endif
//             reset_superstep = false;
//             alternate_reset_remove_superstep = false;
//         }

// #ifdef KL_DEBUG
//     if (parameters.select_all_nodes)
//         std::cout << "KLBase set parameters, select all nodes" << std::endl;
//     else
//         std::cout << "KLBase set parameters, selection threshold: " << parameters.selection_threshold << std::endl;
// #endif


template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::reset_inner_search_structures() {
    average_gain = 0.0;
    node_selection.clear();
    max_gain_heap.clear();
    lock_manager.clear();
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
bool kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::is_local_search_blocked() {
    for (const auto& pair : active_schedule.new_violations) {
        if (lock_manager.is_locked(pair.first))
            return true;                    
    }
    return false;
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::initialize_datastructures(BspSchedule<Graph_t> &schedule) {

    input_schedule = &schedule;
    instance = &schedule.getInstance();
    graph = &instance->getComputationalDag();

    affinity_table = std::vector<std::vector<std::vector<cost_t>>>(
        graph->num_vertices(), std::vector<std::vector<cost_t>>(instance->numberOfProcessors(), std::vector<cost_t>(window_range, std::numeric_limits<cost_t>::lowest())));

    active_schedule.initialize(schedule);
    proc_range.initialize(*instance);
    comm_cost_f.initialize(active_schedule, affinity_table, proc_range);
    active_schedule.initialize_cost(comm_cost_f.compute_schedule_cost());
    reward_penalty_strat.initalize(active_schedule, comm_cost_f.get_max_comm_weight());
    selection_strategy.initialize(*graph, gen);
    
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_avg_gain(cost_t gain, unsigned num_iter) {
    average_gain = static_cast<double>((average_gain * num_iter + gain)) / (num_iter + 1.0);
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::insert_gain_heap(std::unordered_map<VertexType, heap_handle> &nodes) {
    for (const auto &pair : nodes) {
        compute_work_affinity(pair.first);
        comm_cost_f.compute_comm_affinity(pair.first, reward_penalty_strat.penalty, reward_penalty_strat.reward);
        nodes[pair.first] = max_gain_heap.push(compute_best_move<true>(pair.first));
    }
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::insert_new_nodes_gain_heap(std::vector<VertexType>& new_nodes, std::unordered_map<VertexType, heap_handle> &nodes) {
            
    for (const auto &node : new_nodes) {
        compute_work_affinity(node);
        comm_cost_f.compute_comm_affinity(node, reward_penalty_strat.penalty, reward_penalty_strat.reward);
        nodes[node] = max_gain_heap.push(compute_best_move<true>(node));
    }
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::cleanup_datastructures() {
    reset_inner_search_structures();
    affinity_table.clear();
    lock_manager.clear();
    active_schedule.clear();         
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::print_heap() {

    std::cout << "heap current size: " << max_gain_heap.size() << std::endl;
    std::cout << "heap top node " << max_gain_heap.top().node << " gain " << max_gain_heap.top().gain << std::endl;

    unsigned count = 0;
    for (auto it = max_gain_heap.ordered_begin(); it != max_gain_heap.ordered_end(); ++it) {
        std::cout << "node " << it->node << " gain " << it->gain << " to proc " << it->to_proc << " to step "
                    << it->to_step << std::endl;

        if (count++ > 15) {
            break;
        }
    }
}




template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t = no_local_search_memory_constraint,
         unsigned window_size = 1, typename cost_t = double>
class kl_improver_test : public kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t> {
    
    using VertexType = vertex_idx_t<Graph_t>;
    using kl_move = kl_move_struct<cost_t, VertexType>;
    using heap_datastructure = typename boost::heap::fibonacci_heap<kl_move>;
    using heap_handle = typename heap_datastructure::handle_type;
    using kl_gain_update_info = kl_update_info<VertexType>;

    public:

    kl_improver_test() : kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>() {
    }

    virtual ~kl_improver_test() = default;


    kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>& get_active_schedule() {
        return this->active_schedule;
    }

    auto & get_affinity_table() {
        return this->affinity_table;
    }

    auto & get_comm_cost_f() {
        return this->comm_cost_f;
    }

    void setup_schedule(BspSchedule<Graph_t> &schedule) {
        this->initialize_datastructures(schedule);
    }

    void apply_move_test(kl_move move) {
        this->apply_move(move);
    }

    auto & get_max_gain_heap() {
        return this->max_gain_heap;
    }

    std::unordered_map<VertexType, heap_handle> insert_gain_heap_test(const std::vector<VertexType>& n) {

        std::unordered_map<VertexType, heap_handle> nodes;
        for (const auto &node : n) {
            nodes[node] = heap_handle();
        }

        this->insert_gain_heap(nodes);
    
         return nodes;
    }

    std::unordered_map<VertexType, heap_handle> insert_gain_heap_test_penalty(const std::vector<VertexType>& n) {
        
        for (const auto &node : n) {
            this->node_selection[node] = heap_handle();
        }
        
        this->reward_penalty_strat.init_reward_penalty();

        this->insert_gain_heap(this->node_selection);

        return this->node_selection;    
    }

    std::unordered_map<VertexType, heap_handle> insert_gain_heap_test_penalty_reward(const std::vector<VertexType>& n) {
        
        for (const auto &node : n) {
            this->node_selection[node] = heap_handle();
        }
        
        this->reward_penalty_strat.init_reward_penalty();
        this->reward_penalty_strat.reward = 15.0;

        this->insert_gain_heap(this->node_selection);

        return this->node_selection;    
    }

    void update_affinity_table_test(kl_move best_move, const std::unordered_map<VertexType, heap_handle> & node_selection) {
        
        const auto prev_work_data = this->active_schedule.get_pre_move_work_data(best_move);
        this->apply_move(best_move);
            
        std::map<VertexType, kl_update_info<VertexType>> recompute_max_gain = this->update_node_work_affinity(node_selection, best_move, prev_work_data);
        auto new_nodes = this->comm_cost_f.update_node_comm_affinity(best_move, node_selection, this->lock_manager, this->reward_penalty_strat.penalty, this->reward_penalty_strat.reward, recompute_max_gain);
    }


    auto run_inner_iteration_test() {

        this->print_heap();

        kl_move best_move = this->get_best_move(); // locks best_move.node and removes it from node_selection
       
#ifdef KL_DEBUG
        std::cout << "Best move: " << best_move.node << " gain: " << best_move.gain << ", from: " << best_move.from_step << "|" << best_move.from_proc << " to: " << best_move.to_step << "|" << best_move.to_proc << std::endl;
#endif

        const auto prev_work_data = this->active_schedule.get_pre_move_work_data(best_move);
        this->apply_move(best_move);

        this->reward_penalty_strat.update_reward_penalty();
        std::map<VertexType, kl_gain_update_info> recompute_max_gain = this->update_node_work_affinity(this->node_selection, best_move, prev_work_data);
        auto new_nodes = this->comm_cost_f.update_node_comm_affinity(best_move, this->node_selection, this->lock_manager, this->reward_penalty_strat.penalty, this->reward_penalty_strat.reward, recompute_max_gain);

#ifdef KL_DEBUG
        std::cout << "New nodes: { "; 
        for (const auto v : new_nodes) {
            std::cout << v << " ";
        }                
        std::cout << "}" << std::endl;  
#endif

        this->update_max_gain(best_move, recompute_max_gain, this->node_selection);
        this->insert_new_nodes_gain_heap(new_nodes, this->node_selection);

        return recompute_max_gain;
    }

    void get_active_schedule_test(BspSchedule<Graph_t> &schedule) {
        this->active_schedule.write_schedule(schedule);
    } 

};


} // namespace osp