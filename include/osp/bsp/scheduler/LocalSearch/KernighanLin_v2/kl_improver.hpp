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
    double time_quality = 0.5;
    double superstep_remove_strength = 0.5;

    unsigned max_inner_iterations_reset = 500;
    unsigned max_no_improvement_iterations = 50;  

    unsigned max_no_vioaltions_removed_backtrack_reset;    
    unsigned remove_step_epocs;
    unsigned node_max_step_selection_epochs;
    unsigned max_no_vioaltions_removed_backtrack_for_remove_step_reset;
    unsigned max_outer_iterations;
    unsigned try_remove_step_after_num_outer_iterations;
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
    constexpr static bool enable_quick_moves = true;

    using memw_t = v_memw_t<Graph_t>;
    using commw_t = v_commw_t<Graph_t>;
    using work_weight_t = v_workw_t<Graph_t>;
    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;

    using kl_move = kl_move_struct<cost_t, VertexType>;
    using heap_datastructure = typename boost::heap::fibonacci_heap<kl_move>;
    using heap_handle = typename heap_datastructure::handle_type;
    using active_schedule_t = kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>;
    using node_selection_container_t = adaptive_affinity_table<Graph_t, cost_t, heap_handle, active_schedule_t, window_size>;
    using kl_gain_update_info = kl_update_info<VertexType>;

    bool compute_with_time_limit = false;

    BspSchedule<Graph_t> *input_schedule;
    const Graph_t *graph;
    const BspInstance<Graph_t> *instance;

    compatible_processor_range<Graph_t> proc_range;

    kl_parameter parameters;
    std::mt19937 gen;
    
    active_schedule_t active_schedule;
    
    vector_vertex_lock_manger<VertexType> lock_manager;
    vertex_selection_strategy<Graph_t, node_selection_container_t, heap_handle, active_schedule_t> selection_strategy;

    comm_cost_function_t comm_cost_f;
    reward_penalty_strategy<cost_t, comm_cost_function_t, active_schedule_t> reward_penalty_strat;

    heap_datastructure max_gain_heap;

    node_selection_container_t affinity_table;

    double average_gain = 0.0;

    unsigned max_inner_iterations;

    unsigned no_improvement_iterations_reduce_penalty;
    
    unsigned min_inner_iter;
    unsigned no_improvement_iterations_increase_inner_iter;

    unsigned step_selection_epoch_counter = 0;
    unsigned step_selection_counter = 0;

    unsigned step_to_remove = 0;
    unsigned local_search_start_step = 0;

    unsigned unlock_edge_backtrack_counter = 0;
    unsigned unlock_edge_backtrack_counter_reset = 0;

    unsigned max_no_vioaltions_removed_backtrack = 0;

    inline unsigned start_idx(unsigned node_step) { return node_step < window_size ? window_size - node_step : 0; }
    inline unsigned rel_step_idx(const unsigned node_step, const unsigned move_step) { return (move_step >= node_step) ? ((move_step - node_step) + window_size) : (window_size - (node_step - move_step)); }
    inline unsigned end_idx(unsigned node_step) { return node_step + window_size < active_schedule.num_steps() ? window_range : window_range - (node_step + window_size + 1 - active_schedule.num_steps()); }
    inline bool is_compatible(VertexType node, unsigned proc) { return active_schedule.getInstance().isCompatible(node, proc); }

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

        kl_move best_move = (*affinity_table.get_heap_handle(node));
        max_gain_heap.erase(affinity_table.get_heap_handle(node));
        lock_manager.lock(node);
        affinity_table.remove(node);

        return best_move;
    }
    
    inline void process_other_steps_best_move(const unsigned idx, const VertexType& node, const cost_t affinity_current_proc_step, cost_t& max_gain, unsigned& max_proc, unsigned& max_step) {    
        for (const unsigned p : proc_range.compatible_processors_vertex(node)) {
            const cost_t gain = affinity_current_proc_step - affinity_table[node][p][idx];
            if (gain > max_gain) {
                max_gain = gain;
                max_proc = p;
                max_step = idx; 
            }
        }
    }

    template<bool move_to_same_super_step>
    kl_move compute_best_move(VertexType node) {
        const unsigned node_step = active_schedule.assigned_superstep(node);
        const unsigned node_proc = active_schedule.assigned_processor(node);

        cost_t max_gain = std::numeric_limits<cost_t>::lowest();

        unsigned max_proc = std::numeric_limits<unsigned>::max();
        unsigned max_step = std::numeric_limits<unsigned>::max();

        const cost_t affinity_current_proc_step = affinity_table[node][node_proc][window_size];

        unsigned idx = start_idx(node_step);
        for (; idx < window_size; idx++) {
            process_other_steps_best_move(idx, node, affinity_current_proc_step, max_gain, max_proc, max_step);
        }

        if constexpr (move_to_same_super_step) {
            for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                if (proc == node_proc)
                    continue;

                const cost_t gain = affinity_current_proc_step - affinity_table[node][proc][window_size];
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
            process_other_steps_best_move(idx, node, affinity_current_proc_step, max_gain, max_proc, max_step);
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
                    
                    if (node_proc != move.from_proc && is_compatible(node, move.from_proc)) {
                        const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move.from_proc) + graph->vertex_work_weight(move.node);
                        const cost_t prev_other_affinity = compute_same_step_affinity(prev_max_work, prev_new_weight, prev_node_proc_affinity);   
                        const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move.from_proc);
                        const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);           
                        affinity_table[node][move.from_proc][window_size] += (other_affinity - prev_other_affinity);  
                    } 
                    
                    if (node_proc != move.to_proc && is_compatible(node, move.to_proc)) {
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
                                const cost_t prev_affinity = prev_max_work < prev_new_weight ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_max_work) : 0.0;
                                const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                                affinity_table[node][proc][idx] += new_affinity - prev_affinity;                                              
                            } else if (proc == move.to_proc) {
                                const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(move.to_step, proc) - graph->vertex_work_weight(move.node);
                                const cost_t prev_affinity = prev_max_work < prev_new_weight ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_max_work) : 0.0;
                                const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                                affinity_table[node][proc][idx] += new_affinity - prev_affinity;
                            } else {
                                const cost_t prev_affinity = prev_max_work < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(prev_max_work) : 0.0;
                                const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                                affinity_table[node][proc][idx] += new_affinity - prev_affinity;  
                            }
                        }                            
                    } else {
                        // update only move.from_proc and move.to_proc
                        if (is_compatible(node, move.from_proc)) {
                            const work_weight_t from_new_weight = vertex_weight + active_schedule.get_step_processor_work(move.from_step, move.from_proc);
                            const work_weight_t from_prev_new_weight = from_new_weight + graph->vertex_work_weight(move.node);
                            const cost_t from_prev_affinity = prev_max_work < from_prev_new_weight ? static_cast<cost_t>(from_prev_new_weight) - static_cast<cost_t>(prev_max_work) : 0.0;

                            const cost_t from_new_affinity = new_max_weight < from_new_weight ? static_cast<cost_t>(from_new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                            affinity_table[node][move.from_proc][idx] += from_new_affinity - from_prev_affinity;
                        }

                        if (is_compatible(node, move.to_proc)) {
                            const work_weight_t to_new_weight = vertex_weight + active_schedule.get_step_processor_work(move.to_step, move.to_proc);
                            const work_weight_t to_prev_new_weight = to_new_weight - graph->vertex_work_weight(move.node);
                            const cost_t to_prev_affinity = prev_max_work < to_prev_new_weight ? static_cast<cost_t>(to_prev_new_weight) - static_cast<cost_t>(prev_max_work) : 0.0;

                            const cost_t to_new_affinity = new_max_weight < to_new_weight ? static_cast<cost_t>(to_new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                            affinity_table[node][move.to_proc][idx] += to_new_affinity - to_prev_affinity;
                        }
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

    void process_work_update_step(VertexType node, unsigned node_step, unsigned node_proc, work_weight_t vertex_weight, unsigned move_step, unsigned move_proc, work_weight_t move_correction_node_weight, const work_weight_t prev_move_step_max_work, const work_weight_t prev_move_step_second_max_work, unsigned prev_move_step_max_work_processor_count, bool & update_step, bool & update_entire_step, bool & full_update);
    void update_node_work_affinity(const node_selection_container_t &nodes, kl_move move, const pre_move_work_data<work_weight_t> & prev_work_data, std::map<VertexType, kl_gain_update_info> &recompute_max_gain);
    void update_best_move(VertexType node, unsigned step, unsigned proc, node_selection_container_t &nodes);    
    void update_best_move(VertexType node, unsigned step, node_selection_container_t &nodes);
    void update_max_gain(kl_move move, std::map<VertexType, kl_gain_update_info> &recompute_max_gain, node_selection_container_t &nodes);
    void compute_work_affinity(VertexType node);

    inline void recompute_node_max_gain(VertexType node, node_selection_container_t &nodes) {
        const auto best_move = compute_best_move<true>(node);
        heap_handle & node_handle = nodes.get_heap_handle(node);
        (*node_handle).gain = best_move.gain;
        (*node_handle).to_proc = best_move.to_proc;
        (*node_handle).to_step = best_move.to_step;
        max_gain_heap.update(node_handle);
    }

    inline cost_t compute_same_step_affinity(const work_weight_t &max_work_for_step, const work_weight_t &new_weight, const cost_t &node_proc_affinity) {
        const cost_t max_work_after_removal = static_cast<cost_t>(max_work_for_step) - node_proc_affinity;
        if (new_weight > max_work_after_removal) {
            return new_weight - max_work_after_removal;
        }
        return 0.0;
    }
    
    inline cost_t apply_move(kl_move move) {
        active_schedule.apply_move(move);
        comm_cost_f.update_datastructure_after_move(move); 
        cost_t change_in_cost = -move.gain;
        change_in_cost += static_cast<cost_t>(active_schedule.resolved_violations.size()) * reward_penalty_strat.reward;
        change_in_cost -= static_cast<cost_t>(active_schedule.new_violations.size()) * reward_penalty_strat.penalty;
  
#ifdef KL_DEBUG
        std::cout << "penalty: " << reward_penalty_strat.penalty << " num violations: " << active_schedule.get_current_violations().size() <<  " num new violations: " << active_schedule.new_violations.size() << ", num resolved violations: " << active_schedule.resolved_violations.size() <<  ", reward: " << reward_penalty_strat.reward << std::endl;
        std::cout << "apply move, previous cost: " << active_schedule.get_cost() << ", new cost: " << active_schedule.get_cost() + change_in_cost << ", " << (active_schedule.is_feasible() ? "feasible," : "infeasible,") << std::endl;
#endif
        active_schedule.update_cost(change_in_cost);

        return change_in_cost;
    }    

    bool run_local_search() {

#ifdef KL_DEBUG_1
        std::cout << "start local search, initial schedule cost: " << active_schedule.get_cost() << " with " << active_schedule.num_steps() << " supersteps." << std::endl;
#endif
        std::vector<VertexType> new_nodes;
        std::vector<VertexType> unlock_nodes;
        std::map<VertexType, kl_gain_update_info> recompute_max_gain;

        const auto start_time = std::chrono::high_resolution_clock::now();
        cost_t initial_cost = active_schedule.get_cost();
        unsigned no_improvement_iter_counter = 0;
        unsigned outer_iter = 0;

        for (; outer_iter < parameters.max_outer_iterations; outer_iter++) {
            cost_t initial_inner_iter_cost = active_schedule.get_cost();

            reset_inner_search_structures();            
            select_active_nodes(affinity_table);  
            reward_penalty_strat.init_reward_penalty(static_cast<double>(active_schedule.get_current_violations().size()) + 1.0);
            insert_gain_heap(affinity_table);
            
            unsigned inner_iter = 0;
            unsigned violation_removed_count = 0;
            unsigned reset_counter = 0;
            bool iter_inital_feasible = active_schedule.is_feasible();
            
#ifdef KL_DEBUG
            std::cout << "------ start inner loop ------" << std::endl;
            std::cout << "initial node selection: {";
            for (size_t i = 0; i < affinity_table.size() ; ++i) {
                std::cout << affinity_table.get_selected_nodes()[i] << ", ";
            }
            std::cout << "}" << std::endl;
               

            if (not iter_inital_feasible) {
                std::cout << "initial solution not feasible, num violations: " << active_schedule.get_current_violations().size() << ". Penalty: " << reward_penalty_strat.penalty << ", reward: " << reward_penalty_strat.reward << std::endl;
            }
#endif
#ifdef KL_DEBUG_1
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - active_schedule.get_cost()) > 0.00001 ) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << active_schedule.get_cost() << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
#endif


            while (inner_iter < max_inner_iterations && max_gain_heap.size() > 0) {
                kl_move best_move = get_best_move(); // locks best_move.node and removes it from node_selection
                update_avg_gain(best_move.gain, inner_iter);
#ifdef KL_DEBUG
                std::cout << " >>> move node " << best_move.node << " with gain " << best_move.gain << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step << " to: " << best_move.to_proc << "|" << best_move.to_step << ",avg gain: " << average_gain << std::endl;
#endif
                if (inner_iter > min_inner_iter && average_gain < 0.0) {
#ifdef KL_DEBUG
                    std::cout << "Negative average gain: " << average_gain << ", end local search" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG_1
                if (not active_schedule.getInstance().isCompatible(best_move.node, best_move.to_proc)) {
                    std::cout << "move to incompatibe node" << std::endl;
                }
#endif

                const auto prev_work_data = active_schedule.get_pre_move_work_data(best_move);
                const cost_t change_in_cost = apply_move(best_move);

                if constexpr (enable_quick_moves) {
                    if (iter_inital_feasible && active_schedule.new_violations.size() > 0) {

#ifdef KL_DEBUG
                        std::cout << "Starting quick moves sequence." << std::endl;
#endif
                        inner_iter++;

                        const size_t num_applied_moves = active_schedule.getAppliedMoves().size() - 1;
                        const cost_t saved_cost = active_schedule.get_cost() - change_in_cost;

                        std::unordered_set<VertexType> local_lock;
                        local_lock.insert(best_move.node);
                        std::vector<VertexType> quick_moves_stack;
                        quick_moves_stack.reserve(10 + active_schedule.new_violations.size() * 2);

                        for (const auto& [key, value] : active_schedule.new_violations) {
                            quick_moves_stack.push_back(key);
                        }

                        while (quick_moves_stack.size() > 0) {

                            auto next_node_to_move = quick_moves_stack.back();
                            quick_moves_stack.pop_back();

                            affinity_table.insert(next_node_to_move);
                            reward_penalty_strat.init_reward_penalty(static_cast<double>(active_schedule.get_current_violations().size()) + 1.0);
                            compute_node_affinities(next_node_to_move);
                            kl_move best_quick_move = compute_best_move<true>(next_node_to_move);

#ifdef KL_DEBUG
                            std::cout << " >>> move node " << best_quick_move.node << " with gain " << best_quick_move.gain << ", from proc|step: " << best_quick_move.from_proc << "|" << best_quick_move.from_step << " to: " << best_quick_move.to_proc << "|" << best_quick_move.to_step << std::endl;
#endif

                            apply_move(best_quick_move);                          
                            local_lock.insert(next_node_to_move);

                            inner_iter++;

                            if (active_schedule.new_violations.size() > 0) {
                                bool abort = false;

                                for (const auto& [key, value] : active_schedule.new_violations) {
                                    if(local_lock.find(key) != local_lock.end()) {
                                        abort = true;
                                        break;
                                    }                                    
                                    quick_moves_stack.push_back(key);
                                }

                                if (abort) break;

                            } else if (active_schedule.is_feasible()) {
                                break;
                            }
                        }

                        if (!active_schedule.is_feasible()) {
                            active_schedule.revert_schedule_to_bound(num_applied_moves, saved_cost ,true, comm_cost_f);
#ifdef KL_DEBUG
                            std::cout << "Ending quick moves sequence with infeasible solution." << std::endl;
#endif
                        } 
#ifdef KL_DEBUG
                        else {
                            std::cout << "Ending quick moves sequence with feasible solution." << std::endl;
                        }
#endif
                        local_lock.erase(best_move.node);
                        for (const auto & node : local_lock) {
                            affinity_table.remove(node);
                        }

                        affinity_table.trim();
                        max_gain_heap.clear();
                        reward_penalty_strat.init_reward_penalty(static_cast<double>(active_schedule.get_current_violations().size()) + 1.0);
                        insert_gain_heap(affinity_table);

                        

#ifdef KL_DEBUG_1
                        if (std::abs(comm_cost_f.compute_schedule_cost_test() - active_schedule.get_cost()) > 0.00001 ) {
                            std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << active_schedule.get_cost() << std::endl;
                            std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                        }
#endif

                        continue;
                    }
                }

                if (active_schedule.get_current_violations().size() > 0) {
                    if (active_schedule.resolved_violations.size() > 0) {
                        violation_removed_count = 0;
                    } else {
                        violation_removed_count++;

                        if (violation_removed_count > 3) {                           
                           if (reset_counter < max_no_vioaltions_removed_backtrack && ((not iter_inital_feasible) || (active_schedule.get_cost() < active_schedule.get_best_cost()))) {                                   
                                affinity_table.reset_node_selection();
                                max_gain_heap.clear();
                                lock_manager.clear();
                                selection_strategy.select_nodes_violations(affinity_table);
#ifdef KL_DEBUG
                                std::cout << "Infeasible, and no violations resolved for 5 iterations, reset node selection" << std::endl;
#endif
                                reward_penalty_strat.init_reward_penalty(static_cast<double>(active_schedule.get_current_violations().size()));
                                insert_gain_heap(affinity_table);

                                reset_counter++;
                                inner_iter++;
                                continue;
                            } else {
#ifdef KL_DEBUG
                                std::cout << "Infeasible, and no violations resolved for 5 iterations, end local search" << std::endl;
#endif
                                break;
                            } 
                        }
                    }
                } 
                                
                if(is_local_search_blocked()) {
                    if (not blocked_edge_strategy(best_move.node, unlock_nodes)) {
                        break;
                    }
                }

                affinity_table.trim();

                update_node_work_affinity(affinity_table, best_move, prev_work_data, recompute_max_gain);
                comm_cost_f.update_node_comm_affinity(best_move, affinity_table, lock_manager, reward_penalty_strat.penalty, reward_penalty_strat.reward, recompute_max_gain, new_nodes);

                for (const auto v : unlock_nodes) {
                    lock_manager.unlock(v);
                }
                new_nodes.insert(new_nodes.end(), unlock_nodes.begin(), unlock_nodes.end());
                unlock_nodes.clear();

#ifdef KL_DEBUG
                std::cout << "recmopute max gain: {";
                for (const auto [key, value] : recompute_max_gain) {
                    std::cout << key << ", ";
                }
                std::cout << "}" << std::endl;
                std::cout << "new nodes: {";
                for (const auto v : new_nodes) {
                    std::cout << v << ", ";
                }
                std::cout << "}" << std::endl;                
#endif
#ifdef KL_DEBUG_1
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - active_schedule.get_cost()) > 0.00001 ) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << active_schedule.get_cost() << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
#endif
                update_max_gain(best_move, recompute_max_gain, affinity_table);
                insert_new_nodes_gain_heap(new_nodes, affinity_table);

                recompute_max_gain.clear();
                new_nodes.clear();

                inner_iter++;
            }

#ifdef KL_DEBUG
            std::cout << "--- end inner loop after " << inner_iter << " inner iterations, gain heap size: " << max_gain_heap.size() <<  ", outer iteraion " << outer_iter << "/" << parameters.max_outer_iterations << ", current cost: " << active_schedule.get_cost() << ", " << (active_schedule.is_feasible() ? "feasible" : "infeasible") << std::endl;
#endif
               
            active_schedule.revert_to_best_schedule(local_search_start_step, step_to_remove, comm_cost_f);
            if (local_search_start_step > active_schedule.getBestScheduleIdx()) {
                step_selection_counter++;
                if (step_selection_counter >= active_schedule.num_steps()) {
                    step_selection_counter = 0;
                    step_selection_epoch_counter++;
                }            
            }

#ifdef KL_DEBUG_1
            if (std::abs(comm_cost_f.compute_schedule_cost_test() - active_schedule.get_cost()) > 0.00001 ) {
                std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << active_schedule.get_cost() << std::endl;
                std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
            }
#endif

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
#ifdef KL_DEBUG_1
                    std::cout << "no improvement for " << parameters.max_no_improvement_iterations
                              << " iterations, end local search" << std::endl;
#endif
                    break;
                }    
            } else {
                no_improvement_iter_counter = 0;
            } 
            
            adjust_local_search_parameters(outer_iter, no_improvement_iter_counter);
        }

#ifdef KL_DEBUG_1
        std::cout << "local search end after " << outer_iter << " outer iterations, current cost: " << active_schedule.get_cost() << " with " << active_schedule.num_steps() << " supersteps, vs initial cost: " << initial_cost << ", vs " << active_schedule.get_total_work_weight() << " serial costs." << std::endl;
#endif
        if (initial_cost > active_schedule.get_cost())
            return true;
        else
            return false;
    }

    inline bool blocked_edge_strategy(VertexType node, std::vector<VertexType> & unlock_nodes) {                        
        if (unlock_edge_backtrack_counter > 1) {
            for (const auto [v,e] : active_schedule.new_violations) {
                const auto source_v = source(e, *graph);
                const auto target_v = target(e, *graph);

                if (node == source_v && lock_manager.is_locked(target_v)) {
                    unlock_nodes.push_back(target_v);
                } else if (node == target_v && lock_manager.is_locked(source_v)) {
                    unlock_nodes.push_back(source_v);
                }
            }
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, backtrack counter: " << unlock_edge_backtrack_counter <<  std::endl;
#endif
            unlock_edge_backtrack_counter--;
            return true;
        } else {
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, end local search" << std::endl;
#endif
            return false;  //or reset local search and initalize with violating nodes
        }
    }

    inline void adjust_local_search_parameters(unsigned outer_iter, unsigned no_imp_counter) {         
        if (no_imp_counter >= no_improvement_iterations_reduce_penalty && reward_penalty_strat.initial_penalty > 0.0) {
            reward_penalty_strat.initial_penalty = std::floor(std::sqrt(reward_penalty_strat.initial_penalty));
            unlock_edge_backtrack_counter_reset += 1;
            no_improvement_iterations_reduce_penalty += 15;
#ifdef KL_DEBUG_1
            std::cout << "no improvement for " << no_improvement_iterations_reduce_penalty
                        << " iterations, reducing initial penalty to " << reward_penalty_strat.initial_penalty << std::endl;
#endif                   
        } 

        if ((outer_iter + 1) % (parameters.try_remove_step_after_num_outer_iterations) == 0) {
            step_selection_epoch_counter = 0;
#ifdef KL_DEBUG
            std::cout << "reset remove epoc counter after " << outer_iter << " iterations." << std::endl;
#endif
        }

        if (no_imp_counter >= no_improvement_iterations_increase_inner_iter ) {
            min_inner_iter = static_cast<unsigned>(std::ceil(min_inner_iter * 2.2));
            no_improvement_iterations_increase_inner_iter += 20;
#ifdef KL_DEBUG_1
            std::cout << "no improvement for " << no_improvement_iterations_increase_inner_iter
                        << " iterations, increasing min inner iter to " << min_inner_iter << std::endl;
#endif
        }

    }
    
    bool is_local_search_blocked();
    void set_parameters(vertex_idx_t<Graph_t> num_nodes);
    void reset_inner_search_structures();
    void initialize_datastructures(BspSchedule<Graph_t> &schedule);
    void print_heap();
    void cleanup_datastructures();
    void update_avg_gain(cost_t gain, unsigned num_iter); 
    void insert_gain_heap(node_selection_container_t &nodes);
    void insert_new_nodes_gain_heap(std::vector<VertexType>& new_nodes, node_selection_container_t &nodes);

    inline void compute_node_affinities(VertexType node) {
        compute_work_affinity(node);
        comm_cost_f.compute_comm_affinity(node, affinity_table, reward_penalty_strat.penalty, reward_penalty_strat.reward);
    }

    void select_active_nodes(node_selection_container_t & nodes) {   
        if (select_nodes_check_remove_superstep(step_to_remove, nodes)) {
            active_schedule.remove_empty_step(step_to_remove);
            local_search_start_step = static_cast<unsigned>(active_schedule.getAppliedMoves().size());
            active_schedule.update_cost(-1.0 * static_cast<cost_t>(instance->synchronisationCosts()));
            unlock_edge_backtrack_counter = static_cast<unsigned>(active_schedule.get_current_violations().size());
            max_inner_iterations = std::max(unlock_edge_backtrack_counter * 5u, parameters.max_inner_iterations_reset);
            max_no_vioaltions_removed_backtrack = parameters.max_no_vioaltions_removed_backtrack_for_remove_step_reset;
#ifdef KL_DEBUG_1
            std::cout << "Trying to remove step " << step_to_remove << std::endl;
#endif
            return;        
        }
        step_to_remove = 0;
        local_search_start_step = 0;
        selection_strategy.select_active_nodes(nodes);
    }

    bool check_remove_superstep(unsigned step);
    bool select_nodes_check_remove_superstep(unsigned & step, node_selection_container_t & nodes);

    bool scatter_nodes_superstep(unsigned step, node_selection_container_t & nodes) {
        assert(step < active_schedule.num_steps());
        bool abort = false;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            const std::vector<VertexType> step_proc_node_vec(active_schedule.getSetSchedule().step_processor_vertices[step][proc].begin(),active_schedule.getSetSchedule().step_processor_vertices[step][proc].end());
            for (const auto &node : step_proc_node_vec) {
                nodes.insert(node);
                reward_penalty_strat.init_reward_penalty(static_cast<double>(active_schedule.get_current_violations().size()) + 1.0);
                compute_node_affinities(node);
                kl_move best_move = compute_best_move<false>(node);

                if (best_move.gain == std::numeric_limits<double>::lowest()) {
                    abort = true;
                    break;
                }

                apply_move(best_move);

#ifdef KL_DEBUG
                std::cout << "move node " << best_move.node << " with gain " << best_move.gain << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step << " to: " << best_move.to_proc << "|" << best_move.to_step << ",avg gain: " << average_gain << std::endl;
#endif

#ifdef KL_DEBUG_1
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - active_schedule.get_cost()) > 0.00001 ) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << active_schedule.get_cost() << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
#endif
                selection_strategy.add_neighbours_to_selection(best_move.node, nodes);
            }

            if (abort) {
                break;
            }
        }

        if (abort) {
            active_schedule.revert_to_best_schedule(0, 0, comm_cost_f);
            nodes.reset_node_selection();
            return false;
        }
        return true;
    }

  public:
    kl_improver() : ImprovementScheduler<Graph_t>() {

        std::random_device rd;
        gen = std::mt19937(rd());
    }

    virtual ~kl_improver() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule<Graph_t> &schedule) override {
        set_parameters(schedule.getInstance().numberOfVertices());
        initialize_datastructures(schedule);        

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
};



template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::set_parameters(vertex_idx_t<Graph_t> num_nodes) {
    
    const unsigned log_num_nodes = (num_nodes > 1) ? static_cast<unsigned>(std::log(num_nodes)) : 1; 

    // The number of nodes to consider in each inner iteration. Proportional to log(N).
    selection_strategy.selection_threshold = static_cast<std::size_t>(std::ceil(parameters.time_quality * 10 * log_num_nodes + log_num_nodes));
    // Total number of outer iterations. Proportional to N.
    parameters.max_outer_iterations = static_cast<unsigned>(static_cast<double>(num_nodes) * (parameters.time_quality + 0.4));
    // Minimum number of moves to perform in an inner iteration.
    min_inner_iter = static_cast<unsigned>(log_num_nodes + log_num_nodes * (1.0 + parameters.time_quality));
    // Number of times to reset the search for violations before giving up.
    parameters.max_no_vioaltions_removed_backtrack_reset = parameters.time_quality < 0.75 ? 1 : parameters.time_quality < 1.0 ? 2 : 3;

    // Parameters for the superstep removal heuristic.
    parameters.max_no_vioaltions_removed_backtrack_for_remove_step_reset = 3 + static_cast<unsigned>(parameters.superstep_remove_strength * 14);
    parameters.node_max_step_selection_epochs = parameters.superstep_remove_strength < 0.75 ? 1 : parameters.superstep_remove_strength < 1.0 ? 2 : 3;
    parameters.remove_step_epocs = static_cast<unsigned>(parameters.superstep_remove_strength * 5.0);

    // Number of iterations without improvement before reducing penalty.
    no_improvement_iterations_reduce_penalty = parameters.max_no_improvement_iterations / 5;
    // Number of iterations without improvement before increasing the minimum inner iterations.
    no_improvement_iterations_increase_inner_iter = 10;

    if (parameters.remove_step_epocs > 0) {
        parameters.try_remove_step_after_num_outer_iterations = parameters.max_outer_iterations / parameters.remove_step_epocs;
    } else {
        // Effectively disable superstep removal if remove_step_epocs is 0.
        parameters.try_remove_step_after_num_outer_iterations = parameters.max_outer_iterations + 1;
    }
    
    max_inner_iterations = parameters.max_inner_iterations_reset; 

    #ifdef KL_DEBUG_1
                    std::cout << "kl set parameter, number of nodes: " << num_nodes << std::endl;
                    std::cout << "max outer iterations: " << parameters.max_outer_iterations << std::endl;
                    std::cout << "max inner iterations: " << parameters.max_inner_iterations_reset << std::endl; 
                    std::cout << "min inner iterations: " << min_inner_iter << std::endl;
                    std::cout << "no improvement iterations reduce penalty: " << no_improvement_iterations_reduce_penalty << std::endl;
                    std::cout << "selction threshold: " << selection_strategy.selection_threshold << std::endl;
                    std::cout << "remove step epocs: " << parameters.remove_step_epocs << std::endl;
                    std::cout << "try remove step after num outer iterations: " << parameters.try_remove_step_after_num_outer_iterations << std::endl;                   
    #endif

}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_node_work_affinity(const node_selection_container_t &nodes, kl_move move, const pre_move_work_data<work_weight_t> & prev_work_data, std::map<VertexType, kl_gain_update_info> &recompute_max_gain) {
    const size_t active_count = nodes.size();

    for (size_t i = 0; i < active_count; ++i) {
        const VertexType node = nodes.get_selected_nodes()[i];
            
        kl_gain_update_info update_info = update_node_work_affinity_after_move(node, move, prev_work_data);
        if (update_info.update_from_step || update_info.update_to_step) {
            recompute_max_gain[node] = update_info;
        }        
    }
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_max_gain(kl_move move, std::map<VertexType, kl_gain_update_info> &recompute_max_gain, node_selection_container_t &nodes) {
    for (auto& pair : recompute_max_gain) { 
        if (pair.second.full_update) {
            recompute_node_max_gain(pair.first, nodes); 
        } else {
            if (pair.second.update_entire_from_step) {
                update_best_move(pair.first, move.from_step, nodes);
            } else if (pair.second.update_from_step && is_compatible(pair.first, move.from_proc)) {
                update_best_move(pair.first, move.from_step, move.from_proc, nodes);
            } 

            if (move.from_step != move.to_step || not pair.second.update_entire_from_step) {
                if (pair.second.update_entire_to_step) {
                    update_best_move(pair.first, move.to_step, nodes);
                } else if (pair.second.update_to_step && is_compatible(pair.first, move.to_proc)) {
                    update_best_move(pair.first, move.to_step, move.to_proc, nodes);
                }
            }
        } 
    }    
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::compute_work_affinity(VertexType node) {
    const unsigned node_step = active_schedule.assigned_superstep(node);
    const work_weight_t vertex_weight = graph->vertex_work_weight(node);

    unsigned step = (node_step > window_size) ? (node_step - window_size) : 0;
    for (unsigned idx = start_idx(node_step); idx < end_idx(node_step); ++idx, ++step) {
        if (idx == window_size) {
            continue;
        }

        const cost_t max_work_for_step = static_cast<cost_t>(active_schedule.get_step_max_work(step));

        for (const unsigned proc : proc_range.compatible_processors_vertex(node)) { 
            const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(step, proc);
            const cost_t work_diff = static_cast<cost_t>(new_weight) - max_work_for_step;
            affinity_table[node][proc][idx] = std::max(0.0, work_diff);
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

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::process_work_update_step(VertexType node, unsigned node_step, unsigned node_proc, work_weight_t vertex_weight, unsigned move_step, unsigned move_proc, work_weight_t move_correction_node_weight, const work_weight_t prev_move_step_max_work, const work_weight_t prev_move_step_second_max_work, unsigned prev_move_step_max_work_processor_count, bool & update_step, bool & update_entire_step, bool & full_update) {
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
            
            if (node_proc != move_proc && is_compatible(node, move_proc)) {
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

                        const cost_t prev_affinity = prev_move_step_max_work < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(prev_move_step_max_work) : 0.0;
                        const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                        affinity_table[node][proc][idx] += new_affinity - prev_affinity;  

                    } else {
                        const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(move_step, proc) + move_correction_node_weight;
                        const cost_t prev_affinity = prev_move_step_max_work < prev_new_weight ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_move_step_max_work) : 0.0;

                        const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                        affinity_table[node][proc][idx] += new_affinity - prev_affinity;
                    }
                }                        
            } else {
                // update only move_proc
                if (is_compatible(node, move_proc)) {
                    const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(move_step, move_proc);
                    const work_weight_t prev_new_weight = new_weight + move_correction_node_weight;
                    const cost_t prev_affinity = prev_move_step_max_work < prev_new_weight ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_move_step_max_work) : 0.0;

                    const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                    affinity_table[node][move_proc][idx] += new_affinity - prev_affinity;
                }
            }
        }
    }
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
bool kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>:: select_nodes_check_remove_superstep(unsigned & step_to_remove, node_selection_container_t & nodes) {
    if (step_selection_epoch_counter >= parameters.node_max_step_selection_epochs || active_schedule.num_steps() < 3) {
        return false;
    }
    
    for (step_to_remove = step_selection_counter; step_to_remove < active_schedule.num_steps(); step_to_remove++) {
        
#ifdef KL_DEBUG
            std::cout << "Checking to remove step " << step_to_remove << "/" << active_schedule.num_steps() <<  std::endl;
#endif
        if (check_remove_superstep(step_to_remove)) {
#ifdef KL_DEBUG
            std::cout << "Checking to scatter step " << step_to_remove << "/" << active_schedule.num_steps() <<  std::endl;
#endif
            if (scatter_nodes_superstep(step_to_remove, nodes)) {
                step_selection_counter = step_to_remove;

                if (step_selection_counter >= active_schedule.num_steps()) {
                    step_selection_counter = 0;
                    step_selection_epoch_counter++;
                }
                return true;
            }
        }
    }

    step_selection_epoch_counter++;
    step_selection_counter = 0;
    return false;
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
bool kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::check_remove_superstep(unsigned step) {
    if (active_schedule.num_steps() < 2) 
        return false;
    
    if (active_schedule.get_step_max_work(step) < 2 * instance->synchronisationCosts())
        return true;

    return false;
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::reset_inner_search_structures() {
    unlock_edge_backtrack_counter = unlock_edge_backtrack_counter_reset;
    max_inner_iterations = parameters.max_inner_iterations_reset;
    max_no_vioaltions_removed_backtrack = parameters.max_no_vioaltions_removed_backtrack_reset;
    average_gain = 0.0;
    affinity_table.reset_node_selection();
    max_gain_heap.clear();
    lock_manager.clear();
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
bool kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::is_local_search_blocked() {
    for (const auto& pair : active_schedule.new_violations) {
        if (lock_manager.is_locked(pair.first)) {
            return true;                    
        }
    }
    return false;
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::initialize_datastructures(BspSchedule<Graph_t> &schedule) {
    input_schedule = &schedule;
    instance = &schedule.getInstance();
    graph = &instance->getComputationalDag();

    active_schedule.initialize(schedule);
    selection_strategy.initialize(active_schedule, gen);
    affinity_table.initialize(active_schedule, selection_strategy.selection_threshold);

    proc_range.initialize(*instance);
    comm_cost_f.initialize(active_schedule, proc_range);
    active_schedule.initialize_cost(comm_cost_f.compute_schedule_cost());
    reward_penalty_strat.initalize(active_schedule, comm_cost_f.get_max_comm_weight_multiplied(), active_schedule.get_max_work_weight());
    
    lock_manager.initialize(graph->num_vertices());    
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_avg_gain(cost_t gain, unsigned num_iter) {
    average_gain = static_cast<double>((average_gain * num_iter + gain)) / (num_iter + 1.0);
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::insert_gain_heap(node_selection_container_t &nodes) {
    const size_t active_count = nodes.size();

    // #pragma omp parallel for //num_threads(2)
    // for (size_t i = 0; i < active_count; ++i) {
    //     const VertexType node = nodes.get_selected_nodes()[i]; 
    //     compute_node_affinities(node);
    // }

    // for (size_t i = 0; i < active_count; ++i) {
    //     const VertexType node = nodes.get_selected_nodes()[i]; 
    //     nodes.get_heap_handle(node) = max_gain_heap.push(compute_best_move<true>(node));
    // }

    for (size_t i = 0; i < active_count; ++i) {
        const VertexType node = nodes.get_selected_nodes()[i]; 
        compute_node_affinities(node);
        nodes.get_heap_handle(node) = max_gain_heap.push(compute_best_move<true>(node));
    }
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::insert_new_nodes_gain_heap(std::vector<VertexType>& new_nodes, node_selection_container_t &nodes) { 
    for (const auto &node : new_nodes) {
        nodes.insert(node);
        compute_node_affinities(node);
        nodes.get_heap_handle(node) = max_gain_heap.push(compute_best_move<true>(node));        
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


template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_best_move(VertexType node, unsigned step, unsigned proc, node_selection_container_t &nodes) {
    const unsigned node_proc = active_schedule.assigned_processor(node);
    const unsigned node_step = active_schedule.assigned_superstep(node);

    if((node_proc == proc) && (node_step == step)) 
        return;

    const heap_handle & node_handle = affinity_table.get_heap_handle(node);
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
    
template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_best_move(VertexType node, unsigned step, node_selection_container_t &nodes) {
    
    const unsigned node_proc = active_schedule.assigned_processor(node);
    const unsigned node_step = active_schedule.assigned_superstep(node);

    const heap_handle & node_handle = affinity_table.get_heap_handle(node);
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



template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t = no_local_search_memory_constraint,
         unsigned window_size = 1, typename cost_t = double>
class kl_improver_test : public kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t> {
    
    using VertexType = vertex_idx_t<Graph_t>;
    using kl_move = kl_move_struct<cost_t, VertexType>;
    using heap_datastructure = typename boost::heap::fibonacci_heap<kl_move>;
    using heap_handle = typename heap_datastructure::handle_type;
    using kl_gain_update_info = kl_update_info<VertexType>;
    using node_selection_container_t = adaptive_affinity_table<Graph_t, cost_t, heap_handle, kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>, window_size>;

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

    node_selection_container_t  insert_gain_heap_test(const std::vector<VertexType>& n) {        
        this->affinity_table.initialize(this->active_schedule, n.size());

        for (const auto &node : n) {
            this->affinity_table.insert(node);
        }

        this->insert_gain_heap(this->affinity_table);
    
        return this->affinity_table;  
    }

    node_selection_container_t insert_gain_heap_test_penalty(const std::vector<VertexType>& n) {
        this->affinity_table.initialize(this->active_schedule, n.size());
        for (const auto &node : n) {
            this->affinity_table.insert(node);
        }
        this->reward_penalty_strat.penalty = 5.5;
        this->reward_penalty_strat.reward = 0.0;

        this->insert_gain_heap(this->affinity_table);

        return this->affinity_table;    
    }

    node_selection_container_t insert_gain_heap_test_penalty_reward(const std::vector<VertexType>& n) {
        this->affinity_table.initialize(this->active_schedule, n.size());
        for (const auto &node : n) {
            this->affinity_table.insert(node);
        }
        
        this->reward_penalty_strat.init_reward_penalty();
        this->reward_penalty_strat.reward = 15.0;

        this->insert_gain_heap(this->affinity_table);

        return this->affinity_table;    
    }

    void update_affinity_table_test(kl_move best_move, const node_selection_container_t & node_selection) {
        std::map<VertexType, kl_gain_update_info> recompute_max_gain;
        std::vector<VertexType> new_nodes;

        const auto prev_work_data = this->active_schedule.get_pre_move_work_data(best_move);
        this->apply_move(best_move);
            
        this->update_node_work_affinity(node_selection, best_move, prev_work_data, recompute_max_gain);
        this->comm_cost_f.update_node_comm_affinity(best_move, this->affinity_table, this->lock_manager, this->reward_penalty_strat.penalty, this->reward_penalty_strat.reward, recompute_max_gain, new_nodes);
    }


    auto run_inner_iteration_test() {

        std::map<VertexType, kl_gain_update_info> recompute_max_gain;
        std::vector<VertexType> new_nodes;

        this->print_heap();

        kl_move best_move = this->get_best_move(); // locks best_move.node and removes it from node_selection
       
#ifdef KL_DEBUG
        std::cout << "Best move: " << best_move.node << " gain: " << best_move.gain << ", from: " << best_move.from_step << "|" << best_move.from_proc << " to: " << best_move.to_step << "|" << best_move.to_proc << std::endl;
#endif

        const auto prev_work_data = this->active_schedule.get_pre_move_work_data(best_move);
        this->apply_move(best_move);

        this->affinity_table.trim();
        this->update_node_work_affinity(this->affinity_table, best_move, prev_work_data, recompute_max_gain);
        this->comm_cost_f.update_node_comm_affinity(best_move, this->affinity_table, this->lock_manager, this->reward_penalty_strat.penalty, this->reward_penalty_strat.reward, recompute_max_gain, new_nodes);

#ifdef KL_DEBUG
        std::cout << "New nodes: { "; 
        for (const auto v : new_nodes) {
            std::cout << v << " ";
        }                
        std::cout << "}" << std::endl;  
#endif

        this->update_max_gain(best_move, recompute_max_gain, this->affinity_table);
        this->insert_new_nodes_gain_heap(new_nodes, this->affinity_table);

        return recompute_max_gain;
    }

    void get_active_schedule_test(BspSchedule<Graph_t> &schedule) {
        this->active_schedule.write_schedule(schedule);
    } 

};


} // namespace osp