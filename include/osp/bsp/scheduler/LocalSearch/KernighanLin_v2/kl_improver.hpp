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

#include "osp/auxiliary/datastructures/heaps/PairingHeap.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"

#include "kl_active_schedule.hpp"
#include "kl_util.hpp"

namespace osp {

struct kl_parameter {
    double time_quality = 0.8;
    double superstep_remove_strength = 0.5;
    unsigned num_parallel_loops = 4;

    unsigned max_inner_iterations_reset = 500;
    unsigned max_no_improvement_iterations = 50;  

    constexpr static unsigned abort_scatter_nodes_violation_threshold = 500;
    constexpr static unsigned initial_violation_threshold = 250;

    unsigned max_no_vioaltions_removed_backtrack_reset;    
    unsigned remove_step_epocs;
    unsigned node_max_step_selection_epochs;
    unsigned max_no_vioaltions_removed_backtrack_for_remove_step_reset;
    unsigned max_outer_iterations;
    unsigned try_remove_step_after_num_outer_iterations;
    unsigned min_inner_iter_reset;

    unsigned thread_min_range = 8;
    unsigned thread_range_gap = 0;

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
    constexpr static bool enable_preresolving_violations = true;

    using memw_t = v_memw_t<Graph_t>;
    using commw_t = v_commw_t<Graph_t>;
    using work_weight_t = v_workw_t<Graph_t>;
    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;

    using kl_move = kl_move_struct<cost_t, VertexType>;
    using heap_datastructure = MaxPairingHeap<VertexType, kl_move>;
    using active_schedule_t = kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>;
    using node_selection_container_t = adaptive_affinity_table<Graph_t, cost_t, active_schedule_t, window_size>;
    using kl_gain_update_info = kl_update_info<VertexType>;

    struct ThreadSearchContext {

        unsigned thread_id = 0;
        unsigned start_step = 0;
        unsigned end_step = 0;
        unsigned original_end_step = 0;

        vector_vertex_lock_manger<VertexType> lock_manager;
        heap_datastructure max_gain_heap;
        node_selection_container_t affinity_table;
        std::vector<std::vector<cost_t>> local_affinity_table;
        reward_penalty_strategy<cost_t, comm_cost_function_t, active_schedule_t> reward_penalty_strat;
        vertex_selection_strategy<Graph_t, node_selection_container_t, active_schedule_t> selection_strategy;
        thread_local_active_schedule_data<Graph_t, cost_t> active_schedule_data;

        double average_gain = 0.0;
        unsigned max_inner_iterations = 0;
        unsigned no_improvement_iterations_reduce_penalty = 0; 
        unsigned min_inner_iter = 0;
        unsigned no_improvement_iterations_increase_inner_iter = 0;
        unsigned step_selection_epoch_counter = 0;
        unsigned step_selection_counter = 0;
        unsigned step_to_remove = 0;
        unsigned local_search_start_step = 0;
        unsigned unlock_edge_backtrack_counter = 0;
        unsigned unlock_edge_backtrack_counter_reset = 0;
        unsigned max_no_vioaltions_removed_backtrack = 0;

        inline unsigned num_steps() const { return end_step - start_step + 1; }
        inline unsigned start_idx(const unsigned node_step) const { return node_step < start_step + window_size ? window_size - (node_step - start_step) : 0; }
        inline unsigned end_idx(unsigned node_step) const { return node_step + window_size <= end_step ? window_range : window_range - (node_step + window_size - end_step); }

    };

    bool compute_with_time_limit = false;

    BspSchedule<Graph_t> *input_schedule;
    const Graph_t *graph;
    const BspInstance<Graph_t> *instance;

    compatible_processor_range<Graph_t> proc_range;

    kl_parameter parameters;
    std::mt19937 gen;
    
    active_schedule_t active_schedule;
    comm_cost_function_t comm_cost_f;
    std::vector<ThreadSearchContext> thread_data_vec;
    std::vector<bool> thread_finished_vec;
    
    inline unsigned rel_step_idx(const unsigned node_step, const unsigned move_step) const { return (move_step >= node_step) ? ((move_step - node_step) + window_size) : (window_size - (node_step - move_step)); }
    inline bool is_compatible(VertexType node, unsigned proc) const { return active_schedule.getInstance().isCompatible(node, proc); }

    void set_start_step(const unsigned step, ThreadSearchContext& thread_data) {
        thread_data.start_step = step;
        thread_data.step_to_remove = step;
        thread_data.step_selection_counter = step;
       
        thread_data.average_gain = 0.0;
        thread_data.max_inner_iterations = parameters.max_inner_iterations_reset;
        thread_data.no_improvement_iterations_reduce_penalty = parameters.max_no_improvement_iterations / 5;
        thread_data.min_inner_iter = parameters.min_inner_iter_reset;
        thread_data.step_selection_epoch_counter = 0;
        thread_data.no_improvement_iterations_increase_inner_iter = 10;        
        thread_data.unlock_edge_backtrack_counter_reset = 0;
        thread_data.unlock_edge_backtrack_counter = thread_data.unlock_edge_backtrack_counter_reset;        
        thread_data.max_no_vioaltions_removed_backtrack = parameters.max_no_vioaltions_removed_backtrack_reset;
    }


    kl_move get_best_move(node_selection_container_t & affinity_table, vector_vertex_lock_manger<VertexType> & lock_manager, heap_datastructure & max_gain_heap) {
        // To introduce non-determinism and help escape local optima, if there are multiple moves with the same
        // top gain, we randomly select one. We check up to `local_max` ties.
        const unsigned local_max = 50;
        std::vector<VertexType> top_gain_nodes = max_gain_heap.get_top_keys(local_max);

        if (top_gain_nodes.empty()) {
            // This case is guarded by the caller, but for safety:
            top_gain_nodes.push_back(max_gain_heap.top());
        }

        std::uniform_int_distribution<size_t> dis(0, top_gain_nodes.size() - 1);
        const VertexType node = top_gain_nodes[dis(gen)];

        kl_move best_move = max_gain_heap.get_value(node);
        max_gain_heap.erase(node);
        lock_manager.lock(node);
        affinity_table.remove(node);

        return best_move;
    }
    
    inline void process_other_steps_best_move(const unsigned idx, const unsigned node_step, const VertexType& node, const cost_t affinity_current_proc_step, cost_t& max_gain, unsigned& max_proc, unsigned& max_step, const std::vector<std::vector<cost_t>> &affinity_table_node) const {    
        for (const unsigned p : proc_range.compatible_processors_vertex(node)) {
            if constexpr (active_schedule.use_memory_constraint) {
                if( not active_schedule.memory_constraint.can_move(node, p, node_step + idx - window_size)) continue;                
            }

            const cost_t gain = affinity_current_proc_step - affinity_table_node[p][idx];
            if (gain > max_gain) {
                max_gain = gain;
                max_proc = p;
                max_step = idx; 
            }
        }
    }

    template<bool move_to_same_super_step>
    kl_move compute_best_move(VertexType node, const std::vector<std::vector<cost_t>> &affinity_table_node, ThreadSearchContext & thread_data) {
        const unsigned node_step = active_schedule.assigned_superstep(node);
        const unsigned node_proc = active_schedule.assigned_processor(node);

        cost_t max_gain = std::numeric_limits<cost_t>::lowest();

        unsigned max_proc = std::numeric_limits<unsigned>::max();
        unsigned max_step = std::numeric_limits<unsigned>::max();

        const cost_t affinity_current_proc_step = affinity_table_node[node_proc][window_size];

        unsigned idx = thread_data.start_idx(node_step);
        for (; idx < window_size; idx++) {
            process_other_steps_best_move(idx, node_step, node, affinity_current_proc_step, max_gain, max_proc, max_step, affinity_table_node);
        }

        if constexpr (move_to_same_super_step) {
            for (const unsigned proc : proc_range.compatible_processors_vertex(node)) {
                if (proc == node_proc)
                    continue;

                if constexpr (active_schedule.use_memory_constraint) {
                    if( not active_schedule.memory_constraint.can_move(node, proc, node_step + idx - window_size)) continue;                
                }

                const cost_t gain = affinity_current_proc_step - affinity_table_node[proc][window_size];
                if (gain > max_gain) {
                    max_gain = gain;
                    max_proc = proc;
                    max_step = idx; 
                }
            }
        }

        idx++;

        const unsigned bound = thread_data.end_idx(node_step);
        for (; idx < bound; idx++) {
            process_other_steps_best_move(idx, node_step, node, affinity_current_proc_step, max_gain, max_proc, max_step, affinity_table_node);
        }

        return kl_move(node, max_gain, node_proc, node_step, max_proc, node_step + max_step - window_size);
    }
  
    kl_gain_update_info update_node_work_affinity_after_move(VertexType node, kl_move move, const pre_move_work_data<work_weight_t> & prev_work_data, std::vector<std::vector<cost_t>> &affinity_table_node) {
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
                        affinity_table_node[node_proc][window_size] += (new_node_proc_affinity - prev_node_proc_affinity);                    
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
                               
                            affinity_table_node[proc][window_size] += (other_affinity - prev_other_affinity);                            
                        }
                    }  
                    
                    if (node_proc != move.from_proc && is_compatible(node, move.from_proc)) {
                        const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move.from_proc) + graph->vertex_work_weight(move.node);
                        const cost_t prev_other_affinity = compute_same_step_affinity(prev_max_work, prev_new_weight, prev_node_proc_affinity);   
                        const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move.from_proc);
                        const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);           
                        affinity_table_node[move.from_proc][window_size] += (other_affinity - prev_other_affinity);  
                    } 
                    
                    if (node_proc != move.to_proc && is_compatible(node, move.to_proc)) {
                        const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move.to_proc) - graph->vertex_work_weight(move.node);
                        const cost_t prev_other_affinity = compute_same_step_affinity(prev_max_work, prev_new_weight, prev_node_proc_affinity);      
                        const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move.to_proc);
                        const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);            
                        affinity_table_node[move.to_proc][window_size] += (other_affinity - prev_other_affinity);  
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
                                affinity_table_node[proc][idx] += new_affinity - prev_affinity;                                              
                            } else if (proc == move.to_proc) {
                                const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(move.to_step, proc) - graph->vertex_work_weight(move.node);
                                const cost_t prev_affinity = prev_max_work < prev_new_weight ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_max_work) : 0.0;
                                const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                                affinity_table_node[proc][idx] += new_affinity - prev_affinity;
                            } else {
                                const cost_t prev_affinity = prev_max_work < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(prev_max_work) : 0.0;
                                const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                                affinity_table_node[proc][idx] += new_affinity - prev_affinity;  
                            }
                        }                            
                    } else {
                        // update only move.from_proc and move.to_proc
                        if (is_compatible(node, move.from_proc)) {
                            const work_weight_t from_new_weight = vertex_weight + active_schedule.get_step_processor_work(move.from_step, move.from_proc);
                            const work_weight_t from_prev_new_weight = from_new_weight + graph->vertex_work_weight(move.node);
                            const cost_t from_prev_affinity = prev_max_work < from_prev_new_weight ? static_cast<cost_t>(from_prev_new_weight) - static_cast<cost_t>(prev_max_work) : 0.0;

                            const cost_t from_new_affinity = new_max_weight < from_new_weight ? static_cast<cost_t>(from_new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                            affinity_table_node[move.from_proc][idx] += from_new_affinity - from_prev_affinity;
                        }

                        if (is_compatible(node, move.to_proc)) {
                            const work_weight_t to_new_weight = vertex_weight + active_schedule.get_step_processor_work(move.to_step, move.to_proc);
                            const work_weight_t to_prev_new_weight = to_new_weight - graph->vertex_work_weight(move.node);
                            const cost_t to_prev_affinity = prev_max_work < to_prev_new_weight ? static_cast<cost_t>(to_prev_new_weight) - static_cast<cost_t>(prev_max_work) : 0.0;

                            const cost_t to_new_affinity = new_max_weight < to_new_weight ? static_cast<cost_t>(to_new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                            affinity_table_node[move.to_proc][idx] += to_new_affinity - to_prev_affinity;
                        }
                    }
                }
            }
            
        } else {            
            const unsigned node_proc = active_schedule.assigned_processor(node);
            process_work_update_step(node, node_step, node_proc, vertex_weight, move.from_step, move.from_proc, graph->vertex_work_weight(move.node), prev_work_data.from_step_max_work, prev_work_data.from_step_second_max_work, prev_work_data.from_step_max_work_processor_count, update_info.update_from_step, update_info.update_entire_from_step, update_info.full_update, affinity_table_node);
            process_work_update_step(node, node_step, node_proc, vertex_weight, move.to_step, move.to_proc, -graph->vertex_work_weight(move.node), prev_work_data.to_step_max_work, prev_work_data.to_step_second_max_work, prev_work_data.to_step_max_work_processor_count, update_info.update_to_step, update_info.update_entire_to_step, update_info.full_update, affinity_table_node);
        }

        return update_info;
    }

    void process_work_update_step(VertexType node, unsigned node_step, unsigned node_proc, work_weight_t vertex_weight, unsigned move_step, unsigned move_proc, work_weight_t move_correction_node_weight, const work_weight_t prev_move_step_max_work, const work_weight_t prev_move_step_second_max_work, unsigned prev_move_step_max_work_processor_count, bool & update_step, bool & update_entire_step, bool & full_update, std::vector<std::vector<cost_t>> &affinity_table_node);
    void update_node_work_affinity(node_selection_container_t &nodes, kl_move move, const pre_move_work_data<work_weight_t> & prev_work_data, std::map<VertexType, kl_gain_update_info> &recompute_max_gain);
    void update_best_move(VertexType node, unsigned step, unsigned proc, node_selection_container_t &affinity_table, ThreadSearchContext & thread_data);
    void update_best_move(VertexType node, unsigned step, node_selection_container_t &affinity_table, ThreadSearchContext & thread_data);
    void update_max_gain(kl_move move, std::map<VertexType, kl_gain_update_info> &recompute_max_gain, ThreadSearchContext & thread_data);
    void compute_work_affinity(VertexType node, std::vector<std::vector<cost_t>> & affinity_table_node, ThreadSearchContext & thread_data);

    inline void recompute_node_max_gain(VertexType node, node_selection_container_t &affinity_table, ThreadSearchContext & thread_data) {
        const auto best_move = compute_best_move<true>(node, affinity_table[node], thread_data);
        thread_data.max_gain_heap.update(node, best_move);   
    }

    inline cost_t compute_same_step_affinity(const work_weight_t &max_work_for_step, const work_weight_t &new_weight, const cost_t &node_proc_affinity) {
        const cost_t max_work_after_removal = static_cast<cost_t>(max_work_for_step) - node_proc_affinity;
        if (new_weight > max_work_after_removal) {
            return new_weight - max_work_after_removal;
        }
        return 0.0;
    }
    
    inline cost_t apply_move(kl_move move, ThreadSearchContext & thread_data) {
        active_schedule.apply_move(move, thread_data.active_schedule_data);
        comm_cost_f.update_datastructure_after_move(move, thread_data.start_step, thread_data.end_step); 
        cost_t change_in_cost = -move.gain;
        change_in_cost += static_cast<cost_t>(thread_data.active_schedule_data.resolved_violations.size()) * thread_data.reward_penalty_strat.reward;
        change_in_cost -= static_cast<cost_t>(thread_data.active_schedule_data.new_violations.size()) * thread_data.reward_penalty_strat.penalty;
  
#ifdef KL_DEBUG
        std::cout << "penalty: " << thread_data.reward_penalty_strat.penalty << " num violations: " << thread_data.active_schedule_data.current_violations.size() <<  " num new violations: " << thread_data.active_schedule_data.new_violations.size() << ", num resolved violations: " << thread_data.active_schedule_data.resolved_violations.size() <<  ", reward: " << thread_data.reward_penalty_strat.reward << std::endl;
        std::cout << "apply move, previous cost: " << thread_data.active_schedule_data.cost << ", new cost: " << thread_data.active_schedule_data.cost + change_in_cost << ", " << (thread_data.active_schedule_data.feasible ? "feasible," : "infeasible,") << std::endl;
#endif
        
        thread_data.active_schedule_data.update_cost(change_in_cost);
        
        return change_in_cost;
    }    

    void run_quick_moves(unsigned & inner_iter, ThreadSearchContext & thread_data, const cost_t change_in_cost, const VertexType best_move_node) {
#ifdef KL_DEBUG
        std::cout << "Starting quick moves sequence." << std::endl;
#endif
        inner_iter++;

        const size_t num_applied_moves = thread_data.active_schedule_data.applied_moves.size() - 1;
        const cost_t saved_cost = thread_data.active_schedule_data.cost - change_in_cost;

        std::unordered_set<VertexType> local_lock;
        local_lock.insert(best_move_node);
        std::vector<VertexType> quick_moves_stack;
        quick_moves_stack.reserve(10 + thread_data.active_schedule_data.new_violations.size() * 2);

        for (const auto& [key, value] : thread_data.active_schedule_data.new_violations) {
            quick_moves_stack.push_back(key);
        }

        while (quick_moves_stack.size() > 0) {

            auto next_node_to_move = quick_moves_stack.back();
            quick_moves_stack.pop_back();
            
            thread_data.reward_penalty_strat.init_reward_penalty(static_cast<double>(thread_data.active_schedule_data.current_violations.size()) + 1.0);
            compute_node_affinities(next_node_to_move, thread_data.local_affinity_table, thread_data);
            kl_move best_quick_move = compute_best_move<true>(next_node_to_move, thread_data.local_affinity_table, thread_data);

            local_lock.insert(next_node_to_move);
            if (best_quick_move.gain <= std::numeric_limits<cost_t>::lowest()) {
                continue;
            }

#ifdef KL_DEBUG
            std::cout << " >>> move node " << best_quick_move.node << " with gain " << best_quick_move.gain << ", from proc|step: " << best_quick_move.from_proc << "|" << best_quick_move.from_step << " to: " << best_quick_move.to_proc << "|" << best_quick_move.to_step << std::endl;
#endif

            apply_move(best_quick_move, thread_data);                          
            inner_iter++;

            if (thread_data.active_schedule_data.new_violations.size() > 0) {
                bool abort = false;

                for (const auto& [key, value] : thread_data.active_schedule_data.new_violations) {
                    if(local_lock.find(key) != local_lock.end()) {
                        abort = true;
                        break;
                    }                                    
                    quick_moves_stack.push_back(key);
                }

                if (abort) break;

            } else if (thread_data.active_schedule_data.feasible) {
                break;
            }
        }

        if (!thread_data.active_schedule_data.feasible) {
            active_schedule.revert_schedule_to_bound(num_applied_moves, saved_cost ,true, comm_cost_f, thread_data.active_schedule_data, thread_data.start_step, thread_data.end_step);
#ifdef KL_DEBUG
            std::cout << "Ending quick moves sequence with infeasible solution." << std::endl;
#endif
        } 
#ifdef KL_DEBUG
        else {
            std::cout << "Ending quick moves sequence with feasible solution." << std::endl;
        }
#endif

        thread_data.affinity_table.trim();
        thread_data.max_gain_heap.clear();
        thread_data.reward_penalty_strat.init_reward_penalty(1.0);
        insert_gain_heap(thread_data); // Re-initialize the heap with the current state
    }

    void resolve_violations(ThreadSearchContext & thread_data) {    
        auto & current_violations = thread_data.active_schedule_data.current_violations;
        unsigned num_violations = static_cast<unsigned>(current_violations.size());
        if (num_violations > 0) {

#ifdef KL_DEBUG_1
            std::cout << "thread " << thread_data.thread_id << ", Starting preresolving violations with " << num_violations << " initial violations" << std::endl;
#endif
            thread_data.reward_penalty_strat.init_reward_penalty(static_cast<double>(num_violations) + 1.0);
             std::unordered_set<VertexType> local_lock;
            unsigned num_iter = 0;
            const unsigned min_iter = num_violations / 4; 
            while (not current_violations.empty()) {
                std::uniform_int_distribution<size_t> dis(0, current_violations.size() - 1);
                auto it = current_violations.begin();
                std::advance(it, dis(gen));
                const auto &next_edge = *it;
                const VertexType source_v = source(next_edge, *graph);
                const VertexType target_v = target(next_edge, *graph);
                const bool source_locked = local_lock.find(source_v) != local_lock.end();
                const bool target_locked = local_lock.find(target_v) != local_lock.end();
                    
                if (source_locked && target_locked) {
#ifdef KL_DEBUG_1
                    std::cout << "source, target locked" << std::endl;
#endif
                    break;
                }
                
                kl_move best_move;
                if (source_locked || target_locked) {
                    const VertexType node = source_locked ? target_v : source_v;
                    compute_node_affinities(node, thread_data.local_affinity_table, thread_data);
                    best_move = compute_best_move<true>(node, thread_data.local_affinity_table, thread_data);
                } else {
                    compute_node_affinities(source_v, thread_data.local_affinity_table, thread_data);
                    kl_move best_source_v_move = compute_best_move<true>(source_v, thread_data.local_affinity_table, thread_data);
                    compute_node_affinities(target_v, thread_data.local_affinity_table, thread_data);
                    kl_move best_target_v_move = compute_best_move<true>(target_v, thread_data.local_affinity_table, thread_data);
                    best_move = best_target_v_move.gain > best_source_v_move.gain ? std::move(best_target_v_move) : std::move(best_source_v_move);
                }

                local_lock.insert(best_move.node);
                if (best_move.gain <= std::numeric_limits<cost_t>::lowest()) continue;

                apply_move(best_move, thread_data);
                thread_data.affinity_table.insert(best_move.node);
#ifdef KL_DEBUG_1
        std::cout << "move node " << best_move.node << " with gain " << best_move.gain << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step << " to: " << best_move.to_proc << "|" << best_move.to_step << std::endl;
#endif
                const unsigned new_num_violations = static_cast<unsigned>(current_violations.size());
                if (new_num_violations == 0) break;

                if (thread_data.active_schedule_data.new_violations.size() > 0) {                
                    for (const auto & [vertex, edge] : thread_data.active_schedule_data.new_violations) {
                        thread_data.affinity_table.insert(vertex);
                    }
                }

                const double gain = static_cast<double>(num_violations) - static_cast<double>(new_num_violations);
                num_violations = new_num_violations;
                update_avg_gain(gain, num_iter++, thread_data.average_gain);
#ifdef KL_DEBUG_1
                std::cout << "thread " << thread_data.thread_id << ",  preresolving violations with " << num_violations << " violations, " << num_iter << " #iterations, " << thread_data.average_gain << " average gain" << std::endl;
#endif
                if (num_iter > min_iter && thread_data.average_gain < 0.0) {
                    break;
                }
            }
            thread_data.average_gain = 0.0;
        } 
    }

    void run_local_search(ThreadSearchContext & thread_data) {

#ifdef KL_DEBUG_1
        std::cout << "thread " << thread_data.thread_id << ", start local search, initial schedule cost: " << thread_data.active_schedule_data.cost << " with " << thread_data.num_steps() << " supersteps." << std::endl;
#endif
        std::vector<VertexType> new_nodes;
        std::vector<VertexType> unlock_nodes;
        std::map<VertexType, kl_gain_update_info> recompute_max_gain;

        const auto start_time = std::chrono::high_resolution_clock::now();

        unsigned no_improvement_iter_counter = 0;
        unsigned outer_iter = 0;

        for (; outer_iter < parameters.max_outer_iterations; outer_iter++) {
            cost_t initial_inner_iter_cost = thread_data.active_schedule_data.cost;

            reset_inner_search_structures(thread_data);            
            select_active_nodes(thread_data);              
            thread_data.reward_penalty_strat.init_reward_penalty(static_cast<double>(thread_data.active_schedule_data.current_violations.size()) + 1.0);
            insert_gain_heap(thread_data);
            
            unsigned inner_iter = 0;
            unsigned violation_removed_count = 0;
            unsigned reset_counter = 0;
            bool iter_inital_feasible = thread_data.active_schedule_data.feasible;
                        
#ifdef KL_DEBUG
            std::cout << "------ start inner loop ------" << std::endl;
            std::cout << "initial node selection: {";
            for (size_t i = 0; i < thread_data.affinity_table.size() ; ++i) {
                std::cout << thread_data.affinity_table.get_selected_nodes()[i] << ", ";
            }
            std::cout << "}" << std::endl;
#endif
#ifdef KL_DEBUG_1
            if (not iter_inital_feasible) {
                std::cout << "initial solution not feasible, num violations: " << thread_data.active_schedule_data.current_violations.size() << ". Penalty: " << thread_data.reward_penalty_strat.penalty << ", reward: " << thread_data.reward_penalty_strat.reward << std::endl;
            }
#endif
#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001 ) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule.use_memory_constraint) {
                    if ( not active_schedule.memory_constraint.satisfied_memory_constraint())
                        std::cout << "memory constraint not satisfied" << std::endl;
                }
#endif


            while (inner_iter < thread_data.max_inner_iterations && thread_data.max_gain_heap.size() > 0) {
                kl_move best_move = get_best_move(thread_data.affinity_table, thread_data.lock_manager, thread_data.max_gain_heap); // locks best_move.node and removes it from node_selection
                if (best_move.gain <= std::numeric_limits<cost_t>::lowest()) {
                    break;
                }                
                update_avg_gain(best_move.gain, inner_iter, thread_data.average_gain);
#ifdef KL_DEBUG
        std::cout << " >>> move node " << best_move.node << " with gain " << best_move.gain << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step << " to: " << best_move.to_proc << "|" << best_move.to_step << ",avg gain: " << thread_data.average_gain << std::endl;
#endif
                if (inner_iter > thread_data.min_inner_iter && thread_data.average_gain < 0.0) {
#ifdef KL_DEBUG
            std::cout << "Negative average gain: " << thread_data.average_gain << ", end local search" << std::endl;
#endif
                            break;
                }

#ifdef KL_DEBUG
        if (not active_schedule.getInstance().isCompatible(best_move.node, best_move.to_proc)) {
            std::cout << "move to incompatibe node" << std::endl;
        }
#endif

                const auto prev_work_data = active_schedule.get_pre_move_work_data(best_move);
                const cost_t change_in_cost = apply_move(best_move, thread_data);
#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001 ) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule.use_memory_constraint) {
                    if ( not active_schedule.memory_constraint.satisfied_memory_constraint())
                        std::cout << "memory constraint not satisfied" << std::endl;
                }
#endif
                if constexpr (enable_quick_moves) {
                    if (iter_inital_feasible && thread_data.active_schedule_data.new_violations.size() > 0) {
                        run_quick_moves(inner_iter, thread_data, change_in_cost, best_move.node);
#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001 ) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule.use_memory_constraint) {
                    if ( not active_schedule.memory_constraint.satisfied_memory_constraint())
                        std::cout << "memory constraint not satisfied" << std::endl;
                }
#endif
                                continue;
                    }
                }

                if (thread_data.active_schedule_data.current_violations.size() > 0) {
                    if (thread_data.active_schedule_data.resolved_violations.size() > 0) {
                        violation_removed_count = 0;
                    } else {
                        violation_removed_count++;

                        if (violation_removed_count > 3) {
                            if (reset_counter < thread_data.max_no_vioaltions_removed_backtrack && ((not iter_inital_feasible) || (thread_data.active_schedule_data.cost < thread_data.active_schedule_data.best_cost))) {
                                thread_data.affinity_table.reset_node_selection();
                                thread_data.max_gain_heap.clear();
                                thread_data.lock_manager.clear();
                                thread_data.selection_strategy.select_nodes_violations(thread_data.affinity_table, thread_data.active_schedule_data.current_violations, thread_data.start_step, thread_data.end_step);
#ifdef KL_DEBUG
                        std::cout << "Infeasible, and no violations resolved for 5 iterations, reset node selection" << std::endl;
#endif
                                thread_data.reward_penalty_strat.init_reward_penalty(static_cast<double>(thread_data.active_schedule_data.current_violations.size()));
                                insert_gain_heap(thread_data);

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
                                
                if(is_local_search_blocked(thread_data)) {
                    if (not blocked_edge_strategy(best_move.node, unlock_nodes, thread_data)) {
                                break;
                    }
                }

                thread_data.affinity_table.trim();

                update_node_work_affinity(thread_data.affinity_table, best_move, prev_work_data, recompute_max_gain);
                comm_cost_f.update_node_comm_affinity(best_move, thread_data, thread_data.reward_penalty_strat.penalty, thread_data.reward_penalty_strat.reward, recompute_max_gain, new_nodes);

                for (const auto v : unlock_nodes) {
                    thread_data.lock_manager.unlock(v);
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
#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001 ) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule.use_memory_constraint) {
                    if ( not active_schedule.memory_constraint.satisfied_memory_constraint())
                        std::cout << "memory constraint not satisfied" << std::endl;
                }
#endif
                update_max_gain(best_move, recompute_max_gain, thread_data);
                insert_new_nodes_gain_heap(new_nodes, thread_data.affinity_table, thread_data);

                recompute_max_gain.clear();
                new_nodes.clear();

                inner_iter++;
            }

#ifdef KL_DEBUG
            std::cout << "--- end inner loop after " << inner_iter << " inner iterations, gain heap size: " << thread_data.max_gain_heap.size() <<  ", outer iteraion " << outer_iter << "/" << parameters.max_outer_iterations << ", current cost: " << thread_data.active_schedule_data.cost << ", " << (thread_data.active_schedule_data.feasible ? "feasible" : "infeasible") << std::endl;
#endif
#ifdef KL_DEBUG_1
            const unsigned num_steps_tmp = thread_data.end_step;            
#endif
            active_schedule.revert_to_best_schedule(thread_data.local_search_start_step, thread_data.step_to_remove, comm_cost_f, thread_data.active_schedule_data, thread_data.start_step, thread_data.end_step);
#ifdef KL_DEBUG_1
            if (thread_data.local_search_start_step > 0) {
                if(num_steps_tmp == thread_data.end_step) {
                    std::cout << "thread " << thread_data.thread_id << ", removing step " << thread_data.step_to_remove << " succeded " << std::endl;
                } else {
                    std::cout << "thread " << thread_data.thread_id << ", removing step " << thread_data.step_to_remove << " failed " << std::endl;
                }
            } 
#endif


#ifdef KL_DEBUG_COST_CHECK
            active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
            if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001 ) {
                std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
            }
            if constexpr (active_schedule.use_memory_constraint) {
                if ( not active_schedule.memory_constraint.satisfied_memory_constraint())
                    std::cout << "memory constraint not satisfied" << std::endl;
            }
#endif

            if (compute_with_time_limit) {
                auto finish_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();
                if (duration > ImprovementScheduler<Graph_t>::timeLimitSeconds) {
                    break;
                }
            }
  
            if (other_threads_finished(thread_data.thread_id)) {
#ifdef KL_DEBUG_1
                std::cout << "thread " << thread_data.thread_id << ", other threads finished, end local search" << std::endl;
#endif
                break;
            }

            if (initial_inner_iter_cost <= thread_data.active_schedule_data.cost) {
                no_improvement_iter_counter++;

                if (no_improvement_iter_counter >= parameters.max_no_improvement_iterations) {
#ifdef KL_DEBUG_1
                    std::cout << "thread " << thread_data.thread_id << ", no improvement for " << parameters.max_no_improvement_iterations
                              << " iterations, end local search" << std::endl;
#endif
                    break;
                }    
            } else {
                no_improvement_iter_counter = 0;
            } 
            
            adjust_local_search_parameters(outer_iter, no_improvement_iter_counter, thread_data);
        }

#ifdef KL_DEBUG_1
        std::cout << "thread " << thread_data.thread_id << ", local search end after " << outer_iter << " outer iterations, current cost: " << thread_data.active_schedule_data.cost << " with " << thread_data.num_steps() << " supersteps, vs serial cost " << active_schedule.get_total_work_weight() << "." << std::endl;
#endif
        thread_finished_vec[thread_data.thread_id] = true;

    }

    bool other_threads_finished(const unsigned thread_id) {
        const size_t num_threads = thread_finished_vec.size();
        if(num_threads == 1)
            return false;

        for (size_t i = 0; i < num_threads; i++) {
            if (i != thread_id && !thread_finished_vec[i]) 
                return false;
        }
        return true;
    }

    inline bool blocked_edge_strategy(VertexType node, std::vector<VertexType> & unlock_nodes, ThreadSearchContext & thread_data) {
        if (thread_data.unlock_edge_backtrack_counter > 1) {
            for (const auto [v,e] : thread_data.active_schedule_data.new_violations) {
                const auto source_v = source(e, *graph);
                const auto target_v = target(e, *graph);

                if (node == source_v && thread_data.lock_manager.is_locked(target_v)) {
                    unlock_nodes.push_back(target_v);
                } else if (node == target_v && thread_data.lock_manager.is_locked(source_v)) {
                    unlock_nodes.push_back(source_v);
                }
            }
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, backtrack counter: " << thread_data.unlock_edge_backtrack_counter <<  std::endl;
#endif
            thread_data.unlock_edge_backtrack_counter--;
            return true;
        } else {
#ifdef KL_DEBUG
            std::cout << "Nodes of violated edge locked, end local search" << std::endl;
#endif
            return false;  //or reset local search and initalize with violating nodes
        }
    }

    inline void adjust_local_search_parameters(unsigned outer_iter, unsigned no_imp_counter, ThreadSearchContext & thread_data) {
        if (no_imp_counter >= thread_data.no_improvement_iterations_reduce_penalty && thread_data.reward_penalty_strat.initial_penalty > 1.0) {
            thread_data.reward_penalty_strat.initial_penalty = std::floor(std::sqrt(thread_data.reward_penalty_strat.initial_penalty));
            thread_data.unlock_edge_backtrack_counter_reset += 1;
            thread_data.no_improvement_iterations_reduce_penalty += 15;
#ifdef KL_DEBUG_1
            std::cout << "thread " << thread_data.thread_id << ", no improvement for " << thread_data.no_improvement_iterations_reduce_penalty
                        << " iterations, reducing initial penalty to " << thread_data.reward_penalty_strat.initial_penalty << std::endl;
#endif                   
        } 

        if (parameters.try_remove_step_after_num_outer_iterations > 0 && ((outer_iter + 1) % parameters.try_remove_step_after_num_outer_iterations) == 0) {
            thread_data.step_selection_epoch_counter = 0;;
#ifdef KL_DEBUG
            std::cout << "reset remove epoc counter after " << outer_iter << " iterations." << std::endl;
#endif
        }

        if (no_imp_counter >= thread_data.no_improvement_iterations_increase_inner_iter ) {
            thread_data.min_inner_iter = static_cast<unsigned>(std::ceil(thread_data.min_inner_iter * 2.2));
            thread_data.no_improvement_iterations_increase_inner_iter += 20;
#ifdef KL_DEBUG_1
            std::cout << "thread " << thread_data.thread_id << ", no improvement for " << thread_data.no_improvement_iterations_increase_inner_iter
                        << " iterations, increasing min inner iter to " << thread_data.min_inner_iter << std::endl;
#endif
        }

    }
    
    bool is_local_search_blocked(ThreadSearchContext & thread_data);
    void set_parameters(vertex_idx_t<Graph_t> num_nodes);
    void reset_inner_search_structures(ThreadSearchContext & thread_data) const;
    void initialize_datastructures(BspSchedule<Graph_t> &schedule);
    void print_heap(heap_datastructure & max_gain_heap) const;
    void cleanup_datastructures();
    void update_avg_gain(const cost_t gain, const unsigned num_iter, cost_t & average_gain); 
    void insert_gain_heap(ThreadSearchContext & thread_data);
    void insert_new_nodes_gain_heap(std::vector<VertexType>& new_nodes, node_selection_container_t &nodes, ThreadSearchContext & thread_data);

    inline void compute_node_affinities(VertexType node, std::vector<std::vector<cost_t>> & affinity_table_node, ThreadSearchContext & thread_data) {
        compute_work_affinity(node, affinity_table_node, thread_data);
        comm_cost_f.compute_comm_affinity(node, affinity_table_node, thread_data.reward_penalty_strat.penalty, thread_data.reward_penalty_strat.reward, thread_data.start_step, thread_data.end_step);
    }

    void select_active_nodes(ThreadSearchContext & thread_data) {
        if (select_nodes_check_remove_superstep(thread_data.step_to_remove, thread_data)) {
            active_schedule.swap_empty_step_fwd(thread_data.step_to_remove, thread_data.end_step);
            thread_data.end_step--;
            thread_data.local_search_start_step = static_cast<unsigned>(thread_data.active_schedule_data.applied_moves.size());
            thread_data.active_schedule_data.update_cost(-1.0 * static_cast<cost_t>(instance->synchronisationCosts()));

            if constexpr (enable_preresolving_violations) {
                resolve_violations(thread_data);
            }

            if (thread_data.active_schedule_data.current_violations.size() > parameters.initial_violation_threshold) {
                active_schedule.revert_to_best_schedule(thread_data.local_search_start_step, thread_data.step_to_remove, comm_cost_f, thread_data.active_schedule_data, thread_data.start_step, thread_data.end_step);
            } else {
                thread_data.unlock_edge_backtrack_counter = static_cast<unsigned>(thread_data.active_schedule_data.current_violations.size());
                thread_data.max_inner_iterations = std::max(thread_data.unlock_edge_backtrack_counter * 5u, parameters.max_inner_iterations_reset);
                thread_data.max_no_vioaltions_removed_backtrack = parameters.max_no_vioaltions_removed_backtrack_for_remove_step_reset;
    #ifdef KL_DEBUG_1
                std::cout << "thread " << thread_data.thread_id << ", Trying to remove step " << thread_data.step_to_remove << std::endl;
    #endif
                return; 
            }       
        }
        //thread_data.step_to_remove = thread_data.start_step;
        thread_data.local_search_start_step = 0;
        thread_data.selection_strategy.select_active_nodes(thread_data.affinity_table, thread_data.start_step, thread_data.end_step);
    }

    bool check_remove_superstep(unsigned step);
    bool select_nodes_check_remove_superstep(unsigned & step, ThreadSearchContext & thread_data);

    bool scatter_nodes_superstep(unsigned step, ThreadSearchContext & thread_data) {
        assert(step <= thread_data.end_step && thread_data.start_step <= step);
        bool abort = false;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {   
            const std::vector<VertexType> step_proc_node_vec(active_schedule.getSetSchedule().step_processor_vertices[step][proc].begin(),active_schedule.getSetSchedule().step_processor_vertices[step][proc].end());
            for (const auto &node : step_proc_node_vec) {         
                   
                thread_data.reward_penalty_strat.init_reward_penalty(static_cast<double>(thread_data.active_schedule_data.current_violations.size()) + 1.0);
                compute_node_affinities(node, thread_data.local_affinity_table, thread_data);
                kl_move best_move = compute_best_move<false>(node, thread_data.local_affinity_table, thread_data);

                if (best_move.gain <= std::numeric_limits<double>::lowest()) {
                    abort = true;
                    break;
                }

                apply_move(best_move, thread_data);
                if (thread_data.active_schedule_data.current_violations.size() > parameters.abort_scatter_nodes_violation_threshold) {
                    abort = true;
                    break;
                }

                thread_data.affinity_table.insert(node);
                //thread_data.selection_strategy.add_neighbours_to_selection(node, thread_data.affinity_table, thread_data.start_step, thread_data.end_step);
                if (thread_data.active_schedule_data.new_violations.size() > 0) {
                
                    for (const auto & [vertex, edge] : thread_data.active_schedule_data.new_violations) {
                        thread_data.affinity_table.insert(vertex);
                    }
                }

#ifdef KL_DEBUG
                std::cout << "move node " << best_move.node << " with gain " << best_move.gain << ", from proc|step: " << best_move.from_proc << "|" << best_move.from_step << " to: " << best_move.to_proc << "|" << best_move.to_step << std::endl;
#endif

#ifdef KL_DEBUG_COST_CHECK
                active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
                if (std::abs(comm_cost_f.compute_schedule_cost_test() - thread_data.active_schedule_data.cost) > 0.00001 ) {
                    std::cout << "computed cost: " << comm_cost_f.compute_schedule_cost_test() << ", current cost: " << thread_data.active_schedule_data.cost << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>> compute cost not equal to new cost <<<<<<<<<<<<<<<<<<<<" << std::endl;
                }
                if constexpr (active_schedule.use_memory_constraint) {
                    if ( not active_schedule.memory_constraint.satisfied_memory_constraint())
                     std::cout << "memory constraint not satisfied" << std::endl;
                }
#endif
                
            }

            if (abort) {
                break;
            }
        }

        if (abort) {
            active_schedule.revert_to_best_schedule(0, 0, comm_cost_f, thread_data.active_schedule_data, thread_data.start_step, thread_data.end_step);
            thread_data.affinity_table.reset_node_selection();
            return false;
        }
        return true;
    }

    void synchronize_active_schedule(const unsigned num_threads) {
        if (num_threads == 1) { // single thread case
            active_schedule.set_cost(thread_data_vec[0].active_schedule_data.cost);
            active_schedule.getVectorSchedule().number_of_supersteps = thread_data_vec[0].num_steps();
            return;        
        }

        unsigned write_cursor = thread_data_vec[0].end_step + 1;
        for (unsigned i = 1; i < num_threads; ++i) {
            auto& thread = thread_data_vec[i];
            if (thread.start_step <= thread.end_step) {
                for (unsigned j = thread.start_step; j <= thread.end_step; ++j) {
                    if (j != write_cursor) {
                        active_schedule.swap_steps(j, write_cursor);
                    }
                    write_cursor++;
                }
            }
        }
        active_schedule.getVectorSchedule().number_of_supersteps = write_cursor;
        const cost_t new_cost = comm_cost_f.compute_schedule_cost();
        active_schedule.set_cost(new_cost);
    }

  public:
    kl_improver() : ImprovementScheduler<Graph_t>() {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    explicit kl_improver(unsigned seed) : ImprovementScheduler<Graph_t>() {
        gen = std::mt19937(seed);
    }

    virtual ~kl_improver() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule<Graph_t> &schedule) override {
        if (schedule.getInstance().numberOfProcessors() < 2)
            return RETURN_STATUS::BEST_FOUND;

        const unsigned num_threads = 1;
        
        thread_data_vec.resize(num_threads);      
        thread_finished_vec.assign(num_threads, true);

        set_parameters(schedule.getInstance().numberOfVertices());
        initialize_datastructures(schedule); 
        const cost_t initial_cost = active_schedule.get_cost();   
        const unsigned num_steps = schedule.numberOfSupersteps();

        set_start_step(0, thread_data_vec[0]);
        thread_data_vec[0].end_step = (num_steps > 0) ? num_steps - 1 : 0;                   

        auto & thread_data = this->thread_data_vec[0];
        thread_data.active_schedule_data.initialize_cost(active_schedule.get_cost());
        thread_data.selection_strategy.setup(thread_data.start_step, thread_data.end_step);
        run_local_search(thread_data); 
            
        synchronize_active_schedule(num_threads);                       

        if (initial_cost > active_schedule.get_cost()) {
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

    virtual void setTimeQualityParameter(const double time_quality) { this->parameters.time_quality = time_quality; }
    virtual void setSuperstepRemoveStrengthParameter(const double superstep_remove_strength) { this->parameters.superstep_remove_strength = superstep_remove_strength; }

    virtual std::string getScheduleName() const {
        return "kl_improver_" + comm_cost_f.name();
    }
};

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::set_parameters(vertex_idx_t<Graph_t> num_nodes) {    
    const unsigned log_num_nodes = (num_nodes > 1) ? static_cast<unsigned>(std::log(num_nodes)) : 1; 

    // Total number of outer iterations. Proportional to sqrt N.
    parameters.max_outer_iterations = static_cast<unsigned>(std::sqrt(num_nodes) * (parameters.time_quality * 10.0) / parameters.num_parallel_loops);

    // Number of times to reset the search for violations before giving up.
    parameters.max_no_vioaltions_removed_backtrack_reset = parameters.time_quality < 0.75 ? 1 : parameters.time_quality < 1.0 ? 2 : 3;

    // Parameters for the superstep removal heuristic.
    parameters.max_no_vioaltions_removed_backtrack_for_remove_step_reset = 3 + static_cast<unsigned>(parameters.superstep_remove_strength * 7);
    parameters.node_max_step_selection_epochs = parameters.superstep_remove_strength < 0.75 ? 1 : parameters.superstep_remove_strength < 1.0 ? 2 : 3;
    parameters.remove_step_epocs = static_cast<unsigned>(parameters.superstep_remove_strength * 4.0);

    parameters.min_inner_iter_reset = static_cast<unsigned>(log_num_nodes + log_num_nodes * (1.0 + parameters.time_quality));
   
    if (parameters.remove_step_epocs > 0) {
        parameters.try_remove_step_after_num_outer_iterations = parameters.max_outer_iterations / parameters.remove_step_epocs;
    } else {
        // Effectively disable superstep removal if remove_step_epocs is 0.
        parameters.try_remove_step_after_num_outer_iterations = parameters.max_outer_iterations + 1;
    }
    
    unsigned i = 0;
    for (auto & thread : thread_data_vec) {
        thread.thread_id = i++;
        // The number of nodes to consider in each inner iteration. Proportional to log(N).
        thread.selection_strategy.selection_threshold = static_cast<std::size_t>(std::ceil(parameters.time_quality * 10 * log_num_nodes + log_num_nodes));  
    }

    #ifdef KL_DEBUG_1
                    std::cout << "kl set parameter, number of nodes: " << num_nodes << std::endl;
                    std::cout << "max outer iterations: " << parameters.max_outer_iterations << std::endl;
                    std::cout << "max inner iterations: " << parameters.max_inner_iterations_reset << std::endl; 
                    std::cout << "no improvement iterations reduce penalty: " << thread_data_vec[0].no_improvement_iterations_reduce_penalty << std::endl;
                    std::cout << "selction threshold: " << thread_data_vec[0].selection_strategy.selection_threshold << std::endl;
                    std::cout << "remove step epocs: " << parameters.remove_step_epocs << std::endl;
                    std::cout << "try remove step after num outer iterations: " << parameters.try_remove_step_after_num_outer_iterations << std::endl;  
                    std::cout << "number of parallel loops: " << parameters.num_parallel_loops << std::endl;                 
    #endif
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_node_work_affinity(node_selection_container_t &nodes, kl_move move, const pre_move_work_data<work_weight_t> & prev_work_data, std::map<VertexType, kl_gain_update_info> &recompute_max_gain) {
    const size_t active_count = nodes.size();

    for (size_t i = 0; i < active_count; ++i) {
        const VertexType node = nodes.get_selected_nodes()[i];
            
        kl_gain_update_info update_info = update_node_work_affinity_after_move(node, move, prev_work_data, nodes.at(node));
        if (update_info.update_from_step || update_info.update_to_step) {
            recompute_max_gain[node] = update_info;
        }        
    }
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_max_gain(kl_move move, std::map<VertexType, kl_gain_update_info> &recompute_max_gain, ThreadSearchContext & thread_data) {
    for (auto& pair : recompute_max_gain) { 
        if (pair.second.full_update) {
            recompute_node_max_gain(pair.first, thread_data.affinity_table, thread_data); 
        } else {
            if (pair.second.update_entire_from_step) {
                update_best_move(pair.first, move.from_step, thread_data.affinity_table, thread_data);
            } else if (pair.second.update_from_step && is_compatible(pair.first, move.from_proc)) {
                update_best_move(pair.first, move.from_step, move.from_proc, thread_data.affinity_table, thread_data);
            } 

            if (move.from_step != move.to_step || not pair.second.update_entire_from_step) {
                if (pair.second.update_entire_to_step) {
                    update_best_move(pair.first, move.to_step, thread_data.affinity_table, thread_data);
                } else if (pair.second.update_to_step && is_compatible(pair.first, move.to_proc)) {
                    update_best_move(pair.first, move.to_step, move.to_proc, thread_data.affinity_table, thread_data);
                }
            }
        } 
    }    
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::compute_work_affinity(VertexType node, std::vector<std::vector<cost_t>> & affinity_table_node, ThreadSearchContext & thread_data) {
    const unsigned node_step = active_schedule.assigned_superstep(node);
    const work_weight_t vertex_weight = graph->vertex_work_weight(node);

    unsigned step = (node_step > window_size) ? (node_step - window_size) : 0;
    for (unsigned idx = thread_data.start_idx(node_step); idx < thread_data.end_idx(node_step); ++idx, ++step) {
        if (idx == window_size) {
            continue;
        }

        const cost_t max_work_for_step = static_cast<cost_t>(active_schedule.get_step_max_work(step));

        for (const unsigned proc : proc_range.compatible_processors_vertex(node)) { 
            const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(step, proc);
            const cost_t work_diff = static_cast<cost_t>(new_weight) - max_work_for_step;
            affinity_table_node[proc][idx] = std::max(0.0, work_diff);
        }
    }

    const unsigned node_proc = active_schedule.assigned_processor(node);        
    const work_weight_t max_work_for_step = active_schedule.get_step_max_work(node_step);
    const bool is_sole_max_processor = (active_schedule.get_step_max_work_processor_count()[node_step] == 1) && (max_work_for_step == active_schedule.get_step_processor_work(node_step, node_proc));

    const cost_t node_proc_affinity = is_sole_max_processor ? std::min(vertex_weight, max_work_for_step - active_schedule.get_step_second_max_work(node_step)) : 0.0;
    affinity_table_node[node_proc][window_size] = node_proc_affinity;
    
    for (const unsigned proc : proc_range.compatible_processors_vertex(node)) { 
        if(proc == node_proc)
            continue;

        const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, proc);           
        affinity_table_node[proc][window_size] = compute_same_step_affinity(max_work_for_step, new_weight, node_proc_affinity);            
    }
}   

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::process_work_update_step(VertexType node, unsigned node_step, unsigned node_proc, work_weight_t vertex_weight, unsigned move_step, unsigned move_proc, work_weight_t move_correction_node_weight, const work_weight_t prev_move_step_max_work, const work_weight_t prev_move_step_second_max_work, unsigned prev_move_step_max_work_processor_count, bool & update_step, bool & update_entire_step, bool & full_update, std::vector<std::vector<cost_t>> & affinity_table_node) {
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
                affinity_table_node[node_proc][window_size] += (new_node_proc_affinity - prev_node_proc_affinity);
            }
    
            if ((prev_move_step_max_work != new_max_weight) || update_node_proc_affinity) {
                update_entire_step = true;

                for (const unsigned proc : proc_range.compatible_processors_vertex(node)) { 
                    if((proc == node_proc) || (proc == move_proc))
                        continue;

                    const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, proc);
                    const cost_t prev_other_affinity = compute_same_step_affinity(prev_move_step_max_work, new_weight, prev_node_proc_affinity);  
                    const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);            
    
                    affinity_table_node[proc][window_size] += (other_affinity - prev_other_affinity);                             
                }
            }
            
            if (node_proc != move_proc && is_compatible(node, move_proc)) {
                const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move_proc) + move_correction_node_weight;
                const cost_t prev_other_affinity = compute_same_step_affinity(prev_move_step_max_work, prev_new_weight, prev_node_proc_affinity);  
                const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(node_step, move_proc);
                const cost_t other_affinity = compute_same_step_affinity(new_max_weight, new_weight, new_node_proc_affinity);           
    
                affinity_table_node[move_proc][window_size] += (other_affinity - prev_other_affinity); 
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
                        affinity_table_node[proc][idx] += new_affinity - prev_affinity;  

                    } else {
                        const work_weight_t prev_new_weight = vertex_weight + active_schedule.get_step_processor_work(move_step, proc) + move_correction_node_weight;
                        const cost_t prev_affinity = prev_move_step_max_work < prev_new_weight ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_move_step_max_work) : 0.0;

                        const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                        affinity_table_node[proc][idx] += new_affinity - prev_affinity;
                    }
                }                        
            } else {
                // update only move_proc
                if (is_compatible(node, move_proc)) {
                    const work_weight_t new_weight = vertex_weight + active_schedule.get_step_processor_work(move_step, move_proc);
                    const work_weight_t prev_new_weight = new_weight + move_correction_node_weight;
                    const cost_t prev_affinity = prev_move_step_max_work < prev_new_weight ? static_cast<cost_t>(prev_new_weight) - static_cast<cost_t>(prev_move_step_max_work) : 0.0;

                    const cost_t new_affinity = new_max_weight < new_weight ? static_cast<cost_t>(new_weight) - static_cast<cost_t>(new_max_weight) : 0.0;
                    affinity_table_node[move_proc][idx] += new_affinity - prev_affinity;
                }
            }
        }
    }
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
bool kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::select_nodes_check_remove_superstep(unsigned & step_to_remove, ThreadSearchContext & thread_data) {
    if (thread_data.step_selection_epoch_counter >= parameters.node_max_step_selection_epochs || thread_data.num_steps() < 3) {
        return false;
    }
    
    for (step_to_remove = thread_data.step_selection_counter; step_to_remove <= thread_data.end_step; step_to_remove++) {
        assert(step_to_remove >= thread_data.start_step && step_to_remove <= thread_data.end_step);        
#ifdef KL_DEBUG
            std::cout << "Checking to remove step " << step_to_remove << "/" << thread_data.end_step <<  std::endl;
#endif
        if (check_remove_superstep(step_to_remove)) {
#ifdef KL_DEBUG
            std::cout << "Checking to scatter step " << step_to_remove << "/" << thread_data.end_step <<  std::endl;
#endif
            assert(step_to_remove >= thread_data.start_step && step_to_remove <= thread_data.end_step);
            if (scatter_nodes_superstep(step_to_remove, thread_data)) {
                thread_data.step_selection_counter = step_to_remove + 1;

                if (thread_data.step_selection_counter > thread_data.end_step) {
                    thread_data.step_selection_counter = thread_data.start_step;
                    thread_data.step_selection_epoch_counter++;
                }
                return true;
            }
        }
    }

    thread_data.step_selection_epoch_counter++;
    thread_data.step_selection_counter = thread_data.start_step;
    return false;
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
bool kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::check_remove_superstep(unsigned step) {
    if (active_schedule.num_steps() < 2) 
        return false;
    
    if (active_schedule.get_step_max_work(step) < instance->synchronisationCosts())
        return true;

    return false;
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::reset_inner_search_structures(ThreadSearchContext & thread_data) const {
    thread_data.unlock_edge_backtrack_counter = thread_data.unlock_edge_backtrack_counter_reset;
    thread_data.max_inner_iterations = parameters.max_inner_iterations_reset;
    thread_data.max_no_vioaltions_removed_backtrack = parameters.max_no_vioaltions_removed_backtrack_reset;
    thread_data.average_gain = 0.0;
    thread_data.affinity_table.reset_node_selection();
    thread_data.max_gain_heap.clear();
    thread_data.lock_manager.clear();
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
bool kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::is_local_search_blocked(ThreadSearchContext & thread_data) {
    for (const auto& pair : thread_data.active_schedule_data.new_violations) {
        if (thread_data.lock_manager.is_locked(pair.first)) {
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

    proc_range.initialize(*instance);
    comm_cost_f.initialize(active_schedule, proc_range);
    const cost_t initial_cost = comm_cost_f.compute_schedule_cost();
    active_schedule.set_cost(initial_cost);

    for (auto & t_data : thread_data_vec) {
        t_data.affinity_table.initialize(active_schedule, t_data.selection_strategy.selection_threshold);
        t_data.lock_manager.initialize(graph->num_vertices());    
        t_data.reward_penalty_strat.initialize(active_schedule, comm_cost_f.get_max_comm_weight_multiplied(), active_schedule.get_max_work_weight());
        t_data.selection_strategy.initialize(active_schedule, gen, t_data.start_step, t_data.end_step);
         
        t_data.local_affinity_table.resize(instance->numberOfProcessors());
        for (unsigned i = 0; i < instance->numberOfProcessors(); ++i) {
            t_data.local_affinity_table[i].resize(window_range);
        }
    } 
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_avg_gain(const cost_t gain, const unsigned num_iter, cost_t & average_gain) {
    average_gain = static_cast<double>((average_gain * num_iter + gain)) / (num_iter + 1.0);
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::insert_gain_heap(ThreadSearchContext & thread_data) {
    const size_t active_count = thread_data.affinity_table.size();

    for (size_t i = 0; i < active_count; ++i) {
        const VertexType node = thread_data.affinity_table.get_selected_nodes()[i]; 
        compute_node_affinities(node, thread_data.affinity_table.at(node), thread_data);
        const auto best_move = compute_best_move<true>(node, thread_data.affinity_table[node], thread_data);
        thread_data.max_gain_heap.push(node, best_move);
    }
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::insert_new_nodes_gain_heap(std::vector<VertexType>& new_nodes, node_selection_container_t &nodes, ThreadSearchContext & thread_data) {
    for (const auto &node : new_nodes) {
        nodes.insert(node);
        compute_node_affinities(node, thread_data.affinity_table.at(node), thread_data);
        const auto best_move = compute_best_move<true>(node, thread_data.affinity_table[node], thread_data);
        thread_data.max_gain_heap.push(node, best_move);        
    }
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::cleanup_datastructures() {
    thread_data_vec.clear();
    active_schedule.clear();             
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::print_heap(heap_datastructure & max_gain_heap) const {

    if (max_gain_heap.is_empty()) {
        std::cout << "heap is empty" << std::endl;
        return;
    }
    heap_datastructure temp_heap = max_gain_heap; // requires copy constructor

    std::cout << "heap current size: " << temp_heap.size() << std::endl;
    const auto& top_val = temp_heap.get_value(temp_heap.top());
    std::cout << "heap top node " << top_val.node << " gain " << top_val.gain << std::endl;

    unsigned count = 0;
    while (!temp_heap.is_empty() && count++ < 15) {
        const auto& val = temp_heap.get_value(temp_heap.top());
        std::cout << "node " << val.node << " gain " << val.gain << " to proc " << val.to_proc << " to step "
                    << val.to_step << std::endl;
        temp_heap.pop();
    }
}

template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_best_move(VertexType node, unsigned step, unsigned proc, node_selection_container_t &affinity_table, ThreadSearchContext & thread_data) {
    const unsigned node_proc = active_schedule.assigned_processor(node);
    const unsigned node_step = active_schedule.assigned_superstep(node);

    if((node_proc == proc) && (node_step == step)) 
        return;

    kl_move node_move = thread_data.max_gain_heap.get_value(node);
    cost_t max_gain = node_move.gain;
    
    unsigned max_proc = node_move.to_proc;
    unsigned max_step = node_move.to_step;

    if ((max_step == step) && (max_proc == proc)) {
        recompute_node_max_gain(node, affinity_table, thread_data);
    } else {
        if constexpr (active_schedule.use_memory_constraint) {
            if( not active_schedule.memory_constraint.can_move(node, proc, step)) return;                
        }
        const unsigned idx = rel_step_idx(node_step, step);
        const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][proc][idx];
        if (gain > max_gain) {
            max_gain = gain;
            max_proc = proc;
            max_step = step; 
        } 
    
        if ((max_gain != node_move.gain) || (max_proc != node_move.to_proc) || (max_step != node_move.to_step)) {
            node_move.gain = max_gain;
            node_move.to_proc = max_proc;
            node_move.to_step = max_step;
            thread_data.max_gain_heap.update(node, node_move);
        }        
    }
}
    
template<typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
void kl_improver<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t>::update_best_move(VertexType node, unsigned step, node_selection_container_t &affinity_table, ThreadSearchContext & thread_data) {
    
    const unsigned node_proc = active_schedule.assigned_processor(node);
    const unsigned node_step = active_schedule.assigned_superstep(node);

    kl_move node_move = thread_data.max_gain_heap.get_value(node);
    cost_t max_gain = node_move.gain;
    
    unsigned max_proc = node_move.to_proc;
    unsigned max_step = node_move.to_step;

    if (max_step == step) {
        recompute_node_max_gain(node, affinity_table, thread_data);   
    } else {        
        if (node_step != step) {
            const unsigned idx = rel_step_idx(node_step, step);
            for (const unsigned p : proc_range.compatible_processors_vertex(node)) {   
                if constexpr (active_schedule.use_memory_constraint) {
                    if( not active_schedule.memory_constraint.can_move(node, p, step)) continue;                
                }
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
                if constexpr (active_schedule.use_memory_constraint) {
                    if( not active_schedule.memory_constraint.can_move(node, proc, step)) continue;                
                }
                const cost_t gain = affinity_table[node][node_proc][window_size] - affinity_table[node][proc][window_size];
                if (gain > max_gain) {
                    max_gain = gain;
                    max_proc = proc;
                    max_step = step; 
                }
            }
        }        

        if ((max_gain != node_move.gain) || (max_proc != node_move.to_proc) || (max_step != node_move.to_step)) {
            node_move.gain = max_gain;
            node_move.to_proc = max_proc;
            node_move.to_step = max_step;
            thread_data.max_gain_heap.update(node, node_move);
        }        
    } 
}   

} // namespace osp