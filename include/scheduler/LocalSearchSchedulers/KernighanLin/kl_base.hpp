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

#include "auxiliary/auxiliary.hpp"
#include "scheduler/ImprovementScheduler.hpp"
#include "kl_current_schedule.hpp"




struct kl_base_parameter {

  double max_div_best_sol_base_percent = 1.05;
  double max_div_best_sol_rate_percent = 0.002;

  unsigned max_num_unlocks = 2;
  unsigned max_num_failed_branches = 5;

  unsigned max_inner_iterations = 150;
  unsigned max_outer_iterations = 300;

  unsigned max_no_improvement_iterations = 75;

  unsigned selection_threshold;
  bool select_all_nodes = false;

  double initial_penalty = 0.0;

  double gain_threshold = -1.0;
  double change_in_cost_threshold = 0.0;

  bool quick_pass = false;

  unsigned max_step_selection_epochs = 4;
  unsigned reset_epoch_counter_threshold = 10;

  unsigned violations_threshold = 0;

};



class kl_base : public ImprovementScheduler, public Ikl_cost_function {

  protected:

    kl_base_parameter parameters;

    std::mt19937 gen;

    unsigned num_nodes;
    unsigned num_procs;
    
    double penalty = 0.0;
    double reward = 0.0;

    virtual void update_reward_penalty() = 0;
    virtual void set_initial_reward_penalty() = 0;

    boost::heap::fibonacci_heap<kl_move> max_gain_heap;
    using heap_handle = typename boost::heap::fibonacci_heap<kl_move>::handle_type;

    std::unordered_map<VertexType, heap_handle> node_heap_handles;

    std::vector<std::vector<std::vector<double>>> node_gains;
    std::vector<std::vector<std::vector<double>>> node_change_in_costs;
    
    kl_current_schedule *current_schedule;

    BspSchedule *best_schedule;
    double best_schedule_costs;

    std::unordered_set<VertexType> locked_nodes;
    std::unordered_set<VertexType> super_locked_nodes;
    std::vector<unsigned> unlock;

    bool unlock_node(VertexType node);
    bool check_node_unlocked(VertexType node);
    void reset_locked_nodes();

    bool check_violation_locked();

    void reset_gain_heap();
    virtual void initialize_datastructures();   
    
    std::unordered_set<VertexType> nodes_to_update;
    void compute_nodes_to_update(kl_move move);

    void initialize_gain_heap(const std::unordered_set<VertexType> &nodes);
    void initialize_gain_heap_unlocked_nodes(const std::unordered_set<VertexType> &nodes);

    void compute_node_gain(unsigned node);

    double compute_max_gain_insert_or_update_heap(VertexType node);

    void compute_work_gain(unsigned node, unsigned current_step, unsigned current_proc, unsigned new_proc);
        
    virtual void compute_comm_gain(unsigned node, unsigned current_step, unsigned current_proc, unsigned new_proc) = 0;
    //virtual double compute_current_costs() = 0;

    void update_node_gains(const std::unordered_set<VertexType> &nodes);

    kl_move find_best_move();
    kl_move compute_best_move(VertexType node);
    kl_move best_move_change_superstep(VertexType node);
    
    void save_best_schedule(const IBspSchedule &schedule);
    void reverse_move_best_schedule(kl_move move);

    std::unordered_set<VertexType> node_selection;

    void select_nodes(); 
    void select_nodes_threshold(unsigned threshold);
    void select_nodes_permutation_threshold(unsigned threshold);

    void select_nodes_violations();
    
    void select_nodes_conseque_max_work(bool do_not_select_super_locked_nodes = false); 
    void select_nodes_check_remove_superstep();
    unsigned step_selection_counter = 0;
    unsigned step_selection_epoch_counter = 0;

    bool reset_superstep = true;

    virtual bool check_remove_superstep(unsigned step);
    bool scatter_nodes_remove_superstep(unsigned step);

    void select_nodes_check_reset_superstep();
    virtual bool check_reset_superstep(unsigned step);
    bool scatter_nodes_reset_superstep(unsigned step);

    void select_unlock_neighbors(VertexType node);

    void set_parameters();      
    virtual void cleanup_datastructures();
    void reset_run_datastructures();

    bool run_local_search_without_violations();
    bool run_local_search_simple();
    bool run_local_search_remove_supersteps();
 
    bool run_local_search_unlock_delay();

    // virtual void checkMergeSupersteps();
    // virtual void checkInsertSuperstep();

    // virtual void insertSuperstep(unsigned step);
  
    void print_heap();

    bool compute_with_time_limit = false;
    

  public:
    
    kl_base(kl_current_schedule& current_schedule_) : ImprovementScheduler(), current_schedule(&current_schedule_) 
    {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    virtual ~kl_base() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule &schedule) override;

    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule &schedule) override {
        compute_with_time_limit = true;
        return improveSchedule(schedule);
    }

    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override {
        current_schedule->use_memory_constraint = use_memory_constraint_;
    }

    virtual void set_compute_with_time_limit(bool compute_with_time_limit_) {
        compute_with_time_limit = compute_with_time_limit_;
    }

    virtual std::string getScheduleName() const = 0;

    virtual void set_quick_pass(bool quick_pass_) { parameters.quick_pass = quick_pass_; }
};


