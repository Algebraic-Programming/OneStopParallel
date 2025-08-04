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

// #define KL_DEBUG

#pragma once

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/IBspSchedule.hpp"
#include "osp/bsp/model/SetSchedule.hpp"
#include "osp/bsp/model/VectorSchedule.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

template<typename cost_t, typename vertex_idx_t>
struct kl_move_struct {

    vertex_idx_t node;
    cost_t gain;

    unsigned from_proc;
    unsigned from_step;

    unsigned to_proc;
    unsigned to_step;

    kl_move_struct() : node(0), gain(0), from_proc(0), from_step(0), to_proc(0), to_step(0) {}
    kl_move_struct(vertex_idx_t node, cost_t gain, unsigned from_proc, unsigned from_step,
                   unsigned to_proc, unsigned to_step)
        : node(node), gain(gain), from_proc(from_proc), from_step(from_step),
          to_proc(to_proc), to_step(to_step) {}

    bool operator<(kl_move_struct<cost_t, vertex_idx_t> const &rhs) const {
        return (gain < rhs.gain) or (gain == rhs.gain and node > rhs.node);
    }

    kl_move_struct<cost_t, vertex_idx_t> reverse_move() const {
        return kl_move_struct(node, -gain, to_proc, to_step, from_proc, from_step);
    }
};

template<typename work_weight_t>
struct pre_move_work_data {

    work_weight_t from_step_max_work;
    work_weight_t from_step_second_max_work;
    unsigned from_step_max_work_processor_count;

    work_weight_t to_step_max_work;
    work_weight_t to_step_second_max_work;
    unsigned to_step_max_work_processor_count;

    pre_move_work_data() {}
    pre_move_work_data(work_weight_t from_step_max_work_, work_weight_t from_step_second_max_work_, unsigned from_step_max_work_processor_count_,
                 work_weight_t to_step_max_work_, work_weight_t to_step_second_max_work_,
                 unsigned to_step_max_work_processor_count_)
        : from_step_max_work(from_step_max_work_), from_step_second_max_work(from_step_second_max_work_),
          from_step_max_work_processor_count(from_step_max_work_processor_count_),
          to_step_max_work(to_step_max_work_), to_step_second_max_work(to_step_second_max_work_),
          to_step_max_work_processor_count(to_step_max_work_processor_count_) {}          

};

template<typename Graph_t>
struct kl_active_schedule_work_datastructures {

    using work_weight_t = v_workw_t<Graph_t>;

    const BspInstance<Graph_t> *instance;   
    const SetSchedule<Graph_t> *set_schedule;
   
    struct weight_proc {
        work_weight_t work;
        unsigned proc;

        weight_proc() : work(0), proc(0) {}
        weight_proc(work_weight_t work, unsigned proc) : work(work), proc(proc) {}
    
        bool operator<(weight_proc const &rhs) const {
            return (work > rhs.work) or (work == rhs.work and proc < rhs.proc);
        }
    };

    std::vector<std::vector<weight_proc>> step_processor_work_;
    std::vector<std::vector<unsigned>> step_processor_position;
    std::vector<unsigned> step_max_work_processor_count;

    inline work_weight_t step_max_work(unsigned step) const { return step_processor_work_[step][0].work; }
    inline work_weight_t step_second_max_work(unsigned step) const { return step_processor_work_[step][step_max_work_processor_count[step]].work; }
    inline work_weight_t step_proc_work(unsigned step, unsigned proc) const { return step_processor_work_[step][step_processor_position[step][proc]].work; }
    inline work_weight_t & step_proc_work(unsigned step, unsigned proc) { return step_processor_work_[step][step_processor_position[step][proc]].work; }

    template<typename cost_t, typename vertex_idx_t>
    inline pre_move_work_data<work_weight_t> get_pre_move_work_data(kl_move_struct<cost_t, vertex_idx_t> move) { 
        return pre_move_work_data<work_weight_t>(step_max_work(move.from_step), step_second_max_work(move.from_step), step_max_work_processor_count[move.from_step],
                                                        step_max_work(move.to_step), step_second_max_work(move.to_step), step_max_work_processor_count[move.to_step]); 
    }

    inline void initialize(const SetSchedule<Graph_t> &sched, const BspInstance<Graph_t> &inst, unsigned num_steps) {
        instance = &inst;
        set_schedule = &sched;
        step_processor_work_ = std::vector<std::vector<weight_proc>>(num_steps, std::vector<weight_proc>(instance->numberOfProcessors()));
        step_processor_position = std::vector<std::vector<unsigned>>(num_steps, std::vector<unsigned>(instance->numberOfProcessors(), 0));
        step_max_work_processor_count = std::vector<unsigned>(num_steps, 0);
    }

    inline void clear() {
        step_processor_work_.clear();
        step_processor_position.clear();
        step_max_work_processor_count.clear();
    }

    inline void arrange_superstep_data(const unsigned step) {
        std::sort(step_processor_work_[step].begin(), step_processor_work_[step].end());
        unsigned pos = 0;
        const work_weight_t max_work_to = step_processor_work_[step][0].work;

        for (const auto & wp : step_processor_work_[step]) {
            step_processor_position[step][wp.proc] = pos++;

            if (wp.work == max_work_to && pos < instance->numberOfProcessors())
                step_max_work_processor_count[step] = pos; 
        }
    }

    template<typename cost_t, typename vertex_idx_t>
    void apply_move(kl_move_struct<cost_t, vertex_idx_t> move, work_weight_t work_weight) {      

        if (work_weight == 0) 
            return;
        
        if (move.to_step != move.from_step) {
            step_proc_work(move.to_step, move.to_proc) += work_weight;
            step_proc_work(move.from_step, move.from_proc) -= work_weight;

            arrange_superstep_data(move.to_step);
            arrange_superstep_data(move.from_step);

            // const work_weight_t prev_max_work_to = step_max_work(move.to_step);
            // const work_weight_t new_weight_to = step_proc_work(move.to_step, move.to_proc) += work_weight;

            // if (prev_max_work_to < new_weight_to) {
            //     step_max_work_processor_count[move.to_step] = 1;
            // } else if (prev_max_work_to == new_weight_to) {
            //     step_max_work_processor_count[move.to_step]++;
            // }

            // unsigned to_proc_pos = step_processor_position[move.to_step][move.to_proc];
            
            // while (to_proc_pos > 0 && step_processor_work_[move.to_step][to_proc_pos - 1].work < new_weight_to) {
            //     std::swap(step_processor_work_[move.to_step][to_proc_pos], step_processor_work_[move.to_step][to_proc_pos - 1]);
            //     std::swap(step_processor_position[move.to_step][step_processor_work_[move.to_step][to_proc_pos].proc], step_processor_position[move.to_step][step_processor_work_[move.to_step][to_proc_pos - 1].proc]);
            //     to_proc_pos--;
            // }

            // const work_weight_t prev_max_work_from = step_max_work(move.from_step);
            // const work_weight_t prev_weight_from = step_proc_work(move.from_step, move.from_proc);
            // const work_weight_t new_weight_from = step_proc_work(move.from_step, move.from_proc) -= work_weight;

            // unsigned from_proc_pos = step_processor_position[move.from_step][move.from_proc];

            // while (from_proc_pos < instance->numberOfProcessors() - 1 && step_processor_work_[move.from_step][from_proc_pos + 1].work > new_weight_from) {
            //     std::swap(step_processor_work_[move.from_step][from_proc_pos], step_processor_work_[move.from_step][from_proc_pos + 1]);
            //     std::swap(step_processor_position[move.from_step][step_processor_work_[move.from_step][from_proc_pos].proc], step_processor_position[move.from_step][step_processor_work_[move.from_step][from_proc_pos + 1].proc]);
            //     from_proc_pos++;
            // }
                
            // if (prev_max_work_from == prev_weight_from) {
            //     step_max_work_processor_count[move.from_step]--;        
            //     if (step_max_work_processor_count[move.from_step] == 0) {  
            //         step_max_work_processor_count[move.from_step] = from_proc_pos; 
            //     }
            // }    

        } else {            
            step_proc_work(move.to_step, move.to_proc) += work_weight;
            step_proc_work(move.from_step, move.from_proc) -= work_weight;
            arrange_superstep_data(move.to_step);
        }
    }

    void merge_into_previous_superstep(const unsigned step) {
        if (step == 0)
            return;

        const unsigned prev_step = step - 1;
        for (unsigned i = 0; i < instance->numberOfProcessors(); i++) {
            const unsigned proc = step_processor_work_[prev_step][i].proc;
            const unsigned proc_pos_in_next_step = step_processor_position[step][proc];
            step_processor_work_[prev_step][i].work += step_processor_work_[step][proc_pos_in_next_step].work;
        }
        arrange_superstep_data(prev_step);
    }

    void override_previous_superstep(unsigned step) {
        if(step == 0)
            return;

        const unsigned prev_step = step - 1;
        for (unsigned i = 0; i < instance->numberOfProcessors(); i++) {
            step_processor_work_[prev_step][i] = step_processor_work_[step][i]; 
            step_processor_position[prev_step][i] = step_processor_position[step][i];            
        }
        step_max_work_processor_count[prev_step] = step_max_work_processor_count[step];
    }

    void swap_with_next_superstep(const unsigned step) {
        const unsigned next_step = step + 1;

        std::swap(step_processor_work_[step], step_processor_work_[next_step]);
        std::swap(step_processor_position[step], step_processor_position[next_step]);
        std::swap(step_max_work_processor_count[step], step_max_work_processor_count[next_step]);
    }
    

    void override_next_superstep(unsigned step) {

        const unsigned next_step = step + 1;
        for (unsigned i = 0; i < instance->numberOfProcessors(); i++) {
            step_processor_work_[next_step][i] = step_processor_work_[step][i]; 
            step_processor_position[next_step][i] = step_processor_position[step][i];            
        }
        step_max_work_processor_count[next_step] = step_max_work_processor_count[step];
    }

    void reset_superstep(unsigned step) {
        for (unsigned i = 0; i < instance->numberOfProcessors(); i++) {
            step_processor_work_[step][i] = {0,i}; 
            step_processor_position[step][i] = i;            
        }
        step_max_work_processor_count[step] = instance->numberOfProcessors() - 1;
    }

    void compute_work_datastructures(unsigned start_step, unsigned end_step) {
        for (unsigned step = start_step; step <= end_step; step++) {
            step_max_work_processor_count[step] = 0;
            work_weight_t max_work = 0;

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
                step_processor_work_[step][proc].work = 0;
                step_processor_work_[step][proc].proc = proc;

                for (const auto &node : set_schedule->step_processor_vertices[step][proc]) {
                    step_processor_work_[step][proc].work += instance->getComputationalDag().vertex_work_weight(node);
                }

                if (step_processor_work_[step][proc].work > max_work) {
                    max_work = step_processor_work_[step][proc].work;
                    step_max_work_processor_count[step] = 1;
                } else if (step_processor_work_[step][proc].work == max_work && step_max_work_processor_count[step] < (instance->numberOfProcessors() - 1)) {
                    step_max_work_processor_count[step]++;
                } 
            }

            std::sort(step_processor_work_[step].begin(), step_processor_work_[step].end());
            unsigned pos = 0;
            for (const auto & wp : step_processor_work_[step]) {
                step_processor_position[step][wp.proc] = pos++;
            }
        }
    }
};

template<typename Graph_t, typename cost_t, typename MemoryConstraint_t>
class kl_active_schedule {

  private:
    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;

    using kl_move = kl_move_struct<cost_t, VertexType>;

    const BspInstance<Graph_t> *instance;

    VectorSchedule<Graph_t> vector_schedule;
    SetSchedule<Graph_t> set_schedule;

    cost_t cost = 0;
    bool feasible = true;

    std::unordered_set<EdgeType> current_violations;

    std::vector<kl_move> applied_moves;

    cost_t best_cost = 0;
    unsigned best_schedule_idx = 0;

  public:
    virtual ~kl_active_schedule() = default;

    inline const BspInstance<Graph_t> & getInstance() const { return *instance; }
    inline const VectorSchedule<Graph_t> & getVectorSchedule() const { return vector_schedule; }
    inline const SetSchedule<Graph_t> & getSetSchedule() const { return set_schedule; }
    inline const std::vector<kl_move> & getAppliedMoves() const { return applied_moves; }
    inline cost_t get_cost() { return cost; }
    inline cost_t get_best_cost() { return best_cost; }
    inline bool is_feasible() { return feasible; }
    inline const std::unordered_set<EdgeType> &get_current_violations() const { return current_violations; }
    inline unsigned num_steps() const { return vector_schedule.numberOfSupersteps(); }
    inline unsigned assigned_processor(VertexType node) const { return vector_schedule.assignedProcessor(node); }
    inline unsigned assigned_superstep(VertexType node) const { return vector_schedule.assignedSuperstep(node); }
    inline v_workw_t<Graph_t> get_step_max_work(unsigned step) const {return work_datastructures.step_max_work(step); }
    inline v_workw_t<Graph_t> get_step_second_max_work(unsigned step) const {return work_datastructures.step_second_max_work(step); }
    inline std::vector<unsigned> & get_step_max_work_processor_count() {return work_datastructures.step_max_work_processor_count; }    
    inline v_workw_t<Graph_t> get_step_processor_work(unsigned step, unsigned proc) const {return work_datastructures.step_proc_work(step, proc); }
    inline pre_move_work_data<v_workw_t<Graph_t>> get_pre_move_work_data(kl_move move) { return work_datastructures.get_pre_move_work_data(move); }
    
    constexpr static bool use_memory_constraint = is_local_search_memory_constraint_v<MemoryConstraint_t>;

    MemoryConstraint_t memory_constraint;

    kl_active_schedule_work_datastructures<Graph_t> work_datastructures;

    std::unordered_map<VertexType, EdgeType> new_violations;
    std::unordered_set<EdgeType> resolved_violations;

    inline v_workw_t<Graph_t> get_step_total_work(unsigned step) const {        
        v_workw_t<Graph_t> total_work = 0;        
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            total_work += get_step_processor_work(step, proc);
        }       
        return total_work;
    }

    void apply_move(kl_move move) {
        vector_schedule.setAssignedProcessor(move.node, move.to_proc);
        vector_schedule.setAssignedSuperstep(move.node, move.to_step);

        set_schedule.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
        set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
       
        update_violations(move.node);
        applied_moves.push_back(move);

        work_datastructures.apply_move(move, instance->getComputationalDag().vertex_work_weight(move.node));
        if constexpr (use_memory_constraint) {
            memory_constraint.apply_move(move.node, move.from_proc, move.from_step, move.to_proc, move.to_step);
        }       
    }

    template<typename comm_datastructures_t>
    void revert_to_best_schedule(unsigned start_move, unsigned insert_step, comm_datastructures_t & comm_datastructures) {
        const unsigned bound = std::max(start_move, best_schedule_idx);

        while (applied_moves.size() > bound) {
            const auto move = applied_moves.back().reverse_move();
            applied_moves.pop_back();

            vector_schedule.setAssignedProcessor(move.node, move.to_proc);
            vector_schedule.setAssignedSuperstep(move.node, move.to_step);  

            set_schedule.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
            set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
            work_datastructures.apply_move(move, instance->getComputationalDag().vertex_work_weight(move.node));
            comm_datastructures.update_datastructure_after_move(move);
            if constexpr (use_memory_constraint) {
                memory_constraint.apply_move(move.node, move.from_proc, move.from_step, move.to_proc, move.to_step);
            }
        }

        if (start_move > best_schedule_idx) {
            insert_empty_step(insert_step);
        }

        while (applied_moves.size() > best_schedule_idx) {
            const auto move = applied_moves.back().reverse_move();
            applied_moves.pop_back();

            vector_schedule.setAssignedProcessor(move.node, move.to_proc);
            vector_schedule.setAssignedSuperstep(move.node, move.to_step);  

            set_schedule.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
            set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
            work_datastructures.apply_move(move, instance->getComputationalDag().vertex_work_weight(move.node));
            comm_datastructures.update_datastructure_after_move(move);
            if constexpr (use_memory_constraint) {
                memory_constraint.apply_move(move.node, move.from_proc, move.from_step, move.to_proc, move.to_step);
            }
        }

#ifdef KL_DEBUG
        if (not feasible)
            std::cout << "Reverted to best schedule with cost: " << best_cost << " and " << vector_schedule.number_of_supersteps << " supersteps" << std::endl;
#endif

        applied_moves.clear();
        best_schedule_idx = 0;
        current_violations.clear();
        feasible = true;
        cost = best_cost;
    }

    inline void update_cost(cost_t change_in_cost) {
        cost += change_in_cost;        

        if (cost <= best_cost && feasible) {
            best_cost = cost;
            best_schedule_idx = static_cast<unsigned>(applied_moves.size());
        }    
    }

    inline void initialize_cost(cost_t cost_) {
        cost = cost_;
        best_cost = cost_;
    }  

    void compute_violations();
    void compute_work_memory_datastructures(unsigned start_step, unsigned end_step);
    void write_schedule (BspSchedule<Graph_t> &schedule);
    inline void initialize(const IBspSchedule<Graph_t> &schedule);
    inline void clear();
    void remove_empty_step(unsigned step);
    void insert_empty_step(unsigned step);

  private:

    void update_violations(VertexType node) {
        new_violations.clear();
        resolved_violations.clear();

        for (const auto &edge : out_edges(node, instance->getComputationalDag())) {
            const auto &child = target(edge, instance->getComputationalDag());

            if (current_violations.find(edge) == current_violations.end()) {
                if (vector_schedule.assignedSuperstep(node) >= vector_schedule.assignedSuperstep(child)) {
                    if (vector_schedule.assignedProcessor(node) != vector_schedule.assignedProcessor(child) ||
                        vector_schedule.assignedSuperstep(node) > vector_schedule.assignedSuperstep(child)) {

                        current_violations.insert(edge);
                        new_violations[child] = edge;
                    }
                }
            } else {
                if (vector_schedule.assignedSuperstep(node) <= vector_schedule.assignedSuperstep(child)) {
                    if (vector_schedule.assignedProcessor(node) == vector_schedule.assignedProcessor(child) ||
                        vector_schedule.assignedSuperstep(node) < vector_schedule.assignedSuperstep(child)) {

                        current_violations.erase(edge);
                        resolved_violations.insert(edge);
                    }
                }
            }
        }

        for (const auto &edge : in_edges(node, instance->getComputationalDag())) {
            const auto &parent = source(edge, instance->getComputationalDag());

            if (current_violations.find(edge) == current_violations.end()) {
                if (vector_schedule.assignedSuperstep(node) <= vector_schedule.assignedSuperstep(parent)) {
                    if (vector_schedule.assignedProcessor(node) != vector_schedule.assignedProcessor(parent) ||
                        vector_schedule.assignedSuperstep(node) < vector_schedule.assignedSuperstep(parent)) {

                        current_violations.insert(edge);
                        new_violations[parent] = edge;
                    }
                }
            } else {
                if (vector_schedule.assignedSuperstep(node) >= vector_schedule.assignedSuperstep(parent)) {
                    if (vector_schedule.assignedProcessor(node) == vector_schedule.assignedProcessor(parent) ||
                        vector_schedule.assignedSuperstep(node) > vector_schedule.assignedSuperstep(parent)) {

                        current_violations.erase(edge);
                        resolved_violations.insert(edge);
                    }
                }
            }
        }

#ifdef KL_DEBUG

        if (new_violations.size() > 0) {
            std::cout << "New violations: " << std::endl;
            for (const auto &edge : new_violations) {
                std::cout << "Edge: " << source(edge.second, instance->getComputationalDag()) << " -> "
                          << target(edge.second, instance->getComputationalDag()) << std::endl;
            }
        }

        if (resolved_violations.size() > 0) {
            std::cout << "Resolved violations: " << std::endl;
            for (const auto &edge : resolved_violations) {
                std::cout << "Edge: " << source(edge, instance->getComputationalDag()) << " -> "
                          << target(edge, instance->getComputationalDag()) << std::endl;
            }
        }

#endif

        if (current_violations.size() > 0) {
            feasible = false;
        } else {
            feasible = true;
        }
    }

};

template<typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::clear() {

    work_datastructures.clear();

    vector_schedule.clear();
    set_schedule.clear();

    if constexpr (use_memory_constraint) {
        memory_constraint.clear();
    }
}

template<typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::compute_violations() {

    current_violations.clear();

    for (const auto &edge : edges(instance->getComputationalDag())) {

        const auto &source_v = source(edge, instance->getComputationalDag());
        const auto &target_v = target(edge, instance->getComputationalDag());

        const unsigned source_proc = assigned_processor(source_v);
        const unsigned target_proc = assigned_processor(target_v);
        const unsigned source_step = assigned_superstep(source_v);
        const unsigned target_step = assigned_superstep(target_v);
    
        if (source_step > target_step || (source_step == target_step && source_proc != target_proc)) {
            current_violations.insert(edge);
            feasible = false;
        } 
    }  
    
    if (current_violations.empty()) {
        feasible = true;
    }  
}

template<typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::initialize(const IBspSchedule<Graph_t> &schedule) {

    instance = &schedule.getInstance();

    vector_schedule = VectorSchedule(schedule);
    set_schedule = SetSchedule(schedule);
    work_datastructures.initialize(set_schedule, *instance, num_steps());

    cost = 0;
    best_cost = 0;
    feasible = true;
    best_schedule_idx = 0;

    if constexpr (use_memory_constraint) {
        memory_constraint.initialize(set_schedule, vector_schedule);
    }

    compute_work_memory_datastructures(0, num_steps() - 1);
}

template<typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::compute_work_memory_datastructures(unsigned start_step, unsigned end_step) {

    if constexpr (use_memory_constraint) {
        memory_constraint.recompute_memory_datastructure(start_step, end_step);
    }
    work_datastructures.compute_work_datastructures(start_step, end_step);
}

template<typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::write_schedule (BspSchedule<Graph_t> &schedule) {

    for (const auto v : instance->vertices()) {
        schedule.setAssignedProcessor(v, vector_schedule.assignedProcessor(v));
        schedule.setAssignedSuperstep(v, vector_schedule.assignedSuperstep(v));
    }
    schedule.updateNumberOfSupersteps();
}

template<typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::remove_empty_step(unsigned step) {    
    for (unsigned i = step; i < num_steps() - 1; i++) {
        for(unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (const auto node : set_schedule.step_processor_vertices[i + 1][proc]){
                vector_schedule.setAssignedSuperstep(node, i);
            }
        }
        std::swap(set_schedule.step_processor_vertices[i], set_schedule.step_processor_vertices[i + 1]);
        work_datastructures.swap_with_next_superstep(i);
    }
    vector_schedule.number_of_supersteps--;
    // for (unsigned i = step + 1; i < num_steps(); i++) {
    //     work_datastructures.override_previous_superstep(i);
    //     // if constexpr (use_memory_constraint) {
    //     //     memory_constraint.override_superstep(i, proc, i + 1, proc);
    //     // }       
    // }
    // if constexpr (use_memory_constraint) {
    //     memory_constraint.reset_superstep(num_steps());
    // }
}


template<typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::insert_empty_step(unsigned step) {
    unsigned i = vector_schedule.number_of_supersteps++;  
 
    for (; i > step; i--) {
        for(unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (const auto node : set_schedule.step_processor_vertices[i-1][proc]){
                vector_schedule.setAssignedSuperstep(node, i);
            }
        }
        std::swap(set_schedule.step_processor_vertices[i], set_schedule.step_processor_vertices[i - 1]);
        work_datastructures.swap_with_next_superstep(i-1);
    } 
     
}
} // namespace osp

