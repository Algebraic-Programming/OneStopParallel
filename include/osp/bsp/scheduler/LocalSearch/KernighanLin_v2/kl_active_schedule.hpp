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

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/IBspSchedule.hpp"
#include "osp/bsp/model/util/SetSchedule.hpp"
#include "osp/bsp/model/util/VectorSchedule.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

template <typename cost_t, typename vertex_idx_t>
struct kl_move_struct {
    vertex_idx_t node;
    cost_t gain;

    unsigned from_proc;
    unsigned from_step;

    unsigned to_proc;
    unsigned to_step;

    kl_move_struct() : node(0), gain(0), from_proc(0), from_step(0), to_proc(0), to_step(0) {}

    kl_move_struct(vertex_idx_t _node, cost_t _gain, unsigned _from_proc, unsigned _from_step, unsigned _to_proc, unsigned _to_step)
        : node(_node), gain(_gain), from_proc(_from_proc), from_step(_from_step), to_proc(_to_proc), to_step(_to_step) {}

    bool operator<(kl_move_struct<cost_t, vertex_idx_t> const &rhs) const {
        return (gain < rhs.gain) or (gain == rhs.gain and node > rhs.node);
    }

    bool operator>(kl_move_struct<cost_t, vertex_idx_t> const &rhs) const {
        return (gain > rhs.gain) or (gain >= rhs.gain and node < rhs.node);
    }

    kl_move_struct<cost_t, vertex_idx_t> reverse_move() const {
        return kl_move_struct(node, -gain, to_proc, to_step, from_proc, from_step);
    }
};

template <typename work_weight_t>
struct pre_move_work_data {
    work_weight_t from_step_max_work;
    work_weight_t from_step_second_max_work;
    unsigned from_step_max_work_processor_count;

    work_weight_t to_step_max_work;
    work_weight_t to_step_second_max_work;
    unsigned to_step_max_work_processor_count;

    pre_move_work_data() {}

    pre_move_work_data(work_weight_t from_step_max_work_,
                       work_weight_t from_step_second_max_work_,
                       unsigned from_step_max_work_processor_count_,
                       work_weight_t to_step_max_work_,
                       work_weight_t to_step_second_max_work_,
                       unsigned to_step_max_work_processor_count_)
        : from_step_max_work(from_step_max_work_),
          from_step_second_max_work(from_step_second_max_work_),
          from_step_max_work_processor_count(from_step_max_work_processor_count_),
          to_step_max_work(to_step_max_work_),
          to_step_second_max_work(to_step_second_max_work_),
          to_step_max_work_processor_count(to_step_max_work_processor_count_) {}
};

template <typename Graph_t>
struct kl_active_schedule_work_datastructures {
    using work_weight_t = v_workw_t<Graph_t>;

    const BspInstance<Graph_t> *instance;
    const SetSchedule<Graph_t> *set_schedule;

    struct weight_proc {
        work_weight_t work;
        unsigned proc;

        weight_proc() : work(0), proc(0) {}

        weight_proc(work_weight_t _work, unsigned _proc) : work(_work), proc(_proc) {}

        bool operator<(weight_proc const &rhs) const { return (work > rhs.work) or (work == rhs.work and proc < rhs.proc); }
    };

    std::vector<std::vector<weight_proc>> step_processor_work_;
    std::vector<std::vector<unsigned>> step_processor_position;
    std::vector<unsigned> step_max_work_processor_count;
    work_weight_t max_work_weight;
    work_weight_t total_work_weight;

    inline work_weight_t step_max_work(unsigned step) const { return step_processor_work_[step][0].work; }

    inline work_weight_t step_second_max_work(unsigned step) const {
        return step_processor_work_[step][step_max_work_processor_count[step]].work;
    }

    inline work_weight_t step_proc_work(unsigned step, unsigned proc) const {
        return step_processor_work_[step][step_processor_position[step][proc]].work;
    }

    inline work_weight_t &step_proc_work(unsigned step, unsigned proc) {
        return step_processor_work_[step][step_processor_position[step][proc]].work;
    }

    template <typename cost_t, typename vertex_idx_t>
    inline pre_move_work_data<work_weight_t> get_pre_move_work_data(kl_move_struct<cost_t, vertex_idx_t> move) {
        return pre_move_work_data<work_weight_t>(step_max_work(move.from_step),
                                                 step_second_max_work(move.from_step),
                                                 step_max_work_processor_count[move.from_step],
                                                 step_max_work(move.to_step),
                                                 step_second_max_work(move.to_step),
                                                 step_max_work_processor_count[move.to_step]);
    }

    inline void initialize(const SetSchedule<Graph_t> &sched, const BspInstance<Graph_t> &inst, unsigned num_steps) {
        instance = &inst;
        set_schedule = &sched;
        max_work_weight = 0;
        total_work_weight = 0;
        step_processor_work_
            = std::vector<std::vector<weight_proc>>(num_steps, std::vector<weight_proc>(instance->numberOfProcessors()));
        step_processor_position
            = std::vector<std::vector<unsigned>>(num_steps, std::vector<unsigned>(instance->numberOfProcessors(), 0));
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

        for (const auto &wp : step_processor_work_[step]) {
            step_processor_position[step][wp.proc] = pos++;

            if (wp.work == max_work_to && pos < instance->numberOfProcessors()) {
                step_max_work_processor_count[step] = pos;
            }
        }
    }

    template <typename cost_t, typename vertex_idx_t>
    void apply_move(kl_move_struct<cost_t, vertex_idx_t> move, work_weight_t work_weight) {
        if (work_weight == 0) {
            return;
        }

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
            //     std::swap(step_processor_work_[move.to_step][to_proc_pos], step_processor_work_[move.to_step][to_proc_pos -
            //     1]); std::swap(step_processor_position[move.to_step][step_processor_work_[move.to_step][to_proc_pos].proc],
            //     step_processor_position[move.to_step][step_processor_work_[move.to_step][to_proc_pos - 1].proc]);
            //     to_proc_pos--;
            // }

            // const work_weight_t prev_max_work_from = step_max_work(move.from_step);
            // const work_weight_t prev_weight_from = step_proc_work(move.from_step, move.from_proc);
            // const work_weight_t new_weight_from = step_proc_work(move.from_step, move.from_proc) -= work_weight;

            // unsigned from_proc_pos = step_processor_position[move.from_step][move.from_proc];

            // while (from_proc_pos < instance->numberOfProcessors() - 1 && step_processor_work_[move.from_step][from_proc_pos +
            // 1].work > new_weight_from) {
            //     std::swap(step_processor_work_[move.from_step][from_proc_pos],
            //     step_processor_work_[move.from_step][from_proc_pos + 1]);
            //     std::swap(step_processor_position[move.from_step][step_processor_work_[move.from_step][from_proc_pos].proc],
            //     step_processor_position[move.from_step][step_processor_work_[move.from_step][from_proc_pos + 1].proc]);
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

    void swap_steps(const unsigned step1, const unsigned step2) {
        std::swap(step_processor_work_[step1], step_processor_work_[step2]);
        std::swap(step_processor_position[step1], step_processor_position[step2]);
        std::swap(step_max_work_processor_count[step1], step_max_work_processor_count[step2]);
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
            step_processor_work_[step][i] = {0, i};
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
                    const work_weight_t vertex_work_weight = instance->getComputationalDag().vertex_work_weight(node);
                    total_work_weight += vertex_work_weight;
                    max_work_weight = std::max(vertex_work_weight, max_work_weight);
                    step_processor_work_[step][proc].work += vertex_work_weight;
                }

                if (step_processor_work_[step][proc].work > max_work) {
                    max_work = step_processor_work_[step][proc].work;
                    step_max_work_processor_count[step] = 1;
                } else if (step_processor_work_[step][proc].work == max_work
                           && step_max_work_processor_count[step] < (instance->numberOfProcessors() - 1)) {
                    step_max_work_processor_count[step]++;
                }
            }

            std::sort(step_processor_work_[step].begin(), step_processor_work_[step].end());
            unsigned pos = 0;
            for (const auto &wp : step_processor_work_[step]) {
                step_processor_position[step][wp.proc] = pos++;
            }
        }
    }
};

template <typename Graph_t, typename cost_t>
struct thread_local_active_schedule_data {
    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;

    using kl_move = kl_move_struct<cost_t, VertexType>;

    std::unordered_set<EdgeType> current_violations;
    std::vector<kl_move> applied_moves;

    cost_t cost = 0;
    cost_t initial_cost = 0;
    bool feasible = true;

    cost_t best_cost = 0;
    unsigned best_schedule_idx = 0;

    std::unordered_map<VertexType, EdgeType> new_violations;
    std::unordered_set<EdgeType> resolved_violations;

    inline void initialize_cost(cost_t cost_) {
        initial_cost = cost_;
        cost = cost_;
        best_cost = cost_;
        feasible = true;
    }

    inline void update_cost(cost_t change_in_cost) {
        cost += change_in_cost;

        if (cost <= best_cost && feasible) {
            best_cost = cost;
            best_schedule_idx = static_cast<unsigned>(applied_moves.size());
        }
    }
};

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
class kl_active_schedule {
  private:
    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;
    using kl_move = kl_move_struct<cost_t, VertexType>;
    using thread_data_t = thread_local_active_schedule_data<Graph_t, cost_t>;

    const BspInstance<Graph_t> *instance;

    VectorSchedule<Graph_t> vector_schedule;
    SetSchedule<Graph_t> set_schedule;

    cost_t cost = 0;
    bool feasible = true;

  public:
    virtual ~kl_active_schedule() = default;

    inline const BspInstance<Graph_t> &getInstance() const { return *instance; }

    inline const VectorSchedule<Graph_t> &getVectorSchedule() const { return vector_schedule; }

    inline VectorSchedule<Graph_t> &getVectorSchedule() { return vector_schedule; }

    inline const SetSchedule<Graph_t> &getSetSchedule() const { return set_schedule; }

    inline cost_t get_cost() { return cost; }

    inline bool is_feasible() { return feasible; }

    inline unsigned num_steps() const { return vector_schedule.numberOfSupersteps(); }

    inline unsigned assigned_processor(VertexType node) const { return vector_schedule.assignedProcessor(node); }

    inline unsigned assigned_superstep(VertexType node) const { return vector_schedule.assignedSuperstep(node); }

    inline v_workw_t<Graph_t> get_step_max_work(unsigned step) const { return work_datastructures.step_max_work(step); }

    inline v_workw_t<Graph_t> get_step_second_max_work(unsigned step) const {
        return work_datastructures.step_second_max_work(step);
    }

    inline std::vector<unsigned> &get_step_max_work_processor_count() {
        return work_datastructures.step_max_work_processor_count;
    }

    inline v_workw_t<Graph_t> get_step_processor_work(unsigned step, unsigned proc) const {
        return work_datastructures.step_proc_work(step, proc);
    }

    inline pre_move_work_data<v_workw_t<Graph_t>> get_pre_move_work_data(kl_move move) {
        return work_datastructures.get_pre_move_work_data(move);
    }

    inline v_workw_t<Graph_t> get_max_work_weight() { return work_datastructures.max_work_weight; }

    inline v_workw_t<Graph_t> get_total_work_weight() { return work_datastructures.total_work_weight; }

    inline void set_cost(cost_t cost_) { cost = cost_; }

    constexpr static bool use_memory_constraint = is_local_search_memory_constraint_v<MemoryConstraint_t>;

    MemoryConstraint_t memory_constraint;

    kl_active_schedule_work_datastructures<Graph_t> work_datastructures;

    inline v_workw_t<Graph_t> get_step_total_work(unsigned step) const {
        v_workw_t<Graph_t> total_work = 0;
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            total_work += get_step_processor_work(step, proc);
        }
        return total_work;
    }

    void apply_move(kl_move move, thread_data_t &thread_data) {
        vector_schedule.setAssignedProcessor(move.node, move.to_proc);
        vector_schedule.setAssignedSuperstep(move.node, move.to_step);

        set_schedule.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
        set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);

        update_violations(move.node, thread_data);
        thread_data.applied_moves.push_back(move);

        work_datastructures.apply_move(move, instance->getComputationalDag().vertex_work_weight(move.node));
        if constexpr (use_memory_constraint) {
            memory_constraint.apply_move(move.node, move.from_proc, move.from_step, move.to_proc, move.to_step);
        }
    }

    template <typename comm_datastructures_t>
    void revert_to_best_schedule(unsigned start_move,
                                 unsigned insert_step,
                                 comm_datastructures_t &comm_datastructures,
                                 thread_data_t &thread_data,
                                 unsigned start_step,
                                 unsigned &end_step) {
        const unsigned bound = std::max(start_move, thread_data.best_schedule_idx);
        revert_moves(bound, comm_datastructures, thread_data, start_step, end_step);

        if (start_move > thread_data.best_schedule_idx) {
            swap_empty_step_bwd(++end_step, insert_step);
        }

        revert_moves(thread_data.best_schedule_idx, comm_datastructures, thread_data, start_step, end_step);

#ifdef KL_DEBUG
        if (not thread_data.feasible) {
            std::cout << "Reverted to best schedule with cost: " << thread_data.best_cost << " and "
                      << vector_schedule.number_of_supersteps << " supersteps" << std::endl;
        }
#endif

        thread_data.applied_moves.clear();
        thread_data.best_schedule_idx = 0;
        thread_data.current_violations.clear();
        thread_data.feasible = true;
        thread_data.cost = thread_data.best_cost;
    }

    template <typename comm_datastructures_t>
    void revert_schedule_to_bound(const size_t bound,
                                  const cost_t new_cost,
                                  const bool is_feasible,
                                  comm_datastructures_t &comm_datastructures,
                                  thread_data_t &thread_data,
                                  unsigned start_step,
                                  unsigned end_step) {
        revert_moves(bound, comm_datastructures, thread_data, start_step, end_step);

        thread_data.current_violations.clear();
        thread_data.feasible = is_feasible;
        thread_data.cost = new_cost;
    }

    void compute_violations(thread_data_t &thread_data);
    void compute_work_memory_datastructures(unsigned start_step, unsigned end_step);
    void write_schedule(BspSchedule<Graph_t> &schedule);
    inline void initialize(const IBspSchedule<Graph_t> &schedule);
    inline void clear();
    void remove_empty_step(unsigned step);
    void insert_empty_step(unsigned step);
    void swap_empty_step_fwd(const unsigned step, const unsigned to_step);
    void swap_empty_step_bwd(const unsigned to_step, const unsigned empty_step);
    void swap_steps(const unsigned step1, const unsigned step2);

  private:
    template <typename comm_datastructures_t>
    void revert_moves(const size_t bound,
                      comm_datastructures_t &comm_datastructures,
                      thread_data_t &thread_data,
                      unsigned start_step,
                      unsigned end_step) {
        while (thread_data.applied_moves.size() > bound) {
            const auto move = thread_data.applied_moves.back().reverse_move();
            thread_data.applied_moves.pop_back();

            vector_schedule.setAssignedProcessor(move.node, move.to_proc);
            vector_schedule.setAssignedSuperstep(move.node, move.to_step);

            set_schedule.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
            set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
            work_datastructures.apply_move(move, instance->getComputationalDag().vertex_work_weight(move.node));
            comm_datastructures.update_datastructure_after_move(move, start_step, end_step);
            if constexpr (use_memory_constraint) {
                memory_constraint.apply_move(move.node, move.from_proc, move.from_step, move.to_proc, move.to_step);
            }
        }
    }

    void update_violations(VertexType node, thread_data_t &thread_data) {
        thread_data.new_violations.clear();
        thread_data.resolved_violations.clear();

        const unsigned node_step = vector_schedule.assignedSuperstep(node);
        const unsigned node_proc = vector_schedule.assignedProcessor(node);

        for (const auto &edge : out_edges(node, instance->getComputationalDag())) {
            const auto &child = target(edge, instance->getComputationalDag());

            if (thread_data.current_violations.find(edge) == thread_data.current_violations.end()) {
                if ((node_step > vector_schedule.assignedSuperstep(child))
                    || (node_step == vector_schedule.assignedSuperstep(child)
                        && node_proc != vector_schedule.assignedProcessor(child))) {
                    thread_data.current_violations.insert(edge);
                    thread_data.new_violations[child] = edge;
                }
            } else {
                if ((node_step < vector_schedule.assignedSuperstep(child))
                    || (node_step == vector_schedule.assignedSuperstep(child)
                        && node_proc == vector_schedule.assignedProcessor(child))) {
                    thread_data.current_violations.erase(edge);
                    thread_data.resolved_violations.insert(edge);
                }
            }
        }

        for (const auto &edge : in_edges(node, instance->getComputationalDag())) {
            const auto &parent = source(edge, instance->getComputationalDag());

            if (thread_data.current_violations.find(edge) == thread_data.current_violations.end()) {
                if ((node_step < vector_schedule.assignedSuperstep(parent))
                    || (node_step == vector_schedule.assignedSuperstep(parent)
                        && node_proc != vector_schedule.assignedProcessor(parent))) {
                    thread_data.current_violations.insert(edge);
                    thread_data.new_violations[parent] = edge;
                }
            } else {
                if ((node_step > vector_schedule.assignedSuperstep(parent))
                    || (node_step == vector_schedule.assignedSuperstep(parent)
                        && node_proc == vector_schedule.assignedProcessor(parent))) {
                    thread_data.current_violations.erase(edge);
                    thread_data.resolved_violations.insert(edge);
                }
            }
        }

#ifdef KL_DEBUG

        if (thread_data.new_violations.size() > 0) {
            std::cout << "New violations: " << std::endl;
            for (const auto &edge : thread_data.new_violations) {
                std::cout << "Edge: " << source(edge.second, instance->getComputationalDag()) << " -> "
                          << target(edge.second, instance->getComputationalDag()) << std::endl;
            }
        }

        if (thread_data.resolved_violations.size() > 0) {
            std::cout << "Resolved violations: " << std::endl;
            for (const auto &edge : thread_data.resolved_violations) {
                std::cout << "Edge: " << source(edge, instance->getComputationalDag()) << " -> "
                          << target(edge, instance->getComputationalDag()) << std::endl;
            }
        }

#endif

        if (thread_data.current_violations.size() > 0) {
            thread_data.feasible = false;
        } else {
            thread_data.feasible = true;
        }
    }
};

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::clear() {
    work_datastructures.clear();
    vector_schedule.clear();
    set_schedule.clear();
    if constexpr (use_memory_constraint) {
        memory_constraint.clear();
    }
}

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::compute_violations(thread_data_t &thread_data) {
    thread_data.current_violations.clear();
    thread_data.feasible = true;

    for (const auto &edge : edges(instance->getComputationalDag())) {
        const auto &source_v = source(edge, instance->getComputationalDag());
        const auto &target_v = target(edge, instance->getComputationalDag());

        const unsigned source_proc = assigned_processor(source_v);
        const unsigned target_proc = assigned_processor(target_v);
        const unsigned source_step = assigned_superstep(source_v);
        const unsigned target_step = assigned_superstep(target_v);

        if (source_step > target_step || (source_step == target_step && source_proc != target_proc)) {
            thread_data.current_violations.insert(edge);
            thread_data.feasible = false;
        }
    }
}

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::initialize(const IBspSchedule<Graph_t> &schedule) {
    instance = &schedule.getInstance();
    vector_schedule = VectorSchedule(schedule);
    set_schedule = SetSchedule(schedule);
    work_datastructures.initialize(set_schedule, *instance, num_steps());

    cost = 0;
    feasible = true;

    if constexpr (use_memory_constraint) {
        memory_constraint.initialize(set_schedule, vector_schedule);
    }

    compute_work_memory_datastructures(0, num_steps() - 1);
}

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::compute_work_memory_datastructures(unsigned start_step,
                                                                                                 unsigned end_step) {
    if constexpr (use_memory_constraint) {
        memory_constraint.compute_memory_datastructure(start_step, end_step);
    }
    work_datastructures.compute_work_datastructures(start_step, end_step);
}

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::write_schedule(BspSchedule<Graph_t> &schedule) {
    for (const auto v : instance->vertices()) {
        schedule.setAssignedProcessor(v, vector_schedule.assignedProcessor(v));
        schedule.setAssignedSuperstep(v, vector_schedule.assignedSuperstep(v));
    }
    schedule.updateNumberOfSupersteps();
}

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::remove_empty_step(unsigned step) {
    for (unsigned i = step; i < num_steps() - 1; i++) {
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (const auto node : set_schedule.step_processor_vertices[i + 1][proc]) {
                vector_schedule.setAssignedSuperstep(node, i);
            }
        }
        std::swap(set_schedule.step_processor_vertices[i], set_schedule.step_processor_vertices[i + 1]);
        work_datastructures.swap_steps(i, i + 1);
        if constexpr (use_memory_constraint) {
            memory_constraint.swap_steps(i, i + 1);
        }
    }
    vector_schedule.number_of_supersteps--;
}

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::swap_empty_step_fwd(const unsigned step, const unsigned to_step) {
    for (unsigned i = step; i < to_step; i++) {
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (const auto node : set_schedule.step_processor_vertices[i + 1][proc]) {
                vector_schedule.setAssignedSuperstep(node, i);
            }
        }
        std::swap(set_schedule.step_processor_vertices[i], set_schedule.step_processor_vertices[i + 1]);
        work_datastructures.swap_steps(i, i + 1);
        if constexpr (use_memory_constraint) {
            memory_constraint.swap_steps(i, i + 1);
        }
    }
}

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::insert_empty_step(unsigned step) {
    unsigned i = vector_schedule.number_of_supersteps++;

    for (; i > step; i--) {
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (const auto node : set_schedule.step_processor_vertices[i - 1][proc]) {
                vector_schedule.setAssignedSuperstep(node, i);
            }
        }
        std::swap(set_schedule.step_processor_vertices[i], set_schedule.step_processor_vertices[i - 1]);
        work_datastructures.swap_steps(i - 1, i);
        if constexpr (use_memory_constraint) {
            memory_constraint.swap_steps(i - 1, i);
        }
    }
}

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::swap_empty_step_bwd(const unsigned to_step,
                                                                                  const unsigned empty_step) {
    unsigned i = to_step;

    for (; i > empty_step; i--) {
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (const auto node : set_schedule.step_processor_vertices[i - 1][proc]) {
                vector_schedule.setAssignedSuperstep(node, i);
            }
        }
        std::swap(set_schedule.step_processor_vertices[i], set_schedule.step_processor_vertices[i - 1]);
        work_datastructures.swap_steps(i - 1, i);
        if constexpr (use_memory_constraint) {
            memory_constraint.swap_steps(i - 1, i);
        }
    }
}

template <typename Graph_t, typename cost_t, typename MemoryConstraint_t>
void kl_active_schedule<Graph_t, cost_t, MemoryConstraint_t>::swap_steps(const unsigned step1, const unsigned step2) {
    if (step1 == step2) {
        return;
    }

    for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
        for (const auto node : set_schedule.step_processor_vertices[step1][proc]) {
            vector_schedule.setAssignedSuperstep(node, step2);
        }
        for (const auto node : set_schedule.step_processor_vertices[step2][proc]) {
            vector_schedule.setAssignedSuperstep(node, step1);
        }
    }
    std::swap(set_schedule.step_processor_vertices[step1], set_schedule.step_processor_vertices[step2]);
    work_datastructures.swap_steps(step1, step2);
    if constexpr (use_memory_constraint) {
        memory_constraint.swap_steps(step1, step2);
    }
}

}    // namespace osp
