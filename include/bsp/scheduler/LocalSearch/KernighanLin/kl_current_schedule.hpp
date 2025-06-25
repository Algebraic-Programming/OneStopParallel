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

#define KL_DEBUG

#include "bsp/model/BspSchedule.hpp"
#include "bsp/model/IBspSchedule.hpp"
#include "bsp/model/SetSchedule.hpp"
#include "bsp/model/VectorSchedule.hpp"
#include "bsp/scheduler/ImprovementScheduler.hpp"
#include "bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"
#include "graph_algorithms/directed_graph_util.hpp"

namespace osp {

template<typename Graph_t>
struct kl_move {

    vertex_idx_t<Graph_t> node;

    double gain;
    double change_in_cost;

    unsigned from_proc;
    unsigned from_step;

    unsigned to_proc;
    unsigned to_step;

    kl_move() : node(0), gain(0), change_in_cost(0), from_proc(0), from_step(0), to_proc(0), to_step(0) {}
    kl_move(vertex_idx_t<Graph_t> node, double gain, double change_cost, unsigned from_proc, unsigned from_step,
            unsigned to_proc, unsigned to_step)
        : node(node), gain(gain), change_in_cost(change_cost), from_proc(from_proc), from_step(from_step),
          to_proc(to_proc), to_step(to_step) {}

    bool operator<(kl_move const &rhs) const {
        return (gain < rhs.gain) or (gain == rhs.gain and change_in_cost < rhs.change_in_cost) or
               (gain == rhs.gain and change_in_cost == rhs.change_in_cost and node > rhs.node);
    }

    kl_move reverse_move() const {
        return kl_move(node, -gain, -change_in_cost, to_proc, to_step, from_proc, from_step);
    }
};

class Ikl_cost_function {
  public:
    virtual double compute_current_costs() = 0;
};

template<typename Graph_t, typename MemoryConstraint_t>
class kl_current_schedule {

  private:
    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;

  public:

    kl_current_schedule(Ikl_cost_function *cost_f_) : cost_f(cost_f_) {

#ifdef KL_DEBUG        
        if constexpr (use_memory_constraint) {
            std::cout << "KLCurrentSchedule constructor with memory constraint" << std::endl;
        } else {
            std::cout << "KLCurrentSchedule constructor without memory constraint" << std::endl;
        }
#endif

    }

    virtual ~kl_current_schedule() = default;

    Ikl_cost_function *cost_f;

    const BspInstance<Graph_t> *instance;

    VectorSchedule<Graph_t> vector_schedule;
    SetSchedule<Graph_t> set_schedule;

    constexpr static bool use_memory_constraint = is_local_search_memory_constraint_v<MemoryConstraint_t>;

    MemoryConstraint_t memory_constraint;

    std::vector<std::vector<v_workw_t<Graph_t>>> step_processor_work;

    std::vector<v_workw_t<Graph_t>> step_max_work;
    std::vector<v_workw_t<Graph_t>> step_second_max_work;

    double current_cost = 0;

    bool current_feasible = true;
    std::unordered_set<EdgeType> current_violations; // edges

    std::unordered_map<VertexType, EdgeType> new_violations;
    std::unordered_set<EdgeType> resolved_violations;

    void remove_superstep(unsigned step) {

        if (step > 0) {
            vector_schedule.mergeSupersteps(step - 1, step);
            set_schedule.mergeSupersteps(step - 1, step);

            compute_work_memory_datastructures(step - 1, step);

        } else {
            vector_schedule.mergeSupersteps(0, 1);
            set_schedule.mergeSupersteps(0, 1);

            compute_work_memory_datastructures(0, 0);
        }

        for (unsigned i = step + 1; i < num_steps(); i++) {

            step_max_work[i] = step_max_work[i + 1];
            step_second_max_work[i] = step_second_max_work[i + 1];

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                step_processor_work[i][proc] = step_processor_work[i + 1][proc];

                if constexpr (use_memory_constraint) {
                    memory_constraint.override_superstep(i, proc, i + 1, proc);
                }
            }
        }

        step_second_max_work[num_steps()] = 0;
        step_max_work[num_steps()] = 0;

        if constexpr (use_memory_constraint) {
            memory_constraint.reset_superstep(num_steps());
        }

        recompute_current_violations();
        cost_f->compute_current_costs();
    }

    void reset_superstep(unsigned step) {

        if (step > 0) {
            compute_work_memory_datastructures(step - 1, step - 1);
            if (step < num_steps() - 1) {
                compute_work_memory_datastructures(step + 1, step + 1);
            }
        } else {
            compute_work_memory_datastructures(1, 1);
        }

        step_second_max_work[step] = 0;
        step_max_work[step] = 0;

        if constexpr (use_memory_constraint) {
            memory_constraint.reset_superstep(step);
        }

        recompute_current_violations();
        cost_f->compute_current_costs();
    }

    void recompute_neighboring_supersteps(unsigned step) {
        if (step > 0) {
            compute_work_memory_datastructures(step - 1, step);
            if (step < num_steps() - 1) {
                compute_work_memory_datastructures(step + 1, step + 1);
            }
        } else {
            compute_work_memory_datastructures(0, 0);
            if (num_steps() > 1) {
                compute_work_memory_datastructures(1, 1);
            }
        }
    }

    inline unsigned num_steps() const { return vector_schedule.numberOfSupersteps(); }

    virtual void set_current_schedule(const IBspSchedule<Graph_t> &schedule) {

        if (num_steps() == schedule.numberOfSupersteps()) {

#ifdef KL_DEBUG
            std::cout << "KLCurrentSchedule set current schedule, same nr supersteps" << std::endl;
#endif

            for (unsigned step = 0; step < num_steps(); step++) {
                for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
                    set_schedule.step_processor_vertices[step][proc].clear();
                }
            }

            for (const auto &node : instance->getComputationalDag().vertices()) {

                vector_schedule.setAssignedProcessor(node, schedule.assignedProcessor(node));
                vector_schedule.setAssignedSuperstep(node, schedule.assignedSuperstep(node));

                set_schedule.step_processor_vertices[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)]
                    .insert(node);
            }

        } else {

#ifdef KL_DEBUG
            std::cout << "KLCurrentSchedule set current schedule, different nr supersteps" << std::endl;
#endif

            vector_schedule = VectorSchedule(schedule);
            set_schedule = SetSchedule(schedule);

            initialize_superstep_datastructures();
        }

        compute_work_memory_datastructures(0, num_steps() - 1);
        recompute_current_violations();

        cost_f->compute_current_costs();

#ifdef KL_DEBUG
        std::cout << "KLCurrentSchedule set current schedule done, costs: " << current_cost
                  << " number of supersteps: " << num_steps() << std::endl;
#endif
    }

    virtual void initialize_superstep_datastructures() {

#ifdef KL_DEBUG
        std::cout << "KLCurrentSchedule initialize datastructures" << std::endl;
#endif

        const unsigned num_procs = instance->numberOfProcessors();

        if constexpr (use_memory_constraint) {

            memory_constraint.initialize(set_schedule, vector_schedule);
        }

        step_processor_work =
            std::vector<std::vector<v_workw_t<Graph_t>>>(num_steps(), std::vector<v_workw_t<Graph_t>>(num_procs, 0));
        step_max_work = std::vector<v_workw_t<Graph_t>>(num_steps(), 0);
        step_second_max_work = std::vector<v_workw_t<Graph_t>>(num_steps(), 0);
    }

    virtual void cleanup_superstep_datastructures() {

        step_processor_work.clear();
        step_max_work.clear();
        step_second_max_work.clear();

        if constexpr (use_memory_constraint) {
            memory_constraint.clear();
        }
    }

    virtual void compute_work_memory_datastructures(unsigned start_step, unsigned end_step) {

        if constexpr (use_memory_constraint) {
            memory_constraint.recompute_memory_datastructure(start_step, end_step);
        }

        for (unsigned step = start_step; step <= end_step; step++) {

            step_max_work[step] = 0;
            step_second_max_work[step] = 0;

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                step_processor_work[step][proc] = 0;

                for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                    step_processor_work[step][proc] += instance->getComputationalDag().vertex_work_weight(node);
                }

                if (step_processor_work[step][proc] > step_max_work[step]) {

                    step_second_max_work[step] = step_max_work[step];
                    step_max_work[step] = step_processor_work[step][proc];

                } else if (step_processor_work[step][proc] > step_second_max_work[step]) {

                    step_second_max_work[step] = step_processor_work[step][proc];
                }
            }
        }
    }

    virtual void recompute_current_violations() {

        current_violations.clear();

#ifdef KL_DEBUG
        std::cout << "Recompute current violations:" << std::endl;
#endif

        for (const auto &edge : instance->getComputationalDag().edges()) {

            const auto &source_v = source(edge, instance->getComputationalDag());
            const auto &target_v = target(edge, instance->getComputationalDag());

            if (vector_schedule.assignedSuperstep(source_v) >= vector_schedule.assignedSuperstep(target_v)) {

                if (vector_schedule.assignedProcessor(source_v) != vector_schedule.assignedProcessor(target_v) ||
                    vector_schedule.assignedSuperstep(source_v) > vector_schedule.assignedSuperstep(target_v)) {

                    current_violations.insert(edge);

#ifdef KL_DEBUG
                    std::cout << "Edge: " << source_v << " -> " << target_v << std::endl;
#endif
                }
            }
        }

        if (current_violations.size() > 0) {
            current_feasible = false;
        } else {
#ifdef KL_DEBUG
            std::cout << "Current schedule is feasible" << std::endl;
#endif

            current_feasible = true;
        }
    };

    virtual void apply_move(kl_move<Graph_t> move) {

        vector_schedule.setAssignedProcessor(move.node, move.to_proc);
        vector_schedule.setAssignedSuperstep(move.node, move.to_step);

        set_schedule.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
        set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);

        current_cost += move.change_in_cost;

        step_processor_work[move.to_step][move.to_proc] +=
            instance->getComputationalDag().vertex_work_weight(move.node);
        step_processor_work[move.from_step][move.from_proc] -=
            instance->getComputationalDag().vertex_work_weight(move.node);

        update_max_work_datastructures(move);
        update_violations(move.node);

        if constexpr (use_memory_constraint) {

            memory_constraint.apply_move(move.node, move.from_proc, move.from_step, move.to_proc, move.to_step);
        }  
    }

    virtual void initialize_current_schedule(const IBspSchedule<Graph_t> &schedule) {

#ifdef KL_DEBUG
        std::cout << "KLCurrentSchedule initialize current schedule" << std::endl;
#endif

        vector_schedule = VectorSchedule<Graph_t>(schedule);
        set_schedule = SetSchedule<Graph_t>(schedule);

        initialize_superstep_datastructures();

        compute_work_memory_datastructures(0, num_steps() - 1);
        recompute_current_violations();

        cost_f->compute_current_costs();
    }

  private:
    void update_violations(VertexType node) {

        new_violations.clear();
        resolved_violations.clear();

        for (const auto &edge : instance->getComputationalDag().out_edges(node)) {

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

        for (const auto &edge : instance->getComputationalDag().in_edges(node)) {

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
            current_feasible = false;
        } else {
            current_feasible = true;
        }
    }

    void update_max_work_datastructures(kl_move<Graph_t> move) {

        if (move.from_step == move.to_step) {

            recompute_superstep_max_work(move.from_step);

        } else {

            recompute_superstep_max_work(move.from_step);
            recompute_superstep_max_work(move.to_step);
        }
    }

    void recompute_superstep_max_work(unsigned step) {

        step_max_work[step] = 0;
        step_second_max_work[step] = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            if (step_processor_work[step][proc] > step_max_work[step]) {

                step_second_max_work[step] = step_max_work[step];
                step_max_work[step] = step_processor_work[step][proc];

            } else if (step_processor_work[step][proc] > step_second_max_work[step]) {

                step_second_max_work[step] = step_processor_work[step][proc];
            }
        }
    }
};

template<typename Graph_t, typename MemoryConstraint_t>
class kl_current_schedule_max_comm : public kl_current_schedule<Graph_t, MemoryConstraint_t> {

  public:
    std::vector<std::vector<v_commw_t<Graph_t>>> step_processor_send;
    std::vector<v_commw_t<Graph_t>> step_max_send;
    std::vector<v_commw_t<Graph_t>> step_max_receive;

    std::vector<std::vector<v_commw_t<Graph_t>>> step_processor_receive;
    std::vector<v_commw_t<Graph_t>> step_second_max_send;
    std::vector<v_commw_t<Graph_t>> step_second_max_receive;
};

} // namespace osp