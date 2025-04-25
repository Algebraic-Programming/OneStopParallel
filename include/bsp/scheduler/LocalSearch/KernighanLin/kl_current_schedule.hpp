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

#include "bsp/model/SetSchedule.hpp"
#include "bsp/model/VectorSchedule.hpp"
#include "graph_algorithms/directed_graph_util.hpp"
#include "scheduler/ImprovementScheduler.hpp"

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

template<typename Graph_t>
class kl_current_schedule {

  private:
    using VertexType = typename vertex_idx_t<Graph_t>;
    using EdgeType = typename edge_idx_t<Graph_t>;
    using kl_move = typename kl_move<Graph_t>;

  public:
    kl_current_schedule(Ikl_cost_function *cost_f_) : cost_f(cost_f_) {}

    virtual ~kl_current_schedule() = default;

    Ikl_cost_function *cost_f;

    bool use_memory_constraint = false;

    const BspInstance<Graph_t> *instance;

    VectorSchedule<Graph_t> vector_schedule;
    SetSchedule<Graph_t> set_schedule;

    std::vector<std::vector<v_memw_t<Graph_t>>> step_processor_memory;
    std::vector<std::vector<std::unordered_set<VertexType>>> step_processor_pred;

    std::vector<v_memw_t<Graph_t>> current_proc_persistent_memory;
    std::vector<v_commw_t<Graph_t>> current_proc_transient_memory;

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

                if (use_memory_constraint) {
                    if (instance->getArchitecture().getMemoryConstraintType() == LOCAL) {

                        step_processor_memory[i][proc] = step_processor_memory[i + 1][proc];

                    } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_IN_OUT) {

                        step_processor_memory[i][proc] = step_processor_memory[i + 1][proc];
                    } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {
                        step_processor_memory[i][proc] = step_processor_memory[i + 1][proc];
                        step_processor_pred[i][proc] = step_processor_pred[i + 1][proc];
                    } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES_2) {
                        step_processor_memory[i][proc] = step_processor_memory[i + 1][proc];
                        step_processor_pred[i][proc] = step_processor_pred[i + 1][proc];
                    }
                }
            }
        }

        step_second_max_work[num_steps()] = 0;
        step_max_work[num_steps()] = 0;

        if (use_memory_constraint) {
            if (instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
                for (unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); proc++) {
                    step_processor_memory[num_steps()][proc] = 0;
                }
            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_IN_OUT) {
                for (unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); proc++) {
                    step_processor_memory[num_steps()][proc] = 0;
                }
            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {
                for (unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); proc++) {
                    step_processor_memory[num_steps()][proc] = 0;
                    step_processor_pred[num_steps()][proc].clear();
                }
            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES_2) {
                for (unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); proc++) {
                    step_processor_memory[num_steps()][proc] = 0;
                    step_processor_pred[num_steps()][proc].clear();
                }
            }
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

        if (use_memory_constraint) {
            if (instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
                for (unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); proc++) {
                    step_processor_memory[step][proc] = 0;
                }
            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_IN_OUT) {
                for (unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); proc++) {
                    step_processor_memory[step][proc] = 0;
                }
            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {
                for (unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); proc++) {
                    step_processor_memory[step][proc] = 0;
                    step_processor_pred[step][proc].clear();
                }
            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES_2) {
                for (unsigned proc = 0; proc < instance->getArchitecture().numberOfProcessors(); proc++) {
                    step_processor_memory[step][proc] = 0;
                    step_processor_pred[step][proc].clear();
                }
            }
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

        if (use_memory_constraint) {
            if (instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
                step_processor_memory = std::vector<std::vector<v_memw_t<Graph_t>>>(
                    num_steps(), std::vector<v_memw_t<Graph_t>>(num_procs, 0));
            } else if (instance->getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

                throw std::runtime_error("Memory constraint PERSISTENT_AND_TRANSIENT not implemented");

                current_proc_persistent_memory = std::vector<v_memw_t<Graph_t>>(num_procs, 0);
                current_proc_transient_memory = std::vector<v_commw_t<Graph_t>>(num_procs, 0);

            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_IN_OUT) {
                step_processor_memory = std::vector<std::vector<v_memw_t<Graph_t>>>(
                    num_steps(), std::vector<v_memw_t<Graph_t>>(num_procs, 0));

                throw std::runtime_error("Memory constraint LOCAL_IN_OUT not implemented");

            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {
                step_processor_memory = std::vector<std::vector<v_memw_t<Graph_t>>>(
                    num_steps(), std::vector<v_memw_t<Graph_t>>(num_procs, 0));
                step_processor_pred = std::vector<std::vector<std::unordered_set<VertexType>>>(
                    num_steps(), std::vector<std::unordered_set<VertexType>>(num_procs));
            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES_2) {
                step_processor_memory = std::vector<std::vector<v_memw_t<Graph_t>>>(
                    num_steps(), std::vector<v_memw_t<Graph_t>>(num_procs, 0));
                step_processor_pred = std::vector<std::vector<std::unordered_set<VertexType>>>(
                    num_steps(), std::vector<std::unordered_set<VertexType>>(num_procs));
            }
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

        step_processor_memory.clear();
        step_processor_pred.clear();
        current_proc_persistent_memory.clear();
        current_proc_transient_memory.clear();
    }

    virtual void compute_work_memory_datastructures(unsigned start_step, unsigned end_step) {

        if (use_memory_constraint) {

            if (instance->getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
                for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
                    current_proc_persistent_memory[proc] = 0;
                    current_proc_transient_memory[proc] = 0;
                }
            }

            for (unsigned step = start_step; step <= end_step; step++) {

                step_max_work[step] = 0;
                step_second_max_work[step] = 0;

                for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                    step_processor_work[step][proc] = 0;

                    if (instance->getArchitecture().getMemoryConstraintType() == LOCAL) {

                        step_processor_memory[step][proc] = 0;

                    } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {

                        step_processor_memory[step][proc] = 0;
                        step_processor_pred[step][proc].clear();

                    } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES_2) {

                        step_processor_memory[step][proc] = 0;
                        step_processor_pred[step][proc].clear();
                    }

                    for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                        step_processor_work[step][proc] += instance->getComputationalDag().vertex_work_weight(node);

                        if (instance->getArchitecture().getMemoryConstraintType() == LOCAL) {

                            step_processor_memory[step][proc] +=
                                instance->getComputationalDag().vertex_mem_weight(node);

                        } else if (instance->getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
                            current_proc_persistent_memory[proc] +=
                                instance->getComputationalDag().vertex_mem_weight(node);
                            current_proc_transient_memory[proc] =
                                std::max(current_proc_transient_memory[proc],
                                         instance->getComputationalDag().vertex_comm_weight(node));

                            if (current_proc_transient_memory[proc] + current_proc_persistent_memory[proc] >
                                instance->memoryBound(proc)) {
                                throw std::runtime_error(
                                    "Memory constraint PERSISTENT_AND_TRANSIENT not properly implemented");
                            }
                        } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_IN_OUT) {

                            step_processor_memory[step][proc] +=
                                instance->getComputationalDag().vertex_mem_weight(node) +
                                instance->getComputationalDag().vertex_comm_weight(node);

                            for (const auto &pred : instance->getComputationalDag().parents(node)) {

                                if (vector_schedule.assignedProcessor(pred) == proc &&
                                    vector_schedule.assignedSuperstep(pred) == step) {

                                    step_processor_memory[step][proc] -=
                                        instance->getComputationalDag().vertex_comm_weight(pred);
                                }
                            }
                        } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {

                            step_processor_memory[step][proc] +=
                                instance->getComputationalDag().vertex_comm_weight(node);

                            for (const auto &pred : instance->getComputationalDag().parents(node)) {

                                if (vector_schedule.assignedSuperstep(pred) < step) {

                                    auto pair = step_processor_pred[step][proc].insert(pred);
                                    if (pair.second) {
                                        step_processor_memory[step][proc] +=
                                            instance->getComputationalDag().vertex_comm_weight(pred);
                                    }
                                }
                            }
                        } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES_2) {

                            if (is_source(node, instance->getComputationalDag())) {
                                step_processor_memory[step][proc] +=
                                    instance->getComputationalDag().vertex_mem_weight(node);
                            }

                            for (const auto &pred : instance->getComputationalDag().parents(node)) {

                                if (vector_schedule.assignedSuperstep(pred) < step) {

                                    auto pair = step_processor_pred[step][proc].insert(pred);
                                    if (pair.second) {
                                        step_processor_memory[step][proc] +=
                                            instance->getComputationalDag().vertex_comm_weight(pred);
                                    }
                                }
                            }
                        }
                    }

                    if (step_processor_work[step][proc] > step_max_work[step]) {

                        step_second_max_work[step] = step_max_work[step];
                        step_max_work[step] = step_processor_work[step][proc];

                    } else if (step_processor_work[step][proc] > step_second_max_work[step]) {

                        step_second_max_work[step] = step_processor_work[step][proc];
                    }
                }
            }

        } else {

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
    }

    virtual void recompute_current_violations() {

        current_violations.clear();

#ifdef KL_DEBUG
        std::cout << "Recompute current violations:" << std::endl;
#endif

        for (const auto &edge : instance->getComputationalDag().edges()) {

            const auto &source = source(edge, instance->getComputationalDag());
            const auto &target = target(edge, instance->getComputationalDag());

            if (vector_schedule.assignedSuperstep(source) >= vector_schedule.assignedSuperstep(target)) {

                if (vector_schedule.assignedProcessor(source) != vector_schedule.assignedProcessor(target) ||
                    vector_schedule.assignedSuperstep(source) > vector_schedule.assignedSuperstep(target)) {

                    current_violations.insert(edge);

#ifdef KL_DEBUG
                    std::cout << "Edge: " << source << " -> " << target << std::endl;
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

    virtual void apply_move(kl_move move) {

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

        if (use_memory_constraint) {

            if (instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
                step_processor_memory[move.to_step][move.to_proc] +=
                    instance->getComputationalDag().vertex_mem_weight(move.node);
                step_processor_memory[move.from_step][move.from_proc] -=
                    instance->getComputationalDag().vertex_mem_weight(move.node);

            } else if (instance->getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

                if (move.to_proc != move.from_proc) {

                    current_proc_persistent_memory[move.to_proc] +=
                        instance->getComputationalDag().vertex_mem_weight(move.node);
                    current_proc_persistent_memory[move.from_proc] -=
                        instance->getComputationalDag().vertex_mem_weight(move.node);

                    current_proc_transient_memory[move.to_proc] =
                        std::max(current_proc_transient_memory[move.to_proc],
                                 instance->getComputationalDag().vertex_comm_weight(move.node));

                    if (current_proc_transient_memory[move.from_proc] ==
                        instance->getComputationalDag().vertex_comm_weight(move.node)) {

                        current_proc_transient_memory[move.from_proc] = 0;

                        for (unsigned step = 0; step < num_steps(); step++) {
                            for (const auto &node : set_schedule.step_processor_vertices[step][move.from_proc]) {
                                current_proc_transient_memory[move.from_proc] =
                                    std::max(current_proc_transient_memory[move.from_proc],
                                             instance->getComputationalDag().vertex_comm_weight(node));
                            }
                        }
                    }
                }
            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_IN_OUT) {

                step_processor_memory[move.to_step][move.to_proc] +=
                    instance->getComputationalDag().vertex_mem_weight(move.node) +
                    instance->getComputationalDag().vertex_comm_weight(move.node);

                step_processor_memory[move.from_step][move.from_proc] -=
                    (instance->getComputationalDag().vertex_mem_weight(move.node) +
                     instance->getComputationalDag().vertex_comm_weight(move.node));

                for (const auto &pred : instance->getComputationalDag().parents(move.node)) {

                    if (vector_schedule.assignedProcessor(pred) == move.to_proc &&
                        vector_schedule.assignedSuperstep(pred) == move.to_step) {
                        step_processor_memory[move.to_step][move.to_proc] -=
                            instance->getComputationalDag().vertex_comm_weight(pred);
                    } else if (vector_schedule.assignedProcessor(pred) == move.from_proc &&
                               vector_schedule.assignedSuperstep(pred) == move.from_step) {
                        step_processor_memory[move.from_step][move.from_proc] +=
                            instance->getComputationalDag().vertex_comm_weight(pred);
                    }
                }

                for (const auto &succ : instance->getComputationalDag().children(move.node)) {

                    if (vector_schedule.assignedProcessor(succ) == move.to_proc &&
                        vector_schedule.assignedSuperstep(succ) == move.to_step) {
                        step_processor_memory[move.to_step][move.to_proc] -=
                            instance->getComputationalDag().vertex_comm_weight(move.node);
                    } else if (vector_schedule.assignedProcessor(succ) == move.from_proc &&
                               vector_schedule.assignedSuperstep(succ) == move.from_step) {
                        step_processor_memory[move.from_step][move.from_proc] +=
                            instance->getComputationalDag().vertex_comm_weight(move.node);
                    }
                }
            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES) {

                step_processor_memory[move.to_step][move.to_proc] +=
                    instance->getComputationalDag().vertex_comm_weight(move.node);

                step_processor_memory[move.from_step][move.from_proc] -=
                    instance->getComputationalDag().vertex_comm_weight(move.node);

                for (const auto &pred : instance->getComputationalDag().parents(move.node)) {

                    if (vector_schedule.assignedSuperstep(pred) < move.to_step) {

                        auto pair = step_processor_pred[move.to_step][move.to_proc].insert(pred);
                        if (pair.second) {
                            step_processor_memory[move.to_step][move.to_proc] +=
                                instance->getComputationalDag().vertex_comm_weight(pred);
                        }
                    }

                    if (vector_schedule.assignedSuperstep(pred) < move.from_step) {

                        bool remove = true;
                        for (const auto &succ : instance->getComputationalDag().children(pred)) {

                            if (succ == move.node) {
                                continue;
                            }

                            if (vector_schedule.assignedProcessor(succ) == move.from_proc &&
                                vector_schedule.assignedSuperstep(succ) == move.from_step) {
                                remove = false;
                                break;
                            }
                        }

                        if (remove) {
                            step_processor_memory[move.from_step][move.from_proc] -=
                                instance->getComputationalDag().vertex_comm_weight(pred);
                            step_processor_pred[move.from_step][move.from_proc].erase(pred);
                        }
                    }
                }

                if (move.to_step != move.from_step) {

                    for (const auto &succ : instance->getComputationalDag().children(move.node)) {

                        if (move.to_step > move.from_step && vector_schedule.assignedSuperstep(succ) == move.to_step) {

                            if (step_processor_pred[vector_schedule.assignedSuperstep(
                                    succ)][vector_schedule.assignedProcessor(succ)]
                                    .find(move.node) != step_processor_pred[vector_schedule.assignedSuperstep(succ)]
                                                                           [vector_schedule.assignedProcessor(succ)]
                                                                               .end()) {

                                step_processor_memory[vector_schedule.assignedSuperstep(succ)]
                                                     [vector_schedule.assignedProcessor(succ)] -=
                                    instance->getComputationalDag().vertex_comm_weight(move.node);

                                step_processor_pred[vector_schedule.assignedSuperstep(succ)]
                                                   [vector_schedule.assignedProcessor(succ)]
                                                       .erase(move.node);
                            }
                        }

                        if (vector_schedule.assignedSuperstep(succ) > move.to_step) {

                            auto pair = step_processor_pred[vector_schedule.assignedSuperstep(succ)]
                                                           [vector_schedule.assignedProcessor(succ)]
                                                               .insert(move.node);
                            if (pair.second) {
                                step_processor_memory[vector_schedule.assignedSuperstep(succ)]
                                                     [vector_schedule.assignedProcessor(succ)] +=
                                    instance->getComputationalDag().vertex_comm_weight(move.node);
                            }
                        }
                    }
                }
            } else if (instance->getArchitecture().getMemoryConstraintType() == LOCAL_INC_EDGES_2) {

                if (is_source(move.node, instance->getComputationalDag())) {
                    step_processor_memory[move.to_step][move.to_proc] +=
                        instance->getComputationalDag().vertex_mem_weight(move.node);

                    step_processor_memory[move.from_step][move.from_proc] -=
                        instance->getComputationalDag().vertex_mem_weight(move.node);
                }

                for (const auto &pred : instance->getComputationalDag().parents(move.node)) {

                    if (vector_schedule.assignedSuperstep(pred) < move.to_step) {

                        auto pair = step_processor_pred[move.to_step][move.to_proc].insert(pred);
                        if (pair.second) {
                            step_processor_memory[move.to_step][move.to_proc] +=
                                instance->getComputationalDag().vertex_comm_weight(pred);
                        }
                    }

                    if (vector_schedule.assignedSuperstep(pred) < move.from_step) {

                        bool remove = true;
                        for (const auto &succ : instance->getComputationalDag().children(pred)) {

                            if (succ == move.node) {
                                continue;
                            }

                            if (vector_schedule.assignedProcessor(succ) == move.from_proc &&
                                vector_schedule.assignedSuperstep(succ) == move.from_step) {
                                remove = false;
                                break;
                            }
                        }

                        if (remove) {
                            step_processor_memory[move.from_step][move.from_proc] -=
                                instance->getComputationalDag().vertex_comm_weight(pred);
                            step_processor_pred[move.from_step][move.from_proc].erase(pred);
                        }
                    }
                }

                if (move.to_step != move.from_step) {

                    for (const auto &succ : instance->getComputationalDag().children(move.node)) {

                        if (move.to_step > move.from_step && vector_schedule.assignedSuperstep(succ) == move.to_step) {

                            if (step_processor_pred[vector_schedule.assignedSuperstep(
                                    succ)][vector_schedule.assignedProcessor(succ)]
                                    .find(move.node) != step_processor_pred[vector_schedule.assignedSuperstep(succ)]
                                                                           [vector_schedule.assignedProcessor(succ)]
                                                                               .end()) {

                                step_processor_memory[vector_schedule.assignedSuperstep(succ)]
                                                     [vector_schedule.assignedProcessor(succ)] -=
                                    instance->getComputationalDag().vertex_comm_weight(move.node);

                                step_processor_pred[vector_schedule.assignedSuperstep(succ)]
                                                   [vector_schedule.assignedProcessor(succ)]
                                                       .erase(move.node);
                            }
                        }

                        if (vector_schedule.assignedSuperstep(succ) > move.to_step) {

                            auto pair = step_processor_pred[vector_schedule.assignedSuperstep(succ)]
                                                           [vector_schedule.assignedProcessor(succ)]
                                                               .insert(move.node);
                            if (pair.second) {
                                step_processor_memory[vector_schedule.assignedSuperstep(succ)]
                                                     [vector_schedule.assignedProcessor(succ)] +=
                                    instance->getComputationalDag().vertex_comm_weight(move.node);
                            }
                        }
                    }
                }
            }
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
                std::cout << "Edge: " << source(edge, instance->getComputationalDag()) << " -> "
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

    void update_max_work_datastructures(kl_move move) {

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

template<typename Graph_t>
class kl_current_schedule_max_comm : public kl_current_schedule<Graph_t> {

  public:
    std::vector<std::vector<v_commw_t<Graph_t>>> step_processor_send;
    std::vector<v_commw_t<Graph_t>> step_max_send;
    std::vector<v_commw_t<Graph_t>> step_max_receive;

    std::vector<std::vector<v_commw_t<Graph_t>>> step_processor_receive;
    std::vector<v_commw_t<Graph_t>> step_second_max_send;
    std::vector<v_commw_t<Graph_t>> step_second_max_receive;
};

} // namespace osp