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

#include "scheduler/LocalSearchSchedulers/KernighanLin/kl_current_schedule.hpp"

void kl_current_schedule::apply_move(kl_move move) {

    vector_schedule.setAssignedProcessor(move.node, move.to_proc);
    vector_schedule.setAssignedSuperstep(move.node, move.to_step);

    set_schedule.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
    set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);

    current_cost += move.change_in_cost;

    step_processor_work[move.to_step][move.to_proc] += instance->getComputationalDag().nodeWorkWeight(move.node);
    step_processor_work[move.from_step][move.from_proc] -= instance->getComputationalDag().nodeWorkWeight(move.node);

    update_max_work_datastructures(move);
    update_violations(move.node);

    if (use_memory_constraint) {

        if (instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
            step_processor_memory[move.to_step][move.to_proc] +=
                instance->getComputationalDag().nodeMemoryWeight(move.node);
            step_processor_memory[move.from_step][move.from_proc] -=
                instance->getComputationalDag().nodeMemoryWeight(move.node);

        } else if (instance->getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

            if (move.to_proc != move.from_proc) {

                current_proc_persistent_memory[move.to_proc] +=
                    instance->getComputationalDag().nodeMemoryWeight(move.node);
                current_proc_persistent_memory[move.from_proc] -=
                    instance->getComputationalDag().nodeMemoryWeight(move.node);

                current_proc_transient_memory[move.to_proc] =
                    std::max(current_proc_transient_memory[move.to_proc],
                             instance->getComputationalDag().nodeCommunicationWeight(move.node));

                if (current_proc_transient_memory[move.from_proc] ==
                    instance->getComputationalDag().nodeCommunicationWeight(move.node)) {

                    current_proc_transient_memory[move.from_proc] = 0;

                    for (unsigned step = 0; step < num_steps(); step++) {
                        for (const auto &node : set_schedule.step_processor_vertices[step][move.from_proc]) {
                            current_proc_transient_memory[move.from_proc] =
                                std::max(current_proc_transient_memory[move.from_proc],
                                         instance->getComputationalDag().nodeCommunicationWeight(node));
                        }
                    }
                }
            }
        }
    }
}

void kl_current_schedule::update_violations(VertexType node) {

    new_violations.clear();
    resolved_violations.clear();

    for (const auto &edge : instance->getComputationalDag().out_edges(node)) {

        const auto &child = instance->getComputationalDag().target(edge);

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

        const auto &parent = instance->getComputationalDag().source(edge);

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
            std::cout << "Edge: " << instance->getComputationalDag().source(edge.second) << " -> "
                      << instance->getComputationalDag().target(edge.second) << std::endl;
        }
    }

    if (resolved_violations.size() > 0) {
        std::cout << "Resolved violations: " << std::endl;
        for (const auto &edge : resolved_violations) {
            std::cout << "Edge: " << instance->getComputationalDag().source(edge) << " -> "
                      << instance->getComputationalDag().target(edge) << std::endl;
        }
    }

#endif

    if (current_violations.size() > 0) {
        current_feasible = false;
    } else {
        current_feasible = true;
    }
}

void kl_current_schedule::recompute_superstep_max_work(unsigned step) {

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

void kl_current_schedule::update_max_work_datastructures(kl_move move) {

    if (move.from_step == move.to_step) {

        recompute_superstep_max_work(move.from_step);

    } else {

        recompute_superstep_max_work(move.from_step);
        recompute_superstep_max_work(move.to_step);
    }
}

void kl_current_schedule::recompute_current_violations() {

    current_violations.clear();

#ifdef KL_DEBUG
    std::cout << "Recompute current violations:" << std::endl;
#endif

    for (const auto &edge : instance->getComputationalDag().edges()) {

        const auto &source = instance->getComputationalDag().source(edge);
        const auto &target = instance->getComputationalDag().target(edge);

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

void kl_current_schedule::compute_work_memory_datastructures(unsigned start_step, unsigned end_step) {

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
                }
                for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                    step_processor_work[step][proc] += instance->getComputationalDag().nodeWorkWeight(node);

                    if (instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
                        step_processor_memory[step][proc] += instance->getComputationalDag().nodeMemoryWeight(node);
                    } else if (instance->getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
                        current_proc_persistent_memory[proc] += instance->getComputationalDag().nodeMemoryWeight(node);
                        current_proc_transient_memory[proc] =
                            std::max(current_proc_transient_memory[proc],
                                     instance->getComputationalDag().nodeCommunicationWeight(node));
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
                    step_processor_work[step][proc] += instance->getComputationalDag().nodeWorkWeight(node);
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

void kl_current_schedule::remove_superstep(unsigned step) {

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
        }
    }

    step_second_max_work[num_steps()] = 0;
    step_max_work[num_steps()] = 0;

    recompute_current_violations();
    cost_f->compute_current_costs();
}

void kl_current_schedule::set_current_schedule(const IBspSchedule &schedule) {

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

void kl_current_schedule::initialize_superstep_datastructures() {

#ifdef KL_DEBUG
    std::cout << "KLCurrentSchedule initialize datastructures" << std::endl;
#endif

    const unsigned num_procs = instance->numberOfProcessors();

    if (use_memory_constraint) {
        if (instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
            step_processor_memory = std::vector<std::vector<int>>(num_steps(), std::vector<int>(num_procs, 0));
        } else if (instance->getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
            current_proc_persistent_memory = std::vector<int>(num_procs, 0);
            current_proc_transient_memory = std::vector<int>(num_procs, 0);
        }
    }

    step_processor_work = std::vector<std::vector<int>>(num_steps(), std::vector<int>(num_procs, 0));
    step_max_work = std::vector<int>(num_steps(), 0);
    step_second_max_work = std::vector<int>(num_steps(), 0);
}

void kl_current_schedule::cleanup_superstep_datastructures() {

    step_processor_work.clear();
    step_max_work.clear();
    step_second_max_work.clear();

    step_processor_memory.clear();
    current_proc_persistent_memory.clear();
    current_proc_transient_memory.clear();
}