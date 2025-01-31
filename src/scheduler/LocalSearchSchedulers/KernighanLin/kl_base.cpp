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

#include "scheduler/LocalSearchSchedulers/KernighanLin/kl_base.hpp"

RETURN_STATUS kl_base::improveSchedule(BspSchedule &schedule) {

    reset_run_datastructures();

    best_schedule = &schedule;
    current_schedule->instance = &best_schedule->getInstance();

    num_nodes = current_schedule->instance->numberOfVertices();
    num_procs = current_schedule->instance->numberOfProcessors();

    set_parameters();
    initialize_datastructures();

    bool improvement_found = run_local_search_remove_supersteps();

    // assert(best_schedule->satisfiesPrecedenceConstraints());

    schedule.setImprovedLazyCommunicationSchedule();

    if (improvement_found)
        return SUCCESS;
    else
        return BEST_FOUND;
};

void kl_base::initialize_gain_heap_unlocked_nodes(const std::unordered_set<VertexType> &nodes) {

    reset_gain_heap();

    for (const auto &node : nodes) {

        if (locked_nodes.find(node) == locked_nodes.end()) {

            compute_node_gain(node);
            compute_max_gain_insert_or_update_heap(node);
        }
    }
}

void kl_base::initialize_gain_heap(const std::unordered_set<VertexType> &nodes) {

    reset_gain_heap();

    for (const auto &node : nodes) {
        compute_node_gain(node);
        compute_max_gain_insert_or_update_heap(node);
    }
}

void kl_base::update_node_gains(const std::unordered_set<VertexType> &nodes) {

    for (const auto &node : nodes) {

        compute_node_gain(node);
        compute_max_gain_insert_or_update_heap(node);
    }
};

void kl_base::compute_node_gain(unsigned node) {

    const unsigned &current_proc = current_schedule->vector_schedule.assignedProcessor(node);
    const unsigned &current_step = current_schedule->vector_schedule.assignedSuperstep(node);

    for (unsigned new_proc = 0; new_proc < num_procs; new_proc++) {

        if (current_schedule->instance->isCompatible(node, new_proc)) {

            node_gains[node][new_proc][0] = 0.0;
            node_gains[node][new_proc][1] = 0.0;
            node_gains[node][new_proc][2] = 0.0;

            node_change_in_costs[node][new_proc][0] = 0;
            node_change_in_costs[node][new_proc][1] = 0;
            node_change_in_costs[node][new_proc][2] = 0;

            compute_comm_gain(node, current_step, current_proc, new_proc);
            compute_work_gain(node, current_step, current_proc, new_proc);

            if (current_schedule->use_memory_constraint) {

                if (current_schedule->instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
                    if (current_schedule->step_processor_memory[current_schedule->vector_schedule.assignedSuperstep(
                            node)][new_proc] +
                            current_schedule->instance->getComputationalDag().nodeMemoryWeight(node) >
                        current_schedule->instance->memoryBound(new_proc)) {

                        node_gains[node][new_proc][1] = std::numeric_limits<double>::lowest();
                    }
                    if (current_schedule->vector_schedule.assignedSuperstep(node) > 0) {
                        if (current_schedule
                                    ->step_processor_memory[current_schedule->vector_schedule.assignedSuperstep(node) -
                                                            1][new_proc] +
                                current_schedule->instance->getComputationalDag().nodeMemoryWeight(node) >
                            current_schedule->instance->memoryBound(new_proc)) {

                            node_gains[node][new_proc][0] = std::numeric_limits<double>::lowest();
                        }
                    }

                    if (current_schedule->vector_schedule.assignedSuperstep(node) < current_schedule->num_steps() - 1) {
                        if (current_schedule
                                    ->step_processor_memory[current_schedule->vector_schedule.assignedSuperstep(node) +
                                                            1][new_proc] +
                                current_schedule->instance->getComputationalDag().nodeMemoryWeight(node) >
                            current_schedule->instance->memoryBound(new_proc)) {

                            node_gains[node][new_proc][2] = std::numeric_limits<double>::lowest();
                        }
                    }
                } else if (current_schedule->instance->getArchitecture().getMemoryConstraintType() ==
                           PERSISTENT_AND_TRANSIENT) {
                    if (current_schedule->current_proc_persistent_memory[new_proc] +
                            current_schedule->instance->getComputationalDag().nodeMemoryWeight(node) +
                            std::max(current_schedule->current_proc_transient_memory[new_proc],
                                     current_schedule->instance->getComputationalDag().nodeCommunicationWeight(node)) >
                        current_schedule->instance->memoryBound(new_proc)) {

                        node_gains[node][new_proc][0] = std::numeric_limits<double>::lowest();
                        node_gains[node][new_proc][1] = std::numeric_limits<double>::lowest();
                        node_gains[node][new_proc][2] = std::numeric_limits<double>::lowest();
                    }
                }
            }
        } else {

            node_gains[node][new_proc][0] = std::numeric_limits<double>::lowest();
            node_gains[node][new_proc][1] = std::numeric_limits<double>::lowest();
            node_gains[node][new_proc][2] = std::numeric_limits<double>::lowest();
        }
    }
}

void kl_base::initialize_datastructures() {

#ifdef KL_DEBUG
    std::cout << "KLBase initialize datastructures" << std::endl;
#endif

    node_gains = std::vector<std::vector<std::vector<double>>>(
        num_nodes, std::vector<std::vector<double>>(num_procs, std::vector<double>(3, 0)));

    node_change_in_costs = std::vector<std::vector<std::vector<double>>>(
        num_nodes, std::vector<std::vector<double>>(num_procs, std::vector<double>(3, 0)));

    unlock = std::vector<unsigned>(num_nodes, parameters.max_num_unlocks);

    current_schedule->initialize_current_schedule(*best_schedule);
    best_schedule_costs = current_schedule->current_cost;
}

void kl_base::set_parameters() {

    if (num_nodes < 250) {

        parameters.max_outer_iterations = 300;

        parameters.select_all_nodes = true;
        parameters.selection_threshold = num_nodes;

    } else if (num_nodes < 1000) {

        parameters.max_outer_iterations = num_nodes / 2;

        parameters.select_all_nodes = true;
        parameters.selection_threshold = num_nodes;

    } else if (num_nodes < 5000) {

        parameters.max_outer_iterations = 4 * std::sqrt(num_nodes);

        parameters.selection_threshold = 0.33 * num_nodes;

    } else if (num_nodes < 10000) {

        parameters.max_outer_iterations = 3 * std::sqrt(num_nodes);

        parameters.selection_threshold = num_nodes * 0.33;

    } else if (num_nodes < 50000) {

        parameters.max_outer_iterations = std::sqrt(num_nodes);

        parameters.selection_threshold = num_nodes * 0.2;

    } else if (num_nodes < 100000) {

        parameters.max_outer_iterations = 2 * std::log(num_nodes);

        parameters.selection_threshold = num_nodes * 0.1;

    } else {

        parameters.max_outer_iterations = std::log(num_nodes);

        parameters.selection_threshold = num_nodes * 0.1;
    }

    if (parameters.quick_pass) {
        parameters.max_outer_iterations = 25;
        parameters.max_no_improvement_iterations = 3;
    }

#ifdef KL_DEBUG
    if (parameters.select_all_nodes)
        std::cout << "KLBase set parameters, select all nodes" << std::endl;
    else
        std::cout << "KLBase set parameters, selection threshold: " << parameters.selection_threshold << std::endl;
#endif
}

void kl_base::select_unlock_neighbors(VertexType node) {

    for (const auto &target : current_schedule->instance->getComputationalDag().children(node)) {

        if (check_node_unlocked(target)) {

            node_selection.insert(target);
            nodes_to_update.insert(target);
        }
    }

    for (const auto &source : current_schedule->instance->getComputationalDag().parents(node)) {

        if (check_node_unlocked(source)) {

            node_selection.insert(source);
            nodes_to_update.insert(source);
        }
    }
}

bool kl_base::check_node_unlocked(VertexType node) {

    if (super_locked_nodes.find(node) == super_locked_nodes.end() && locked_nodes.find(node) == locked_nodes.end()) {
        return true;
    }
    return false;
};

bool kl_base::unlock_node(VertexType node) {

    if (super_locked_nodes.find(node) == super_locked_nodes.end()) {

        if (locked_nodes.find(node) == locked_nodes.end()) {
            return true;
        } else if (locked_nodes.find(node) != locked_nodes.end() && unlock[node] > 0) {
            unlock[node]--;

            locked_nodes.erase(node);

            return true;
        }
    }
    return false;
};

void kl_base::compute_nodes_to_update(kl_move move) {

    nodes_to_update.clear();

    for (const auto &target : current_schedule->instance->getComputationalDag().children(move.node)) {

        if (node_selection.find(target) != node_selection.end() && locked_nodes.find(target) == locked_nodes.end() &&
            super_locked_nodes.find(target) == super_locked_nodes.end()) {

            nodes_to_update.insert(target);
        }
    }

    for (const auto &source : current_schedule->instance->getComputationalDag().parents(move.node)) {

        if (node_selection.find(source) != node_selection.end() && locked_nodes.find(source) == locked_nodes.end() &&
            super_locked_nodes.find(source) == super_locked_nodes.end()) {

            nodes_to_update.insert(source);
        }
    }

    const unsigned start_step =
        std::min(move.from_step, move.to_step) == 0 ? 0 : std::min(move.from_step, move.to_step) - 1;
    const unsigned end_step = std::min(current_schedule->num_steps(), std::max(move.from_step, move.to_step) + 2);

    for (unsigned step = start_step; step < end_step; step++) {

        for (unsigned proc = 0; proc < num_procs; proc++) {

            for (const auto &node : current_schedule->set_schedule.step_processor_vertices[step][proc]) {

                if (node_selection.find(node) != node_selection.end() &&
                    locked_nodes.find(node) == locked_nodes.end() &&
                    super_locked_nodes.find(node) == super_locked_nodes.end()) {

                    nodes_to_update.insert(node);
                }
            }
        }
    }
}

void kl_base::compute_work_gain(unsigned node, unsigned current_step, unsigned current_proc, unsigned new_proc) {

    if (current_proc == new_proc) {

        node_gains[node][current_proc][1] = std::numeric_limits<double>::lowest();

    } else {

        if (current_schedule->step_max_work[current_step] ==
                current_schedule->step_processor_work[current_step][current_proc] &&
            current_schedule->step_processor_work[current_step][current_proc] >
                current_schedule->step_second_max_work[current_step]) {

            // new max
            const double new_max_work =
                std::max(current_schedule->step_processor_work[current_step][current_proc] -
                             current_schedule->instance->getComputationalDag().nodeWorkWeight(node),
                         current_schedule->step_second_max_work[current_step]);

            if (current_schedule->step_processor_work[current_step][new_proc] +
                    current_schedule->instance->getComputationalDag().nodeWorkWeight(node) >
                new_max_work) {

                const double gain = current_schedule->step_max_work[current_step] -
                                    (current_schedule->step_processor_work[current_step][new_proc] +
                                     current_schedule->instance->getComputationalDag().nodeWorkWeight(node));

                node_gains[node][new_proc][1] += gain;
                node_change_in_costs[node][new_proc][1] -= gain;

            } else {

                const double gain = current_schedule->step_max_work[current_step] - new_max_work;
                node_gains[node][new_proc][1] += gain;
                node_change_in_costs[node][new_proc][1] -= gain;
            }

        } else {

            if (current_schedule->step_max_work[current_step] <
                current_schedule->instance->getComputationalDag().nodeWorkWeight(node) +
                    current_schedule->step_processor_work[current_step][new_proc]) {

                const double gain = (current_schedule->instance->getComputationalDag().nodeWorkWeight(node) +
                                     current_schedule->step_processor_work[current_step][new_proc] -
                                     current_schedule->step_max_work[current_step]);

                node_gains[node][new_proc][1] -= gain;
                node_change_in_costs[node][new_proc][1] += gain;
            }
        }
    }

    if (current_step > 0) {

        if (current_schedule->step_max_work[current_step - 1] <
            current_schedule->step_processor_work[current_step - 1][new_proc] +
                current_schedule->instance->getComputationalDag().nodeWorkWeight(node)) {

            const double gain = current_schedule->step_processor_work[current_step - 1][new_proc] +
                                current_schedule->instance->getComputationalDag().nodeWorkWeight(node) -
                                current_schedule->step_max_work[current_step - 1];

            node_gains[node][new_proc][0] -= gain;

            node_change_in_costs[node][new_proc][0] += gain;
        }

        if (current_schedule->step_max_work[current_step] ==
                current_schedule->step_processor_work[current_step][current_proc] &&
            current_schedule->step_processor_work[current_step][current_proc] >
                current_schedule->step_second_max_work[current_step]) {

            if (current_schedule->step_max_work[current_step] -
                    current_schedule->instance->getComputationalDag().nodeWorkWeight(node) >
                current_schedule->step_second_max_work[current_step]) {

                const double gain = current_schedule->instance->getComputationalDag().nodeWorkWeight(node);
                node_gains[node][new_proc][0] += gain;
                node_change_in_costs[node][new_proc][0] -= gain;
            } else {

                const double gain = current_schedule->step_max_work[current_step] -
                                    current_schedule->step_second_max_work[current_step];
                node_gains[node][new_proc][0] += gain;
                node_change_in_costs[node][new_proc][0] -= gain;
            }
        }

    } else {

        node_gains[node][new_proc][0] = std::numeric_limits<double>::lowest();
    }

    if (current_step < current_schedule->num_steps() - 1) {

        if (current_schedule->step_max_work[current_step + 1] <
            current_schedule->step_processor_work[current_step + 1][new_proc] +
                current_schedule->instance->getComputationalDag().nodeWorkWeight(node)) {

            const double gain = current_schedule->step_processor_work[current_step + 1][new_proc] +
                                current_schedule->instance->getComputationalDag().nodeWorkWeight(node) -
                                current_schedule->step_max_work[current_step + 1];

            node_gains[node][new_proc][2] -= gain;
            node_change_in_costs[node][new_proc][2] += gain;
        }

        if (current_schedule->step_max_work[current_step] ==
                current_schedule->step_processor_work[current_step][current_proc] &&
            current_schedule->step_processor_work[current_step][current_proc] >
                current_schedule->step_second_max_work[current_step]) {

            if ((current_schedule->step_max_work[current_step] -
                 current_schedule->instance->getComputationalDag().nodeWorkWeight(node)) >
                current_schedule->step_second_max_work[current_step]) {

                const double gain = current_schedule->instance->getComputationalDag().nodeWorkWeight(node);
                node_gains[node][new_proc][2] += gain;
                node_change_in_costs[node][new_proc][2] -= gain;

            } else {

                const double gain = current_schedule->step_max_work[current_step] -
                                    current_schedule->step_second_max_work[current_step];
                node_gains[node][new_proc][2] += gain;
                node_change_in_costs[node][new_proc][2] -= gain;
            }
        }
    } else {

        node_gains[node][new_proc][2] = std::numeric_limits<double>::lowest();
    }
}

void kl_base::select_nodes() {

    if (parameters.select_all_nodes) {

        for (unsigned i = 0; i < num_nodes; i++) {
            if (super_locked_nodes.find(i) == super_locked_nodes.end())
                node_selection.insert(i);
        }

    } else {
        select_nodes_threshold(parameters.selection_threshold - super_locked_nodes.size());
    }
}

void kl_base::select_nodes_threshold(unsigned threshold) {

    std::uniform_int_distribution<> dis(0, num_nodes - 1);

    while (node_selection.size() < threshold) {

        auto node = dis(gen);

        if (super_locked_nodes.find(node) == super_locked_nodes.end()) {
            node_selection.insert(node);
        }
    }
}

void kl_base::reset_locked_nodes() {

    for (const auto &i : locked_nodes) {

        unlock[i] = parameters.max_num_unlocks;
    }

    locked_nodes.clear();
}

void kl_base::select_nodes_permutation_threshold(unsigned threshold) {

    std::vector<VertexType> permutation(num_nodes);
    std::iota(std::begin(permutation), std::end(permutation), 0);

    std::shuffle(permutation.begin(), permutation.end(), gen);

    for (unsigned i = 0; i < threshold; i++) {

        if (super_locked_nodes.find(permutation[i]) == super_locked_nodes.end())
            node_selection.insert(permutation[i]);
    }
}

kl_move kl_base::find_best_move() {

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

    unsigned i = randInt(count);
    kl_move best_move = kl_move((*node_heap_handles[max_nodes[i]]));

    max_gain_heap.erase(node_heap_handles[max_nodes[i]]);
    node_heap_handles.erase(max_nodes[i]);

    return best_move;
};

kl_move kl_base::compute_best_move(VertexType node) {

    double node_max_gain = std::numeric_limits<double>::lowest();
    double node_change_in_cost = 0;
    unsigned node_best_step = 0;
    unsigned node_best_proc = 0;

    double proc_change_in_cost = 0;
    double proc_max = 0;
    unsigned best_step = 0;
    for (unsigned proc = 0; proc < num_procs; proc++) {

        unsigned rand_count = 0;

        if (current_schedule->vector_schedule.assignedSuperstep(node) > 0 &&
            current_schedule->vector_schedule.assignedSuperstep(node) < current_schedule->num_steps() - 1) {

            if (node_gains[node][proc][0] > node_gains[node][proc][1]) {

                if (node_gains[node][proc][0] > node_gains[node][proc][2]) {
                    proc_max = node_gains[node][proc][0];
                    proc_change_in_cost = node_change_in_costs[node][proc][0];
                    best_step = 0;

                } else {
                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                }

            } else {

                if (node_gains[node][proc][1] > node_gains[node][proc][2]) {

                    proc_max = node_gains[node][proc][1];
                    proc_change_in_cost = node_change_in_costs[node][proc][1];
                    best_step = 1;
                } else {

                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                }
            }

        } else if (current_schedule->vector_schedule.assignedSuperstep(node) == 0 &&
                   current_schedule->vector_schedule.assignedSuperstep(node) < current_schedule->num_steps() - 1) {

            if (node_gains[node][proc][2] > node_gains[node][proc][1]) {

                proc_max = node_gains[node][proc][2];
                proc_change_in_cost = node_change_in_costs[node][proc][2];
                best_step = 2;
            } else {

                proc_max = node_gains[node][proc][1];
                proc_change_in_cost = node_change_in_costs[node][proc][1];
                best_step = 1;
            }

        } else if (current_schedule->vector_schedule.assignedSuperstep(node) > 0 &&
                   current_schedule->vector_schedule.assignedSuperstep(node) == current_schedule->num_steps() - 1) {

            if (node_gains[node][proc][1] > node_gains[node][proc][0]) {

                proc_max = node_gains[node][proc][1];
                proc_change_in_cost = node_change_in_costs[node][proc][1];
                best_step = 1;
            } else {

                proc_max = node_gains[node][proc][0];
                proc_change_in_cost = node_change_in_costs[node][proc][0];
                best_step = 0;
            }
        } else {
            proc_max = node_gains[node][proc][1];
            proc_change_in_cost = node_change_in_costs[node][proc][1];
            best_step = 1;
        }

        if (node_max_gain < proc_max) {

            node_max_gain = proc_max;
            node_change_in_cost = proc_change_in_cost;
            node_best_step = current_schedule->vector_schedule.assignedSuperstep(node) + best_step - 1;
            node_best_proc = proc;
            rand_count = 0;

        } else if (node_max_gain == proc_max) {

            if (rand() % (2 + rand_count) == 0) {
                node_max_gain = proc_max;
                node_change_in_cost = proc_change_in_cost;
                node_best_step = current_schedule->vector_schedule.assignedSuperstep(node) + best_step - 1;
                node_best_proc = proc;
                rand_count++;
            }
        }
    }

    return kl_move(node, node_max_gain, node_change_in_cost, current_schedule->vector_schedule.assignedProcessor(node),
                   current_schedule->vector_schedule.assignedSuperstep(node), node_best_proc, node_best_step);
}

kl_move kl_base::best_move_change_superstep(VertexType node) {

    double node_max_gain = std::numeric_limits<double>::lowest();
    double node_change_in_cost = 0;
    unsigned node_best_step = 0;
    unsigned node_best_proc = 0;

    double proc_change_in_cost = 0;
    double proc_max = 0;
    unsigned best_step = 0;
    for (unsigned proc = 0; proc < num_procs; proc++) {

        if (current_schedule->vector_schedule.assignedSuperstep(node) > 0 &&
            current_schedule->vector_schedule.assignedSuperstep(node) < current_schedule->num_steps() - 1) {

            if (node_gains[node][proc][0] > node_gains[node][proc][2]) {
                proc_max = node_gains[node][proc][0];
                proc_change_in_cost = node_change_in_costs[node][proc][0];
                best_step = 0;

            } else {
                proc_max = node_gains[node][proc][2];
                proc_change_in_cost = node_change_in_costs[node][proc][2];
                best_step = 2;
            }

        } else if (current_schedule->vector_schedule.assignedSuperstep(node) == 0 &&
                   current_schedule->vector_schedule.assignedSuperstep(node) < current_schedule->num_steps() - 1) {

            proc_max = node_gains[node][proc][2];
            proc_change_in_cost = node_change_in_costs[node][proc][2];
            best_step = 2;

        } else if (current_schedule->vector_schedule.assignedSuperstep(node) > 0 &&
                   current_schedule->vector_schedule.assignedSuperstep(node) == current_schedule->num_steps() - 1) {

            proc_max = node_gains[node][proc][0];
            proc_change_in_cost = node_change_in_costs[node][proc][0];
            best_step = 0;

        } else {
            throw std::invalid_argument("error lk base best_move_change_superstep");
        }

        if (node_max_gain < proc_max) {

            node_max_gain = proc_max;
            node_change_in_cost = proc_change_in_cost;
            node_best_step = current_schedule->vector_schedule.assignedSuperstep(node) + best_step - 1;
            node_best_proc = proc;
        }
    }

    return kl_move(node, node_max_gain, node_change_in_cost, current_schedule->vector_schedule.assignedProcessor(node),
                   current_schedule->vector_schedule.assignedSuperstep(node), node_best_proc, node_best_step);
}

double kl_base::compute_max_gain_insert_or_update_heap(VertexType node) {

    double node_max_gain = std::numeric_limits<double>::lowest();
    double node_change_in_cost = 0;
    unsigned node_best_step = 0;
    unsigned node_best_proc = 0;

    double proc_change_in_cost = 0;
    double proc_max = 0;
    unsigned best_step = 0;

    for (unsigned proc = 0; proc < num_procs; proc++) {

        unsigned rand_count = 0;

        if (current_schedule->vector_schedule.assignedSuperstep(node) > 0 &&
            current_schedule->vector_schedule.assignedSuperstep(node) < current_schedule->num_steps() - 1) {

            if (node_gains[node][proc][0] > node_gains[node][proc][1]) {

                if (node_gains[node][proc][0] > node_gains[node][proc][2]) {
                    proc_max = node_gains[node][proc][0];
                    proc_change_in_cost = node_change_in_costs[node][proc][0];
                    best_step = 0;

                } else {
                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                }

            } else {

                if (node_gains[node][proc][1] > node_gains[node][proc][2]) {

                    proc_max = node_gains[node][proc][1];
                    proc_change_in_cost = node_change_in_costs[node][proc][1];
                    best_step = 1;
                } else {

                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                }
            }

        } else if (current_schedule->vector_schedule.assignedSuperstep(node) == 0 &&
                   current_schedule->vector_schedule.assignedSuperstep(node) < current_schedule->num_steps() - 1) {

            if (node_gains[node][proc][2] > node_gains[node][proc][1]) {

                proc_max = node_gains[node][proc][2];
                proc_change_in_cost = node_change_in_costs[node][proc][2];
                best_step = 2;
            } else {

                proc_max = node_gains[node][proc][1];
                proc_change_in_cost = node_change_in_costs[node][proc][1];
                best_step = 1;
            }

        } else if (current_schedule->vector_schedule.assignedSuperstep(node) > 0 &&
                   current_schedule->vector_schedule.assignedSuperstep(node) == current_schedule->num_steps() - 1) {

            if (node_gains[node][proc][1] > node_gains[node][proc][0]) {

                proc_max = node_gains[node][proc][1];
                proc_change_in_cost = node_change_in_costs[node][proc][1];
                best_step = 1;
            } else {

                proc_max = node_gains[node][proc][0];
                proc_change_in_cost = node_change_in_costs[node][proc][0];
                best_step = 0;
            }
        } else {
            proc_max = node_gains[node][proc][1];
            proc_change_in_cost = node_change_in_costs[node][proc][1];
            best_step = 1;
        }

        if (node_max_gain < proc_max) {

            node_max_gain = proc_max;
            node_change_in_cost = proc_change_in_cost;
            node_best_step = current_schedule->vector_schedule.assignedSuperstep(node) + best_step - 1;
            node_best_proc = proc;
            rand_count = 0;

        } else if (node_max_gain == proc_max) {

            if (rand() % (2 + rand_count) == 0) {
                node_max_gain = proc_max;
                node_change_in_cost = proc_change_in_cost;
                node_best_step = current_schedule->vector_schedule.assignedSuperstep(node) + best_step - 1;
                node_best_proc = proc;
                rand_count++;
            }
        }
    }

    if (node_heap_handles.find(node) != node_heap_handles.end()) {

        (*node_heap_handles[node]).to_proc = node_best_proc;
        (*node_heap_handles[node]).to_step = node_best_step;
        (*node_heap_handles[node]).change_in_cost = node_change_in_cost;

        if ((*node_heap_handles[node]).gain != node_max_gain) {

            (*node_heap_handles[node]).gain = node_max_gain;
            max_gain_heap.update(node_heap_handles[node]);
        }

    } else {

        // if (node_max_gain < parameters.gain_threshold && node_change_in_cost > parameters.change_in_cost_threshold)
        //     return node_max_gain;

        kl_move move(node, node_max_gain, node_change_in_cost,
                     current_schedule->vector_schedule.assignedProcessor(node),
                     current_schedule->vector_schedule.assignedSuperstep(node), node_best_proc, node_best_step);
        node_heap_handles[node] = max_gain_heap.push(move);
    }

    return node_max_gain;
}

void kl_base::reset_gain_heap() {

    max_gain_heap.clear();
    node_heap_handles.clear();
}

void kl_base::save_best_schedule(const IBspSchedule &schedule) {

    for (unsigned node = 0; node < num_nodes; node++) {

        best_schedule->setAssignedProcessor(node, schedule.assignedProcessor(node));
        best_schedule->setAssignedSuperstep(node, schedule.assignedSuperstep(node));
    }
    best_schedule->updateNumberOfSupersteps();
}

void kl_base::reverse_move_best_schedule(kl_move move) {
    best_schedule->setAssignedProcessor(move.node, move.from_proc);
    best_schedule->setAssignedSuperstep(move.node, move.from_step);
}

void kl_base::cleanup_datastructures() {

    node_change_in_costs.clear();
    node_gains.clear();

    unlock.clear();

    max_gain_heap.clear();
    node_heap_handles.clear();

    current_schedule->cleanup_superstep_datastructures();
}

bool kl_base::check_violation_locked() {

    if (current_schedule->current_violations.empty())
        return false;

    for (auto &edge : current_schedule->current_violations) {

        const auto &source = current_schedule->instance->getComputationalDag().source(edge);
        const auto &target = current_schedule->instance->getComputationalDag().target(edge);

        if (locked_nodes.find(source) == locked_nodes.end() || locked_nodes.find(target) == locked_nodes.end()) {
            return false;
        }

        bool abort = false;
        if (locked_nodes.find(source) != locked_nodes.end()) {

            if (unlock_node(source)) {
                nodes_to_update.insert(source);
                node_selection.insert(source);
            } else {
                abort = true;
            }
        }

        if (locked_nodes.find(target) != locked_nodes.end()) {

            if (unlock_node(target)) {
                nodes_to_update.insert(target);
                node_selection.insert(target);
                abort = false;
            }
        }

        if (abort) {
            return true;
        }
    }

    return false;
}

void kl_base::print_heap() {

    std::cout << "heap current size: " << max_gain_heap.size() << std::endl;
    std::cout << "heap top node " << max_gain_heap.top().node << " gain " << max_gain_heap.top().gain << std::endl;

    unsigned count = 0;
    for (auto it = max_gain_heap.ordered_begin(); it != max_gain_heap.ordered_end(); ++it) {
        std::cout << "node " << it->node << " gain " << it->gain << " to proc " << it->to_proc << " to step "
                  << it->to_step << std::endl;

        if (count++ > 25) {
            break;
        }
    }
}

void kl_base::select_nodes_violations() {

    if (current_schedule->current_violations.empty()) {
        select_nodes();
        return;
    }

    for (const auto &edge : current_schedule->current_violations) {

        const auto &source = current_schedule->instance->getComputationalDag().source(edge);
        const auto &target = current_schedule->instance->getComputationalDag().target(edge);

        node_selection.insert(source);
        node_selection.insert(target);

        for (const auto &child : current_schedule->instance->getComputationalDag().children(source)) {
            if (child != target) {
                node_selection.insert(child);
            }
        }

        for (const auto &parent : current_schedule->instance->getComputationalDag().parents(source)) {
            if (parent != target) {
                node_selection.insert(parent);
            }
        }

        for (const auto &child : current_schedule->instance->getComputationalDag().children(target)) {
            if (child != source) {
                node_selection.insert(child);
            }
        }

        for (const auto &parent : current_schedule->instance->getComputationalDag().parents(target)) {
            if (parent != source) {
                node_selection.insert(parent);
            }
        }
    }
}

void kl_base::select_nodes_conseque_max_work(bool do_not_select_super_locked_nodes) {

    if (step_selection_epoch_counter > parameters.max_step_selection_epochs) {

#ifdef KL_DEBUG
        std::cout << "step selection epoch counter exceeded. conseque max work" << std::endl;
#endif

        select_nodes();
        return;
    }

    unsigned max_work_step = 0;
    unsigned max_step = 0;
    unsigned second_max_work_step = 0;
    unsigned second_max_step = 0;

    for (unsigned proc = 0; proc < num_procs; proc++) {

        if (current_schedule->step_processor_work[step_selection_counter][proc] > max_work_step) {
            second_max_work_step = max_work_step;
            second_max_step = max_step;
            max_work_step = current_schedule->step_processor_work[step_selection_counter][proc];
            max_step = proc;

        } else if (current_schedule->step_processor_work[step_selection_counter][proc] > second_max_work_step) {
            second_max_work_step = current_schedule->step_processor_work[step_selection_counter][proc];
            second_max_step = proc;
        }
    }

    if (current_schedule->set_schedule.step_processor_vertices[step_selection_counter][max_step].size() <
        parameters.selection_threshold * .66) {

        node_selection.insert(
            current_schedule->set_schedule.step_processor_vertices[step_selection_counter][max_step].begin(),
            current_schedule->set_schedule.step_processor_vertices[step_selection_counter][max_step].end());

    } else {

        std::sample(current_schedule->set_schedule.step_processor_vertices[step_selection_counter][max_step].begin(),
                    current_schedule->set_schedule.step_processor_vertices[step_selection_counter][max_step].end(),
                    std::inserter(node_selection, node_selection.end()),
                    (unsigned)std::round(parameters.selection_threshold * .66), gen);
    }

    if (current_schedule->set_schedule.step_processor_vertices[step_selection_counter][second_max_step].size() <
        parameters.selection_threshold * .33) {

        node_selection.insert(
            current_schedule->set_schedule.step_processor_vertices[step_selection_counter][second_max_step].begin(),
            current_schedule->set_schedule.step_processor_vertices[step_selection_counter][second_max_step].end());

    } else {

        std::sample(
            current_schedule->set_schedule.step_processor_vertices[step_selection_counter][second_max_step].begin(),
            current_schedule->set_schedule.step_processor_vertices[step_selection_counter][second_max_step].end(),
            std::inserter(node_selection, node_selection.end()),
            (unsigned)std::round(parameters.selection_threshold * .33), gen);
    }

    if (do_not_select_super_locked_nodes) {
        for (const auto &node : super_locked_nodes) {
            node_selection.erase(node);
        }
    }

#ifdef KL_DEBUG
    std::cout << "step selection conseque max work, node selection size " << node_selection.size()
              << " ... selected nodes assigend to superstep " << step_selection_counter << " and procs " << max_step
              << " and " << second_max_step << std::endl;
#endif

    step_selection_counter++;
    if (step_selection_counter >= current_schedule->num_steps()) {
        step_selection_counter = 0;
        step_selection_epoch_counter++;
    }
}


void kl_base::select_nodes_check_reset_superstep() {

    if (step_selection_epoch_counter > parameters.max_step_selection_epochs) {

#ifdef KL_DEBUG
        std::cout << "step selection epoch counter exceeded, reset supersteps" << std::endl;
#endif

        select_nodes();
        return;
    }

    for (unsigned step_to_remove = step_selection_counter; step_to_remove < current_schedule->num_steps();
         step_to_remove++) {

#ifdef KL_DEBUG
        std::cout << "checking step to reset " << step_to_remove << " / " << current_schedule->num_steps()
                  << std::endl;
#endif

        if (check_reset_superstep(step_to_remove)) {

#ifdef KL_DEBUG
            std::cout << "trying to reset superstep " << step_to_remove << std::endl;
#endif

            if (scatter_nodes_reset_superstep(step_to_remove)) {

                for (unsigned proc = 0; proc < num_procs; proc++) {

                    if (step_to_remove < current_schedule->num_steps()) {
                        node_selection.insert(
                            current_schedule->set_schedule.step_processor_vertices[step_to_remove][proc].begin(),
                            current_schedule->set_schedule.step_processor_vertices[step_to_remove][proc].end());
                    }

                    if (step_to_remove > 0) {
                        node_selection.insert(
                            current_schedule->set_schedule.step_processor_vertices[step_to_remove - 1][proc].begin(),
                            current_schedule->set_schedule.step_processor_vertices[step_to_remove - 1][proc].end());
                    }
                }

                step_selection_counter = step_to_remove + 1;

                if (step_selection_counter >= current_schedule->num_steps()) {
                    step_selection_counter = 0; 
                    step_selection_epoch_counter++;
                }

                parameters.violations_threshold = 0;
                super_locked_nodes.clear();
#ifdef KL_DEBUG
                std::cout << "---- reset super locked nodes" << std::endl;
#endif

                return;
            }
        }
    }

#ifdef KL_DEBUG
    std::cout << "no superstep to reset" << std::endl;
#endif

    step_selection_epoch_counter++;
    select_nodes();
    return;
}


bool kl_base::check_reset_superstep(unsigned step) {

    if (current_schedule->num_steps() <= 2) {
        return false;
    }

    unsigned total_work = 0;
    int max_total_work = 0;
    int min_total_work = std::numeric_limits<int>::max();

    for (unsigned proc = 0; proc < num_procs; proc++) {
        total_work += current_schedule->step_processor_work[step][proc];
        max_total_work = std::max(max_total_work, current_schedule->step_processor_work[step][proc]);
        min_total_work = std::min(min_total_work, current_schedule->step_processor_work[step][proc]);
    }


#ifdef KL_DEBUG

    std::cout << " avg " << static_cast<double>(total_work) / static_cast<double>(current_schedule->instance->numberOfProcessors()) << " max " << max_total_work << " min " << min_total_work << std::endl;
#endif

    if ( static_cast<double>(total_work) / static_cast<double>(current_schedule->instance->numberOfProcessors()) * 0.1 > static_cast<double>(min_total_work)) {
        return true;
    }
    
    return false;
}

bool kl_base::scatter_nodes_reset_superstep(unsigned step) {

    assert(step < current_schedule->num_steps());

    std::vector<kl_move> moves;

    bool abort = false;

    for (unsigned proc = 0; proc < num_procs; proc++) {
        for (const auto &node : current_schedule->set_schedule.step_processor_vertices[step][proc]) {

            compute_node_gain(node);
            moves.push_back(best_move_change_superstep(node));

            if (moves.back().gain == std::numeric_limits<double>::lowest()) {
                abort = true;
                break;
            }

            if (current_schedule->use_memory_constraint) {

                if (current_schedule->instance->getArchitecture().getMemoryConstraintType() ==
                    PERSISTENT_AND_TRANSIENT) {

                    if (moves.back().to_proc != moves.back().from_proc) {
                        current_schedule->current_proc_persistent_memory[moves.back().to_proc] +=
                            current_schedule->instance->getComputationalDag().nodeMemoryWeight(node);
                        current_schedule->current_proc_persistent_memory[moves.back().from_proc] -=
                            current_schedule->instance->getComputationalDag().nodeMemoryWeight(node);

                        // current_schedule->current_proc_transient_memory[moves.back().to_proc] =
                        //     std::max(current_schedule->current_proc_transient_memory[moves.back().to_proc],
                        //              current_schedule->instance->getComputationalDag().nodeCommunicationWeight(node));
                        // TODO: implement this properly if PERSISTENT_AND_TRANSIENT becomes relevant
                    }

                } else if (current_schedule->instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
                    current_schedule->step_processor_work[moves.back().to_step][moves.back().to_proc] +=
                        current_schedule->instance->getComputationalDag().nodeWorkWeight(node);
                }
            }
        }

        if (abort) {
            break;
        }

        current_schedule->set_schedule.step_processor_vertices[step][proc].clear();
    }

    if (abort) {

        for (const auto &move : moves) {

            current_schedule->set_schedule.step_processor_vertices[step][move.from_proc].insert(move.node);

            if (current_schedule->use_memory_constraint) {

                if (current_schedule->instance->getArchitecture().getMemoryConstraintType() ==
                    PERSISTENT_AND_TRANSIENT) {

                    if (move.to_proc != move.from_proc) {
                        current_schedule->current_proc_persistent_memory[move.to_proc] -=
                            current_schedule->instance->getComputationalDag().nodeMemoryWeight(move.node);
                        current_schedule->current_proc_persistent_memory[move.from_proc] +=
                            current_schedule->instance->getComputationalDag().nodeMemoryWeight(move.node);
                    }

                } else if (current_schedule->instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
                    current_schedule->step_processor_work[move.to_step][move.to_proc] -=
                        current_schedule->instance->getComputationalDag().nodeWorkWeight(move.node);
                }
            }
        }
        return false;
    }

    for (const auto &move : moves) {

#ifdef KL_DEBUG
        std::cout << "scatter node " << move.node << " to proc " << move.to_proc << " to step " << move.to_step
                  << std::endl;
#endif

        current_schedule->vector_schedule.setAssignedSuperstep(move.node, move.to_step);
        current_schedule->vector_schedule.setAssignedProcessor(move.node, move.to_proc);
        current_schedule->set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
    }

    current_schedule->reset_superstep(step);

    return true;
}


void kl_base::select_nodes_check_remove_superstep() {

    if (step_selection_epoch_counter > parameters.max_step_selection_epochs) {

#ifdef KL_DEBUG
        std::cout << "step selection epoch counter exceeded, remove supersteps" << std::endl;
#endif

        select_nodes();
        return;
    }

    for (unsigned step_to_remove = step_selection_counter; step_to_remove < current_schedule->num_steps();
         step_to_remove++) {

#ifdef KL_DEBUG
        std::cout << "checking step to remove " << step_to_remove << " / " << current_schedule->num_steps()
                  << std::endl;
#endif

        if (check_remove_superstep(step_to_remove)) {

#ifdef KL_DEBUG
            std::cout << "trying to remove superstep " << step_to_remove << std::endl;
#endif

            if (scatter_nodes_remove_superstep(step_to_remove)) {

                for (unsigned proc = 0; proc < num_procs; proc++) {

                    if (step_to_remove < current_schedule->num_steps()) {
                        node_selection.insert(
                            current_schedule->set_schedule.step_processor_vertices[step_to_remove][proc].begin(),
                            current_schedule->set_schedule.step_processor_vertices[step_to_remove][proc].end());
                    }

                    if (step_to_remove > 0) {
                        node_selection.insert(
                            current_schedule->set_schedule.step_processor_vertices[step_to_remove - 1][proc].begin(),
                            current_schedule->set_schedule.step_processor_vertices[step_to_remove - 1][proc].end());
                    }
                }

                step_selection_counter = step_to_remove + 1;

                if (step_selection_counter >= current_schedule->num_steps()) {
                    step_selection_counter = 0;
                    step_selection_epoch_counter++;
                }

                parameters.violations_threshold = 0;
                super_locked_nodes.clear();
#ifdef KL_DEBUG
                std::cout << "---- reset super locked nodes" << std::endl;
#endif

                return;
            }
        }
    }

#ifdef KL_DEBUG
    std::cout << "no superstep to remove" << std::endl;
#endif

    step_selection_epoch_counter++;
    select_nodes();
    return;
}

bool kl_base::check_remove_superstep(unsigned step) {

    if (current_schedule->num_steps() <= 2) {
        return false;
    }

    unsigned total_work = 0;

    for (unsigned proc = 0; proc < num_procs; proc++) {

        total_work += current_schedule->step_processor_work[step][proc];
    }

    if (total_work < current_schedule->instance->synchronisationCosts()) {
        return true;
    }
    return false;
}

bool kl_base::scatter_nodes_remove_superstep(unsigned step) {

    assert(step < current_schedule->num_steps());

    std::vector<kl_move> moves;

    bool abort = false;

    for (unsigned proc = 0; proc < num_procs; proc++) {
        for (const auto &node : current_schedule->set_schedule.step_processor_vertices[step][proc]) {

            compute_node_gain(node);
            moves.push_back(best_move_change_superstep(node));

            if (moves.back().gain == std::numeric_limits<double>::lowest()) {
                abort = true;
                break;
            }

            if (current_schedule->use_memory_constraint) {

                if (current_schedule->instance->getArchitecture().getMemoryConstraintType() ==
                    PERSISTENT_AND_TRANSIENT) {

                    if (moves.back().to_proc != moves.back().from_proc) {
                        current_schedule->current_proc_persistent_memory[moves.back().to_proc] +=
                            current_schedule->instance->getComputationalDag().nodeMemoryWeight(node);
                        current_schedule->current_proc_persistent_memory[moves.back().from_proc] -=
                            current_schedule->instance->getComputationalDag().nodeMemoryWeight(node);

                        current_schedule->current_proc_transient_memory[moves.back().to_proc] =
                            std::max(current_schedule->current_proc_transient_memory[moves.back().to_proc],
                                     current_schedule->instance->getComputationalDag().nodeCommunicationWeight(node));
                        // TODO: implement this properly if PERSISTENT_AND_TRANSIENT becomes relevant
                    }

                } else if (current_schedule->instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
                    current_schedule->step_processor_work[moves.back().to_step][moves.back().to_proc] +=
                        current_schedule->instance->getComputationalDag().nodeWorkWeight(node);
                }
            }
        }

        if (abort) {
            break;
        }

        current_schedule->set_schedule.step_processor_vertices[step][proc].clear();
    }

    if (abort) {

        for (const auto &move : moves) {

            current_schedule->set_schedule.step_processor_vertices[step][move.from_proc].insert(move.node);

            if (current_schedule->use_memory_constraint) {

                if (current_schedule->instance->getArchitecture().getMemoryConstraintType() ==
                    PERSISTENT_AND_TRANSIENT) {

                    if (move.to_proc != move.from_proc) {
                        current_schedule->current_proc_persistent_memory[move.to_proc] -=
                            current_schedule->instance->getComputationalDag().nodeMemoryWeight(move.node);
                        current_schedule->current_proc_persistent_memory[move.from_proc] +=
                            current_schedule->instance->getComputationalDag().nodeMemoryWeight(move.node);
                    }

                } else if (current_schedule->instance->getArchitecture().getMemoryConstraintType() == LOCAL) {
                    current_schedule->step_processor_work[move.to_step][move.to_proc] -=
                        current_schedule->instance->getComputationalDag().nodeWorkWeight(move.node);
                }
            }
        }
        return false;
    }

    for (const auto &move : moves) {

#ifdef KL_DEBUG
        std::cout << "scatter node " << move.node << " to proc " << move.to_proc << " to step " << move.to_step
                  << std::endl;
#endif

        current_schedule->vector_schedule.setAssignedSuperstep(move.node, move.to_step);
        current_schedule->vector_schedule.setAssignedProcessor(move.node, move.to_proc);
        current_schedule->set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
    }

    current_schedule->remove_superstep(step);

    return true;
}

void kl_base::reset_run_datastructures() {
    node_selection.clear();
    nodes_to_update.clear();
    locked_nodes.clear();
    super_locked_nodes.clear();
}