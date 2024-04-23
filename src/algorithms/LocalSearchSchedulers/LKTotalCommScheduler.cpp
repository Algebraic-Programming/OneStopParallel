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

#include "algorithms/LocalSearchSchedulers/LKTotalCommScheduler.hpp"

bool LKTotalCommScheduler::start() {

    vector_schedule = VectorSchedule(*best_schedule);
    set_schedule = SetSchedule(*best_schedule);

    comm_multiplier = 1.0 / instance->numberOfProcessors();
    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
    double base_reward = base_reward_factor * penalty_factor;

    initalize_datastructures();
    compute_superstep_datastructures();

    best_schedule_costs = current_costs();

    double initial_costs = best_schedule_costs;
    double current_costs = initial_costs;

    // std::cout << "LKTotalComm start() best schedule costs: " << best_schedule_costs << std::endl;

    setParameters();

    std::unordered_set<VertexType> node_selection = selectNodesPermutationThreshold(selection_threshold);

    for (const auto &node : node_selection) {
        locked[node] = false;
    }

    initalize_gain_heap(node_selection);

    //  std::cout << "Initial costs " << current_costs() << std::endl;
    unsigned improvement_counter = 0;

    for (unsigned i = 0; i < max_iterations; i++) {

        unsigned failed_branches = 0;
        double best_iter_costs = current_costs;
        counter = 0;

        while (failed_branches < 3 && locked_nodes.size() < max_iterations_inner && max_gain_heap.size() > 0) {

            // std::cout << "failed branches: " << failed_branches << std::endl;

            Move best_move = findMove(); // O(log n)

            if (checkAbortCondition(best_move)) {
                //    std::cout << "abort condition met" << std::endl;
                break;
            }

            unsigned current_proc = vector_schedule.assignedProcessor(best_move.node);
            unsigned current_step = vector_schedule.assignedSuperstep(best_move.node);

            applyMove(best_move, current_proc, current_step); // O(p + log n)

            current_costs -= best_move.change_in_cost;

            // std::cout << "current costs: " << current_costs << " best move gain: " << best_move.gain
            //           << " best move costs: " << best_move.change_in_cost << std::endl;

            locked_nodes.insert(best_move.node);
            locked[best_move.node] = true;

            updateViolations(best_move.node); // O(Delta_max * log(current_violations.size())

            penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                             comm_multiplier * current_violations.size();

            reward_factor = base_reward + comm_multiplier * current_violations.size() * current_violations.size() * current_violations.size();

            std::unordered_set<VertexType> nodes_to_update =
                collectNodesToUpdate(best_move, current_proc, current_step);
            unlockNeighbours(best_move.node, nodes_to_update);
            updateNodesGain(nodes_to_update);

            if (best_move.change_in_cost < 0 && current_violations.empty() && current_feasible) {

                if (best_schedule_costs > current_costs + best_move.change_in_cost) {

                    //                    std::cout << "costs increased .. save best schedule with costs "
                    //                              << current_costs + best_move.change_in_cost << std::endl;

                    best_schedule_costs = current_costs + best_move.change_in_cost;
                    setBestSchedule(vector_schedule); // O(n)
                    reverseMoveBestSchedule(best_move, current_proc, current_step);
                }
            }

            if (current_violations.empty() && not current_feasible) {

                //                std::cout << "<=============== moved from infeasible to feasible" << std::endl;
                current_feasible = true;

                if (current_costs <= best_schedule_costs) {
                    //                    std::cout << "new schdule better than previous best schedule" << std::endl;

                    //                setBestSchedule(vector_schedule);
                    //                best_schedule_costs = current_costs;
                } else {

                    //                    std::cout << "... but costs did not improve: " << current_costs
                    //                              << " vs best schedule: " << best_schedule_costs << std::endl;

                    if (current_costs > (1.1 - counter * 0.001) * best_schedule_costs) {
                        //                        std::cout << "rollback to best schedule" << std::endl;
                        setCurrentSchedule(*best_schedule); // O(n + p*s)
                        compute_superstep_datastructures(); // O(n)
                        current_costs = best_schedule_costs;
                        current_violations.clear();
                        current_feasible = true;
                        resetGainHeap();
                        setup_gain_heap_unlocked_nodes();
                        counter = 0;
                        failed_branches++;
                    }
                }

            } else if (not current_violations.empty() && current_feasible) {
                //                std::cout << "================> moved from feasible to infeasible" << std::endl;
                current_feasible = false;

                // unlockNeighbours(best_move.node);

                if (current_costs + best_move.change_in_cost <= best_schedule_costs) {
                    //                    std::cout << "save best schedule with costs " << current_costs +
                    //                    best_move.change_in_cost
                    //                              << std::endl;
                    best_schedule_costs = current_costs + best_move.change_in_cost;
                    setBestSchedule(vector_schedule); // O(n)
                    reverseMoveBestSchedule(best_move, current_proc, current_step);
                }
            }

            if (not current_feasible) {

                if (current_costs > (1.2 - counter * 0.001) * best_schedule_costs) {

                    // std::cout << "current cost " << current_costs
                    //           << " too far away from best schedule costs: " << best_schedule_costs << "rollback to
                    //           best schedule" << std::endl;

                    setCurrentSchedule(*best_schedule); // O(n + p*s)
                    compute_superstep_datastructures(); // O(n)
                    current_costs = best_schedule_costs;
                    current_violations.clear();
                    current_feasible = true;
                    resetGainHeap();
                    setup_gain_heap_unlocked_nodes();
                    counter = 0;
                    failed_branches++;

                } //else {

                //    checkInsertSuperstep();
                //}
            }

            counter++;

        } // while

        // std::cout << "current costs end while: " << current_costs_double() << std::endl;
        // std::cout << "number of violations " << current_violations.size() << std::endl;
        if (current_violations.empty()) {

            if (current_costs <= best_schedule_costs) {
                setBestSchedule(vector_schedule);
                best_schedule_costs = current_costs;
            }

            resetLockedNodesAndComputeGains();

        } else {

            //            std::cout << "current solution not feasible .. rolling back to best solution with costs "
            //                      << best_schedule_costs << std::endl;

            resetLockedNodes();
            lockAll();

            std::unordered_set<VertexType> node_selection = selectNodesThreshold(selection_threshold);

            for (const auto &node : node_selection) {
                locked[node] = false;
            }

            setCurrentSchedule(*best_schedule); // O (n + p*s)

            compute_superstep_datastructures(); // O(n)
            current_costs = best_schedule_costs;
            current_violations.clear();
            current_feasible = true;

            resetGainHeap();
            initalize_gain_heap(node_selection);
        }

        if (best_iter_costs <= current_costs) {

            if (improvement_counter++ == std::sqrt(max_iterations)) {
                // std::cout << "no improvement ... end local search " << std::endl;
                break;
            }
        } else {
            improvement_counter = 0;
        }

    } // for

    // std::cout << "LKTotalComm end best schedule costs: " << best_schedule_costs << std::endl;
    cleanup_datastructures();

    if (initial_costs > current_costs)
        return true;
    else
        return false;
}

void LKTotalCommScheduler::insertSuperstep(unsigned step_before) {

    set_schedule.insertSupersteps(step_before, 1);
    vector_schedule.insertSupersteps(step_before, 1);

    step_processor_work.push_back(std::move(step_processor_work[num_steps - 1]));
    step_max_work.push_back(step_max_work[num_steps - 1]);
    step_second_max_work.push_back(step_second_max_work[num_steps - 1]);

    for (unsigned step = num_steps - 2; step > step_before; step--) {

        step_max_work[step + 1] = step_max_work[step];

        step_second_max_work[step + 1] = step_second_max_work[step];

        step_processor_work[step + 1] = std::move(step_processor_work[step]);
    }

    num_steps += 1;
 
    step_max_work[step_before + 1] = 0;
    step_second_max_work[step_before + 1] = 0;
    step_processor_work[step_before + 1] = std::vector<double>(num_procs, 0);

    unsigned last_step = std::min(num_steps - 1, step_before + 2);

    for (unsigned i = step_before; i <= last_step; i++) {

        if (i != step_before + 1) {
            for (unsigned proc = 0; proc < num_procs; proc++) {

                for (const auto &node : set_schedule.step_processor_vertices[i][proc]) {

                    if (locked[node]) {

                        if (unlockNode(node)) {
                            computeNodeGain(node);
                            computeMaxGain(node);
                        }

                    } else {
                        computeNodeGain(node);
                        computeMaxGain(node);
                    }
                }
            }
        }
    }
}

void LKTotalCommScheduler::update_superstep_datastructures(Move move, unsigned from_proc, unsigned from_step) {

    step_processor_work[move.to_step][move.to_proc] += instance->getComputationalDag().nodeWorkWeight(move.node);

    if (from_step == move.to_step) {
        step_processor_work[from_step][from_proc] -= instance->getComputationalDag().nodeWorkWeight(move.node);
        recompute_superstep_max_work(from_step);
    } else {

        if (step_max_work[move.to_step] < step_processor_work[move.to_step][move.to_proc]) {
            step_second_max_work[move.to_step] = step_max_work[move.to_step];
            step_max_work[move.to_step] = step_processor_work[move.to_step][move.to_proc];

        } else if (step_second_max_work[move.to_step] < step_processor_work[move.to_step][move.to_proc]) {
            step_second_max_work[move.to_step] = step_processor_work[move.to_step][move.to_proc];
        }

        if (step_max_work[from_step] == step_processor_work[from_step][from_proc] &&
            step_max_work[from_step] > step_second_max_work[from_step]) {

            step_processor_work[from_step][from_proc] -= instance->getComputationalDag().nodeWorkWeight(move.node);

            if (step_processor_work[from_step][from_proc] >= step_second_max_work[from_step]) {
                step_max_work[from_step] = step_processor_work[from_step][from_proc];

            } else {
                recompute_superstep_max_work(from_step);
            }
        } else if (step_processor_work[from_step][from_proc] == step_second_max_work[from_step]) {
            step_processor_work[from_step][from_proc] -= instance->getComputationalDag().nodeWorkWeight(move.node);

            recompute_superstep_max_work(from_step);
        }
    }
}

void LKTotalCommScheduler::commputeCommGain(unsigned node, unsigned current_step, unsigned current_proc,
                                            unsigned new_proc) {

    if (current_proc == new_proc) {

        for (const auto &target : instance->getComputationalDag().children(node)) {

            if (current_step == vector_schedule.assignedSuperstep(target)) {
                node_gains[node][current_proc][2] -=
                    (double)instance->getComputationalDag().nodeCommunicationWeight(node) * penalty_factor;

            } else if (current_step > vector_schedule.assignedSuperstep(target)) {

                node_gains[node][current_proc][0] +=
                    (double)instance->getComputationalDag().nodeCommunicationWeight(node) * reward_factor;
            }
        }

        for (const auto &source : instance->getComputationalDag().parents(node)) {

            if (current_step == vector_schedule.assignedSuperstep(source)) {
                node_gains[node][current_proc][0] -=
                    (double)instance->getComputationalDag().nodeCommunicationWeight(node) * penalty_factor;

            } else if (current_step < vector_schedule.assignedSuperstep(source)) {

                node_gains[node][current_proc][2] +=
                    (double)instance->getComputationalDag().nodeCommunicationWeight(node) * reward_factor;
            }
        }
    } else {

        // current_proc != new_proc

        for (const auto &target : instance->getComputationalDag().children(node)) {

            const unsigned &target_proc = vector_schedule.assignedProcessor(target);
            if (target_proc == current_proc) {

                const double loss = (double)instance->getComputationalDag().nodeCommunicationWeight(node) *
                                    instance->communicationCosts(new_proc, target_proc) * comm_multiplier;

                node_gains[node][new_proc][0] -= loss;
                node_gains[node][new_proc][1] -= loss;
                node_gains[node][new_proc][2] -= loss;

                node_change_in_costs[node][new_proc][0] -= loss;
                node_change_in_costs[node][new_proc][1] -= loss;
                node_change_in_costs[node][new_proc][2] -= loss;

                if (vector_schedule.assignedSuperstep(target) == current_step) {

                    node_gains[node][new_proc][1] -=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(node) * penalty_factor;
                    node_gains[node][new_proc][2] -=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(node) * penalty_factor;

                } else if (vector_schedule.assignedSuperstep(target) == current_step + 1) {

                    node_gains[node][new_proc][2] -=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(node) * penalty_factor;

                } else if (vector_schedule.assignedSuperstep(target) < current_step) {
                    // not resolved but step towrads it ...
                    node_gains[node][new_proc][0] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(node) * reward_factor * 0.5;
                }

            } else if (target_proc == new_proc) {

                const double gain = (double)instance->getComputationalDag().nodeCommunicationWeight(node) *
                                    instance->communicationCosts(current_proc, target_proc) * comm_multiplier;

                node_gains[node][new_proc][0] += gain;
                node_gains[node][new_proc][1] += gain;
                node_gains[node][new_proc][2] += gain;

                node_change_in_costs[node][new_proc][0] += gain;
                node_change_in_costs[node][new_proc][1] += gain;
                node_change_in_costs[node][new_proc][2] += gain;

                if (vector_schedule.assignedSuperstep(target) == current_step) {

                    node_gains[node][new_proc][1] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(node) * reward_factor;

                } else if (vector_schedule.assignedSuperstep(target) < current_step) {

                    node_gains[node][new_proc][0] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(node) * reward_factor;
                }

            } else {

                assert(target_proc != current_proc && target_proc != new_proc);

                const double gain = (double)(instance->communicationCosts(new_proc, target_proc) -
                                             instance->communicationCosts(current_proc, target_proc)) *
                                    instance->getComputationalDag().nodeCommunicationWeight(node) * comm_multiplier;

                node_gains[node][new_proc][0] += gain;
                node_gains[node][new_proc][1] += gain;
                node_gains[node][new_proc][2] += gain;

                node_change_in_costs[node][new_proc][0] += gain;
                node_change_in_costs[node][new_proc][1] += gain;
                node_change_in_costs[node][new_proc][2] += gain;

                if (vector_schedule.assignedSuperstep(target) == current_step + 1) {

                    node_gains[node][new_proc][2] -=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(node) * penalty_factor;
                } else if (vector_schedule.assignedSuperstep(target) == current_step) {

                    node_gains[node][new_proc][0] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(node) * reward_factor;
                } else if (vector_schedule.assignedSuperstep(target) < current_step) {

                    node_gains[node][new_proc][0] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(node) * reward_factor * 0.5;
                }
            }
        }

        for (const auto &source : instance->getComputationalDag().parents(node)) {

            const unsigned &source_proc = vector_schedule.assignedProcessor(source);
            if (source_proc == current_proc) {

                const double loss = (double)instance->getComputationalDag().nodeCommunicationWeight(source) *
                                    instance->communicationCosts(current_proc, new_proc) * comm_multiplier;

                node_gains[node][new_proc][0] -= loss;
                node_gains[node][new_proc][1] -= loss;
                node_gains[node][new_proc][2] -= loss;

                node_change_in_costs[node][new_proc][0] -= loss;
                node_change_in_costs[node][new_proc][1] -= loss;
                node_change_in_costs[node][new_proc][2] -= loss;

                if (vector_schedule.assignedSuperstep(source) == current_step) {

                    node_gains[node][new_proc][0] -=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * penalty_factor;
                    node_gains[node][new_proc][1] -=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * penalty_factor;

                } else if (vector_schedule.assignedSuperstep(source) == current_step - 1) {

                    node_gains[node][new_proc][0] -=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * penalty_factor;

                } else if (vector_schedule.assignedSuperstep(source) > current_step) {

                    node_gains[node][new_proc][0] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * reward_factor * 0.5;
                }

            } else if (source_proc == new_proc) {

                assert(source_proc != current_proc);
                const double gain = (double)instance->getComputationalDag().nodeCommunicationWeight(source) *
                                    instance->communicationCosts(current_proc, new_proc) * comm_multiplier;

                node_gains[node][new_proc][0] += gain;
                node_gains[node][new_proc][1] += gain;
                node_gains[node][new_proc][2] += gain;

                node_change_in_costs[node][new_proc][0] += gain;
                node_change_in_costs[node][new_proc][1] += gain;
                node_change_in_costs[node][new_proc][2] += gain;

                if (vector_schedule.assignedSuperstep(source) == current_step) {

                    node_gains[node][new_proc][1] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * reward_factor;

                    node_gains[node][new_proc][2] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * reward_factor;

                } else if (vector_schedule.assignedSuperstep(source) == current_step + 1) {

                    node_gains[node][new_proc][2] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * reward_factor;
                } else if (vector_schedule.assignedSuperstep(source) > current_step + 1) {

                    node_gains[node][new_proc][2] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * reward_factor * 0.5;
                }

            } else {

                assert(source_proc != current_proc && source_proc != new_proc);
                const double gain = (double)(instance->communicationCosts(new_proc, source_proc) -
                                             instance->communicationCosts(current_proc, source_proc)) *
                                    instance->getComputationalDag().nodeCommunicationWeight(source) * comm_multiplier;

                node_gains[node][new_proc][0] += gain;
                node_gains[node][new_proc][1] += gain;
                node_gains[node][new_proc][2] += gain;

                node_change_in_costs[node][new_proc][0] += gain;
                node_change_in_costs[node][new_proc][1] += gain;
                node_change_in_costs[node][new_proc][2] += gain;

                if (vector_schedule.assignedSuperstep(source) == current_step - 1) {

                    node_gains[node][new_proc][0] -=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * penalty_factor;

                } else if (vector_schedule.assignedSuperstep(source) == current_step) {

                    node_gains[node][new_proc][2] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * reward_factor;
                } else if (vector_schedule.assignedSuperstep(source) > current_step) {

                    node_gains[node][new_proc][2] +=
                        (double)instance->getComputationalDag().nodeCommunicationWeight(source) * reward_factor * 0.5;
                }
            }
        }
    }
}

double LKTotalCommScheduler::current_costs() {

    double work_costs = 0;
    for (unsigned step = 0; step < num_steps; step++) {
        work_costs += step_max_work[step];
    }

    double comm_costs = 0;
    for (const auto &edge : instance->getComputationalDag().edges()) {
        const unsigned &source = instance->getComputationalDag().source(edge);

        const unsigned &source_proc = vector_schedule.assignedProcessor(source);
        const unsigned &target_proc = vector_schedule.assignedProcessor(instance->getComputationalDag().target(edge));

        if (source_proc != target_proc) {
            comm_costs += instance->getComputationalDag().nodeCommunicationWeight(source) *
                          instance->communicationCosts(source_proc, target_proc) * comm_multiplier;
        }
    }

    return work_costs + comm_costs + (num_steps - 1) * instance->synchronisationCosts();
}

void LKTotalCommScheduler::compute_superstep_datastructures() {

    for (unsigned step = 0; step < num_steps; step++) {

        step_max_work[step] = 0;
        step_second_max_work[step] = 0;

        for (unsigned proc = 0; proc < num_procs; proc++) {

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

void LKTotalCommScheduler::cleanup_superstep_datastructures() {

    step_processor_work.clear();
    step_max_work.clear();
    step_second_max_work.clear();
}

void LKTotalCommScheduler::initalize_superstep_datastructures() {

    step_processor_work = std::vector<std::vector<double>>(num_steps, std::vector<double>(num_procs, 0));
    step_max_work = std::vector<double>(num_steps, 0);
    step_second_max_work = std::vector<double>(num_steps, 0);
}

void LKTotalCommScheduler::initializeRewardPenaltyFactors() {

    comm_multiplier = 1.0 / instance->numberOfProcessors();
    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
    base_reward = base_reward_factor * penalty_factor;
}

void LKTotalCommScheduler::updateRewardPenaltyFactors() {

    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                     comm_multiplier * std::sqrt(current_violations.size());

    reward_factor = base_reward + comm_multiplier * current_violations.size();
}
