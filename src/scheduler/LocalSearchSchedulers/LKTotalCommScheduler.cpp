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

#include "scheduler/LocalSearchSchedulers/LKTotalCommScheduler.hpp"

bool LKTotalCommScheduler::start() {

    vector_schedule = VectorSchedule(*best_schedule);
    set_schedule = SetSchedule(*best_schedule);

    comm_multiplier = 1.0 / instance->numberOfProcessors();
    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
    base_reward = penalty_factor + 5.0;

    auto start_time = std::chrono::high_resolution_clock::now();

    initalize_datastructures();
    compute_superstep_datastructures();

    best_schedule_costs = compute_current_costs();

    double initial_costs = best_schedule_costs;
    current_cost = initial_costs;

    setParameters();

    select_nodes();

    initalize_gain_heap(node_selection);

    // std::cout << "Initial costs " << current_cost << std::endl;
    unsigned improvement_counter = 0;
    counter = 0;

    for (unsigned i = 0; i < max_iterations; i++) {

        unsigned failed_branches = 0;
        double best_iter_costs = current_cost;

        unsigned inner_counter = 0;
        unsigned barrier = max_inner_iterations;
        if (current_violations.size() > 0) {
            barrier *= 2;
        }

        bool initial_feasile = current_violations.size() == 0;

        while (failed_branches < 3 && inner_counter < barrier && max_gain_heap.size() > 0) {

            inner_counter++;

            double iter_costs = current_cost;

            if (current_feasible) {
                initial_depth = 3;

            } else {
                initial_depth = 1;
            }

            Move best_move = findMove(); // O(log n)

            applyMove(best_move); // O(p + log n)

            locked_nodes.insert(best_move.node);

            std::unordered_map<VertexType, EdgeType> nodes_to_move;

            updateViolations(best_move.node, nodes_to_move); // O(Delta_max * log(current_violations.size())

#ifdef LK_DEBUG
            std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                      << best_move.change_in_cost << " from step " << best_move.from_step << " to " << best_move.to_step
                      << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                      << " violations: " << current_violations.size() << " cost " << current_cost << " computed costs "
                      << compute_current_costs() << std::endl;
#endif
            updateRewardPenaltyFactors();

            std::unordered_set<VertexType> nodes_to_update;

            std::vector<Move> moves;
            moves.push_back(best_move);

            if (nodes_to_move.size() == 0) {

                collectNodesToUpdate(best_move, nodes_to_update);

                std::unordered_set<VertexType> unlocked_nodes;
                unlockNeighbours(best_move, unlocked_nodes);

                for (const auto &node : unlocked_nodes) {
                    nodes_to_update.insert(node);
                    node_selection.insert(node);
                }

            } else {

                int depth = initial_depth;

                moves.reserve(nodes_to_move.size() * depth);

                std::unordered_map<VertexType, EdgeType> next_nodes_to_move;

                bool abort = false;

                while (depth > 0) {

                    for (const auto &[node, edge] : nodes_to_move) {

                        computeNodeGain(node);
                        Move node_move = compute_best_move(node);

                        if (node_move.gain < base_reward_factor) {
                            next_nodes_to_move[node] = edge;
                            continue;
                        }

                        if (node_heap_handles.find(node) != node_heap_handles.end()) {
                            max_gain_heap.erase(node_heap_handles[node]);
                            node_heap_handles.erase(node);
                        }

                        applyMove(node_move); // O(p + log n)

                        locked_nodes.insert(node);

                        moves.push_back(node_move);

                        std::unordered_set<EdgeType, EdgeType_hash> resolved;
                        updateViolations(node, next_nodes_to_move, &resolved);

                        collectNodesToUpdate(node_move, nodes_to_update);

#ifdef LK_DEBUG
                        std::cout << "node " << node << " gain " << node_move.gain << " chng in cost "
                                  << node_move.change_in_cost << " from step " << node_move.from_step << " to "
                                  << node_move.to_step << ", from proc " << node_move.from_proc << " to "
                                  << node_move.to_proc << " violations: " << current_violations.size() << std::endl;

#endif

                        std::unordered_set<VertexType> unlocked_nodes;
                        unlockNeighbours(node_move, unlocked_nodes);

                        for (const auto &node_neighbor : unlocked_nodes) {
                            nodes_to_update.insert(node_neighbor);
                            node_selection.insert(node_neighbor);
                        }

                        updateRewardPenaltyFactors();
                        if (current_violations.size() == 0) {
                            depth = 0;
                            break;
                        }

                        if (resolved.find(edge) != resolved.end()) {

                            if (unlockNode(node)) {

                                next_nodes_to_move[node] = edge;

                            } else {

                                abort = true;
                                depth = 0;
                                break;
                            }
                        }
                    }

                    depth--;

                    nodes_to_move = next_nodes_to_move;
                    next_nodes_to_move.clear();
                }

                if (abort) {

                    std::unordered_map<VertexType, EdgeType> new_violations;

#ifdef LK_DEBUG
                    std::cout << "abort depth search, rolling back to start schedule" << std::endl;
#endif

                    for (auto it = moves.rbegin(); it != moves.rend(); ++it) {
                        reverseMove(*it);
                        updateViolations((*it).node, new_violations);
                    }
                }
            }

            bool abort = false;
            if (not current_violations.empty()) {

                for (auto &edge : current_violations) {

                    const auto &source = instance->getComputationalDag().source(edge);
                    const auto &target = instance->getComputationalDag().target(edge);

                    if (locked_nodes.find(source) != locked_nodes.end() && not unlockNode(source)) {
                        abort = true;
                    }

                    if (abort && locked_nodes.find(target) != locked_nodes.end() && not unlockNode(target)) {
                        // abort = true;
                        break;
                    } else {
                        abort = false;
                    }

                    if (locked_nodes.find(source) == locked_nodes.end()) {

                        nodes_to_update.insert(source);
                        node_selection.insert(source);
                    }

                    if (locked_nodes.find(target) == locked_nodes.end()) {
                        nodes_to_update.insert(target);
                        node_selection.insert(target);
                    }
                }
            }

            // if (current_violations.size() > 40) {

            //     for (auto &edge : current_violations) {

            //         const auto &source = instance->getComputationalDag().source(edge);
            //         const auto &target = instance->getComputationalDag().target(edge);

            //         computeNodeGain(source);
            //         computeNodeGain(target);
            //         Move source_move = compute_best_move(source);
            //         Move target_move = compute_best_move(target);

            //         if ( source_move.gain > target_move.gain) {

            //         if (node_heap_handles.find(source) != node_heap_handles.end()) {
            //             max_gain_heap.erase(node_heap_handles[source]);
            //             node_heap_handles.erase(source);
            //         }

            //         } else {

            //         if (node_heap_handles.find(target) != node_heap_handles.end()) {
            //             max_gain_heap.erase(node_heap_handles[target]);
            //             node_heap_handles.erase(target);
            //         }

            //         }

            //     }
            // }

            if (nodes_to_update.find(best_move.node) != nodes_to_update.end()) {
                nodes_to_update.erase(best_move.node);
            }

            updateNodesGain(nodes_to_update);

            if (iter_costs < current_cost && current_violations.empty() && current_feasible && initial_feasile) {

                if (best_schedule_costs > iter_costs) {

#ifdef LK_DEBUG
                    std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                    best_schedule_costs = iter_costs;
                    setBestSchedule(vector_schedule); // O(n)

                    for (auto it = moves.rbegin(); it != moves.rend(); ++it) {
                        reverseMoveBestSchedule(*it);
                    }
                }
            }

            if (current_cost < best_schedule_costs && current_violations.empty()) {
                // std::cout << "new schdule cost " << current_cost << " lower than best schedule costs "
                //           << best_schedule_costs << std::endl;
            }

            if (current_violations.empty() && not current_feasible) {

#ifdef LK_DEBUG
                std::cout << "<=============== moved from infeasible to feasible" << std::endl;
#endif
                current_feasible = true;

                if (current_cost <= best_schedule_costs) {
#ifdef LK_DEBUG
                    std::cout << "new schdule better than previous best schedule" << std::endl;
#endif
                    setBestSchedule(vector_schedule);
                    best_schedule_costs = current_cost;
                } else {

#ifdef LK_DEBUG
                    std::cout << "... but costs did not improve: " << current_cost
                              << " vs best schedule: " << best_schedule_costs << std::endl;
#endif

                    if (current_cost > (1.02 + counter * 0.002) * best_schedule_costs) {

#ifdef LK_DEBUG
                        std::cout << " rollback to best schedule" << std::endl;
#endif
                        setCurrentSchedule(*best_schedule); // O(n + p*s)
                        compute_superstep_datastructures(); // O(n)

                        penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
                        reward_factor = 1.0;

                        current_cost = best_schedule_costs;
                        current_violations.clear();
                        current_feasible = true;
                        resetGainHeap();
                        setup_gain_heap_unlocked_nodes();

                        failed_branches++;
                    }
                }

            } else if (not current_violations.empty() && current_feasible) {
#ifdef LK_DEBUG
                std::cout << "================> moved from feasible to infeasible" << std::endl;
#endif
                current_feasible = false;

                if (iter_costs <= best_schedule_costs) {
#ifdef LK_DEBUG
                    std::cout << "save best schedule with costs " << iter_costs << std::endl;
#endif

                    best_schedule_costs = iter_costs;
                    setBestSchedule(vector_schedule); // O(n)
                    for (auto it = moves.rbegin(); it != moves.rend(); ++it) {
                        reverseMoveBestSchedule(*it);
                    }
                }
            }

            if (abort) {
#ifdef LK_DEBUG
                std::cout << "abort condition met" << std::endl;
#endif
                break;
            }

            if (not current_feasible) {

                if (current_cost > (1.02 + counter * 0.002) * best_schedule_costs) {

#ifdef LK_DEBUG
                    std::cout << "current cost " << current_cost
                              << " too far away from best schedule costs: " << best_schedule_costs
                              << " rollback to best schedule" << std::endl;
#endif

                    setCurrentSchedule(*best_schedule); // O(n + p*s)
                    compute_superstep_datastructures(); // O(n)

                    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
                    reward_factor = 1.0;

                    current_cost = best_schedule_costs;
                    current_violations.clear();
                    current_feasible = true;
                    resetGainHeap();
                    setup_gain_heap_unlocked_nodes();
                    // counter = 0;
                    failed_branches++;
                }
            }

        } // while

        counter++;
#ifdef LK_DEBUG
        std::cout << "end inner loop current cost: " << current_cost << " with " << current_violations.size()
                  << " violation, best sol cost: " << best_schedule_costs << " with "
                  << best_schedule->numberOfSupersteps() << " supersteps, counter: " << counter << "/" << max_iterations
                  << std::endl;
#endif
        // if (step_selection_counter >= num_steps && counter % std::max(num_steps, 100u) == 0) {
        if (step_selection_counter >= num_steps && counter % max_iterations / 3 == 0) {
            step_selection_counter = 1;
        }

        if (current_violations.empty()) {

            if (current_cost <= best_schedule_costs) {
                setBestSchedule(vector_schedule);
                best_schedule_costs = current_cost;
            }

            select_nodes();

            resetLockedNodes();

            resetGainHeap();
            initalize_gain_heap(node_selection);

        } else {

            // std::cout << "current solution not feasible .. rolling back to best solution with costs "
            //           << best_schedule_costs << std::endl;

            setCurrentSchedule(*best_schedule); // O (n + p*s)

            compute_superstep_datastructures(); // O(n)

            current_cost = best_schedule_costs;
            current_violations.clear();
            current_feasible = true;

            select_nodes();

            resetLockedNodes();
            resetGainHeap();
            initalize_gain_heap(node_selection);
        }

        if (compute_with_time_limit) {

            auto finish_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

            if (duration > timeLimitSeconds) {
                break;
            }
        }

        if (best_iter_costs <= current_cost) {

            if (improvement_counter++ == std::log(max_iterations)) {
                // std::cout << "no improvement ... end local search " << std::endl;
                break;
            }
        } else {
            improvement_counter = 0;
        }

    } // for

    // std::cout << "LKTotalComm end best schedule costs: " << best_schedule_costs
    //           << " computed costs: " << compute_current_costs() << std::endl;
    cleanup_datastructures();

    if (initial_costs > current_cost)
        return true;
    else
        return false;
}

void LKTotalCommScheduler::insertSuperstep(unsigned step_before) {
    /*
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
    */
}

void LKTotalCommScheduler::concurrent_updateNodesGain(const std::vector<VertexType> &nodes) {

#pragma omp parallel for
    for (auto node : nodes) {
        computeNodeGain(node);
        computeMaxGain(node);
    }
};

void LKTotalCommScheduler::select_nodes() {

    node_selection = selectNodesFindRemoveSteps(selection_threshold);

    if (current_violations.size() > 0) {

        updateRewardPenaltyFactors();

        current_cost = compute_current_costs();
        current_feasible = false;

    } else {

        penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
        reward_factor = 1.0;
    }
}

void LKTotalCommScheduler::update_superstep_datastructures(Move move) {

    step_processor_work[move.to_step][move.to_proc] += instance->getComputationalDag().nodeWorkWeight(move.node);

    if (use_memory_constraint) {
        step_processor_memory[move.to_step][move.to_proc] +=
            instance->getComputationalDag().nodeMemoryWeight(move.node);
        step_processor_memory[move.from_step][move.from_proc] -=
            instance->getComputationalDag().nodeMemoryWeight(move.node);
    }

    if (move.from_step == move.to_step) {
        step_processor_work[move.from_step][move.from_proc] -=
            instance->getComputationalDag().nodeWorkWeight(move.node);
        recompute_superstep_max_work(move.from_step);
    } else {

        if (step_max_work[move.to_step] < step_processor_work[move.to_step][move.to_proc]) {
            step_second_max_work[move.to_step] = step_max_work[move.to_step];
            step_max_work[move.to_step] = step_processor_work[move.to_step][move.to_proc];

        } else if (step_second_max_work[move.to_step] < step_processor_work[move.to_step][move.to_proc]) {
            step_second_max_work[move.to_step] = step_processor_work[move.to_step][move.to_proc];
        }

        if (step_max_work[move.from_step] == step_processor_work[move.from_step][move.from_proc] &&
            step_max_work[move.from_step] > step_second_max_work[move.from_step]) {

            step_processor_work[move.from_step][move.from_proc] -=
                instance->getComputationalDag().nodeWorkWeight(move.node);

            if (step_processor_work[move.from_step][move.from_proc] >= step_second_max_work[move.from_step]) {
                step_max_work[move.from_step] = step_processor_work[move.from_step][move.from_proc];

            } else {
                recompute_superstep_max_work(move.from_step);
            }
        } else if (step_processor_work[move.from_step][move.from_proc] == step_second_max_work[move.from_step]) {
            step_processor_work[move.from_step][move.from_proc] -=
                instance->getComputationalDag().nodeWorkWeight(move.node);

            recompute_superstep_max_work(move.from_step);
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

                } // else if (vector_schedule.assignedSuperstep(target) < current_step) {
                  //  not resolved but step towrads it ...
                //     node_gains[node][new_proc][0] +=
                //         (double)instance->getComputationalDag().nodeCommunicationWeight(node) * reward_factor *
                //         0.5;
                // }

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
                } // else if (vector_schedule.assignedSuperstep(target) < current_step) {

                /////         node_gains[node][new_proc][0] +=
                //              (double)instance->getComputationalDag().nodeCommunicationWeight(node) *
                //              reward_factor * 0.5;
                //      }
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

                } // else if (vector_schedule.assignedSuperstep(source) > current_step) {

                //           node_gains[node][new_proc][0] +=
                //               (double)instance->getComputationalDag().nodeCommunicationWeight(source) *
                //               reward_factor
                //               * 0.5;
                //       }

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
                } // else if (vector_schedule.assignedSuperstep(source) > current_step + 1) {

                //      node_gains[node][new_proc][2] +=
                //           (double)instance->getComputationalDag().nodeCommunicationWeight(source) * reward_factor
                //           * 0.5;
                //   }

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
                } // else if (vector_schedule.assignedSuperstep(source) > current_step) {

                //   node_gains[node][new_proc][2] +=
                //       (double)instance->getComputationalDag().nodeCommunicationWeight(source) * reward_factor *
                //       0.5;
                // }
            }
        }
    }
}

double LKTotalCommScheduler::compute_current_costs() {

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

/* needed */
void LKTotalCommScheduler::compute_superstep_datastructures() {

    for (unsigned step = 0; step < num_steps; step++) {

        step_max_work[step] = 0;
        step_second_max_work[step] = 0;

        for (unsigned proc = 0; proc < num_procs; proc++) {

            step_processor_work[step][proc] = 0;

            for (const auto &node : set_schedule.step_processor_vertices[step][proc]) {
                step_processor_work[step][proc] += instance->getComputationalDag().nodeWorkWeight(node);

                if (use_memory_constraint) {
                    step_processor_memory[step][proc] += instance->getComputationalDag().nodeMemoryWeight(node);
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
}

void LKTotalCommScheduler::cleanup_superstep_datastructures() {

    step_processor_work.clear();
    step_max_work.clear();
    step_second_max_work.clear();

    step_processor_memory.clear();
}

void LKTotalCommScheduler::initalize_superstep_datastructures() {

    step_processor_work = std::vector<std::vector<double>>(num_steps, std::vector<double>(num_procs, 0));
    step_max_work = std::vector<double>(num_steps, 0);
    step_second_max_work = std::vector<double>(num_steps, 0);

    if (use_memory_constraint) {
        step_processor_memory = std::vector<std::vector<unsigned>>(num_steps, std::vector<unsigned>(num_procs, 0));
    }
}

void LKTotalCommScheduler::initializeRewardPenaltyFactors() {

    comm_multiplier = 1.0 / instance->numberOfProcessors();
    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
    base_reward = base_reward_factor * penalty_factor;
}

void LKTotalCommScheduler::updateRewardPenaltyFactors() {

    //base_penalty_factor = 10.0 + 10.0 * (max_iterations - counter) / max_iterations;

    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                     (current_violations.size() + base_penalty_factor) * current_violations.size();

    reward_factor = base_reward + (current_violations.size() + 2 + base_penalty_factor) * current_violations.size() *
                                      (current_violations.size() + 10);
}

/*
bool LKTotalCommScheduler::start() {


    vector_schedule = VectorSchedule(*best_schedule);
    set_schedule = SetSchedule(*best_schedule);

    comm_multiplier = 1.0 / instance->numberOfProcessors();
    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
    double base_reward = penalty_factor + 5.0;

    // std::uniform_int_distribution<> coin(0, 2);
    auto start_time = std::chrono::high_resolution_clock::now();

    initalize_datastructures();
    compute_superstep_datastructures();

    best_schedule_costs = current_costs();

    double initial_costs = best_schedule_costs;
    current_cost = initial_costs;

    std::cout << "LKTotalComm start() best schedule costs: " << best_schedule_costs << std::endl;

    setParameters();

    std::unordered_set<VertexType> node_selection = selectNodesFindRemoveSteps(selection_threshold);
    //    selectNodesConseqStepsReduceNumSteps(selection_threshold); //
    // selectNodesConseqStepsMaxWork(selection_threshold); // selectNodesPermutationThreshold(selection_threshold);

    for (const auto &node : node_selection) {
        locked[node] = false;
    }

    initalize_gain_heap(node_selection);

    std::cout << "Initial costs " << current_cost << std::endl;
    unsigned improvement_counter = 0;
    counter = 0;

    VertexType node_causing_first_violation = 0;

    for (unsigned i = 0; i < max_iterations; i++) {

        unsigned failed_branches = 0;
        double best_iter_costs = current_cost;

        unsigned inner_counter = 0;
        unsigned barrier = 1500;
        if (current_violations.size() > 0) {
            barrier = 3000;
        }

        while (failed_branches < 3 && inner_counter < barrier && max_gain_heap.size() > 0) {

            inner_counter++;

            Move best_move = findMove(); // O(log n)

            //             if (best_move.gain < 0) {

            //                     std::cout << "abort condition met" << std::endl;
            // std::cout << "current costs: " << current_cost << " best move gain: " << best_move.gain
            //                        << " best move costs: " << best_move.change_in_cost << std::endl;

            //                //break;
            //             }

            applyMove(best_move); // O(p + log n)

            //  std::cout << inner_counter << " best move node: " << best_move.node << " current costs: " <<
            //  current_cost
            //            << " best move gain: " << best_move.gain << " best move costs: " << best_move.change_in_cost
            //           << " number of violations: " << current_violations.size() << std::endl;

            locked_nodes.insert(best_move.node);
            locked[best_move.node] = true;

      //      std::unordered_map<VertexType, Ed> nodes_to_move;

            updateViolations(best_move.node, nodes_to_move); // O(Delta_max * log(current_violations.size())

            penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                             (current_violations.size() + base_penalty_factor) * current_violations.size();

            reward_factor = base_reward + (current_violations.size() + 2 + base_penalty_factor) *
                                              current_violations.size() * current_violations.size();

            std::unordered_set<VertexType> nodes_to_update;

            if (current_violations.size() == 0) {

                nodes_to_update = collectNodesToUpdate(best_move);

                unlockNeighbours(best_move, nodes_to_update);

            } else {

                std::vector<Move> moves;
                moves.reserve(4 * current_violations.size());
                moves.push_back(best_move);

                unsigned depth = 1;

                std::unordered_set<EdgeType, EdgeType_hash> old_violations(current_violations);

                std::deque<VertexType> next_nodes_to_move;

                nodes_to_update = node_selection;

                bool abort = false;

                while (depth > 0) {

                    for (const auto &node : nodes_to_move) {

                        computeNodeGain(node);
                        Move node_move = compute_best_move(node);

                        applyMove(node_move); // O(p + log n)

                        moves.push_back(node_move);

                        locked_nodes.insert(node);
                        locked[node] = true;

                        updateViolations(best_move.node, next_nodes_to_move);

                        penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                                         (current_violations.size() + base_penalty_factor) * current_violations.size();

                        reward_factor = base_reward + (current_violations.size() + 2 + base_penalty_factor) *
                                                          current_violations.size() * current_violations.size();
                    }

                    depth--;
                    nodes_to_move = next_nodes_to_move;
                    next_nodes_to_move.clear();

                    for (const auto &edge : old_violations) {

                        if (current_violations.find(edge) != current_violations.end()) {

                            const auto &source = instance->getComputationalDag().source(edge);
                            const auto &target = instance->getComputationalDag().target(edge);

                            if (unlockNode(source)) {
                                nodes_to_update.insert(source);
                            } else {
                                super_locked[source] = true;
                                locked[source] = true;
                                abort = true;
                            }

                            if (unlockNode(target)) {
                                nodes_to_update.insert(target);
                            } else {
                                super_locked[target] = true;
                                locked[target] = true;
                                abort = true;
                            }
                        }
                    }
                    old_violations.clear();
                    old_violations = current_violations;
                }

                if (abort) {
                    // std::cout << "abort condition met" << std::endl;
                    break;
                }

                // for (const auto &node : node_selection) {
                //     if (locked[node] == false) {
                //         nodes_to_update.insert(node);
                //     }
                // }

                // bool abort = false;
                // for (auto &edge : current_violations) {

                //     const auto &source = instance->getComputationalDag().source(edge);
                //     const auto &target = instance->getComputationalDag().target(edge);

                //     if (unlock[source] == 0 && unlock[target] == 0) {

                //         super_locked[source] = true;
                //         super_locked[target] = true;
                //         super_locked[node_causing_first_violation] = true;
                //         abort = true;

                //         break;
                //     }

                //     unlockEdgeNeighbors(edge, nodes_to_update);
                // }
            }

            if (not current_violations.empty() && best_move.gain < 5) {

                bool abort = false;
                for (auto &edge : current_violations) {

                    const auto &source = instance->getComputationalDag().source(edge);
                    const auto &target = instance->getComputationalDag().target(edge);

                    if (unlock[source] == 0 && unlock[target] == 0) {

                        super_locked[source] = true;
                        super_locked[target] = true;
                        super_locked[node_causing_first_violation] = true;
                        abort = true;

                        break;
                    }

                    unlockEdgeNeighbors(edge, nodes_to_update);

                    if (source != best_move.node) {

                        if (unlockNode(source)) {
                            nodes_to_update.insert(source);
                        }
                    }

                    if (target != best_move.node) {

                        if (unlockNode(target)) {
                            nodes_to_update.insert(target);
                        }
                    }
                }

                if (abort) {
                    // std::cout << "abort condition met" << std::endl;
                    break;
                }
            }

            // if (current_violations.size() > 200) {

            //     std::cout << "abort condition 2 met" << std::endl;
            //     //  break;
            // }

            // std::vector<VertexType> nodes_to_update_vec(nodes_to_update.begin(), nodes_to_update.end());
            // #pragma omp parallel for
            // for (auto node : nodes_to_update_vec) {
            //     computeNodeGain(node);
            // }

            // for (auto node : nodes_to_update_vec) {
            //     computeMaxGain(node);
            // }

            if (nodes_to_update.find(best_move.node) != nodes_to_update.end()) {
                nodes_to_update.erase(best_move.node);
            }

            updateNodesGain(nodes_to_update);

            if (best_move.change_in_cost < 0 && current_violations.empty() && current_feasible) {

                if (best_schedule_costs > current_cost + best_move.change_in_cost) {

                    // std::cout << "costs increased .. save best schedule with costs "
                    //           << current_cost + best_move.change_in_cost << std::endl;

                    best_schedule_costs = current_cost + best_move.change_in_cost;
                    setBestSchedule(vector_schedule); // O(n)
                    reverseMoveBestSchedule(best_move);
                }
            }

            if (current_violations.empty() && not current_feasible) {

                // std::cout << "<=============== moved from infeasible to feasible" << std::endl;
                current_feasible = true;

                if (current_cost <= best_schedule_costs) {
                    // std::cout << "new schdule better than previous best schedule" << std::endl;

                    //                setBestSchedule(vector_schedule);
                    //                best_schedule_costs = current_costs;
                } else {

                    // std::cout << "... but costs did not improve: " << current_cost
                    //           << " vs best schedule: " << best_schedule_costs << std::endl;

                    if (current_cost > (1.02 + counter * 0.002) * best_schedule_costs) {
                        //    std::cout << " rollback to best schedule" << std::endl;
                        setCurrentSchedule(*best_schedule); // O(n + p*s)
                        compute_superstep_datastructures(); // O(n)

                        penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
                        reward_factor = 1.0;

                        current_cost = best_schedule_costs;
                        current_violations.clear();
                        current_feasible = true;
                        resetGainHeap();
                        setup_gain_heap_unlocked_nodes();
                        // counter = 0;

                        failed_branches++;
                    }
                }

            } else if (not current_violations.empty() && current_feasible) {
                // std::cout << "================> moved from feasible to infeasible" << std::endl;
                current_feasible = false;
                node_causing_first_violation = best_move.node;
                // unlockNeighbours(best_move.node);

                if (current_cost + best_move.change_in_cost <= best_schedule_costs) {
                    // std::cout << "save best schedule with costs " << current_cost + best_move.change_in_cost
                    //         << std::endl;
                    best_schedule_costs = current_cost + best_move.change_in_cost;
                    setBestSchedule(vector_schedule); // O(n)
                    reverseMoveBestSchedule(best_move);
                }
            }

            if (not current_feasible) {

                if (current_cost > (1.02 + counter * 0.002) * best_schedule_costs) {

                    //            std::cout << "current cost " << current_cost
                    //                      << " too far away from best schedule costs: " << best_schedule_costs << "
                    //                      metric "
                    //                      << (1.20 + counter * 0.001) * best_schedule_costs << " counter " << counter
                    //                     << " #violations " << current_violations.size() << " rollback to best
                    //                     schedule"
                    //                     << std::endl;

                    setCurrentSchedule(*best_schedule); // O(n + p*s)
                    compute_superstep_datastructures(); // O(n)

                    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
                    reward_factor = 1.0;

                    current_cost = best_schedule_costs;
                    current_violations.clear();
                    current_feasible = true;
                    resetGainHeap();
                    setup_gain_heap_unlocked_nodes();
                    // counter = 0;
                    failed_branches++;

                } // else {

                //    checkInsertSuperstep();
                //}
            }

        } // while

        counter++;

        std::cout << "current costs end while: " << current_cost << " with " << current_violations.size()
                  << " violation, computed costs: " << current_costs()
                  << " violations, best sol costs: " << best_schedule_costs << " counter: " << counter << std::endl;

        // if (step_selection_counter >= num_steps && counter % std::max(num_steps, 100u) == 0) {
        if (step_selection_counter >= num_steps && counter % 20 == 0) {
            step_selection_counter = 1;
        }

        if (current_violations.empty()) {

            if (current_cost <= best_schedule_costs) {
                setBestSchedule(vector_schedule);
                best_schedule_costs = current_cost;
            }

            penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
            reward_factor = 1.0;

            std::unordered_set<VertexType> node_selection = selectNodesFindRemoveSteps(selection_threshold);
            // selectNodesConseqStepsReduceNumSteps(
            //   selection_threshold); //

            penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                             (current_violations.size() + base_penalty_factor) * current_violations.size();

            reward_factor = base_reward + (current_violations.size() + 2 + base_penalty_factor) *
                                              current_violations.size() * current_violations.size();

            current_cost = current_costs();
            resetLockedNodesAndComputeGains();

            if (current_violations.size() > 0) {
                current_feasible = false;
                for (const auto &node : node_selection) {
                    locked[node] = false;
                    unlock[node] = 2 * max_num_unlocks;
                }
            }

            updateNodesGain(node_selection);

        } else {

            std::cout << "current solution not feasible .. rolling back to best solution with costs "
                      << best_schedule_costs << std::endl;

            resetLockedNodes();
            lockAll();

            setCurrentSchedule(*best_schedule); // O (n + p*s)

            current_cost = best_schedule_costs;
            current_violations.clear();
            current_feasible = true;

            std::unordered_set<VertexType> node_selection =
                selectNodesFindRemoveSteps(selection_threshold); // selectNodesConseqStepsReduceNumSteps(
            //     selection_threshold); //
            current_cost = current_costs();

            for (const auto &node : node_selection) {
                locked[node] = false;
                unlock[node] = max_num_unlocks;
            }

            if (current_violations.size() > 0) {
                current_feasible = false;

                for (const auto &node : node_selection) {
                    locked[node] = false;
                    unlock[node] = 2 * max_num_unlocks;
                }
            } else {
                for (const auto &node : node_selection) {
                    locked[node] = false;
                    unlock[node] = max_num_unlocks;
                }
            }

            compute_superstep_datastructures(); // O(n)

            // penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
            // reward_factor = 1.0;

            penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                             (current_violations.size() + base_penalty_factor) * current_violations.size();

            reward_factor = base_reward + (current_violations.size() + 2 + base_penalty_factor) *
                                              current_violations.size() * current_violations.size();

            resetGainHeap();
            initalize_gain_heap(node_selection);
        }

        std::cout << "number of supersteps: " << num_steps << " selection step counter: " << step_selection_counter
                  << std::endl;

        if (compute_with_time_limit) {

            auto finish_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

            if (duration > timeLimitSeconds) {
                break;
            }
        }

        if (epoch_counter >= max_epochs) {
            break;
        }

        if (best_iter_costs <= current_cost) {

            if (improvement_counter++ == std::log(max_iterations)) {
                // std::cout << "no improvement ... end local search " << std::endl;
                break;
            }
        } else {
            improvement_counter = 0;
        }

        if (counter % 500 == 0) {
            super_locked = std::vector<bool>(num_nodes, false);
        }

    } // for

    std::cout << "LKTotalComm end best schedule costs: " << best_schedule_costs
              << " computed costs: " << current_costs() << std::endl;
    cleanup_datastructures();

    if (initial_costs > current_cost)
        return true;
    else
        return false;


}
*/
/*
bool LKTotalCommScheduler::start_feas() {



    vector_schedule = VectorSchedule(*best_schedule);
    set_schedule = SetSchedule(*best_schedule);

    comm_multiplier = 1.0 / instance->numberOfProcessors();
    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
    double base_reward = penalty_factor + 5.0;

    // std::uniform_int_distribution<> coin(0, 2);
    auto start_time = std::chrono::high_resolution_clock::now();

    initalize_datastructures();
    compute_superstep_datastructures();

    // checkMergeSupersteps();

    best_schedule_costs = current_costs();

    double initial_costs = best_schedule_costs;
    current_cost = initial_costs;

    std::cout << "LKTotalComm start() best schedule costs: " << best_schedule_costs << std::endl;

    setParameters();

    std::unordered_set<VertexType> node_selection = selectNodesFindRemoveSteps(selection_threshold);
    // selectNodesConseqStepsReduceNumSteps(selection_threshold); //
    // selectNodesConseqStepsMaxWork(selection_threshold); // selectNodesPermutationThreshold(selection_threshold);

    for (const auto &node : node_selection) {
        locked[node] = false;
    }

    initalize_gain_heap(node_selection);

    std::cout << "Initial costs " << current_cost << std::endl;
    unsigned improvement_counter = 0;
    counter = 0;

    VertexType node_causing_first_violation = 0;

    for (unsigned i = 0; i < max_iterations; i++) {

        unsigned failed_branches = 0;
        double best_iter_costs = current_cost;

        unsigned inner_counter = 0;
        unsigned barrier = 1500;
        if (current_violations.size() > 0) {
            barrier = 3000;
        }

        while (failed_branches < 3 && inner_counter < barrier && max_gain_heap.size() > 0) {

            inner_counter++;

            Move best_move = findMove(); // O(log n)

            //             if (best_move.gain < 0) {

            //                     std::cout << "abort condition met" << std::endl;
            // std::cout << "current costs: " << current_cost << " best move gain: " << best_move.gain
            //                        << " best move costs: " << best_move.change_in_cost << std::endl;

            //                //break;
            //             }

            applyMove(best_move); // O(p + log n)

            //  std::cout << inner_counter << " best move node: " << best_move.node << " current costs: " <<
            //  current_cost
            //            << " best move gain: " << best_move.gain << " best move costs: " << best_move.change_in_cost
            //           << " number of violations: " << current_violations.size() << std::endl;

            locked_nodes.insert(best_move.node);
            locked[best_move.node] = true;

            std::deque<VertexType> nodes_to_move;

            updateViolations(best_move.node, nodes_to_move); // O(Delta_max * log(current_violations.size())

            penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                             (current_violations.size() + base_penalty_factor) * current_violations.size();

            reward_factor = base_reward + (current_violations.size() + 2 + base_penalty_factor) *
                                              current_violations.size() * current_violations.size();

            std::unordered_set<VertexType> nodes_to_update = collectNodesToUpdate(best_move);

            while (!nodes_to_move.empty()) {

                VertexType node = nodes_to_move.front();
                nodes_to_move.pop_front();

                computeNodeGain(node);
                Move node_move = compute_best_move(node);

                applyMove(node_move); // O(p + log n)

                locked_nodes.insert(node);
                locked[node] = true;

                updateViolations(best_move.node, nodes_to_move);

                penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                                 (current_violations.size() + base_penalty_factor) * current_violations.size();

                reward_factor = base_reward + (current_violations.size() + 2 + base_penalty_factor) *
                                                  current_violations.size() * current_violations.size();
            }

            unlockNeighbours(best_move, nodes_to_update);

            if (not current_violations.empty() && best_move.gain < 5) {

                bool abort = false;
                for (auto &edge : current_violations) {

                    const auto &source = instance->getComputationalDag().source(edge);
                    const auto &target = instance->getComputationalDag().target(edge);

                    if (unlock[source] == 0 && unlock[target] == 0) {

                        super_locked[source] = true;
                        super_locked[target] = true;
                        super_locked[node_causing_first_violation] = true;
                        abort = true;

                        break;
                    }

                    unlockEdgeNeighbors(edge, nodes_to_update);

                    if (source != best_move.node) {

                        if (unlockNode(source)) {
                            nodes_to_update.insert(source);
                        }
                    }

                    if (target != best_move.node) {

                        if (unlockNode(target)) {
                            nodes_to_update.insert(target);
                        }
                    }
                }

                if (abort) {
                    // std::cout << "abort condition met" << std::endl;
                    break;
                }
            }

            // if (current_violations.size() > 200) {

            //     std::cout << "abort condition 2 met" << std::endl;
            //     //  break;
            // }

            // std::vector<VertexType> nodes_to_update_vec(nodes_to_update.begin(), nodes_to_update.end());
            // #pragma omp parallel for
            // for (auto node : nodes_to_update_vec) {
            //     computeNodeGain(node);
            // }

            // for (auto node : nodes_to_update_vec) {
            //     computeMaxGain(node);
            // }

            if (nodes_to_update.find(best_move.node) != nodes_to_update.end()) {
                nodes_to_update.erase(best_move.node);
            }

            updateNodesGain(nodes_to_update);

            if (best_move.change_in_cost < 0 && current_violations.empty() && current_feasible) {

                if (best_schedule_costs > current_cost + best_move.change_in_cost) {

                    // std::cout << "costs increased .. save best schedule with costs "
                    //           << current_cost + best_move.change_in_cost << std::endl;

                    best_schedule_costs = current_cost + best_move.change_in_cost;
                    setBestSchedule(vector_schedule); // O(n)
                    reverseMoveBestSchedule(best_move);
                }
            }

            if (current_violations.empty() && not current_feasible) {

                // std::cout << "<=============== moved from infeasible to feasible" << std::endl;
                current_feasible = true;

                if (current_cost <= best_schedule_costs) {
                    // std::cout << "new schdule better than previous best schedule" << std::endl;

                    //                setBestSchedule(vector_schedule);
                    //                best_schedule_costs = current_costs;
                } else {

                    // std::cout << "... but costs did not improve: " << current_cost
                    //           << " vs best schedule: " << best_schedule_costs << std::endl;

                    if (current_cost > (1.02 + counter * 0.002) * best_schedule_costs) {
                        //    std::cout << " rollback to best schedule" << std::endl;
                        setCurrentSchedule(*best_schedule); // O(n + p*s)
                        compute_superstep_datastructures(); // O(n)

                        penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
                        reward_factor = 1.0;

                        current_cost = best_schedule_costs;
                        current_violations.clear();
                        current_feasible = true;
                        resetGainHeap();
                        setup_gain_heap_unlocked_nodes();
                        // counter = 0;

                        failed_branches++;
                    }
                }

            } else if (not current_violations.empty() && current_feasible) {
                // std::cout << "================> moved from feasible to infeasible" << std::endl;
                current_feasible = false;
                node_causing_first_violation = best_move.node;
                // unlockNeighbours(best_move.node);

                if (current_cost + best_move.change_in_cost <= best_schedule_costs) {
                    // std::cout << "save best schedule with costs " << current_cost + best_move.change_in_cost
                    //         << std::endl;
                    best_schedule_costs = current_cost + best_move.change_in_cost;
                    setBestSchedule(vector_schedule); // O(n)
                    reverseMoveBestSchedule(best_move);
                }
            }

            if (not current_feasible) {

                if (current_cost > (1.02 + counter * 0.002) * best_schedule_costs) {

                    //            std::cout << "current cost " << current_cost
                    //                      << " too far away from best schedule costs: " << best_schedule_costs << "
                    //                      metric "
                    //                      << (1.20 + counter * 0.001) * best_schedule_costs << " counter " << counter
                    //                     << " #violations " << current_violations.size() << " rollback to best
                    //                     schedule"
                    //                     << std::endl;

                    setCurrentSchedule(*best_schedule); // O(n + p*s)
                    compute_superstep_datastructures(); // O(n)

                    penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
                    reward_factor = 1.0;

                    current_cost = best_schedule_costs;
                    current_violations.clear();
                    current_feasible = true;
                    resetGainHeap();
                    setup_gain_heap_unlocked_nodes();
                    // counter = 0;
                    failed_branches++;

                } // else {

                //    checkInsertSuperstep();
                //}
            }

        } // while

        counter++;

        std::cout << "current costs end while: " << current_cost << " with " << current_violations.size()
                  << " violation, computed costs: " << current_costs()
                  << " violations, best sol costs: " << best_schedule_costs << " counter: " << counter << std::endl;

        // if (step_selection_counter >= num_steps && counter % std::max(num_steps, 100u) == 0) {
        if (step_selection_counter >= num_steps && counter % 20 == 0) {
            step_selection_counter = 1;
        }

        if (current_violations.empty()) {

            if (current_cost <= best_schedule_costs) {
                setBestSchedule(vector_schedule);
                best_schedule_costs = current_cost;
            }

            penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
            reward_factor = 1.0;

            std::unordered_set<VertexType> node_selection =
                selectNodesFindRemoveSteps(selection_threshold); // selectNodesConseqStepsReduceNumSteps(
            //    selection_threshold); //

            penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                             (current_violations.size() + base_penalty_factor) * current_violations.size();

            reward_factor = base_reward + (current_violations.size() + 2 + base_penalty_factor) *
                                              current_violations.size() * current_violations.size();

            current_cost = current_costs();
            resetLockedNodesAndComputeGains();

            if (current_violations.size() > 0) {
                current_feasible = false;
                for (const auto &node : node_selection) {
                    locked[node] = false;
                    unlock[node] = 2 * max_num_unlocks;
                }
            }

            updateNodesGain(node_selection);

        } else {

            std::cout << "current solution not feasible .. rolling back to best solution with costs "
                      << best_schedule_costs << std::endl;

            resetLockedNodes();
            lockAll();

            setCurrentSchedule(*best_schedule); // O (n + p*s)

            current_cost = best_schedule_costs;
            current_violations.clear();
            current_feasible = true;

            std::unordered_set<VertexType> node_selection =
                selectNodesFindRemoveSteps(selection_threshold); // selectNodesConseqStepsReduceNumSteps(
            //    selection_threshold); // selectNodesFindRemoveSteps(selection_threshold);
            current_cost = current_costs();

            for (const auto &node : node_selection) {
                locked[node] = false;
                unlock[node] = max_num_unlocks;
            }

            if (current_violations.size() > 0) {
                current_feasible = false;

                for (const auto &node : node_selection) {
                    locked[node] = false;
                    unlock[node] = 2 * max_num_unlocks;
                }
            } else {
                for (const auto &node : node_selection) {
                    locked[node] = false;
                    unlock[node] = max_num_unlocks;
                }
            }

            compute_superstep_datastructures(); // O(n)

            // penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts();
            // reward_factor = 1.0;

            penalty_factor = base_penalty_factor * comm_multiplier * instance->communicationCosts() +
                             (current_violations.size() + base_penalty_factor) * current_violations.size();

            reward_factor = base_reward + (current_violations.size() + 2 + base_penalty_factor) *
                                              current_violations.size() * current_violations.size();

            resetGainHeap();
            initalize_gain_heap(node_selection);
        }

        std::cout << "number of supersteps: " << num_steps << " selection step counter: " << step_selection_counter
                  << std::endl;

        if (compute_with_time_limit) {

            auto finish_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

            if (duration > timeLimitSeconds) {
                break;
            }
        }

        if (epoch_counter >= max_epochs) {
            break;
        }

        if (best_iter_costs <= current_cost) {

            if (improvement_counter++ == std::log(max_iterations)) {
                // std::cout << "no improvement ... end local search " << std::endl;
                break;
            }
        } else {
            improvement_counter = 0;
        }

        if (counter % 200 == 0) {
            super_locked = std::vector<bool>(num_nodes, false);
        }

    } // for

    std::cout << "LKTotalComm end best schedule costs: " << best_schedule_costs
              << " computed costs: " << current_costs() << std::endl;
    cleanup_datastructures();

    if (initial_costs > current_cost)
        return true;
    else
        return false;


}
*/