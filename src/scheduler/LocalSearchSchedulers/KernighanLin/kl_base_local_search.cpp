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

bool kl_base::run_local_search_unlock_delay() {

    const double initial_costs = current_schedule->current_cost;

#ifdef KL_DEBUG
    std::cout << "Initial costs " << initial_costs << std::endl;
#endif

#ifdef KL_PRINT_SCHEDULE
    print_best_schedule(0);
#endif

    unsigned no_improvement_iter_counter = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    select_nodes_check_remove_superstep();

    update_reward_penalty();

    initialize_gain_heap(node_selection);

    for (unsigned outer_counter = 0; outer_counter < parameters.max_outer_iterations; outer_counter++) {
#ifdef KL_DEBUG
        std::cout << "outer iteration " << outer_counter << " current costs: " << current_schedule->current_cost
                  << std::endl;
        if (max_gain_heap.size() == 0) {
            std::cout << "max gain heap empty" << std::endl;
        }
#endif

        unsigned conseq_no_gain_moves_counter = 0;

        unsigned failed_branches = 0;
        double best_iter_costs = current_schedule->current_cost;

        VertexType node_causing_first_violation = 0;

        unsigned inner_counter = 0;

        while (failed_branches < parameters.max_num_failed_branches &&
               inner_counter < parameters.max_inner_iterations && max_gain_heap.size() > 0) {

            inner_counter++;

            const bool iter_feasible = current_schedule->current_feasible;
            const double iter_costs = current_schedule->current_cost;

            kl_move best_move = find_best_move(); // O(log n)

            if (best_move.gain < -std::numeric_limits<double>::max() * .25) {
#ifdef KL_DEBUG
                std::cout << "abort iteration on very negative max gain" << std::endl;
#endif
                break;
            }

#ifdef KL_DEBUG
            std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                      << best_move.change_in_cost << " from step " << best_move.from_step << " to " << best_move.to_step
                      << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                      << " violations: " << current_schedule->current_violations.size() << " old cost "
                      << current_schedule->current_cost << " new cost "
                      << current_schedule->current_cost + best_move.change_in_cost << std::endl;
#endif

            current_schedule->apply_move(best_move); // O(p + log n)

            //             if (best_move.gain <= 0.000000001) {
            //                 conseq_no_gain_moves_counter++;

            //                 if (conseq_no_gain_moves_counter > 15) {

            //                     conseq_no_gain_moves_counter = 0;
            //                     parameters.initial_penalty = 0.0;
            //                     parameters.violations_threshold = 3;
            // #ifdef KL_DEBUG
            //                     std::cout << "more than 15 moves with gain <= 0, set " << parameters.initial_penalty
            //                               << " violations threshold " << parameters.violations_threshold <<
            //                               std::endl;
            // #endif
            //                 }

            //             } else {
            //                 conseq_no_gain_moves_counter = 0;
            //             }

            update_reward_penalty();
            locked_nodes.insert(best_move.node);

#ifdef KL_DEBUG
            double tmp_costs = current_schedule->current_cost;
            if (tmp_costs != compute_current_costs()) {

                std::cout << "current costs: " << current_schedule->current_cost
                          << " best move gain: " << best_move.gain << " best move costs: " << best_move.change_in_cost
                          << " tmp cost: " << tmp_costs << std::endl;

                std::cout << "! costs not equal " << std::endl;
            }
#endif

            if (iter_feasible != current_schedule->current_feasible) {

                if (iter_feasible) {
#ifdef KL_DEBUG
                    std::cout << "===> current schedule changed from feasible to infeasible" << std::endl;
#endif

                    node_causing_first_violation = best_move.node;

                    if (iter_costs < best_schedule_costs) {
#ifdef KL_DEBUG
                        std::cout << "save best schedule with costs " << iter_costs << std::endl;
#endif
                        best_schedule_costs = iter_costs;
                        save_best_schedule(current_schedule->vector_schedule); // O(n)
                        reverse_move_best_schedule(best_move);
#ifdef KL_DEBUG
                        std::cout << "KLBase save best schedule with (source node comm) cost "
                                  << best_schedule->computeCostsTotalCommunication() << " and number of supersteps "
                                  << best_schedule->numberOfSupersteps() << std::endl;
#endif
                    }

                } else {
#ifdef KL_DEBUG
                    std::cout << "===> current schedule changed from infeasible to feasible" << std::endl;
#endif
                }
            } else if (best_move.change_in_cost > 0 && current_schedule->current_feasible) {

                if (iter_costs < best_schedule_costs) {
#ifdef KL_DEBUG
                    std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                    best_schedule_costs = iter_costs;
                    save_best_schedule(current_schedule->vector_schedule); // O(n)
                    reverse_move_best_schedule(best_move);
#ifdef KL_DEBUG
                    std::cout << "KLBase save best schedule with (source node comm) cost "
                              << best_schedule->computeCostsTotalCommunication() << " and number of supersteps "
                              << best_schedule->numberOfSupersteps() << std::endl;
#endif
                }
            }

            compute_nodes_to_update(best_move);

            select_unlock_neighbors(best_move.node);

            if (check_violation_locked()) {

                if (iter_feasible != current_schedule->current_feasible && iter_feasible) {
                    node_causing_first_violation = best_move.node;
                }
                super_locked_nodes.insert(node_causing_first_violation);
#ifdef KL_DEBUG
                std::cout << "abort iteration on locked violation, super locking node " << node_causing_first_violation
                          << std::endl;
#endif
                break;
            }

            update_node_gains(nodes_to_update);

            if (not current_schedule->current_violations.size() > 4 && not iter_feasible) {
                const auto &iter = max_gain_heap.ordered_begin();
                if (iter->gain < parameters.gain_threshold) {

                    node_selection.clear();
                    select_nodes_violations();

                    update_reward_penalty();

                    initialize_gain_heap(node_selection);

#ifdef KL_DEBUG
                    std::cout << "max gain below gain threshold" << std::endl;
#endif
                }
            }

            if (current_schedule->current_cost >
                (parameters.max_div_best_sol_base_percent + outer_counter * parameters.max_div_best_sol_rate_percent) *
                    best_schedule_costs) {

#ifdef KL_DEBUG
                std::cout << "current cost " << current_schedule->current_cost
                          << " too far away from best schedule costs: " << best_schedule_costs
                          << " rollback to best schedule" << std::endl;
#endif

                current_schedule->set_current_schedule(*best_schedule);

                set_initial_reward_penalty();
                initialize_gain_heap_unlocked_nodes(node_selection);

#ifdef KL_DEBUG
                std::cout << "new current cost " << current_schedule->current_cost << std::endl;
#endif

                failed_branches++;
            }

        } // while

#ifdef KL_DEBUG
        std::cout << std::setprecision(12) << "end inner loop current cost: " << current_schedule->current_cost
                  << " with " << current_schedule->current_violations.size()
                  << " violation, best sol cost: " << best_schedule_costs << " with "
                  << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                  << parameters.max_outer_iterations << std::endl;
#endif

        if (current_schedule->current_feasible) {
            if (current_schedule->current_cost <= best_schedule_costs) {
                save_best_schedule(current_schedule->vector_schedule);
                best_schedule_costs = current_schedule->current_cost;
#ifdef KL_DEBUG
                std::cout << "KLBase save best schedule with (source node comm) cost "
                          << best_schedule->computeCostsTotalCommunication() << " and number of supersteps "
                          << best_schedule->numberOfSupersteps() << std::endl;
#endif
            } else {
                current_schedule->set_current_schedule(*best_schedule);
            }
        } else {
            current_schedule->set_current_schedule(*best_schedule);
        }

        if (outer_counter > 0 && outer_counter % 30 == 0) {
            super_locked_nodes.clear();
#ifdef KL_DEBUG
            std::cout << "---- reset super locked nodes" << std::endl;
#endif
        }


#ifdef KL_PRINT_SCHEDULE
if (best_iter_costs > current_schedule->current_cost) {
        print_best_schedule(outer_counter + 1);
}
#endif

        reset_locked_nodes();

        node_selection.clear();

        if (reset_superstep) {
            select_nodes_check_reset_superstep();
        } else {
            select_nodes_check_remove_superstep();
        }

        update_reward_penalty();

        initialize_gain_heap(node_selection);

#ifdef KL_DEBUG
        std::cout << "end of while, current cost " << current_schedule->current_cost << std::endl;
#endif

        if (compute_with_time_limit) {
            auto finish_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();
            if (duration > timeLimitSeconds) {
                break;
            }
        }

        if (best_iter_costs <= current_schedule->current_cost) {

            no_improvement_iter_counter++;

            if (no_improvement_iter_counter > parameters.reset_epoch_counter_threshold) {

                step_selection_epoch_counter = 0;
                parameters.reset_epoch_counter_threshold += current_schedule->num_steps();
#ifdef KL_DEBUG
                std::cout << "no improvement for " << no_improvement_iter_counter
                          << " iterations, reset epoc counter. Increase reset threshold to "
                          << parameters.reset_epoch_counter_threshold << std::endl;
#endif
            }

            if (no_improvement_iter_counter > 10 && no_improvement_iter_counter % 15 == 0) {

                step_selection_epoch_counter = 0;

                if (alternate_reset_remove_superstep) {
                    reset_superstep = !reset_superstep;
                }

#ifdef KL_DEBUG
                std::cout << "no improvement for " << no_improvement_iter_counter << " reset superstep "
                          << reset_superstep << std::endl;
#endif
            }

            if (no_improvement_iter_counter > 50 && no_improvement_iter_counter % 3 == 0) {

                parameters.initial_penalty = 0.0;
                parameters.violations_threshold = 5;

            } else if (no_improvement_iter_counter > 30 && no_improvement_iter_counter % 5 == 0) {

                parameters.initial_penalty = 0.0;
                parameters.violations_threshold = 4;

            } else if (no_improvement_iter_counter > 9 && no_improvement_iter_counter % 10 == 0) {

                parameters.initial_penalty = 0.0;
                parameters.violations_threshold = 3;
#ifdef KL_DEBUG
                std::cout << "---- reset initial penalty " << parameters.initial_penalty << " violations threshold "
                          << parameters.violations_threshold << std::endl;
#endif
            }

            if (no_improvement_iter_counter == 35) {

                parameters.max_div_best_sol_base_percent *= 1.02;
#ifdef KL_DEBUG
                std::cout << "no improvement for " << no_improvement_iter_counter
                          << " iterations, increase max_div_best_sol_base_percent to "
                          << parameters.max_div_best_sol_base_percent << std::endl;
#endif
            }

            if (no_improvement_iter_counter >= parameters.max_no_improvement_iterations) {
#ifdef KL_DEBUG
                std::cout << "no improvement for " << parameters.max_no_improvement_iterations
                          << " iterations, end local search" << std::endl;
#endif
                break;
            }
        } else {
            no_improvement_iter_counter = 0;
        }

    } // for

    cleanup_datastructures();

#ifdef KL_DEBUG
    std::cout << "kl done, current cost " << best_schedule_costs << " vs " << initial_costs << " initial costs"
              << std::endl;
    assert(best_schedule->satisfiesPrecedenceConstraints());
#endif

    if (initial_costs > current_schedule->current_cost)
        return true;
    else
        return false;
}

bool kl_base::run_local_search_remove_supersteps() {

    const double initial_costs = current_schedule->current_cost;

#ifdef KL_DEBUG
    std::cout << "Initial costs " << initial_costs << std::endl;
#endif

    unsigned no_improvement_iter_counter = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    select_nodes_check_remove_superstep();

    update_reward_penalty();

    initialize_gain_heap(node_selection);

    for (unsigned outer_counter = 0; outer_counter < parameters.max_outer_iterations; outer_counter++) {
#ifdef KL_DEBUG
        std::cout << "outer iteration " << outer_counter << " current costs: " << current_schedule->current_cost
                  << std::endl;
        if (max_gain_heap.size() == 0) {
            std::cout << "max gain heap empty" << std::endl;
        }
#endif

        unsigned conseq_no_gain_moves_counter = 0;

        unsigned failed_branches = 0;
        double best_iter_costs = current_schedule->current_cost;

        VertexType node_causing_first_violation = 0;

        unsigned inner_counter = 0;

        while (failed_branches < parameters.max_num_failed_branches &&
               inner_counter < parameters.max_inner_iterations && max_gain_heap.size() > 0) {

            inner_counter++;

            const bool iter_feasible = current_schedule->current_feasible;
            const double iter_costs = current_schedule->current_cost;

            kl_move best_move = find_best_move(); // O(log n)

            if (best_move.gain < -std::numeric_limits<double>::max() * .25) {
#ifdef KL_DEBUG
                std::cout << "abort iteration on very negative max gain" << std::endl;
#endif
                break;
            }

#ifdef KL_DEBUG
            std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                      << best_move.change_in_cost << " from step " << best_move.from_step << " to " << best_move.to_step
                      << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                      << " violations: " << current_schedule->current_violations.size() << " old cost "
                      << current_schedule->current_cost << " new cost "
                      << current_schedule->current_cost + best_move.change_in_cost << std::endl;
#endif

            current_schedule->apply_move(best_move); // O(p + log n)

            update_reward_penalty();
            locked_nodes.insert(best_move.node);

#ifdef KL_DEBUG
            double tmp_costs = current_schedule->current_cost;
            if (tmp_costs != compute_current_costs()) {

                std::cout << "current costs: " << current_schedule->current_cost
                          << " best move gain: " << best_move.gain << " best move costs: " << best_move.change_in_cost
                          << " tmp cost: " << tmp_costs << std::endl;

                std::cout << "! costs not equal " << std::endl;
            }
#endif

            if (iter_feasible != current_schedule->current_feasible) {

                if (iter_feasible) {
#ifdef KL_DEBUG
                    std::cout << "===> current schedule changed from feasible to infeasible" << std::endl;
#endif

                    node_causing_first_violation = best_move.node;

                    if (iter_costs < best_schedule_costs) {
#ifdef KL_DEBUG
                        std::cout << "save best schedule with costs " << iter_costs << std::endl;
#endif
                        best_schedule_costs = iter_costs;
                        save_best_schedule(current_schedule->vector_schedule); // O(n)
                        reverse_move_best_schedule(best_move);
#ifdef KL_DEBUG
                        std::cout << "KLBase save best schedule with (source node comm) cost "
                                  << best_schedule->computeCostsTotalCommunication() << " and number of supersteps "
                                  << best_schedule->numberOfSupersteps() << std::endl;
#endif
                    }

                } else {
#ifdef KL_DEBUG
                    std::cout << "===> current schedule changed from infeasible to feasible" << std::endl;
#endif
                }
            } else if (best_move.change_in_cost > 0 && current_schedule->current_feasible) {

                if (iter_costs < best_schedule_costs) {
#ifdef KL_DEBUG
                    std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                    best_schedule_costs = iter_costs;
                    save_best_schedule(current_schedule->vector_schedule); // O(n)
                    reverse_move_best_schedule(best_move);
#ifdef KL_DEBUG
                    std::cout << "KLBase save best schedule with (source node comm) cost "
                              << best_schedule->computeCostsTotalCommunication() << " and number of supersteps "
                              << best_schedule->numberOfSupersteps() << std::endl;
#endif
                }
            }

            compute_nodes_to_update(best_move);

            select_unlock_neighbors(best_move.node);

            if (check_violation_locked()) {

                if (iter_feasible != current_schedule->current_feasible && iter_feasible) {
                    node_causing_first_violation = best_move.node;
                }
                super_locked_nodes.insert(node_causing_first_violation);
#ifdef KL_DEBUG
                std::cout << "abort iteration on locked violation, super locking node " << node_causing_first_violation
                          << std::endl;
#endif
                break;
            }

            update_node_gains(nodes_to_update);

            if (current_schedule->current_cost >
                (parameters.max_div_best_sol_base_percent + outer_counter * parameters.max_div_best_sol_rate_percent) *
                    best_schedule_costs) {

#ifdef KL_DEBUG
                std::cout << "current cost " << current_schedule->current_cost
                          << " too far away from best schedule costs: " << best_schedule_costs
                          << " rollback to best schedule" << std::endl;
#endif

                current_schedule->set_current_schedule(*best_schedule);

                set_initial_reward_penalty();
                initialize_gain_heap_unlocked_nodes(node_selection);

#ifdef KL_DEBUG
                std::cout << "new current cost " << current_schedule->current_cost << std::endl;
#endif

                failed_branches++;
            }

        } // while

#ifdef KL_DEBUG
        std::cout << std::setprecision(12) << "end inner loop current cost: " << current_schedule->current_cost
                  << " with " << current_schedule->current_violations.size()
                  << " violation, best sol cost: " << best_schedule_costs << " with "
                  << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                  << parameters.max_outer_iterations << std::endl;
#endif

        if (current_schedule->current_feasible) {
            if (current_schedule->current_cost <= best_schedule_costs) {
                save_best_schedule(current_schedule->vector_schedule);
                best_schedule_costs = current_schedule->current_cost;
#ifdef KL_DEBUG
                std::cout << "KLBase save best schedule with (source node comm) cost "
                          << best_schedule->computeCostsTotalCommunication() << " and number of supersteps "
                          << best_schedule->numberOfSupersteps() << std::endl;
#endif
            } else {
                current_schedule->set_current_schedule(*best_schedule);
            }
        } else {
            current_schedule->set_current_schedule(*best_schedule);
        }

        if (outer_counter > 0 && outer_counter % 30 == 0) {
            super_locked_nodes.clear();
#ifdef KL_DEBUG
            std::cout << "---- reset super locked nodes" << std::endl;
#endif
        }

        reset_locked_nodes();

        node_selection.clear();
        select_nodes_check_remove_superstep();

        update_reward_penalty();

        initialize_gain_heap(node_selection);

#ifdef KL_DEBUG
        std::cout << "end of while, current cost " << current_schedule->current_cost << std::endl;
#endif

        if (compute_with_time_limit) {
            auto finish_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();
            if (duration > timeLimitSeconds) {
                break;
            }
        }

        if (best_iter_costs <= current_schedule->current_cost) {

            no_improvement_iter_counter++;

            if (no_improvement_iter_counter > parameters.reset_epoch_counter_threshold) {

                step_selection_epoch_counter = 0;
                parameters.reset_epoch_counter_threshold += current_schedule->num_steps();
#ifdef KL_DEBUG
                std::cout << "no improvement for " << no_improvement_iter_counter
                          << " iterations, reset epoc counter. Increase reset threshold to "
                          << parameters.reset_epoch_counter_threshold << std::endl;
#endif
            }

            if (no_improvement_iter_counter > 10) {

                parameters.initial_penalty = 0.0;
                parameters.violations_threshold = 3;
#ifdef KL_DEBUG
                std::cout << "---- reset initial penalty " << parameters.initial_penalty << " violations threshold "
                          << parameters.violations_threshold << std::endl;
#endif
            }

            if (no_improvement_iter_counter == 35) {

                parameters.max_div_best_sol_base_percent *= 1.02;
#ifdef KL_DEBUG
                std::cout << "no improvement for " << no_improvement_iter_counter
                          << " iterations, increase max_div_best_sol_base_percent to "
                          << parameters.max_div_best_sol_base_percent << std::endl;
#endif
            }

            if (no_improvement_iter_counter >= parameters.max_no_improvement_iterations) {
#ifdef KL_DEBUG
                std::cout << "no improvement for " << parameters.max_no_improvement_iterations
                          << " iterations, end local search" << std::endl;
#endif
                break;
            }
        } else {
            no_improvement_iter_counter = 0;
        }

    } // for

    cleanup_datastructures();

#ifdef KL_DEBUG
    std::cout << "kl done, current cost " << best_schedule_costs << " vs " << initial_costs << " initial costs"
              << std::endl;
    assert(best_schedule->satisfiesPrecedenceConstraints());
#endif

    if (initial_costs > current_schedule->current_cost)
        return true;
    else
        return false;
}

bool kl_base::run_local_search_simple() {

    set_initial_reward_penalty();

    const double initial_costs = current_schedule->current_cost;

    unsigned improvement_counter = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    select_nodes();

    initialize_gain_heap(node_selection);

#ifdef KL_DEBUG
    std::cout << "Initial costs " << initial_costs << std::endl;
#endif

    for (unsigned outer_counter = 0; outer_counter < parameters.max_outer_iterations; outer_counter++) {
#ifdef KL_DEBUG
        std::cout << "outer iteration " << outer_counter << std::endl;
        if (max_gain_heap.size() == 0) {
            std::cout << "max gain heap empty" << std::endl;
        }
#endif
        unsigned failed_branches = 0;
        double best_iter_costs = current_schedule->current_cost;

        VertexType node_causing_first_violation = 0;

        unsigned inner_counter = 0;

        while (failed_branches < parameters.max_num_failed_branches &&
               inner_counter < parameters.max_inner_iterations && max_gain_heap.size() > 0) {

            inner_counter++;

            const bool iter_feasible = current_schedule->current_feasible;
            const double iter_costs = current_schedule->current_cost;

            kl_move best_move = find_best_move(); // O(log n)

            if (best_move.gain < -std::numeric_limits<double>::max() * .25) {
#ifdef KL_DEBUG
                std::cout << "abort iteration on very negative max gain" << std::endl;
#endif
                break;
            }

#ifdef KL_DEBUG
            std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                      << best_move.change_in_cost << " from step " << best_move.from_step << " to " << best_move.to_step
                      << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                      << " violations: " << current_schedule->current_violations.size() << " cost "
                      << current_schedule->current_cost << std::endl;
#endif

            current_schedule->apply_move(best_move); // O(p + log n)

            update_reward_penalty();
            locked_nodes.insert(best_move.node);

#ifdef KL_DEBUG
            double tmp_costs = current_schedule->current_cost;
            if (tmp_costs != compute_current_costs()) {

                std::cout << "current costs: " << current_schedule->current_cost
                          << " best move gain: " << best_move.gain << " best move costs: " << best_move.change_in_cost
                          << " tmp cost: " << tmp_costs << std::endl;

                std::cout << "! costs not equal " << std::endl;
            }
#endif

            if (iter_feasible != current_schedule->current_feasible) {

                if (iter_feasible) {
#ifdef KL_DEBUG
                    std::cout << "===> current schedule changed from feasible to infeasible" << std::endl;
#endif

                    node_causing_first_violation = best_move.node;

                    if (iter_costs < best_schedule_costs) {
#ifdef KL_DEBUG
                        std::cout << "save best schedule with costs " << iter_costs << std::endl;
#endif
                        best_schedule_costs = iter_costs;
                        save_best_schedule(current_schedule->vector_schedule); // O(n)
                        reverse_move_best_schedule(best_move);
                    }

                } else {
#ifdef KL_DEBUG
                    std::cout << "===> current schedule changed from infeasible to feasible" << std::endl;
#endif
                }
            } else if (best_move.change_in_cost > 0 && current_schedule->current_feasible) {

                if (iter_costs < best_schedule_costs) {
#ifdef KL_DEBUG
                    std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                    best_schedule_costs = iter_costs;
                    save_best_schedule(current_schedule->vector_schedule); // O(n)
                    reverse_move_best_schedule(best_move);
                }
            }

            compute_nodes_to_update(best_move);

            select_unlock_neighbors(best_move.node);

            if (check_violation_locked()) {

                if (iter_feasible != current_schedule->current_feasible && iter_feasible) {
                    node_causing_first_violation = best_move.node;
                }
                super_locked_nodes.insert(node_causing_first_violation);
#ifdef KL_DEBUG
                std::cout << "abort iteration on locked violation, super locking node " << node_causing_first_violation
                          << std::endl;
#endif
                break;
            }

            update_node_gains(nodes_to_update);

            if (current_schedule->current_cost >
                (parameters.max_div_best_sol_base_percent + outer_counter * parameters.max_div_best_sol_rate_percent) *
                    best_schedule_costs) {

#ifdef KL_DEBUG
                std::cout << "current cost " << current_schedule->current_cost
                          << " too far away from best schedule costs: " << best_schedule_costs
                          << " rollback to best schedule" << std::endl;
#endif

                current_schedule->set_current_schedule(*best_schedule);

                set_initial_reward_penalty();
                initialize_gain_heap_unlocked_nodes(node_selection);

                failed_branches++;
            }

        } // while

#ifdef KL_DEBUG
        std::cout << "end inner loop current cost: " << current_schedule->current_cost << " with "
                  << current_schedule->current_violations.size() << " violation, best sol cost: " << best_schedule_costs
                  << " with " << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                  << parameters.max_outer_iterations << std::endl;
#endif

        if (current_schedule->current_feasible) {
            if (current_schedule->current_cost <= best_schedule_costs) {
                save_best_schedule(current_schedule->vector_schedule);
                best_schedule_costs = current_schedule->current_cost;
            } else {
                current_schedule->set_current_schedule(*best_schedule);
            }
        } else {
            current_schedule->set_current_schedule(*best_schedule);
        }

        if (outer_counter == 20) {
            parameters.initial_penalty = 0.0;
#ifdef KL_DEBUG
            std::cout << "---- reset initial penalty" << std::endl;
#endif
        }
        if (outer_counter > 0 && outer_counter % 30 == 0) {
            super_locked_nodes.clear();
#ifdef KL_DEBUG
            std::cout << "---- reset super locked nodes" << std::endl;
#endif
        }

        reset_locked_nodes();

        node_selection.clear();
        select_nodes();

        set_initial_reward_penalty();

        initialize_gain_heap(node_selection);

#ifdef KL_DEBUG
        std::cout << "end of while, current cost " << current_schedule->current_cost << std::endl;
#endif

        if (compute_with_time_limit) {
            auto finish_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();
            if (duration > timeLimitSeconds) {
                break;
            }
        }

        if (best_iter_costs <= current_schedule->current_cost) {
            if (improvement_counter++ >= parameters.max_no_improvement_iterations) {
#ifdef KL_DEBUG
                std::cout << "no improvement for " << parameters.max_no_improvement_iterations
                          << " iterations, end local search" << std::endl;
#endif
                break;
            }
        } else {
            improvement_counter = 0;
        }

    } // for

    cleanup_datastructures();

#ifdef KL_DEBUG
    std::cout << "kl done, current cost " << best_schedule_costs << " vs " << initial_costs << " initial costs"
              << std::endl;
    assert(best_schedule->satisfiesPrecedenceConstraints());
#endif

    if (initial_costs > current_schedule->current_cost)
        return true;
    else
        return false;
}

bool kl_base::run_local_search_without_violations() {

    penalty = std::numeric_limits<double>::max() * .24;

    double initial_costs = current_schedule->current_cost;

    auto start_time = std::chrono::high_resolution_clock::now();

    select_nodes_threshold(parameters.selection_threshold);

    initialize_gain_heap(node_selection);

#ifdef KL_DEBUG
    std::cout << "Initial costs " << initial_costs << std::endl;
#endif

    for (unsigned outer_counter = 0; outer_counter < parameters.max_outer_iterations; outer_counter++) {
#ifdef KL_DEBUG
        std::cout << "outer iteration " << outer_counter << std::endl;
#endif
        unsigned failed_branches = 0;
        double best_iter_costs = current_schedule->current_cost;

        unsigned inner_counter = 0;

        while (failed_branches < 3 && inner_counter < parameters.max_inner_iterations && max_gain_heap.size() > 0) {

            inner_counter++;

            const double iter_costs = current_schedule->current_cost;

            kl_move best_move = find_best_move(); // O(log n)

            if (best_move.gain < -std::numeric_limits<double>::max() * .25) {
                continue;
            }

            current_schedule->apply_move(best_move); // O(p + log n)

            locked_nodes.insert(best_move.node);

#ifdef KL_DEBUG
            double tmp_costs = current_schedule->current_cost;
            if (tmp_costs != compute_current_costs()) {

                std::cout << "current costs: " << current_schedule->current_cost
                          << " best move gain: " << best_move.gain << " best move costs: " << best_move.change_in_cost
                          << " tmp cost: " << tmp_costs << std::endl;

                std::cout << "! costs not equal " << std::endl;
            }
#endif

            if (best_move.change_in_cost > 0 && current_schedule->current_feasible) {

                if (best_schedule_costs > iter_costs) {
#ifdef KL_DEBUG
                    std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                    best_schedule_costs = iter_costs;
                    save_best_schedule(current_schedule->vector_schedule); // O(n)
                    reverse_move_best_schedule(best_move);
                }
            }

            compute_nodes_to_update(best_move);

            select_unlock_neighbors(best_move.node);

            update_node_gains(nodes_to_update);

#ifdef KL_DEBUG
            std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                      << best_move.change_in_cost << " from step " << best_move.from_step << " to " << best_move.to_step
                      << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                      << " violations: " << current_schedule->current_violations.size() << " cost "
                      << current_schedule->current_cost << std::endl;
#endif

            // if (not current_schedule->current_feasible) {

            if (current_schedule->current_cost > (1.04 + outer_counter * 0.002) * best_schedule_costs) {

#ifdef KL_DEBUG
                std::cout << "current cost " << current_schedule->current_cost
                          << " too far away from best schedule costs: " << best_schedule_costs
                          << " rollback to best schedule" << std::endl;
#endif

                current_schedule->set_current_schedule(*best_schedule);

                // set_initial_reward_penalty();
                initialize_gain_heap_unlocked_nodes(node_selection);

                failed_branches++;
            }
            //}

        } // while

#ifdef KL_DEBUG
        std::cout << "end inner loop current cost: " << current_schedule->current_cost << " with "
                  << current_schedule->current_violations.size() << " violation, best sol cost: " << best_schedule_costs
                  << " with " << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                  << parameters.max_outer_iterations << std::endl;
#endif

        if (current_schedule->current_feasible) {
            if (current_schedule->current_cost <= best_schedule_costs) {
                save_best_schedule(current_schedule->vector_schedule);
                best_schedule_costs = current_schedule->current_cost;
            } else {
                current_schedule->set_current_schedule(*best_schedule);
            }
        } else {
            current_schedule->set_current_schedule(*best_schedule);
        }

        reset_locked_nodes();
        node_selection.clear();
        select_nodes_threshold(parameters.selection_threshold);

        initialize_gain_heap(node_selection);

#ifdef KL_DEBUG
        std::cout << "end of while, current cost " << current_schedule->current_cost << std::endl;
#endif

        if (compute_with_time_limit) {

            auto finish_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

            if (duration > timeLimitSeconds) {
                break;
            }
        }

    } // for

    // std::cout << "LKTotalComm end best schedule costs: " << best_schedule_costs
    //           << " computed costs: " << compute_current_costs() << std::endl;
    cleanup_datastructures();

    if (initial_costs > current_schedule->current_cost)
        return true;
    else
        return false;
}