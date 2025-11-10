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

#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "kl_current_schedule.hpp"

#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

// #define KL_PRINT_SCHEDULE

#ifdef KL_PRINT_SCHEDULE
#include "file_interactions/DotFileWriter.hpp"
#endif

namespace osp {

struct kl_base_parameter {

    double max_div_best_sol_base_percent = 1.05;
    double max_div_best_sol_rate_percent = 0.002;

    unsigned max_num_unlocks = 1;
    unsigned max_num_failed_branches = 5;

    unsigned max_inner_iterations = 150;
    unsigned max_outer_iterations = 100;

    unsigned max_no_improvement_iterations = 75;

    std::size_t selection_threshold;
    bool select_all_nodes = false;

    double initial_penalty = 0.0;

    double gain_threshold = -10.0;
    double change_in_cost_threshold = 0.0;

    bool quick_pass = false;

    unsigned max_step_selection_epochs = 4;
    unsigned reset_epoch_counter_threshold = 10;

    unsigned violations_threshold = 0;
};

template<typename Graph_t, typename MemoryConstraint_t>
class kl_base : public ImprovementScheduler<Graph_t>, public Ikl_cost_function {

    static_assert(is_directed_graph_edge_desc_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(has_hashable_edge_desc_v<Graph_t>, "Graph_t must satisfy the has_hashable_edge_desc concept");
    static_assert(is_computational_dag_v<Graph_t>, "Graph_t must satisfy the computational_dag concept");

  private:
    using memw_t = v_memw_t<Graph_t>;
    using commw_t = v_commw_t<Graph_t>;
    using workw_t = v_workw_t<Graph_t>;

  protected:
    using VertexType = vertex_idx_t<Graph_t>;

    kl_base_parameter parameters;

    std::mt19937 gen;

    VertexType num_nodes;
    unsigned num_procs;

    double penalty = 0.0;
    double reward = 0.0;

    virtual void update_reward_penalty() = 0;
    virtual void set_initial_reward_penalty() = 0;

    boost::heap::fibonacci_heap<kl_move<Graph_t>> max_gain_heap;
    using heap_handle = typename boost::heap::fibonacci_heap<kl_move<Graph_t>>::handle_type;

    std::unordered_map<VertexType, heap_handle> node_heap_handles;

    std::vector<std::vector<std::vector<double>>> node_gains;
    std::vector<std::vector<std::vector<double>>> node_change_in_costs;

    kl_current_schedule<Graph_t, MemoryConstraint_t> &current_schedule;

    BspSchedule<Graph_t> *best_schedule;
    double best_schedule_costs;

    std::unordered_set<VertexType> locked_nodes;
    std::unordered_set<VertexType> super_locked_nodes;
    std::vector<unsigned> unlock;

    bool unlock_node(VertexType node) {

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
    }

    bool check_node_unlocked(VertexType node) {

        if (super_locked_nodes.find(node) == super_locked_nodes.end() &&
            locked_nodes.find(node) == locked_nodes.end()) {
            return true;
        }
        return false;
    };

    void reset_locked_nodes() {

        for (const auto &i : locked_nodes) {

            unlock[i] = parameters.max_num_unlocks;
        }

        locked_nodes.clear();
    }

    bool check_violation_locked() {

        if (current_schedule.current_violations.empty())
            return false;

        for (auto &edge : current_schedule.current_violations) {

            const auto &source_v = source(edge, current_schedule.instance->getComputationalDag());
            const auto &target_v = target(edge, current_schedule.instance->getComputationalDag());

            if (locked_nodes.find(source_v) == locked_nodes.end() ||
                locked_nodes.find(target_v) == locked_nodes.end()) {
                return false;
            }

            bool abort = false;
            if (locked_nodes.find(source_v) != locked_nodes.end()) {

                if (unlock_node(source_v)) {
                    nodes_to_update.insert(source_v);
                    node_selection.insert(source_v);
                } else {
                    abort = true;
                }
            }

            if (locked_nodes.find(target_v) != locked_nodes.end()) {

                if (unlock_node(target_v)) {
                    nodes_to_update.insert(target_v);
                    node_selection.insert(target_v);
                    abort = false;
                }
            }

            if (abort) {
                return true;
            }
        }

        return false;
    }

    void reset_gain_heap() {

        max_gain_heap.clear();
        node_heap_handles.clear();
    }

    virtual void initialize_datastructures() {

#ifdef KL_DEBUG
        std::cout << "KLBase initialize datastructures" << std::endl;
#endif

        node_gains = std::vector<std::vector<std::vector<double>>>(
            num_nodes, std::vector<std::vector<double>>(num_procs, std::vector<double>(3, 0)));

        node_change_in_costs = std::vector<std::vector<std::vector<double>>>(
            num_nodes, std::vector<std::vector<double>>(num_procs, std::vector<double>(3, 0)));

        unlock = std::vector<unsigned>(num_nodes, parameters.max_num_unlocks);

        current_schedule.initialize_current_schedule(*best_schedule);
        best_schedule_costs = current_schedule.current_cost;
    }

    std::unordered_set<VertexType> nodes_to_update;

    void compute_nodes_to_update(kl_move<Graph_t> move) {

        nodes_to_update.clear();

        for (const auto &target : current_schedule.instance->getComputationalDag().children(move.node)) {

            if (node_selection.find(target) != node_selection.end() &&
                locked_nodes.find(target) == locked_nodes.end() &&
                super_locked_nodes.find(target) == super_locked_nodes.end()) {

                nodes_to_update.insert(target);
            }
        }

        for (const auto &source : current_schedule.instance->getComputationalDag().parents(move.node)) {

            if (node_selection.find(source) != node_selection.end() &&
                locked_nodes.find(source) == locked_nodes.end() &&
                super_locked_nodes.find(source) == super_locked_nodes.end()) {

                nodes_to_update.insert(source);
            }
        }

        const unsigned start_step =
            std::min(move.from_step, move.to_step) == 0 ? 0 : std::min(move.from_step, move.to_step) - 1;
        const unsigned end_step = std::min(current_schedule.num_steps(), std::max(move.from_step, move.to_step) + 2);

#ifdef KL_DEBUG
        std::cout << "updating from step " << start_step << " to step " << end_step << std::endl;
#endif

        for (unsigned step = start_step; step < end_step; step++) {

            for (unsigned proc = 0; proc < num_procs; proc++) {

                for (const auto &node : current_schedule.set_schedule.step_processor_vertices[step][proc]) {

                    if (node_selection.find(node) != node_selection.end() &&
                        locked_nodes.find(node) == locked_nodes.end() &&
                        super_locked_nodes.find(node) == super_locked_nodes.end()) {

                        nodes_to_update.insert(node);
                    }
                }
            }
        }
    }

    void initialize_gain_heap(const std::unordered_set<VertexType> &nodes) {

        reset_gain_heap();

        for (const auto &node : nodes) {
            compute_node_gain(node);
            compute_max_gain_insert_or_update_heap(node);
        }
    }

    void initialize_gain_heap_unlocked_nodes(const std::unordered_set<VertexType> &nodes) {

        reset_gain_heap();

        for (const auto &node : nodes) {

            if (locked_nodes.find(node) == locked_nodes.end() &&
                super_locked_nodes.find(node) == super_locked_nodes.end()) {

                compute_node_gain(node);
                compute_max_gain_insert_or_update_heap(node);
            }
        }
    }

    void compute_node_gain(VertexType node) {

        const unsigned &current_proc = current_schedule.vector_schedule.assignedProcessor(node);
        const unsigned &current_step = current_schedule.vector_schedule.assignedSuperstep(node);

        for (unsigned new_proc = 0; new_proc < num_procs; new_proc++) {

            if (current_schedule.instance->isCompatible(node, new_proc)) {

                node_gains[node][new_proc][0] = 0.0;
                node_gains[node][new_proc][1] = 0.0;
                node_gains[node][new_proc][2] = 0.0;

                node_change_in_costs[node][new_proc][0] = 0;
                node_change_in_costs[node][new_proc][1] = 0;
                node_change_in_costs[node][new_proc][2] = 0;

                compute_comm_gain(node, current_step, current_proc, new_proc);
                compute_work_gain(node, current_step, current_proc, new_proc);

                if constexpr (current_schedule.use_memory_constraint) {

                    if (not current_schedule.memory_constraint.can_move(
                            node, new_proc, current_schedule.vector_schedule.assignedSuperstep(node))) {
                        node_gains[node][new_proc][1] = std::numeric_limits<double>::lowest();
                    }

                    if (current_schedule.vector_schedule.assignedSuperstep(node) > 0) {
                        if (not current_schedule.memory_constraint.can_move(
                                node, new_proc, current_schedule.vector_schedule.assignedSuperstep(node) - 1)) {
                            node_gains[node][new_proc][0] = std::numeric_limits<double>::lowest();
                        }
                    }
                    if (current_schedule.vector_schedule.assignedSuperstep(node) < current_schedule.num_steps() - 1) {
                        if (not current_schedule.memory_constraint.can_move(
                                node, new_proc, current_schedule.vector_schedule.assignedSuperstep(node) + 1)) {
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

    double compute_max_gain_insert_or_update_heap(VertexType node) {

        double node_max_gain = std::numeric_limits<double>::lowest();
        double node_change_in_cost = 0;
        unsigned node_best_step = 0;
        unsigned node_best_proc = 0;

        double proc_change_in_cost = 0;
        double proc_max = 0;
        unsigned best_step = 0;

        for (unsigned proc = 0; proc < num_procs; proc++) {

            int rand_count = 0;

            if (current_schedule.vector_schedule.assignedSuperstep(node) > 0 &&
                current_schedule.vector_schedule.assignedSuperstep(node) < current_schedule.num_steps() - 1) {

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

            } else if (current_schedule.vector_schedule.assignedSuperstep(node) == 0 &&
                       current_schedule.vector_schedule.assignedSuperstep(node) < current_schedule.num_steps() - 1) {

                if (node_gains[node][proc][2] > node_gains[node][proc][1]) {

                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                } else {

                    proc_max = node_gains[node][proc][1];
                    proc_change_in_cost = node_change_in_costs[node][proc][1];
                    best_step = 1;
                }

            } else if (current_schedule.vector_schedule.assignedSuperstep(node) > 0 &&
                       current_schedule.vector_schedule.assignedSuperstep(node) == current_schedule.num_steps() - 1) {

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
                node_best_step = current_schedule.vector_schedule.assignedSuperstep(node) + best_step - 1;
                node_best_proc = proc;
                rand_count = 0;

            } else if (node_max_gain <= proc_max) {

                if (rand() % (2 + rand_count) == 0) {
                    node_max_gain = proc_max;
                    node_change_in_cost = proc_change_in_cost;
                    node_best_step = current_schedule.vector_schedule.assignedSuperstep(node) + best_step - 1;
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

            // if (node_max_gain < parameters.gain_threshold && node_change_in_cost >
            // parameters.change_in_cost_threshold)
            //     return node_max_gain;

            kl_move<Graph_t> move(
                node, node_max_gain, node_change_in_cost, current_schedule.vector_schedule.assignedProcessor(node),
                current_schedule.vector_schedule.assignedSuperstep(node), node_best_proc, node_best_step);
            node_heap_handles[node] = max_gain_heap.push(move);
        }

        return node_max_gain;
    }

    void compute_work_gain(VertexType node, unsigned current_step, unsigned current_proc, unsigned new_proc) {

        if (current_proc == new_proc) {

            node_gains[node][current_proc][1] = std::numeric_limits<double>::lowest();

        } else {

            if (current_schedule.step_max_work[current_step] ==
                    current_schedule.step_processor_work[current_step][current_proc] &&
                current_schedule.step_processor_work[current_step][current_proc] >
                    current_schedule.step_second_max_work[current_step]) {

                // new max
                const double new_max_work =
                    std::max(current_schedule.step_processor_work[current_step][current_proc] -
                                 current_schedule.instance->getComputationalDag().vertex_work_weight(node),
                             current_schedule.step_second_max_work[current_step]);

                if (current_schedule.step_processor_work[current_step][new_proc] +
                        current_schedule.instance->getComputationalDag().vertex_work_weight(node) >
                    new_max_work) {

                    const double gain =
                        static_cast<double>(current_schedule.step_max_work[current_step]) -
                        (static_cast<double>(current_schedule.step_processor_work[current_step][new_proc]) +
                         static_cast<double>(
                             current_schedule.instance->getComputationalDag().vertex_work_weight(node)));

                    node_gains[node][new_proc][1] += gain;
                    node_change_in_costs[node][new_proc][1] -= gain;

                } else {

                    const double gain = static_cast<double>(current_schedule.step_max_work[current_step]) -
                                        static_cast<double>(new_max_work);

                    node_gains[node][new_proc][1] += gain;
                    node_change_in_costs[node][new_proc][1] -= gain;
                }

            } else {

                if (current_schedule.step_max_work[current_step] <
                    current_schedule.instance->getComputationalDag().vertex_work_weight(node) +
                        current_schedule.step_processor_work[current_step][new_proc]) {

                    const double gain =
                        (static_cast<double>(
                             current_schedule.instance->getComputationalDag().vertex_work_weight(node)) +
                         static_cast<double>(current_schedule.step_processor_work[current_step][new_proc]) -
                         static_cast<double>(current_schedule.step_max_work[current_step]));

                    node_gains[node][new_proc][1] -= gain;
                    node_change_in_costs[node][new_proc][1] += gain;
                }
            }
        }

        if (current_step > 0) {

            if (current_schedule.step_max_work[current_step - 1] <
                current_schedule.step_processor_work[current_step - 1][new_proc] +
                    current_schedule.instance->getComputationalDag().vertex_work_weight(node)) {

                const double gain =
                    static_cast<double>(current_schedule.step_processor_work[current_step - 1][new_proc]) +
                    static_cast<double>(current_schedule.instance->getComputationalDag().vertex_work_weight(node)) -
                    static_cast<double>(current_schedule.step_max_work[current_step - 1]);

                node_gains[node][new_proc][0] -= gain;

                node_change_in_costs[node][new_proc][0] += gain;
            }

            if (current_schedule.step_max_work[current_step] ==
                    current_schedule.step_processor_work[current_step][current_proc] &&
                current_schedule.step_processor_work[current_step][current_proc] >
                    current_schedule.step_second_max_work[current_step]) {

                if (current_schedule.step_max_work[current_step] -
                        current_schedule.instance->getComputationalDag().vertex_work_weight(node) >
                    current_schedule.step_second_max_work[current_step]) {

                    const double gain = current_schedule.instance->getComputationalDag().vertex_work_weight(node);
                    node_gains[node][new_proc][0] += gain;
                    node_change_in_costs[node][new_proc][0] -= gain;

                } else {

                    const double gain = current_schedule.step_max_work[current_step] -
                                        current_schedule.step_second_max_work[current_step];

                    node_gains[node][new_proc][0] += gain;
                    node_change_in_costs[node][new_proc][0] -= gain;
                }
            }

        } else {

            node_gains[node][new_proc][0] = std::numeric_limits<double>::lowest();
        }

        if (current_step < current_schedule.num_steps() - 1) {

            if (current_schedule.step_max_work[current_step + 1] <
                current_schedule.step_processor_work[current_step + 1][new_proc] +
                    current_schedule.instance->getComputationalDag().vertex_work_weight(node)) {

                const double gain =
                    static_cast<double>(current_schedule.step_processor_work[current_step + 1][new_proc]) +
                    static_cast<double>(current_schedule.instance->getComputationalDag().vertex_work_weight(node)) -
                    static_cast<double>(current_schedule.step_max_work[current_step + 1]);

                node_gains[node][new_proc][2] -= gain;
                node_change_in_costs[node][new_proc][2] += gain;
            }

            if (current_schedule.step_max_work[current_step] ==
                    current_schedule.step_processor_work[current_step][current_proc] &&
                current_schedule.step_processor_work[current_step][current_proc] >
                    current_schedule.step_second_max_work[current_step]) {

                if ((current_schedule.step_max_work[current_step] -
                     current_schedule.instance->getComputationalDag().vertex_work_weight(node)) >
                    current_schedule.step_second_max_work[current_step]) {

                    const double gain = current_schedule.instance->getComputationalDag().vertex_work_weight(node);

                    node_gains[node][new_proc][2] += gain;
                    node_change_in_costs[node][new_proc][2] -= gain;

                } else {

                    const double gain = current_schedule.step_max_work[current_step] -
                                        current_schedule.step_second_max_work[current_step];

                    node_gains[node][new_proc][2] += gain;
                    node_change_in_costs[node][new_proc][2] -= gain;
                }
            }
        } else {

            node_gains[node][new_proc][2] = std::numeric_limits<double>::lowest();
        }
    }

    virtual void compute_comm_gain(vertex_idx_t<Graph_t> node, unsigned current_step, unsigned current_proc,
                                   unsigned new_proc) = 0;

    void update_node_gains(const std::unordered_set<VertexType> &nodes) {

        for (const auto &node : nodes) {

            compute_node_gain(node);
            compute_max_gain_insert_or_update_heap(node);
        }
    };

    kl_move<Graph_t> find_best_move() {

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

        std::uniform_int_distribution<unsigned> dis(0, count - 1);
        unsigned i = dis(gen);

        kl_move<Graph_t> best_move = kl_move<Graph_t>((*node_heap_handles[max_nodes[i]]));

        max_gain_heap.erase(node_heap_handles[max_nodes[i]]);
        node_heap_handles.erase(max_nodes[i]);

        return best_move;
    }

    kl_move<Graph_t> compute_best_move(VertexType node) {

        double node_max_gain = std::numeric_limits<double>::lowest();
        double node_change_in_cost = 0;
        unsigned node_best_step = 0;
        unsigned node_best_proc = 0;

        double proc_change_in_cost = 0;
        double proc_max = 0;
        unsigned best_step = 0;
        for (unsigned proc = 0; proc < num_procs; proc++) {

            unsigned rand_count = 0;

            if (current_schedule.vector_schedule.assignedSuperstep(node) > 0 &&
                current_schedule.vector_schedule.assignedSuperstep(node) < current_schedule.num_steps() - 1) {

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

            } else if (current_schedule.vector_schedule.assignedSuperstep(node) == 0 &&
                       current_schedule.vector_schedule.assignedSuperstep(node) < current_schedule.num_steps() - 1) {

                if (node_gains[node][proc][2] > node_gains[node][proc][1]) {

                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                } else {

                    proc_max = node_gains[node][proc][1];
                    proc_change_in_cost = node_change_in_costs[node][proc][1];
                    best_step = 1;
                }

            } else if (current_schedule.vector_schedule.assignedSuperstep(node) > 0 &&
                       current_schedule.vector_schedule.assignedSuperstep(node) == current_schedule.num_steps() - 1) {

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
                node_best_step = current_schedule.vector_schedule.assignedSuperstep(node) + best_step - 1;
                node_best_proc = proc;
                rand_count = 0;

            } else if (node_max_gain <= proc_max) {

                if (rand() % (2 + rand_count) == 0) {
                    node_max_gain = proc_max;
                    node_change_in_cost = proc_change_in_cost;
                    node_best_step = current_schedule.vector_schedule.assignedSuperstep(node) + best_step - 1;
                    node_best_proc = proc;
                    rand_count++;
                }
            }
        }

        return kl_move<Graph_t>(
            node, node_max_gain, node_change_in_cost, current_schedule.vector_schedule.assignedProcessor(node),
            current_schedule.vector_schedule.assignedSuperstep(node), node_best_proc, node_best_step);
    }

    kl_move<Graph_t> best_move_change_superstep(VertexType node) {

        double node_max_gain = std::numeric_limits<double>::lowest();
        double node_change_in_cost = 0;
        unsigned node_best_step = 0;
        unsigned node_best_proc = 0;

        double proc_change_in_cost = 0;
        double proc_max = 0;
        unsigned best_step = 0;
        for (unsigned proc = 0; proc < num_procs; proc++) {

            if (current_schedule.vector_schedule.assignedSuperstep(node) > 0 &&
                current_schedule.vector_schedule.assignedSuperstep(node) < current_schedule.num_steps() - 1) {

                if (node_gains[node][proc][0] > node_gains[node][proc][2]) {
                    proc_max = node_gains[node][proc][0];
                    proc_change_in_cost = node_change_in_costs[node][proc][0];
                    best_step = 0;

                } else {
                    proc_max = node_gains[node][proc][2];
                    proc_change_in_cost = node_change_in_costs[node][proc][2];
                    best_step = 2;
                }

            } else if (current_schedule.vector_schedule.assignedSuperstep(node) == 0 &&
                       current_schedule.vector_schedule.assignedSuperstep(node) < current_schedule.num_steps() - 1) {

                proc_max = node_gains[node][proc][2];
                proc_change_in_cost = node_change_in_costs[node][proc][2];
                best_step = 2;

            } else if (current_schedule.vector_schedule.assignedSuperstep(node) > 0 &&
                       current_schedule.vector_schedule.assignedSuperstep(node) == current_schedule.num_steps() - 1) {

                proc_max = node_gains[node][proc][0];
                proc_change_in_cost = node_change_in_costs[node][proc][0];
                best_step = 0;

            } else {
                throw std::invalid_argument("error lk base best_move_change_superstep");
            }

            if (node_max_gain < proc_max) {

                node_max_gain = proc_max;
                node_change_in_cost = proc_change_in_cost;
                node_best_step = current_schedule.vector_schedule.assignedSuperstep(node) + best_step - 1;
                node_best_proc = proc;
            }
        }

        return kl_move<Graph_t>(
            node, node_max_gain, node_change_in_cost, current_schedule.vector_schedule.assignedProcessor(node),
            current_schedule.vector_schedule.assignedSuperstep(node), node_best_proc, node_best_step);
    }

    void save_best_schedule(const IBspSchedule<Graph_t> &schedule) {

        for (const auto &node : current_schedule.instance->vertices()) {

            best_schedule->setAssignedProcessor(node, schedule.assignedProcessor(node));
            best_schedule->setAssignedSuperstep(node, schedule.assignedSuperstep(node));
        }
        best_schedule->updateNumberOfSupersteps();
    }

    void reverse_move_best_schedule(kl_move<Graph_t> move) {
        best_schedule->setAssignedProcessor(move.node, move.from_proc);
        best_schedule->setAssignedSuperstep(move.node, move.from_step);
    }

    std::unordered_set<VertexType> node_selection;

    void select_nodes() {

        if (parameters.select_all_nodes) {

            for (const auto &node : current_schedule.instance->vertices()) {
                if (super_locked_nodes.find(node) == super_locked_nodes.end())
                    node_selection.insert(node);
            }

        } else {
            select_nodes_threshold(parameters.selection_threshold - super_locked_nodes.size());
        }
    }

    virtual void select_nodes_comm() {

        for (const auto &node : current_schedule.instance->vertices()) {

            if (super_locked_nodes.find(node) != super_locked_nodes.end()) {
                continue;
            }

            for (const auto &source : current_schedule.instance->getComputationalDag().parents(node)) {

                if (current_schedule.vector_schedule.assignedProcessor(node) !=
                    current_schedule.vector_schedule.assignedProcessor(source)) {

                    node_selection.insert(node);
                    break;
                }
            }

            for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

                if (current_schedule.vector_schedule.assignedProcessor(node) !=
                    current_schedule.vector_schedule.assignedProcessor(target)) {

                    node_selection.insert(node);
                    break;
                }
            }
        }
    }

    void select_nodes_threshold(std::size_t threshold) {

        std::uniform_int_distribution<vertex_idx_t<Graph_t>> dis(0, num_nodes - 1);

        while (node_selection.size() < threshold) {

            auto node = dis(gen);

            if (super_locked_nodes.find(node) == super_locked_nodes.end()) {
                node_selection.insert(node);
            }
        }
    }

    void select_nodes_permutation_threshold(std::size_t threshold) {

        std::vector<VertexType> permutation(num_nodes);
        std::iota(std::begin(permutation), std::end(permutation), 0);

        std::shuffle(permutation.begin(), permutation.end(), gen);

        for (std::size_t i = 0; i < threshold; i++) {

            if (super_locked_nodes.find(permutation[i]) == super_locked_nodes.end())
                node_selection.insert(permutation[i]);
        }
    }

    void select_nodes_violations() {

        if (current_schedule.current_violations.empty()) {
            select_nodes();
            return;
        }

        for (const auto &edge : current_schedule.current_violations) {

            const auto &source_v = source(edge, current_schedule.instance->getComputationalDag());
            const auto &target_v = target(edge, current_schedule.instance->getComputationalDag());

            node_selection.insert(source_v);
            node_selection.insert(target_v);

            for (const auto &child : current_schedule.instance->getComputationalDag().children(source_v)) {
                if (child != target_v) {
                    node_selection.insert(child);
                }
            }

            for (const auto &parent : current_schedule.instance->getComputationalDag().parents(source_v)) {
                if (parent != target_v) {
                    node_selection.insert(parent);
                }
            }

            for (const auto &child : current_schedule.instance->getComputationalDag().children(target_v)) {
                if (child != source_v) {
                    node_selection.insert(child);
                }
            }

            for (const auto &parent : current_schedule.instance->getComputationalDag().parents(target_v)) {
                if (parent != source_v) {
                    node_selection.insert(parent);
                }
            }
        }
    }

    void select_nodes_conseque_max_work(bool do_not_select_super_locked_nodes = false) {

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

            if (current_schedule.step_processor_work[step_selection_counter][proc] > max_work_step) {
                second_max_work_step = max_work_step;
                second_max_step = max_step;
                max_work_step = current_schedule.step_processor_work[step_selection_counter][proc];
                max_step = proc;

            } else if (current_schedule.step_processor_work[step_selection_counter][proc] > second_max_work_step) {
                second_max_work_step = current_schedule.step_processor_work[step_selection_counter][proc];
                second_max_step = proc;
            }
        }

        if (current_schedule.set_schedule.step_processor_vertices[step_selection_counter][max_step].size() <
            parameters.selection_threshold * .66) {

            node_selection.insert(
                current_schedule.set_schedule.step_processor_vertices[step_selection_counter][max_step].begin(),
                current_schedule.set_schedule.step_processor_vertices[step_selection_counter][max_step].end());

        } else {

            std::sample(current_schedule.set_schedule.step_processor_vertices[step_selection_counter][max_step].begin(),
                        current_schedule.set_schedule.step_processor_vertices[step_selection_counter][max_step].end(),
                        std::inserter(node_selection, node_selection.end()),
                        static_cast<unsigned>(std::round(parameters.selection_threshold * .66)), gen);
        }

        if (current_schedule.set_schedule.step_processor_vertices[step_selection_counter][second_max_step].size() <
            parameters.selection_threshold * .33) {

            node_selection.insert(
                current_schedule.set_schedule.step_processor_vertices[step_selection_counter][second_max_step].begin(),
                current_schedule.set_schedule.step_processor_vertices[step_selection_counter][second_max_step].end());

        } else {

            std::sample(
                current_schedule.set_schedule.step_processor_vertices[step_selection_counter][second_max_step].begin(),
                current_schedule.set_schedule.step_processor_vertices[step_selection_counter][second_max_step].end(),
                std::inserter(node_selection, node_selection.end()),
                static_cast<unsigned>(std::round(parameters.selection_threshold * .33)), gen);
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
        if (step_selection_counter >= current_schedule.num_steps()) {
            step_selection_counter = 0;
            step_selection_epoch_counter++;
        }
    }

    void select_nodes_check_remove_superstep() {

        if (step_selection_epoch_counter > parameters.max_step_selection_epochs) {

#ifdef KL_DEBUG
            std::cout << "step selection epoch counter exceeded, remove supersteps" << std::endl;
#endif

            select_nodes();
            return;
        }

        for (unsigned step_to_remove = step_selection_counter; step_to_remove < current_schedule.num_steps();
             step_to_remove++) {

#ifdef KL_DEBUG
            std::cout << "checking step to remove " << step_to_remove << " / " << current_schedule.num_steps()
                      << std::endl;
#endif

            if (check_remove_superstep(step_to_remove)) {

#ifdef KL_DEBUG
                std::cout << "trying to remove superstep " << step_to_remove << std::endl;
#endif

                if (scatter_nodes_remove_superstep(step_to_remove)) {

                    for (unsigned proc = 0; proc < num_procs; proc++) {

                        if (step_to_remove < current_schedule.num_steps()) {
                            node_selection.insert(
                                current_schedule.set_schedule.step_processor_vertices[step_to_remove][proc].begin(),
                                current_schedule.set_schedule.step_processor_vertices[step_to_remove][proc].end());
                        }

                        if (step_to_remove > 0) {
                            node_selection.insert(
                                current_schedule.set_schedule.step_processor_vertices[step_to_remove - 1][proc].begin(),
                                current_schedule.set_schedule.step_processor_vertices[step_to_remove - 1][proc].end());
                        }
                    }

                    step_selection_counter = step_to_remove + 1;

                    if (step_selection_counter >= current_schedule.num_steps()) {
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

    unsigned step_selection_counter = 0;
    unsigned step_selection_epoch_counter = 0;

    bool auto_alternate = false;
    bool alternate_reset_remove_superstep = false;
    bool reset_superstep = false;

    virtual bool check_remove_superstep(unsigned step) {

        if (current_schedule.num_steps() <= 2) {
            return false;
        }

        v_workw_t<Graph_t> total_work = 0;

        for (unsigned proc = 0; proc < num_procs; proc++) {

            total_work += current_schedule.step_processor_work[step][proc];
        }

        if (total_work < 2.0 * current_schedule.instance->synchronisationCosts()) {
            return true;
        }
        return false;
    }

    bool scatter_nodes_remove_superstep(unsigned step) {

        assert(step < current_schedule.num_steps());

        std::vector<kl_move<Graph_t>> moves;

        bool abort = false;

        for (unsigned proc = 0; proc < num_procs; proc++) {
            for (const auto &node : current_schedule.set_schedule.step_processor_vertices[step][proc]) {

                compute_node_gain(node);
                moves.push_back(best_move_change_superstep(node));

                if (moves.back().gain == std::numeric_limits<double>::lowest()) {
                    abort = true;
                    break;
                }

                if constexpr (current_schedule.use_memory_constraint) {
                    current_schedule.memory_constraint.apply_move(node, proc, step, moves.back().to_proc,
                                                                  moves.back().to_step);
                }
               
            }

            if (abort) {
                break;
            }
        }

        if (abort) {
            current_schedule.recompute_neighboring_supersteps(step);

#ifdef KL_DEBUG
            BspSchedule<Graph_t> tmp_schedule(current_schedule.set_schedule);
            if (not tmp_schedule.satisfiesMemoryConstraints())
                std::cout << "Mem const violated" << std::endl;
#endif

            return false;
        }

        for (unsigned proc = 0; proc < num_procs; proc++) {
            current_schedule.set_schedule.step_processor_vertices[step][proc].clear();
        }

        for (const auto &move : moves) {

#ifdef KL_DEBUG
            std::cout << "scatter node " << move.node << " to proc " << move.to_proc << " to step " << move.to_step
                      << std::endl;
#endif

            current_schedule.vector_schedule.setAssignedSuperstep(move.node, move.to_step);
            current_schedule.vector_schedule.setAssignedProcessor(move.node, move.to_proc);
            current_schedule.set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
        }

        current_schedule.remove_superstep(step);

#ifdef KL_DEBUG
        BspSchedule<Graph_t> tmp_schedule(current_schedule.set_schedule);
        if (not tmp_schedule.satisfiesMemoryConstraints())
            std::cout << "Mem const violated" << std::endl;
#endif

        return true;
    }

    void select_nodes_check_reset_superstep() {

        if (step_selection_epoch_counter > parameters.max_step_selection_epochs) {

#ifdef KL_DEBUG
            std::cout << "step selection epoch counter exceeded, reset supersteps" << std::endl;
#endif

            select_nodes();
            return;
        }

        for (unsigned step_to_remove = step_selection_counter; step_to_remove < current_schedule.num_steps();
             step_to_remove++) {

#ifdef KL_DEBUG
            std::cout << "checking step to reset " << step_to_remove << " / " << current_schedule.num_steps()
                      << std::endl;
#endif

            if (check_reset_superstep(step_to_remove)) {

#ifdef KL_DEBUG
                std::cout << "trying to reset superstep " << step_to_remove << std::endl;
#endif

                if (scatter_nodes_reset_superstep(step_to_remove)) {

                    for (unsigned proc = 0; proc < num_procs; proc++) {

                        if (step_to_remove < current_schedule.num_steps() - 1) {
                            node_selection.insert(
                                current_schedule.set_schedule.step_processor_vertices[step_to_remove + 1][proc].begin(),
                                current_schedule.set_schedule.step_processor_vertices[step_to_remove + 1][proc].end());
                        }

                        if (step_to_remove > 0) {
                            node_selection.insert(
                                current_schedule.set_schedule.step_processor_vertices[step_to_remove - 1][proc].begin(),
                                current_schedule.set_schedule.step_processor_vertices[step_to_remove - 1][proc].end());
                        }
                    }

                    step_selection_counter = step_to_remove + 1;

                    if (step_selection_counter >= current_schedule.num_steps()) {
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

    virtual bool check_reset_superstep(unsigned step) {

        if (current_schedule.num_steps() <= 2) {
            return false;
        }

        v_workw_t<Graph_t> total_work = 0;
        v_workw_t<Graph_t> max_total_work = 0;
        v_workw_t<Graph_t> min_total_work = std::numeric_limits<v_workw_t<Graph_t>>::max();

        for (unsigned proc = 0; proc < num_procs; proc++) {
            total_work += current_schedule.step_processor_work[step][proc];
            max_total_work = std::max(max_total_work, current_schedule.step_processor_work[step][proc]);
            min_total_work = std::min(min_total_work, current_schedule.step_processor_work[step][proc]);
        }

#ifdef KL_DEBUG

        std::cout << " avg "
                  << static_cast<double>(total_work) /
                         static_cast<double>(current_schedule.instance->numberOfProcessors())
                  << " max " << max_total_work << " min " << min_total_work << std::endl;
#endif

        if (static_cast<double>(total_work) / static_cast<double>(current_schedule.instance->numberOfProcessors()) -
                static_cast<double>(min_total_work) >
            0.1 * static_cast<double>(min_total_work)) {
            return true;
        }

        return false;
    }

    bool scatter_nodes_reset_superstep(unsigned step) {

        assert(step < current_schedule.num_steps());

        std::vector<kl_move<Graph_t>> moves;

        bool abort = false;

        for (unsigned proc = 0; proc < num_procs; proc++) {
            for (const auto &node : current_schedule.set_schedule.step_processor_vertices[step][proc]) {

                compute_node_gain(node);
                moves.push_back(best_move_change_superstep(node));

                if (moves.back().gain == std::numeric_limits<double>::lowest()) {
                    abort = true;
                    break;
                }

                if constexpr (current_schedule.use_memory_constraint) {
                    current_schedule.memory_constraint.apply_forward_move(node, proc, step, moves.back().to_proc,
                                                                          moves.back().to_step);
                }               
            }

            if (abort) {
                break;
            }
        }

        if (abort) {

            current_schedule.recompute_neighboring_supersteps(step);
            return false;
        }

        for (unsigned proc = 0; proc < num_procs; proc++) {
            current_schedule.set_schedule.step_processor_vertices[step][proc].clear();
        }

        for (const auto &move : moves) {

#ifdef KL_DEBUG
            std::cout << "scatter node " << move.node << " to proc " << move.to_proc << " to step " << move.to_step
                      << std::endl;
#endif

            current_schedule.vector_schedule.setAssignedSuperstep(move.node, move.to_step);
            current_schedule.vector_schedule.setAssignedProcessor(move.node, move.to_proc);
            current_schedule.set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
        }

        current_schedule.reset_superstep(step);

        return true;
    }

    void select_unlock_neighbors(VertexType node) {

        for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

            if (check_node_unlocked(target)) {

                node_selection.insert(target);
                nodes_to_update.insert(target);
            }
        }

        for (const auto &source : current_schedule.instance->getComputationalDag().parents(node)) {

            if (check_node_unlocked(source)) {

                node_selection.insert(source);
                nodes_to_update.insert(source);
            }
        }
    }

    void set_parameters() {

        if (num_nodes < 250) {

            parameters.max_outer_iterations = 300;

            parameters.select_all_nodes = true;
            parameters.selection_threshold = num_nodes;

        } else if (num_nodes < 1000) {

            parameters.max_outer_iterations = static_cast<unsigned>(num_nodes / 2);

            parameters.select_all_nodes = true;
            parameters.selection_threshold = num_nodes;

        } else if (num_nodes < 5000) {

            parameters.max_outer_iterations = 4 * static_cast<unsigned>(std::sqrt(num_nodes));

            parameters.selection_threshold = num_nodes / 3;

        } else if (num_nodes < 10000) {

            parameters.max_outer_iterations = 3 * static_cast<unsigned>(std::sqrt(num_nodes));

            parameters.selection_threshold = num_nodes / 3;

        } else if (num_nodes < 50000) {

            parameters.max_outer_iterations = static_cast<unsigned>(std::sqrt(num_nodes));

            parameters.selection_threshold = num_nodes / 5;

        } else if (num_nodes < 100000) {

            parameters.max_outer_iterations = 2 * static_cast<unsigned>(std::log(num_nodes));

            parameters.selection_threshold = num_nodes / 10;

        } else {

            parameters.max_outer_iterations = static_cast<unsigned>(std::min(10000.0, std::log(num_nodes)));

            parameters.selection_threshold = num_nodes / 10;
        }

        if (parameters.quick_pass) {
            parameters.max_outer_iterations = 50;
            parameters.max_no_improvement_iterations = 25;
        }

        if (auto_alternate && current_schedule.instance->getArchitecture().synchronisationCosts() > 10000.0) {
#ifdef KL_DEBUG
            std::cout << "KLBase set parameters, large synchchost: only remove supersets" << std::endl;
#endif
            reset_superstep = false;
            alternate_reset_remove_superstep = false;
        }

#ifdef KL_DEBUG
        if (parameters.select_all_nodes)
            std::cout << "KLBase set parameters, select all nodes" << std::endl;
        else
            std::cout << "KLBase set parameters, selection threshold: " << parameters.selection_threshold << std::endl;
#endif
    }

    virtual void cleanup_datastructures() {

        node_change_in_costs.clear();
        node_gains.clear();

        unlock.clear();

        max_gain_heap.clear();
        node_heap_handles.clear();

        current_schedule.cleanup_superstep_datastructures();
    }

    void reset_run_datastructures() {
        node_selection.clear();
        nodes_to_update.clear();
        locked_nodes.clear();
        super_locked_nodes.clear();
    }

    bool run_local_search_without_violations() {

        penalty = std::numeric_limits<double>::max() * .24;

        double initial_costs = current_schedule.current_cost;

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
            // double best_iter_costs = current_schedule.current_cost;

            unsigned inner_counter = 0;

            while (failed_branches < 3 && inner_counter < parameters.max_inner_iterations && max_gain_heap.size() > 0) {

                inner_counter++;

                const double iter_costs = current_schedule.current_cost;

                kl_move<Graph_t> best_move = find_best_move(); // O(log n)

                if (best_move.gain < -std::numeric_limits<double>::max() * .25) {
                    continue;
                }

                current_schedule.apply_move(best_move); // O(p + log n)

                locked_nodes.insert(best_move.node);

#ifdef KL_DEBUG
                double tmp_costs = current_schedule.current_cost;
                if (tmp_costs != compute_current_costs()) {

                    std::cout << "current costs: " << current_schedule.current_cost
                              << " best move gain: " << best_move.gain
                              << " best move costs: " << best_move.change_in_cost << " tmp cost: " << tmp_costs
                              << std::endl;

                    std::cout << "! costs not equal " << std::endl;
                }
#endif

                if (best_move.change_in_cost > 0 && current_schedule.current_feasible) {

                    if (best_schedule_costs > iter_costs) {
#ifdef KL_DEBUG
                        std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                        best_schedule_costs = iter_costs;
                        save_best_schedule(current_schedule.vector_schedule); // O(n)
                        reverse_move_best_schedule(best_move);
                    }
                }

                compute_nodes_to_update(best_move);

                select_unlock_neighbors(best_move.node);

                update_node_gains(nodes_to_update);

#ifdef KL_DEBUG
                std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                          << best_move.change_in_cost << " from step " << best_move.from_step << " to "
                          << best_move.to_step << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                          << " violations: " << current_schedule.current_violations.size() << " cost "
                          << current_schedule.current_cost << std::endl;
#endif

                // if (not current_schedule.current_feasible) {

                if (current_schedule.current_cost > (1.04 + outer_counter * 0.002) * best_schedule_costs) {

#ifdef KL_DEBUG
                    std::cout << "current cost " << current_schedule.current_cost
                              << " too far away from best schedule costs: " << best_schedule_costs
                              << " rollback to best schedule" << std::endl;
#endif

                    current_schedule.set_current_schedule(*best_schedule);

                    // set_initial_reward_penalty();
                    initialize_gain_heap_unlocked_nodes(node_selection);

                    failed_branches++;
                }
                //}

            } // while

#ifdef KL_DEBUG
            std::cout << "end inner loop current cost: " << current_schedule.current_cost << " with "
                      << current_schedule.current_violations.size()
                      << " violation, best sol cost: " << best_schedule_costs << " with "
                      << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                      << parameters.max_outer_iterations << std::endl;
#endif

            if (current_schedule.current_feasible) {
                if (current_schedule.current_cost <= best_schedule_costs) {
                    save_best_schedule(current_schedule.vector_schedule);
                    best_schedule_costs = current_schedule.current_cost;
                } else {
                    current_schedule.set_current_schedule(*best_schedule);
                }
            } else {
                current_schedule.set_current_schedule(*best_schedule);
            }

            reset_locked_nodes();
            node_selection.clear();
            select_nodes_threshold(parameters.selection_threshold);

            initialize_gain_heap(node_selection);

#ifdef KL_DEBUG
            std::cout << "end of while, current cost " << current_schedule.current_cost << std::endl;
#endif

            if (compute_with_time_limit) {

                auto finish_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

                if (duration > ImprovementScheduler<Graph_t>::timeLimitSeconds) {
                    break;
                }
            }

        } // for

        cleanup_datastructures();

        if (initial_costs > current_schedule.current_cost)
            return true;
        else
            return false;
    }

    bool run_local_search_simple() {

        set_initial_reward_penalty();

        const double initial_costs = current_schedule.current_cost;

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
            double best_iter_costs = current_schedule.current_cost;

            VertexType node_causing_first_violation = 0;

            unsigned inner_counter = 0;

            while (failed_branches < parameters.max_num_failed_branches &&
                   inner_counter < parameters.max_inner_iterations && max_gain_heap.size() > 0) {

                inner_counter++;

                const bool iter_feasible = current_schedule.current_feasible;
                const double iter_costs = current_schedule.current_cost;

                kl_move<Graph_t> best_move = find_best_move(); // O(log n)

                if (best_move.gain < -std::numeric_limits<double>::max() * .25) {
#ifdef KL_DEBUG
                    std::cout << "abort iteration on very negative max gain" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                          << best_move.change_in_cost << " from step " << best_move.from_step << " to "
                          << best_move.to_step << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                          << " violations: " << current_schedule.current_violations.size() << " cost "
                          << current_schedule.current_cost << std::endl;
#endif

                current_schedule.apply_move(best_move); // O(p + log n)

                update_reward_penalty();
                locked_nodes.insert(best_move.node);

#ifdef KL_DEBUG
                double tmp_costs = current_schedule.current_cost;
                if (tmp_costs != compute_current_costs()) {

                    std::cout << "current costs: " << current_schedule.current_cost
                              << " best move gain: " << best_move.gain
                              << " best move costs: " << best_move.change_in_cost << " tmp cost: " << tmp_costs
                              << std::endl;

                    std::cout << "! costs not equal " << std::endl;
                }
#endif

                if (iter_feasible != current_schedule.current_feasible) {

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
                            save_best_schedule(current_schedule.vector_schedule); // O(n)
                            reverse_move_best_schedule(best_move);
                        }

                    } else {
#ifdef KL_DEBUG
                        std::cout << "===> current schedule changed from infeasible to feasible" << std::endl;
#endif
                    }
                } else if (best_move.change_in_cost > 0 && current_schedule.current_feasible) {

                    if (iter_costs < best_schedule_costs) {
#ifdef KL_DEBUG
                        std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                        best_schedule_costs = iter_costs;
                        save_best_schedule(current_schedule.vector_schedule); // O(n)
                        reverse_move_best_schedule(best_move);
                    }
                }

                compute_nodes_to_update(best_move);

                select_unlock_neighbors(best_move.node);

                if (check_violation_locked()) {

                    if (iter_feasible != current_schedule.current_feasible && iter_feasible) {
                        node_causing_first_violation = best_move.node;
                    }
                    super_locked_nodes.insert(node_causing_first_violation);
#ifdef KL_DEBUG
                    std::cout << "abort iteration on locked violation, super locking node "
                              << node_causing_first_violation << std::endl;
#endif
                    break;
                }

                update_node_gains(nodes_to_update);

                if (current_schedule.current_cost > (parameters.max_div_best_sol_base_percent +
                                                     outer_counter * parameters.max_div_best_sol_rate_percent) *
                                                        best_schedule_costs) {

#ifdef KL_DEBUG
                    std::cout << "current cost " << current_schedule.current_cost
                              << " too far away from best schedule costs: " << best_schedule_costs
                              << " rollback to best schedule" << std::endl;
#endif

                    current_schedule.set_current_schedule(*best_schedule);

                    set_initial_reward_penalty();
                    initialize_gain_heap_unlocked_nodes(node_selection);

                    failed_branches++;
                }

            } // while

#ifdef KL_DEBUG
            std::cout << "end inner loop current cost: " << current_schedule.current_cost << " with "
                      << current_schedule.current_violations.size()
                      << " violation, best sol cost: " << best_schedule_costs << " with "
                      << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                      << parameters.max_outer_iterations << std::endl;
#endif

            if (current_schedule.current_feasible) {
                if (current_schedule.current_cost <= best_schedule_costs) {
                    save_best_schedule(current_schedule.vector_schedule);
                    best_schedule_costs = current_schedule.current_cost;
                } else {
                    current_schedule.set_current_schedule(*best_schedule);
                }
            } else {
                current_schedule.set_current_schedule(*best_schedule);
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
            std::cout << "end of while, current cost " << current_schedule.current_cost << std::endl;
#endif

            if (compute_with_time_limit) {
                auto finish_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();
                if (duration > ImprovementScheduler<Graph_t>::timeLimitSeconds) {
                    break;
                }
            }

            if (best_iter_costs <= current_schedule.current_cost) {
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

        if (initial_costs > current_schedule.current_cost)
            return true;
        else
            return false;
    }

    bool run_local_search_remove_supersteps() {

        const double initial_costs = current_schedule.current_cost;

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
            std::cout << "outer iteration " << outer_counter << " current costs: " << current_schedule.current_cost
                      << std::endl;
            if (max_gain_heap.size() == 0) {
                std::cout << "max gain heap empty" << std::endl;
            }
#endif

            unsigned conseq_no_gain_moves_counter = 0;

            unsigned failed_branches = 0;
            double best_iter_costs = current_schedule.current_cost;

            VertexType node_causing_first_violation = 0;

            unsigned inner_counter = 0;

            while (failed_branches < parameters.max_num_failed_branches &&
                   inner_counter < parameters.max_inner_iterations && max_gain_heap.size() > 0) {

                inner_counter++;

                const bool iter_feasible = current_schedule.current_feasible;
                const double iter_costs = current_schedule.current_cost;

                kl_move<Graph_t> best_move = find_best_move(); // O(log n)

                if (best_move.gain < -std::numeric_limits<double>::max() * .25) {
#ifdef KL_DEBUG
                    std::cout << "abort iteration on very negative max gain" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                          << best_move.change_in_cost << " from step " << best_move.from_step << " to "
                          << best_move.to_step << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                          << " violations: " << current_schedule.current_violations.size() << " old cost "
                          << current_schedule.current_cost << " new cost "
                          << current_schedule.current_cost + best_move.change_in_cost << std::endl;
#endif

                current_schedule.apply_move(best_move); // O(p + log n)

                update_reward_penalty();
                locked_nodes.insert(best_move.node);

#ifdef KL_DEBUG
                double tmp_costs = current_schedule.current_cost;
                if (tmp_costs != compute_current_costs()) {

                    std::cout << "current costs: " << current_schedule.current_cost
                              << " best move gain: " << best_move.gain
                              << " best move costs: " << best_move.change_in_cost << " tmp cost: " << tmp_costs
                              << std::endl;

                    std::cout << "! costs not equal " << std::endl;
                }
#endif

                if (iter_feasible != current_schedule.current_feasible) {

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
                            save_best_schedule(current_schedule.vector_schedule); // O(n)
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
                } else if (best_move.change_in_cost > 0 && current_schedule.current_feasible) {

                    if (iter_costs < best_schedule_costs) {
#ifdef KL_DEBUG
                        std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                        best_schedule_costs = iter_costs;
                        save_best_schedule(current_schedule.vector_schedule); // O(n)
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

                    if (iter_feasible != current_schedule.current_feasible && iter_feasible) {
                        node_causing_first_violation = best_move.node;
                    }
                    super_locked_nodes.insert(node_causing_first_violation);
#ifdef KL_DEBUG
                    std::cout << "abort iteration on locked violation, super locking node "
                              << node_causing_first_violation << std::endl;
#endif
                    break;
                }

                update_node_gains(nodes_to_update);

                if (current_schedule.current_cost > (parameters.max_div_best_sol_base_percent +
                                                     outer_counter * parameters.max_div_best_sol_rate_percent) *
                                                        best_schedule_costs) {

#ifdef KL_DEBUG
                    std::cout << "current cost " << current_schedule.current_cost
                              << " too far away from best schedule costs: " << best_schedule_costs
                              << " rollback to best schedule" << std::endl;
#endif

                    current_schedule.set_current_schedule(*best_schedule);

                    set_initial_reward_penalty();
                    initialize_gain_heap_unlocked_nodes(node_selection);

#ifdef KL_DEBUG
                    std::cout << "new current cost " << current_schedule.current_cost << std::endl;
#endif

                    failed_branches++;
                }

            } // while

#ifdef KL_DEBUG
            std::cout << std::setprecision(12) << "end inner loop current cost: " << current_schedule.current_cost
                      << " with " << current_schedule.current_violations.size()
                      << " violation, best sol cost: " << best_schedule_costs << " with "
                      << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                      << parameters.max_outer_iterations << std::endl;
#endif

            if (current_schedule.current_feasible) {
                if (current_schedule.current_cost <= best_schedule_costs) {
                    save_best_schedule(current_schedule.vector_schedule);
                    best_schedule_costs = current_schedule.current_cost;
#ifdef KL_DEBUG
                    std::cout << "KLBase save best schedule with (source node comm) cost "
                              << best_schedule->computeCostsTotalCommunication() << " and number of supersteps "
                              << best_schedule->numberOfSupersteps() << std::endl;
#endif
                } else {
                    current_schedule.set_current_schedule(*best_schedule);
                }
            } else {
                current_schedule.set_current_schedule(*best_schedule);
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
            std::cout << "end of while, current cost " << current_schedule.current_cost << std::endl;
#endif

            if (compute_with_time_limit) {
                auto finish_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();
                if (duration > ImprovementScheduler<Graph_t>::timeLimitSeconds) {
                    break;
                }
            }

            if (best_iter_costs <= current_schedule.current_cost) {

                no_improvement_iter_counter++;

                if (no_improvement_iter_counter > parameters.reset_epoch_counter_threshold) {

                    step_selection_epoch_counter = 0;
                    parameters.reset_epoch_counter_threshold += current_schedule.num_steps();
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

        if (initial_costs > current_schedule.current_cost)
            return true;
        else
            return false;
    }

    bool run_local_search_unlock_delay() {

        const double initial_costs = current_schedule.current_cost;

#ifdef KL_DEBUG_1
        std::cout << "Initial costs " << initial_costs << " with " << best_schedule->numberOfSupersteps() << " supersteps."<< std::endl;
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
            std::cout << "outer iteration " << outer_counter << " current costs: " << current_schedule.current_cost
                      << std::endl;
            if (max_gain_heap.size() == 0) {
                std::cout << "max gain heap empty" << std::endl;
            }
#endif

            // unsigned conseq_no_gain_moves_counter = 0;

            unsigned failed_branches = 0;
            double best_iter_costs = current_schedule.current_cost;

            VertexType node_causing_first_violation = 0;

            unsigned inner_counter = 0;

            while (failed_branches < parameters.max_num_failed_branches &&
                   inner_counter < parameters.max_inner_iterations && max_gain_heap.size() > 0) {

                inner_counter++;

                const bool iter_feasible = current_schedule.current_feasible;
                const double iter_costs = current_schedule.current_cost;
#ifdef KL_DEBUG
                print_heap();
#endif
                kl_move<Graph_t> best_move = find_best_move(); // O(log n)

                if (best_move.gain < -std::numeric_limits<double>::max() * .25) {
#ifdef KL_DEBUG
                    std::cout << "abort iteration on very negative max gain" << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                std::cout << "best move: " << best_move.node << " gain " << best_move.gain << " chng in cost "
                          << best_move.change_in_cost << " from step " << best_move.from_step << " to "
                          << best_move.to_step << ", from proc " << best_move.from_proc << " to " << best_move.to_proc
                          << " violations: " << current_schedule.current_violations.size() << " old cost "
                          << current_schedule.current_cost << " new cost "
                          << current_schedule.current_cost + best_move.change_in_cost << std::endl;

                if constexpr (current_schedule.use_memory_constraint) {
                    std::cout << "memory to step/proc "
                              << current_schedule.memory_constraint
                                     .step_processor_memory[best_move.to_step][best_move.to_proc]
                              << std::endl;
                }

                printSetScheduleWorkMemNodesGrid(std::cout, current_schedule.set_schedule, true);
#endif

                current_schedule.apply_move(best_move); // O(p + log n)

                //             if (best_move.gain <= 0.000000001) {
                //                 conseq_no_gain_moves_counter++;

                //                 if (conseq_no_gain_moves_counter > 15) {

                //                     conseq_no_gain_moves_counter = 0;
                //                     parameters.initial_penalty = 0.0;
                //                     parameters.violations_threshold = 3;
                // #ifdef KL_DEBUG
                //                     std::cout << "more than 15 moves with gain <= 0, set " <<
                //                     parameters.initial_penalty
                //                               << " violations threshold " << parameters.violations_threshold <<
                //                               std::endl;
                // #endif
                //                 }

                //             } else {
                //                 conseq_no_gain_moves_counter = 0;
                //             }

#ifdef KL_DEBUG
                BspSchedule<Graph_t> tmp_schedule(current_schedule.set_schedule);
                if (not tmp_schedule.satisfiesMemoryConstraints())
                    std::cout << "Mem const violated" << std::endl;
#endif

                update_reward_penalty();
                locked_nodes.insert(best_move.node);

#ifdef KL_DEBUG
                double tmp_costs = current_schedule.current_cost;
                if (tmp_costs != compute_current_costs()) {

                    std::cout << "current costs: " << current_schedule.current_cost
                              << " best move gain: " << best_move.gain
                              << " best move costs: " << best_move.change_in_cost << " tmp cost: " << tmp_costs
                              << std::endl;

                    std::cout << "! costs not equal " << std::endl;
                }
#endif

                if (iter_feasible != current_schedule.current_feasible) {

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
                            save_best_schedule(current_schedule.vector_schedule); // O(n)
                            reverse_move_best_schedule(best_move);
#ifdef KL_DEBUG
                            std::cout << "KLBase save best schedule with (source node comm) cost "
                                      << best_schedule->computeTotalCosts() << " and number of supersteps "
                                      << best_schedule->numberOfSupersteps() << std::endl;
#endif
                        }

                    } else {
#ifdef KL_DEBUG
                        std::cout << "===> current schedule changed from infeasible to feasible" << std::endl;
#endif
                    }
                } else if (best_move.change_in_cost > 0 && current_schedule.current_feasible) {

                    if (iter_costs < best_schedule_costs) {
#ifdef KL_DEBUG
                        std::cout << "costs increased .. save best schedule with costs " << iter_costs << std::endl;
#endif
                        best_schedule_costs = iter_costs;
                        save_best_schedule(current_schedule.vector_schedule); // O(n)
                        reverse_move_best_schedule(best_move);
#ifdef KL_DEBUG
                        std::cout << "KLBase save best schedule with (source node comm) cost "
                                  << best_schedule->computeTotalCosts() << " and number of supersteps "
                                  << best_schedule->numberOfSupersteps() << std::endl;
#endif
                    }
                }

#ifdef KL_DEBUG
                std::cout << "Node selection: [";
                for (auto it = node_selection.begin(); it != node_selection.end(); ++it) {
                    std::cout << *it << " ";
                }
                std::cout << "]" << std::endl;

                std::cout << "Locked nodes: [";
                for (auto it = locked_nodes.begin(); it != locked_nodes.end(); ++it) {
                    std::cout << *it << " ";
                }
                std::cout << "]" << std::endl;

                std::cout << "Super locked nodes: [";
                for (auto it = super_locked_nodes.begin(); it != super_locked_nodes.end(); ++it) {
                    std::cout << *it << " ";
                }
                std::cout << "]" << std::endl;

#endif

                compute_nodes_to_update(best_move);

                select_unlock_neighbors(best_move.node);

                if (check_violation_locked()) {

                    if (iter_feasible != current_schedule.current_feasible && iter_feasible) {
                        node_causing_first_violation = best_move.node;
                    }
                    super_locked_nodes.insert(node_causing_first_violation);
#ifdef KL_DEBUG
                    std::cout << "abort iteration on locked violation, super locking node "
                              << node_causing_first_violation << std::endl;
#endif
                    break;
                }

#ifdef KL_DEBUG
                std::cout << "Nodes to update: [";
                for (auto it = nodes_to_update.begin(); it != nodes_to_update.end(); ++it) {
                    std::cout << *it << " ";
                }
                std::cout << "]" << std::endl;
#endif

                update_node_gains(nodes_to_update);

                if (not(current_schedule.current_violations.size() > 4) && not iter_feasible &&
                    not max_gain_heap.empty()) {
                    const auto &iter = max_gain_heap.ordered_begin();
                    if (iter->gain < parameters.gain_threshold) {

                        node_selection.clear();
                        locked_nodes.clear();
                        super_locked_nodes.clear();
                        select_nodes_violations();

                        update_reward_penalty();

                        initialize_gain_heap(node_selection);

#ifdef KL_DEBUG
                        std::cout << "max gain below gain threshold" << std::endl;
#endif
                    }
                }

                if (current_schedule.current_cost > (parameters.max_div_best_sol_base_percent +
                                                     outer_counter * parameters.max_div_best_sol_rate_percent) *
                                                        best_schedule_costs) {

#ifdef KL_DEBUG
                    std::cout << "current cost " << current_schedule.current_cost
                              << " too far away from best schedule costs: " << best_schedule_costs
                              << " rollback to best schedule" << std::endl;
#endif

                    current_schedule.set_current_schedule(*best_schedule);

                    set_initial_reward_penalty();
                    initialize_gain_heap_unlocked_nodes(node_selection);

#ifdef KL_DEBUG
                    std::cout << "new current cost " << current_schedule.current_cost << std::endl;
#endif

                    failed_branches++;
                }

            } // while

#ifdef KL_DEBUG
            std::cout << std::setprecision(12) << "end inner loop current cost: " << current_schedule.current_cost
                      << " with " << current_schedule.current_violations.size()
                      << " violation, best sol cost: " << best_schedule_costs << " with "
                      << best_schedule->numberOfSupersteps() << " supersteps, counter: " << outer_counter << "/"
                      << parameters.max_outer_iterations << std::endl;
#endif

            if (current_schedule.current_feasible) {
                if (current_schedule.current_cost <= best_schedule_costs) {
                    save_best_schedule(current_schedule.vector_schedule);
                    best_schedule_costs = current_schedule.current_cost;
#ifdef KL_DEBUG
                    std::cout << "KLBase save best schedule with (source node comm) cost "
                              << best_schedule->computeTotalCosts() << " and number of supersteps "
                              << best_schedule->numberOfSupersteps() << std::endl;
#endif
                } else {
                    current_schedule.set_current_schedule(*best_schedule);
                }
            } else {
                current_schedule.set_current_schedule(*best_schedule);
            }

            if (compute_with_time_limit) {
                auto finish_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();
                if (duration > ImprovementScheduler<Graph_t>::timeLimitSeconds) {
                    break;
                }
            }

            if (outer_counter > 0 && outer_counter % 30 == 0) {
                super_locked_nodes.clear();
#ifdef KL_DEBUG
                std::cout << "---- reset super locked nodes" << std::endl;
#endif
            }

#ifdef KL_PRINT_SCHEDULE
            if (best_iter_costs > current_schedule.current_cost) {
                print_best_schedule(outer_counter + 1);
            }
#endif

            reset_locked_nodes();

            node_selection.clear();

            // if (reset_superstep) {
            //     select_nodes_check_reset_superstep();
            // } else {
            select_nodes_check_remove_superstep();
            // }

            update_reward_penalty();

            initialize_gain_heap(node_selection);

#ifdef KL_DEBUG
            std::cout << "end of while, current cost " << current_schedule.current_cost << std::endl;
#endif

            if (best_iter_costs <= current_schedule.current_cost) {

                no_improvement_iter_counter++;

                if (no_improvement_iter_counter > parameters.reset_epoch_counter_threshold) {

                    step_selection_epoch_counter = 0;
                    parameters.reset_epoch_counter_threshold += current_schedule.num_steps();
#ifdef KL_DEBUG
                    std::cout << "no improvement for " << no_improvement_iter_counter
                              << " iterations, reset epoc counter. Increase reset threshold to "
                              << parameters.reset_epoch_counter_threshold << std::endl;
#endif
                }

                //             if (no_improvement_iter_counter > 10 && no_improvement_iter_counter % 15 == 0) {

                //                 step_selection_epoch_counter = 0;

                //                 if (alternate_reset_remove_superstep) {
                //                     reset_superstep = !reset_superstep;
                //                 }

                // #ifdef KL_DEBUG
                //                 std::cout << "no improvement for " << no_improvement_iter_counter << " reset
                //                 superstep "
                //                           << reset_superstep << std::endl;
                // #endif
                //             }

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

#ifdef KL_DEBUG
            std::cout << "end of while, current cost " << current_schedule.current_cost << std::endl;
#endif

        } // for

        cleanup_datastructures();

#ifdef KL_DEBUG_1
        std::cout << "kl done, current cost " << best_schedule_costs << " with " << best_schedule->numberOfSupersteps() << " supersteps vs " << initial_costs << " initial costs"
                  << std::endl;
        assert(best_schedule->satisfiesPrecedenceConstraints());
#endif

        if (initial_costs > current_schedule.current_cost)
            return true;
        else
            return false;
    }

    // virtual void checkMergeSupersteps();
    // virtual void checkInsertSuperstep();

    // virtual void insertSuperstep(unsigned step);

    void print_heap() {

        std::cout << "heap current size: " << max_gain_heap.size() << std::endl;
        std::cout << "heap top node " << max_gain_heap.top().node << " gain " << max_gain_heap.top().gain << std::endl;

        unsigned count = 0;
        for (auto it = max_gain_heap.ordered_begin(); it != max_gain_heap.ordered_end(); ++it) {
            std::cout << "node " << it->node << " gain " << it->gain << " to proc " << it->to_proc << " to step "
                      << it->to_step << std::endl;

            if (count++ > 15 || it->gain <= 0.0) {
                break;
            }
        }
    }

    bool compute_with_time_limit = false;

#ifdef KL_PRINT_SCHEDULE
    std::string file_name_write_schedule = "kl_schedule_iter_";
    void print_best_schedule(unsigned iteration);
#endif

  public:
    kl_base(kl_current_schedule<Graph_t, MemoryConstraint_t> &current_schedule_)
        : ImprovementScheduler<Graph_t>(), current_schedule(current_schedule_) {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    virtual ~kl_base() = default;

    virtual RETURN_STATUS improveSchedule(BspSchedule<Graph_t> &schedule) override {

        reset_run_datastructures();

        best_schedule = &schedule;
        current_schedule.instance = &best_schedule->getInstance();

        num_nodes = current_schedule.instance->numberOfVertices();
        num_procs = current_schedule.instance->numberOfProcessors();

        set_parameters();
        initialize_datastructures();

        bool improvement_found = run_local_search_unlock_delay();

        if (improvement_found)
            return RETURN_STATUS::OSP_SUCCESS;
        else
            return RETURN_STATUS::BEST_FOUND;
    }

    virtual RETURN_STATUS improveScheduleWithTimeLimit(BspSchedule<Graph_t> &schedule) override {
        compute_with_time_limit = true;
        return improveSchedule(schedule);
    }

    virtual void set_compute_with_time_limit(bool compute_with_time_limit_) {
        compute_with_time_limit = compute_with_time_limit_;
    }

    virtual std::string getScheduleName() const = 0;

    virtual void set_quick_pass(bool quick_pass_) { parameters.quick_pass = quick_pass_; }

    virtual void set_alternate_reset_remove_superstep(bool alternate_reset_remove_superstep_) {
        auto_alternate = false;
        alternate_reset_remove_superstep = alternate_reset_remove_superstep_;
    }
};

} // namespace osp