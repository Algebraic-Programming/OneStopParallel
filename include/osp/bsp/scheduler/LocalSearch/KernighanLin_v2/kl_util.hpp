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

#include <unordered_set>
#include "kl_active_schedule.hpp"


namespace osp {


template<typename cost_t, typename comm_cost_function_t, typename kl_active_schedule_t>
struct reward_penalty_strategy {
    
    kl_active_schedule_t *active_schedule;
    cost_t instance_comm_cost;
    cost_t max_comm_weight;

    unsigned violations_threshold = 0;
    cost_t initial_penalty = 10.0;
    cost_t penalty = 0;
    cost_t reward = 0; 

    void initalize(kl_active_schedule_t & sched, const cost_t max_comm) {
        max_comm_weight = max_comm;
        active_schedule = &sched;
        instance_comm_cost = sched.getInstance().communicationCosts();
        initial_penalty = max_comm_weight * instance_comm_cost;
    }
 
    void init_reward_penalty() {
        penalty = initial_penalty;
        reward = max_comm_weight * instance_comm_cost;
    }

    void update_reward_penalty() {

        const size_t num_violations = active_schedule->get_current_violations().size();

        if (num_violations <= violations_threshold) {
            penalty = initial_penalty;
            reward = 0.0;
        } else {
            violations_threshold = 0;
            penalty = std::log((num_violations)) * max_comm_weight * instance_comm_cost;
            reward = std::sqrt((num_violations + 4)) * max_comm_weight * instance_comm_cost;
        }
    }
};


template<typename VertexType>
struct set_vertex_lock_manger {

    std::unordered_set<VertexType> locked_nodes;

    void lock(VertexType node) {
        locked_nodes.insert(node);
    }

    void unlock(VertexType node) {
        locked_nodes.erase(node);
    }

    bool is_locked(VertexType node) {
        return locked_nodes.find(node) != locked_nodes.end();
    }

    void clear() {
        locked_nodes.clear();
    }

//    bool unlock_node(VertexType node) {

//         if (super_locked_nodes.find(node) == super_locked_nodes.end()) {

//             if (locked_nodes.find(node) == locked_nodes.end()) {
//                 return true;
//             } else if (locked_nodes.find(node) != locked_nodes.end() && unlock[node] > 0) {
//                 unlock[node]--;

//                 locked_nodes.erase(node);

//                 return true;
//             }
//         }
//         return false;
//     }

//     bool check_node_unlocked(VertexType node) {

//         if (super_locked_nodes.find(node) == super_locked_nodes.end() &&
//             locked_nodes.find(node) == locked_nodes.end()) {
//             return true;
//         }
//         return false;
//     };

//    void reset_locked_nodes() {

//         for (const auto &i : locked_nodes) {

//             unlock[i] = parameters.max_num_unlocks;
//         }

//         locked_nodes.clear();
//     }

    // template<typename edge_container>
    // bool check_violation_locked(const edge_container& violations) {

    //     if (violations.empty())
    //         return false;

    //     for (auto &edge : violations) {

    //         const auto &source_v = source(edge, current_schedule.instance->getComputationalDag());
    //         const auto &target_v = target(edge, current_schedule.instance->getComputationalDag());

    //         if (locked_nodes.find(source_v) == locked_nodes.end() ||
    //             locked_nodes.find(target_v) == locked_nodes.end()) {
    //             return false;
    //         }

    //         bool abort = false;
    //         if (locked_nodes.find(source_v) != locked_nodes.end()) {

    //             if (unlock_node(source_v)) {
    //                 nodes_to_update.insert(source_v);
    //                 node_selection.insert(source_v);
    //             } else {
    //                 abort = true;
    //             }
    //         }

    //         if (locked_nodes.find(target_v) != locked_nodes.end()) {

    //             if (unlock_node(target_v)) {
    //                 nodes_to_update.insert(target_v);
    //                 node_selection.insert(target_v);
    //                 abort = false;
    //             }
    //         }

    //         if (abort) {
    //             return true;
    //         }
    //     }

    //     return false;
    // }

};


template<typename Graph_t, typename container_t, typename handle_t>
struct vertex_selection_strategy {

    const Graph_t * graph;
    std::mt19937 * gen;
    std::size_t selection_threshold;

    inline void initialize(const Graph_t & graph_, std::mt19937 & gen_) {
        graph = &graph_;
        gen = &gen_;

        selection_threshold = graph->num_vertices() / 3;
    }

    inline void select_active_nodes(container_t & node_selection) {

        select_nodes_threshold(selection_threshold, node_selection);
    }

    inline void select_nodes_threshold(const std::size_t & threshold, container_t & node_selection) {

        std::uniform_int_distribution<vertex_idx_t<Graph_t>> dis(0, graph->num_vertices() - 1);

        while (node_selection.size() < threshold) {
            node_selection[dis(*gen)] = handle_t();
        }
    }


    // void select_nodes_threshold(std::size_t threshold) {
        // if (parameters.select_all_nodes) {

        //     for (const auto &node : current_schedule.instance->vertices()) {
        //         if (super_locked_nodes.find(node) == super_locked_nodes.end())
        //             node_selection.insert(node);
        //     }

        // } else {
        //     select_nodes_threshold(parameters.selection_threshold - super_locked_nodes.size());
        // }
   // }

//     virtual void select_nodes_comm() {

//         for (const auto &node : current_schedule.instance->vertices()) {

//             if (super_locked_nodes.find(node) != super_locked_nodes.end()) {
//                 continue;
//             }

//             for (const auto &source : current_schedule.instance->getComputationalDag().parents(node)) {

//                 if (current_schedule.vector_schedule.assignedProcessor(node) !=
//                     current_schedule.vector_schedule.assignedProcessor(source)) {

//                     node_selection.insert(node);
//                     break;
//                 }
//             }

//             for (const auto &target : current_schedule.instance->getComputationalDag().children(node)) {

//                 if (current_schedule.vector_schedule.assignedProcessor(node) !=
//                     current_schedule.vector_schedule.assignedProcessor(target)) {

//                     node_selection.insert(node);
//                     break;
//                 }
//             }
//         }
//     }

//     void select_nodes_permutation_threshold(std::size_t threshold) {

//         std::vector<VertexType> permutation(num_nodes);
//         std::iota(std::begin(permutation), std::end(permutation), 0);

//         std::shuffle(permutation.begin(), permutation.end(), gen);

//         for (std::size_t i = 0; i < threshold; i++) {

//             if (super_locked_nodes.find(permutation[i]) == super_locked_nodes.end())
//                 node_selection.insert(permutation[i]);
//         }
//     }

//     void select_nodes_violations() {

//         if (current_schedule.current_violations.empty()) {
//             select_nodes();
//             return;
//         }

//         for (const auto &edge : current_schedule.current_violations) {

//             const auto &source_v = source(edge, current_schedule.instance->getComputationalDag());
//             const auto &target_v = target(edge, current_schedule.instance->getComputationalDag());

//             node_selection.insert(source_v);
//             node_selection.insert(target_v);

//             for (const auto &child : current_schedule.instance->getComputationalDag().children(source_v)) {
//                 if (child != target_v) {
//                     node_selection.insert(child);
//                 }
//             }

//             for (const auto &parent : current_schedule.instance->getComputationalDag().parents(source_v)) {
//                 if (parent != target_v) {
//                     node_selection.insert(parent);
//                 }
//             }

//             for (const auto &child : current_schedule.instance->getComputationalDag().children(target_v)) {
//                 if (child != source_v) {
//                     node_selection.insert(child);
//                 }
//             }

//             for (const auto &parent : current_schedule.instance->getComputationalDag().parents(target_v)) {
//                 if (parent != source_v) {
//                     node_selection.insert(parent);
//                 }
//             }
//         }
//     }

//     void select_nodes_conseque_max_work(bool do_not_select_super_locked_nodes = false) {

//         if (step_selection_epoch_counter > parameters.max_step_selection_epochs) {

// #ifdef KL_DEBUG
//             std::cout << "step selection epoch counter exceeded. conseque max work" << std::endl;
// #endif

//             select_nodes();
//             return;
//         }

//         unsigned max_work_step = 0;
//         unsigned max_step = 0;
//         unsigned second_max_work_step = 0;
//         unsigned second_max_step = 0;

//         for (unsigned proc = 0; proc < num_procs; proc++) {

//             if (current_schedule.step_processor_work[step_selection_counter][proc] > max_work_step) {
//                 second_max_work_step = max_work_step;
//                 second_max_step = max_step;
//                 max_work_step = current_schedule.step_processor_work[step_selection_counter][proc];
//                 max_step = proc;

//             } else if (current_schedule.step_processor_work[step_selection_counter][proc] > second_max_work_step) {
//                 second_max_work_step = current_schedule.step_processor_work[step_selection_counter][proc];
//                 second_max_step = proc;
//             }
//         }

//         if (current_schedule.set_schedule.step_processor_vertices[step_selection_counter][max_step].size() <
//             parameters.selection_threshold * .66) {

//             node_selection.insert(
//                 current_schedule.set_schedule.step_processor_vertices[step_selection_counter][max_step].begin(),
//                 current_schedule.set_schedule.step_processor_vertices[step_selection_counter][max_step].end());

//         } else {

//             std::sample(current_schedule.set_schedule.step_processor_vertices[step_selection_counter][max_step].begin(),
//                         current_schedule.set_schedule.step_processor_vertices[step_selection_counter][max_step].end(),
//                         std::inserter(node_selection, node_selection.end()),
//                         (unsigned)std::round(parameters.selection_threshold * .66), gen);
//         }

//         if (current_schedule.set_schedule.step_processor_vertices[step_selection_counter][second_max_step].size() <
//             parameters.selection_threshold * .33) {

//             node_selection.insert(
//                 current_schedule.set_schedule.step_processor_vertices[step_selection_counter][second_max_step].begin(),
//                 current_schedule.set_schedule.step_processor_vertices[step_selection_counter][second_max_step].end());

//         } else {

//             std::sample(
//                 current_schedule.set_schedule.step_processor_vertices[step_selection_counter][second_max_step].begin(),
//                 current_schedule.set_schedule.step_processor_vertices[step_selection_counter][second_max_step].end(),
//                 std::inserter(node_selection, node_selection.end()),
//                 (unsigned)std::round(parameters.selection_threshold * .33), gen);
//         }

//         if (do_not_select_super_locked_nodes) {
//             for (const auto &node : super_locked_nodes) {
//                 node_selection.erase(node);
//             }
//         }

// #ifdef KL_DEBUG
//         std::cout << "step selection conseque max work, node selection size " << node_selection.size()
//                   << " ... selected nodes assigend to superstep " << step_selection_counter << " and procs " << max_step
//                   << " and " << second_max_step << std::endl;
// #endif

//         step_selection_counter++;
//         if (step_selection_counter >= current_schedule.num_steps()) {
//             step_selection_counter = 0;
//             step_selection_epoch_counter++;
//         }
//     }


//       void select_nodes_check_remove_superstep() {

//         if (step_selection_epoch_counter > parameters.max_step_selection_epochs) {

// #ifdef KL_DEBUG
//             std::cout << "step selection epoch counter exceeded, remove supersteps" << std::endl;
// #endif

//             select_nodes();
//             return;
//         }

//         for (unsigned step_to_remove = step_selection_counter; step_to_remove < current_schedule.num_steps();
//              step_to_remove++) {

// #ifdef KL_DEBUG
//             std::cout << "checking step to remove " << step_to_remove << " / " << current_schedule.num_steps()
//                       << std::endl;
// #endif

//             if (check_remove_superstep(step_to_remove)) {

// #ifdef KL_DEBUG
//                 std::cout << "trying to remove superstep " << step_to_remove << std::endl;
// #endif

//                 if (scatter_nodes_remove_superstep(step_to_remove)) {

//                     for (unsigned proc = 0; proc < num_procs; proc++) {

//                         if (step_to_remove < current_schedule.num_steps()) {
//                             node_selection.insert(
//                                 current_schedule.set_schedule.step_processor_vertices[step_to_remove][proc].begin(),
//                                 current_schedule.set_schedule.step_processor_vertices[step_to_remove][proc].end());
//                         }

//                         if (step_to_remove > 0) {
//                             node_selection.insert(
//                                 current_schedule.set_schedule.step_processor_vertices[step_to_remove - 1][proc].begin(),
//                                 current_schedule.set_schedule.step_processor_vertices[step_to_remove - 1][proc].end());
//                         }
//                     }

//                     step_selection_counter = step_to_remove + 1;

//                     if (step_selection_counter >= current_schedule.num_steps()) {
//                         step_selection_counter = 0;
//                         step_selection_epoch_counter++;
//                     }

//                     parameters.violations_threshold = 0;
//                     super_locked_nodes.clear();
// #ifdef KL_DEBUG
//                     std::cout << "---- reset super locked nodes" << std::endl;
// #endif

//                     return;
//                 }
//             }
//         }

// #ifdef KL_DEBUG
//         std::cout << "no superstep to remove" << std::endl;
// #endif

//         step_selection_epoch_counter++;
//         select_nodes();
//         return;
//     }

//     virtual bool check_remove_superstep(unsigned step) {

//         if (current_schedule.num_steps() <= 2) {
//             return false;
//         }

//         v_workw_t<Graph_t> total_work = 0;

//         for (unsigned proc = 0; proc < num_procs; proc++) {

//             total_work += current_schedule.step_processor_work[step][proc];
//         }

//         if (total_work < 2.0 * current_schedule.instance->synchronisationCosts()) {
//             return true;
//         }
//         return false;
//     }

//     bool scatter_nodes_remove_superstep(unsigned step) {

//         assert(step < current_schedule.num_steps());

//         std::vector<kl_move<Graph_t>> moves;

//         bool abort = false;

//         for (unsigned proc = 0; proc < num_procs; proc++) {
//             for (const auto &node : current_schedule.set_schedule.step_processor_vertices[step][proc]) {

//                 compute_node_gain(node);
//                 moves.push_back(best_move_change_superstep(node));

//                 if (moves.back().gain == std::numeric_limits<double>::lowest()) {
//                     abort = true;
//                     break;
//                 }

//                 if constexpr (current_schedule.use_memory_constraint) {
//                     current_schedule.memory_constraint.apply_move(node, proc, step, moves.back().to_proc,
//                                                                   moves.back().to_step);
//                 }
//             }

//             if (abort) {
//                 break;
//             }
//         }

//         if (abort) {
//             current_schedule.recompute_neighboring_supersteps(step);

// #ifdef KL_DEBUG
//             BspSchedule<Graph_t> tmp_schedule(current_schedule.set_schedule);
//             if (not tmp_schedule.satisfiesMemoryConstraints())
//                 std::cout << "Mem const violated" << std::endl;
// #endif

//             return false;
//         }

//         for (unsigned proc = 0; proc < num_procs; proc++) {
//             current_schedule.set_schedule.step_processor_vertices[step][proc].clear();
//         }

//         for (const auto &move : moves) {

// #ifdef KL_DEBUG
//             std::cout << "scatter node " << move.node << " to proc " << move.to_proc << " to step " << move.to_step
//                       << std::endl;
// #endif

//             current_schedule.vector_schedule.setAssignedSuperstep(move.node, move.to_step);
//             current_schedule.vector_schedule.setAssignedProcessor(move.node, move.to_proc);
//             current_schedule.set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
//         }

//         current_schedule.remove_superstep(step);

// #ifdef KL_DEBUG
//         BspSchedule<Graph_t> tmp_schedule(current_schedule.set_schedule);
//         if (not tmp_schedule.satisfiesMemoryConstraints())
//             std::cout << "Mem const violated" << std::endl;
// #endif

//         return true;
//     }

//     void select_nodes_check_reset_superstep() {

//         if (step_selection_epoch_counter > parameters.max_step_selection_epochs) {

// #ifdef KL_DEBUG
//             std::cout << "step selection epoch counter exceeded, reset supersteps" << std::endl;
// #endif

//             select_nodes();
//             return;
//         }

//         for (unsigned step_to_remove = step_selection_counter; step_to_remove < current_schedule.num_steps();
//              step_to_remove++) {

// #ifdef KL_DEBUG
//             std::cout << "checking step to reset " << step_to_remove << " / " << current_schedule.num_steps()
//                       << std::endl;
// #endif

//             if (check_reset_superstep(step_to_remove)) {

// #ifdef KL_DEBUG
//                 std::cout << "trying to reset superstep " << step_to_remove << std::endl;
// #endif

//                 if (scatter_nodes_reset_superstep(step_to_remove)) {

//                     for (unsigned proc = 0; proc < num_procs; proc++) {

//                         if (step_to_remove < current_schedule.num_steps() - 1) {
//                             node_selection.insert(
//                                 current_schedule.set_schedule.step_processor_vertices[step_to_remove + 1][proc].begin(),
//                                 current_schedule.set_schedule.step_processor_vertices[step_to_remove + 1][proc].end());
//                         }

//                         if (step_to_remove > 0) {
//                             node_selection.insert(
//                                 current_schedule.set_schedule.step_processor_vertices[step_to_remove - 1][proc].begin(),
//                                 current_schedule.set_schedule.step_processor_vertices[step_to_remove - 1][proc].end());
//                         }
//                     }

//                     step_selection_counter = step_to_remove + 1;

//                     if (step_selection_counter >= current_schedule.num_steps()) {
//                         step_selection_counter = 0;
//                         step_selection_epoch_counter++;
//                     }

//                     parameters.violations_threshold = 0;
//                     super_locked_nodes.clear();
// #ifdef KL_DEBUG
//                     std::cout << "---- reset super locked nodes" << std::endl;
// #endif

//                     return;
//                 }
//             }
//         }

// #ifdef KL_DEBUG
//         std::cout << "no superstep to reset" << std::endl;
// #endif

//         step_selection_epoch_counter++;
//         select_nodes();
//         return;
//     }

//     virtual bool check_reset_superstep(unsigned step) {

//         if (current_schedule.num_steps() <= 2) {
//             return false;
//         }

//         v_workw_t<Graph_t> total_work = 0;
//         v_workw_t<Graph_t> max_total_work = 0;
//         v_workw_t<Graph_t> min_total_work = std::numeric_limits<v_workw_t<Graph_t>>::max();

//         for (unsigned proc = 0; proc < num_procs; proc++) {
//             total_work += current_schedule.step_processor_work[step][proc];
//             max_total_work = std::max(max_total_work, current_schedule.step_processor_work[step][proc]);
//             min_total_work = std::min(min_total_work, current_schedule.step_processor_work[step][proc]);
//         }

// #ifdef KL_DEBUG

//         std::cout << " avg "
//                   << static_cast<double>(total_work) /
//                          static_cast<double>(current_schedule.instance->numberOfProcessors())
//                   << " max " << max_total_work << " min " << min_total_work << std::endl;
// #endif

//         if (static_cast<double>(total_work) / static_cast<double>(current_schedule.instance->numberOfProcessors()) -
//                 static_cast<double>(min_total_work) >
//             0.1 * static_cast<double>(min_total_work)) {
//             return true;
//         }

//         return false;
//     }

//     bool scatter_nodes_reset_superstep(unsigned step) {

//         assert(step < current_schedule.num_steps());

//         std::vector<kl_move<Graph_t>> moves;

//         bool abort = false;

//         for (unsigned proc = 0; proc < num_procs; proc++) {
//             for (const auto &node : current_schedule.set_schedule.step_processor_vertices[step][proc]) {

//                 compute_node_gain(node);
//                 moves.push_back(best_move_change_superstep(node));

//                 if (moves.back().gain == std::numeric_limits<double>::lowest()) {
//                     abort = true;
//                     break;
//                 }

//                 if constexpr (current_schedule.use_memory_constraint) {
//                     current_schedule.memory_constraint.apply_forward_move(node, proc, step, moves.back().to_proc,
//                                                                           moves.back().to_step);
//                 }
//             }

//             if (abort) {
//                 break;
//             }
//         }

//         if (abort) {

//             current_schedule.recompute_neighboring_supersteps(step);
//             return false;
//         }

//         for (unsigned proc = 0; proc < num_procs; proc++) {
//             current_schedule.set_schedule.step_processor_vertices[step][proc].clear();
//         }

//         for (const auto &move : moves) {

// #ifdef KL_DEBUG
//             std::cout << "scatter node " << move.node << " to proc " << move.to_proc << " to step " << move.to_step
//                       << std::endl;
// #endif

//             current_schedule.vector_schedule.setAssignedSuperstep(move.node, move.to_step);
//             current_schedule.vector_schedule.setAssignedProcessor(move.node, move.to_proc);
//             current_schedule.set_schedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);
//         }

//         current_schedule.reset_superstep(step);

//         return true;
//     }



};

} // namespace osp