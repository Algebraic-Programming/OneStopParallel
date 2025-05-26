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

#include <chrono>
#include <climits>
#include <cmath>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/heap/fibonacci_heap.hpp>

#include "MemoryConstraintModules.hpp"
#include "auxiliary/misc.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

/**
 * @brief The GreedyBspLocking class represents a scheduler that uses a greedy algorithm to compute schedules for
 * BspInstance.
 *
 * This class inherits from the Scheduler class and implements the computeSchedule() and getScheduleName() methods.
 * The computeSchedule() method computes a schedule for a given BspInstance using a greedy algorithm.
 * The getScheduleName() method returns the name of the schedule, which is "BspGreedy" in this case.
 */

template<typename Graph_t, typename MemoryConstraint_t = no_memory_constraint>
class BspLocking : public Scheduler<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "BspLocking can only be used with computational DAGs.");

  private:
    using VertexType = vertex_idx_t<Graph_t>;

    constexpr static bool use_memory_constraint =
        is_memory_constraint_v<MemoryConstraint_t> or is_memory_constraint_schedule_v<MemoryConstraint_t>;

    static_assert(not use_memory_constraint or std::is_same_v<Graph_t, typename MemoryConstraint_t::Graph_impl_t>,
                  "Graph_t must be the same as MemoryConstraint_t::Graph_impl_t.");

    MemoryConstraint_t memory_constraint;

    struct heap_node {

        VertexType node;

        int score;
        unsigned secondary_score;

        heap_node() : node(0), score(0), secondary_score(0) {}
        heap_node(VertexType node, int score, unsigned secondary_score)
            : node(node), score(score), secondary_score(secondary_score) {}

        bool operator<(heap_node const &rhs) const {
            return (score < rhs.score) || (score == rhs.score and secondary_score < rhs.secondary_score) ||
                   (score == rhs.score and secondary_score == rhs.secondary_score and node > rhs.node);
        }
    };

    std::vector<v_workw_t<Graph_t>> get_longest_path(const Graph_t &graph) const {

        std::vector<v_workw_t<Graph_t>> longest_path(graph.num_vertices(), 0);

        const std::vector<VertexType> top_order = GetTopOrder(graph);

        for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
            longest_path[*r_iter] = graph.vertex_work_weight(*r_iter);
            if (graph.out_degree(*r_iter) > 0) {
                v_workw_t<Graph_t> max = 0;
                for (const auto &child : graph.children(*r_iter)) {
                    if (max <= longest_path[child])
                        max = longest_path[child];
                }
                longest_path[*r_iter] += max;
            }
        }

        return longest_path;
    }

    std::vector<boost::heap::fibonacci_heap<heap_node>> max_proc_score_heap;
    std::vector<boost::heap::fibonacci_heap<heap_node>> max_all_proc_score_heap;

    using heap_handle = typename boost::heap::fibonacci_heap<heap_node>::handle_type;

    std::vector<std::unordered_map<VertexType, heap_handle>> node_proc_heap_handles;
    std::vector<std::unordered_map<VertexType, heap_handle>> node_all_proc_heap_handles;

    std::deque<VertexType> locked_set;
    std::vector<unsigned> locked;
    int lock_penalty = 1;
    std::vector<unsigned> ready_phase;

    std::vector<int> default_value;

    double max_percent_idle_processors;
    bool increase_parallelism_in_new_superstep;

    std::pair<int, double> computeScore(VertexType node, unsigned proc, const BspInstance<Graph_t> &instance) {

        int score = 0;
        for (const auto &succ : instance.getComputationalDag().children(node)) {
            if (locked[succ] < instance.numberOfProcessors() && locked[succ] != proc)
                score -= lock_penalty;
        }

        return std::pair<int, double>(score + default_value[node], 0);
    };

    bool check_mem_feasibility(const BspInstance<Graph_t> &instance, const std::set<VertexType> &allReady,
                               const std::vector<std::set<VertexType>> &procReady) const {

        if constexpr (use_memory_constraint) {

            if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

                for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
                    if (!procReady[i].empty()) {

                        heap_node top_node = max_proc_score_heap[i].top();

                        if (memory_constraint.can_add(top_node.node, i)) {
                            return true;
                        }
                    }
                }

                if (!allReady.empty())
                    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {

                        heap_node top_node = max_all_proc_score_heap[i].top();

                        if (memory_constraint.can_add(top_node.node, i)) {
                            return true;
                        }
                    }

                return false;
            }
        }

        return true;
    }

    bool Choose(const BspInstance<Graph_t> &instance, std::set<VertexType> &allReady,
                std::vector<std::set<VertexType>> &procReady, const std::vector<bool> &procFree, VertexType &node,
                unsigned &p, const bool endSupStep, const v_workw_t<Graph_t> remaining_time) {

        for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {

            if (procFree[proc] && !procReady[proc].empty()) {

                // select node
                heap_node top_node = max_proc_score_heap[proc].top();

                // filling up
                bool procready_empty = false;
                while (endSupStep &&
                       (remaining_time < instance.getComputationalDag().vertex_work_weight(top_node.node))) {
                    procReady[proc].erase(top_node.node);
                    ready_phase[top_node.node] = std::numeric_limits<unsigned>::max();
                    max_proc_score_heap[proc].pop();
                    node_proc_heap_handles[proc].erase(top_node.node);
                    if (!procReady[proc].empty()) {
                        top_node = max_proc_score_heap[proc].top();
                    } else {
                        procready_empty = true;
                        break;
                    }
                }
                if (procready_empty) {
                    continue;
                }

                node = top_node.node;
                p = proc;
            }
        }

        if (p < instance.numberOfProcessors())
            return true;

        heap_node best_node(instance.numberOfVertices(), std::numeric_limits<int>::min(), 0);

        for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
            if (!procFree[proc] or max_all_proc_score_heap[proc].empty())
                continue;

            heap_node top_node = max_all_proc_score_heap[proc].top();

            // filling up
            bool all_procready_empty = false;
            while (endSupStep && (remaining_time < instance.getComputationalDag().vertex_work_weight(top_node.node))) {
                allReady.erase(top_node.node);
                for (unsigned proc_del = 0; proc_del < instance.numberOfProcessors(); proc_del++) {
                    if (proc_del == proc || !instance.isCompatible(top_node.node, proc_del))
                        continue;
                    max_all_proc_score_heap[proc_del].erase(node_all_proc_heap_handles[proc_del][top_node.node]);
                    node_all_proc_heap_handles[proc_del].erase(top_node.node);
                }
                max_all_proc_score_heap[proc].pop();
                node_all_proc_heap_handles[proc].erase(top_node.node);
                ready_phase[top_node.node] = std::numeric_limits<unsigned>::max();
                if (!max_all_proc_score_heap[proc].empty()) {
                    top_node = max_all_proc_score_heap[proc].top();
                } else {
                    all_procready_empty = true;
                    break;
                }
            }
            if (all_procready_empty) {
                continue;
            }

            if (best_node < top_node) {

                if constexpr (use_memory_constraint) {

                    if (memory_constraint.can_add(top_node.node, proc)) {

                        best_node = top_node;
                        node = top_node.node;
                        p = proc;
                    }

                } else {

                    best_node = top_node;
                    node = top_node.node;
                    p = proc;
                }
            }
        }
        return (best_node.score > -3);
    }

    bool CanChooseNode(const BspInstance<Graph_t> &instance, const std::vector<std::set<VertexType>> &procReady,
                       const std::vector<bool> &procFree) const {

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
            if (procFree[i] && !procReady[i].empty())
                return true;

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
            if (procFree[i] && !max_all_proc_score_heap[i].empty())
                return true;

        return false;
    }

    unsigned get_nr_parallelizable_nodes(const BspInstance<Graph_t> &instance,
                                         const std::vector<unsigned> &nr_ready_nodes_per_type,
                                         const std::vector<unsigned> &nr_procs_per_type) const {
        unsigned nr_nodes = 0;

        std::vector<unsigned> ready_nodes_per_type = nr_ready_nodes_per_type;
        std::vector<unsigned> procs_per_type = nr_procs_per_type;
        for (unsigned proc_type = 0; proc_type < instance.getArchitecture().getNumberOfProcessorTypes(); ++proc_type)
            for (unsigned node_type = 0; node_type < instance.getComputationalDag().num_vertex_types(); ++node_type)
                if (instance.isCompatibleType(node_type, proc_type)) {
                    unsigned matched = std::min(ready_nodes_per_type[node_type], procs_per_type[proc_type]);
                    nr_nodes += matched;
                    ready_nodes_per_type[node_type] -= matched;
                    procs_per_type[proc_type] -= matched;
                }

        return nr_nodes;
    }

  public:
    /**
     * @brief Default constructor for GreedyBspLocking.
     */
    BspLocking(float max_percent_idle_processors_ = 0.4f, bool increase_parallelism_in_new_superstep_ = true)
        : max_percent_idle_processors(max_percent_idle_processors_),
          increase_parallelism_in_new_superstep(increase_parallelism_in_new_superstep_) {}

    /**
     * @brief Default destructor for GreedyBspLocking.
     */
    virtual ~BspLocking() = default;

    /**
     * @brief Compute a schedule for the given BspInstance.
     *
     * This method computes a schedule for the given BspInstance using a greedy algorithm.
     *
     * @param instance The BspInstance object representing the instance to compute the schedule for.
     * @return A pair containing the return status and the computed BspSchedule.
     */
    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {

        const auto &instance = schedule.getInstance();

        for (const auto &v : instance.getComputationalDag().vertices()) {
            schedule.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
        }

        unsigned supstepIdx = 0;

        if constexpr (is_memory_constraint_v<MemoryConstraint_t>) {
            memory_constraint.initialize(instance);
        } else if constexpr (is_memory_constraint_schedule_v<MemoryConstraint_t>) {
            memory_constraint.initialize(schedule, supstepIdx);
        }

        const auto &N = instance.numberOfVertices();
        const unsigned &params_p = instance.numberOfProcessors();
        const auto &G = instance.getComputationalDag();

        const std::vector<v_workw_t<Graph_t>> path_length = get_longest_path(G);
        v_workw_t<Graph_t> max_path = 1;
        for (const auto &i : instance.vertices())
            if (path_length[i] > max_path)
                max_path = path_length[i];

        default_value.clear();
        default_value.resize(N, 0);
        for (const auto &i : instance.vertices()) {
            //assert(path_length[i] * 20 / max_path <= std::numeric_limits<int>::max());
            default_value[i] = static_cast<int>(path_length[i] * static_cast<v_workw_t<Graph_t>>(20) / max_path);
        }

        max_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);
        max_all_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);

        node_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);
        node_all_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);

        locked_set.clear();
        locked.clear();
        locked.resize(N, std::numeric_limits<unsigned>::max());

        std::set<VertexType> ready;
        ready_phase.clear();
        ready_phase.resize(N, std::numeric_limits<unsigned>::max());

        std::vector<std::set<VertexType>> procReady(params_p);
        std::set<VertexType> allReady;

        std::vector<unsigned> nrPredecDone(N, 0);
        std::vector<bool> procFree(params_p, true);
        unsigned free = params_p;

        std::vector<unsigned> nr_ready_nodes_per_type(G.num_vertex_types(), 0);
        std::vector<unsigned> nr_procs_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
        for (unsigned proc = 0; proc < params_p; ++proc)
            ++nr_procs_per_type[instance.getArchitecture().processorType(proc)];

        std::set<std::pair<v_workw_t<Graph_t>, VertexType>> finishTimes;
        finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

        for (const auto &v : source_vertices_view(G)) {
            ready.insert(v);
            allReady.insert(v);
            ++nr_ready_nodes_per_type[G.vertex_type(v)];
            ready_phase[v] = params_p;

            for (unsigned proc = 0; proc < params_p; ++proc) {
                if (instance.isCompatible(v, proc)) {
                    heap_node new_node(v, default_value[v], static_cast<unsigned>(G.out_degree(v)));
                    node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
                }
            }
        }

        bool endSupStep = false;

        while (!ready.empty() || !finishTimes.empty()) {

            if (finishTimes.empty() && endSupStep) {
                for (unsigned proc = 0; proc < params_p; ++proc) {
                    procReady[proc].clear();
                    max_proc_score_heap[proc].clear();
                    node_proc_heap_handles[proc].clear();

                    if constexpr (use_memory_constraint) {
                        memory_constraint.reset(proc);
                    }
                }

                allReady = ready;

                for (const auto &node : locked_set)
                    locked[node] = std::numeric_limits<unsigned>::max();
                locked_set.clear();

                for (unsigned proc = 0; proc < params_p; ++proc) {
                    max_all_proc_score_heap[proc].clear();
                    node_all_proc_heap_handles[proc].clear();
                }

                for (const auto &v : ready) {
                    ready_phase[v] = params_p;
                    for (unsigned proc = 0; proc < params_p; ++proc) {

                        if (!instance.isCompatible(v, proc))
                            continue;

                        std::pair<int, double> score = computeScore(v, proc, instance);
                        heap_node new_node(v, score.first, static_cast<unsigned>(G.out_degree(v)));
                        node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
                    }
                }

                ++supstepIdx;

                endSupStep = false;

                finishTimes.emplace(0, std::numeric_limits<VertexType>::max());
            }

            const v_workw_t<Graph_t> time = finishTimes.begin()->first;
            const v_workw_t<Graph_t> max_finish_time = finishTimes.rbegin()->first;

            // Find new ready jobs
            while (!finishTimes.empty() && finishTimes.begin()->first == time) {

                const VertexType node = finishTimes.begin()->second;
                finishTimes.erase(finishTimes.begin());

                if (node != std::numeric_limits<VertexType>::max()) {
                    for (const auto &succ : G.children(node)) {

                        ++nrPredecDone[succ];
                        if (nrPredecDone[succ] == G.in_degree(succ)) {
                            ready.insert(succ);
                            ++nr_ready_nodes_per_type[G.vertex_type(succ)];

                            bool canAdd = true;
                            for (const auto &pred : G.parents(succ)) {

                                if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) &&
                                    schedule.assignedSuperstep(pred) == supstepIdx) {
                                    canAdd = false;
                                    break;
                                }
                            }

                            if constexpr (use_memory_constraint) {

                                if (canAdd) {
                                    if (not memory_constraint.can_add(succ, schedule.assignedProcessor(node)))
                                        canAdd = false;
                                }
                            }

                            if (!instance.isCompatible(succ, schedule.assignedProcessor(node)))
                                canAdd = false;

                            if (canAdd) {
                                procReady[schedule.assignedProcessor(node)].insert(succ);
                                ready_phase[succ] = schedule.assignedProcessor(node);

                                std::pair<int, double> score =
                                    computeScore(succ, schedule.assignedProcessor(node), instance);
                                heap_node new_node(succ, score.first, static_cast<unsigned>(G.out_degree(succ)));

                                node_proc_heap_handles[schedule.assignedProcessor(node)][succ] =
                                    max_proc_score_heap[schedule.assignedProcessor(node)].push(new_node);
                            }
                        }
                    }
                    procFree[schedule.assignedProcessor(node)] = true;
                    ++free;
                }
            }

            // Assign new jobs to processors
            if (!CanChooseNode(instance, procReady, procFree)) {
                endSupStep = true;
            }

            while (CanChooseNode(instance, procReady, procFree)) {

                VertexType nextNode = std::numeric_limits<VertexType>::max();
                unsigned nextProc = instance.numberOfProcessors();
                Choose(instance, allReady, procReady, procFree, nextNode, nextProc, endSupStep, max_finish_time - time);

                if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == instance.numberOfProcessors()) {
                    endSupStep = true;
                    break;
                }

                if (ready_phase[nextNode] < params_p) {

                    procReady[nextProc].erase(nextNode);

                    max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][nextNode]);
                    node_proc_heap_handles[nextProc].erase(nextNode);

                } else {

                    allReady.erase(nextNode);

                    for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                        if (instance.isCompatible(nextNode, proc)) {
                            max_all_proc_score_heap[proc].erase(node_all_proc_heap_handles[proc][nextNode]);
                            node_all_proc_heap_handles[proc].erase(nextNode);
                        }
                    }
                }

                ready.erase(nextNode);
                --nr_ready_nodes_per_type[G.vertex_type(nextNode)];
                schedule.setAssignedProcessor(nextNode, nextProc);
                schedule.setAssignedSuperstep(nextNode, supstepIdx);

                ready_phase[nextNode] = std::numeric_limits<unsigned>::max();

                if constexpr (use_memory_constraint) {
                    memory_constraint.add(nextNode, nextProc);

                    std::vector<VertexType> toErase;
                    for (const auto &node : procReady[nextProc]) {
                        if (not memory_constraint.can_add(node, nextProc)) {
                            toErase.push_back(node);
                        }
                    }

                    for (const auto &node : toErase) {
                        procReady[nextProc].erase(node);
                        max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][node]);
                        node_proc_heap_handles[nextProc].erase(node);
                        ready_phase[node] = std::numeric_limits<unsigned>::max();
                    }
                }

                finishTimes.emplace(time + G.vertex_work_weight(nextNode), nextNode);
                procFree[nextProc] = false;
                --free;

                // update auxiliary structures

                for (const auto &succ : G.children(nextNode)) {

                    if (locked[succ] < params_p && locked[succ] != nextProc) {
                        for (const auto &parent : G.parents(succ)) {
                            if (ready_phase[parent] < std::numeric_limits<unsigned>::max() &&
                                ready_phase[parent] < params_p && ready_phase[parent] != locked[succ]) {
                                (*node_proc_heap_handles[ready_phase[parent]][parent]).score += lock_penalty;
                                max_proc_score_heap[ready_phase[parent]].update(
                                    node_proc_heap_handles[ready_phase[parent]][parent]);
                            }
                            if (ready_phase[parent] == params_p) {
                                for (unsigned proc = 0; proc < params_p; ++proc) {
                                    if (proc == locked[succ] || !instance.isCompatible(parent, proc))
                                        continue;

                                    (*node_all_proc_heap_handles[proc][parent]).score += lock_penalty;
                                    max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
                                }
                            }
                        }
                        locked[succ] = params_p;
                    } else if (locked[succ] == std::numeric_limits<unsigned>::max()) {
                        locked_set.push_back(succ);
                        locked[succ] = nextProc;

                        for (const auto &parent : G.parents(succ)) {
                            if (ready_phase[parent] < std::numeric_limits<unsigned>::max() &&
                                ready_phase[parent] < params_p && ready_phase[parent] != nextProc) {
                                (*node_proc_heap_handles[ready_phase[parent]][parent]).score -= lock_penalty;
                                max_proc_score_heap[ready_phase[parent]].update(
                                    node_proc_heap_handles[ready_phase[parent]][parent]);
                            }
                            if (ready_phase[parent] == params_p) {
                                for (unsigned proc = 0; proc < params_p; ++proc) {
                                    if (proc == nextProc || !instance.isCompatible(parent, proc))
                                        continue;

                                    (*node_all_proc_heap_handles[proc][parent]).score -= lock_penalty;
                                    max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
                                }
                            }
                        }
                    }
                }
            }

            if constexpr (use_memory_constraint) {

                if (not check_mem_feasibility(instance, allReady, procReady)) {

                    return ERROR;
                }
            }

            if (free > params_p * max_percent_idle_processors &&
                ((!increase_parallelism_in_new_superstep) ||
                 get_nr_parallelizable_nodes(instance, nr_ready_nodes_per_type, nr_procs_per_type) >=
                     std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
                              params_p - free + ((unsigned)(0.5 * free))))) {
                endSupStep = true;
            }
        }



        assert(schedule.satisfiesPrecedenceConstraints());

        return SUCCESS;
    }

    // std::pair<RETURN_STATUS, BspSchedule<Graph_t>>
    // computeSchedule_with_preassignment(const BspInstance<Graph_t> &instance,
    //                                    const std::vector<VertexType> &preassign_nodes) {

    //     init_mem_const_data_structures(instance.getArchitecture());

    //     const unsigned &N = instance.numberOfVertices();
    //     const unsigned &params_p = instance.numberOfProcessors();
    //     const auto &G = instance.getComputationalDag();

    //     const std::vector<v_workw_t<Graph_t>> path_length = get_longest_path(G);
    //     v_workw_t<Graph_t> max_path = 0;
    //     for (const auto &i : instance.vertices())
    //         if (path_length[i] > max_path)
    //             max_path = path_length[i];

    //     default_value.clear();
    //     default_value.resize(N, 0);
    //     for (const auto &i : instance.vertices())
    //         default_value[i] = path_length[i] * 20 / max_path;

    //     max_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);
    //     max_all_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);

    //     node_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);
    //     node_all_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);

    //     locked_set.clear();
    //     locked.clear();
    //     locked.resize(N, -1);

    //     BspSchedule<Graph_t> schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), ),
    //                          std::vector<unsigned>(instance.numberOfVertices()));

    //     std::set<VertexType> ready;
    //     ready_phase.clear();
    //     ready_phase.resize(N, -1);

    //     std::vector<std::set<VertexType>> procReady(params_p);
    //     std::set<VertexType> allReady;

    //     std::vector<unsigned> nrPredecDone(N, 0);
    //     std::vector<bool> procFree(params_p, true);
    //     unsigned free = params_p;

    //     std::vector<unsigned> nr_ready_nodes_per_type(G.num_vertex_types(), 0);
    //     std::vector<unsigned> nr_procs_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
    //     for (unsigned proc = 0; proc < params_p; ++proc)
    //         ++nr_procs_per_type[instance.getArchitecture().processorType(proc)];

    //     std::set<std::pair<size_t, VertexType>> finishTimes;
    //     finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

    //     std::vector<VertexType> default_proc_type(params_p, 0);
    //     for (unsigned proc = 0; proc < params_p; ++proc) {
    //         default_proc_type[instance.getArchitecture().processorType(proc)] = proc;
    //     }

    //     for (const auto &node : preassign_nodes) {
    //         schedule.setAssignedProcessor(node, default_proc_type[G.vertex_type(node)]);
    //         schedule.setAssignedSuperstep(node, 0);
    //     }

    //     std::unordered_set<VertexType> visited;
    //     std::unordered_set<VertexType> pre_assigned_set(preassign_nodes.begin(), preassign_nodes.end());

    //     for (const auto &v : G.sourceVertices()) {

    //         if (pre_assigned_set.find(v) == pre_assigned_set.end()) {

    //             ready.insert(v);
    //             allReady.insert(v);
    //             ++nr_ready_nodes_per_type[G.vertex_type(v)];
    //             ready_phase[v] = params_p;

    //             for (unsigned proc = 0; proc < params_p; ++proc) {
    //                 if (instance.isCompatible(v, proc)) {
    //                     heap_node new_node(v, default_value[v], G.out_degree(v));
    //                     node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
    //                 }
    //             }
    //         } else {

    //             for (const auto &succ : G.children(v)) {

    //                 if (visited.find(succ) == visited.end()) {

    //                     bool is_ready_ = true;
    //                     for (const auto &pred : G.parents(succ)) {
    //                         if (pre_assigned_set.find(pred) == pre_assigned_set.end()) {
    //                             is_ready_ = false;
    //                             break;
    //                         }
    //                     }

    //                     visited.insert(succ);

    //                     if (is_ready_) {

    //                         ready.insert(v);
    //                         allReady.insert(v);
    //                         ++nr_ready_nodes_per_type[G.vertex_type(v)];
    //                         ready_phase[v] = params_p;

    //                         for (unsigned proc = 0; proc < params_p; ++proc) {
    //                             if (instance.isCompatible(v, proc)) {
    //                                 heap_node new_node(v, default_value[v], G.out_degree(v));
    //                                 node_all_proc_heap_handles[proc][v] =
    //                                 max_all_proc_score_heap[proc].push(new_node);
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     unsigned supstepIdx = 1;
    //     bool endSupStep = false;

    //     while (!ready.empty() || !finishTimes.empty()) {

    //         if (finishTimes.empty() && endSupStep) {
    //             for (unsigned proc = 0; proc < params_p; ++proc) {
    //                 procReady[proc].clear();
    //                 max_proc_score_heap[proc].clear();
    //                 node_proc_heap_handles[proc].clear();

    //                 reset_mem_const_datastructures_new_superstep(proc);
    //             }

    //             allReady = ready;

    //             for (int node : locked_set)
    //                 locked[node] = -1;
    //             locked_set.clear();

    //             for (unsigned proc = 0; proc < params_p; ++proc) {
    //                 max_all_proc_score_heap[proc].clear();
    //                 node_all_proc_heap_handles[proc].clear();
    //             }

    //             for (const auto &v : ready) {
    //                 ready_phase[v] = params_p;
    //                 for (unsigned proc = 0; proc < params_p; ++proc) {

    //                     if (!instance.isCompatible(v, proc))
    //                         continue;

    //                     std::pair<int, double> score = computeScore(v, proc, instance);
    //                     heap_node new_node(v, score.first, G.out_degree(v));
    //                     node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
    //                 }
    //             }

    //             ++supstepIdx;

    //             endSupStep = false;

    //             finishTimes.emplace(0, std::numeric_limits<VertexType>::max());
    //         }

    //         const size_t time = finishTimes.begin()->first;
    //         const size_t max_finish_time = finishTimes.rbegin()->first;

    //         // Find new ready jobs
    //         while (!finishTimes.empty() && finishTimes.begin()->first == time) {

    //             const VertexType node = finishTimes.begin()->second;
    //             finishTimes.erase(finishTimes.begin());

    //             if (node != std::numeric_limits<VertexType>::max()) {
    //                 for (const auto &succ : G.children(node)) {

    //                     ++nrPredecDone[succ];
    //                     if (nrPredecDone[succ] == G.in_degree(succ)) {
    //                         ready.insert(succ);
    //                         ++nr_ready_nodes_per_type[G.vertex_type(succ)];

    //                         bool canAdd = true;
    //                         for (const auto &pred : G.parents(succ)) {

    //                             if ((schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) &&
    //                                  schedule.assignedSuperstep(pred) == supstepIdx) ||
    //                                 (pre_assigned_set.find(pred) != pre_assigned_set.end())) {
    //                                 canAdd = false;
    //                                 break;
    //                             }
    //                         }

    //                         if (use_memory_constraint && canAdd) {
    //                             canAdd = check_can_add(schedule, instance, node, succ, supstepIdx);
    //                         }

    //                         if (!instance.isCompatible(succ, schedule.assignedProcessor(node)))
    //                             canAdd = false;

    //                         if (canAdd) {
    //                             procReady[schedule.assignedProcessor(node)].insert(succ);
    //                             ready_phase[succ] = schedule.assignedProcessor(node);

    //                             std::pair<int, double> score =
    //                                 computeScore(succ, schedule.assignedProcessor(node), instance);
    //                             heap_node new_node(succ, score.first, G.out_degree(succ));

    //                             node_proc_heap_handles[schedule.assignedProcessor(node)][succ] =
    //                                 max_proc_score_heap[schedule.assignedProcessor(node)].push(new_node);
    //                         }
    //                     }
    //                 }
    //                 procFree[schedule.assignedProcessor(node)] = true;
    //                 ++free;
    //             }
    //         }

    //         // Assign new jobs to processors
    //         if (!CanChooseNode(instance, procReady, procFree)) {
    //             endSupStep = true;
    //         }

    //         while (CanChooseNode(instance, procReady, procFree)) {

    //             VertexType nextNode = std::numeric_limits<VertexType>::max();
    //             unsigned nextProc = instance.numberOfProcessors();
    //             Choose(instance, allReady, procReady, procFree, nextNode, nextProc, endSupStep,
    //                    max_finish_time - time);

    //             if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == instance.numberOfProcessors())
    //             {
    //                 endSupStep = true;
    //                 break;
    //             }

    //             if (ready_phase[nextNode] < params_p) {

    //                 procReady[nextProc].erase(nextNode);

    //                 max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][nextNode]);
    //                 node_proc_heap_handles[nextProc].erase(nextNode);

    //             } else {

    //                 allReady.erase(nextNode);

    //                 for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
    //                     if (instance.isCompatible(nextNode, proc)) {
    //                         max_all_proc_score_heap[proc].erase(node_all_proc_heap_handles[proc][nextNode]);
    //                         node_all_proc_heap_handles[proc].erase(nextNode);
    //                     }
    //                 }
    //             }

    //             ready.erase(nextNode);
    //             --nr_ready_nodes_per_type[G.vertex_type(nextNode)];
    //             schedule.setAssignedProcessor(nextNode, nextProc);
    //             schedule.setAssignedSuperstep(nextNode, supstepIdx);

    //             ready_phase[nextNode] = -1;

    //             if (use_memory_constraint) {

    //                 std::vector<VertexType> toErase = update_mem_const_datastructure_after_assign(
    //                     schedule, instance, nextNode, nextProc, supstepIdx, procReady);

    //                 for (const auto &node : toErase) {
    //                     procReady[nextProc].erase(node);
    //                     max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][node]);
    //                     node_proc_heap_handles[nextProc].erase(node);
    //                     ready_phase[node] = -1;
    //                 }
    //             }

    //             finishTimes.emplace(time + G.vertex_work_weight(nextNode), nextNode);
    //             procFree[nextProc] = false;
    //             --free;

    //             // update auxiliary structures

    //             for (const auto &succ : G.children(nextNode)) {

    //                 if (locked[succ] >= 0 && locked[succ] < params_p && locked[succ] != nextProc) {
    //                     for (const auto &parent : G.parents(succ)) {
    //                         if (ready_phase[parent] >= 0 && ready_phase[parent] < params_p &&
    //                             ready_phase[parent] != locked[succ]) {
    //                             (*node_proc_heap_handles[ready_phase[parent]][parent]).score += lock_penalty;
    //                             max_proc_score_heap[ready_phase[parent]].update(
    //                                 node_proc_heap_handles[ready_phase[parent]][parent]);
    //                         }
    //                         if (ready_phase[parent] == params_p) {
    //                             for (int proc = 0; proc < params_p; ++proc) {
    //                                 if (proc == locked[succ] || !instance.isCompatible(parent, proc))
    //                                     continue;

    //                                 (*node_all_proc_heap_handles[proc][parent]).score += lock_penalty;
    //                                 max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
    //                             }
    //                         }
    //                     }
    //                     locked[succ] = params_p;
    //                 } else if (locked[succ] == -1) {
    //                     locked_set.push_back(succ);
    //                     locked[succ] = nextProc;

    //                     for (const auto &parent : G.parents(succ)) {
    //                         if (ready_phase[parent] >= 0 && ready_phase[parent] < params_p &&
    //                             ready_phase[parent] != nextProc) {
    //                             (*node_proc_heap_handles[ready_phase[parent]][parent]).score -= lock_penalty;
    //                             max_proc_score_heap[ready_phase[parent]].update(
    //                                 node_proc_heap_handles[ready_phase[parent]][parent]);
    //                         }
    //                         if (ready_phase[parent] == params_p) {
    //                             for (int proc = 0; proc < params_p; ++proc) {
    //                                 if (proc == nextProc || !instance.isCompatible(parent, proc))
    //                                     continue;

    //                                 (*node_all_proc_heap_handles[proc][parent]).score -= lock_penalty;
    //                                 max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }

    //         if (use_memory_constraint && not check_mem_feasibility(instance, allReady, procReady)) {

    //             return {ERROR, schedule};
    //         }

    //         if (free > params_p * max_percent_idle_processors &&
    //             ((!increase_parallelism_in_new_superstep) ||
    //              get_nr_parallelizable_nodes(instance, nr_ready_nodes_per_type, nr_procs_per_type) >=
    //                  std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
    //                           params_p - free + ((unsigned)(0.5 * free))))) {
    //             endSupStep = true;
    //         }
    //     }

    //     assert(schedule.satisfiesPrecedenceConstraints());

    //     schedule.setAutoCommunicationSchedule();

    //     return {SUCCESS, schedule};
    // }

    /**
     * @brief Get the name of the schedule.
     *
     * This method returns the name of the schedule, which is "BspGreedy" in this case.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override {

        if (use_memory_constraint) {
            return "BspGreedyLockingMemory";
        } else {
            return "BspGreedyLocking";
        }
    }

    void set_max_percent_idle_processors(float max_percent_idle_processors_) {
        max_percent_idle_processors = max_percent_idle_processors_;
    }
};

} // namespace osp