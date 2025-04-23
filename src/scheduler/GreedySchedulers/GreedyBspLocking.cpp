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

#include <algorithm>
#include <stdexcept>

#include "scheduler/GreedySchedulers/GreedyBspLocking.hpp"

std::vector<int> GreedyBspLocking::get_longest_path(const ComputationalDag &graph) const {
    std::vector<int> longest_path(graph.numberOfVertices(), 0);

    const std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
        longest_path[*r_iter] = graph.nodeWorkWeight(*r_iter);
        if (graph.numberOfChildren(*r_iter) > 0) {
            int max = 0;
            for (const auto &child : graph.children(*r_iter)) {
                if (max <= longest_path[child])
                    max = longest_path[child];
            }
            longest_path[*r_iter] += max;
        }
    }

    return longest_path;
}

std::pair<int, double> GreedyBspLocking::computeScore(VertexType node, unsigned proc, const BspInstance &instance) {

    int score = 0;
    for (const auto &succ : instance.getComputationalDag().children(node)) {
        if (locked[succ] >= 0 && locked[succ] < instance.numberOfProcessors() && locked[succ] != proc)
            score -= lock_penalty;
    }

    return std::pair<int, double>(score + default_value[node], 0);
};

std::pair<RETURN_STATUS, BspSchedule> GreedyBspLocking::computeSchedule(const BspInstance &instance) {

    init_mem_const_data_structures(instance.getArchitecture());

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    const std::vector<int> path_length = get_longest_path(G);
    int max_path = 1;
    for (int i = 0; i < N; ++i)
        if (path_length[i] > max_path)
            max_path = path_length[i];

    default_value.clear();
    default_value.resize(N, 0);
    for (int i = 0; i < N; ++i)
        default_value[i] = path_length[i] * 20 / max_path;

    max_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);
    max_all_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);

    node_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);
    node_all_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);

    locked_set.clear();
    locked.clear();
    locked.resize(N, -1);

    BspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1),
                         std::vector<unsigned>(instance.numberOfVertices()));

    std::set<VertexType> ready;
    ready_phase.clear();
    ready_phase.resize(N, -1);

    std::vector<std::set<VertexType>> procReady(params_p);
    std::set<VertexType> allReady;

    std::vector<unsigned> nrPredecDone(N, 0);
    std::vector<bool> procFree(params_p, true);
    unsigned free = params_p;

    std::vector<unsigned> nr_ready_nodes_per_type(G.getNumberOfNodeTypes(), 0);
    std::vector<unsigned> nr_procs_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
    for (unsigned proc = 0; proc < params_p; ++proc)
        ++nr_procs_per_type[instance.getArchitecture().processorType(proc)];

    std::set<std::pair<size_t, VertexType>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

    for (const auto &v : G.sourceVertices()) {
        ready.insert(v);
        allReady.insert(v);
        ++nr_ready_nodes_per_type[G.nodeType(v)];
        ready_phase[v] = params_p;

        for (unsigned proc = 0; proc < params_p; ++proc) {
            if (instance.isCompatible(v, proc)) {
                heap_node new_node(v, default_value[v], G.numberOfChildren(v));
                node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
            }
        }
    }

    unsigned supstepIdx = 0;
    bool endSupStep = false;

    while (!ready.empty() || !finishTimes.empty()) {

        if (finishTimes.empty() && endSupStep) {
            for (unsigned proc = 0; proc < params_p; ++proc) {
                procReady[proc].clear();
                max_proc_score_heap[proc].clear();
                node_proc_heap_handles[proc].clear();

                reset_mem_const_datastructures_new_superstep(proc);
            }

            allReady = ready;

            for (int node : locked_set)
                locked[node] = -1;
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
                    heap_node new_node(v, score.first, G.numberOfChildren(v));
                    node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
                }
            }

            ++supstepIdx;

            endSupStep = false;

            finishTimes.emplace(0, std::numeric_limits<VertexType>::max());
        }

        const size_t time = finishTimes.begin()->first;
        const size_t max_finish_time = finishTimes.rbegin()->first;

        // Find new ready jobs
        while (!finishTimes.empty() && finishTimes.begin()->first == time) {

            const VertexType node = finishTimes.begin()->second;
            finishTimes.erase(finishTimes.begin());

            if (node != std::numeric_limits<VertexType>::max()) {
                for (const auto &succ : G.children(node)) {

                    ++nrPredecDone[succ];
                    if (nrPredecDone[succ] == G.numberOfParents(succ)) {
                        ready.insert(succ);
                        ++nr_ready_nodes_per_type[G.nodeType(succ)];

                        bool canAdd = true;
                        for (const auto &pred : G.parents(succ)) {

                            if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) &&
                                schedule.assignedSuperstep(pred) == supstepIdx) {
                                canAdd = false;
                                break;
                            }
                        }

                        if (use_memory_constraint && canAdd) {
                            canAdd = check_can_add(schedule, instance, node, succ, supstepIdx);
                        }

                        if (!instance.isCompatible(succ, schedule.assignedProcessor(node)))
                            canAdd = false;

                        if (canAdd) {
                            procReady[schedule.assignedProcessor(node)].insert(succ);
                            ready_phase[succ] = schedule.assignedProcessor(node);

                            std::pair<int, double> score =
                                computeScore(succ, schedule.assignedProcessor(node), instance);
                            heap_node new_node(succ, score.first, G.numberOfChildren(succ));

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
        if (!CanChooseNode(instance, allReady, procReady, procFree)) {
            endSupStep = true;
        }

        while (CanChooseNode(instance, allReady, procReady, procFree)) {

            VertexType nextNode = std::numeric_limits<VertexType>::max();
            unsigned nextProc = instance.numberOfProcessors();
            Choose(instance, schedule, allReady, procReady, procFree, nextNode, nextProc, endSupStep, supstepIdx,
                   max_finish_time - time);

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
            --nr_ready_nodes_per_type[G.nodeType(nextNode)];
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);

            ready_phase[nextNode] = -1;

            if (use_memory_constraint) {

                std::vector<VertexType> toErase = update_mem_const_datastructure_after_assign(
                    schedule, instance, nextNode, nextProc, supstepIdx, procReady);

                for (const auto &node : toErase) {
                    procReady[nextProc].erase(node);
                    max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][node]);
                    node_proc_heap_handles[nextProc].erase(node);
                    ready_phase[node] = -1;
                }
            }

            finishTimes.emplace(time + G.nodeWorkWeight(nextNode), nextNode);
            procFree[nextProc] = false;
            --free;

            // update auxiliary structures

            for (const auto &succ : G.children(nextNode)) {

                if (locked[succ] >= 0 && locked[succ] < params_p && locked[succ] != nextProc) {
                    for (const auto &parent : G.parents(succ)) {
                        if (ready_phase[parent] >= 0 && ready_phase[parent] < params_p &&
                            ready_phase[parent] != locked[succ]) {
                            (*node_proc_heap_handles[ready_phase[parent]][parent]).score += lock_penalty;
                            max_proc_score_heap[ready_phase[parent]].update(
                                node_proc_heap_handles[ready_phase[parent]][parent]);
                        }
                        if (ready_phase[parent] == params_p) {
                            for (int proc = 0; proc < params_p; ++proc) {
                                if (proc == locked[succ] || !instance.isCompatible(parent, proc))
                                    continue;

                                (*node_all_proc_heap_handles[proc][parent]).score += lock_penalty;
                                max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
                            }
                        }
                    }
                    locked[succ] = params_p;
                } else if (locked[succ] == -1) {
                    locked_set.push_back(succ);
                    locked[succ] = nextProc;

                    for (const auto &parent : G.parents(succ)) {
                        if (ready_phase[parent] >= 0 && ready_phase[parent] < params_p &&
                            ready_phase[parent] != nextProc) {
                            (*node_proc_heap_handles[ready_phase[parent]][parent]).score -= lock_penalty;
                            max_proc_score_heap[ready_phase[parent]].update(
                                node_proc_heap_handles[ready_phase[parent]][parent]);
                        }
                        if (ready_phase[parent] == params_p) {
                            for (int proc = 0; proc < params_p; ++proc) {
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

        if (use_memory_constraint && not check_mem_feasibility(instance, allReady, procReady)) {

            return {ERROR, schedule};
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

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};

// return value currently unused, remains a future improvement option
// (returning false indicates that the best move is still not particularly promising)
bool GreedyBspLocking::Choose(const BspInstance &instance, const BspSchedule &schedule, std::set<VertexType> &allReady,
                              std::vector<std::set<VertexType>> &procReady, const std::vector<bool> &procFree,
                              VertexType &node, unsigned &p, const bool endSupStep, const unsigned current_superstep,
                              const size_t remaining_time) {

    for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {

        if (procFree[proc] && !procReady[proc].empty()) {

            // select node
            heap_node top_node = max_proc_score_heap[proc].top();

            // filling up
            bool procready_empty = false;
            while (endSupStep && (remaining_time < instance.getComputationalDag().nodeWorkWeight(top_node.node))) {
                procReady[proc].erase(top_node.node);
                ready_phase[top_node.node] = -1;
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

    heap_node best_node(instance.numberOfVertices(), -instance.numberOfVertices() - 1, 0.0);

    for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
        if (!procFree[proc] or max_all_proc_score_heap[proc].empty())
            continue;

        heap_node top_node = max_all_proc_score_heap[proc].top();

        // filling up
        bool all_procready_empty = false;
        while (endSupStep && (remaining_time < instance.getComputationalDag().nodeWorkWeight(top_node.node))) {
            allReady.erase(top_node.node);
            for (unsigned proc_del = 0; proc_del < instance.numberOfProcessors(); proc_del++) {
                if (proc_del == proc || !instance.isCompatible(top_node.node, proc_del))
                    continue;
                max_all_proc_score_heap[proc_del].erase(node_all_proc_heap_handles[proc_del][top_node.node]);
                node_all_proc_heap_handles[proc_del].erase(top_node.node);
            }
            max_all_proc_score_heap[proc].pop();
            node_all_proc_heap_handles[proc].erase(top_node.node);
            ready_phase[top_node.node] = -1;
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

            if (use_memory_constraint) {

                if (check_choose_node(schedule, instance, top_node.node, proc, current_superstep)) {
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
};

bool GreedyBspLocking::check_mem_feasibility(const BspInstance &instance, const std::set<VertexType> &allReady,
                                             const std::vector<std::set<VertexType>> &procReady) const {

    if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (!procReady[i].empty()) {

                heap_node top_node = max_proc_score_heap[i].top();

                if (current_proc_persistent_memory[i] + instance.getComputationalDag().nodeMemoryWeight(top_node.node) +
                        std::max(current_proc_transient_memory[i],
                                 instance.getComputationalDag().nodeCommunicationWeight(top_node.node)) <=
                    instance.getArchitecture().memoryBound(i)) {
                    return true;
                }
            }
        }

        if (!allReady.empty())
            for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {

                heap_node top_node = max_all_proc_score_heap[i].top();

                if (current_proc_persistent_memory[i] + instance.getComputationalDag().nodeMemoryWeight(top_node.node) +
                        std::max(current_proc_transient_memory[i],
                                 instance.getComputationalDag().nodeCommunicationWeight(top_node.node)) <=
                    instance.getArchitecture().memoryBound(i)) {
                    return true;
                }
            }

        return false;
    }

    return true;
};

// auxiliary - check if it is possible to assign a node at all
bool GreedyBspLocking::CanChooseNode(const BspInstance &instance, const std::set<VertexType> &allReady,
                                     const std::vector<std::set<VertexType>> &procReady,
                                     const std::vector<bool> &procFree) const {
    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
        if (procFree[i] && !procReady[i].empty())
            return true;

    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
        if (procFree[i] && !max_all_proc_score_heap[i].empty())
            return true;

    return false;
};

// get number of ready nodes that can be run in parallel, to check whether more parallelism is available
// (currently OK for triangular compatibility matrix, otherwise just heuristic - can be changed to matching-based
// solution later)
unsigned GreedyBspLocking::get_nr_parallelizable_nodes(const BspInstance &instance,
                                                       const std::vector<unsigned> &nr_ready_nodes_per_type,
                                                       const std::vector<unsigned> &nr_procs_per_type) const {
    unsigned nr_nodes = 0;

    std::vector<unsigned> ready_nodes_per_type = nr_ready_nodes_per_type;
    std::vector<unsigned> procs_per_type = nr_procs_per_type;
    for (unsigned proc_type = 0; proc_type < instance.getArchitecture().getNumberOfProcessorTypes(); ++proc_type)
        for (unsigned node_type = 0; node_type < instance.getComputationalDag().getNumberOfNodeTypes(); ++node_type)
            if (instance.isCompatibleType(node_type, proc_type)) {
                unsigned matched = std::min(ready_nodes_per_type[node_type], procs_per_type[proc_type]);
                nr_nodes += matched;
                ready_nodes_per_type[node_type] -= matched;
                procs_per_type[proc_type] -= matched;
            }

    return nr_nodes;
}

std::pair<RETURN_STATUS, BspSchedule>
GreedyBspLocking::computeSchedule_with_preassignment(const BspInstance &instance,
                                                     const std::vector<VertexType> &preassign_nodes) {

    init_mem_const_data_structures(instance.getArchitecture());

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    const std::vector<int> path_length = get_longest_path(G);
    int max_path = 0;
    for (int i = 0; i < N; ++i)
        if (path_length[i] > max_path)
            max_path = path_length[i];

    default_value.clear();
    default_value.resize(N, 0);
    for (int i = 0; i < N; ++i)
        default_value[i] = path_length[i] * 20 / max_path;

    max_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);
    max_all_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);

    node_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);
    node_all_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);

    locked_set.clear();
    locked.clear();
    locked.resize(N, -1);

    BspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1),
                         std::vector<unsigned>(instance.numberOfVertices()));

    std::set<VertexType> ready;
    ready_phase.clear();
    ready_phase.resize(N, -1);

    std::vector<std::set<VertexType>> procReady(params_p);
    std::set<VertexType> allReady;

    std::vector<unsigned> nrPredecDone(N, 0);
    std::vector<bool> procFree(params_p, true);
    unsigned free = params_p;

    std::vector<unsigned> nr_ready_nodes_per_type(G.getNumberOfNodeTypes(), 0);
    std::vector<unsigned> nr_procs_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
    for (unsigned proc = 0; proc < params_p; ++proc)
        ++nr_procs_per_type[instance.getArchitecture().processorType(proc)];

    std::set<std::pair<size_t, VertexType>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

    std::vector<VertexType> default_proc_type(params_p, 0);
    for (unsigned proc = 0; proc < params_p; ++proc) {
        default_proc_type[instance.getArchitecture().processorType(proc)] = proc;
    }

    for (const auto &node : preassign_nodes) {
        schedule.setAssignedProcessor(node, default_proc_type[G.nodeType(node)]);
        schedule.setAssignedSuperstep(node, 0);
    }

    std::unordered_set<VertexType> visited;
    std::unordered_set<VertexType> pre_assigned_set(preassign_nodes.begin(), preassign_nodes.end());

    for (const auto &v : G.sourceVertices()) {

        if (pre_assigned_set.find(v) == pre_assigned_set.end()) {

            ready.insert(v);
            allReady.insert(v);
            ++nr_ready_nodes_per_type[G.nodeType(v)];
            ready_phase[v] = params_p;

            for (unsigned proc = 0; proc < params_p; ++proc) {
                if (instance.isCompatible(v, proc)) {
                    heap_node new_node(v, default_value[v], G.numberOfChildren(v));
                    node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
                }
            }
        } else {

            for (const auto &succ : G.children(v)) {

                if (visited.find(succ) == visited.end()) {

                    bool is_ready_ = true;
                    for (const auto &pred : G.parents(succ)) {
                        if (pre_assigned_set.find(pred) == pre_assigned_set.end()) {
                            is_ready_ = false;
                            break;
                        }
                    }

                    visited.insert(succ);

                    if (is_ready_) {

                        ready.insert(v);
                        allReady.insert(v);
                        ++nr_ready_nodes_per_type[G.nodeType(v)];
                        ready_phase[v] = params_p;

                        for (unsigned proc = 0; proc < params_p; ++proc) {
                            if (instance.isCompatible(v, proc)) {
                                heap_node new_node(v, default_value[v], G.numberOfChildren(v));
                                node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
                            }
                        }
                    }
                }
            }
        }
    }

    unsigned supstepIdx = 1;
    bool endSupStep = false;

    while (!ready.empty() || !finishTimes.empty()) {

        if (finishTimes.empty() && endSupStep) {
            for (unsigned proc = 0; proc < params_p; ++proc) {
                procReady[proc].clear();
                max_proc_score_heap[proc].clear();
                node_proc_heap_handles[proc].clear();

                reset_mem_const_datastructures_new_superstep(proc);
            }

            allReady = ready;

            for (int node : locked_set)
                locked[node] = -1;
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
                    heap_node new_node(v, score.first, G.numberOfChildren(v));
                    node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
                }
            }

            ++supstepIdx;

            endSupStep = false;

            finishTimes.emplace(0, std::numeric_limits<VertexType>::max());
        }

        const size_t time = finishTimes.begin()->first;
        const size_t max_finish_time = finishTimes.rbegin()->first;

        // Find new ready jobs
        while (!finishTimes.empty() && finishTimes.begin()->first == time) {

            const VertexType node = finishTimes.begin()->second;
            finishTimes.erase(finishTimes.begin());

            if (node != std::numeric_limits<VertexType>::max()) {
                for (const auto &succ : G.children(node)) {

                    ++nrPredecDone[succ];
                    if (nrPredecDone[succ] == G.numberOfParents(succ)) {
                        ready.insert(succ);
                        ++nr_ready_nodes_per_type[G.nodeType(succ)];

                        bool canAdd = true;
                        for (const auto &pred : G.parents(succ)) {

                            if ((schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) &&
                                schedule.assignedSuperstep(pred) == supstepIdx) || (pre_assigned_set.find(pred) != pre_assigned_set.end())) {
                                canAdd = false;
                                break;
                            }
                        }

                        if (use_memory_constraint && canAdd) {
                            canAdd = check_can_add(schedule, instance, node, succ, supstepIdx);
                        }

                        if (!instance.isCompatible(succ, schedule.assignedProcessor(node)))
                            canAdd = false;

                        if (canAdd) {
                            procReady[schedule.assignedProcessor(node)].insert(succ);
                            ready_phase[succ] = schedule.assignedProcessor(node);

                            std::pair<int, double> score =
                                computeScore(succ, schedule.assignedProcessor(node), instance);
                            heap_node new_node(succ, score.first, G.numberOfChildren(succ));

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
        if (!CanChooseNode(instance, allReady, procReady, procFree)) {
            endSupStep = true;
        }

        while (CanChooseNode(instance, allReady, procReady, procFree)) {

            VertexType nextNode = std::numeric_limits<VertexType>::max();
            unsigned nextProc = instance.numberOfProcessors();
            Choose(instance, schedule, allReady, procReady, procFree, nextNode, nextProc, endSupStep, supstepIdx,
                   max_finish_time - time);

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
            --nr_ready_nodes_per_type[G.nodeType(nextNode)];
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);

            ready_phase[nextNode] = -1;

            if (use_memory_constraint) {

                std::vector<VertexType> toErase = update_mem_const_datastructure_after_assign(
                    schedule, instance, nextNode, nextProc, supstepIdx, procReady);

                for (const auto &node : toErase) {
                    procReady[nextProc].erase(node);
                    max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][node]);
                    node_proc_heap_handles[nextProc].erase(node);
                    ready_phase[node] = -1;
                }
            }

            finishTimes.emplace(time + G.nodeWorkWeight(nextNode), nextNode);
            procFree[nextProc] = false;
            --free;

            // update auxiliary structures

            for (const auto &succ : G.children(nextNode)) {

                if (locked[succ] >= 0 && locked[succ] < params_p && locked[succ] != nextProc) {
                    for (const auto &parent : G.parents(succ)) {
                        if (ready_phase[parent] >= 0 && ready_phase[parent] < params_p &&
                            ready_phase[parent] != locked[succ]) {
                            (*node_proc_heap_handles[ready_phase[parent]][parent]).score += lock_penalty;
                            max_proc_score_heap[ready_phase[parent]].update(
                                node_proc_heap_handles[ready_phase[parent]][parent]);
                        }
                        if (ready_phase[parent] == params_p) {
                            for (int proc = 0; proc < params_p; ++proc) {
                                if (proc == locked[succ] || !instance.isCompatible(parent, proc))
                                    continue;

                                (*node_all_proc_heap_handles[proc][parent]).score += lock_penalty;
                                max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
                            }
                        }
                    }
                    locked[succ] = params_p;
                } else if (locked[succ] == -1) {
                    locked_set.push_back(succ);
                    locked[succ] = nextProc;

                    for (const auto &parent : G.parents(succ)) {
                        if (ready_phase[parent] >= 0 && ready_phase[parent] < params_p &&
                            ready_phase[parent] != nextProc) {
                            (*node_proc_heap_handles[ready_phase[parent]][parent]).score -= lock_penalty;
                            max_proc_score_heap[ready_phase[parent]].update(
                                node_proc_heap_handles[ready_phase[parent]][parent]);
                        }
                        if (ready_phase[parent] == params_p) {
                            for (int proc = 0; proc < params_p; ++proc) {
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

        if (use_memory_constraint && not check_mem_feasibility(instance, allReady, procReady)) {

            return {ERROR, schedule};
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

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};