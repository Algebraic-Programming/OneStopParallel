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

std::pair<int, double> GreedyBspLocking::computeScore(VertexType node, unsigned proc,
                                                      const std::vector<std::vector<bool>> &procInHyperedge,
                                                      const BspInstance &instance) {

    int score = 0;
    for (const auto &succ : instance.getComputationalDag().children(node)) {
        if (locked[succ] >= 0 && locked[succ] < instance.numberOfProcessors() && locked[succ] != proc)
            score -= lock_penalty;
    }

    return std::pair<int, double>(score + default_value[node], 0);
};

std::pair<RETURN_STATUS, BspSchedule> GreedyBspLocking::computeSchedule(const BspInstance &instance) {

    if (use_memory_constraint) {

        switch (instance.getArchitecture().getMemoryConstraintType()) {

        case LOCAL:
            current_proc_persistent_memory = std::vector<int>(instance.numberOfProcessors(), 0);
            break;

        case PERSISTENT_AND_TRANSIENT:
            current_proc_persistent_memory = std::vector<int>(instance.numberOfProcessors(), 0);
            current_proc_transient_memory = std::vector<int>(instance.numberOfProcessors(), 0);
            break;

        case GLOBAL:
            throw std::invalid_argument("Global memory constraint not supported");

        case NONE:
            use_memory_constraint = false;
            std::cerr << "Warning: Memory constraint type set to NONE, ignoring memory constraint" << std::endl;
            break;

        default:
            break;
        }
    }

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

    std::vector<std::vector<bool>> procInHyperedge =
        std::vector<std::vector<bool>>(N, std::vector<bool>(params_p, false));

    std::vector<std::set<VertexType>> procReady(params_p);
    std::set<VertexType> allReady;

    std::vector<unsigned> nrPredecDone(N, 0);
    std::vector<bool> procFree(params_p, true);
    unsigned free = params_p;

    std::set<std::pair<size_t, VertexType>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

    for (const auto &v : G.sourceVertices()) {
        ready.insert(v);
        allReady.insert(v);
        ready_phase[v] = params_p;

        for (unsigned proc = 0; proc < params_p; ++proc) {

            heap_node new_node(v, default_value[v], G.numberOfChildren(v));
            node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
        }
    }

    unsigned supstepIdx = 0;
    bool endSupStep = false;
    counter = 0;
    while (!ready.empty() || !finishTimes.empty()) {

        if (finishTimes.empty() && endSupStep) {
            for (unsigned proc = 0; proc < params_p; ++proc) {
                procReady[proc].clear();
                max_proc_score_heap[proc].clear();
                node_proc_heap_handles[proc].clear();

                if (use_memory_constraint && instance.getArchitecture().getMemoryConstraintType() == LOCAL) {
                    current_proc_persistent_memory[proc] = 0;
                }
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

                    std::pair<int, double> score = computeScore(v, proc, procInHyperedge, instance);
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

                        bool canAdd = true;
                        for (const auto &pred : G.parents(succ)) {

                            if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) &&
                                schedule.assignedSuperstep(pred) == supstepIdx) {
                                canAdd = false;
                                break;
                            }
                        }

                        if (use_memory_constraint && canAdd) {

                            if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                                if (current_proc_persistent_memory[schedule.assignedProcessor(node)] +
                                        instance.getComputationalDag().nodeMemoryWeight(succ) >
                                    instance.getArchitecture().memoryBound()) {
                                    canAdd = false;
                                }

                            } else if (instance.getArchitecture().getMemoryConstraintType() ==
                                       PERSISTENT_AND_TRANSIENT) {

                                if (current_proc_persistent_memory[schedule.assignedProcessor(node)] +
                                        instance.getComputationalDag().nodeMemoryWeight(succ) +
                                        std::max(current_proc_transient_memory[schedule.assignedProcessor(node)],
                                                 instance.getComputationalDag().nodeCommunicationWeight(succ)) >
                                    instance.getArchitecture().memoryBound()) {
                                    canAdd = false;
                                }
                            }
                        }

                        if (canAdd) {
                            procReady[schedule.assignedProcessor(node)].insert(succ);
                            ready_phase[succ] = schedule.assignedProcessor(node);

                            std::pair<int, double> score =
                                computeScore(succ, schedule.assignedProcessor(node), procInHyperedge, instance);
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

        /*if (endSupStep)
            continue;*/

        // Assign new jobs to processors
        if (!CanChooseNode(instance, allReady, procReady, procFree)) {
            endSupStep = true;
        }

        while (CanChooseNode(instance, allReady, procReady, procFree)) {

            VertexType nextNode = std::numeric_limits<VertexType>::max();
            unsigned nextProc = instance.numberOfProcessors();
            bool should = Choose(instance, procInHyperedge, allReady, procReady, procFree, nextNode, nextProc,
                                 endSupStep, max_finish_time - time);

            if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == instance.numberOfProcessors()) {
                endSupStep = true;
                break;
            }

            /*if (!should && (float)free < (float)params_p * max_percent_idle_processors) {
                break;
            }*/

            if (ready_phase[nextNode] < params_p) {

                procReady[nextProc].erase(nextNode);

                max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][nextNode]);
                node_proc_heap_handles[nextProc].erase(nextNode);

            } else {

                allReady.erase(nextNode);

                for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                    max_all_proc_score_heap[proc].erase(node_all_proc_heap_handles[proc][nextNode]);
                    node_all_proc_heap_handles[proc].erase(nextNode);
                }
            }

            ready.erase(nextNode);
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);

            ready_phase[nextNode] = -1;
            ++counter;

            if (use_memory_constraint) {

                std::vector<VertexType> toErase;
                if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                    current_proc_persistent_memory[nextProc] +=
                        instance.getComputationalDag().nodeMemoryWeight(nextNode);

                    for (const auto &node : procReady[nextProc]) {
                        if (current_proc_persistent_memory[nextProc] +
                                instance.getComputationalDag().nodeMemoryWeight(node) >
                            instance.getArchitecture().memoryBound()) {
                            toErase.push_back(node);
                        }
                    }

                } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

                    current_proc_persistent_memory[nextProc] +=
                        instance.getComputationalDag().nodeMemoryWeight(nextNode);
                    current_proc_transient_memory[nextProc] =
                        std::max(current_proc_transient_memory[nextProc],
                                 instance.getComputationalDag().nodeCommunicationWeight(nextNode));

                    for (const auto &node : procReady[nextProc]) {
                        if (current_proc_persistent_memory[nextProc] +
                                instance.getComputationalDag().nodeMemoryWeight(node) +
                                std::max(current_proc_transient_memory[nextProc],
                                         instance.getComputationalDag().nodeCommunicationWeight(node)) >
                            instance.getArchitecture().memoryBound()) {
                            toErase.push_back(node);
                        }
                    }
                }

                for (const auto &node : toErase) {
                    procReady[nextProc].erase(node);
                    max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][node]);
                    node_proc_heap_handles[nextProc].erase(node);
                }
            }

            finishTimes.emplace(time + G.nodeWorkWeight(nextNode), nextNode);
            procFree[nextProc] = false;
            --free;

            // update auxiliary structures
            /*procInHyperedge[nextNode][nextProc] = true;

            for (const auto &pred : G.parents(nextNode)) {

                if (procInHyperedge[pred][nextProc]) {
                    continue;
                }

                procInHyperedge[pred][nextProc] = true;

                for (const auto &child : G.children(pred)) {

                    if (child != nextNode && procReady[nextProc].find(child) != procReady[nextProc].end()) {

                        (*node_proc_heap_handles[nextProc][child]).secondary_score +=
                            (double)instance.getComputationalDag().nodeCommunicationWeight(pred) /
                            (double)instance.getComputationalDag().numberOfChildren(pred);
                        max_proc_score_heap[nextProc].update(node_proc_heap_handles[nextProc][child]);
                    }

                    if (child != nextNode && allReady.find(child) != allReady.end()) {

                        (*node_all_proc_heap_handles[nextProc][child]).secondary_score +=
                            (double)instance.getComputationalDag().nodeCommunicationWeight(pred) /
                            (double)instance.getComputationalDag().numberOfChildren(pred);
                        max_all_proc_score_heap[nextProc].update(node_all_proc_heap_handles[nextProc][child]);
                    }
                }
            }*/

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
                                if (proc == locked[succ])
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
                                if (proc == nextProc)
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

        if (allReady.empty() && free > params_p * max_percent_idle_processors &&
            ((!increase_parallelism_in_new_superstep) ||
             ready.size() >= std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
                                      params_p - free + ((unsigned)(0.5 * free))))) {
            endSupStep = true;
        }
    }

    assert(schedule.satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};

bool GreedyBspLocking::Choose(const BspInstance &instance, const std::vector<std::vector<bool>> &procInHyperedge,
                              std::set<VertexType> &allReady, std::vector<std::set<VertexType>> &procReady,
                              const std::vector<bool> &procFree, VertexType &node, unsigned &p, const bool endSupStep,
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
                if (proc_del == proc)
                    continue;
                max_all_proc_score_heap[proc_del].erase(node_all_proc_heap_handles[proc][top_node.node]);
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

                if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                    if (current_proc_persistent_memory[proc] +
                            instance.getComputationalDag().nodeMemoryWeight(top_node.node) <=
                        instance.getArchitecture().memoryBound()) {

                        best_node = top_node;
                        node = top_node.node;
                        p = proc;
                    }

                } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
                    if (current_proc_persistent_memory[proc] +
                            instance.getComputationalDag().nodeMemoryWeight(top_node.node) +
                            std::max(current_proc_transient_memory[proc],
                                     instance.getComputationalDag().nodeCommunicationWeight(top_node.node)) <=
                        instance.getArchitecture().memoryBound()) {

                        best_node = top_node;
                        node = top_node.node;
                        p = proc;
                    }
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

    if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {
        return true;
    } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (!procReady[i].empty()) {

                heap_node top_node = max_proc_score_heap[i].top();

                if (current_proc_persistent_memory[i] + instance.getComputationalDag().nodeMemoryWeight(top_node.node) +
                        std::max(current_proc_transient_memory[i],
                                 instance.getComputationalDag().nodeCommunicationWeight(top_node.node)) <=
                    instance.getArchitecture().memoryBound()) {
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
                    instance.getArchitecture().memoryBound()) {
                    return true;
                }
            }

        return false;
    }
};

// auxiliary - check if it is possible to assign a node at all
bool GreedyBspLocking::CanChooseNode(const BspInstance &instance, const std::set<VertexType> &allReady,
                                     const std::vector<std::set<VertexType>> &procReady,
                                     const std::vector<bool> &procFree) const {
    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
        if (procFree[i] && !procReady[i].empty())
            return true;

    if (!allReady.empty())
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
            if (procFree[i])
                return true;

    return false;
};