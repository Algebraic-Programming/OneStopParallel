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

#include "scheduler/GreedySchedulers/GreedyBspFillupScheduler.hpp"

double GreedyBspFillupScheduler::computeScore(VertexType node, unsigned proc,
                                              const std::vector<std::vector<bool>> &procInHyperedge,
                                              const BspInstance &instance) {

    double score = 0;
    for (const auto &pred : instance.getComputationalDag().parents(node)) {

        if (procInHyperedge[pred][proc]) {
            score += (double)instance.getComputationalDag().nodeCommunicationWeight(pred) /
                     (double)instance.getComputationalDag().numberOfChildren(pred);
        }
    }
    return score;
};

bool GreedyBspFillupScheduler::check_mem_feasibility(const BspInstance &instance, const std::set<VertexType> &allReady,
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
};

std::pair<RETURN_STATUS, BspSchedule> GreedyBspFillupScheduler::computeSchedule(const BspInstance &instance) {

    if (use_memory_constraint) {

        switch (instance.getArchitecture().getMemoryConstraintType()) {

        case LOCAL: {
            current_proc_persistent_memory = std::vector<int>(instance.numberOfProcessors(), 0);
            break;
        }

        case PERSISTENT_AND_TRANSIENT: {
            current_proc_persistent_memory = std::vector<int>(instance.numberOfProcessors(), 0);
            current_proc_transient_memory = std::vector<int>(instance.numberOfProcessors(), 0);
            break;
        }

        case GLOBAL: {
            throw std::invalid_argument("Global memory constraint not supported");
        }
        case NONE: {
            use_memory_constraint = false;
            std::cerr << "Warning: Memory constraint type set to NONE, ignoring memory constraint" << std::endl;
            break;
        }

        default:
            break;
        }
    }

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    max_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);
    max_all_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);

    node_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);
    node_all_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);

    BspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1),
                         std::vector<unsigned>(instance.numberOfVertices()));

    std::set<VertexType> ready;

    std::vector<std::vector<bool>> procInHyperedge =
        std::vector<std::vector<bool>>(N, std::vector<bool>(params_p, false));

    std::vector<std::set<VertexType>> procReady(params_p);
    std::set<VertexType> allReady;

    std::vector<unsigned> nr_ready_nodes_per_type(G.getNumberOfNodeTypes(), 0);
    std::vector<unsigned> nr_procs_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
    for (unsigned proc = 0; proc < params_p; ++proc)
        ++nr_procs_per_type[instance.getArchitecture().processorType(proc)];

    std::vector<unsigned> nrPredecRemain(N);
    for (VertexType node = 0; node < N; node++) {
        unsigned num_parents = G.numberOfParents(node);
        nrPredecRemain[node] = num_parents;
        if (num_parents == 0) {
            ready.insert(node);
            allReady.insert(node);
            ++nr_ready_nodes_per_type[G.nodeType(node)];

            for (unsigned proc = 0; proc < params_p; ++proc) {
                if(instance.isCompatible(node, proc)){

                    heap_node new_node(node, 0.0);
                    node_all_proc_heap_handles[proc][node] = max_all_proc_score_heap[proc].push(new_node);
                }
            }
        }
    }

    std::vector<bool> procFree(params_p, true);
    unsigned free = params_p;

    std::set<std::pair<size_t, VertexType>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

    unsigned supstepIdx = 0;
    bool endSupStep = false;
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

            for (unsigned proc = 0; proc < params_p; ++proc) {
                max_all_proc_score_heap[proc].clear();
                node_all_proc_heap_handles[proc].clear();
            }

            for (const auto &v : ready) {
                for (unsigned proc = 0; proc < params_p; ++proc) {

                    if(!instance.isCompatible(v, proc))
                        continue;

                    double score = computeScore(v, proc, procInHyperedge, instance);
                    heap_node new_node(v, score);
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

                    nrPredecRemain[succ]--;
                    if (nrPredecRemain[succ] == 0) {
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

                            if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                                if (current_proc_persistent_memory[schedule.assignedProcessor(node)] +
                                        instance.getComputationalDag().nodeMemoryWeight(succ) >
                                    instance.getArchitecture().memoryBound(schedule.assignedProcessor(node))) {
                                    canAdd = false;
                                }

                            } else if (instance.getArchitecture().getMemoryConstraintType() ==
                                       PERSISTENT_AND_TRANSIENT) {

                                if (current_proc_persistent_memory[schedule.assignedProcessor(node)] +
                                        instance.getComputationalDag().nodeMemoryWeight(succ) +
                                        std::max(current_proc_transient_memory[schedule.assignedProcessor(node)],
                                                 instance.getComputationalDag().nodeCommunicationWeight(succ)) >
                                    instance.getArchitecture().memoryBound(schedule.assignedProcessor(node))) {
                                    canAdd = false;
                                }
                            }
                        }

                        if (!instance.isCompatible(succ, schedule.assignedProcessor(node)))
                            canAdd = false;

                        if (canAdd) {
                            procReady[schedule.assignedProcessor(node)].insert(succ);

                            double score =
                                computeScore(succ, schedule.assignedProcessor(node), procInHyperedge, instance);

                            heap_node new_node(succ, score);
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
            Choose(instance, procInHyperedge, allReady, procReady, procFree, nextNode, nextProc, endSupStep,
                   max_finish_time - time);

            if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == instance.numberOfProcessors()) {
                endSupStep = true;
                break;
            }

            if (procReady[nextProc].find(nextNode) != procReady[nextProc].end()) {

                procReady[nextProc].erase(nextNode);

                max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc][nextNode]);
                node_proc_heap_handles[nextProc].erase(nextNode);

            } else {

                allReady.erase(nextNode);

                for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                    if(instance.isCompatible(nextNode, proc)) {
                        max_all_proc_score_heap[proc].erase(node_all_proc_heap_handles[proc][nextNode]);
                        node_all_proc_heap_handles[proc].erase(nextNode);
                    }
                }
            }

            ready.erase(nextNode);
            --nr_ready_nodes_per_type[G.nodeType(nextNode)];
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);

            if (use_memory_constraint) {

                std::vector<VertexType> toErase;
                if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                    current_proc_persistent_memory[nextProc] +=
                        instance.getComputationalDag().nodeMemoryWeight(nextNode);

                    for (const auto &node : procReady[nextProc]) {
                        if (current_proc_persistent_memory[nextProc] +
                                instance.getComputationalDag().nodeMemoryWeight(node) >
                            instance.getArchitecture().memoryBound(nextProc)) {
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
                            instance.getArchitecture().memoryBound(nextProc)) {
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

            // update comm auxiliary structure
            procInHyperedge[nextNode][nextProc] = true;

            for (const auto &pred : G.parents(nextNode)) {
                // for (const int i : G.In[nextNode]) {

                if (procInHyperedge[pred][nextProc]) {
                    continue;
                }

                procInHyperedge[pred][nextProc] = true;

                for (const auto &child : G.children(pred)) {

                    if (child != nextNode && procReady[nextProc].find(child) != procReady[nextProc].end()) {

                        (*node_proc_heap_handles[nextProc][child]).score +=
                            (double)instance.getComputationalDag().nodeCommunicationWeight(pred) /
                            (double)instance.getComputationalDag().numberOfChildren(pred);
                        max_proc_score_heap[nextProc].update(node_proc_heap_handles[nextProc][child]);
                    }

                    if (child != nextNode && allReady.find(child) != allReady.end() && instance.isCompatible(child, nextProc)) {

                        (*node_all_proc_heap_handles[nextProc][child]).score +=
                            (double)instance.getComputationalDag().nodeCommunicationWeight(pred) /
                            (double)instance.getComputationalDag().numberOfChildren(pred);
                        max_all_proc_score_heap[nextProc].update(node_all_proc_heap_handles[nextProc][child]);
                    }
                }
            }
        }

        if (use_memory_constraint && not check_mem_feasibility(instance, allReady, procReady)) {

            return {ERROR, schedule};
        }

        if (allReady.empty() && free > params_p * max_percent_idle_processors &&
            ((!increase_parallelism_in_new_superstep) ||
             get_nr_parallelizable_nodes(instance, nr_ready_nodes_per_type, nr_procs_per_type) >= std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
                                      params_p - free + ((unsigned)(0.5 * free))))) {
            endSupStep = true;
        }
    }

    assert(schedule.satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};

void GreedyBspFillupScheduler::Choose(const BspInstance &instance,
                                      const std::vector<std::vector<bool>> &procInHyperedge,
                                      std::set<VertexType> &allReady, std::vector<std::set<VertexType>> &procReady,
                                      const std::vector<bool> &procFree, VertexType &node, unsigned &p,
                                      const bool endSupStep, const size_t remaining_time) {

    double max_score = -1.0;

    for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {

        if (procFree[proc] && !procReady[proc].empty()) {

            // select node
            heap_node top_node = max_proc_score_heap[proc].top();
            bool procready_empty = false;
            while (endSupStep && (remaining_time < instance.getComputationalDag().nodeWorkWeight(top_node.node))) {
                procReady[proc].erase(top_node.node);
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

            if (top_node.score > max_score) {
                max_score = top_node.score;
                node = top_node.node;
                p = proc;
                return;
            }
        }
    }

    for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
        if (!procFree[proc] or max_all_proc_score_heap[proc].empty())
            continue;

        heap_node top_node = max_all_proc_score_heap[proc].top();
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

        if (top_node.score > max_score) {

            if (use_memory_constraint) {

                if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                    if (current_proc_persistent_memory[proc] +
                            instance.getComputationalDag().nodeMemoryWeight(top_node.node) <=
                        instance.getArchitecture().memoryBound(proc)) {

                        max_score = top_node.score;
                        node = top_node.node;
                        p = proc;
                    }

                } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
                    if (current_proc_persistent_memory[proc] +
                            instance.getComputationalDag().nodeMemoryWeight(top_node.node) +
                            std::max(current_proc_transient_memory[proc],
                                     instance.getComputationalDag().nodeCommunicationWeight(top_node.node)) <=
                        instance.getArchitecture().memoryBound(proc)) {

                        max_score = top_node.score;
                        node = top_node.node;
                        p = proc;
                    }
                }

            } else {

                max_score = top_node.score;
                node = top_node.node;
                p = proc;
            }
        }
    }
};

// auxiliary - check if it is possible to assign a node at all
bool GreedyBspFillupScheduler::CanChooseNode(const BspInstance &instance, const std::set<VertexType> &allReady,
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

// get number of ready nodes that can be run in parallel, to check whether more parallelism is available
// (currently OK for triangular compatibility matrix, otherwise just heuristic - can be changed to matching-based solution later)
unsigned GreedyBspFillupScheduler::get_nr_parallelizable_nodes(const BspInstance &instance,
                                                                const std::vector<unsigned>& nr_ready_nodes_per_type,
                                                                const std::vector<unsigned>& nr_procs_per_type) const {
    unsigned nr_nodes = 0;

    std::vector<unsigned> ready_nodes_per_type = nr_ready_nodes_per_type;
    std::vector<unsigned> procs_per_type = nr_procs_per_type;
    for(unsigned proc_type = 0; proc_type < instance.getArchitecture().getNumberOfProcessorTypes(); ++proc_type)
        for(unsigned node_type = 0; node_type < instance.getComputationalDag().getNumberOfNodeTypes(); ++node_type)
            if(instance.isCompatibleType(node_type, proc_type))
            {
                unsigned matched = std::min(ready_nodes_per_type[node_type], procs_per_type[proc_type]);
                nr_nodes += matched;
                ready_nodes_per_type[node_type] -= matched;
                procs_per_type[proc_type] -= matched;
            }

    return nr_nodes;
}