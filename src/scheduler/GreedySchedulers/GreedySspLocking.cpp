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

#include "scheduler/GreedySchedulers/GreedySspLocking.hpp"

std::vector<int> GreedySspLocking::get_longest_path(const ComputationalDag &graph) const {
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

void GreedySspLocking::computeDefaultScores(const ComputationalDag &graph, const unsigned staleness) {
    default_value = std::vector<int>(graph.numberOfVertices(), 0);

    const std::vector<int> path_length = get_longest_path(graph);

    int max_path = 1;
    for (VertexType i = 0; i < graph.numberOfVertices(); ++i) {
        max_path = std::max(max_path, path_length[i]);
    }

    for (VertexType i = 0; i < graph.numberOfVertices(); ++i) {
        const int weighted_path_length = path_length[i] * 20; 
        default_value[i] = (weighted_path_length / max_path) * staleness;
        default_value[i] += ((weighted_path_length % max_path) * staleness) / max_path;
    }
}

// int GreedySspLocking::computeLockingPenalty(const ComputationalDag &graph) {
//     int avg_work_weight = 0;
//     int work_weight_remainder = 0;

//     // Minimising the risk of overflow
//     for (VertexType vert = 0; vert < graph.numberOfVertices(); vert++) {
//         avg_work_weight += (graph.nodeWorkWeight(vert) / graph.numberOfVertices());
//         work_weight_remainder += (graph.nodeWorkWeight(vert) % graph.numberOfVertices());

//         avg_work_weight += (work_weight_remainder / graph.numberOfVertices());
//         work_weight_remainder = work_weight_remainder % graph.numberOfVertices();
//     }

//     return std::max(1, avg_work_weight);
// }

std::pair<int, double> GreedySspLocking::computeScore(VertexType node, unsigned proc, const BspInstance &instance, const unsigned supstepIdx, const unsigned staleness) {

    int score = default_value[node];
    for (const auto &succ : instance.getComputationalDag().children(node)) {
        if (locked[succ] >= 0 && locked[succ] < instance.numberOfProcessors() && locked[succ] != proc)
            score -= lock_penalty * (std::max(staleness + locked_superstep[succ], supstepIdx) - supstepIdx);
    }

    return std::pair<int, double>(score, 0);
};

std::pair<RETURN_STATUS, SspSchedule> GreedySspLocking::computeSspSchedule(const BspInstance &instance, unsigned staleness) {

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

    // lock_penalty = computeLockingPenalty(G);

    std::vector<std::vector<unsigned>> procTypesCompatibleWithNodeType = instance.getProcTypesCompatibleWithNodeType();

    computeDefaultScores(G, staleness);

    max_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);
    max_all_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);

    node_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);
    node_all_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);

    locked_set.clear();
    locked = std::vector<int>(N, -1);
    locked_superstep = std::vector<unsigned>(N, 0);

    SspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1),
                         std::vector<unsigned>(instance.numberOfVertices()));

    std::set<VertexType> old_ready;
    std::vector<std::set<VertexType>> ready(staleness);
    ready_phase = std::vector<int>(N, -1);

    std::vector<std::vector<std::set<VertexType>>> procReady(staleness, std::vector<std::set<VertexType>>(params_p));
    std::vector<std::set<VertexType>> allReady(instance.getArchitecture().getNumberOfProcessorTypes());

    std::vector<unsigned> nrPredecDone(N, 0);
    std::vector<bool> procFree(params_p, true);
    unsigned free = params_p;

    std::vector<unsigned> nr_old_ready_nodes_per_type(G.getNumberOfNodeTypes(), 0);
    std::vector<std::vector<unsigned>> nr_ready_stale_nodes_per_type(staleness, std::vector<unsigned>(G.getNumberOfNodeTypes(), 0));
    std::vector<unsigned> nr_procs_per_type(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
    for (unsigned proc = 0; proc < params_p; ++proc)
        ++nr_procs_per_type[instance.getArchitecture().processorType(proc)];

    std::set<std::pair<size_t, VertexType>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

    for (const auto &v : G.sourceVertices()) {
        ready[0].insert(v);
        // allReady.insert(v);
        ++nr_ready_stale_nodes_per_type[0][G.nodeType(v)];
        // ready_phase[v] = params_p;

        // for (unsigned proc = 0; proc < params_p; ++proc) {
        //     if(instance.isCompatible(v, proc)) {
        //         heap_node new_node(v, default_value[v], G.numberOfChildren(v));
        //         node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
        //     }
        // }
    }

    std::vector<unsigned> number_of_allocated_allReady_tasks_in_superstep(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
    std::vector<unsigned> limit_of_number_of_allocated_allReady_tasks_in_superstep(instance.getArchitecture().getNumberOfProcessorTypes(), 0);

    unsigned supstepIdx = 0;
    bool endSupStep = true;
    bool begin_outer_while = true;
    bool able_to_schedule_in_step = false;
    unsigned successive_empty_supersteps = 0;
    while (!old_ready.empty() || std::any_of(ready.begin(), ready.end(), [](const std::set<VertexType>& ready_set) { return !ready_set.empty(); } ) || !finishTimes.empty()) {

        if (finishTimes.empty() && endSupStep) {
            able_to_schedule_in_step = false;
            number_of_allocated_allReady_tasks_in_superstep = std::vector<unsigned>(instance.getArchitecture().getNumberOfProcessorTypes(), 0);
            ready_phase = std::vector<int>(N, -1);
            
            for (unsigned proc = 0; proc < params_p; ++proc) {
                procReady[supstepIdx % staleness][proc].clear();
            }
            
            if (!begin_outer_while) {
                supstepIdx++;
            } else {
                begin_outer_while = false;
            }

            for (unsigned proc = 0; proc < params_p; ++proc) {
                max_proc_score_heap[proc].clear();
                node_proc_heap_handles[proc].clear();

                if (use_memory_constraint && instance.getArchitecture().getMemoryConstraintType() == LOCAL) {
                    current_proc_persistent_memory[proc] = 0;
                }
            }

            for (unsigned proc = 0; proc < params_p; proc++) {
                for (const VertexType& node : procReady[supstepIdx % staleness][proc]) {
                    std::pair<int, double> score =
                        computeScore(node, proc, instance, supstepIdx, staleness);
                    heap_node new_node(node, score.first, G.numberOfChildren(node));

                    node_proc_heap_handles[proc][node] =
                        max_proc_score_heap[proc].push(new_node);

                    ready_phase[node] = proc;
                }
            }

            for(int procType = 0; procType < instance.getArchitecture().getNumberOfProcessorTypes(); ++procType) {
                allReady[procType].clear();
            }

            for (auto node_it = locked_set.begin(); node_it !=locked_set.end(); ) {
                if (locked_superstep[*node_it] + staleness <= supstepIdx) {
                    locked[*node_it] = -1;
                    node_it = locked_set.erase(node_it);
                } else {
                    node_it++;
                }
            }

            for (unsigned proc = 0; proc < params_p; ++proc) {
                max_all_proc_score_heap[proc].clear();
                node_all_proc_heap_handles[proc].clear();
            }

            old_ready.insert(ready[supstepIdx % staleness].begin(), ready[supstepIdx % staleness].end());
            ready[supstepIdx % staleness].clear();
            for(unsigned node_type = 0; node_type < instance.getComputationalDag().getNumberOfNodeTypes(); ++node_type) {
                nr_old_ready_nodes_per_type[node_type] += nr_ready_stale_nodes_per_type[supstepIdx % staleness][node_type];
                nr_ready_stale_nodes_per_type[supstepIdx % staleness][node_type] = 0;
            }

            for (const auto &v : old_ready) {
                ready_phase[v] = params_p;
                
                for(unsigned procType : procTypesCompatibleWithNodeType[G.nodeType(v)]) {
                    allReady[procType].emplace(v);
                }

                for (unsigned proc = 0; proc < params_p; ++proc) {

                    if(!instance.isCompatible(v, proc)) continue;

                    std::pair<int, double> score = computeScore(v, proc, instance, supstepIdx, staleness);
                    heap_node new_node(v, score.first, G.numberOfChildren(v));
                    node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
                }
            }

            for (unsigned procType = 0; procType < instance.getArchitecture().getNumberOfProcessorTypes(); procType++) {
                unsigned equal_split = (allReady[procType].size() + staleness - 1) / staleness;
                unsigned at_least_for_long_step = 3 * nr_procs_per_type[procType];

                limit_of_number_of_allocated_allReady_tasks_in_superstep[procType] = std::max(at_least_for_long_step, equal_split);
            }

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
                        ready[supstepIdx % staleness].insert(succ);
                        ++nr_ready_stale_nodes_per_type[supstepIdx % staleness][G.nodeType(succ)];

                        bool canAdd = instance.isCompatible(succ, schedule.assignedProcessor(node));
                        unsigned earliest_add = supstepIdx;
                        bool memory_ok = true;

                        for (const auto &pred : G.parents(succ)) {

                            if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node)) {
                                earliest_add = std::max(earliest_add, staleness + schedule.assignedSuperstep(pred));
                            }
                        }

                        if (use_memory_constraint && canAdd && earliest_add == supstepIdx) {

                            if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                                if (current_proc_persistent_memory[schedule.assignedProcessor(node)] +
                                        instance.getComputationalDag().nodeMemoryWeight(succ) >
                                    instance.getArchitecture().memoryBound(schedule.assignedProcessor(node))) {
                                    memory_ok = false;
                                }

                            } else if (instance.getArchitecture().getMemoryConstraintType() ==
                                       PERSISTENT_AND_TRANSIENT) {

                                if (current_proc_persistent_memory[schedule.assignedProcessor(node)] +
                                        instance.getComputationalDag().nodeMemoryWeight(succ) +
                                        std::max(current_proc_transient_memory[schedule.assignedProcessor(node)],
                                                 instance.getComputationalDag().nodeCommunicationWeight(succ)) >
                                    instance.getArchitecture().memoryBound(schedule.assignedProcessor(node))) {
                                    memory_ok = false;
                                }
                            }
                        }
                        
                        if (canAdd) {
                            for (unsigned step_to_add = earliest_add; step_to_add < supstepIdx + staleness; step_to_add++) {
                                if ((step_to_add == supstepIdx) && !memory_ok) {
                                    continue;
                                }
                                if (step_to_add == supstepIdx) {
                                    std::pair<int, double> score =
                                        computeScore(succ, schedule.assignedProcessor(node), instance, supstepIdx, staleness);
                                    heap_node new_node(succ, score.first, G.numberOfChildren(succ));

                                    node_proc_heap_handles[schedule.assignedProcessor(node)][succ] =
                                        max_proc_score_heap[schedule.assignedProcessor(node)].push(new_node);

                                    ready_phase[succ] = schedule.assignedProcessor(node);
                                }
                                procReady[step_to_add % staleness][schedule.assignedProcessor(node)].emplace(succ);
                            }

                            // if ((memory_ok && earliest_add == supstepIdx) || (supstepIdx < earliest_add && earliest_add < supstepIdx + staleness)) {
                            //     ready_phase[succ] = schedule.assignedProcessor(node);
                            // }
                        }
                    }
                }
                procFree[schedule.assignedProcessor(node)] = true;
                ++free;
            }
        }

        // Assign new jobs to processors
        if (!CanChooseNode(instance, procReady[supstepIdx % staleness], procFree)) {
            endSupStep = true;
        }

        while (CanChooseNode(instance, procReady[supstepIdx % staleness], procFree)) {

            VertexType nextNode = std::numeric_limits<VertexType>::max();
            unsigned nextProc = instance.numberOfProcessors();
            Choose(instance, allReady, procReady[supstepIdx % staleness], procFree, nextNode, nextProc,
                                 endSupStep, max_finish_time - time);

            if (nextNode == std::numeric_limits<VertexType>::max() || nextProc == instance.numberOfProcessors()) {
                endSupStep = true;
                break;
            }

            if (ready_phase[nextNode] < params_p) {

                for (unsigned i = 0; i < staleness; i++) {
                    procReady[i][nextProc].erase(nextNode);
                }

                max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc].at(nextNode));
                node_proc_heap_handles[nextProc].erase(nextNode);

            } else {

                for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
                    if(instance.isCompatible(nextNode, proc)) {
                        unsigned procType = instance.getArchitecture().processorType(proc);
                        if (allReady[procType].find(nextNode) != allReady[procType].end()) {
                            max_all_proc_score_heap[proc].erase(node_all_proc_heap_handles[proc].at(nextNode));
                            node_all_proc_heap_handles[proc].erase(nextNode);
                        }
                    }
                }

                for(unsigned procType : procTypesCompatibleWithNodeType[G.nodeType(nextNode)]) {
                    allReady[procType].erase(nextNode);
                }

                --nr_old_ready_nodes_per_type[G.nodeType(nextNode)];
                const unsigned nextProcType = instance.getArchitecture().processorType(nextProc);
                number_of_allocated_allReady_tasks_in_superstep[nextProcType]++;
                if (number_of_allocated_allReady_tasks_in_superstep[nextProcType] >= limit_of_number_of_allocated_allReady_tasks_in_superstep[nextProcType]) {
                    allReady[nextProcType].clear();
                    std::set<VertexType> toErase;
                    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                        if (instance.getArchitecture().processorType(proc) == nextProcType) {
                            for (const auto& [node, _] : node_all_proc_heap_handles[proc]) {
                                toErase.emplace(node);
                            }
                            max_all_proc_score_heap[proc].clear();
                            node_all_proc_heap_handles[proc].clear();
                        }
                    }
                    for (const VertexType& vert : toErase) {
                        bool remove_from_ready_phase = true;
                        for (unsigned procType : procTypesCompatibleWithNodeType[G.nodeType(vert)]) {
                            if (procType == nextProcType) continue;
                            
                            if (allReady[procType].find(vert) != allReady[procType].end()) {
                                remove_from_ready_phase = false;
                                break;
                            }
                        }
                        if (remove_from_ready_phase) {
                            ready_phase[vert] = -1;
                        }
                    }
                }
            }

            for (unsigned i = 0; i < staleness; i++) {
                ready[i].erase(nextNode);
            }
            old_ready.erase(nextNode);
            
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);
            able_to_schedule_in_step = true;

            ready_phase[nextNode] = -1;

            if (use_memory_constraint) {

                std::vector<VertexType> toErase;
                if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                    current_proc_persistent_memory[nextProc] +=
                        instance.getComputationalDag().nodeMemoryWeight(nextNode);

                    for (const auto &node : procReady[supstepIdx % staleness][nextProc]) {
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

                    for (const auto &node : procReady[supstepIdx % staleness][nextProc]) {
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
                    procReady[supstepIdx % staleness][nextProc].erase(node);
                    max_proc_score_heap[nextProc].erase(node_proc_heap_handles[nextProc].at(node));
                    node_proc_heap_handles[nextProc].erase(node);
                    ready_phase[node] = -1;
                }
            }

            finishTimes.emplace(time + G.nodeWorkWeight(nextNode), nextNode);
            procFree[nextProc] = false;
            --free;

            // update auxiliary structures

            for (const auto &succ : G.children(nextNode)) {

                if (locked[succ] >= 0 && locked[succ] < params_p) {
                    for (const auto &parent : G.parents(succ)) {
                        if (ready_phase[parent] >= 0 && ready_phase[parent] < params_p) {
                            if (ready_phase[parent] == nextProc) {
                                (*node_proc_heap_handles[nextProc].at(parent)).score += lock_penalty * (std::max(staleness + locked_superstep[succ], supstepIdx) - supstepIdx);
                                max_proc_score_heap[nextProc].update(
                                    node_proc_heap_handles[nextProc][parent]);
                            } else if (ready_phase[parent] == locked[succ]) {
                                (*node_proc_heap_handles[locked[succ]].at(parent)).score -= lock_penalty * staleness;
                                max_proc_score_heap[locked[succ]].update(node_proc_heap_handles[locked[succ]][parent]);
                            } else {
                                (*node_proc_heap_handles[ready_phase[parent]].at(parent)).score -= lock_penalty * std::min(supstepIdx - locked_superstep[succ], staleness);
                                max_proc_score_heap[ready_phase[parent]].update(node_proc_heap_handles[ready_phase[parent]][parent]);
                            }
                        }
                        if (ready_phase[parent] == params_p) {
                            for (unsigned proc = 0; proc < params_p; proc++) {
                                if (!instance.isCompatible(parent, proc)) continue;

                                unsigned procType = instance.getArchitecture().processorType(proc);
                                if (allReady[procType].find(parent) == allReady[procType].end()) continue;

                                if (proc == nextProc) {
                                    (*node_all_proc_heap_handles[proc].at(parent)).score += lock_penalty * (std::max(staleness + locked_superstep[succ], supstepIdx) - supstepIdx);
                                    max_all_proc_score_heap[proc].update(
                                        node_all_proc_heap_handles[proc][parent]);
                                } else if (proc == locked[succ]) {
                                    (*node_all_proc_heap_handles[proc].at(parent)).score -= lock_penalty * staleness;
                                    max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
                                } else {
                                    (*node_all_proc_heap_handles[proc].at(parent)).score -= lock_penalty * std::min(supstepIdx - locked_superstep[succ], staleness);
                                    max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
                                }
                            }
                        }
                    }
                    locked[succ] = nextProc;
                    locked_superstep[succ] = supstepIdx;
                } else if (locked[succ] == -1) {
                    locked_set.emplace(succ);
                    locked[succ] = nextProc;
                    locked_superstep[succ] = supstepIdx;

                    for (const auto &parent : G.parents(succ)) {
                        if (ready_phase[parent] >= 0 && ready_phase[parent] < params_p &&
                            ready_phase[parent] != nextProc) {
                            
                            (*node_proc_heap_handles[ready_phase[parent]].at(parent)).score -= lock_penalty * staleness;
                            max_proc_score_heap[ready_phase[parent]].update(
                                node_proc_heap_handles[ready_phase[parent]][parent]);
                        }
                        if (ready_phase[parent] == params_p) {
                            for (int proc = 0; proc < params_p; ++proc) {
                                if (proc == nextProc || !instance.isCompatible(parent, proc)) continue;

                                unsigned procType = instance.getArchitecture().processorType(proc);
                                if (allReady[procType].find(parent) == allReady[procType].end()) continue;

                                (*node_all_proc_heap_handles[proc].at(parent)).score -= lock_penalty * staleness;
                                max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
                            }
                        }
                    }
                }
            }
        }

        // if (use_memory_constraint && not check_mem_feasibility(instance, allReady, procReady)) {
        if (able_to_schedule_in_step) {
            successive_empty_supersteps = 0;
        } else {
            successive_empty_supersteps++;
            if (successive_empty_supersteps > 100 + staleness) {
                return {ERROR, schedule};
            }
        }

        if (free > params_p * max_percent_idle_processors &&
            ((!increase_parallelism_in_new_superstep) ||
             get_nr_parallelizable_nodes(instance, staleness, nr_old_ready_nodes_per_type, nr_ready_stale_nodes_per_type[(supstepIdx + 1) % staleness], procReady[(supstepIdx + 1) % staleness], nr_procs_per_type) >= std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
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
bool GreedySspLocking::Choose(const BspInstance &instance, std::vector<std::set<VertexType>> &allReady, std::vector<std::set<VertexType>> &procReady,
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

    heap_node best_node(std::numeric_limits<VertexType>::max(), std::numeric_limits<int>::min(), std::numeric_limits<int>::min());

    for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
        if (!procFree[proc] or max_all_proc_score_heap[proc].empty())
            continue;

        heap_node top_node = max_all_proc_score_heap[proc].top();

        // filling up
        bool all_procready_empty = false;
        while (endSupStep && (remaining_time < instance.getComputationalDag().nodeWorkWeight(top_node.node))) {
            for (unsigned procType = 0; procType < instance.getArchitecture().getNumberOfProcessorTypes(); procType++) {
                if (instance.isCompatibleType(instance.getComputationalDag().nodeType(top_node.node), procType)) {
                    allReady[procType].erase(top_node.node);
                }
            }
            for (unsigned proc_del = 0; proc_del < instance.numberOfProcessors(); proc_del++) {
                if (proc_del == proc || !instance.isCompatible(top_node.node, proc_del))
                    continue;
                
                if (node_all_proc_heap_handles[proc_del].find(top_node.node) != node_all_proc_heap_handles[proc_del].end()) {
                    max_all_proc_score_heap[proc_del].erase(node_all_proc_heap_handles[proc_del].at(top_node.node));
                    node_all_proc_heap_handles[proc_del].erase(top_node.node);
                }
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
                        instance.getArchitecture().memoryBound(proc)) {

                        best_node = top_node;
                        node = top_node.node;
                        p = proc;
                    }

                } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {
                    if (current_proc_persistent_memory[proc] +
                            instance.getComputationalDag().nodeMemoryWeight(top_node.node) +
                            std::max(current_proc_transient_memory[proc],
                                     instance.getComputationalDag().nodeCommunicationWeight(top_node.node)) <=
                        instance.getArchitecture().memoryBound(proc)) {

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
    return (best_node.score > std::numeric_limits<int>::min());
};

bool GreedySspLocking::check_mem_feasibility(const BspInstance &instance, const std::set<VertexType> &allReady,
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

// auxiliary - check if it is possible to assign a node at all
bool GreedySspLocking::CanChooseNode(const BspInstance &instance,
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

// // get number of ready nodes that can be run in parallel, to check whether more parallelism is available
// // (currently OK for triangular compatibility matrix, otherwise just heuristic - can be changed to matching-based solution later)
// unsigned GreedySspLocking::get_nr_parallelizable_nodes(const BspInstance &instance,
//                                             const std::vector<unsigned>& nr_ready_nodes_per_type,
//                                             const std::vector<unsigned>& nr_procs_per_type) const {
//     unsigned nr_nodes = 0;

//     std::vector<unsigned> ready_nodes_per_type = nr_ready_nodes_per_type;
//     std::vector<unsigned> procs_per_type = nr_procs_per_type;
//     for(unsigned proc_type = 0; proc_type < instance.getArchitecture().getNumberOfProcessorTypes(); ++proc_type)
//         for(unsigned node_type = 0; node_type < instance.getComputationalDag().getNumberOfNodeTypes(); ++node_type)
//             if(instance.isCompatibleType(node_type, proc_type))
//             {
//                 unsigned matched = std::min(ready_nodes_per_type[node_type], procs_per_type[proc_type]);
//                 nr_nodes += matched;
//                 ready_nodes_per_type[node_type] -= matched;
//                 procs_per_type[proc_type] -= matched;
//             }

//     return nr_nodes;
// }

// get number of ready nodes that can be run in parallel, to check whether more parallelism is available
// (currently OK for triangular compatibility matrix, otherwise just heuristic - can be changed to matching-based solution later)
unsigned GreedySspLocking::get_nr_parallelizable_nodes(const BspInstance &instance,
                                            const unsigned &stale,
                                            const std::vector<unsigned>& nr_old_ready_nodes_per_type,
                                            const std::vector<unsigned>& nr_ready_nodes_per_type,
                                            const std::vector<std::set<VertexType>> &procReady,
                                            const std::vector<unsigned>& nr_procs_per_type) const {
    unsigned nr_nodes = 0;
    unsigned num_proc_types = instance.getArchitecture().getNumberOfProcessorTypes();

    std::vector<unsigned> procs_per_type = nr_procs_per_type;

    if (stale > 1) {
        for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
            if (!(procReady[proc].empty())) {
                procs_per_type[instance.getArchitecture().processorType(proc)]--; // Cannot go below zero (due to logic)
                nr_nodes++;
            }
        }
    }

    std::vector<unsigned> ready_nodes_per_type = nr_ready_nodes_per_type;
    for (unsigned node_type = 0; node_type < ready_nodes_per_type.size(); node_type++) {
        ready_nodes_per_type[node_type] += nr_old_ready_nodes_per_type[node_type];
    }

    for(unsigned proc_type = 0; proc_type < num_proc_types; ++proc_type) {
        for(unsigned node_type = 0; node_type < instance.getComputationalDag().getNumberOfNodeTypes(); ++node_type) {
            if(instance.isCompatibleType(node_type, proc_type)) {
                unsigned matched = std::min(ready_nodes_per_type[node_type], procs_per_type[proc_type]);
                nr_nodes += matched;
                ready_nodes_per_type[node_type] -= matched;
                procs_per_type[proc_type] -= matched;
            }
        }
    }

    return nr_nodes;
}