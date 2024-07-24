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

#include "algorithms/GreedySchedulers/GreedyBspScheduler.hpp"

std::pair<RETURN_STATUS, BspSchedule> GreedyBspScheduler::computeScheduleNoHeap(const BspInstance &instance) {

    if (use_memory_constraint) {
        current_proc_memory = std::vector<unsigned>(instance.numberOfProcessors(), 0);
    }

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    BspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1),
                         std::vector<unsigned>(instance.numberOfVertices()));

    std::set<int> ready;

    std::vector<std::vector<int>> procInHyperedge = std::vector<std::vector<int>>(N, std::vector<int>(params_p, false));

    std::vector<std::set<int>> procReady(params_p);
    std::set<int> allReady;

    std::vector<unsigned> nrPredecDone(N, 0);
    std::vector<bool> procFree(params_p, true);
    unsigned free = params_p;

    std::set<intPair> finishTimes;
    finishTimes.insert(intPair(0, -1));

    for (const auto &v : G.sourceVertices()) {
        ready.insert(v);
        allReady.insert(v);
    }

    unsigned supstepIdx = 0;
    bool endSupStep = false;
    while (!ready.empty() || !finishTimes.empty()) {
        if (finishTimes.empty() && endSupStep) {
            for (unsigned i = 0; i < params_p; ++i)
                procReady[i].clear();

            allReady = ready;

            ++supstepIdx;

            if (use_memory_constraint) {
                for (unsigned proc = 0; proc < params_p; proc++) {
                    current_proc_memory[proc] = 0;
                }
            }

            endSupStep = false;

            finishTimes.insert(intPair(0, -1));
        }

        const int time = finishTimes.begin()->a;

        // Find new ready jobs
        while (!finishTimes.empty() && finishTimes.begin()->a == time) {
            const intPair currentPair = *finishTimes.begin();
            finishTimes.erase(finishTimes.begin());
            const int node = currentPair.b;
            if (node != -1) {
                for (const auto &succ : G.children(node)) {
                    // for (size_t j = 0; j < G.Out[node].size(); ++j) {
                    //     int succ = G.Out[node][j];
                    ++nrPredecDone[succ];
                    if (nrPredecDone[succ] == G.numberOfParents(succ)) {
                        ready.insert(succ);

                        bool canAdd = true;
                        for (const auto &pred : G.parents(succ)) {
                            // for (const int i : G.In[succ]) {
                            if (schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) &&
                                schedule.assignedSuperstep(pred) == supstepIdx)
                                canAdd = false;
                        }

                        if (use_memory_constraint) {
                            if (current_proc_memory[schedule.assignedProcessor(node)] +
                                        instance.getComputationalDag().nodeMemoryWeight(succ) <=
                                    instance.getArchitecture().memoryBound() &&
                                canAdd) {
                                procReady[schedule.assignedProcessor(node)].insert(succ);
                            }

                        } else {

                            if (canAdd) {
                                procReady[schedule.assignedProcessor(node)].insert(succ);
                            }
                        }
                    }
                }
                procFree[schedule.assignedProcessor(node)] = true;
                ++free;
            }
        }

        if (endSupStep)
            continue;

        // Assign new jobs to processors
        if (!CanChooseNode(instance, allReady, procReady, procFree)) {
            endSupStep = true;
        }
        while (CanChooseNode(instance, allReady, procReady, procFree)) {

            int nextNode = -1, nextProc = -1;
            Choose(instance, procInHyperedge, allReady, procReady, procFree, nextNode, nextProc);

            if (nextNode == -1 || nextProc == -1) {
                endSupStep = true;
                break;
            }

            if (procReady[nextProc].find(nextNode) != procReady[nextProc].end())
                procReady[nextProc].erase(nextNode);
            else
                allReady.erase(nextNode);

            ready.erase(nextNode);
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);

            if (use_memory_constraint) {
                current_proc_memory[nextProc] += instance.getComputationalDag().nodeMemoryWeight(nextNode);

                std::vector<int> toErase;
                for (const auto &node : procReady[nextProc]) {
                    if (current_proc_memory[nextProc] + instance.getComputationalDag().nodeMemoryWeight(node) >
                        instance.getArchitecture().memoryBound()) {
                        toErase.push_back(node);
                    }
                }

                for (const auto &node : toErase) {
                    procReady[nextProc].erase(node);
                }
            }

            // schedule.proc[nextNode] = nextProc;
            // schedule.supstep[nextNode] = supstepIdx;

            finishTimes.insert(intPair(time + G.nodeWorkWeight(nextNode), nextNode));
            procFree[nextProc] = false;
            --free;

            // update comm auxiliary structure
            procInHyperedge[nextNode][nextProc] = true;

            for (const auto &pred : G.parents(nextNode)) {
                // for (const int i : G.In[nextNode]) {
                procInHyperedge[pred][nextProc] = true;
            }
        }
        if (allReady.empty() && free > params_p * max_percent_idle_processors &&
            ready.size() >= std::min(std::min(params_p, (unsigned)1.2 * (params_p - free)),
                                     params_p - free + ((unsigned)0.5 * free)))
            endSupStep = true;
    }

    assert(schedule.satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};

// auxiliary - check if it is possible to assign a node at all
bool GreedyBspScheduler::CanChooseNode(const BspInstance &instance, const std::set<int> &allReady,
                                       const std::vector<std::set<int>> &procReady,
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

void GreedyBspScheduler::Choose(const BspInstance &instance, const std::vector<std::vector<int>> &procInHyperedge,
                                const std::set<int> &allReady, const std::vector<std::set<int>> &procReady,
                                const std::vector<bool> &procFree, int &node, int &p) const {

    double maxScore = -1;
    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
        if (procFree[i] && !procReady[i].empty()) {
            p = i;
            // select node
            for (auto &r : procReady[i]) {
                double score = 0;
                for (const auto &pred : instance.getComputationalDag().parents(r)) {

                    if (procInHyperedge[pred][i])
                        score += (double)instance.getComputationalDag().nodeCommunicationWeight(pred) /
                                 (double)instance.getComputationalDag().numberOfChildren(pred);
                }

                if (score > maxScore) {
                    maxScore = score;
                    node = r;
                }
            }
            return;
        }
    }
    for (auto &r : allReady) {
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (!procFree[i])
                continue;

            double score = 0;
            for (const auto &pred : instance.getComputationalDag().parents(r)) {

                if (procInHyperedge[pred][i]) {
                    score += (double)instance.getComputationalDag().nodeCommunicationWeight(pred) /
                             (double)instance.getComputationalDag().numberOfChildren(pred);
                }
            }

            if (score > maxScore) {

                if (use_memory_constraint) {

                    if (current_proc_memory[i] + instance.getComputationalDag().nodeMemoryWeight(r) <=
                        instance.getArchitecture().memoryBound()) {

                        maxScore = score;
                        node = r;
                        p = i;
                    }
                } else {

                    maxScore = score;
                    node = r;
                    p = i;
                }
            }
        }
    }
};

double GreedyBspScheduler::computeScore(VertexType node, unsigned proc,
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

std::pair<RETURN_STATUS, BspSchedule> GreedyBspScheduler::computeSchedule(const BspInstance &instance) {

    if (use_memory_constraint) {
        current_proc_memory = std::vector<unsigned>(instance.numberOfProcessors(), 0);
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

    std::vector<unsigned> nrPredecDone(N, 0);
    std::vector<bool> procFree(params_p, true);
    unsigned free = params_p;

    std::set<std::pair<size_t, VertexType>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<VertexType>::max());

    for (const auto &v : G.sourceVertices()) {
        ready.insert(v);
        allReady.insert(v);

        for (unsigned proc = 0; proc < params_p; ++proc) {

            // double score = computeScore(v, proc, procInHyperedge, instance);
            heap_node new_node(v, 0.0);
            node_all_proc_heap_handles[proc][v] = max_all_proc_score_heap[proc].push(new_node);
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

                if (use_memory_constraint) {
                    current_proc_memory[proc] = 0;
                }
            }

            allReady = ready;

            for (unsigned proc = 0; proc < params_p; ++proc) {
                max_all_proc_score_heap[proc].clear();
                node_all_proc_heap_handles[proc].clear();
            }

            for (const auto &v : ready) {
                for (unsigned proc = 0; proc < params_p; ++proc) {

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
                            if (current_proc_memory[schedule.assignedProcessor(node)] +
                                    instance.getComputationalDag().nodeMemoryWeight(succ) >
                                instance.getArchitecture().memoryBound()) {
                                canAdd = false;
                            }
                        }

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

        if (endSupStep)
            continue;

        // Assign new jobs to processors
        if (!CanChooseNodeHeap(instance, allReady, procReady, procFree)) {
            endSupStep = true;
        }

        while (CanChooseNodeHeap(instance, allReady, procReady, procFree)) {

            VertexType nextNode = std::numeric_limits<VertexType>::max();
            unsigned nextProc = instance.numberOfProcessors();
            ChooseHeap(instance, procInHyperedge, allReady, procReady, procFree, nextNode, nextProc);

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
                    max_all_proc_score_heap[proc].erase(node_all_proc_heap_handles[proc][nextNode]);
                    node_all_proc_heap_handles[proc].erase(nextNode);
                }
            }

            ready.erase(nextNode);
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);

            if (use_memory_constraint) {
                current_proc_memory[nextProc] += instance.getComputationalDag().nodeMemoryWeight(nextNode);

                std::vector<VertexType> toErase;
                for (const auto &node : procReady[nextProc]) {
                    if (current_proc_memory[nextProc] + instance.getComputationalDag().nodeMemoryWeight(node) >
                        instance.getArchitecture().memoryBound()) {
                        toErase.push_back(node);
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
                            (double)instance.getComputationalDag().nodeCommunicationWeight(child) /
                            (double)instance.getComputationalDag().numberOfChildren(child);
                        max_proc_score_heap[nextProc].update(node_proc_heap_handles[nextProc][child]);
                    }
                }
            }
        }

        if (allReady.empty() && free > params_p * max_percent_idle_processors &&
            ready.size() >= std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
                                     params_p - free + ((unsigned)(0.5 * free)))) {
            endSupStep = true;
        }
    }

    assert(schedule.satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};

void GreedyBspScheduler::ChooseHeap(const BspInstance &instance, const std::vector<std::vector<bool>> &procInHyperedge,
                                    const std::set<VertexType> &allReady,
                                    const std::vector<std::set<VertexType>> &procReady,
                                    const std::vector<bool> &procFree, VertexType &node, unsigned &p) const {

    double max_score = -1.0;

    for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {

        if (procFree[proc] && !procReady[proc].empty()) {
            // if (procFree[proc] && !node_proc_heap_handles[proc].empty()) {

            // select node
            heap_node top_node = max_proc_score_heap[proc].top();

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

        if (top_node.score > max_score) {

            if (use_memory_constraint) {

                if (current_proc_memory[proc] + instance.getComputationalDag().nodeMemoryWeight(top_node.node) <=
                    instance.getArchitecture().memoryBound()) {

                    max_score = top_node.score;
                    node = top_node.node;
                    p = proc;
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
bool GreedyBspScheduler::CanChooseNodeHeap(const BspInstance &instance, const std::set<VertexType> &allReady,
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