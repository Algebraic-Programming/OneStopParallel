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

#include "algorithms/GreedySchedulers/GreedyBspLocking.hpp"

std::vector<int> GreedyBspLocking::get_longest_path(const ComputationalDag& graph) const {
    std::vector<int> longest_path(graph.numberOfVertices(), 0);

    const std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
        longest_path[*r_iter] = graph.nodeWorkWeight(*r_iter);
        if(graph.numberOfChildren(*r_iter)>0)
        {
            int max = 0;
            for (const auto &child : graph.children(*r_iter)) {
                if(max <= longest_path[child])
                    max = longest_path[child];
            }
            longest_path[*r_iter] += max;
        }
    }

    return longest_path;
}

std::pair<RETURN_STATUS, BspSchedule> GreedyBspLocking::computeScheduleNoHeap(const BspInstance &instance) {

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

    locked_set.clear();
    locked.clear();
    locked.resize(N, -1);

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
    counter = 0;
    while (!ready.empty() || !finishTimes.empty()) {
        if (finishTimes.empty() && endSupStep) {
            for (unsigned i = 0; i < params_p; ++i)
                procReady[i].clear();

            allReady = ready;

            for(int node : locked_set)
                locked[node] = -1;
            locked_set.clear();

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
            bool should = Choose(instance, procInHyperedge, allReady, procReady, procFree, nextNode, nextProc);

            if (nextNode == -1 || nextProc == -1) {
                endSupStep = true;
                break;
            }

            /*if (!should && (float)free < (float)params_p * max_percent_idle_processors) {
                break;
            }*/

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

            for (const auto &succ : G.children(nextNode)) {
                locked_set.push_back(succ);
                if(locked[succ]>=0 && locked[succ]!=nextProc)
                    locked[succ] = params_p;
                else
                    locked[succ] = nextProc;
            }

            ++counter;

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
bool GreedyBspLocking::CanChooseNode(const BspInstance &instance, const std::set<int> &allReady,
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

bool GreedyBspLocking::Choose(const BspInstance &instance, const std::vector<std::vector<int>> &procInHyperedge,
                                const std::set<int> &allReady, const std::vector<std::set<int>> &procReady,
                                const std::vector<bool> &procFree, int &node, int &p) const {

    int maxScore = -instance.numberOfVertices()-1;
    double maxScore2 = 0;
    //double maxScore2 = (double)instance.numberOfVertices();
    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
        if (procFree[i] && !procReady[i].empty()) {
            p = i;
            // select node
            for (auto &r : procReady[i]) {
                int score = 0;
                //double score2 = 0;
                /*for (const auto &pred : instance.getComputationalDag().parents(r)) {

                    if (procInHyperedge[pred][i])
                        score2 += (double)instance.getComputationalDag().nodeCommunicationWeight(pred) /
                                 (double)instance.getComputationalDag().numberOfChildren(pred);
                }*/

                for (const auto &succ : instance.getComputationalDag().children(r)) {
                    if (locked[succ]>=0 && locked[succ] < instance.numberOfProcessors() && locked[succ]!=i)
                        score -= 1;
                }

                if (score > maxScore) {
                    maxScore = score;
                    //maxScore2 = score2;
                    node = r;
                }
            }
            return true;
        }
    }

    maxScore = -instance.numberOfVertices()-1;
    maxScore2 = 0;
    for (auto &r : allReady) {
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
            if (!procFree[i])
                continue;

            int score = 0;
            double score2 = 0;
            for (const auto &pred : instance.getComputationalDag().parents(r)) {

                if (procInHyperedge[pred][i]) {
                    score2 += (double)instance.getComputationalDag().nodeCommunicationWeight(pred) / 
                              (double)instance.getComputationalDag().numberOfChildren(pred);
                }
            }
            for (const auto &succ : instance.getComputationalDag().children(r)) {
                if (locked[succ]>=0 && locked[succ] < instance.numberOfProcessors()  && locked[succ]!=i)
                    score -= 1;
            }

            if (score > maxScore || (score == maxScore && score2 > maxScore2 + 0.0001 )) {

                if (use_memory_constraint) {

                    if (current_proc_memory[i] + instance.getComputationalDag().nodeMemoryWeight(r) <=
                        instance.getArchitecture().memoryBound()) {

                        maxScore = score;
                        maxScore2 = score2;
                        node = r;
                        p = i;
                    }
                } else {

                    maxScore = score;
                    maxScore2 = score2;
                    node = r;
                    p = i;
                }
            }
        }
    }
    return (maxScore>-3);
};

std::pair<int, double> GreedyBspLocking::computeScore(VertexType node, unsigned proc,
                                        const std::vector<std::vector<bool>> &procInHyperedge,
                                        const BspInstance &instance) {

    int score = 0;
    for (const auto &succ : instance.getComputationalDag().children(node)) {
        if (locked[succ]>=0 && locked[succ] < instance.numberOfProcessors() && locked[succ]!=proc)
            score -= lock_penalty;
    }
    
    return std::pair<int, double>(score + default_value[node], 0);
};

std::pair<RETURN_STATUS, BspSchedule> GreedyBspLocking::computeSchedule(const BspInstance &instance) {

    if (use_memory_constraint) {
        current_proc_memory = std::vector<unsigned>(instance.numberOfProcessors(), 0);
    }

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    const std::vector<int> path_length = get_longest_path(G);
    int max_path=0;
    for(int i=0; i<N; ++i)
        if(path_length[i]>max_path)
            max_path = path_length[i];

    default_value.clear();
    default_value.resize(N, 0);
    for(int i=0; i<N; ++i)
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

                if (use_memory_constraint) {
                    current_proc_memory[proc] = 0;
                }
            }

            allReady = ready;

            for(int node : locked_set)
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
                            if (current_proc_memory[schedule.assignedProcessor(node)] +
                                    instance.getComputationalDag().nodeMemoryWeight(succ) >
                                instance.getArchitecture().memoryBound()) {
                                canAdd = false;
                            }
                        }

                        if (canAdd) {
                            procReady[schedule.assignedProcessor(node)].insert(succ);
                            ready_phase[succ] = schedule.assignedProcessor(node);

                            std::pair<int, double> score = computeScore(succ, schedule.assignedProcessor(node), procInHyperedge, instance);
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
        if (!CanChooseNodeHeap(instance, allReady, procReady, procFree)) {
            endSupStep = true;
        }

        while (CanChooseNodeHeap(instance, allReady, procReady, procFree)) {

            VertexType nextNode = std::numeric_limits<VertexType>::max();
            unsigned nextProc = instance.numberOfProcessors();
            bool should = ChooseHeap(instance, procInHyperedge, allReady, procReady, procFree, nextNode, nextProc, endSupStep, max_finish_time - time);


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

                if(locked[succ] >= 0 && locked[succ] < params_p && locked[succ] != nextProc)
                {    
                    for (const auto &parent : G.parents(succ))
                    {
                        if(ready_phase[parent]>=0 && ready_phase[parent]<params_p && ready_phase[parent] != locked[succ])
                        {
                            (*node_proc_heap_handles[ready_phase[parent]][parent]).score += lock_penalty;
                            max_proc_score_heap[ready_phase[parent]].update(node_proc_heap_handles[ready_phase[parent]][parent]);
                        }
                        if(ready_phase[parent]==params_p)
                        {
                            for(int proc=0; proc< params_p; ++proc)
                            {
                                if(proc == locked[succ])
                                    continue;
                                
                                (*node_all_proc_heap_handles[proc][parent]).score += lock_penalty;
                                max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
                            }
                        }
                    }
                    locked[succ] = params_p;
                }
                else if (locked[succ]==-1)
                {
                    locked_set.push_back(succ);
                    locked[succ] = nextProc;   

                    for (const auto &parent : G.parents(succ))
                    {
                        if(ready_phase[parent]>=0 && ready_phase[parent]<params_p && ready_phase[parent] != nextProc)
                        {
                            (*node_proc_heap_handles[ready_phase[parent]][parent]).score -= lock_penalty;
                            max_proc_score_heap[ready_phase[parent]].update(node_proc_heap_handles[ready_phase[parent]][parent]);
                        }
                        if(ready_phase[parent]==params_p)
                        {
                            for(int proc=0; proc< params_p; ++proc)
                            {
                                if(proc == nextProc)
                                    continue;

                                (*node_all_proc_heap_handles[proc][parent]).score -= lock_penalty;
                                max_all_proc_score_heap[proc].update(node_all_proc_heap_handles[proc][parent]);
                            }
                        }
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

bool GreedyBspLocking::ChooseHeap(const BspInstance &instance, const std::vector<std::vector<bool>> &procInHyperedge,
                                    std::set<VertexType> &allReady,
                                    std::vector<std::set<VertexType>> &procReady,
                                    const std::vector<bool> &procFree, VertexType &node, unsigned &p,
                                    const bool endSupStep, const size_t remaining_time) {

    for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {

        if (procFree[proc] && !procReady[proc].empty()) {

            // select node
            heap_node top_node = max_proc_score_heap[proc].top();


            // filling up
            bool procready_empty = false;
            while (endSupStep && ( remaining_time < instance.getComputationalDag().nodeWorkWeight(top_node.node) )) {
                procReady[proc].erase(top_node.node);
                ready_phase[top_node.node] = -1;
                max_proc_score_heap[proc].pop();
                node_proc_heap_handles[proc].erase(top_node.node);
                if ( !procReady[proc].empty() ) {
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

    if(p < instance.numberOfProcessors())
        return true;

    heap_node best_node(instance.numberOfVertices(), -instance.numberOfVertices()-1, 0.0);

    for (unsigned proc = 0; proc < instance.numberOfProcessors(); ++proc) {
        if (!procFree[proc] or max_all_proc_score_heap[proc].empty())
            continue;

        heap_node top_node = max_all_proc_score_heap[proc].top();


        // filling up
        bool all_procready_empty = false;
        while (endSupStep && ( remaining_time < instance.getComputationalDag().nodeWorkWeight(top_node.node) )) {
            allReady.erase(top_node.node);
            for (unsigned proc_del = 0; proc_del < instance.numberOfProcessors(); proc_del++) {
                if (proc_del == proc) continue;
                max_all_proc_score_heap[proc_del].erase(node_all_proc_heap_handles[proc][top_node.node]);
                node_all_proc_heap_handles[proc_del].erase(top_node.node);
            }
            max_all_proc_score_heap[proc].pop();
            node_all_proc_heap_handles[proc].erase(top_node.node);
            ready_phase[top_node.node] = -1;
            if ( !max_all_proc_score_heap[proc].empty() ) {
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

                if (current_proc_memory[proc] + instance.getComputationalDag().nodeMemoryWeight(top_node.node) <=
                    instance.getArchitecture().memoryBound()) {

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
    return (best_node.score>-3);
};

// auxiliary - check if it is possible to assign a node at all
bool GreedyBspLocking::CanChooseNodeHeap(const BspInstance &instance, const std::set<VertexType> &allReady,
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