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

#include "scheduler/GreedySchedulers/GreedyBspPebbler.hpp"

std::pair<RETURN_STATUS, BspSchedule> GreedyBspPebbler::computeSchedule(const BspInstance &instance) {

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    max_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);
    max_all_proc_score_heap = std::vector<boost::heap::fibonacci_heap<heap_node>>(params_p);

    node_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);
    node_all_proc_heap_handles = std::vector<std::unordered_map<VertexType, heap_handle>>(params_p);

    std::vector<std::set<unsigned> > in_mem(params_p);
    std::vector<unsigned > mem_used(params_p, 0);

    std::vector<std::vector<double> > input_fraction_in_mem(N, std::vector<double>(params_p, 0));
    std::vector<double> total_inweight(N, 0);
    for(int node=0; node < N; ++node)
        for (const auto &pred : G.parents(node))
            total_inweight[node] += (double)G.nodeWorkWeight(pred);


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
            }

            allReady = ready;

            for (unsigned proc = 0; proc < params_p; ++proc) {
                max_all_proc_score_heap[proc].clear();
                node_all_proc_heap_handles[proc].clear();
            }

            for (const auto &v : ready) {
                for (unsigned proc = 0; proc < params_p; ++proc) {

                    heap_node new_node(v, input_fraction_in_mem[v][proc]);
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

                        if (canAdd) {
                            procReady[schedule.assignedProcessor(node)].insert(succ);

                            heap_node new_node(succ, input_fraction_in_mem[succ][schedule.assignedProcessor(node)]);
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

            finishTimes.emplace(time + G.nodeWorkWeight(nextNode), nextNode);
            procFree[nextProc] = false;
            --free;

            //update memory
            mem_used[nextProc] += G.nodeWorkWeight(nextNode);
            in_mem[nextProc].insert(nextNode);
            for(const auto &succ : G.children(nextNode))
                input_fraction_in_mem[succ][nextProc] += (double) G.nodeWorkWeight(nextNode) / total_inweight[succ]; 

            std::set<unsigned> non_evictable{nextNode};
            for (const auto &pred : G.parents(nextNode))
            {
                non_evictable.insert(pred);
                if(in_mem[nextProc].find(pred) == in_mem[nextProc].end())
                {
                    in_mem[nextProc].insert(pred);
                    mem_used[nextProc] += G.nodeWorkWeight(pred);
                    for(const auto &succ : G.children(pred))
                    {
                        input_fraction_in_mem[succ][nextProc] += (double) G.nodeWorkWeight(pred) / total_inweight[succ];
                        if(procReady[nextProc].find(succ) != procReady[nextProc].end())
                        {
                            (*node_proc_heap_handles[nextProc][succ]).score = input_fraction_in_mem[succ][nextProc];
                            max_proc_score_heap[nextProc].update(node_proc_heap_handles[nextProc][succ]);
                        }
                        if(allReady.find(succ) != allReady.end())
                        {
                            (*node_all_proc_heap_handles[nextProc][succ]).score = input_fraction_in_mem[succ][nextProc];
                            max_all_proc_score_heap[nextProc].update(node_all_proc_heap_handles[nextProc][succ]);
                        }
                    }
                }
            }
            auto itr = (--in_mem[nextProc].end());
            while(mem_used[nextProc] > mem_limit)
            {
                if(non_evictable.find(*itr) == non_evictable.end())
                {
                    mem_used[nextProc] -= G.nodeWorkWeight(*itr);
                    for(const auto &succ : G.children(*itr))
                    {
                        input_fraction_in_mem[succ][nextProc] -= (double) G.nodeWorkWeight(*itr) / total_inweight[succ];
                        if(procReady[nextProc].find(succ) != procReady[nextProc].end())
                        {
                            (*node_proc_heap_handles[nextProc][succ]).score = input_fraction_in_mem[succ][nextProc];
                            max_proc_score_heap[nextProc].update(node_proc_heap_handles[nextProc][succ]);
                        }
                        if(allReady.find(succ) != allReady.end())
                        {
                            (*node_all_proc_heap_handles[nextProc][succ]).score = input_fraction_in_mem[succ][nextProc];
                            max_all_proc_score_heap[nextProc].update(node_all_proc_heap_handles[nextProc][succ]);
                        }
                    }

                    auto itr_copy = itr;
                    if(itr == in_mem[nextProc].begin())
                    {
                        in_mem[nextProc].erase(itr);
                        break;
                    }
                    --itr;

                    in_mem[nextProc].erase(itr_copy);
                }
                else
                {
                    if(itr == in_mem[nextProc].begin())
                        break;
                    --itr;
                }
            }

        }

        if (allReady.empty() && free > params_p * max_percent_idle_processors && ((!increase_parallelism_in_new_superstep) ||
            ready.size() >= std::min(std::min(params_p, (unsigned)(1.2 * (params_p - free))),
                                     params_p - free + ((unsigned)(0.5 * free))))) {
            endSupStep = true;
        }
    }

    assert(schedule.satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};

void GreedyBspPebbler::ChooseHeap(const BspInstance &instance, const std::vector<std::vector<bool>> &procInHyperedge,
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
            max_score = top_node.score;
            node = top_node.node;
            p = proc;
        }
    }
};

// auxiliary - check if it is possible to assign a node at all
bool GreedyBspPebbler::CanChooseNodeHeap(const BspInstance &instance, const std::set<VertexType> &allReady,
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