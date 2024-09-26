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

#include "scheduler/GreedySchedulers/GreedyEtfScheduler.hpp"

std::pair<RETURN_STATUS, BspSchedule> GreedyEtfScheduler::computeSchedule(const BspInstance &instance) {

    if (use_memory_constraint) {

        switch (instance.getArchitecture().getMemoryConstraintType()) {

        case LOCAL:
            throw std::invalid_argument("Local memory constraint not supported");

        case PERSISTENT_AND_TRANSIENT:
            current_proc_persistent_memory = std::vector<int>(instance.numberOfProcessors(), 0);
            current_proc_transient_memory = std::vector<int>(instance.numberOfProcessors(), 0);

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

    const unsigned &avgCommCost = instance.getArchitecture().computeCommAverage();

    CSchedule schedule(instance.numberOfVertices());

    std::vector<std::deque<unsigned>> greedyProcLists(instance.numberOfProcessors());

    std::vector<unsigned> predecProcessed(instance.numberOfVertices(), 0);

    std::vector<int> finishTimes(instance.numberOfProcessors(), 0), send(instance.numberOfProcessors(), 0),
        rec(instance.numberOfProcessors(), 0);

    std::vector<int> BL;
    if (mode == BL_EST)
        BL = ComputeBottomLevel(instance, avgCommCost);
    else
        BL = std::vector<int>(instance.numberOfVertices(), 0);

    std::set<intPair> ready;

    for (const auto &v : instance.getComputationalDag().sourceVertices()) {
        ready.insert(intPair(BL[v], v));
    }

    while (!ready.empty()) {
        intTriple best(0, 0, 0);
        if (mode == BL_EST) {
            std::vector<int> nodeList{ready.begin()->b};
            ready.erase(ready.begin());
            best = GetBestESTforNodes(instance, schedule, nodeList, finishTimes, send, rec, avgCommCost);
        }
        if (mode == ETF) {
            std::vector<int> nodeList;
            for (const intPair next : ready)
                nodeList.push_back(next.b);
            best = GetBestESTforNodes(instance, schedule, nodeList, finishTimes, send, rec, avgCommCost);
            ready.erase(intPair(0, best.a));
        }
        int node = best.a;
        int proc = best.b;

        schedule.proc[node] = proc;
        greedyProcLists[proc].push_back(node);

        schedule.time[node] = best.c;
        finishTimes[proc] = schedule.time[node] + instance.getComputationalDag().nodeWorkWeight(node);

        if (use_memory_constraint) {

            if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

                current_proc_persistent_memory[proc] += instance.getComputationalDag().nodeMemoryWeight(node);
                current_proc_transient_memory[proc] = std::max(current_proc_transient_memory[proc],
                                                               instance.getComputationalDag().nodeMemoryWeight(node));
            }
        }

        // for (const int succ : G.Out[node]) {

        for (const auto &succ : instance.getComputationalDag().children(node)) {
            ++predecProcessed[succ];
            if (predecProcessed[succ] == instance.getComputationalDag().numberOfParents(succ))
                ready.insert(intPair(BL[succ], succ));
        }

        if (use_memory_constraint && not check_mem_feasibility(instance, ready)) {

            return {ERROR, schedule.convertToBspSchedule(instance, greedyProcLists)};
        }
    }

    return {SUCCESS, schedule.convertToBspSchedule(instance, greedyProcLists)};
};

bool GreedyEtfScheduler::check_mem_feasibility(const BspInstance &instance, const std::set<intPair> &ready) const {

    if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

        for (const auto &node_pair : ready) {
            for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {

                auto node = node_pair.b;

                if (current_proc_persistent_memory[i] + instance.getComputationalDag().nodeMemoryWeight(node) +
                        std::max(current_proc_transient_memory[i],
                                 instance.getComputationalDag().nodeCommunicationWeight(node)) <=
                    instance.getArchitecture().memoryBound()) {
                    return true;
                }
            }
        }

        return false;
    }
};

std::vector<int> GreedyEtfScheduler::ComputeBottomLevel(const BspInstance &instance, unsigned avg_) const {

    std::vector<int> BL(instance.numberOfVertices(), 0.0);

    // const std::vector<VertexType> topOrder = instance.getComputationalDag().topologicalOrderVertices();
    for (const auto &node : instance.getComputationalDag().dfs_reverse_topoOrder()) {
        // for (unsigned i = instance.numberOfVertices() - 1; i >= 0; --i) {
        // const int node = topOrder[i];
        int maxval = 0;
        //        VertexType maxNode = 0;

        for (const auto &out_edge : instance.getComputationalDag().out_edges(node)) {

            const auto &succ = out_edge.m_target;
            const int tmp_val = BL[succ] + instance.getComputationalDag().edgeCommunicationWeight(out_edge);

            if (tmp_val > maxval) {
                maxval = tmp_val;
                //              maxNode = succ;
            }
        }
        BL[node] = maxval + instance.getComputationalDag().nodeWorkWeight(node);
    }
    return BL;
};

int GreedyEtfScheduler::GetESTforProc(const BspInstance &instance, CSchedule &schedule, int node, int proc,
                                      const int procAvailableFrom, std::vector<int> &send, std::vector<int> &rec,
                                      unsigned avg_) const {

    std::vector<intPair> predec;
    for (const auto &pred : instance.getComputationalDag().parents(node)) {
        // for (const int pred : schedule.G.In[node])
        predec.emplace_back(schedule.time[pred] + instance.getComputationalDag().nodeWorkWeight(pred), pred);
    }

    std::sort(predec.begin(), predec.end());

    int EST = procAvailableFrom;
    for (const intPair next : predec) {
        int t = schedule.time[next.b] + instance.getComputationalDag().nodeWorkWeight(next.b);
        if (schedule.proc[next.b] != proc) {
            t = std::max(t, send[schedule.proc[next.b]]);
            t = std::max(t, rec[proc]);
            t += instance.getComputationalDag().edgeCommunicationWeight(
                     boost::edge(next.b, node, instance.getComputationalDag().getGraph()).first) *
                 instance.sendCosts(schedule.proc[next.b], proc);

            // t += instance.getComputationalDag().nodeCommunicationWeight(next.b) *
            //      (use_numa ? instance.communicationCosts() * instance.sendCosts(schedule.proc[next.b], proc) : avg_);

            send[schedule.proc[next.b]] = t;
            rec[proc] = t;
        }
        EST = std::max(EST, t);
    }
    return EST;
};

// auxiliary: compute EST of node over all processors
intTriple GreedyEtfScheduler::GetBestESTforNodes(const BspInstance &instance, CSchedule &schedule,
                                                 const std::vector<int> &nodeList,
                                                 const std::vector<int> &procAvailableFrom, std::vector<int> &send,
                                                 std::vector<int> &rec, unsigned avg_) const {

    int bestEST = INT_MAX, bestProc = 0, bestNode = 0;
    std::vector<int> bestSend, bestRec;
    for (int node : nodeList)
        for (unsigned j = 0; j < instance.numberOfProcessors(); ++j) {

            if (use_memory_constraint) {

                if (instance.getArchitecture().getMemoryConstraintType() == LOCAL) {

                    if (current_proc_persistent_memory[j] + instance.getComputationalDag().nodeMemoryWeight(node) >
                        instance.getArchitecture().memoryBound()) {
                        continue;
                    }

                } else if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

                    if (current_proc_persistent_memory[j] + instance.getComputationalDag().nodeMemoryWeight(node) +
                            std::max(current_proc_transient_memory[j],
                                     instance.getComputationalDag().nodeCommunicationWeight(node)) >
                        instance.getArchitecture().memoryBound()) {
                        continue;
                    }
                }
            }

            std::vector<int> newSend = send;
            std::vector<int> newRec = rec;
            int EST = GetESTforProc(instance, schedule, node, j, procAvailableFrom[j], newSend, newRec, avg_);
            if (EST < bestEST) {
                bestEST = EST;
                bestProc = j;
                bestNode = node;
                bestSend = newSend;
                bestRec = newRec;
            }
        }

    send = bestSend;
    rec = bestRec;
    return intTriple(bestNode, bestProc, bestEST);
};