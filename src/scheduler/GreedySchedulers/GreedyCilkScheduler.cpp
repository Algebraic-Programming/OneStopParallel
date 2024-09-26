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

#include "scheduler/GreedySchedulers/GreedyCilkScheduler.hpp"

std::pair<RETURN_STATUS, BspSchedule> GreedyCilkScheduler::computeSchedule(const BspInstance &instance) {

    if (use_memory_constraint) {

        switch (instance.getArchitecture().getMemoryConstraintType()) {

        case LOCAL:
            throw std::invalid_argument("Local memory constraint not supported");

        case PERSISTENT_AND_TRANSIENT:

            throw std::invalid_argument("Local memory constraint not supported");
            // TODO: Implement memory constraint for Cilk scheduler
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

    CSchedule schedule(instance.numberOfVertices());

    std::set<unsigned> ready;
    std::vector<unsigned> nrPredecDone(instance.numberOfVertices(), 0);
    std::vector<bool> procFree(instance.numberOfProcessors(), true);
    int nrProcFree = instance.numberOfProcessors();

    std::vector<std::deque<unsigned>> procQueue(instance.numberOfProcessors());
    std::vector<std::deque<unsigned>> greedyProcLists(instance.numberOfProcessors());

    std::set<intPair> finishTimes;
    const intPair start(0, -1);
    finishTimes.insert(start);

    for (const auto &v : instance.getComputationalDag().sourceVertices()) {
        ready.insert(v);
        if (mode == CILK)
            procQueue[0].push_front(v);
    }

    while (!finishTimes.empty()) {
        const int time = finishTimes.begin()->a;

        // Find new ready jobs
        while (!finishTimes.empty() && finishTimes.begin()->a == time) {
            const intPair currentPair = *finishTimes.begin();
            finishTimes.erase(finishTimes.begin());
            const int node = currentPair.b;
            if (node != -1) {

                for (const auto &succ : instance.getComputationalDag().children(node)) {
                    // for (int j = 0; j < G.Out[node].size(); ++j) {

                    ++nrPredecDone[succ];
                    if (nrPredecDone[succ] == instance.getComputationalDag().numberOfParents(succ)) {
                        // G.In[G.Out[node][j]].size()) {
                        ready.insert(succ);
                        if (mode == CILK)
                            procQueue[schedule.proc[node]].push_back(succ);
                    }
                }
                procFree[schedule.proc[node]] = true;
                ++nrProcFree;
            }
        }

        // Assign new jobs to processors
        while (nrProcFree > 0 && !ready.empty()) {
            unsigned nextNode, nextProc;
            Choose(instance, procQueue, ready, procFree, nextNode, nextProc);

            ready.erase(nextNode);
            schedule.proc[nextNode] = nextProc;
            schedule.time[nextNode] = time;

            if (use_memory_constraint) {

                if (instance.getArchitecture().getMemoryConstraintType() == PERSISTENT_AND_TRANSIENT) {

                    current_proc_persistent_memory[nextProc] +=
                        instance.getComputationalDag().nodeMemoryWeight(nextNode);
                    current_proc_transient_memory[nextProc] =
                        std::max(current_proc_transient_memory[nextProc],
                                 instance.getComputationalDag().nodeMemoryWeight(nextNode));
                }
            }

            finishTimes.insert(intPair(time + instance.getComputationalDag().nodeWorkWeight(nextNode), nextNode));
            procFree[nextProc] = false;
            --nrProcFree;

            greedyProcLists[nextProc].push_back(nextNode);
        }
    }

    return {SUCCESS, schedule.convertToBspSchedule(instance, greedyProcLists)};
};

// Choosing a node to assign for classical greedy methods (cilk, SJF, random)
void GreedyCilkScheduler::Choose(const BspInstance &instance, std::vector<std::deque<unsigned>> &procQueue,
                                 const std::set<unsigned> &readyNodes, const std::vector<bool> &procFree,
                                 unsigned &node, unsigned &p) {
    if (mode == SJF) {
        node = *readyNodes.begin();
        for (auto &r : readyNodes)
            if (instance.getComputationalDag().nodeWorkWeight(r) < instance.getComputationalDag().nodeWorkWeight(node))
                node = r;

        p = 0;
        for (; p < instance.numberOfProcessors(); ++p)
            if (procFree[p])
                break;
    } else if (mode == RANDOM) {
        unsigned i = 0, rnd = randInt(readyNodes.size());
        for (auto &r : readyNodes) {
            if (i == rnd) {
                node = r;
                break;
            }
            ++i;
        }
        p = pickRandom(procFree);
    } else if (mode == CILK) {
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
            if (procFree[i] && !procQueue[i].empty()) {
                p = i;
                node = procQueue[i].back();
                procQueue[i].pop_back();
                return;
            }

        // Time to steal
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
            if (procFree[i]) {
                p = i;
                break;
            }

        std::vector<bool> canStealFrom(instance.numberOfProcessors(), false);
        for (unsigned i = 0; i < instance.numberOfProcessors(); ++i)
            canStealFrom[i] = !procQueue[i].empty();

        int chosenQueue = pickRandom(canStealFrom);
        node = procQueue[chosenQueue].front();
        procQueue[chosenQueue].pop_front();
    }
};