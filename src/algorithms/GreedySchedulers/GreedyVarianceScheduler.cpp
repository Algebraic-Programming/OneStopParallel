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

#include "algorithms/GreedySchedulers/GreedyVarianceScheduler.hpp"

std::vector<double> GreedyVarianceScheduler::compute_work_variance(const ComputationalDag& graph) const {
    std::vector<double> work_variance(graph.numberOfVertices(), 0.0);

    const std::vector<VertexType> top_order = graph.GetTopOrder();

    for (auto r_iter = top_order.rbegin(); r_iter != top_order.crend(); r_iter++) {
        work_variance[*r_iter] = (double) graph.nodeWorkWeight(*r_iter);
        double temp = 0;
        for (const auto& child : graph.children(*r_iter)) {
            temp += pow(work_variance[child], 2);
        }
        work_variance[*r_iter] += pow(temp, 0.5);
    }

    return work_variance;
}

std::pair<RETURN_STATUS, BspSchedule> GreedyVarianceScheduler::computeSchedule(const BspInstance &instance) {

    const unsigned &N = instance.numberOfVertices();
    const unsigned &params_p = instance.numberOfProcessors();
    const auto &G = instance.getComputationalDag();

    BspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1),
                         std::vector<unsigned>(instance.numberOfVertices()));

    const std::vector<double> work_variances = compute_work_variance(G);

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

                        if (canAdd)
                            procReady[schedule.assignedProcessor(node)].insert(succ);
                    }
                }
                procFree[schedule.assignedProcessor(node)] = true;
                ++free;
            }
        }

        if (endSupStep)
            continue;

        // Assign new jobs to processors
        while (CanChooseNode(instance, allReady, procReady, procFree)) {

            int nextNode = -1, nextProc = -1;
            Choose(instance, work_variances, procInHyperedge, allReady, procReady, procFree, nextNode, nextProc);

            if (procReady[nextProc].find(nextNode) != procReady[nextProc].end())
                procReady[nextProc].erase(nextNode);
            else
                allReady.erase(nextNode);

            ready.erase(nextNode);
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstepIdx);
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

        if (allReady.empty() && free >= params_p / 2)
            endSupStep = true;
    }

    assert(schedule.satisfiesPrecedenceConstraints());

    schedule.setAutoCommunicationSchedule();
    
    return {SUCCESS, schedule};
};

// auxiliary - check if it is possible to assign a node at all
bool GreedyVarianceScheduler::CanChooseNode(const BspInstance &instance, const std::set<int> &allReady,
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

void GreedyVarianceScheduler::Choose(   const BspInstance &instance, const std::vector<double> &work_variance,
                                        const std::vector<std::vector<int>> &procInHyperedge,
                                        const std::set<int> &allReady, const std::vector<std::set<int>> &procReady,
                                        const std::vector<bool> &procFree, int &node, int &p) const {

    double maxScore = -1;
    for (unsigned i = 0; i < instance.numberOfProcessors(); ++i) {
        if (procFree[i] && !procReady[i].empty()) {
            p = i;
            // select node
            for (auto &r : procReady[i]) {
                double score = work_variance[r];
                // for (const auto &pred : instance.getComputationalDag().parents(r)) {
                //     // for (const int pred : schedule.G.In[r]) {
                //     if (procInHyperedge[pred][i])
                //         score += (double)instance.getComputationalDag().nodeCommunicationWeight(pred) /
                //                  (double)instance.getComputationalDag().numberOfChildren(pred);
                // }

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

            double score = work_variance[r];
            // for (const auto &pred : instance.getComputationalDag().parents(r)) {
            //     // for (const int pred : schedule.G.In[r]) {

            //     if (procInHyperedge[pred][i]) {
            //         score += (double)instance.getComputationalDag().nodeCommunicationWeight(pred) /
            //                  (double)instance.getComputationalDag().numberOfChildren(pred);
            //     }
            // }
            if (score > maxScore) {
                maxScore = score;
                node = r;
                p = i;
            }
        }
    }
};