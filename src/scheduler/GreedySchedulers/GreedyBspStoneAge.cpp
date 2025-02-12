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

#include "scheduler/GreedySchedulers/GreedyBspStoneAge.hpp"
#include <algorithm>
#include <stdexcept>

std::pair<RETURN_STATUS, BspSchedule> GreedyBspStoneAge::computeSchedule(const BspInstance &instance) {

    const unsigned N = instance.numberOfVertices();
    const unsigned P = instance.numberOfProcessors();
    const ComputationalDag& G = instance.getComputationalDag();

    BspSchedule schedule(instance, std::vector<unsigned>(instance.numberOfVertices(), -1), std::vector<unsigned>(instance.numberOfVertices()));

    std::set<int> ready;

    std::vector<std::set<int> > procReady(P);
    std::set<int> allReady;

    std::vector<int> predec(N, 0);
    std::vector<bool> procFree(P, true);
    int free = P;

    procInHyperedge.clear();
    procInHyperedge.resize(N, std::vector<int>(P, false));

    std::set<std::pair<int, int> > finishTimes;
    finishTimes.emplace(0, -1);

    for(int node=0; node<N; ++node)
        if(G.numberOfParents(node) == 0)
        {
            ready.insert(node);
            allReady.insert(node);
        }

    int supstep = 0;
    bool endSupStep = false;
    while(!ready.empty() || !finishTimes.empty())
    {
        if(finishTimes.empty() && endSupStep)
        {
            for(int i=0; i<P; ++i)
                procReady[i].clear();

            allReady = ready;

            ++supstep;
            endSupStep = false;

            finishTimes.emplace(0, -1);
        }

        int time = finishTimes.begin()->first;

        // Find new ready jobs
        while(!finishTimes.empty() && finishTimes.begin()->first == time)
        {
            auto currentPair = *finishTimes.begin();
            finishTimes.erase(finishTimes.begin());
            int node = currentPair.second;
            if(node!=-1)
            {
                for (const auto &succ : G.children(node))
                {
                    ++predec[succ];
                    if(predec[succ]==G.numberOfParents(succ))
                    {
                        ready.insert(succ);

                        bool canAdd = true;
                        for(const auto &pred : G.parents(succ))
                            if(schedule.assignedProcessor(pred) != schedule.assignedProcessor(node) && schedule.assignedSuperstep(pred) == supstep)
                                canAdd= false;

                        if(canAdd)
                            procReady[schedule.assignedProcessor(node)].insert(succ);
                    }

                }
                procFree[schedule.assignedProcessor(node)] = true;
                ++free;
            }

        }

        if(endSupStep)
            continue;

        //Assign new jobs to processors
        while(true)
        {
            if(!CanChooseNode(instance, allReady, procReady, procFree))
                break;

            int nextNode=-1, nextProc;
            Choose(instance, allReady, procReady, procFree, nextNode, nextProc);

            if(procReady[nextProc].find(nextNode) != procReady[nextProc].end())
                procReady[nextProc].erase(nextNode);
            else
                allReady.erase(nextNode);

            ready.erase(nextNode);
            schedule.setAssignedProcessor(nextNode, nextProc);
            schedule.setAssignedSuperstep(nextNode, supstep);

            finishTimes.emplace(time+G.nodeWorkWeight(nextNode), nextNode);
            procFree[nextProc]=false;
            --free;

            // update comm auxiliary structure
            procInHyperedge[nextNode][nextProc]=true;
            for(const auto &pred : G.parents(nextNode))
                procInHyperedge[pred][nextProc]=true;
        }

        if(allReady.empty() && free>=P/2)
            endSupStep = true;

    }

    schedule.setAutoCommunicationSchedule();

    return {SUCCESS, schedule};
};

void GreedyBspStoneAge::Choose(const BspInstance &instance, const std::set<int> &allReady,
                                const std::vector<std::set<int>> &procReady, const std::vector<bool> &procFree,
                                int &node, int &p) const {
    double maxScore = -1;
    for(int i=0; i<instance.numberOfProcessors(); ++i)
        if(procFree[i] && !procReady[i].empty())
        {
            for(auto& r : procReady[i])
            {
                double score = 0;
                for (const auto &pred : instance.getComputationalDag().parents(r))
                    if(procInHyperedge[pred][i])
                        score += (double)instance.getComputationalDag().nodeCommunicationWeight(pred)/(double)instance.getComputationalDag().numberOfChildren(pred);

                if(score>maxScore)
                {
                    maxScore=score;
                    node = r;
                    p = i;
                }
            }
            return;
        }

    for(auto& r : allReady)
    {
        for(int i=0; i<instance.numberOfProcessors(); ++i)
        {
            if(!procFree[i])
                continue;

            double score = 0;
            for (const auto &pred : instance.getComputationalDag().parents(r))
                if(procInHyperedge[pred][i])
                    score += (double)instance.getComputationalDag().nodeCommunicationWeight(pred)/(double)(double)instance.getComputationalDag().numberOfChildren(pred);


            if(score>maxScore)
            {
                maxScore=score;
                node = r;
                p = i;
            }
        }
    }
};

// auxiliary - check if it is possible to assign a node at all./Ones
bool GreedyBspStoneAge::CanChooseNode(const BspInstance &instance, const std::set<int> &allReady,
                                       const std::vector<std::set<int>> &procReady,
                                       const std::vector<bool> &procFree) const {
    for(int i=0; i<instance.numberOfProcessors(); ++i)
        if(procFree[i] && !procReady[i].empty())
            return true;

    if(!allReady.empty())
        for(int i=0; i<instance.numberOfProcessors(); ++i)
            if(procFree[i])
                return true;

    return false;
};