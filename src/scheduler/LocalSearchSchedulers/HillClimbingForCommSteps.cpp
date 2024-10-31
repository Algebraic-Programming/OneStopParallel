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

#include "scheduler/LocalSearchSchedulers/HillClimbingForCommSteps.hpp"

RETURN_STATUS HillClimbingForCommSteps::improveSchedule(BspSchedule &input_schedule) {

    return improveScheduleWithTimeLimit(input_schedule, 180);
}

// Main method for hill climbing (with time limit)
RETURN_STATUS HillClimbingForCommSteps::improveScheduleWithTimeLimit(BspSchedule &input_schedule, const unsigned timeLimit) {

    schedule = &input_schedule;

    if(schedule->numberOfSupersteps() <=2)
        return SUCCESS;

    Init();
    ConvertCommSchedule();
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    std::cout<<"CommX: "<<schedule->computeCosts()-schedule->computeWorkCosts()<<" vs "<<cost<<std::endl;

    int counter = 0;
    while (Improve())
        if ((++counter) == 100) {
            counter = 0;
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            if (elapsed >= timeLimit) {
                std::cout << "Comm. Sched. Hill Climbing was shut down due to time limit." << std::endl;
                break;
            }
        }


    ConvertCommSchedule();

    std::cout<<"CommY: "<<schedule->computeCosts()-schedule->computeWorkCosts()<<std::endl;

    return SUCCESS;

}


// Initialization for comm. schedule hill climbing
void HillClimbingForCommSteps::Init() {
    const unsigned N = schedule->getInstance().getComputationalDag().numberOfVertices();
    const unsigned P = schedule->getInstance().getArchitecture().numberOfProcessors();
    const unsigned M = schedule->numberOfSupersteps();
    const auto &G = schedule->getInstance().getComputationalDag();

    CreateSupstepLists();
    cost = schedule->computeCosts()-schedule->computeWorkCosts();

    nextSupstep = 0;
    commSchedule.clear();
    commSchedule.resize(N, std::vector<int>(P, -1));
    sent.clear();
    sent.resize(M - 1, std::vector<int>(P, 0));
    received.clear();
    received.resize(M - 1, std::vector<int>(P, 0));
    commCost.clear();
    commCost.resize(M - 1, std::vector<int>(P));
    commCostList.clear();
    commCostList.resize(M - 1);
    commCostPointer.clear();
    commCostPointer.resize(M - 1, std::vector<std::set<intPair>::iterator>(P));
    commBounds.clear();
    commBounds.resize(N, std::vector<intPair>(P));
    commSchedSendLists.clear();
    commSchedSendLists.resize(M - 1, std::vector<std::list<intPair>>(P));
    commSchedRecLists.clear();
    commSchedRecLists.resize(M - 1, std::vector<std::list<intPair>>(P));
    commSchedSendListPointer.clear();
    commSchedSendListPointer.resize(N, std::vector<std::list<intPair>::iterator>(P));
    commSchedRecListPointer.clear();
    commSchedRecListPointer.resize(N, std::vector<std::list<intPair>::iterator>(P));

    // initialize to lazy comm schedule first - to make sure it's correct even if e.g. com scehdule has indirect sending
    for (size_t step = 1; step < M; ++step)
        for (int proc = 0; proc < P; ++proc)
            for (const int node : supsteplists[step][proc])
                for (const auto &pred : G.parents(node))
                    if (schedule->assignedProcessor(pred) != schedule->assignedProcessor(node) &&
                        commSchedule[pred][schedule->assignedProcessor(node)] == -1) {
                            commSchedule[pred][schedule->assignedProcessor(node)] = step - 1;
                            commBounds[pred][schedule->assignedProcessor(node)] = intPair(schedule->assignedSuperstep(pred), step - 1);
                    }

    // overwrite with original comm schedule, wherever possible
    const std::map<KeyTriple, unsigned int> originalCommSchedule = schedule->getCommunicationSchedule();
    for(unsigned node = 0; node < N; ++node)
        for (int proc = 0; proc < P; ++proc)
        {
            if(commSchedule[node][proc] == -1 )
                continue;
            
            const auto comm_schedule_key = std::make_tuple(node, schedule->assignedProcessor(node), proc);
            auto mapIterator = originalCommSchedule.find(comm_schedule_key);
            if (mapIterator != originalCommSchedule.end())
            {
                int originalStep = mapIterator->second;
                if(originalStep >= commBounds[node][proc].a && originalStep <= commBounds[node][proc].b)
                    commSchedule[node][proc] = originalStep;
            }

            unsigned step = commSchedule[node][proc];
            commSchedSendLists[step][schedule->assignedProcessor(node)].emplace_front(node, proc);
            commSchedSendListPointer[node][proc] =
                commSchedSendLists[step][schedule->assignedProcessor(node)].begin();
            commSchedRecLists[step][proc].emplace_front(node, proc);
            commSchedRecListPointer[node][proc] =
                commSchedRecLists[step][proc].begin();
            
            sent[step][schedule->assignedProcessor(node)] +=
                            (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(node), proc);
            received[step][proc] +=
                            (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(node), proc);
                    

        }
    
    for (int step = 0; step < M - 1; ++step)
        for (int proc = 0; proc < P; ++proc)
        {
            commCost[step][proc] = std::max(sent[step][proc], received[step][proc]);
            intPair entry(commCost[step][proc], proc);
            commCostPointer[step][proc] = commCostList[step].insert(entry).first;
        }
};

// compute cost change incurred by a potential move
int HillClimbingForCommSteps::moveCostChange(const int node, const int p, const int step) {
    const int oldStep = commSchedule[node][p];
    const int sourceProc = schedule->assignedProcessor(node);
    int change = 0;

    // Change at old place
    auto itr = commCostList[oldStep].rbegin();
    int oldMax = itr->a;
    const int maxSource =
        std::max(sent[oldStep][sourceProc] - (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(sourceProc, p),
                 received[oldStep][sourceProc]);
    const int maxTarget = std::max(sent[oldStep][p],
                                received[oldStep][p] - (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(sourceProc, p));
    int maxOther = 0;
    for (; itr != commCostList[oldStep].rend(); ++itr)
        if (itr->b != sourceProc && itr->b != p) {
            maxOther = itr->a;
            break;
        }

    int newMax = std::max(std::max(maxSource, maxTarget), maxOther);
    change += (newMax - oldMax) * (int)schedule->getInstance().getArchitecture().communicationCosts();
    if(newMax==0)
        change -= (int)schedule->getInstance().getArchitecture().synchronisationCosts();

    // Change at new place
    oldMax = commCostList[step].rbegin()->a;
    newMax = std::max(
        std::max(oldMax, sent[step][sourceProc] + (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(sourceProc, p)),
        received[step][p] + (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(sourceProc, p));
    change += (newMax - oldMax) * (int)schedule->getInstance().getArchitecture().communicationCosts();
    if(oldMax==0)
        change += (int)schedule->getInstance().getArchitecture().synchronisationCosts();

    return change;
};

// execute a move, updating the comm. schedule and the data structures
void HillClimbingForCommSteps::executeMove(int node, int p, const int step, const int changeCost) {
    const int oldStep = commSchedule[node][p];
    const int sourceProc = schedule->assignedProcessor(node);
    cost += changeCost;

    unsigned max_comm0 = 0;

    // Old step update
    if (sent[oldStep][sourceProc] > received[oldStep][sourceProc]) {
        commCostList[oldStep].erase(commCostPointer[oldStep][sourceProc]);
        sent[oldStep][sourceProc] -= (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(sourceProc, p);
        commCost[oldStep][sourceProc] = std::max(sent[oldStep][sourceProc], received[oldStep][sourceProc]);
        commCostPointer[oldStep][sourceProc] =
            commCostList[oldStep].insert(intPair(commCost[oldStep][sourceProc], sourceProc)).first;
    } else
        sent[oldStep][sourceProc] -= (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(sourceProc, p);

    if (received[oldStep][p] > sent[oldStep][p]) {
        commCostList[oldStep].erase(commCostPointer[oldStep][p]);
        received[oldStep][p] -= (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(sourceProc, p);
        commCost[oldStep][p] = std::max(sent[oldStep][p], received[oldStep][p]);
        commCostPointer[oldStep][p] = commCostList[oldStep].insert(intPair(commCost[oldStep][p], p)).first;
    } else
        received[oldStep][p] -= (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(sourceProc, p);

    // New step update
    sent[step][sourceProc] += (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(sourceProc, p);
    if (sent[step][sourceProc] > received[step][sourceProc]) {
        commCostList[step].erase(commCostPointer[step][sourceProc]);
        commCost[step][sourceProc] = sent[step][sourceProc];
        commCostPointer[step][sourceProc] =
            commCostList[step].insert(intPair(commCost[step][sourceProc], sourceProc)).first;
    }

    received[step][p] += (int)schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(sourceProc, p);
    if (received[step][p] > sent[step][p]) {
        commCostList[step].erase(commCostPointer[step][p]);
        commCost[step][p] = received[step][p];
        commCostPointer[step][p] = commCostList[step].insert(intPair(commCost[step][p], p)).first;
    }

    // CommSched update
    commSchedule[node][p] = step;

    // Comm lists
    commSchedSendLists[oldStep][sourceProc].erase(commSchedSendListPointer[node][p]);
    commSchedSendLists[step][sourceProc].emplace_front(node, p);
    commSchedSendListPointer[node][p] = commSchedSendLists[step][sourceProc].begin();

    commSchedRecLists[oldStep][p].erase(commSchedRecListPointer[node][p]);
    commSchedRecLists[step][p].emplace_front(node, p);
    commSchedRecListPointer[node][p] = commSchedRecLists[step][p].begin();
};

// Single comm. schedule hill climbing step
bool HillClimbingForCommSteps::Improve() {
    const int M = schedule->numberOfSupersteps();
    int bestNode = 0, bestProc = 0, bestStep = 0, bestDiff = 0;
    int startingSupstep = nextSupstep;

    // iterate over supersteps
    while(true) {
        auto itr = commCostList[nextSupstep].rbegin();

        if (itr == commCostList[nextSupstep].crend())
            break;

        // find maximal comm cost that dominates the h-relation
        const int commMax = itr->a;
        if (commMax == 0)
        {
            nextSupstep = (nextSupstep+1)%(M-1);
            if(nextSupstep == startingSupstep)
                break;
            else
                continue;
        }

        // go over all processors that incur this maximal comm cost in superstep nextSupstep
        for (; itr != commCostList[nextSupstep].rend() && itr->a == commMax; ++itr) {
            const int maxProc = itr->b;

            if (sent[nextSupstep][maxProc] == commMax)
                for (const intPair entry : commSchedSendLists[nextSupstep][maxProc]) {
                    const int node = entry.a;
                    const int p = entry.b;
                    // iterate over alternative supsteps to place this communication step
                    for (int step = commBounds[node][p].a; step < commBounds[node][p].b; ++step) {
                        if (step == commSchedule[node][p])
                            continue;

                        const int costDiff = moveCostChange(node, p, step);

                        if (!steepestAscent && costDiff < 0) {
                            executeMove(node, p, step, costDiff);
                            return true;
                        } else if (costDiff < bestDiff) {
                            bestNode = node;
                            bestProc = p;
                            bestStep = step;
                            bestDiff = costDiff;
                        }
                    }
                }

            if (received[nextSupstep][maxProc] == commMax)
                for (const intPair entry : commSchedRecLists[nextSupstep][maxProc]) {
                    const int node = entry.a;
                    const int p = entry.b;
                    // iterate over alternative supsteps to place this communication step
                    for (int step = commBounds[node][p].a; step < commBounds[node][p].b; ++step) {
                        if (step == commSchedule[node][p])
                            continue;

                        const int costDiff = moveCostChange(node, p, step);

                        if (!steepestAscent && costDiff < 0) {
                            executeMove(node, p, step, costDiff);
                            return true;
                        }
                        if (costDiff < bestDiff) {
                            bestNode = node;
                            bestProc = p;
                            bestStep = step;
                            bestDiff = costDiff;
                        }
                    }
                }
        }

        nextSupstep = (nextSupstep+1)%(M-1);
        if(nextSupstep == startingSupstep)
            break;
    }

    if (bestDiff == 0)
        return false;

    executeMove(bestNode, bestProc, bestStep, bestDiff);

    return true;
};

void HillClimbingForCommSteps::CreateSupstepLists() {
    
    const unsigned P = schedule->getInstance().getArchitecture().numberOfProcessors();
    const unsigned N = schedule->getInstance().getComputationalDag().numberOfVertices();
    const auto &G = schedule->getInstance().getComputationalDag();

    schedule->updateNumberOfSupersteps();
    const unsigned M = schedule->numberOfSupersteps();

    supsteplists.clear();
    supsteplists.resize(M, std::vector<std::list<int>>(P));

    const auto topOrder = G.GetTopOrder();
    for (unsigned node : topOrder)
        supsteplists[schedule->assignedSuperstep(node)][schedule->assignedProcessor(node)].push_back(node);

};

void HillClimbingForCommSteps::ConvertCommSchedule()
{
    const unsigned N = schedule->getInstance().getComputationalDag().numberOfVertices();
    const unsigned P = schedule->getInstance().getArchitecture().numberOfProcessors();

    std::map<KeyTriple, unsigned int> newCommSchedule;

    for(unsigned node=0; node < N; ++node)
        for(unsigned proc=0; proc < P; ++proc)
            if(commSchedule[node][proc]>=0)
            {
                const auto comm_schedule_key = std::make_tuple(node, schedule->assignedProcessor(node), proc);
                newCommSchedule[comm_schedule_key] = commSchedule[node][proc];
            }

    schedule->setCommunicationSchedule(newCommSchedule);
};