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

#include "scheduler/LocalSearchSchedulers/HillClimbingScheduler.hpp"

RETURN_STATUS HillClimbingScheduler::improveSchedule(BspSchedule &input_schedule) {

    return improveScheduleWithTimeLimit(input_schedule, 600);
}

// Main method for hill climbing (with time limit)
RETURN_STATUS HillClimbingScheduler::improveScheduleWithTimeLimit(BspSchedule &input_schedule, const unsigned timeLimit) {

    schedule = &input_schedule;

    std::cout<<schedule->computeCosts()<<" "<<schedule->computeWorkCosts()<<std::endl;

    CreateSupstepLists();
    Init();
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    int counter = 0;
    while (Improve())
        if ((++counter) == 10) {
            counter = 0;
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            if (elapsed >= timeLimit) {
                std::cout << "Hill Climbing was shut down due to time limit." << std::endl;
                break;
            }
        }

    /*
    // check cost calculation, if desired
    schedule->setLazyCommunicationSchedule();
    if(cost != schedule->computeCosts())
        std::cout << "ERROR: Cost calculation in HillClimbing is incorrect!" << std::endl;*/
    
    schedule->setAutoCommunicationSchedule();

    return SUCCESS;
}

// Hill climbing with step limit (designed as an ingredient for multilevel algorithms, no safety checks)
RETURN_STATUS HillClimbingScheduler::improveScheduleWithStepLimit(BspSchedule &input_schedule, const unsigned stepLimit) {

    schedule = &input_schedule;
    
    CreateSupstepLists();
    Init();
    for (int step = 0; step < stepLimit; ++step)
        if (!Improve())
            break;
    
    schedule->setLazyCommunicationSchedule();

    return SUCCESS;
}


void HillClimbingScheduler::Init() {
    if(shrink)
    {
        RemoveNeedlessSupSteps();
        CreateSupstepLists();
    }

    const unsigned N = schedule->getInstance().getComputationalDag().numberOfVertices();
    const unsigned P = schedule->getInstance().getArchitecture().numberOfProcessors();
    const unsigned M = schedule->numberOfSupersteps();
    const auto &G = schedule->getInstance().getComputationalDag();

    // Movement options
    canMove.clear();
    canMove.resize(NumDirections, std::vector<std::vector<bool>>(N, std::vector<bool>(P, false)));
    moveOptions.clear();
    moveOptions.resize(NumDirections);
    movePointer.clear();
    movePointer.resize(NumDirections, std::vector<std::vector<std::list<intPair>::iterator>>(
                                          N, std::vector<std::list<intPair>::iterator>(P)));

    // Value use lists
    succSteps.clear();
    succSteps.resize(N, std::vector<std::map<int, int>>(P));
    for (unsigned node = 0; node < N; ++node)
        for (const auto &succ : G.children(node)) {
            if (succSteps[node][schedule->assignedProcessor(succ)].find(schedule->assignedSuperstep(succ)) ==
                succSteps[node][schedule->assignedProcessor(succ)].end())
                succSteps[node][schedule->assignedProcessor(succ)].insert({schedule->assignedSuperstep(succ), 1});
            else
                succSteps[node][schedule->assignedProcessor(succ)].at(schedule->assignedSuperstep(succ)) += 1;
        }
    
    // Cost data
    workCost.clear();
    workCost.resize(M, std::vector<int>(P, 0));
    sent.clear();
    sent.resize(M - 1, std::vector<int>(P, 0));
    received.clear();
    received.resize(M - 1, std::vector<int>(P, 0));
    commCost.clear();
    commCost.resize(M - 1, std::vector<int>(P));

    workCostList.clear();
    workCostList.resize(M);
    commCostList.clear();
    commCostList.resize(M - 1);
    workCostPointer.clear();
    workCostPointer.resize(M, std::vector<std::set<intPair>::iterator>(P));
    commCostPointer.clear();
    commCostPointer.resize(M - 1, std::vector<std::set<intPair>::iterator>(P));

    // Supstep std::list pointers
    supStepListPointer.clear();
    supStepListPointer.resize(N);
    for (int step = 0; step < M; ++step)
        for (int proc = 0; proc < P; ++proc)
            for (auto it = supsteplists[step][proc].begin(); it != supsteplists[step][proc].end(); ++it)
                supStepListPointer[*it] = it;

    // Compute movement options
    for (int node = 0; node < N; ++node)
        updateNodeMoves(node);

    nextMove.first = 0;
    nextMove.second = moveOptions[0].begin();

    // Compute cost data
    cost = 0;
    for (int step = 0; step < M; ++step) {
        for (int proc = 0; proc < P; ++proc) {
            for (const int node : supsteplists[step][proc])
                workCost[step][proc] += schedule->getInstance().getComputationalDag().nodeWorkWeight(node);

            intPair entry(workCost[step][proc], proc);
            workCostPointer[step][proc] = workCostList[step].insert(entry).first;
        }
        cost += (--workCostList[step].end())->a;
    }

    std::vector<std::vector<bool>> present(N, std::vector<bool>(P, false));
    for (int step = 0; step < M - 1; ++step) {
        for (int proc = 0; proc < P; ++proc)
            for (const int node : supsteplists[step + 1][proc])
                for (const auto &pred : G.parents(node))
                    if (schedule->assignedProcessor(node) != schedule->assignedProcessor(pred) && !present[pred][schedule->assignedProcessor(node)]) {
                        present[pred][schedule->assignedProcessor(node)] = true;
                        sent[step][schedule->assignedProcessor(pred)] +=
                            schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), schedule->assignedProcessor(node));
                        received[step][schedule->assignedProcessor(node)] +=
                            schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), schedule->assignedProcessor(node));
                    }

        for (int proc = 0; proc < P; ++proc) {
            commCost[step][proc] = std::max(sent[step][proc], received[step][proc]);
            intPair entry(commCost[step][proc], proc);
            commCostPointer[step][proc] = commCostList[step].insert(entry).first;
        }
        cost += (int)schedule->getInstance().getArchitecture().communicationCosts() * commCostList[step].rbegin()->a
                + (int)schedule->getInstance().getArchitecture().synchronisationCosts();
    }

    updatePromisingMoves();

    // memory_constraints
    if(use_memory_constraint)
    {
        memory_used.clear();
        memory_used.resize(P, std::vector<int>(M, 0));
        for (unsigned node = 0; node < N; ++node)
            memory_used[schedule->assignedProcessor(node)][schedule->assignedSuperstep(node)] += schedule->getInstance().getComputationalDag().nodeMemoryWeight(node);
    }

};


void HillClimbingScheduler::updatePromisingMoves()
{
    if(!findPromisingMoves)
        return;

    const unsigned P = schedule->getInstance().getArchitecture().numberOfProcessors();
    const auto &G = schedule->getInstance().getComputationalDag();

    promisingMoves.clear();
    for(int node=0; node < schedule->getInstance().getComputationalDag().numberOfVertices(); ++node)
    {
        std::vector<int> nrPredOnProc(P, 0);
        for(const auto &pred : G.parents(node))
            ++nrPredOnProc[schedule->assignedProcessor(pred)];

        int otherProcUsed = 0;
        for(int proc=0; proc<P; ++proc)
            if(schedule->assignedProcessor(node)!=proc && nrPredOnProc[proc]>0)
                ++otherProcUsed;
                
        if(otherProcUsed==1)
            for(int proc=0; proc<P; ++proc)
                if(schedule->assignedProcessor(node)!=proc && nrPredOnProc[proc]>0 && schedule->getInstance().isCompatible(node,proc))
                {
                    promisingMoves.push_back(intTriple(node, proc, EARLIER));
                    promisingMoves.push_back(intTriple(node, proc, AT));
                    promisingMoves.push_back(intTriple(node, proc, LATER));
                }

        std::vector<int> nrSuccOnProc(P, 0);
        for(const auto &succ : G.children(node))
            ++nrSuccOnProc[schedule->assignedProcessor(succ)];

        otherProcUsed = 0;
        for(int proc=0; proc<P; ++proc)
            if(schedule->assignedProcessor(node)!=proc && nrSuccOnProc[proc]>0)
                ++otherProcUsed;

        if(otherProcUsed==1)
            for(int proc=0; proc<P; ++proc)
                if(schedule->assignedProcessor(node)!=proc && nrSuccOnProc[proc]>0 && schedule->getInstance().isCompatible(node,proc))
                {
                    promisingMoves.push_back(intTriple(node, proc, EARLIER));
                    promisingMoves.push_back(intTriple(node, proc, AT));
                    promisingMoves.push_back(intTriple(node, proc, LATER));
                }
        }

    for(int step=0; step < schedule->numberOfSupersteps(); ++step)
    {
        std::list<int> minProcs, maxProcs;
        int minWork=INT_MAX, maxWork=-1;
        for(int proc=0; proc<P; ++proc)
        {
            if(workCost[step][proc]> maxWork)
                maxWork=workCost[step][proc];
            if(workCost[step][proc]< minWork)
                minWork=workCost[step][proc];
        }
        for(int proc=0; proc<P; ++proc)
        {
            if(workCost[step][proc]==minWork)
                minProcs.push_back(proc);
            if(workCost[step][proc]==maxWork)
                maxProcs.push_back(proc);
        }
        for(int to: minProcs)
            for(int from: maxProcs)
                for(int node : supsteplists[step][from])
                    if(schedule->getInstance().isCompatible(node, to))
                        promisingMoves.push_back(intTriple(node,to, AT));
    }
}


// Functions to compute and update the std::list of possible moves
void HillClimbingScheduler::updateNodeMovesEarlier(const int node) {
    if (schedule->assignedSuperstep(node) == 0)
        return;

    std::set<int> predProc; 
    for (const auto &pred : schedule->getInstance().getComputationalDag().parents(node)) {
        if (schedule->assignedSuperstep(pred) == schedule->assignedSuperstep(node))
            return;
        if (schedule->assignedSuperstep(pred) == schedule->assignedSuperstep(node) - 1)
            predProc.insert(schedule->assignedProcessor(pred));
    }

    if (predProc.size() > 1)
        return;

    if (predProc.size() == 1)
        addMoveOption(node, *predProc.begin(), EARLIER);
    else
        for (int proc = 0; proc < schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
            addMoveOption(node, proc, EARLIER);
};

void HillClimbingScheduler::updateNodeMovesAt(const int node) {
    for (const auto &pred : schedule->getInstance().getComputationalDag().parents(node))
        if (schedule->assignedSuperstep(pred) == schedule->assignedSuperstep(node))
            return;

    for (const auto &succ : schedule->getInstance().getComputationalDag().children(node))
        if (schedule->assignedSuperstep(succ) == schedule->assignedSuperstep(node))
            return;

    for (int proc = 0; proc < schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
        if (proc != schedule->assignedProcessor(node))
            addMoveOption(node, proc, AT);
};

void HillClimbingScheduler::updateNodeMovesLater(const int node) {
    if (schedule->assignedSuperstep(node) == schedule->numberOfSupersteps() - 1)
        return;

    std::set<int> succProc;
    for (const auto &succ : schedule->getInstance().getComputationalDag().children(node)) {
        if (schedule->assignedSuperstep(succ) == schedule->assignedSuperstep(node))
            return;
        if (schedule->assignedSuperstep(succ) == schedule->assignedSuperstep(node) + 1)
            succProc.insert(schedule->assignedProcessor(succ));
    }

    if (succProc.size() > 1)
        return;

    if (succProc.size() == 1)
        addMoveOption(node, *succProc.begin(), LATER);
    else
        for (int proc = 0; proc < schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
            addMoveOption(node, proc, LATER);
};

void HillClimbingScheduler::updateNodeMoves(const int node) {
    eraseMoveOptions(node);
    updateNodeMovesEarlier(node);
    updateNodeMovesAt(node);
    updateNodeMovesLater(node);
};

void HillClimbingScheduler::updateMoveOptions(int node, int where)
{
    const auto &G = schedule->getInstance().getComputationalDag();
    
    updateNodeMoves(node);
    if(where==0)
    {
        for(const auto &pred : G.parents(node))
        {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
        }
        for(const auto &succ : G.children(node))
        {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
        }
    }
    if(where==-1)
    {
        for(const auto &pred : G.parents(node))
        {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
            eraseMoveOptionsAt(pred);
            updateNodeMovesAt(pred);
        }
        for(const auto &succ : G.children(node))
        {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
        }
    }
    if(where==1)
    {
        for(const auto &pred : G.parents(node))
        {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
        }
        for(const auto &succ : G.children(node))
        {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
            eraseMoveOptionsAt(succ);
            updateNodeMovesAt(succ);
        }
    }
}

void HillClimbingScheduler::addMoveOption(const int node, const int p, const Direction dir) {
    if (!canMove[dir][node][p] && schedule->getInstance().isCompatible(node, p)) {
        canMove[dir][node][p] = true;
        moveOptions[dir].emplace_back(node, p);
        movePointer[dir][node][p] = --moveOptions[dir].end();
    }
};

void HillClimbingScheduler::eraseMoveOption(int node, int p, Direction dir)
{
    canMove[dir][node][p] = false;
    if(nextMove.first == dir && nextMove.second->a == node && nextMove.second->b == p)
        ++nextMove.second;
    moveOptions[dir].erase(movePointer[dir][node][p]);
}

void HillClimbingScheduler::eraseMoveOptionsEarlier(int node)
{
    for(int proc=0; proc<schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
        if(canMove[EARLIER][node][proc])
            eraseMoveOption(node, proc, EARLIER);
}

void HillClimbingScheduler::eraseMoveOptionsAt(int node)
{
    for(int proc=0; proc<schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
        if(canMove[AT][node][proc])
            eraseMoveOption(node, proc, AT);
}

void HillClimbingScheduler::eraseMoveOptionsLater(int node)
{
    for(int proc=0; proc<schedule->getInstance().getArchitecture().numberOfProcessors(); ++proc)
        if(canMove[LATER][node][proc])
            eraseMoveOption(node, proc, LATER);
}

void HillClimbingScheduler::eraseMoveOptions(int node)
{
    eraseMoveOptionsEarlier(node);
    eraseMoveOptionsAt(node);
    eraseMoveOptionsLater(node);
}

// Compute the cost change incurred by a potential move
int HillClimbingScheduler::moveCostChange(const int node, int p, const int where, stepAuxData &changing) {
    const int step = schedule->assignedSuperstep(node);
    int oldProc = schedule->assignedProcessor(node);
    int change = 0;

    const auto &G = schedule->getInstance().getComputationalDag();

    // Work cost change
    const auto itBest = --workCostList[step].end();
    int maxAfterRemoval = itBest->a;
    if (itBest->b == oldProc) {
        auto itNext = itBest;
        --itNext;
        maxAfterRemoval = std::max(itBest->a - schedule->getInstance().getComputationalDag().nodeWorkWeight(node), itNext->a);
        change -= itBest->a - maxAfterRemoval;
    }

    const int maxBeforeAddition = (where == 0) ? maxAfterRemoval : workCostList[step + where].rbegin()->a;
    if (workCost[step + where][p] + schedule->getInstance().getComputationalDag().nodeWorkWeight(node) > maxBeforeAddition)
        change += workCost[step + where][p] + schedule->getInstance().getComputationalDag().nodeWorkWeight(node) - maxBeforeAddition;

    // Comm cost change
    std::list<intTriple> sentInc, recInc;
    //  -outputs
    if (p != oldProc) {
        for (int j = 0; j < schedule->getInstance().getArchitecture().numberOfProcessors(); ++j) {
            if (succSteps[node][j].empty())
                continue;

            int affectedStep = succSteps[node][j].begin()->first - 1;
            if (j == p) {
                sentInc.emplace_back(affectedStep, oldProc, 
                                     -schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(oldProc, j));
                recInc.emplace_back(affectedStep, p, -schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(oldProc, j));
            } else if (j == oldProc) {
                recInc.emplace_back(affectedStep, oldProc, schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(p, j));
                sentInc.emplace_back(affectedStep, p, schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(p, j));
            } else {
                sentInc.emplace_back(affectedStep, oldProc,
                                     -schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(oldProc, j));
                recInc.emplace_back(affectedStep, j, -schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(oldProc, j));
                sentInc.emplace_back(affectedStep, p, schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(p, j));
                recInc.emplace_back(affectedStep, j, schedule->getInstance().getComputationalDag().nodeCommunicationWeight(node) * (int)schedule->getInstance().getArchitecture().sendCosts(p, j));
            }
        }
    }

    //  -inputs
    if (p == oldProc)
        for (const auto &pred : G.parents(node)) {
            if (schedule->assignedProcessor(pred) == p)
                continue;

            const auto firstUse = *succSteps[pred][p].begin();
            const bool skip = firstUse.first < step || (firstUse.first == step && where >= 0 && firstUse.second > 1);
            if (!skip) {
                sentInc.emplace_back(step - 1, schedule->assignedProcessor(pred),
                                     -schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p));
                recInc.emplace_back(step - 1, p,
                                    -schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p));
                sentInc.emplace_back(step + where - 1, schedule->assignedProcessor(pred),
                                     schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p));
                recInc.emplace_back(step + where - 1, p,
                                    schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p));
            }
        }
    else
        for (const auto &pred : G.parents(node)) {
            // Comm. cost of sending pred to oldProc
            auto firstUse = succSteps[pred][oldProc].begin();
            bool skip = (schedule->assignedProcessor(pred) == oldProc) || firstUse->first < step ||
                        (firstUse->first == step && firstUse->second > 1);
            if (!skip) {
                sentInc.emplace_back(step - 1, schedule->assignedProcessor(pred),
                                     -schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), oldProc));
                recInc.emplace_back(step - 1, oldProc,
                                    -schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), oldProc));
                ++firstUse;
                if (firstUse != succSteps[pred][oldProc].end()) {
                    const int nextStep = firstUse->first;
                    sentInc.emplace_back(nextStep - 1, schedule->assignedProcessor(pred),
                                         schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) *
                                             (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), oldProc));
                    recInc.emplace_back(nextStep - 1, oldProc,
                                        schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) *
                                            (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), oldProc));
                }
            }

            // Comm. cost of sending pred to p
            firstUse = succSteps[pred][p].begin();
            skip = (schedule->assignedProcessor(pred) == p) ||
                   ((firstUse != succSteps[pred][p].end()) && (firstUse->first <= step + where));
            if (!skip) {
                sentInc.emplace_back(step + where - 1, schedule->assignedProcessor(pred),
                                     schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p));
                recInc.emplace_back(step + where - 1, p,
                                    schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p));
                if (firstUse != succSteps[pred][p].end()) {
                    sentInc.emplace_back(firstUse->first - 1, schedule->assignedProcessor(pred),
                                         -schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p));
                    recInc.emplace_back(firstUse->first - 1, p,
                                        -schedule->getInstance().getComputationalDag().nodeCommunicationWeight(pred) * (int)schedule->getInstance().getArchitecture().sendCosts(schedule->assignedProcessor(pred), p));
                }
            }
        }

    //  -process changes
    changing.sentChange.clear();
    changing.recChange.clear();
    std::set<int> affectedSteps;
    for (auto entry : sentInc) {
        affectedSteps.insert(entry.a);
        auto itr = changing.sentChange.find(intPair(entry.a, entry.b));
        if (itr == changing.sentChange.end())
            changing.sentChange.insert({intPair(entry.a, entry.b), entry.c});
        else
            itr->second += entry.c;
    }
    for (auto entry : recInc) {
        affectedSteps.insert(entry.a);
        auto itr = changing.recChange.find(intPair(entry.a, entry.b));
        if (itr == changing.recChange.end())
            changing.recChange.insert({intPair(entry.a, entry.b), entry.c});
        else
            itr->second += entry.c;
    }

    auto itrSent = changing.sentChange.begin(), itrRec = changing.recChange.begin();
    for (const int sstep : affectedSteps) {
        int newMax = 0;
        for (int j = 0; j < schedule->getInstance().getArchitecture().numberOfProcessors(); ++j) {
            int diff = (itrSent != changing.sentChange.end() && itrSent->first.a == sstep && itrSent->first.b == j)
                           ? (itrSent++)->second
                           : 0;
            if (sent[sstep][j] + diff > newMax)
                newMax = sent[sstep][j] + diff;
            diff = (itrRec != changing.recChange.end() && itrRec->first.a == sstep && itrRec->first.b == j)
                       ? (itrRec++)->second
                       : 0;
            if (received[sstep][j] + diff > newMax)
                newMax = received[sstep][j] + diff;
        }
        change += (int)schedule->getInstance().getArchitecture().communicationCosts() * (newMax - commCostList[sstep].rbegin()->a);

        if (HCwithLatency) {
            if (newMax > 0 && commCostList[sstep].rbegin()->a == 0) {
                change += (int)schedule->getInstance().getArchitecture().synchronisationCosts();
            }
            if (newMax == 0 && commCostList[sstep].rbegin()->a > 0) {
                change -= (int)schedule->getInstance().getArchitecture().synchronisationCosts();
                changing.canShrink = true;
            }
        }
    }

    changing.newCost = cost + change;
    return change;
};

// Execute a chosen move, updating the schedule and the data structures
void HillClimbingScheduler::executeMove(const int node, const int newProc, const int where, const stepAuxData &changing) {
    int oldStep = schedule->assignedSuperstep(node), newStep = oldStep + where;
    const int oldProc = schedule->assignedProcessor(node);
    cost = changing.newCost;

    // Work cost change
    workCostList[oldStep].erase(workCostPointer[oldStep][oldProc]);
    workCost[oldStep][oldProc] -= schedule->getInstance().getComputationalDag().nodeWorkWeight(node);
    workCostPointer[oldStep][oldProc] =
        workCostList[oldStep].insert(intPair(workCost[oldStep][oldProc], oldProc)).first;

    workCostList[newStep].erase(workCostPointer[newStep][newProc]);
    workCost[newStep][newProc] += schedule->getInstance().getComputationalDag().nodeWorkWeight(node);
    workCostPointer[newStep][newProc] =
        workCostList[newStep].insert(intPair(workCost[newStep][newProc], newProc)).first;

    // Comm cost change
    for (const auto update : changing.sentChange)
        sent[update.first.a][update.first.b] += update.second;
    for (const auto update : changing.recChange)
        received[update.first.a][update.first.b] += update.second;

    std::set<intPair> toUpdate;
    for (const auto update : changing.sentChange)
        if (std::max(sent[update.first.a][update.first.b], received[update.first.a][update.first.b]) !=
            commCost[update.first.a][update.first.b])
            toUpdate.insert(intPair(update.first.a, update.first.b));

    for (const auto update : changing.recChange)
        if (std::max(sent[update.first.a][update.first.b], received[update.first.a][update.first.b]) !=
            commCost[update.first.a][update.first.b])
            toUpdate.insert(intPair(update.first.a, update.first.b));

    for (const auto update : toUpdate) {
        commCostList[update.a].erase(commCostPointer[update.a][update.b]);
        commCost[update.a][update.b] = std::max(sent[update.a][update.b], received[update.a][update.b]);
        commCostPointer[update.a][update.b] =
            commCostList[update.a].insert(intPair(commCost[update.a][update.b], update.b)).first;
    }

    // update successor lists
    for (const auto &pred : schedule->getInstance().getComputationalDag().parents(node)) {
        auto itr = succSteps[pred][oldProc].find(oldStep);
        if ((--(itr->second)) == 0)
            succSteps[pred][oldProc].erase(itr);

        itr = succSteps[pred][newProc].find(newStep);
        if (itr == succSteps[pred][newProc].end())
            succSteps[pred][newProc].insert({newStep, 1});
        else
            itr->second += 1;
    }

    // memory constraints, if any
    if(use_memory_constraint)
    {
        memory_used[schedule->assignedProcessor(node)][schedule->assignedSuperstep(node)] -= schedule->getInstance().getComputationalDag().nodeMemoryWeight(node);
        memory_used[newProc][newStep] += schedule->getInstance().getComputationalDag().nodeMemoryWeight(node);
    }

    // update data
    schedule->setAssignedProcessor(node, newProc);
    schedule->setAssignedSuperstep(node, newStep);
    supsteplists[oldStep][oldProc].erase(supStepListPointer[node]);
    supsteplists[newStep][newProc].push_back(node);
    supStepListPointer[node] = (--supsteplists[newStep][newProc].end());

    updateMoveOptions(node, where);
};

// Single hill climbing step
bool HillClimbingScheduler::Improve() {
    int bestCost = cost;
    stepAuxData bestMoveData;
    intPair bestMove;
    int bestDir = 0;
    int startingDir = nextMove.first;

    // pre-selected "promising" moves
    while(!promisingMoves.empty() && !steepestAscent)
    {
        intTriple next = promisingMoves.front();
        promisingMoves.pop_front();

        if(!canMove[static_cast<Direction>(next.c)][next.a][next.b])
            continue;
        
        if(use_memory_constraint && violatesMemConstraint(next.a, next.b, next.c-1))
            continue;

        stepAuxData moveData;
        int costDiff = moveCostChange(next.a, next.b, next.c-1, moveData);

        if(costDiff<0)
        {
            executeMove(next.a, next.b, next.c-1, moveData);
            if(shrink && moveData.canShrink)
                Init();
            
            return true;
        }

    }

    // standard moves
    int dir = startingDir;
    while(true)
    {
        bool reachedBeginning = false;
        while(nextMove.second == moveOptions[nextMove.first].end())
        {
            dir = (nextMove.first+1)%NumDirections;
            if(dir == startingDir)
            {
                reachedBeginning = true;
                break;
            }
            nextMove.first = dir;
            nextMove.second = moveOptions[nextMove.first].begin();
        }
        if(reachedBeginning)
            break;

        intPair next = *nextMove.second;
        ++nextMove.second;

        if(use_memory_constraint && violatesMemConstraint(next.a, next.b, dir-1))
            continue;

        stepAuxData moveData;
        int costDiff = moveCostChange(next.a, next.b, dir-1, moveData);

        if(!steepestAscent && costDiff<0)
        {
            executeMove(next.a, next.b, dir-1, moveData);
            if(shrink && moveData.canShrink)
                Init();

            return true;
        }
        else if(cost+costDiff<bestCost)
        {
            bestCost = cost+costDiff;
            bestMove = next;
            bestMoveData = moveData;
            bestDir = dir-1;
        }


    }

    if (bestCost == cost)
        return false;

    executeMove(bestMove.a, bestMove.b, bestDir, bestMoveData);
    if(shrink && bestMoveData.canShrink)
        Init();

    return true;
};

// Check if move violates mem constraints
bool HillClimbingScheduler::violatesMemConstraint(int node, int processor, int where)
{
    if(memory_used[processor][schedule->assignedSuperstep(node)+where]
        + schedule->getInstance().getComputationalDag().nodeMemoryWeight(node) > (int)schedule->getInstance().memoryBound())
        return true;
    
    return false;
}

void HillClimbingScheduler::RemoveNeedlessSupSteps() {

    const unsigned P = schedule->getInstance().getArchitecture().numberOfProcessors();
    const unsigned M = schedule->numberOfSupersteps();
    const auto &G = schedule->getInstance().getComputationalDag();
    
    int current_step = 0;

    auto nextBreak = schedule->numberOfSupersteps();
    for (unsigned step = 0; step < M; ++step) {
        if (nextBreak == step) {
            ++current_step;
            nextBreak = M;
        }
        for (int proc = 0; proc < P; ++proc)
            for (const int node : supsteplists[step][proc]) {
                schedule->setAssignedSuperstep(node, current_step);
                for (const auto &succ : G.children(node))
                    if (schedule->assignedProcessor(node) != schedule->assignedProcessor(succ) && schedule->assignedSuperstep(succ) < nextBreak)
                        nextBreak = schedule->assignedSuperstep(succ);
            }
    }

    schedule->updateNumberOfSupersteps();
};

void HillClimbingScheduler::CreateSupstepLists() {
    
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