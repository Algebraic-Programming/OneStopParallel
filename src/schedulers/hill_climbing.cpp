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

#include <iostream>

#include "schedulers/hill_climbing.hpp"
#include "structures/schedule.hpp"

void HillClimbing::Init() {
    const int N = schedule.G.n;
    const int M = schedule.supsteplists.size();
    schedule.cost = schedule.GetCost();

    // Movement options
    canMove.clear();
    canMove.resize(NumDirections, std::vector<std::vector<bool>>(N, std::vector<bool>(schedule.params.p, false)));
    moveOptions.clear();
    moveOptions.resize(NumDirections);
    movePointer.clear();
    movePointer.resize(NumDirections, std::vector<std::vector<std::list<intPair>::iterator>>(
                                          N, std::vector<std::list<intPair>::iterator>(schedule.params.p)));

    // Value use lists
    succSteps.clear();
    succSteps.resize(N, std::vector<std::map<int, int>>(schedule.params.p));
    for (unsigned i = 0; i < N; ++i)
        for (const int succ : schedule.G.Out[i]) {
            if (succSteps[i][schedule.proc[succ]].find(schedule.supstep[succ]) ==
                succSteps[i][schedule.proc[succ]].end())
                succSteps[i][schedule.proc[succ]].insert({schedule.supstep[succ], 1});
            else
                succSteps[i][schedule.proc[succ]].at(schedule.supstep[succ]) += 1;
        }

    // Cost data
    workCost.clear();
    workCost.resize(M, std::vector<int>(schedule.params.p, 0));
    sent.clear();
    sent.resize(M - 1, std::vector<int>(schedule.params.p, 0));
    received.clear();
    received.resize(M - 1, std::vector<int>(schedule.params.p, 0));
    commCost.clear();
    commCost.resize(M - 1, std::vector<int>(schedule.params.p));

    workCostList.clear();
    workCostList.resize(M);
    commCostList.clear();
    commCostList.resize(M - 1);
    workCostPointer.clear();
    workCostPointer.resize(M, std::vector<std::set<intPair>::iterator>(schedule.params.p));
    commCostPointer.clear();
    commCostPointer.resize(M - 1, std::vector<std::set<intPair>::iterator>(schedule.params.p));

    // Supstep std::list pointers
    supStepListPointer.clear();
    supStepListPointer.resize(N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < schedule.params.p; ++j)
            for (auto it = schedule.supsteplists[i][j].begin(); it != schedule.supsteplists[i][j].end(); ++it)
                supStepListPointer[*it] = it;

    // Compute movement options
    for (int i = 0; i < N; ++i)
        updateNodeMoves(i);

    nextMove.first = 0;
    nextMove.second = moveOptions[0].begin();

    // Compute cost data
    schedule.cost = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < schedule.params.p; ++j) {
            for (const int node : schedule.supsteplists[i][j])
                workCost[i][j] += schedule.G.workW[node];

            intPair entry(workCost[i][j], j);
            workCostPointer[i][j] = workCostList[i].insert(entry).first;
        }
        schedule.cost += (--workCostList[i].end())->a;
    }

    std::vector<std::vector<bool>> present(N, std::vector<bool>(schedule.params.p, false));
    for (int i = 0; i < M - 1; ++i) {
        for (int j = 0; j < schedule.params.p; ++j)
            for (const int node : schedule.supsteplists[i + 1][j])
                for (const int pred : schedule.G.In[node])
                    if (schedule.proc[node] != schedule.proc[pred] && !present[pred][schedule.proc[node]]) {
                        present[pred][schedule.proc[node]] = true;
                        sent[i][schedule.proc[pred]] +=
                            schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][schedule.proc[node]];
                        received[i][schedule.proc[node]] +=
                            schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][schedule.proc[node]];
                    }

        for (int j = 0; j < schedule.params.p; ++j) {
            commCost[i][j] = std::max(sent[i][j], received[i][j]);
            intPair entry(commCost[i][j], j);
            commCostPointer[i][j] = commCostList[i].insert(entry).first;
        }
        schedule.cost += schedule.params.g * commCostList[i].rbegin()->a + schedule.params.L;
    }

    updatePromisingMoves();
};


void HillClimbing::updatePromisingMoves()
{
    if(!findPromisingMoves)
        return;

    promisingMoves.clear();
    for(int i=0; i<schedule.G.n; ++i)
    {
        std::vector<int> nrPredOnProc(schedule.params.p, 0);
        for(int pred : schedule.G.In[i])
            ++nrPredOnProc[schedule.proc[pred]];

        int otherProcUsed = 0;
        for(int j=0; j<schedule.params.p; ++j)
            if(schedule.proc[i]!=j && nrPredOnProc[j]>0)
                ++otherProcUsed;
                
        if(otherProcUsed==1)
            for(int j=0; j<schedule.params.p; ++j)
                if(schedule.proc[i]!=j && nrPredOnProc[j]>0)
                {
                    promisingMoves.push_back(intTriple(i, j, EARLIER));
                    promisingMoves.push_back(intTriple(i, j, AT));
                    promisingMoves.push_back(intTriple(i, j, LATER));
                }

        std::vector<int> nrSuccOnProc(schedule.params.p, 0);
        for(int succ : schedule.G.Out[i])
            ++nrSuccOnProc[schedule.proc[succ]];

        otherProcUsed = 0;
        for(int j=0; j<schedule.params.p; ++j)
            if(schedule.proc[i]!=j && nrSuccOnProc[j]>0)
                ++otherProcUsed;

        if(otherProcUsed==1)
            for(int j=0; j<schedule.params.p; ++j)
                if(schedule.proc[i]!=j && nrSuccOnProc[j]>0)
                {
                    promisingMoves.push_back(intTriple(i, j, EARLIER));
                    promisingMoves.push_back(intTriple(i, j, AT));
                    promisingMoves.push_back(intTriple(i, j, LATER));
                }
        }

    for(int i=0; i<schedule.supsteplists.size(); ++i)
    {
        std::list<int> minProcs, maxProcs;
        int minWork=INT_MAX, maxWork=-1;
        for(int j=0; j<schedule.params.p; ++j)
        {
            if(workCost[i][j]> maxWork)
                maxWork=workCost[i][j];
            if(workCost[i][j]< minWork)
                minWork=workCost[i][j];
        }
        for(int j=0; j<schedule.params.p; ++j)
        {
            if(workCost[i][j]==minWork)
                minProcs.push_back(j);
            if(workCost[i][j]==maxWork)
                maxProcs.push_back(j);
        }
        for(int to: minProcs)
            for(int from: maxProcs)
                for(int node : schedule.supsteplists[i][from])
                    promisingMoves.push_back(intTriple(node,to, AT));
    }
}


// Functions to compute and update the std::list of possible moves
void HillClimbing::updateNodeMovesEarlier(const int node) {
    if (schedule.supstep[node] == 0)
        return;

    std::set<int> predProc;
    for (const int pred : schedule.G.In[node]) {
        if (schedule.supstep[pred] == schedule.supstep[node])
            return;
        if (schedule.supstep[pred] == schedule.supstep[node] - 1)
            predProc.insert(schedule.proc[pred]);
    }

    if (predProc.size() > 1)
        return;

    if (predProc.size() == 1)
        addMoveOption(node, *predProc.begin(), EARLIER);
    else
        for (int j = 0; j < schedule.params.p; ++j)
            addMoveOption(node, j, EARLIER);
};

void HillClimbing::updateNodeMovesAt(const int node) {
    for (const int pred : schedule.G.In[node])
        if (schedule.supstep[pred] == schedule.supstep[node])
            return;

    for (const int succ : schedule.G.Out[node])
        if (schedule.supstep[succ] == schedule.supstep[node])
            return;

    for (int j = 0; j < schedule.params.p; ++j)
        if (j != schedule.proc[node])
            addMoveOption(node, j, AT);
};

void HillClimbing::updateNodeMovesLater(const int node) {
    if (schedule.supstep[node] == schedule.supsteplists.size() - 1)
        return;

    std::set<int> succProc;
    for (const int succ : schedule.G.Out[node]) {
        if (schedule.supstep[succ] == schedule.supstep[node])
            return;
        if (schedule.supstep[succ] == schedule.supstep[node] + 1)
            succProc.insert(schedule.proc[succ]);
    }

    if (succProc.size() > 1)
        return;

    if (succProc.size() == 1)
        addMoveOption(node, *succProc.begin(), LATER);
    else
        for (int j = 0; j < schedule.params.p; ++j)
            addMoveOption(node, j, LATER);
};

void HillClimbing::updateNodeMoves(const int node) {
    eraseMoveOptions(node);
    updateNodeMovesEarlier(node);
    updateNodeMovesAt(node);
    updateNodeMovesLater(node);
};

void HillClimbing::updateMoveOptions(int node, int where)
{
    updateNodeMoves(node);
    if(where==0)
    {
        for(int pred : schedule.G.In[node])
        {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
        }
        for(int succ : schedule.G.Out[node])
        {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
        }
    }
    if(where==-1)
    {
        for(int pred : schedule.G.In[node])
        {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
            eraseMoveOptionsAt(pred);
            updateNodeMovesAt(pred);
        }
        for(int succ : schedule.G.Out[node])
        {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
        }
    }
    if(where==1)
    {
        for(int pred : schedule.G.In[node])
        {
            eraseMoveOptionsLater(pred);
            updateNodeMovesLater(pred);
        }
        for(int succ : schedule.G.Out[node])
        {
            eraseMoveOptionsEarlier(succ);
            updateNodeMovesEarlier(succ);
            eraseMoveOptionsAt(succ);
            updateNodeMovesAt(succ);
        }
    }
}

void HillClimbing::addMoveOption(const int node, const int p, const Direction dir) {
    if (!canMove[dir][node][p]) {
        canMove[dir][node][p] = true;
        moveOptions[dir].emplace_back(node, p);
        movePointer[dir][node][p] = --moveOptions[dir].end();
    }
};

void HillClimbing::eraseMoveOption(int node, int p, Direction dir)
{
    canMove[dir][node][p] = false;
    if(nextMove.first == dir && nextMove.second->a == node && nextMove.second->b == p)
        ++nextMove.second;
    moveOptions[dir].erase(movePointer[dir][node][p]);
}

void HillClimbing::eraseMoveOptionsEarlier(int node)
{
    for(int j=0; j<schedule.params.p; ++j)
        if(canMove[EARLIER][node][j])
            eraseMoveOption(node, j, EARLIER);
}

void HillClimbing::eraseMoveOptionsAt(int node)
{
    for(int j=0; j<schedule.params.p; ++j)
        if(canMove[AT][node][j])
            eraseMoveOption(node, j, AT);
}

void HillClimbing::eraseMoveOptionsLater(int node)
{
    for(int j=0; j<schedule.params.p; ++j)
        if(canMove[LATER][node][j])
            eraseMoveOption(node, j, LATER);
}

void HillClimbing::eraseMoveOptions(int node)
{
    eraseMoveOptionsEarlier(node);
    eraseMoveOptionsAt(node);
    eraseMoveOptionsLater(node);
}

// Compute the cost change incurred by a potential move
int HillClimbing::moveCostChange(const int node, int p, const int where, stepAuxData &changing) {
    const int step = schedule.supstep[node];
    int oldProc = schedule.proc[node];
    int change = 0;

    // Work cost change
    const auto itBest = --workCostList[step].end();
    int maxAfterRemoval = itBest->a;
    if (itBest->b == oldProc) {
        auto itNext = itBest;
        --itNext;
        maxAfterRemoval = std::max(itBest->a - schedule.G.workW[node], itNext->a);
        change -= itBest->a - maxAfterRemoval;
    }

    const int maxBeforeAddition = (where == 0) ? maxAfterRemoval : workCostList[step + where].rbegin()->a;
    if (workCost[step + where][p] + schedule.G.workW[node] > maxBeforeAddition)
        change += workCost[step + where][p] + schedule.G.workW[node] - maxBeforeAddition;

    // Comm cost change
    std::list<intTriple> sentInc, recInc;
    //  -outputs
    if (p != oldProc) {
        for (int j = 0; j < schedule.params.p; ++j) {
            if (succSteps[node][j].empty())
                continue;

            int affectedStep = succSteps[node][j].begin()->first - 1;
            if (j == p) {
                sentInc.emplace_back(affectedStep, oldProc,
                                     -schedule.G.commW[node] * schedule.params.sendCost[oldProc][j]);
                recInc.emplace_back(affectedStep, p, -schedule.G.commW[node] * schedule.params.sendCost[oldProc][j]);
            } else if (j == oldProc) {
                recInc.emplace_back(affectedStep, oldProc, schedule.G.commW[node] * schedule.params.sendCost[p][j]);
                sentInc.emplace_back(affectedStep, p, schedule.G.commW[node] * schedule.params.sendCost[p][j]);
            } else {
                sentInc.emplace_back(affectedStep, oldProc,
                                     -schedule.G.commW[node] * schedule.params.sendCost[oldProc][j]);
                recInc.emplace_back(affectedStep, j, -schedule.G.commW[node] * schedule.params.sendCost[oldProc][j]);
                sentInc.emplace_back(affectedStep, p, schedule.G.commW[node] * schedule.params.sendCost[p][j]);
                recInc.emplace_back(affectedStep, j, schedule.G.commW[node] * schedule.params.sendCost[p][j]);
            }
        }
    }

    //  -inputs
    if (p == oldProc)
        for (const int pred : schedule.G.In[node]) {
            if (schedule.proc[pred] == p)
                continue;

            const auto firstUse = *succSteps[pred][p].begin();
            const bool skip = firstUse.first < step || (firstUse.first == step && where >= 0 && firstUse.second > 1);
            if (!skip) {
                sentInc.emplace_back(step - 1, schedule.proc[pred],
                                     -schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][p]);
                recInc.emplace_back(step - 1, p,
                                    -schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][p]);
                sentInc.emplace_back(step + where - 1, schedule.proc[pred],
                                     schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][p]);
                recInc.emplace_back(step + where - 1, p,
                                    schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][p]);
            }
        }
    else
        for (const int pred : schedule.G.In[node]) {
            // Comm. cost of sending pred to oldProc
            auto firstUse = succSteps[pred][oldProc].begin();
            bool skip = (schedule.proc[pred] == oldProc) || firstUse->first < step ||
                        (firstUse->first == step && firstUse->second > 1);
            if (!skip) {
                sentInc.emplace_back(step - 1, schedule.proc[pred],
                                     -schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][oldProc]);
                recInc.emplace_back(step - 1, oldProc,
                                    -schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][oldProc]);
                ++firstUse;
                if (firstUse != succSteps[pred][oldProc].end()) {
                    const int nextStep = firstUse->first;
                    sentInc.emplace_back(nextStep - 1, schedule.proc[pred],
                                         schedule.G.commW[pred] *
                                             schedule.params.sendCost[schedule.proc[pred]][oldProc]);
                    recInc.emplace_back(nextStep - 1, oldProc,
                                        schedule.G.commW[pred] *
                                            schedule.params.sendCost[schedule.proc[pred]][oldProc]);
                }
            }

            // Comm. cost of sending pred to p
            firstUse = succSteps[pred][p].begin();
            skip = (schedule.proc[pred] == p) ||
                   ((firstUse != succSteps[pred][p].end()) && (firstUse->first <= step + where));
            if (!skip) {
                sentInc.emplace_back(step + where - 1, schedule.proc[pred],
                                     schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][p]);
                recInc.emplace_back(step + where - 1, p,
                                    schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][p]);
                if (firstUse != succSteps[pred][p].end()) {
                    sentInc.emplace_back(firstUse->first - 1, schedule.proc[pred],
                                         -schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][p]);
                    recInc.emplace_back(firstUse->first - 1, p,
                                        -schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][p]);
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
        for (int j = 0; j < schedule.params.p; ++j) {
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
        change += schedule.params.g * (newMax - commCostList[sstep].rbegin()->a);

        if (HCwithLatency) {
            if (newMax > 0 && commCostList[sstep].rbegin()->a == 0) {
                change += schedule.params.L;
            }
            if (newMax == 0 && commCostList[sstep].rbegin()->a > 0) {
                change -= schedule.params.L;
                changing.canShrink = true;
            }
        }
    }

    changing.newCost = schedule.cost + change;
    return change;
};

// Execute a chosen move, updating the schedule and the data structures
void HillClimbing::executeMove(const int node, const int newProc, const int where, const stepAuxData &changing) {
    int oldStep = schedule.supstep[node], newStep = oldStep + where;
    const int oldProc = schedule.proc[node];
    schedule.cost = changing.newCost;

    // Work cost change
    workCostList[oldStep].erase(workCostPointer[oldStep][oldProc]);
    workCost[oldStep][oldProc] -= schedule.G.workW[node];
    workCostPointer[oldStep][oldProc] =
        workCostList[oldStep].insert(intPair(workCost[oldStep][oldProc], oldProc)).first;

    workCostList[newStep].erase(workCostPointer[newStep][newProc]);
    workCost[newStep][newProc] += schedule.G.workW[node];
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
    for (const int pred : schedule.G.In[node]) {
        auto itr = succSteps[pred][oldProc].find(oldStep);
        if ((--(itr->second)) == 0)
            succSteps[pred][oldProc].erase(itr);

        itr = succSteps[pred][newProc].find(newStep);
        if (itr == succSteps[pred][newProc].end())
            succSteps[pred][newProc].insert({newStep, 1});
        else
            itr->second += 1;
    }

    // update data
    schedule.proc[node] = newProc;
    schedule.supstep[node] = newStep;
    schedule.supsteplists[oldStep][oldProc].erase(supStepListPointer[node]);
    schedule.supsteplists[newStep][newProc].push_back(node);
    supStepListPointer[node] = (--schedule.supsteplists[newStep][newProc].end());

    updateMoveOptions(node, where);
};

// Single hill climbing step
bool HillClimbing::Improve(const bool SteepestAscent, const bool shrink) {
    int bestCost = schedule.cost;
    stepAuxData bestMoveData;
    intPair bestMove;
    int bestDir = 0;
    int startingDir = nextMove.first;

    // pre-selected "promising" moves
    while(!promisingMoves.empty() && !SteepestAscent)
    {
        intTriple next = promisingMoves.front();
        promisingMoves.pop_front();

        if(!canMove[static_cast<Direction>(next.c)][next.a][next.b])
            continue;

        stepAuxData moveData;
        int costDiff = moveCostChange(next.a, next.b, next.c-1, moveData);

        if(costDiff<0)
        {
            executeMove(next.a, next.b, next.c-1, moveData);
            if(shrink && moveData.canShrink)
            {
                schedule.RemoveNeedlessSupSteps();
                Init();
            }
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

        stepAuxData moveData;
        int costDiff = moveCostChange(next.a, next.b, dir-1, moveData);

        if(!SteepestAscent && costDiff<0)
        {
            executeMove(next.a, next.b, dir-1, moveData);
            if(shrink && moveData.canShrink)
            {
                schedule.RemoveNeedlessSupSteps();
                Init();
            }

            return true;
        }
        else if(schedule.cost+costDiff<bestCost)
        {
            bestCost = schedule.cost+costDiff;
            bestMove = next;
            bestMoveData = moveData;
            bestDir = dir-1;
        }


    }

    if (bestCost == schedule.cost)
        return false;

    executeMove(bestMove.a, bestMove.b, bestDir, bestMoveData);
    if(shrink && bestMoveData.canShrink)
    {
        schedule.RemoveNeedlessSupSteps();
        Init();
    }

    return true;
};

// Main method for hill climbing (with time limit)
void HillClimbing::HillClimb(const int TimeLimit, const bool SteepestAscent, const bool shrink) {
    if(shrink)
        schedule.RemoveNeedlessSupSteps();
    
    Init();
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    int counter = 0;
    while (Improve(SteepestAscent, shrink))
        if ((++counter) == 10) {
            counter = 0;
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            if (elapsed >= TimeLimit) {
                std::cout << "Hill Climbing was shut down due to time limit." << std::endl;
                break;
            }
        }

    if(schedule.cost != schedule.GetCost())
        std::cout << "ERROR: Cost calculation in HillClimbing is incorrect!" << std::endl;
};

// Hill climbing for limited number of steps
void HillClimbing::HillClimbSteps(const int StepsLimit, const bool SteepestAscent, const bool shrink) {
    Init();
    for (int i = 0; i < StepsLimit; ++i)
        if (!Improve(SteepestAscent, shrink))
            break;
};

// Initialization for comm. schedule hill climbing
void HillClimbingCS::Init() {
    const int M = schedule.supsteplists.size();
    nextSupstep = 0;
    schedule.commSchedule.clear();
    schedule.commSchedule.resize(schedule.G.n, std::vector<int>(schedule.params.p, -1));
    sent.clear();
    sent.resize(M - 1, std::vector<int>(schedule.params.p, 0));
    received.clear();
    received.resize(M - 1, std::vector<int>(schedule.params.p, 0));
    commCost.clear();
    commCost.resize(M - 1, std::vector<int>(schedule.params.p));
    commCostList.clear();
    commCostList.resize(M - 1);
    commCostPointer.clear();
    commCostPointer.resize(M - 1, std::vector<std::set<intPair>::iterator>(schedule.params.p));
    commBounds.clear();
    commBounds.resize(schedule.G.n, std::vector<intPair>(schedule.params.p));
    commSchedSendLists.clear();
    commSchedSendLists.resize(M - 1, std::vector<std::list<intPair>>(schedule.params.p));
    commSchedRecLists.clear();
    commSchedRecLists.resize(M - 1, std::vector<std::list<intPair>>(schedule.params.p));
    commSchedSendListPointer.clear();
    commSchedSendListPointer.resize(schedule.G.n, std::vector<std::list<intPair>::iterator>(schedule.params.p));
    commSchedRecListPointer.clear();
    commSchedRecListPointer.resize(schedule.G.n, std::vector<std::list<intPair>::iterator>(schedule.params.p));

    for (size_t i = 1; i < schedule.supsteplists.size(); ++i)
        for (int j = 0; j < schedule.params.p; ++j)
            for (const int node : schedule.supsteplists[i][j])
                for (int pred : schedule.G.In[node])
                    if (schedule.proc[pred] != schedule.proc[node] &&
                        schedule.commSchedule[pred][schedule.proc[node]] == -1) {
                        schedule.commSchedule[pred][schedule.proc[node]] = i - 1;
                        commBounds[pred][schedule.proc[node]] = intPair(schedule.supstep[pred], i - 1);

                        commSchedSendLists[i - 1][schedule.proc[pred]].emplace_front(pred, schedule.proc[node]);
                        commSchedSendListPointer[pred][schedule.proc[node]] =
                            commSchedSendLists[i - 1][schedule.proc[pred]].begin();
                        commSchedRecLists[i - 1][schedule.proc[node]].emplace_front(pred, schedule.proc[node]);
                        commSchedRecListPointer[pred][schedule.proc[node]] =
                            commSchedRecLists[i - 1][schedule.proc[node]].begin();
                    }

    std::vector<std::vector<bool>> present(schedule.G.n, std::vector<bool>(schedule.params.p, false));
    for (int i = 0; i < M - 1; ++i) {
        for (int j = 0; j < schedule.params.p; ++j)
            for (const int node : schedule.supsteplists[i + 1][j])
                for (const int pred : schedule.G.In[node])
                    if (schedule.proc[node] != schedule.proc[pred] && !present[pred][schedule.proc[node]]) {
                        present[pred][schedule.proc[node]] = true;
                        sent[i][schedule.proc[pred]] +=
                            schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][schedule.proc[node]];
                        received[i][schedule.proc[node]] +=
                            schedule.G.commW[pred] * schedule.params.sendCost[schedule.proc[pred]][schedule.proc[node]];
                    }

        for (int j = 0; j < schedule.params.p; ++j) {
            commCost[i][j] = std::max(sent[i][j], received[i][j]);
            intPair entry(commCost[i][j], j);
            commCostPointer[i][j] = commCostList[i].insert(entry).first;
        }
    }
};

// compute cost change incurred by a potential move
int HillClimbingCS::moveCostChange(const int node, const int p, const int step) {
    const int oldStep = schedule.commSchedule[node][p];
    const int sourceProc = schedule.proc[node];
    int change = 0;

    // Change at old place
    auto itr = commCostList[oldStep].rbegin();
    int oldMax = itr->a;
    const int maxSource =
        std::max(sent[oldStep][sourceProc] - schedule.G.commW[node] * schedule.params.sendCost[sourceProc][p],
                 received[oldStep][sourceProc]);
    const int maxTarget = std::max(
        sent[oldStep][p], received[oldStep][p] - schedule.G.commW[node] * schedule.params.sendCost[sourceProc][p]);
    int maxOther = 0;
    for (; itr != commCostList[oldStep].rend(); ++itr)
        if (itr->b != sourceProc && itr->b != p) {
            maxOther = itr->a;
            break;
        }

    int newMax = std::max(std::max(maxSource, maxTarget), maxOther);
    change += newMax - oldMax;
    if(newMax==0)
        change -= schedule.params.L;

    // Change at new place
    oldMax = commCostList[step].rbegin()->a;
    newMax = std::max(
        std::max(oldMax, sent[step][sourceProc] + schedule.G.commW[node] * schedule.params.sendCost[sourceProc][p]),
        received[step][p] + schedule.G.commW[node] * schedule.params.sendCost[sourceProc][p]);
    change += newMax - oldMax;
    if(oldMax==0)
        change += schedule.params.L;

    return change;
};

// execute a move, updating the comm. schedule and the data structures
void HillClimbingCS::executeMove(int node, int p, const int step, const int changeCost) {
    const int oldStep = schedule.commSchedule[node][p];
    const int sourceProc = schedule.proc[node];
    schedule.cost += changeCost * schedule.params.g;

    // Old step update
    if (sent[oldStep][sourceProc] > received[oldStep][sourceProc]) {
        commCostList[oldStep].erase(commCostPointer[oldStep][sourceProc]);
        sent[oldStep][sourceProc] -= schedule.G.commW[node] * schedule.params.sendCost[sourceProc][p];
        commCost[oldStep][sourceProc] = std::max(sent[oldStep][sourceProc], received[oldStep][sourceProc]);
        commCostPointer[oldStep][sourceProc] =
            commCostList[oldStep].insert(intPair(commCost[oldStep][sourceProc], sourceProc)).first;
    } else
        sent[oldStep][sourceProc] -= schedule.G.commW[node] * schedule.params.sendCost[sourceProc][p];

    if (received[oldStep][p] > sent[oldStep][p]) {
        commCostList[oldStep].erase(commCostPointer[oldStep][p]);
        received[oldStep][p] -= schedule.G.commW[node] * schedule.params.sendCost[sourceProc][p];
        commCost[oldStep][p] = std::max(sent[oldStep][p], received[oldStep][p]);
        commCostPointer[oldStep][p] = commCostList[oldStep].insert(intPair(commCost[oldStep][p], p)).first;
    } else
        received[oldStep][p] -= schedule.G.commW[node] * schedule.params.sendCost[sourceProc][p];

    // New step update
    sent[step][sourceProc] += schedule.G.commW[node] * schedule.params.sendCost[sourceProc][p];
    if (sent[step][sourceProc] > received[step][sourceProc]) {
        commCostList[step].erase(commCostPointer[step][sourceProc]);
        commCost[step][sourceProc] = sent[step][sourceProc];
        commCostPointer[step][sourceProc] =
            commCostList[step].insert(intPair(commCost[step][sourceProc], sourceProc)).first;
    }

    received[step][p] += schedule.G.commW[node] * schedule.params.sendCost[sourceProc][p];
    if (received[step][p] > sent[step][p]) {
        commCostList[step].erase(commCostPointer[step][p]);
        commCost[step][p] = received[step][p];
        commCostPointer[step][p] = commCostList[step].insert(intPair(commCost[step][p], p)).first;
    }

    // CommSched update
    schedule.commSchedule[node][p] = step;

    // Comm lists
    commSchedSendLists[oldStep][sourceProc].erase(commSchedSendListPointer[node][p]);
    commSchedSendLists[step][sourceProc].emplace_front(node, p);
    commSchedSendListPointer[node][p] = commSchedSendLists[step][sourceProc].begin();

    commSchedRecLists[oldStep][p].erase(commSchedRecListPointer[node][p]);
    commSchedRecLists[step][p].emplace_front(node, p);
    commSchedRecListPointer[node][p] = commSchedRecLists[step][p].begin();
};

// Single comm. schedule hill climbing step
bool HillClimbingCS::Improve(const bool SteepestAscent) {
    const int M = schedule.supsteplists.size();
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
                        if (step == schedule.commSchedule[node][p])
                            continue;

                        const int costDiff = moveCostChange(node, p, step);

                        if (!SteepestAscent && costDiff < 0) {
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
                        if (step == schedule.commSchedule[node][p])
                            continue;

                        const int costDiff = moveCostChange(node, p, step);

                        if (!SteepestAscent && costDiff < 0) {
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

// Main function for comm. schedule hill climbing
void HillClimbingCS::HillClimb(const int TimeLimit, const bool SteepestAscent) {
    if(schedule.supsteplists.size()<=2)
        return;

    Init();
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    int counter = 0;
    while (Improve(SteepestAscent))
        if ((++counter) == 1) {
            counter = 0;
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            if (elapsed >= TimeLimit) {
                std::cout << "Comm. Sched. Hill Climbing was shut down due to time limit." << std::endl;
                break;
            }
        }
    schedule.RemoveNeedlessSupSteps();
};
