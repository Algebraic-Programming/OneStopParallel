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
#include <deque>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <cmath>

#include "file_interactions/FileReader.hpp"
#include "scheduler/LocalSearchSchedulers/hill_climbing.hpp"
#include "scheduler/ContractRefineScheduler/multilevel.hpp"
#include <structures/schedule.hpp>

bool Multilevel::isContractable(const DAG &G, const int source, const int target, const std::vector<int> &topOrderPos,
                                const std::vector<bool> &valid) {
    std::deque<int> Queue;
    std::set<int> visited;
    for (int succ : G.Out[source])
        if (valid[succ] && topOrderPos[succ] < topOrderPos[target]) {
            Queue.push_back(succ);
            visited.insert(succ);
        }

    while (!Queue.empty()) {
        const int node = Queue.front();
        Queue.pop_front();
        for (int succ : G.Out[node]) {
            if (succ == target)
                return false;

            if (valid[succ] && topOrderPos[succ] < topOrderPos[target] && visited.count(succ) == 0) {
                Queue.push_back(succ);
                visited.insert(succ);
            }
        }
    }
    return true;
};

std::vector<bool> Multilevel::areOutEdgesContractable(const DAG &G, const int node, const std::vector<int> &topOrderPos,
                                const std::vector<bool> &valid) {
    std::deque<int> Queue;
    std::set<int> visited;
    std::map<int, bool> succ_contractable;
    int topOrderMax = topOrderPos[node];

    for (int succ : G.Out[node])
    {
        succ_contractable[succ] = valid[succ];
        
        if(topOrderPos[succ] > topOrderMax)
            topOrderMax = topOrderPos[succ];

        if (valid[succ]) {
            Queue.push_back(succ);
            visited.insert(succ);
        }
    }

    while (!Queue.empty()) {
        const int node = Queue.front();
        Queue.pop_front();
        for (int succ : G.Out[node]) {
            if (succ_contractable.find(succ) != succ_contractable.end())
                succ_contractable[succ] = false;

            if (valid[succ] && topOrderPos[succ] < topOrderMax && visited.count(succ) == 0) {
                Queue.push_back(succ);
                visited.insert(succ);
            }
        }
    }

    std::vector<bool> isOutEdgeContractable(G.Out[node].size());
    for (unsigned i=0; i<G.Out[node].size(); ++i)
        isOutEdgeContractable[i] = succ_contractable[G.Out[node][i]];

    return isOutEdgeContractable;
};

std::vector<bool> Multilevel::areInEdgesContractable(const DAG &G, const int node, const std::vector<int> &topOrderPos,
                                const std::vector<bool> &valid) {
    std::deque<int> Queue;
    std::set<int> visited;
    std::map<int, bool> pred_contractable;
    int topOrderMin = topOrderPos[node];

    for (int pred : G.In[node])
    {
        pred_contractable[pred] = valid[pred];
        
        if(topOrderPos[pred] < topOrderMin)
            topOrderMin = topOrderPos[pred];

        if (valid[pred]) {
            Queue.push_back(pred);
            visited.insert(pred);
        }
    }

    while (!Queue.empty()) {
        const int node = Queue.front();
        Queue.pop_front();
        for (int pred : G.In[node]) {
            if (pred_contractable.find(pred) != pred_contractable.end())
                pred_contractable[pred] = false;

            if (valid[pred] && topOrderPos[pred] > topOrderMin && visited.count(pred) == 0) {
                Queue.push_back(pred);
                visited.insert(pred);
            }
        }
    }

    std::vector<bool> isInEdgeContractable(G.In[node].size());
    for (unsigned i=0; i<G.In[node].size(); ++i)
        isInEdgeContractable[i] = pred_contractable[G.In[node][i]];

    return isInEdgeContractable;
};

std::map<intPair, int> Multilevel::GetContractableEdges() const {

    std::vector<int> topOrderPos = G_full.GetTopOrderIdx();
    const std::vector<bool> valid(G_full.n, true);
    std::map<intPair, int> contractable;

    for (int i = 0; i < G_full.n; ++i)
    {
        std::vector<bool> out_edges = areOutEdgesContractable(G_full, i, topOrderPos, valid);
        for(int j=0; j<G_full.Out[i].size(); ++j)
            if(out_edges[j])
                contractable[intPair(i, G_full.Out[i][j])] = G_full.commW[i];
    }

    return contractable;
};

void Multilevel::updateDistantEdgeContractibility(const DAG &G, const int source, const int target,
                                                  std::map<intPair, int> &contractable) {
    std::unordered_set<int> ancestors, descendant;

    std::deque<int> Queue;
    for (int succ : G.Out[source])
        if (succ != target) {
            Queue.push_back(succ);
            descendant.insert(succ);
        }
    while (!Queue.empty()) {
        const int node = Queue.front();
        Queue.pop_front();
        for (int succ : G.Out[node])
            if (descendant.count(succ) == 0) {
                Queue.push_back(succ);
                descendant.insert(succ);
            }
    }

    for (int pred : G.In[target])
        if (pred != source) {
            Queue.push_back(pred);
            ancestors.insert(pred);
        }
    while (!Queue.empty()) {
        const int node = Queue.front();
        Queue.pop_front();
        for (int pred : G.In[node])
            if (ancestors.count(pred) == 0) {
                Queue.push_back(pred);
                ancestors.insert(pred);
            }
    }

    for (const int node : ancestors)
        for (const int succ : G.Out[node])
            if (descendant.count(succ) > 0)
                contractable.erase(intPair(node, succ));
};

DAG Multilevel::Coarsify(int newN, const std::string& outfilename, bool FastCoarsify, int randomSeed) {
    const int N = G_full.n;
    std::vector<bool> validNode(N, true);
    contractionHistory.clear();
    if(newN<1)
        newN=1;
    // list of original node indices contained in each contracted node
    std::vector<std::set<int>> contains(N);
    for (int i = 0; i < N; ++i)
        contains[i].insert(i);

    DAG coarse(G_full);

    //used for original, slow coarsening
    std::map<intPair, int> edgeWeights;
    std::map<intPair, int> contractable;
    bool random = false;
    if(!FastCoarsify)
    {
        // Init edge weights
        for (int i = 0; i < N; ++i)
            for (int j: G_full.Out[i])
                edgeWeights[intPair(i, j)] = G_full.commW[i];

        // get original contractable edges
        contractable = GetContractableEdges();

        // seed random
        random = (randomSeed != -1);
        if (random)
            srand(randomSeed);
    }

    for (int NrOfNodes = N; NrOfNodes > newN; ) {
        // Single contraction step

        std::vector<intPair> edgesToContract;

        // choose edges to contract in this step
        if(!FastCoarsify)
        {
            std::vector<contractionEdge> candidates = CreateEdgeCandidateList(coarse, contains, contractable);
            if(candidates.empty())
            {
                std::cout<<"Error: no edge to contract"<<std::endl;
                break;
            }
            intPair chosenEdge = PickEdgeToContract(coarse, candidates, random);
            edgesToContract.push_back(chosenEdge);

            //Update far-away edges that become uncontractable now
            updateDistantEdgeContractibility(coarse, chosenEdge.a, chosenEdge.b, contractable);
        }
        else
            edgesToContract = ClusterCoarsen(coarse, validNode);

        if(edgesToContract.empty())
            break;
        
        // contract these edges
        for(intPair edge : edgesToContract)
        {
            if(!FastCoarsify)
            {
                //Update contractable edges - edge.b
                for(int pred : coarse.In[edge.b])
                    contractable.erase(intPair(pred, edge.b));
                
                for(int succ : coarse.Out[edge.b])
                    contractable.erase(intPair(edge.b, succ));
            }

            ContractSingleEdge(coarse, edge, contains, edgeWeights);
            validNode[edge.b] = false;

            if(!FastCoarsify)
            {
                std::vector<int> topOrderPos = coarse.GetTopOrderIdx(validNode);

                //Update contractable edges - edge.a
                std::vector<bool> inEdgeContractable = areInEdgesContractable(coarse, edge.a, topOrderPos, validNode);
                for (int j=0; j<coarse.In[edge.a].size(); ++j)
                {
                    if(inEdgeContractable[j])
                        contractable[intPair(coarse.In[edge.a][j], edge.a)] = edgeWeights[intPair(coarse.In[edge.a][j], edge.a)];
                    else
                        contractable.erase(intPair(coarse.In[edge.a][j], edge.a));
                }
                
                std::vector<bool> outEdgeContractable = areOutEdgesContractable(coarse, edge.a, topOrderPos, validNode);
                for (int j=0; j<coarse.Out[edge.a].size(); ++j)
                {
                    if(outEdgeContractable[j])
                        contractable[intPair(edge.a, coarse.Out[edge.a][j])] = edgeWeights[intPair(edge.a, coarse.Out[edge.a][j])];
                    else
                        contractable.erase(intPair(edge.a, coarse.Out[edge.a][j]));
                }
            }
            --NrOfNodes;
            if(NrOfNodes==newN)
                break;
        }
    }

    if(pebbling_mode)
        MergeSourcesInPebbling(coarse, validNode, contains, edgeWeights);

    //Print contraction steps to file
    if(!outfilename.empty())
    {
        std::ofstream outfile(outfilename);
        if (outfile.is_open()) {
            outfile << N << " " << newN << std::endl;
            for (intPair entry: contractionHistory)
                outfile << entry.a << " " << entry.b << std::endl;

            std::vector<int> image = GetFinalImage(G_full, contractionHistory);
            for (int i = 0; i < N; ++i)
                outfile << i << " " << image[i] << std::endl;

            outfile.close();
        }
        else
            std::cout << "ERROR: Unable to write/open contraction output log file for DAG.\n";
    }

    //Return contracted DAG
    return Contract(contractionHistory);
};

void Multilevel::ContractSingleEdge(DAG& coarse, intPair edge, std::vector<std::set<int>>& contains, std::map<intPair, int>& edgeWeights) {

    coarse.workW[edge.a] += coarse.workW[edge.b];
    coarse.commW[edge.a] += coarse.commW[edge.b];
    contractionHistory.emplace_back(edge.a, edge.b);
    for (int pred: coarse.In[edge.b]) {
        if (pred == edge.a)
            continue;

        if(find(coarse.In[edge.a].begin(), coarse.In[edge.a].end(), pred) != coarse.In[edge.a].end()) // Combine edges
        {
            if(!edgeWeights.empty())
            {
                edgeWeights[intPair(pred, edge.a)] = 0;
                for (int node: contains[pred])
                    for (int succ: coarse.Out[node])
                        if (succ == edge.a || succ == edge.b)
                            edgeWeights[intPair(pred, edge.a)] += G_full.commW[node];
            }
        }
        else // Add incoming edge
        {
            if(!edgeWeights.empty())
                edgeWeights[intPair(pred, edge.a)] = edgeWeights[intPair(pred, edge.b)];

            coarse.Out[pred].push_back(edge.a);
            coarse.In[edge.a].push_back(pred);
        }
        for (auto it = coarse.Out[pred].begin(); it != coarse.Out[pred].end(); ++it)
            if (*it == edge.b) {
                coarse.Out[pred].erase(it);
                break;
            }
    }
    for (int succ: coarse.Out[edge.b]) {

        if(find(coarse.Out[edge.a].begin(), coarse.Out[edge.a].end(), succ) != coarse.Out[edge.a].end()) // Combine edges
        {
            if(!edgeWeights.empty())
                edgeWeights[intPair(edge.a, succ)] += edgeWeights[intPair(edge.b, succ)];
        }
        else // Add outgoing edge
        {
            if(!edgeWeights.empty())
                edgeWeights[intPair(edge.a, succ)] = edgeWeights[intPair(edge.b, succ)];

            coarse.In[succ].push_back(edge.a);
            coarse.Out[edge.a].push_back(succ);
        }
        for (auto it = coarse.In[succ].begin(); it != coarse.In[succ].end(); ++it)
            if (*it == edge.b) {
                coarse.In[succ].erase(it);
                break;
            }
    }
    for (auto it = coarse.Out[edge.a].begin(); it != coarse.Out[edge.a].end(); ++it)
        if (*it == edge.b) {
            coarse.Out[edge.a].erase(it);
            break;
        }

    coarse.In[edge.b].clear();
    coarse.Out[edge.b].clear();

    for (int node: contains[edge.b])
        contains[edge.a].insert(node);

    contains[edge.b].clear();
}

DAG Multilevel::Contract(const std::vector<intPair> &contractionSteps) const {
    const std::vector<int> target = GetFinalImage(G_full, contractionSteps);

    DAG coarseG;
    coarseG.Resize(G_full.n - contractionSteps.size());
    for (int i = 0; i < coarseG.n; ++i) {
        coarseG.workW[i] = 0;
        coarseG.commW[i] = 0;
    }

    for (int i = 0; i < G_full.n; ++i) {
        coarseG.workW[target[i]] += G_full.workW[i];
        coarseG.commW[target[i]] += G_full.commW[i];
    }

    std::set<intPair> edges;
    for (int i = 0; i < G_full.n; ++i)
        for (const int succ : G_full.Out[i]) {
            intPair edge(target[i], target[succ]);
            if (edges.find(edge) != edges.end() || edge.a == edge.b)
                continue;
            coarseG.addEdge(edge.a, edge.b);
            edges.insert(edge);
        }

    for (int i = 0; i < coarseG.n; ++i) {
        sort(coarseG.In[i].begin(), coarseG.In[i].end());
        sort(coarseG.Out[i].begin(), coarseG.Out[i].end());
    }

    return coarseG;
};

std::vector<int> Multilevel::GetFinalImage(const DAG &G, const std::vector<intPair> &contractionSteps) {
    std::vector<int> target(G.n), pointsTo(G.n, -1);

    for (const intPair &step : contractionSteps)
        pointsTo[step.b] = step.a;

    for (int i = 0; i < G.n; ++i) {
        target[i] = i;
        while (pointsTo[target[i]] != -1)
            target[i] = pointsTo[target[i]];
    }

    if (contractionSteps.empty()) 
        return target;

    std::vector<bool> valid(G.n, false);
    for (int i = 0; i < G.n; ++i)
        valid[target[i]] = true;

    DAG coarseG;
    coarseG.Resize(G.n);
    std::set<intPair> edges;
    for (int i = 0; i < G.n; ++i)
        for (const int succ : G.Out[i]) {
            intPair edge(target[i], target[succ]);
            if (edges.find(edge) != edges.end() || edge.a == edge.b)
                continue;
            coarseG.Out[edge.a].push_back(edge.b);
            coarseG.In[edge.b].push_back(edge.a);
            edges.insert(edge);
        }

    std::vector<int> newIdx = coarseG.GetTopOrderIdx(valid);
    for (int i = 0; i < G.n; ++i)
        target[i] = newIdx[target[i]];

    return target;
};

// run refinement: uncoarsify the DAG in small batches, and apply some steps of
// hill climbing after each iteration
Schedule Multilevel::Refine(const Schedule &CoarseSchedule, const int HCsteps) const {
    Schedule S = CoarseSchedule;
    S.commSchedule.clear();

    Schedule uncontractedS = ComputeUncontractedSchedule(S, contractionHistory);

    for (int nextN : refinementPoints){

        std::vector<intPair> contractionPrefix = contractionHistory;
        contractionPrefix.resize(G_full.n - nextN);

        const DAG nextG = Contract(contractionPrefix);
        S.G = nextG;
        S.params = CoarseSchedule.params;
        S.proc.clear();
        S.proc.resize(nextN);
        S.supstep.clear();
        S.supstep.resize(nextN);

        // Project full schedule to current graph
        std::vector<int> target = Multilevel::GetFinalImage(G_full, contractionPrefix);
        for (int i = 0; i < G_full.n; ++i) {
            S.proc[target[i]] = uncontractedS.proc[i];
            S.supstep[target[i]] = uncontractedS.supstep[i];
        }
        S.CreateSupStepLists();

        HillClimbing improve(S);
        improve.HillClimbSteps(HCsteps);
        S = improve.getSchedule();
        uncontractedS = ComputeUncontractedSchedule(S, contractionPrefix);
    }

    std::cout << "Refined cost: " << S.GetCost() << std::endl;
    return S;
};

bool Multilevel::ReadContractionHistory(const std::string &contractFile) {
    std::ifstream infile(contractFile);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input DAG-contraction file.\n";
        return false;
    }

    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    int oldN, newN;
    sscanf(line.c_str(), "%d %d", &oldN, &newN);

    contractionHistory.clear();

    for (int i = 0; i < oldN - newN; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect contraction file format (file terminated too early).\n";
            return false;
        }
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int target, source;
        sscanf(line.c_str(), "%d %d", &target, &source);

        if (target < 0 || source < 0 || target >= oldN || source >= oldN) {
            std::cout << "Incorrect contraction file format (index out of range).\n";
            return false;
        }

        contractionHistory.push_back(intPair(target, source));
    }

    infile.close();
    return true;
};

// given an original DAG G, a schedule on the coarsified G and the contraction
// steps, project the coarse schedule to the entire G
Schedule Multilevel::ComputeUncontractedSchedule(const Schedule &CoarseSchedule,
                                                 const std::vector<intPair> &contractionSteps) const {
    const std::vector<int> target = Multilevel::GetFinalImage(G_full, contractionSteps);

    Schedule S;
    S.G = G_full;
    S.params = CoarseSchedule.params;
    S.proc.clear();
    S.proc.resize(G_full.n);
    S.supstep.clear();
    S.supstep.resize(G_full.n);

    for (int i = 0; i < G_full.n; ++i) {
        S.proc[i] = CoarseSchedule.proc[target[i]];
        S.supstep[i] = CoarseSchedule.supstep[target[i]];
    }

    if (!CoarseSchedule.commSchedule.empty()) {
        S.commSchedule.clear();
        S.commSchedule.resize(G_full.n, std::vector<int>(S.params.p, -1));
        for (int i = 0; i < G_full.n; ++i)
            for (int j = 0; j < S.params.p; ++j)
                if (CoarseSchedule.commSchedule[target[i]][j] >= 0) {
                    // Check if this comm. step is needed at all after uncoarsening
                    bool needed = false;
                    for (const int succ : G_full.Out[i])
                        if (S.proc[succ] == j) {
                            needed = true;
                            break;
                        }

                    if (needed)
                        S.commSchedule[i][j] = CoarseSchedule.commSchedule[target[i]][j];
                }
    }

    S.CreateSupStepLists();
    return S;
};

// read, compute projected schedule (see above), and write to file
Schedule Multilevel::ComputeUncontractedSchedule(const Schedule &CoarseSchedule, const std::string &contractFile,
                                                 const std::string &outfilename) {
    std::vector<intPair> contractionSteps;
    if (!ReadContractionHistory(contractFile))
        return Schedule();

    Schedule S = ComputeUncontractedSchedule(CoarseSchedule, contractionHistory);

    std::cout << "Uncontracted cost: " << S.GetCost() << std::endl;

    if (!outfilename.empty())
        S.WriteToFile(outfilename);

    return S;
}

void Multilevel::setLinearRefinementPoints(int newN, int stepSize)
{
    refinementPoints.clear();
    if(stepSize<5)
        stepSize = 5;

    for (int nextN = newN + stepSize; nextN < G_full.n; nextN += stepSize)
        refinementPoints.push_back(nextN);

    refinementPoints.pop_back();
    refinementPoints.push_back(G_full.n);
}

void Multilevel::setExponentialRefinementPoints(int newN, double stepRatio)
{
    refinementPoints.clear();
    if(stepRatio<1.01)
        stepRatio = 1.01;

    for (int nextN = std::max(static_cast<int>(std::round(newN * stepRatio)), newN+5); nextN < G_full.n;
                        nextN = std::max(static_cast<int>(std::round(nextN * stepRatio)), refinementPoints.back()+5))
        refinementPoints.push_back(nextN);

    refinementPoints.push_back(G_full.n);
}

std::vector<int> Multilevel::ComputeTopLevel(const DAG& G, const std::vector<bool>& valid) const
{
    std::vector<int> TopLevel(G_full.n);
    const std::vector<int> topOrder = G.GetFilteredTopOrder(valid);
    for (int i = 0; i < topOrder.size(); ++i) {
        const int node = topOrder[i];
        int maxval = -1;
        for (const int pred: G.In[node])
            if (TopLevel[pred] > maxval)
                maxval = TopLevel[pred];
        TopLevel[node] = maxval + 1;
    }
    return TopLevel;
}

std::vector<contractionEdge> Multilevel::CreateEdgeCandidateList(const DAG& coarse, const std::vector<std::set<int>>& contains,
                                        const std::map<intPair, int>& contractable) const
{
    std::vector<contractionEdge> candidates;

    for (auto it= contractable.begin(); it != contractable.end(); ++it)
    {
        if(pebbling_mode && IncontractableForPebbling(coarse, it->first))
            continue;

        candidates.emplace_back(it->first.a, it->first.b, contains[it->first.a].size() + contains[it->first.b].size(), it->second);
    }

    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

intPair Multilevel::PickEdgeToContract(const DAG& coarse, const std::vector<contractionEdge>& candidates, bool random) const
{
    int limit = (candidates.size() + 2) / 3;
    int limitCardinality = candidates[limit].nodeW;
    while (limit < candidates.size() - 1 && candidates[limit + 1].nodeW == limitCardinality)
        ++limit;

    // an edge case
    if (candidates.size() == 1)
        limit = 0;

    contractionEdge chosen = candidates[randInt(candidates.size())];
    if (!random) {
        int best = 0;
        for (int i = 1; i <= limit; ++i)
            if (candidates[i].edgeW > candidates[best].edgeW)
                best = i;

        chosen = candidates[best];
    }
    return chosen.edge;
}

/**
 * @brief Acyclic graph contractor based on (Herrmann, Julien, et al. "Acyclic partitioning of large directed acyclic graphs." 2017 17th IEEE/ACM international symposium on cluster, cloud and grid computing (CCGRID). IEEE, 2017.))
 * @brief with minor changes and fixes
 * 
 */
std::vector<intPair> Multilevel::ClusterCoarsen(const DAG& G, const std::vector<bool>& valid) const
{
    std::vector<bool> singleton(G_full.n, true);
    std::vector<int> leader(G_full.n);
    std::vector<int> weight(G_full.n);
    std::vector<int> nrBadNeighbors(G_full.n);
    std::vector<int> leaderBadNeighbors(G_full.n);

    std::vector<int> minTopLevel(G_full.n);
    std::vector<int> maxTopLevel(G_full.n);
    std::vector<int> clusterNewID(G_full.n);

    std::vector<intPair> contractionSteps;
    std::vector<int> topLevel = ComputeTopLevel(G, valid);
    for(int i=0; i<G_full.n; ++i)
        if(valid[i])
        {
            leader[i]=i;
            weight[i]=1 /*G.workW[i]*/;
            nrBadNeighbors[i]=0;
            leaderBadNeighbors[i]=-1;
            clusterNewID[i]=i;
            minTopLevel[i]=topLevel[i];
            maxTopLevel[i]=topLevel[i];
        }

    for(int i=0; i<G_full.n; ++i)
    {
        if(!valid[i] || !singleton[i])
            continue;

        if(nrBadNeighbors[i]>1)
            continue;

        std::vector<int> validNeighbors;
        for(int pred: G.In[i])
        {
            // direct check of condition 1
            if(topLevel[i]<maxTopLevel[leader[pred]]-1 || topLevel[i]>minTopLevel[leader[pred]]+1)
                continue;
            // indirect check of condition 2
            if(nrBadNeighbors[i]>1 || (nrBadNeighbors[i]==1 && leaderBadNeighbors[i]!=leader[pred]))
                continue;
            //check condition 2 for pred if it is a singleton
            if(singleton[pred] && nrBadNeighbors[pred]>0)
                continue;

            // check viability for pebbling
            if(pebbling_mode && IncontractableForPebbling(G, intPair(pred, i)))
                continue;

            validNeighbors.push_back(pred);
        }
        for(int succ: G.Out[i])
        {
            // direct check of condition 1
            if(topLevel[i]<maxTopLevel[leader[succ]]-1 || topLevel[i]>minTopLevel[leader[succ]]+1)
                continue;
            // indirect check of condition 2
            if(nrBadNeighbors[i]>1 || (nrBadNeighbors[i]==1 && leaderBadNeighbors[i]!=leader[succ]))
                continue;
            //check condition 2 for pred if it is a singleton
            if(singleton[succ] && nrBadNeighbors[succ]>0)
                continue;

            // check viability for pebbling
            if(pebbling_mode && IncontractableForPebbling(G, intPair(i, succ)))
                continue;

            validNeighbors.push_back(succ);
        }

        int bestNeighbor = -1;
        for(int neigh : validNeighbors)
            if(bestNeighbor == -1 || weight[leader[neigh]]<weight[leader[bestNeighbor]])
                bestNeighbor = neigh;

        if(bestNeighbor==-1)
            continue;

        int newLead = leader[bestNeighbor];
        leader[i] = newLead;
        weight[newLead] += weight[i];

        if(std::find(G.In[i].begin(), G.In[i].end(), bestNeighbor) != G.In[i].end())
            contractionSteps.emplace_back(clusterNewID[newLead], i);
        else
        {
            contractionSteps.emplace_back(i, clusterNewID[newLead]);
            clusterNewID[newLead]=i;
        }

        minTopLevel[newLead]=std::min(minTopLevel[newLead], topLevel[i]);
        maxTopLevel[newLead]=std::max(maxTopLevel[newLead], topLevel[i]);

        for(int pred: G.In[i])
        {
            int abs1 = static_cast<int>(std::round(std::abs(topLevel[pred]-maxTopLevel[newLead])));
            int abs2 = static_cast<int>(std::round(std::abs(topLevel[pred]-minTopLevel[newLead])));
            if(abs1 != 1 && abs2 != 1)
                continue;

            if(nrBadNeighbors[pred]==0)
            {
                ++nrBadNeighbors[pred];
                leaderBadNeighbors[pred]=newLead;
            }
            else if(nrBadNeighbors[pred]==1 && leaderBadNeighbors[pred]!=newLead)
                ++nrBadNeighbors[pred];
        }
        for(int succ: G.Out[i])
        {
            int abs1 = static_cast<int>(std::round(std::abs(topLevel[succ]-maxTopLevel[newLead])));
            int abs2 = static_cast<int>(std::round(std::abs(topLevel[succ]-minTopLevel[newLead])));
            if(abs1 != 1 && abs2 != 1)
                continue;

            if(nrBadNeighbors[succ]==0)
            {
                ++nrBadNeighbors[succ];
                leaderBadNeighbors[succ]=newLead;
            }
            else if(nrBadNeighbors[succ]==1 && leaderBadNeighbors[succ]!=newLead)
                ++nrBadNeighbors[succ];
        }

        if(singleton[bestNeighbor])
        {
            for(int pred: G.In[bestNeighbor])
            {
                int abs1 = static_cast<int>(std::round(std::abs(topLevel[pred]-maxTopLevel[newLead])));
                int abs2 = static_cast<int>(std::round(std::abs(topLevel[pred]-minTopLevel[newLead])));
                if(abs1 != 1 && abs2 != 1)
                    continue;

                if(nrBadNeighbors[pred]==0)
                {
                    ++nrBadNeighbors[pred];
                    leaderBadNeighbors[pred]=newLead;
                }
                else if(nrBadNeighbors[pred]==1 && leaderBadNeighbors[pred]!=newLead)
                    ++nrBadNeighbors[pred];
            }
            for(int succ: G.Out[bestNeighbor])
            {
                int abs1 = static_cast<int>(std::round(std::abs(topLevel[succ]-maxTopLevel[newLead])));
                int abs2 = static_cast<int>(std::round(std::abs(topLevel[succ]-minTopLevel[newLead])));
                if(abs1 != 1 && abs2 != 1)
                    continue;

                if(nrBadNeighbors[succ]==0)
                {
                    ++nrBadNeighbors[succ];
                    leaderBadNeighbors[succ]=newLead;
                }
                else if(nrBadNeighbors[succ]==1 && leaderBadNeighbors[succ]!=newLead)
                    ++nrBadNeighbors[succ];
            }
            singleton[bestNeighbor] = false;
        }
        singleton[i] = false;
    }

    return contractionSteps;
}


ComputationalDag Multilevel::CoarsenForPebbling(const ComputationalDag& dag, double coarsen_ratio, std::vector<unsigned>& new_node_IDs, unsigned hard_constraint, bool FastCoarsify)
{
    new_node_IDs.clear();
    new_node_IDs.resize(dag.numberOfVertices());

    if(hard_constraint > 0)
        setHardMemConstraint(hard_constraint);

    pebbling_mode = true;
    unsigned nr_non_sources = 0;
    for(unsigned node = 0; node < dag.numberOfVertices(); ++node)
        if(dag.numberOfParents(node) > 0)
            ++nr_non_sources;

    DAG dag_format(dag);
    G_full = dag_format;
    unsigned new_size = (double)nr_non_sources * coarsen_ratio + (dag_format.n - nr_non_sources);
    DAG coarse_dag_format = Coarsify(new_size, "", FastCoarsify);

    std::vector<int> mapping_to_coarse = GetFinalImage(dag_format, contractionHistory);
    for(unsigned node = 0; node < dag.numberOfVertices(); ++node)
        new_node_IDs[node] = (unsigned)mapping_to_coarse[node];

    ComputationalDag final_coarsened(coarse_dag_format.n);
    for(unsigned node = 0; node < final_coarsened.numberOfVertices(); ++node)
    {
        final_coarsened.setNodeWorkWeight(node, 0);
        final_coarsened.setNodeCommunicationWeight(node, 0);
        final_coarsened.setNodeMemoryWeight(node, 0);
    }

    for(unsigned node = 0; node < dag.numberOfVertices(); ++node)
    {
        final_coarsened.setNodeWorkWeight(new_node_IDs[node], final_coarsened.nodeWorkWeight(new_node_IDs[node]) + dag.nodeWorkWeight(node));
        final_coarsened.setNodeCommunicationWeight(new_node_IDs[node], final_coarsened.nodeCommunicationWeight(new_node_IDs[node]) + dag.nodeCommunicationWeight(node));
        final_coarsened.setNodeMemoryWeight(new_node_IDs[node], final_coarsened.nodeMemoryWeight(new_node_IDs[node]) + dag.nodeMemoryWeight(node));
        for(unsigned succ : dag.children(node))
            if(new_node_IDs[node] != new_node_IDs[succ])
                final_coarsened.addEdge(new_node_IDs[node], new_node_IDs[succ]);
    }
    final_coarsened.mergeMultipleEdges();

    return final_coarsened;
}

bool Multilevel::IncontractableForPebbling(const DAG& coarse, const intPair& edge) const
{
    if(coarse.In[edge.a].size() == 0)
        return true;

    unsigned sum_weight = coarse.commW[edge.a] + coarse.commW[edge.b];
    std::set<unsigned> parents;
    for(int pred: coarse.In[edge.a])
        parents.insert(pred);
    for(int pred: coarse.In[edge.b])
        if(pred != edge.a)
            parents.insert(pred);
    for(unsigned node : parents)
        sum_weight += coarse.commW[node];

    if(sum_weight > hard_mem_constraint)
        return true;
    
    std::set<unsigned> children;
    for(int succ: coarse.Out[edge.b])
        children.insert(succ);
    for(int succ: coarse.Out[edge.a])
        if(succ != edge.b)
            children.insert(succ);

    for(unsigned child : children)
    {
        sum_weight = coarse.commW[edge.a] + coarse.commW[edge.b] + coarse.commW[child];
        for(int pred: coarse.In[child])
        {
            if(pred != edge.a && pred != edge.b)
                sum_weight += coarse.commW[pred];
        }
        
        if(sum_weight > hard_mem_constraint)
            return true;
    }
    return false;
}

void Multilevel::MergeSourcesInPebbling(DAG& coarse, const std::vector<bool>& validNode, std::vector<std::set<int>>& contains, std::map<intPair, int>& edgeWeights)
{
    // initialize memory requirement sums to check viability later
    std::vector<unsigned> memory_sum(coarse.n, 0);
    std::vector<unsigned> sources;
    for(unsigned node = 0; node < coarse.n; ++node)
    {
        if(!validNode[node])
            continue;

        if(coarse.In[node].size()>0)
        {
            memory_sum[node] = coarse.commW[node];
            for(int pred: coarse.In[node])
                memory_sum[node] += coarse.commW[pred];
        }
        else 
            sources.push_back(node);
    }
    
    std::set<unsigned> invalidated_sources;
    bool could_merge = true;
    while(could_merge)
    {
        could_merge = false;
        for(unsigned idx1 = 0; idx1 < sources.size(); ++idx1)
        {
            unsigned source_a = sources[idx1];
            if(invalidated_sources.find(source_a) != invalidated_sources.end())
                continue;
            
            for(unsigned idx2 = idx1 + 1; idx2 < sources.size(); ++idx2)
            {
                unsigned source_b = sources[idx2];
                if(invalidated_sources.find(source_b) != invalidated_sources.end())
                    continue;
                
                // check if we can merge source_a and source_b
                std::set<unsigned> a_children, b_children;
                for(int succ: coarse.Out[source_a])
                    a_children.insert(succ);
                for(int succ: coarse.Out[source_b])
                    b_children.insert(succ);
                
                std::set<unsigned> only_a, only_b, both;
                for(int succ: coarse.Out[source_a])
                {
                    if(b_children.find(succ) == b_children.end())
                        only_a.insert(succ);
                    else
                        both.insert(succ);
                }
                for(int succ: coarse.Out[source_b])
                {
                    if(a_children.find(succ) == a_children.end())
                        only_b.insert(succ);
                }

                bool violates_constraint = false;
                for(unsigned node : only_a)
                    if(memory_sum[node] + coarse.commW[source_b] > hard_mem_constraint)
                        violates_constraint = true;
                for(unsigned node : only_b)
                    if(memory_sum[node] + coarse.commW[source_a] > hard_mem_constraint)
                        violates_constraint = true;

                if(violates_constraint)
                    continue;

                // check if we want to merge source_a and source_b
                double sim_diff = (only_a.size() + only_b.size() == 0) ? 0.0001 : (double)(only_a.size() + only_b.size());
                double ratio = (double) both.size() / sim_diff;
                
                if(ratio > 2)
                {
                    ContractSingleEdge(coarse, intPair(source_a, source_b), contains, edgeWeights);
                    invalidated_sources.insert(source_b);
                    could_merge = true;

                    for(unsigned node : only_a)
                        memory_sum[node] += coarse.commW[source_b];
                    for(unsigned node : only_b)
                        memory_sum[node] += coarse.commW[source_a];
                }
            }
        }
    }
}
