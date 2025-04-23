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

#pragma once

#include "structures/dag.hpp"
#include <string>
#include <vector>

#include "auxiliary/auxiliary.hpp"
#include "structures/dag.hpp"
#include "model/ComputationalDag.hpp"

struct Schedule;

struct DAG;

struct Multilevel {

    DAG G_full;
    std::vector<intPair> contractionHistory;
    Multilevel() {}
    Multilevel(const DAG &G) : G_full(G) {}

    // main functions for multilevel approach
    DAG Coarsify(int newN, const std::string& outfile, bool FastCoarsify = false, int randomSeed = -1);
    DAG Contract(const std::vector<intPair>& contractionSteps) const;

    // Contract single edge while coarsifying DAG
    void ContractSingleEdge(DAG& coarseG, intPair edge, std::vector<std::set<int>>& contains, std::map<intPair, int>& edgeWeights);

    // further auxiliary for slow contraction
    std::vector<contractionEdge> CreateEdgeCandidateList(const DAG& coarse, const std::vector<std::set<int>>& contains,
                                        const std::map<intPair, int>& contractable) const;
    intPair PickEdgeToContract(const DAG& coarse, const std::vector<contractionEdge>& candidates, bool random) const;

    // auxiliary functions
    std::map<intPair, int> GetContractableEdges() const;
    static bool isContractable(const DAG &G, int source, int target, const std::vector<int> &topOrder,
                               const std::vector<bool> &valid);
    static std::vector<bool> areOutEdgesContractable(const DAG &G, int node, const std::vector<int> &topOrderPos,
                                const std::vector<bool> &valid);
    static std::vector<bool> areInEdgesContractable(const DAG &G, int node, const std::vector<int> &topOrderPos,
                                const std::vector<bool> &valid);
    static void updateDistantEdgeContractibility(const DAG &G, int source, int target,
                                                 std::map<intPair, int> &contractable);
    static std::vector<int> GetFinalImage(const DAG &G, const std::vector<intPair> &contractionSteps);

    // run refinement: uncoarsify the DAG in small batches, and apply some steps
    // of hill climbing after each iteration
    Schedule Refine(const Schedule &CoarseSchedule, int HCsteps = 20) const;
    bool ReadContractionHistory(const std::string &contractFile);

    // given a schedule on the coarsified G and the contraction steps, project the
    // coarse schedule to the entire G
    Schedule ComputeUncontractedSchedule(const Schedule &CoarseSchedule,
                                         const std::vector<intPair> &contractionSteps) const;

    // read contraction steps, and evaluate coarse schedule on contracted DAG
    Schedule ComputeUncontractedSchedule(const Schedule &CoarseSchedule, const std::string &contractFile,
                                         const std::string &outfilename = "");

    // ingredients for the new, faster version of multilevel
    std::vector<int> refinementPoints;
    void setLinearRefinementPoints(int newN, int stepSize);
    void setExponentialRefinementPoints(int newN, double stepRatio);

    std::vector<intPair> ClusterCoarsen(const DAG& G, const std::vector<bool>& valid) const;
    std::vector<int> ComputeTopLevel(const DAG& G, const std::vector<bool>& valid) const;

    // utility: coarsening function for pebbling problems - leaves the source nodes intact
    bool pebbling_mode = false;
    ComputationalDag CoarsenForPebbling(const ComputationalDag& dag, double coarsen_ratio, std::vector<unsigned>& new_node_IDs, unsigned hard_constraint = 0, bool FastCoarsify = false);

    // for pebbling: hard memory constraint to ensure feasibility of resulting graph
    unsigned hard_mem_constraint = 0;
    void setHardMemConstraint(unsigned new_constraint){ hard_mem_constraint = new_constraint;}
    bool IncontractableForPebbling(const DAG& coarse, const intPair&) const;
    void MergeSourcesInPebbling(DAG& coarse, const std::vector<bool>& validNode, std::vector<std::set<int>>& contains, std::map<intPair, int>& edgeWeights);
};
