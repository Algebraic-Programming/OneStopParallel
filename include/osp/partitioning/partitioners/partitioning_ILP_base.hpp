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

#include <callbackbase.h>
#include <coptcpp_pch.h>

#include "osp/bsp/model/BspInstance.hpp"    // for return statuses (stati?)
#include "osp/partitioning/model/partitioning_problem.hpp"

namespace osp {

template <typename HypergraphT>
class HypergraphPartitioningILPBase {
  protected:
    std::vector<VarArray> nodeInPartition_;
    std::vector<VarArray> hyperedgeUsesPartition_;

    unsigned timeLimitSeconds_ = 3600;
    bool useInitialSolution_ = false;

    std::vector<std::vector<unsigned> > ReadAllCoptAssignments(const PartitioningProblem<HypergraphT> &instance, Model &model);

    void SetupFundamentalVariablesConstraintsObjective(const PartitioningProblem<HypergraphT> &instance, Model &model);

    void SolveIlp(Model &model);

  public:
    virtual std::string GetAlgorithmName() const = 0;

    inline unsigned GetTimeLimitSeconds() const { return timeLimitSeconds_; }

    inline void SetTimeLimitSeconds(unsigned limit) { timeLimitSeconds_ = limit; }

    inline void SetUseInitialSolution(bool use) { useInitialSolution_ = use; }

    virtual ~HypergraphPartitioningILPBase() = default;
};

template <typename HypergraphT>
void HypergraphPartitioningILPBase<HypergraphT>::SolveIlp(Model &model) {
    // model.SetIntParam(COPT_INTPARAM_LOGTOCONSOLE, 0);

    model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds_);
    model.SetIntParam(COPT_INTPARAM_THREADS, 128);

    // model.SetIntParam(COPT_INTPARAM_STRONGBRANCHING, 1);
    // model.SetIntParam(COPT_INTPARAM_LPMETHOD, 1);
    // model.SetIntParam(COPT_INTPARAM_ROUNDINGHEURLEVEL, 1);

    model.SetIntParam(COPT_INTPARAM_PRESOLVE, 0);
    model.SetIntParam(COPT_INTPARAM_LPMETHOD, 1);
    model.SetIntParam(COPT_INTPARAM_CUTLEVEL, 1);
    model.SetIntParam(COPT_INTPARAM_ROOTCUTLEVEL, 2);
    model.SetIntParam(COPT_INTPARAM_TREECUTLEVEL, 1);

    // model.SetIntParam(COPT_INTPARAM_SUBMIPHEURLEVEL, 1);
    //  model.SetIntParam(COPT_INTPARAM_PRESOLVE, 1);
    //  model.SetIntParam(COPT_INTPARAM_CUTLEVEL, 0);
    // model.SetIntParam(COPT_INTPARAM_TREECUTLEVEL, 2);
    //  model.SetIntParam(COPT_INTPARAM_DIVINGHEURLEVEL, 2);

    model.Solve();
}

template <typename HypergraphT>
void HypergraphPartitioningILPBase<HypergraphT>::SetupFundamentalVariablesConstraintsObjective(
    const PartitioningProblem<HypergraphT> &instance, Model &model) {
    using IndexType = typename HypergraphT::VertexIdx;
    using WorkwType = typename HypergraphT::VertexWorkWeightType;
    using MemwType = typename HypergraphT::VertexMemWeightType;

    const IndexType numberOfParts = instance.GetNumberOfPartitions();
    const IndexType numberOfVertices = instance.GetHypergraph().NumVertices();
    const IndexType numberOfHyperedges = instance.GetHypergraph().NumHyperedges();

    // Variables

    nodeInPartition_ = std::vector<VarArray>(numberOfVertices);

    for (IndexType node = 0; node < numberOfVertices; node++) {
        nodeInPartition_[node] = model.AddVars(static_cast<int>(numberOfParts), COPT_BINARY, "node_in_partition");
    }

    hyperedgeUsesPartition_ = std::vector<VarArray>(numberOfHyperedges);

    for (IndexType hyperedge = 0; hyperedge < numberOfHyperedges; hyperedge++) {
        hyperedgeUsesPartition_[hyperedge]
            = model.AddVars(static_cast<int>(numberOfParts), COPT_BINARY, "hyperedge_uses_partition");
    }

    // partition size constraints
    if (instance.GetMaxWorkWeightPerPartition() < std::numeric_limits<WorkwType>::max()) {
        for (unsigned part = 0; part < numberOfParts; part++) {
            Expr expr;
            for (IndexType node = 0; node < numberOfVertices; node++) {
                expr += instance.GetHypergraph().GetVertexWorkWeight(node) * nodeInPartition_[node][static_cast<int>(part)];
            }

            model.AddConstr(expr <= instance.GetMaxWorkWeightPerPartition());
        }
    }
    if (instance.GetMaxMemoryWeightPerPartition() < std::numeric_limits<MemwType>::max()) {
        for (unsigned part = 0; part < numberOfParts; part++) {
            Expr expr;
            for (IndexType node = 0; node < numberOfVertices; node++) {
                expr += instance.GetHypergraph().GetVertexMemoryWeight(node) * nodeInPartition_[node][static_cast<int>(part)];
            }

            model.AddConstr(expr <= instance.GetMaxMemoryWeightPerPartition());
        }
    }

    // set objective
    Expr expr;
    for (IndexType hyperedge = 0; hyperedge < numberOfHyperedges; hyperedge++) {
        expr -= instance.GetHypergraph().GetHyperedgeWeight(hyperedge);
        for (unsigned part = 0; part < numberOfParts; part++) {
            expr += instance.GetHypergraph().GetHyperedgeWeight(hyperedge)
                    * hyperedgeUsesPartition_[hyperedge][static_cast<int>(part)];
        }
    }

    model.SetObjective(expr, COPT_MINIMIZE);
}

template <typename HypergraphT>
std::vector<std::vector<unsigned> > HypergraphPartitioningILPBase<HypergraphT>::ReadAllCoptAssignments(
    const PartitioningProblem<HypergraphT> &instance, Model &model) {
    using IndexType = typename HypergraphT::VertexIdx;

    std::vector<std::vector<unsigned> > nodeToPartitions(instance.GetHypergraph().NumVertices());

    std::set<unsigned> nonemptyPartitionIds;
    for (IndexType node = 0; node < instance.GetHypergraph().NumVertices(); node++) {
        for (unsigned part = 0; part < instance.GetNumberOfPartitions(); part++) {
            if (nodeInPartition_[node][static_cast<int>(part)].Get(COPT_DBLINFO_VALUE) >= .99) {
                nodeToPartitions[node].push_back(part);
                nonemptyPartitionIds.insert(part);
            }
        }
    }

    for (std::vector<unsigned> &chosenPartitions : nodeToPartitions) {
        if (chosenPartitions.empty()) {
            std::cout << "Error: partitioning returned by ILP seems incomplete!" << std::endl;
            chosenPartitions.push_back(std::numeric_limits<unsigned>::max());
        }
    }

    unsigned currentIndex = 0;
    std::map<unsigned, unsigned> newPartIndex;
    for (unsigned partIndex : nonemptyPartitionIds) {
        newPartIndex[partIndex] = currentIndex;
        ++currentIndex;
    }

    for (IndexType node = 0; node < instance.GetHypergraph().NumVertices(); node++) {
        for (unsigned entryIdx = 0; entryIdx < nodeToPartitions[node].size(); ++entryIdx) {
            nodeToPartitions[node][entryIdx] = newPartIndex[nodeToPartitions[node][entryIdx]];
        }
    }

    std::cout << "Hypergraph partitioning ILP best solution value: " << model.GetDblAttr(COPT_DBLATTR_BESTOBJ)
              << ", best lower bound: " << model.GetDblAttr(COPT_DBLATTR_BESTBND) << std::endl;

    return nodeToPartitions;
}

}    // namespace osp
