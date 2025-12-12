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

#include "osp/auxiliary/return_status.hpp"
#include "osp/partitioning/model/partitioning.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP_base.hpp"

namespace osp {

template <typename HypergraphT>
class HypergraphPartitioningILP : public HypergraphPartitioningILPBase<HypergraphT> {
  protected:
    std::vector<unsigned> ReadCoptAssignment(const PartitioningProblem<HypergraphT> &instance, Model &model);

    void SetupExtraVariablesConstraints(const PartitioningProblem<HypergraphT> &instance, Model &model);

    void SetInitialSolution(const Partitioning<HypergraphT> &partition, Model &model);

  public:
    virtual ~HypergraphPartitioningILP() override = default;

    RETURN_STATUS ComputePartitioning(Partitioning<HypergraphT> &result);

    virtual std::string GetAlgorithmName() const override { return "HypergraphPartitioningILP"; }
};

template <typename HypergraphT>
RETURN_STATUS HypergraphPartitioningILP<HypergraphT>::ComputePartitioning(Partitioning<HypergraphT> &result) {
    Envr env;
    Model model = env.CreateModel("HypergraphPart");

    this->SetupFundamentalVariablesConstraintsObjective(result.GetInstance(), model);
    SetupExtraVariablesConstraints(result.GetInstance(), model);

    if (this->useInitialSolution_) {
        SetInitialSolution(result, model);
    }

    this->SolveIlp(model);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        result.setAssignedPartitions(ReadCoptAssignment(result.GetInstance(), model));
        return RETURN_STATUS::OSP_SUCCESS;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;

    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            result.setAssignedPartitions(ReadCoptAssignment(result.GetInstance(), model));
            return RETURN_STATUS::OSP_SUCCESS;

        } else {
            return RETURN_STATUS::ERROR;
        }
    }
}

template <typename HypergraphT>
void HypergraphPartitioningILP<HypergraphT>::SetupExtraVariablesConstraints(const PartitioningProblem<HypergraphT> &instance,
                                                                            Model &model) {
    using IndexType = typename HypergraphT::vertex_idx;

    const IndexType numberOfParts = instance.getNumberOfPartitions();
    const IndexType numberOfVertices = instance.getHypergraph().NumVertices();

    // Constraints

    // each node assigned to exactly one partition
    for (IndexType node = 0; node < numberOfVertices; node++) {
        Expr expr;
        for (unsigned part = 0; part < numberOfParts; part++) {
            expr += this->nodeInPartition_[node][static_cast<int>(part)];
        }

        model.AddConstr(expr == 1);
    }

    // hyperedge indicators match node variables
    for (unsigned part = 0; part < numberOfParts; part++) {
        for (IndexType node = 0; node < numberOfVertices; node++) {
            for (const IndexType &hyperedge : instance.getHypergraph().get_incident_hyperedges(node)) {
                model.AddConstr(this->hyperedgeUsesPartition_[hyperedge][static_cast<int>(part)]
                                >= this->nodeInPartition_[node][static_cast<int>(part)]);
            }
        }
    }
}

// convert generic one-to-many assingment (of base class function) to one-to-one
template <typename HypergraphT>
std::vector<unsigned> HypergraphPartitioningILP<HypergraphT>::ReadCoptAssignment(const PartitioningProblem<HypergraphT> &instance,
                                                                                 Model &model) {
    using IndexType = typename HypergraphT::vertex_idx;

    std::vector<unsigned> nodeToPartition(instance.getHypergraph().NumVertices(), std::numeric_limits<unsigned>::max());
    std::vector<std::vector<unsigned>> assignmentsGenericForm = this->ReadAllCoptAssignments(instance, model);

    for (IndexType node = 0; node < instance.getHypergraph().NumVertices(); node++) {
        nodeToPartition[node] = assignmentsGenericForm[node].front();
    }

    return nodeToPartition;
}

template <typename HypergraphT>
void HypergraphPartitioningILP<HypergraphT>::SetInitialSolution(const Partitioning<HypergraphT> &partition, Model &model) {
    using IndexType = typename HypergraphT::vertex_idx;

    const std::vector<unsigned> &assignment = partition.assignedPartitions();
    const unsigned &numPartitions = partition.GetInstance().getNumberOfPartitions();
    if (assignment.size() != partition.GetInstance().getHypergraph().NumVertices()) {
        return;
    }

    for (IndexType node = 0; node < assignment.size(); ++node) {
        if (assignment[node] >= numPartitions) {
            continue;
        }

        for (unsigned part = 0; part < numPartitions; ++part) {
            model.SetMipStart(this->nodeInPartition_[node][static_cast<int>(part)], static_cast<int>(assignment[node] == part));
        }
    }
    model.LoadMipStart();
}

}    // namespace osp
