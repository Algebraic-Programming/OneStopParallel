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

#include "osp/partitioning/model/partitioning_replication.hpp"
#include "osp/partitioning/partitioners/partitioning_ILP_base.hpp"

namespace osp {

template <typename HypergraphT>
class HypergraphPartitioningILPWithReplication : public HypergraphPartitioningILPBase<HypergraphT> {
  public:
    enum class ReplicationModelInIlp { ONLY_TWICE, GENERAL };

  protected:
    void SetupExtraVariablesConstraints(const PartitioningProblem<HypergraphT> &instance, Model &model);

    void SetInitialSolution(const PartitioningWithReplication<HypergraphT> &partition, Model &model);

    ReplicationModelInIlp replicationModel_ = ReplicationModelInIlp::ONLY_TWICE;

  public:
    virtual ~HypergraphPartitioningILPWithReplication() override = default;

    RETURN_STATUS ComputePartitioning(PartitioningWithReplication<HypergraphT> &result);

    virtual std::string GetAlgorithmName() const override { return "HypergraphPartitioningILPWithReplication"; }

    void SetReplicationModel(ReplicationModelInIlp replicationModel) { replicationModel_ = replicationModel; }
};

template <typename HypergraphT>
RETURN_STATUS HypergraphPartitioningILPWithReplication<HypergraphT>::ComputePartitioning(
    PartitioningWithReplication<HypergraphT> &result) {
    Envr env;
    Model model = env.CreateModel("HypergraphPartRepl");

    this->SetupFundamentalVariablesConstraintsObjective(result.getInstance(), model);
    SetupExtraVariablesConstraints(result.getInstance(), model);

    if (this->useInitialSolution_) {
        SetInitialSolution(result, model);
    }

    this->SolveIlp(model);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        result.setAssignedPartitionVectors(this->ReadAllCoptAssignments(result.getInstance(), model));
        return RETURN_STATUS::OSP_SUCCESS;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return RETURN_STATUS::ERROR;

    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            result.setAssignedPartitionVectors(this->ReadAllCoptAssignments(result.getInstance(), model));
            return RETURN_STATUS::OSP_SUCCESS;

        } else {
            return RETURN_STATUS::ERROR;
        }
    }
}

template <typename HypergraphT>
void HypergraphPartitioningILPWithReplication<HypergraphT>::SetupExtraVariablesConstraints(
    const PartitioningProblem<HypergraphT> &instance, Model &model) {
    using IndexType = typename HypergraphT::vertex_idx;

    const IndexType numberOfParts = instance.getNumberOfPartitions();
    const IndexType numberOfVertices = instance.getHypergraph().num_vertices();

    if (replicationModel_ == ReplicationModelInIlp::GENERAL) {
        // create variables for each pin+partition combination
        std::map<std::pair<IndexType, unsigned>, IndexType> pinIdMap;
        IndexType nrOfPins = 0;
        for (IndexType node = 0; node < numberOfVertices; node++) {
            for (const IndexType &hyperedge : instance.getHypergraph().get_incident_hyperedges(node)) {
                pinIdMap[std::make_pair(node, hyperedge)] = nrOfPins++;
            }
        }

        std::vector<VarArray> pinCoveredByPartition = std::vector<VarArray>(nrOfPins);

        for (IndexType pin = 0; pin < nrOfPins; pin++) {
            pinCoveredByPartition[pin] = model.AddVars(static_cast<int>(numberOfParts), COPT_BINARY, "pin_covered_by_partition");
        }

        //  each pin covered exactly once
        for (IndexType pin = 0; pin < nrOfPins; pin++) {
            Expr expr;
            for (unsigned part = 0; part < numberOfParts; part++) {
                expr += pinCoveredByPartition[pin][static_cast<int>(part)];
            }

            model.AddConstr(expr == 1);
        }

        // pin covering requires node assignment
        for (unsigned part = 0; part < numberOfParts; part++) {
            for (IndexType node = 0; node < numberOfVertices; node++) {
                for (const IndexType &hyperedge : instance.getHypergraph().get_incident_hyperedges(node)) {
                    model.AddConstr(this->nodeInPartition_[node][static_cast<int>(part)]
                                    >= pinCoveredByPartition[pinIdMap[std::make_pair(node, hyperedge)]][static_cast<int>(part)]);
                }
            }
        }

        // pin covering requires hyperedge use
        for (unsigned part = 0; part < numberOfParts; part++) {
            for (IndexType node = 0; node < numberOfVertices; node++) {
                for (const IndexType &hyperedge : instance.getHypergraph().get_incident_hyperedges(node)) {
                    model.AddConstr(this->hyperedgeUsesPartition_[hyperedge][static_cast<int>(part)]
                                    >= pinCoveredByPartition[pinIdMap[std::make_pair(node, hyperedge)]][static_cast<int>(part)]);
                }
            }
        }

    } else if (replicationModel_ == ReplicationModelInIlp::ONLY_TWICE) {
        // each node has one or two copies
        VarArray nodeReplicated = model.AddVars(static_cast<int>(numberOfVertices), COPT_BINARY, "node_replicated");

        for (IndexType node = 0; node < numberOfVertices; node++) {
            Expr expr = -1;
            for (unsigned part = 0; part < numberOfParts; part++) {
                expr += this->nodeInPartition_[node][static_cast<int>(part)];
            }

            model.AddConstr(expr == nodeReplicated[static_cast<int>(node)]);
        }

        // hyperedge indicators if node is not replicated
        for (unsigned part = 0; part < numberOfParts; part++) {
            for (IndexType node = 0; node < numberOfVertices; node++) {
                for (const IndexType &hyperedge : instance.getHypergraph().get_incident_hyperedges(node)) {
                    model.AddConstr(this->hyperedgeUsesPartition_[hyperedge][static_cast<int>(part)]
                                    >= this->nodeInPartition_[node][static_cast<int>(part)]
                                           - nodeReplicated[static_cast<int>(node)]);
                }
            }
        }

        // hyperedge indicators if node is replicated
        for (IndexType node = 0; node < numberOfVertices; node++) {
            for (const IndexType &hyperedge : instance.getHypergraph().get_incident_hyperedges(node)) {
                for (unsigned part1 = 0; part1 < numberOfParts; part1++) {
                    for (unsigned part2 = part1 + 1; part2 < numberOfParts; part2++) {
                        model.AddConstr(this->hyperedgeUsesPartition_[hyperedge][static_cast<int>(part1)]
                                            + this->hyperedgeUsesPartition_[hyperedge][static_cast<int>(part2)]
                                        >= this->nodeInPartition_[node][static_cast<int>(part1)]
                                               + this->nodeInPartition_[node][static_cast<int>(part2)] - 1);
                    }
                }
            }
        }
    }
}

template <typename HypergraphT>
void HypergraphPartitioningILPWithReplication<HypergraphT>::SetInitialSolution(
    const PartitioningWithReplication<HypergraphT> &partition, Model &model) {
    using IndexType = typename HypergraphT::vertex_idx;

    const std::vector<std::vector<unsigned> > &assignments = partition.assignedPartitions();
    const unsigned &numPartitions = partition.getInstance().getNumberOfPartitions();
    if (assignments.size() != partition.getInstance().getHypergraph().num_vertices()) {
        return;
    }

    for (IndexType node = 0; node < assignments.size(); ++node) {
        std::vector<bool> assingedToPart(numPartitions, false);
        for (unsigned part : assignments[node]) {
            if (part < numPartitions) {
                assingedToPart[part] = true;
            }
        }

        for (unsigned part = 0; part < numPartitions; ++part) {
            model.SetMipStart(this->nodeInPartition_[node][static_cast<int>(part)], static_cast<int>(assingedToPart[part]));
        }
    }
    model.LoadMipStart();
}

}    // namespace osp
