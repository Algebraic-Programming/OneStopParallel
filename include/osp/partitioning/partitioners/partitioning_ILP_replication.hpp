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

    ReturnStatus ComputePartitioning(PartitioningWithReplication<HypergraphT> &result);

    virtual std::string GetAlgorithmName() const override { return "HypergraphPartitioningILPWithReplication"; }

    void SetReplicationModel(ReplicationModelInIlp replicationModel) { replicationModel_ = replicationModel; }
};

template <typename HypergraphT>
ReturnStatus HypergraphPartitioningILPWithReplication<HypergraphT>::ComputePartitioning(
    PartitioningWithReplication<HypergraphT> &result) {
    Envr env;
    Model model = env.CreateModel("HypergraphPartRepl");

    this->SetupFundamentalVariablesConstraintsObjective(result.GetInstance(), model);
    SetupExtraVariablesConstraints(result.GetInstance(), model);

    if (this->useInitialSolution_) {
        SetInitialSolution(result, model);
    }

    this->SolveIlp(model);

    if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        result.SetAssignedPartitionVectors(this->ReadAllCoptAssignments(result.GetInstance(), model));
        return ReturnStatus::OSP_SUCCESS;

    } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        return ReturnStatus::ERROR;

    } else {
        if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            result.SetAssignedPartitionVectors(this->ReadAllCoptAssignments(result.GetInstance(), model));
            return ReturnStatus::OSP_SUCCESS;

        } else {
            return ReturnStatus::ERROR;
        }
    }
}

template <typename HypergraphT>
void HypergraphPartitioningILPWithReplication<HypergraphT>::SetupExtraVariablesConstraints(
    const PartitioningProblem<HypergraphT> &instance, Model &model) {
    using IndexType = typename HypergraphT::VertexIdx;

    const IndexType numberOfParts = instance.GetNumberOfPartitions();
    const IndexType numberOfVertices = instance.GetHypergraph().NumVertices();

    IndexType heaviestNode = 0;
    auto maxWeightFound = -1.0;

    for (IndexType node = 0; node < numberOfVertices; node++) {
        auto w = instance.GetHypergraph().GetVertexWorkWeight(node);
        if (w > maxWeightFound) {
            maxWeightFound = w;
            heaviestNode = node;
        }
    }
    // symmetry breaking
    model.AddConstr(this->nodeInPartition_[heaviestNode][0] == 1);

    if (replicationModel_ == ReplicationModelInIlp::GENERAL) {
        // create variables for each pin+partition combination
        std::map<std::pair<IndexType, unsigned>, IndexType> pinIdMap;
        IndexType nrOfPins = 0;
        for (IndexType node = 0; node < numberOfVertices; node++) {
            for (const IndexType &hyperedge : instance.GetHypergraph().GetIncidentHyperedges(node)) {
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
                for (const IndexType &hyperedge : instance.GetHypergraph().GetIncidentHyperedges(node)) {
                    model.AddConstr(this->nodeInPartition_[node][static_cast<int>(part)]
                                    >= pinCoveredByPartition[pinIdMap[std::make_pair(node, hyperedge)]][static_cast<int>(part)]);
                }
            }
        }

        // pin covering requires hyperedge use
        for (unsigned part = 0; part < numberOfParts; part++) {
            for (IndexType node = 0; node < numberOfVertices; node++) {
                for (const IndexType &hyperedge : instance.GetHypergraph().GetIncidentHyperedges(node)) {
                    model.AddConstr(this->hyperedgeUsesPartition_[hyperedge][static_cast<int>(part)]
                                    >= pinCoveredByPartition[pinIdMap[std::make_pair(node, hyperedge)]][static_cast<int>(part)]);
                }
            }
        }

    } else if (replicationModel_ == ReplicationModelInIlp::ONLY_TWICE) {
        // each node has one or two copies
        VarArray nodeReplicated = model.AddVars(static_cast<int>(numberOfVertices), COPT_BINARY, "node_replicated");

        double totalNodeWeight = 0;
        for (IndexType node = 0; node < numberOfVertices; node++) {
            totalNodeWeight += instance.GetHypergraph().GetVertexWorkWeight(node);
        }

        double totalSystemCapacity = static_cast<double>(numberOfParts) * instance.GetMaxWorkWeightPerPartition();
        double maxSlack = totalSystemCapacity - totalNodeWeight;

        // Constraint: Sum(Weight * ReplicatedStatus) <= MaxSlack
        Expr replicationMass;
        for (IndexType node = 0; node < numberOfVertices; node++) {
            replicationMass += instance.GetHypergraph().GetVertexWorkWeight(node) * nodeReplicated[static_cast<int>(node)];
        }

        model.AddConstr(replicationMass <= maxSlack);

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
                for (const IndexType &hyperedge : instance.GetHypergraph().GetIncidentHyperedges(node)) {
                    model.AddConstr(this->hyperedgeUsesPartition_[hyperedge][static_cast<int>(part)]
                                    >= this->nodeInPartition_[node][static_cast<int>(part)]
                                           - nodeReplicated[static_cast<int>(node)]);
                }
            }
        }

        // hyperedge indicators if node is replicated
        for (IndexType node = 0; node < numberOfVertices; node++) {
            for (const IndexType &hyperedge : instance.GetHypergraph().GetIncidentHyperedges(node)) {
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

    double totalNodeWeight = 0;
    for (IndexType v = 0; v < numberOfVertices; ++v) {
        totalNodeWeight += instance.GetHypergraph().GetVertexWorkWeight(v);
    }
    double totalCapacity = static_cast<double>(numberOfParts) * instance.GetMaxWorkWeightPerPartition();
    double replicationBudget = totalCapacity - totalNodeWeight;

    // If budget is negative (infeasible inputs), treat as 0
    if (replicationBudget < 0) {
        replicationBudget = 0;
    }

    struct StarCandidate {
        double savings;
        double cost;
        double ratio;
    };

    std::vector<StarCandidate> candidates;
    double totalPotentialCuts = 0.0;
    double partitionCapacity = instance.GetMaxWorkWeightPerPartition();

    for (IndexType centerNode = 0; centerNode < numberOfVertices; ++centerNode) {
        double centerWeight = instance.GetHypergraph().GetVertexWorkWeight(centerNode);

        // 1. Build the Star
        // Note: Check if your Hypergraph class uses .Pins(e) or .GetPins(e)
        std::vector<std::pair<double, double>> incidentEdges;
        double currentStarWeight = centerWeight;

        for (const IndexType &edgeIdx : instance.GetHypergraph().GetIncidentHyperedges(centerNode)) {
            double edgeWeight = instance.GetHypergraph().GetHyperedgeWeight(edgeIdx);
            double addedWeight = 0.0;

            // Iterate over nodes in this edge
            for (const IndexType &pin : instance.GetHypergraph().GetVerticesInHyperedge(edgeIdx)) {
                if (pin != centerNode) {
                    addedWeight += instance.GetHypergraph().GetVertexWorkWeight(pin);
                }
            }

            if (addedWeight > 0) {
                incidentEdges.push_back({edgeWeight, addedWeight});
                currentStarWeight += addedWeight;
            }
        }

        // 2. Only consider stars that don't fit in one partition
        if (currentStarWeight <= partitionCapacity) {
            continue;
        }

        // 3. Calculate Cut Cost if NOT replicated (Local Knapsack)
        std::sort(
            incidentEdges.begin(), incidentEdges.end(), [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
                return (a.first / a.second) < (b.first / b.second);
            });

        double localBoundCost = 0.0;
        double tempWeight = currentStarWeight;
        for (const auto &edge : incidentEdges) {
            if (tempWeight <= partitionCapacity) {
                break;
            }
            localBoundCost += edge.first;
            tempWeight -= edge.second;
        }

        if (localBoundCost > 1e-6) {
            // "Savings" is the cost we avoid by replicating this node
            // "Cost" is the weight of the node (capacity consumed)
            candidates.push_back({localBoundCost, centerWeight, localBoundCost / centerWeight});
            totalPotentialCuts += localBoundCost;
        }
    }

    // 4. Global Greedy Knapsack: Use budget to fix the most efficient stars
    std::sort(
        candidates.begin(), candidates.end(), [](const StarCandidate &a, const StarCandidate &b) { return a.ratio > b.ratio; });

    double currentBudget = replicationBudget;
    double attainableSavings = 0.0;

    for (const auto &cand : candidates) {
        if (currentBudget >= cand.cost) {
            currentBudget -= cand.cost;
            attainableSavings += cand.savings;
        } else {
            // Budget exhausted
            break;
        }
    }

    // 5. Apply the Valid Lower Bound
    double finalGlobalLB = totalPotentialCuts - attainableSavings;

    if (finalGlobalLB > 1e-6) {
        Expr objExpr;
        double totalEdgeConst = 0.0;

        // Reconstruct the objective expression to constrain it
        for (IndexType e = 0; e < instance.GetHypergraph().NumHyperedges(); ++e) {
            double w = instance.GetHypergraph().GetHyperedgeWeight(e);
            totalEdgeConst += w;
            for (unsigned k = 0; k < numberOfParts; ++k) {
                objExpr += w * this->hyperedgeUsesPartition_[e][static_cast<int>(k)];
            }
        }

        // Objective Function >= FinalGlobalLB
        // (Sum(w_e * y_{e,k}) - Sum(w_e)) >= finalGlobalLB
        model.AddConstr(objExpr - totalEdgeConst >= finalGlobalLB);

        std::cout << "Injecting Combinatorial Lower Bound: " << finalGlobalLB << std::endl;
    }
}

template <typename HypergraphT>
void HypergraphPartitioningILPWithReplication<HypergraphT>::SetInitialSolution(
    const PartitioningWithReplication<HypergraphT> &partition, Model &model) {
    using IndexType = typename HypergraphT::VertexIdx;

    const std::vector<std::vector<unsigned>> &assignments = partition.AssignedPartitions();
    const unsigned &numPartitions = partition.GetInstance().GetNumberOfPartitions();
    if (assignments.size() != partition.GetInstance().GetHypergraph().NumVertices()) {
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
