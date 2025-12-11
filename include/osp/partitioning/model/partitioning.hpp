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

#include <vector>

#include "osp/partitioning/model/partitioning_problem.hpp"

namespace osp {

// Represents a partitioning where each vertex of a hypergraph is assigned to a specifc partition

template <typename HypergraphT>
class Partitioning {
  private:
    using IndexType = typename HypergraphT::VertexIdx;
    using WorkwType = typename HypergraphT::VertexWorkWeightType;
    using MemwType = typename HypergraphT::VertexMemWeightType;
    using CommwType = typename HypergraphT::VertexCommWeightType;

    const PartitioningProblem<HypergraphT> *instance_;

    std::vector<unsigned> nodeToPartitionAssignment_;

  public:
    Partitioning() = delete;

    Partitioning(const PartitioningProblem<HypergraphT> &inst)
        : instance_(&inst), nodeToPartitionAssignment_(std::vector<unsigned>(inst.GetHypergraph().NumVertices(), 0)) {}

    Partitioning(const PartitioningProblem<HypergraphT> &inst, const std::vector<unsigned> &partitionAssignment)
        : instance_(&inst), nodeToPartitionAssignment_(partitionAssignment) {}

    Partitioning(const Partitioning<HypergraphT> &partitioning) = default;
    Partitioning(Partitioning<HypergraphT> &&partitioning) = default;

    Partitioning &operator=(const Partitioning<HypergraphT> &partitioning) = default;

    virtual ~Partitioning() = default;

    // getters and setters

    inline const PartitioningProblem<HypergraphT> &GetInstance() const { return *instance_; }

    inline unsigned AssignedPartition(IndexType node) const { return nodeToPartitionAssignment_[node]; }

    inline const std::vector<unsigned> &AssignedPartitions() const { return nodeToPartitionAssignment_; }

    inline std::vector<unsigned> &AssignedPartitions() { return nodeToPartitionAssignment_; }

    inline void SetAssignedPartition(IndexType node, unsigned part) { nodeToPartitionAssignment_.at(node) = part; }

    void SetAssignedPartitions(const std::vector<unsigned> &vec) {
        if (vec.size() == static_cast<std::size_t>(instance_->getHypergraph().num_vertices())) {
            nodeToPartitionAssignment_ = vec;
        } else {
            throw std::invalid_argument("Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }

    void SetAssignedPartitions(std::vector<unsigned> &&vec) {
        if (vec.size() == static_cast<std::size_t>(instance_->getHypergraph().num_vertices())) {
            nodeToPartitionAssignment_ = vec;
        } else {
            throw std::invalid_argument("Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }

    std::vector<IndexType> GetPartitionContent(unsigned part) const {
        std::vector<IndexType> content;
        for (IndexType node = 0; node < nodeToPartitionAssignment_.size(); ++node) {
            if (nodeToPartitionAssignment_[node] == part) {
                content.push_back(node);
            }
        }

        return content;
    }

    void ResetPartition() {
        nodeToPartitionAssignment_.clear();
        nodeToPartitionAssignment_.resize(instance_->GetHypergraph().NumVertices(), 0);
    }

    // costs and validity

    std::vector<unsigned> ComputeLambdaForHyperedges() const;
    CommwType ComputeConnectivityCost() const;
    CommwType ComputeCutNetCost() const;

    bool SatisfiesBalanceConstraint() const;
};

template <typename HypergraphT>
std::vector<unsigned> Partitioning<HypergraphT>::ComputeLambdaForHyperedges() const {
    std::vector<unsigned> lambda(instance_->GetHypergraph().NumHyperedges(), 0);
    for (IndexType edgeIdx = 0; edgeIdx < instance_->GetHypergraph().NumHyperedges(); ++edgeIdx) {
        const std::vector<IndexType> &hyperedge = instance_->GetHypergraph().GetVerticesInHyperedge(edgeIdx);
        if (hyperedge.empty()) {
            continue;
        }
        std::vector<bool> intersectsPart(instance_->GetNumberOfPartitions(), false);
        for (const IndexType &node : hyperedge) {
            intersectsPart[nodeToPartitionAssignment_[node]] = true;
        }
        for (unsigned part = 0; part < instance_->GetNumberOfPartitions(); ++part) {
            if (intersectsPart[part]) {
                ++lambda[edgeIdx];
            }
        }
    }
    return lambda;
}

template <typename HypergraphT>
typename HypergraphT::vertex_comm_weight_type Partitioning<HypergraphT>::ComputeConnectivityCost() const {
    CommwType total = 0;
    std::vector<unsigned> lambda = ComputeLambdaForHyperedges();

    for (IndexType edgeIdx = 0; edgeIdx < instance_->GetHypergraph().NumHyperedges(); ++edgeIdx) {
        if (lambda[edgeIdx] >= 1) {
            total += (static_cast<CommwType>(lambda[edgeIdx]) - 1) * instance_->GetHypergraph().GetHyperedgeWeight(edgeIdx);
        }
    }

    return total;
}

template <typename HypergraphT>
typename HypergraphT::vertex_comm_weight_type Partitioning<HypergraphT>::ComputeCutNetCost() const {
    CommwType total = 0;
    std::vector<unsigned> lambda = ComputeLambdaForHyperedges();
    for (IndexType edgeIdx = 0; edgeIdx < instance_->GetHypergraph().NumHyperedges(); ++edgeIdx) {
        if (lambda[edgeIdx] > 1) {
            total += instance_->GetHypergraph().GetHyperedgeWeight(edgeIdx);
        }
    }

    return total;
}

template <typename HypergraphT>
bool Partitioning<HypergraphT>::SatisfiesBalanceConstraint() const {
    std::vector<WorkwType> workWeight(instance_->GetNumberOfPartitions(), 0);
    std::vector<MemwType> memoryWeight(instance_->GetNumberOfPartitions(), 0);
    for (IndexType node = 0; node < nodeToPartitionAssignment_.size(); ++node) {
        if (nodeToPartitionAssignment_[node] > instance_->GetNumberOfPartitions()) {
            throw std::invalid_argument("Invalid Argument while checking balance constraint: partition ID out of range.");
        } else {
            workWeight[nodeToPartitionAssignment_[node]] += instance_->GetHypergraph().GetVertexWorkWeight(node);
            memoryWeight[nodeToPartitionAssignment_[node]] += instance_->GetHypergraph().GetVertexMemoryWeight(node);
        }
    }

    for (unsigned part = 0; part < instance_->GetNumberOfPartitions(); ++part) {
        if (workWeight[part] > instance_->GetMaxWorkWeightPerPartition()) {
            return false;
        }
        if (memoryWeight[part] > instance_->GetMaxMemoryWeightPerPartition()) {
            return false;
        }
    }

    return true;
}

}    // namespace osp
