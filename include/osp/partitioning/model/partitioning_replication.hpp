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

// Represents a partitioning where each vertex of a hypergraph can be assinged to one or more partitions

template <typename HypergraphT>
class PartitioningWithReplication {
  private:
    using IndexType = typename HypergraphT::VertexIdx;
    using WorkwType = typename HypergraphT::VertexWorkWeightType;
    using MemwType = typename HypergraphT::VertexMemWeightType;
    using CommwType = typename HypergraphT::VertexCommWeightType;

    const PartitioningProblem<HypergraphT> *instance_;

    std::vector<std::vector<unsigned>> nodeToPartitionsAssignment_;

  public:
    PartitioningWithReplication() = delete;

    PartitioningWithReplication(const PartitioningProblem<HypergraphT> &inst)
        : instance_(&inst),
          nodeToPartitionsAssignment_(std::vector<std::vector<unsigned>>(inst.GetHypergraph().NumVertices(), {0})) {}

    PartitioningWithReplication(const PartitioningProblem<HypergraphT> &inst,
                                const std::vector<std::vector<unsigned>> &partitionAssignment)
        : instance_(&inst), nodeToPartitionsAssignment_(partitionAssignment) {}

    PartitioningWithReplication(const PartitioningWithReplication<HypergraphT> &partitioning) = default;
    PartitioningWithReplication(PartitioningWithReplication<HypergraphT> &&partitioning) = default;

    PartitioningWithReplication &operator=(const PartitioningWithReplication<HypergraphT> &partitioning) = default;

    virtual ~PartitioningWithReplication() = default;

    // getters and setters

    inline const PartitioningProblem<HypergraphT> &GetInstance() const { return *instance_; }

    inline std::vector<unsigned> AssignedPartitions(IndexType node) const { return nodeToPartitionsAssignment_[node]; }

    inline const std::vector<std::vector<unsigned>> &AssignedPartitions() const { return nodeToPartitionsAssignment_; }

    inline std::vector<std::vector<unsigned>> &AssignedPartitions() { return nodeToPartitionsAssignment_; }

    inline void SetAssignedPartitions(IndexType node, const std::vector<unsigned> &parts) {
        nodeToPartitionsAssignment_.at(node) = parts;
    }

    void SetAssignedPartitionVectors(const std::vector<std::vector<unsigned>> &vec) {
        if (vec.size() == static_cast<std::size_t>(instance_->getHypergraph().NumVertices())) {
            nodeToPartitionsAssignment_ = vec;
        } else {
            throw std::invalid_argument("Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }

    void SetAssignedPartitionVectors(std::vector<std::vector<unsigned>> &&vec) {
        if (vec.size() == static_cast<std::size_t>(instance_->getHypergraph().NumVertices())) {
            nodeToPartitionsAssignment_ = vec;
        } else {
            throw std::invalid_argument("Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }

    std::vector<std::vector<IndexType>> GetPartitionContents() const {
        std::vector<std::vector<IndexType>> content(instance_->GetNumberOfPartitions());
        for (IndexType node = 0; node < nodeToPartitionsAssignment_.size(); ++node) {
            for (unsigned part : nodeToPartitionsAssignment_[node]) {
                content[part].push_back(node);
            }
        }

        return content;
    }

    void ResetPartition() {
        nodeToPartitionsAssignment_.clear();
        nodeToPartitionsAssignment_.resize(instance_->GetHypergraph().NumVertices(), {0});
    }

    // costs and validity

    CommwType ComputeConnectivityCost() const;
    CommwType ComputeCutNetCost() const;

    bool SatisfiesBalanceConstraint() const;
};

template <typename HypergraphT>
typename HypergraphT::VertexCommWeightType PartitioningWithReplication<HypergraphT>::ComputeConnectivityCost() const {
    // naive implementation. in the worst-case this is exponential in the number of parts
    if (instance_->GetNumberOfPartitions() > 16) {
        throw std::invalid_argument("Computing connectivity cost is not supported for more than 16 partitions.");
    }

    CommwType total = 0;
    std::vector<bool> partUsed(instance_->GetNumberOfPartitions(), false);
    for (IndexType edgeIdx = 0; edgeIdx < instance_->GetHypergraph().NumHyperedges(); ++edgeIdx) {
        const std::vector<IndexType> &hyperedge = instance_->GetHypergraph().GetVerticesInHyperedge(edgeIdx);
        if (hyperedge.empty()) {
            continue;
        }

        unsigned long mask = 0UL;

        std::vector<IndexType> nrNodesCoveredByPart(instance_->GetNumberOfPartitions(), 0);
        for (const IndexType &node : hyperedge) {
            if (nodeToPartitionsAssignment_[node].size() == 1) {
                mask = mask | (1UL << nodeToPartitionsAssignment_[node].front());
            }
        }

        unsigned minPartsToCover = instance_->GetNumberOfPartitions();
        unsigned long maskLimit = 1UL << instance_->GetNumberOfPartitions();
        for (unsigned long subsetMask = 1UL; subsetMask < maskLimit; ++subsetMask) {
            if ((subsetMask & mask) != mask) {
                continue;
            }

            unsigned nrPartsUsed = 0;
            for (unsigned part = 0; part < instance_->GetNumberOfPartitions(); ++part) {
                partUsed[part] = (((1UL << part) & subsetMask) > 0);
                nrPartsUsed += static_cast<unsigned>(partUsed[part]);
            }

            bool allNodesCovered = true;
            for (const IndexType &node : hyperedge) {
                bool nodeCovered = false;
                for (unsigned part : nodeToPartitionsAssignment_[node]) {
                    if (partUsed[part]) {
                        nodeCovered = true;
                        break;
                    }
                }
                if (!nodeCovered) {
                    allNodesCovered = false;
                    break;
                }
            }
            if (allNodesCovered) {
                minPartsToCover = std::min(minPartsToCover, nrPartsUsed);
            }
        }

        total += static_cast<CommwType>(minPartsToCover - 1) * instance_->GetHypergraph().GetHyperedgeWeight(edgeIdx);
    }

    return total;
}

template <typename HypergraphT>
typename HypergraphT::VertexCommWeightType PartitioningWithReplication<HypergraphT>::ComputeCutNetCost() const {
    CommwType total = 0;
    for (IndexType edgeIdx = 0; edgeIdx < instance_->GetHypergraph().NumHyperedges(); ++edgeIdx) {
        const std::vector<IndexType> &hyperedge = instance_->GetHypergraph().GetVerticesInHyperedge(edgeIdx);
        if (hyperedge.empty()) {
            continue;
        }
        std::vector<IndexType> nrNodesCoveredByPart(instance_->GetNumberOfPartitions(), 0);
        for (const IndexType &node : hyperedge) {
            for (unsigned part : nodeToPartitionsAssignment_[node]) {
                ++nrNodesCoveredByPart[part];
            }
        }

        bool coversAll = false;
        for (unsigned part = 0; part < instance_->GetNumberOfPartitions(); ++part) {
            if (nrNodesCoveredByPart[part] == hyperedge.size()) {
                coversAll = true;
            }
        }

        if (!coversAll) {
            total += instance_->GetHypergraph().GetHyperedgeWeight(edgeIdx);
        }
    }

    return total;
}

template <typename HypergraphT>
bool PartitioningWithReplication<HypergraphT>::SatisfiesBalanceConstraint() const {
    std::vector<WorkwType> workWeight(instance_->GetNumberOfPartitions(), 0);
    std::vector<MemwType> memoryWeight(instance_->GetNumberOfPartitions(), 0);
    for (IndexType node = 0; node < nodeToPartitionsAssignment_.size(); ++node) {
        for (unsigned part : nodeToPartitionsAssignment_[node]) {
            if (part > instance_->GetNumberOfPartitions()) {
                throw std::invalid_argument("Invalid Argument while checking balance constraint: partition ID out of range.");
            } else {
                workWeight[part] += instance_->GetHypergraph().GetVertexWorkWeight(node);
                memoryWeight[part] += instance_->GetHypergraph().GetVertexMemoryWeight(node);
            }
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
