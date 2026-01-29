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

#include <cmath>
#include <iostream>

#include "osp/partitioning/model/hypergraph_utility.hpp"

namespace osp {

// represents a hypergraph partitioning problem into a fixed number of parts with a balance constraint
template <typename HypergraphT>
class PartitioningProblem {
  private:
    using ThisT = PartitioningProblem<HypergraphT>;

    using IndexType = typename HypergraphT::VertexIdx;
    using WorkwType = typename HypergraphT::VertexWorkWeightType;
    using MemwType = typename HypergraphT::VertexMemWeightType;
    using CommwType = typename HypergraphT::VertexCommWeightType;

    HypergraphT hgraph_;

    unsigned nrOfPartitions_;
    WorkwType maxWorkWeightPerPartition_ = std::numeric_limits<WorkwType>::max();
    MemwType maxMemoryWeightPerPartition_ = std::numeric_limits<MemwType>::max();

    bool allowsReplication_ = false;

  public:
    PartitioningProblem() = default;

    PartitioningProblem(const HypergraphT &hgraph,
                        unsigned nrParts = 2,
                        WorkwType maxWorkWeight = std::numeric_limits<WorkwType>::max(),
                        MemwType maxMemoryWeight = std::numeric_limits<MemwType>::max())
        : hgraph_(hgraph),
          nrOfPartitions_(nrParts),
          maxWorkWeightPerPartition_(maxWorkWeight),
          maxMemoryWeightPerPartition_(maxMemoryWeight) {}

    PartitioningProblem(const HypergraphT &&hgraph,
                        unsigned nrParts = 2,
                        WorkwType maxWorkWeight = std::numeric_limits<WorkwType>::max(),
                        MemwType maxMemoryWeight = std::numeric_limits<MemwType>::max())
        : hgraph_(hgraph),
          nrOfPartitions_(nrParts),
          maxWorkWeightPerPartition_(maxWorkWeight),
          maxMemoryWeightPerPartition_(maxMemoryWeight) {}

    PartitioningProblem(const ThisT &other) = default;
    PartitioningProblem(ThisT &&other) = default;

    PartitioningProblem &operator=(const ThisT &other) = default;
    PartitioningProblem &operator=(ThisT &&other) = default;

    // getters
    inline const HypergraphT &GetHypergraph() const { return hgraph_; }

    inline HypergraphT &GetHypergraph() { return hgraph_; }

    inline unsigned GetNumberOfPartitions() const { return nrOfPartitions_; }

    inline WorkwType GetMaxWorkWeightPerPartition() const { return maxWorkWeightPerPartition_; }

    inline MemwType GetMaxMemoryWeightPerPartition() const { return maxMemoryWeightPerPartition_; }

    inline bool GetAllowsReplication() const { return allowsReplication_; }

    // setters
    inline void SetHypergraph(const HypergraphT &hgraph) { hgraph_ = hgraph; }

    inline void SetNumberOfPartitions(unsigned nrParts) { nrOfPartitions_ = nrParts; }

    inline void SetAllowsReplication(bool allowed) { allowsReplication_ = allowed; }

    inline void SetMaxWorkWeightExplicitly(WorkwType maxWeight) { maxWorkWeightPerPartition_ = maxWeight; }

    void SetMaxWorkWeightViaImbalanceFactor(double imbalance) {
        if (imbalance < 0) {
            throw std::invalid_argument("Invalid Argument while setting imbalance parameter: parameter is negative.");
        } else {
            maxWorkWeightPerPartition_ = static_cast<WorkwType>(
                ceil(ComputeTotalVertexWorkWeight(hgraph_) / static_cast<double>(nrOfPartitions_) * (1.0 + imbalance)));
        }
    }

    inline void SetMaxMemoryWeightExplicitly(MemwType maxWeight) { maxMemoryWeightPerPartition_ = maxWeight; }

    void SetMaxMemoryWeightViaImbalanceFactor(double imbalance) {
        if (imbalance < 0) {
            throw std::invalid_argument("Invalid Argument while setting imbalance parameter: parameter is negative.");
        } else {
            maxMemoryWeightPerPartition_ = static_cast<MemwType>(
                ceil(ComputeTotalVertexMemoryWeight(hgraph_) / static_cast<double>(nrOfPartitions_) * (1.0 + imbalance)));
        }
    }
};

}    // namespace osp
