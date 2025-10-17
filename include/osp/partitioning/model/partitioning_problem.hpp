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

#include <iostream>
#include <cmath>

#include "osp/partitioning/model/hypergraph.hpp"

namespace osp {

// represents a hypergraph partitioning problem into a fixed number of parts with a balance constraint
template<typename index_type = size_t, typename workw_type = int, typename memw_type = int, typename commw_type = int>
class PartitioningProblem {

  private:
    Hypergraph<index_type, workw_type, memw_type, commw_type> hgraph;

    unsigned nr_of_partitions;
    workw_type max_work_weight_per_partition;
    memw_type max_memory_weight_per_partition;

    bool allows_replication = false;

  public:

    PartitioningProblem() = default;

    PartitioningProblem(const Hypergraph<index_type, workw_type, memw_type, commw_type> &hgraph_, unsigned nr_parts_ = 2,
                        workw_type max_work_weight_ = std::numeric_limits<workw_type>::max(),
                        memw_type max_memory_weight_ = std::numeric_limits<memw_type>::max()) :
                        hgraph(hgraph_), nr_of_partitions(nr_parts_),
                        max_work_weight_per_partition(max_work_weight_), max_memory_weight_per_partition(max_memory_weight_) {}

    PartitioningProblem(const Hypergraph<index_type, workw_type, memw_type, commw_type> &&hgraph_, unsigned nr_parts_ = 2,
                        workw_type max_work_weight_ = std::numeric_limits<workw_type>::max(),
                        memw_type max_memory_weight_ = std::numeric_limits<memw_type>::max()) :
                        hgraph(hgraph_), nr_of_partitions(nr_parts_),
                        max_work_weight_per_partition(max_work_weight_), max_memory_weight_per_partition(max_memory_weight_) {}

    PartitioningProblem(const PartitioningProblem<index_type, workw_type, memw_type, commw_type> &other) = default;
    PartitioningProblem(PartitioningProblem<index_type, workw_type, memw_type, commw_type> &&other) = default;

    PartitioningProblem &operator=(const PartitioningProblem<index_type, workw_type, memw_type, commw_type> &other) = default;
    PartitioningProblem &operator=(PartitioningProblem<index_type, workw_type, memw_type, commw_type> &&other) = default;

    // getters
    inline const Hypergraph<index_type, workw_type, memw_type, commw_type> &getHypergraph() const { return hgraph; }
    inline Hypergraph<index_type, workw_type, memw_type, commw_type> &getHypergraph() { return hgraph; }

    inline unsigned getNumberOfPartitions() const { return nr_of_partitions; }
    inline workw_type getMaxWorkWeightPerPartition() const { return max_work_weight_per_partition; }
    inline memw_type getMaxMemoryWeightPerPartition() const { return max_memory_weight_per_partition; }
    inline bool getAllowsReplication() const { return allows_replication; }

    // setters
    inline void setHypergraph(const Hypergraph<index_type, workw_type, memw_type, commw_type> &hgraph_) { hgraph = hgraph_; }
    
    inline void setNumberOfPartitions(unsigned nr_parts_) { nr_of_partitions = nr_parts_; }
    inline void setAllowsReplication(bool allowed_) { allows_replication = allowed_; }

    inline void setMaxWorkWeightExplicitly(workw_type max_weight_) { max_work_weight_per_partition = max_weight_; }
    void setMaxWorkWeightViaImbalanceFactor(double imbalance){
        if(imbalance < 0 )
            throw std::invalid_argument("Invalid Argument while setting imbalance parameter: parameter is negative.");
        else
            max_work_weight_per_partition = static_cast<workw_type>(ceil(hgraph.compute_total_vertex_work_weight()/ static_cast<double>(nr_of_partitions) * (1.0+imbalance)));
    }
    inline void setMaxMemoryWeightExplicitly(memw_type max_weight_) { max_memory_weight_per_partition = max_weight_; }
    void setMaxMemoryWeightViaImbalanceFactor(double imbalance){
        if(imbalance < 0 )
            throw std::invalid_argument("Invalid Argument while setting imbalance parameter: parameter is negative.");
        else
            max_memory_weight_per_partition = static_cast<memw_type>(ceil(hgraph.compute_total_vertex_memory_weight()/ static_cast<double>(nr_of_partitions) * (1.0+imbalance)));
    }
};


} // namespace osp