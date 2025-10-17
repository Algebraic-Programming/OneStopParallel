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

template<typename index_type = size_t, typename workw_type = int, typename memw_type = int, typename commw_type = int>
class Partitioning {

  private:

    const PartitioningProblem<index_type, workw_type, memw_type, commw_type> *instance;

    std::vector<unsigned> node_to_partition_assignment;

  public:
  
    Partitioning() = delete;

    Partitioning(const PartitioningProblem<index_type, workw_type, memw_type, commw_type> &inst)
        : instance(&inst), node_to_partition_assignment(std::vector<unsigned>(inst.getHypergraph().num_vertices(), 0)) {}

    Partitioning(const PartitioningProblem<index_type, workw_type, memw_type, commw_type> &inst, const std::vector<unsigned> &partition_assignment_)
        : instance(&inst), node_to_partition_assignment(partition_assignment_) {}

    Partitioning(const Partitioning<index_type, workw_type, memw_type, commw_type> &partitioning_) = default;
    Partitioning(Partitioning<index_type, workw_type, memw_type, commw_type> &&partitioning_) = default;

    Partitioning &operator=(const Partitioning<index_type, workw_type, memw_type, commw_type> &partitioning_) = default;

    virtual ~Partitioning() = default;


    // getters and setters

    inline const PartitioningProblem<index_type, workw_type, memw_type, commw_type> &getInstance() const { return *instance; }

    inline unsigned assignedPartition(index_type node) const { return node_to_partition_assignment[node]; }
    inline const std::vector<unsigned> &assignedPartitions() const { return node_to_partition_assignment; }
    inline std::vector<unsigned> &assignedPartitions() { return node_to_partition_assignment; }

    inline void setAssignedPartition(index_type node, unsigned part) { node_to_partition_assignment.at(node) = part; }
    void setAssignedPartitions(const std::vector<unsigned> &vec) {

        if (vec.size() == static_cast<std::size_t>(instance->getHypergraph().num_vertices()) ) {
            node_to_partition_assignment = vec;
        } else {
            throw std::invalid_argument(
                "Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }
    void setAssignedPartitions(std::vector<unsigned> &&vec) {

        if (vec.size() == static_cast<std::size_t>(instance->getHypergraph().num_vertices()) ) {
            node_to_partition_assignment = vec;
        } else {
            throw std::invalid_argument(
                "Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }

    std::vector<index_type> getPartitionContent(unsigned part) const {

        std::vector<index_type> content;
        for (index_type node = 0; node < node_to_partition_assignment.size(); ++node) {

            if (node_to_partition_assignment[node] == part) {
                content.push_back(node);
            }
        }

        return content;
    }

    void resetPartition() {
        node_to_partition_assignment.clear();
        node_to_partition_assignment.resize(instance->getHypergraph().num_vertices(), 0);
    }

    // costs and validity

    std::vector<unsigned> computeLambdaForHyperedges() const;
    commw_type computeConnectivityCost() const;
    commw_type computeCutNetCost() const;

    bool satisfiesBalanceConstraint() const;

};

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
std::vector<unsigned> Partitioning<index_type, workw_type, memw_type, commw_type>::computeLambdaForHyperedges() const
{
    std::vector<unsigned> lambda(instance->getHypergraph().num_hyperedges(), 0);
    for(index_type edge_idx = 0; edge_idx < instance->getHypergraph().num_hyperedges(); ++edge_idx)
    {
        const std::vector<index_type> &hyperedge = instance->getHypergraph().get_vertices_in_hyperedge(edge_idx);
        if(hyperedge.empty())
            continue;
        std::vector<bool> intersects_part(instance->getNumberOfPartitions(), false);
        for(const index_type& node : hyperedge)
            intersects_part[node_to_partition_assignment[node]] = true;
        for(unsigned part = 0; part < instance->getNumberOfPartitions(); ++part)
            if(intersects_part[part])
                ++lambda[edge_idx];
    }
    return lambda;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
commw_type Partitioning<index_type, workw_type, memw_type, commw_type>::computeConnectivityCost() const {

    commw_type total = 0;
    std::vector<unsigned> lambda = computeLambdaForHyperedges();
    
    for(index_type edge_idx = 0; edge_idx < instance->getHypergraph().num_hyperedges(); ++edge_idx)
        if(lambda[edge_idx] >= 1)
            total += (static_cast<commw_type>(lambda[edge_idx])-1) * instance->getHypergraph().get_hyperedge_weight(edge_idx);
    
    return total;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
commw_type Partitioning<index_type, workw_type, memw_type, commw_type>::computeCutNetCost() const {

    commw_type total = 0;
    std::vector<unsigned> lambda = computeLambdaForHyperedges();
    for(index_type edge_idx = 0; edge_idx < instance->getHypergraph().num_hyperedges(); ++edge_idx)
        if(lambda[edge_idx] > 1)
            total += instance->getHypergraph().get_hyperedge_weight(edge_idx);
    
    return total;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
bool Partitioning<index_type, workw_type, memw_type, commw_type>::satisfiesBalanceConstraint() const {
    std::vector<workw_type> work_weight(instance->getNumberOfPartitions(), 0);
    std::vector<memw_type> memory_weight(instance->getNumberOfPartitions(), 0);
    for (index_type node = 0; node < node_to_partition_assignment.size(); ++node) {
        if (node_to_partition_assignment[node] > instance->getNumberOfPartitions())
            throw std::invalid_argument("Invalid Argument while checking balance constraint: partition ID out of range.");
        else
        {
            work_weight[node_to_partition_assignment[node]] += instance->getHypergraph().get_vertex_work_weight(node);
            memory_weight[node_to_partition_assignment[node]] += instance->getHypergraph().get_vertex_memory_weight(node);
        }
    }

    for(unsigned part = 0; part < instance->getNumberOfPartitions(); ++part)
    {
        if(work_weight[part] > instance->getMaxWorkWeightPerPartition())
            return false;
        if(memory_weight[part] > instance->getMaxMemoryWeightPerPartition())
            return false;
    }

    return true;
};

} // namespace osp