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

class PartitioningWithReplication {

  private:

    const PartitioningProblem *instance;

    std::vector<std::vector<unsigned> > node_to_partitions_assignment;

  public:
  
    PartitioningWithReplication() = delete;

    PartitioningWithReplication(const PartitioningProblem &inst)
        : instance(&inst), node_to_partitions_assignment(std::vector<std::vector<unsigned>>(inst.getHypergraph().num_vertices(), {0})) {}

    PartitioningWithReplication(const PartitioningProblem &inst, const std::vector<std::vector<unsigned> > &partition_assignment_)
        : instance(&inst), node_to_partitions_assignment(partition_assignment_) {}

    PartitioningWithReplication(const PartitioningWithReplication &partitioning_) = default;
    PartitioningWithReplication(PartitioningWithReplication &&partitioning_) = default;

    PartitioningWithReplication &operator=(const PartitioningWithReplication &partitioning_) = default;

    virtual ~PartitioningWithReplication() = default;


    // getters and setters

    inline const PartitioningProblem &getInstance() const { return *instance; }

    inline std::vector<unsigned> assignedPartitions(unsigned node) const { return node_to_partitions_assignment[node]; }
    inline const std::vector<std::vector<unsigned> > &assignedPartitions() const { return node_to_partitions_assignment; }
    inline std::vector<std::vector<unsigned> > &assignedPartitions() { return node_to_partitions_assignment; }

    inline void setAssignedPartitions(unsigned node, const std::vector<unsigned>& parts) { node_to_partitions_assignment.at(node) = parts; }
    void setAssignedPartitionVectors(const std::vector<std::vector<unsigned> > &vec) {

        if (vec.size() == static_cast<std::size_t>(instance->getHypergraph().num_vertices()) ) {
            node_to_partitions_assignment = vec;
        } else {
            throw std::invalid_argument(
                "Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }
    void setAssignedPartitionVectors(std::vector<std::vector<unsigned> > &&vec) {

        if (vec.size() == static_cast<std::size_t>(instance->getHypergraph().num_vertices()) ) {
            node_to_partitions_assignment = vec;
        } else {
            throw std::invalid_argument(
                "Invalid Argument while assigning processors: size does not match number of nodes.");
        }
    }

    std::vector<std::vector<unsigned> > getPartitionContents() const {

        std::vector<std::vector<unsigned> > content(instance->getNumberOfPartitions());
        for (unsigned node = 0; node < node_to_partitions_assignment.size(); ++node)
            for(unsigned part : node_to_partitions_assignment[node])
                content[part].push_back(node);

        return content;
    }

    void resetPartition() {
        node_to_partitions_assignment.clear();
        node_to_partitions_assignment.resize(instance->getHypergraph().num_vertices(), {0});
    }

    // costs and validity

    int computeConnectivityCost() const;
    int computeCutNetCost() const;

    bool satisfiesBalanceConstraint() const;

};

int PartitioningWithReplication::computeConnectivityCost() const {

    // naive implementation. in the worst-case this is exponential in the number of parts
    if(instance->getNumberOfPartitions() > 16)
        throw std::invalid_argument("Computing connectivity cost is not supported for more than 16 partitions.");

    int total = 0;
    std::vector<bool> part_used(instance->getNumberOfPartitions(), false);
    for(unsigned edge_idx = 0; edge_idx < instance->getHypergraph().num_hyperedges(); ++edge_idx)
    {
        const std::vector<unsigned> &hyperedge = instance->getHypergraph().get_vertices_in_hyperedge(edge_idx);
        if(hyperedge.empty())
            continue;

        unsigned long mask = 0UL;

        std::vector<unsigned> nr_nodes_covered_by_part(instance->getNumberOfPartitions(), 0);
        for(const unsigned& node : hyperedge)
            if(node_to_partitions_assignment[node].size() == 1)
                mask = mask | (1UL << node_to_partitions_assignment[node].front());

        unsigned min_parts_to_cover = instance->getNumberOfPartitions();
        unsigned long mask_limit = 1UL << instance->getNumberOfPartitions();
        for(unsigned long subset_mask = 1UL; subset_mask < mask_limit; ++subset_mask)
        {
            if((subset_mask & mask)!= mask)
                continue;
            
            unsigned nr_parts_used = 0;
            for(unsigned part = 0; part < instance->getNumberOfPartitions(); ++part)
            {
                part_used[part] = (((1UL << part) & subset_mask) > 0);
                nr_parts_used += static_cast<unsigned>(part_used[part]);
            }
            
            bool all_nodes_covered = true;
            for(const unsigned& node : hyperedge)
            {
                bool node_covered=false;
                for(unsigned part : node_to_partitions_assignment[node])
                    if(part_used[part])
                    {
                        node_covered = true;
                        break;
                    }
                if(!node_covered)
                {
                    all_nodes_covered = false;
                    break;
                }
            }
            if(all_nodes_covered)
                min_parts_to_cover = std::min(min_parts_to_cover, nr_parts_used);
        }
 
        total += static_cast<int>(min_parts_to_cover-1) * instance->getHypergraph().get_hyperedge_weight(edge_idx);
    }

    return total;
}

int PartitioningWithReplication::computeCutNetCost() const {

    int total = 0;
    for(unsigned edge_idx = 0; edge_idx < instance->getHypergraph().num_hyperedges(); ++edge_idx)
    {
        const std::vector<unsigned> &hyperedge = instance->getHypergraph().get_vertices_in_hyperedge(edge_idx);
        if(hyperedge.empty())
            continue;
        std::vector<unsigned> nr_nodes_covered_by_part(instance->getNumberOfPartitions(), 0);
        for(const unsigned& node : hyperedge)
            for(unsigned part : node_to_partitions_assignment[node])
                ++nr_nodes_covered_by_part[part];
        
        bool covers_all = false;
        for(unsigned part = 0; part < instance->getNumberOfPartitions(); ++part)
            if(nr_nodes_covered_by_part[part] == hyperedge.size())
                covers_all = true;
        
        if(!covers_all)
            total += instance->getHypergraph().get_hyperedge_weight(edge_idx);
    }

    return total;
}

bool PartitioningWithReplication::satisfiesBalanceConstraint() const {
    std::vector<int> work_weight(instance->getNumberOfPartitions(), 0);
    std::vector<int> memory_weight(instance->getNumberOfPartitions(), 0);
    for (unsigned node = 0; node < node_to_partitions_assignment.size(); ++node)
        for(unsigned part : node_to_partitions_assignment[node]){
            if (part > instance->getNumberOfPartitions())
                throw std::invalid_argument("Invalid Argument while checking balance constraint: partition ID out of range.");
            else
            {
                work_weight[part] += instance->getHypergraph().get_vertex_work_weight(node);
                memory_weight[part] += instance->getHypergraph().get_vertex_memory_weight(node);
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