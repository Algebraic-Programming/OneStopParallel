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

#include "osp/partitioning/model/partitioning.hpp"
#include <cmath>
#include <algorithm>

namespace osp{

class GenericFM {

  protected:
    unsigned max_number_of_passes = 10;
    unsigned max_nodes_in_part = 0;

  public:

    void ImprovePartitioning(Partitioning& partition);

    inline unsigned getMaxNumberOfPasses() const { return max_number_of_passes; }
    inline void setMaxNumberOfPasses(unsigned passes_) { max_number_of_passes = passes_; }
    inline unsigned getMaxNodesInPart() const { return max_nodes_in_part; }
    inline void setMaxNodesInPart(unsigned max_nodes_) { max_nodes_in_part = max_nodes_; }
};

void GenericFM::ImprovePartitioning(Partitioning& partition)
{
    // Note: this algorithm disregards hyperedge weights, in order to keep the size of the gain bucket array bounded!

    if(partition.getInstance().getNumberOfPartitions() != 2)
    {
        std::cout << "Error: FM can only be used for 2 partitions." << std::endl;
        return;
    }
    
    if(!partition.satisfiesBalanceConstraint())
    {
        std::cout << "Error: initial partition to FM does not satisfy balance constraint." << std::endl;
        return;
    }

    const Hypergraph& Hgraph = partition.getInstance().getHypergraph();

    unsigned max_degree = 0;
    for(unsigned node = 0; node < Hgraph.num_vertices(); ++node)
        max_degree = std::max(max_degree, static_cast<unsigned>(Hgraph.get_incident_hyperedges(node).size()));

    if(max_nodes_in_part == 0) // if not initialized
        max_nodes_in_part = static_cast<unsigned>(ceil(static_cast<double>(Hgraph.num_vertices()) * static_cast<double>(partition.getInstance().getMaxWorkWeightPerPartition())
                                         / static_cast<double>(Hgraph.compute_total_vertex_work_weight()) ));

    for(unsigned pass_idx = 0; pass_idx < max_number_of_passes; ++pass_idx)
    {
        std::vector<unsigned> node_to_new_part = partition.assignedPartitions();
        std::vector<bool> locked(Hgraph.num_vertices(), false);
        std::vector<int> gain(Hgraph.num_vertices(), 0);
        std::vector<std::vector<unsigned> > nr_nodes_in_hyperedge_on_side(Hgraph.num_hyperedges(), std::vector<unsigned>(2, 0));
        int cost = 0;

        unsigned left_side = 0;
        for(unsigned node = 0; node < Hgraph.num_vertices(); ++node)
            if(partition.assignedPartition(node) == 0)
                ++left_side;

        // Initialize gain values
        for(unsigned hyperedge = 0; hyperedge < Hgraph.num_hyperedges(); ++hyperedge)
        {
            for(unsigned node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                ++nr_nodes_in_hyperedge_on_side[hyperedge][partition.assignedPartition(node)];

            if(Hgraph.get_vertices_in_hyperedge(hyperedge).size() < 2)
                continue;
            
            for(unsigned part = 0; part < 2; ++part)
            {
                if(nr_nodes_in_hyperedge_on_side[hyperedge][part] == 1)
                    for(unsigned node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                        if(partition.assignedPartition(node) == part)
                            ++gain[node];
            }
        }

        // build gain bucket array
        std::vector<int> max_gain(2, -static_cast<int>(max_degree)-1);
        std::vector<std::vector<std::vector<unsigned> > > gain_bucket_array(2, std::vector<std::vector<unsigned> >(2*max_degree+1));
        for(unsigned node = 0; node < Hgraph.num_vertices(); ++node)
        {
            const unsigned& part = partition.assignedPartition(node);
            gain_bucket_array[part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))].push_back(node);
            max_gain[part] = std::max(max_gain[part], gain[node]);
        }

        unsigned best_index = 0;
        int best_cost = 0;
        std::vector<unsigned> moved_nodes;

        // the pass itself: make moves
        while(moved_nodes.size() < Hgraph.num_vertices())
        {
            // select move
            unsigned to_move = std::numeric_limits<unsigned>::max();
            unsigned chosen_part = std::numeric_limits<unsigned>::max();

            unsigned gain_index = static_cast<unsigned>(std::max(max_gain[0], max_gain[1]) + static_cast<int>(max_degree));
            while(gain_index < std::numeric_limits<unsigned>::max())
            {
                bool can_choose_left = (Hgraph.num_vertices() - left_side < max_nodes_in_part) && !gain_bucket_array[0][gain_index].empty();
                bool can_choose_right = (left_side < max_nodes_in_part) && !gain_bucket_array[1][gain_index].empty();

                if(can_choose_left && can_choose_right)
                    chosen_part = (left_side >= Hgraph.num_vertices() / 2) ? 1 : 0;
                else if(can_choose_left)
                    chosen_part = 0;
                else if(can_choose_right)
                    chosen_part = 1;

                if(chosen_part < 2)
                {
                    to_move = gain_bucket_array[chosen_part][gain_index].back();
                    gain_bucket_array[chosen_part][gain_index].pop_back();
                    break;
                }
                --gain_index;
            }

            if(to_move == std::numeric_limits<unsigned>::max())
                break;
            
            // make move

            moved_nodes.push_back(to_move);
            cost -= gain[to_move];
            if(cost < best_cost)
            {
                best_cost = cost;
                best_index = static_cast<unsigned>(moved_nodes.size()) + 1;
            }
            locked[to_move] = true;
            node_to_new_part[to_move] = 1 - node_to_new_part[to_move];

            if(chosen_part == 0)
                --left_side;
            else
                ++left_side;

            unsigned other_part = 1-chosen_part;

            // update gain values
            for(unsigned hyperedge : Hgraph.get_incident_hyperedges(to_move))
            {
                if(nr_nodes_in_hyperedge_on_side[hyperedge][chosen_part] == 1)
                {
                    for(unsigned node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                    {
                        if(locked[node])
                            continue;

                        std::vector<unsigned>& vec = gain_bucket_array[other_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))];
                        vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
                        --gain[node];
                        gain_bucket_array[other_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))].push_back(node);
                    }
                }
                else if(nr_nodes_in_hyperedge_on_side[hyperedge][chosen_part] == 2)
                {
                    for(unsigned node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                    {
                        if(node_to_new_part[node] == chosen_part && !locked[node])
                        {
                            std::vector<unsigned>& vec = gain_bucket_array[chosen_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))];
                            vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
                            ++gain[node];
                            gain_bucket_array[chosen_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))].push_back(node);
                            max_gain[chosen_part] = std::max(max_gain[chosen_part], gain[node]);
                            break;
                        }
                    }
                }
                if(nr_nodes_in_hyperedge_on_side[hyperedge][other_part] == 1)
                {
                    for(unsigned node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                    {
                        if(node_to_new_part[node] == other_part && !locked[node])
                        {
                            std::vector<unsigned>& vec = gain_bucket_array[other_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))];
                            vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
                            --gain[node];
                            gain_bucket_array[other_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))].push_back(node);
                            break;
                        }
                    }
                }
                else if(nr_nodes_in_hyperedge_on_side[hyperedge][other_part] == 0)
                {
                    for(unsigned node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                    {
                        if(locked[node])
                            continue;

                        std::vector<unsigned>& vec = gain_bucket_array[chosen_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))];
                        vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
                        ++gain[node];
                        gain_bucket_array[chosen_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))].push_back(node);
                        max_gain[chosen_part] = std::max(max_gain[chosen_part], gain[node]);
                    }
                }
                --nr_nodes_in_hyperedge_on_side[hyperedge][chosen_part];
                ++nr_nodes_in_hyperedge_on_side[hyperedge][other_part];
            }
        }

        // apply best configuration seen
        if(best_index == 0)
            break;

        for(unsigned node_idx = 0; node_idx < best_index; ++node_idx)
            partition.setAssignedPartition(moved_nodes[node_idx], 1U-partition.assignedPartition(moved_nodes[node_idx]));

    }
}

} // namespace osp