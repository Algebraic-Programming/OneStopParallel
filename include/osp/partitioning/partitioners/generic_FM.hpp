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

template<typename hypergraph_t>
class GenericFM {

    using index_type = typename hypergraph_t::vertex_idx;
    using workw_type = typename hypergraph_t::vertex_work_weight_type;
    using memw_type = typename hypergraph_t::vertex_mem_weight_type;
    using commw_type = typename hypergraph_t::vertex_comm_weight_type;



  protected:
    unsigned max_number_of_passes = 10;
    index_type max_nodes_in_part = 0;

    // auxiliary for RecursiveFM
    std::vector<index_type> getMaxNodesOnLevel(index_type nr_nodes, unsigned nr_parts) const;

  public:

    void ImprovePartitioning(Partitioning<hypergraph_t>& partition);

    void RecursiveFM(Partitioning<hypergraph_t>& partition);

    inline unsigned getMaxNumberOfPasses() const { return max_number_of_passes; }
    inline void setMaxNumberOfPasses(unsigned passes_) { max_number_of_passes = passes_; }
    inline index_type getMaxNodesInPart() const { return max_nodes_in_part; }
    inline void setMaxNodesInPart(index_type max_nodes_) { max_nodes_in_part = max_nodes_; }
};

template<typename hypergraph_t>
void GenericFM<hypergraph_t>::ImprovePartitioning(Partitioning<hypergraph_t>& partition)
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

    const Hypergraph<index_type, workw_type, memw_type, commw_type>& Hgraph = partition.getInstance().getHypergraph();

    index_type max_degree = 0;
    for(index_type node = 0; node < Hgraph.num_vertices(); ++node)
        max_degree = std::max(max_degree, static_cast<index_type>(Hgraph.get_incident_hyperedges(node).size()));

    if(max_nodes_in_part == 0) // if not initialized
        max_nodes_in_part = static_cast<index_type>(ceil(static_cast<double>(Hgraph.num_vertices()) * static_cast<double>(partition.getInstance().getMaxWorkWeightPerPartition())
                                         / static_cast<double>(compute_total_vertex_work_weight(Hgraph)) ));

    for(unsigned pass_idx = 0; pass_idx < max_number_of_passes; ++pass_idx)
    {
        std::vector<unsigned> node_to_new_part = partition.assignedPartitions();
        std::vector<bool> locked(Hgraph.num_vertices(), false);
        std::vector<int> gain(Hgraph.num_vertices(), 0);
        std::vector<std::vector<index_type> > nr_nodes_in_hyperedge_on_side(Hgraph.num_hyperedges(), std::vector<index_type>(2, 0));
        int cost = 0;

        index_type left_side = 0;
        for(index_type node = 0; node < Hgraph.num_vertices(); ++node)
            if(partition.assignedPartition(node) == 0)
                ++left_side;

        if(left_side > max_nodes_in_part || Hgraph.num_vertices() - left_side > max_nodes_in_part)
        {
            if(pass_idx == 0)
            {
                std::cout<<"Error: initial partitioning of FM is not balanced."<<std::endl;
                return;
            }
            else
            {
                std::cout<<"Error during FM: partitionming somehow became imbalanced."<<std::endl;
                return;
            }
        }

        // Initialize gain values
        for(index_type hyperedge = 0; hyperedge < Hgraph.num_hyperedges(); ++hyperedge)
        {
            for(index_type node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                ++nr_nodes_in_hyperedge_on_side[hyperedge][partition.assignedPartition(node)];

            if(Hgraph.get_vertices_in_hyperedge(hyperedge).size() < 2)
                continue;
            
            for(unsigned part = 0; part < 2; ++part)
            {
                if(nr_nodes_in_hyperedge_on_side[hyperedge][part] == 1)
                    for(index_type node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                        if(partition.assignedPartition(node) == part)
                            ++gain[node];

                if(nr_nodes_in_hyperedge_on_side[hyperedge][part] == 0)
                    for(index_type node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                        if(partition.assignedPartition(node) != part)
                            --gain[node];
            }
        }

        // build gain bucket array
        std::vector<int> max_gain(2, -static_cast<int>(max_degree)-1);
        std::vector<std::vector<std::vector<index_type> > > gain_bucket_array(2, std::vector<std::vector<index_type> >(2*max_degree+1));
        for(index_type node = 0; node < Hgraph.num_vertices(); ++node)
        {
            const unsigned& part = partition.assignedPartition(node);
            gain_bucket_array[part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))].push_back(node);
            max_gain[part] = std::max(max_gain[part], gain[node]);
        }

        index_type best_index = 0;
        int best_cost = 0;
        std::vector<index_type> moved_nodes;

        // the pass itself: make moves
        while(moved_nodes.size() < Hgraph.num_vertices())
        {
            // select move
            index_type to_move = std::numeric_limits<index_type>::max();
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

            if(to_move == std::numeric_limits<index_type>::max())
                break;
            
            // make move

            moved_nodes.push_back(to_move);
            cost -= gain[to_move];
            if(cost < best_cost)
            {
                best_cost = cost;
                best_index = static_cast<index_type>(moved_nodes.size()) + 1;
            }
            locked[to_move] = true;
            node_to_new_part[to_move] = 1 - node_to_new_part[to_move];

            if(chosen_part == 0)
                --left_side;
            else
                ++left_side;

            unsigned other_part = 1-chosen_part;

            // update gain values
            for(index_type hyperedge : Hgraph.get_incident_hyperedges(to_move))
            {
                if(nr_nodes_in_hyperedge_on_side[hyperedge][chosen_part] == 1)
                {
                    for(index_type node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                    {
                        if(locked[node])
                            continue;

                        std::vector<index_type>& vec = gain_bucket_array[other_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))];
                        vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
                        --gain[node];
                        gain_bucket_array[other_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))].push_back(node);
                    }
                }
                else if(nr_nodes_in_hyperedge_on_side[hyperedge][chosen_part] == 2)
                {
                    for(index_type node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                    {
                        if(node_to_new_part[node] == chosen_part && !locked[node])
                        {
                            std::vector<index_type>& vec = gain_bucket_array[chosen_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))];
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
                    for(index_type node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                    {
                        if(node_to_new_part[node] == other_part && !locked[node])
                        {
                            std::vector<index_type>& vec = gain_bucket_array[other_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))];
                            vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
                            --gain[node];
                            gain_bucket_array[other_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))].push_back(node);
                            break;
                        }
                    }
                }
                else if(nr_nodes_in_hyperedge_on_side[hyperedge][other_part] == 0)
                {
                    for(index_type node : Hgraph.get_vertices_in_hyperedge(hyperedge))
                    {
                        if(locked[node])
                            continue;

                        std::vector<index_type>& vec = gain_bucket_array[chosen_part][static_cast<unsigned>(gain[node] + static_cast<int>(max_degree))];
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

        for(index_type node_idx = 0; node_idx < best_index && node_idx < static_cast<index_type>(moved_nodes.size()); ++node_idx)
            partition.setAssignedPartition(moved_nodes[node_idx], 1U-partition.assignedPartition(moved_nodes[node_idx]));

    }
}

template<typename hypergraph_t>
void GenericFM<hypergraph_t>::RecursiveFM(Partitioning<hypergraph_t>& partition)
{
    const unsigned& nr_parts = partition.getInstance().getNumberOfPartitions();
    const index_type& nr_nodes = partition.getInstance().getHypergraph().num_vertices();

    using Hgraph = Hypergraph<index_type, workw_type, memw_type, commw_type>;

    // Note: this is just a simple recursive heuristic for the case when the partitions are a small power of 2
    if(nr_parts != 4 && nr_parts != 8 && nr_parts != 16 && nr_parts != 32)
    {
        std::cout << "Error: Recursive FM can only be used for 4, 8, 16 or 32 partitions currently." << std::endl;
        return;
    }

    for(index_type node = 0; node < nr_nodes; ++node)
        partition.setAssignedPartition(node, static_cast<unsigned>(node % 2));

    if(max_nodes_in_part == 0) // if not initialized
        max_nodes_in_part = static_cast<index_type>(ceil(static_cast<double>(nr_nodes) * static_cast<double>(partition.getInstance().getMaxWorkWeightPerPartition())
                                         / static_cast<double>(compute_total_vertex_work_weight(partition.getInstance().getHypergraph())) ));

    const std::vector<index_type> max_nodes_on_level = getMaxNodesOnLevel(nr_nodes, nr_parts);
    
    unsigned parts = 1;
    unsigned level = 0;
    std::vector<Hgraph> sub_hgraphs({partition.getInstance().getHypergraph()});
    unsigned start_index = 0;

    std::map<index_type, std::pair<unsigned, index_type> > node_to_new_hgraph_and_id;
    std::map<std::pair<unsigned, index_type>, index_type> hgraph_and_id_to_old_idx;
    for(index_type node = 0; node < nr_nodes; ++node)
    {
        node_to_new_hgraph_and_id[node] = std::make_pair(0, node);
        hgraph_and_id_to_old_idx[std::make_pair(0, node)] = node;
    }

    while(parts < nr_parts)
    {
        unsigned end_idx = static_cast<unsigned>(sub_hgraphs.size());
        for(unsigned sub_hgraph_index = start_index; sub_hgraph_index < end_idx; ++sub_hgraph_index)
        {
            const Hgraph& hgraph = sub_hgraphs[sub_hgraph_index];
            PartitioningProblem instance(hgraph, 2);
            Partitioning sub_partition(instance);
            for(index_type node = 0; node < hgraph.num_vertices(); ++node)
                sub_partition.setAssignedPartition(node, node%2);
            
            GenericFM sub_fm;
            sub_fm.setMaxNodesInPart(max_nodes_on_level[level]);
            //std::cout<<"Hgraph of size "<<hgraph.num_vertices()<<" split into two parts of at most "<<max_nodes_on_level[level]<<std::endl;
            sub_fm.ImprovePartitioning(sub_partition);

            std::vector<unsigned> current_idx(2, 0);
            std::vector<std::vector<bool> > part_indicator(2, std::vector<bool>(hgraph.num_vertices(), false));
            for(index_type node = 0; node < hgraph.num_vertices(); ++node)
            {
                const unsigned part_id = sub_partition.assignedPartition(node);
                const index_type original_id = hgraph_and_id_to_old_idx[std::make_pair(sub_hgraph_index, node)];
                node_to_new_hgraph_and_id[original_id] = std::make_pair(sub_hgraphs.size()+part_id, current_idx[part_id]);
                hgraph_and_id_to_old_idx[std::make_pair(sub_hgraphs.size()+part_id, current_idx[part_id])] = original_id;
                ++current_idx[part_id];
                part_indicator[part_id][node] = true;
            }

            for(unsigned part = 0; part < 2; ++part)
                sub_hgraphs.push_back(create_induced_hypergraph(sub_hgraphs[sub_hgraph_index], part_indicator[part]));

            ++start_index;
        }

        parts *= 2;
        ++level;
    }
    
    for(index_type node = 0; node < nr_nodes; ++node)
        partition.setAssignedPartition(node, node_to_new_hgraph_and_id[node].first - (static_cast<unsigned>(sub_hgraphs.size())-nr_parts));    
    
}

template<typename hypergraph_t>
std::vector<typename hypergraph_t::vertex_idx> GenericFM<hypergraph_t>::getMaxNodesOnLevel(typename hypergraph_t::vertex_idx nr_nodes, unsigned nr_parts) const
{
    std::vector<index_type> max_nodes_on_level;
    std::vector<index_type> limit_per_level({static_cast<index_type>(ceil(static_cast<double>(nr_nodes) / 2.0))});
    for(unsigned parts = nr_parts / 4; parts > 0; parts /= 2)
        limit_per_level.push_back(static_cast<index_type>(ceil(static_cast<double>(limit_per_level.back()) / 2.0)));

    max_nodes_on_level.push_back(max_nodes_in_part);
    for(unsigned parts = 2; parts < nr_parts; parts *= 2)
    {
        index_type next_limit = max_nodes_on_level.back()*2;
        if(next_limit > limit_per_level.back())
            --next_limit;
        
        limit_per_level.pop_back();
        max_nodes_on_level.push_back(next_limit);
    }

    std::reverse(max_nodes_on_level.begin(),max_nodes_on_level.end());
    return max_nodes_on_level;
}

} // namespace osp