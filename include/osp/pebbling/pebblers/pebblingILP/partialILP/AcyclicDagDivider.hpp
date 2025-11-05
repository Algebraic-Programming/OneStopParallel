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

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/dag_divider/ConnectedComponentDivider.hpp"
#include "osp/pebbling/pebblers/pebblingILP/partialILP/AcyclicPartitioningILP.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

namespace osp{

template<typename Graph_t>
class AcyclicDagDivider {

    static_assert(is_computational_dag_v<Graph_t>, "PebblingSchedule can only be used with computational DAGs."); 

  protected:
    using vertex_idx = vertex_idx_t<Graph_t>;

    std::vector<unsigned> node_to_part;

    unsigned minPartitionSize = 40, maxPartitionSize = 80;
    bool ignore_sources_in_size = true;

    std::vector<unsigned> getTopologicalSplit(const Graph_t &G, std::pair<unsigned, unsigned> min_and_max, const std::vector<bool>& is_original_source) const;

    v_commw_t<Graph_t> static getSplitCost(const Graph_t &G, const std::vector<unsigned>& node_to_part);

  public:
    AcyclicDagDivider() {}

    virtual ~AcyclicDagDivider() = default;

    std::vector<unsigned> computePartitioning(const BspInstance<Graph_t> &instance);

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> getMinAndMaxSize() const { return std::make_pair(minPartitionSize, maxPartitionSize); }
    inline void setMinAndMaxSize(const std::pair<unsigned, unsigned> min_and_max) {minPartitionSize = min_and_max.first; maxPartitionSize = min_and_max.second; }
    inline void setIgnoreSources(const bool ignore_) {ignore_sources_in_size = ignore_; }
};

template<typename Graph_t>
std::vector<unsigned> AcyclicDagDivider<Graph_t>::computePartitioning(const BspInstance<Graph_t> &instance)
{
    const unsigned N = static_cast<unsigned>(instance.numberOfVertices());

    // split to connected components first
    ConnectedComponentDivider<Graph_t, Graph_t> connected_comp;
    connected_comp.divide(instance.getComputationalDag());

    std::vector<Graph_t> subDags = connected_comp.get_sub_dags();
    std::vector<std::pair<unsigned, vertex_idx> > node_to_subdag_and_index(N);
    std::vector<std::vector<vertex_idx> > original_id(subDags.size());
    for(vertex_idx node = 0; node < N; ++node)
    {
        node_to_subdag_and_index[node] = {connected_comp.get_component()[node], connected_comp.get_vertex_map()[node]};
        original_id[connected_comp.get_component()[node]].push_back(node);
    }
    
    // TODO extend with splits at directed articulation points in future?

    // split components further with ILPs or heuristics
    while(true)
    {
        bool exists_too_large = false;
        std::vector<bool> dag_is_too_large(subDags.size(), false);
        std::vector<unsigned> dag_real_size(subDags.size(), 0);

        for(unsigned idx = 0; idx < subDags.size(); ++idx)
        {
            const Graph_t& dag = subDags[idx];
            if(!ignore_sources_in_size)
            {
                dag_real_size[idx] = static_cast<unsigned>(dag.num_vertices());
                if(dag.num_vertices() > maxPartitionSize)
                {
                    dag_is_too_large[idx] = true;
                    exists_too_large = true;
                }
            }
            else
            {
                for(vertex_idx local_ID = 0; local_ID < dag.num_vertices(); ++local_ID)
                    if(instance.getComputationalDag().in_degree(original_id[idx][local_ID]) > 0)
                        ++dag_real_size[idx];        
            }
            if(dag_real_size[idx] > maxPartitionSize)
            {
                dag_is_too_large[idx] = true;
                exists_too_large = true;
            }

        }
        
        if(!exists_too_large)
            break;
        
        std::vector<Graph_t > newDagList;
        std::vector<std::vector<vertex_idx> > original_id_updated;

        for(unsigned idx = 0; idx < subDags.size(); ++idx)
        {
            const Graph_t& dag = subDags[idx];
            if(!dag_is_too_large[idx])
            {
                for(vertex_idx local_ID = 0; local_ID < dag.num_vertices(); ++local_ID)
                    node_to_subdag_and_index[original_id[idx][local_ID]].first = static_cast<unsigned>(newDagList.size());

                original_id_updated.push_back(original_id[idx]);
                newDagList.push_back(dag);
            }
            else
            {
                std::vector<unsigned> ILP_assignment;
                //unsigned newMin = dag_real_size[idx]/3, minPartitionSize); minimum condition removed - it can cause very strict bisections
                unsigned newMin = dag_real_size[idx]/3;
                unsigned newMax =  dag_real_size[idx] - newMin;

                // mark the source nodes of the original DAG
                std::vector<bool> is_original_source(dag.num_vertices());
                for(vertex_idx local_ID = 0; local_ID < dag.num_vertices(); ++local_ID)
                    is_original_source[local_ID] = (instance.getComputationalDag().in_degree(original_id[idx][local_ID]) == 0);

                // heuristic splitting
                std::vector<unsigned> heuristic_assignment = getTopologicalSplit(dag, {newMin, newMax}, is_original_source);
                unsigned heuristicCost = getSplitCost(dag, heuristic_assignment);
                unsigned ILPCost = UINT_MAX;

                // ILP-based splitting
                AcyclicPartitioningILP<Graph_t> partitioner;
                partitioner.setTimeLimitSeconds(120);
                partitioner.setMinAndMaxSize({newMin, newMax});
                partitioner.setIsOriginalSource(is_original_source);
                partitioner.setNumberOfParts(2); // note - if set to more than 2, ILP is MUCH more inefficient
                BspInstance partial_instance(dag, instance.getArchitecture(), instance.getNodeProcessorCompatibilityMatrix());
                RETURN_STATUS status = partitioner.computePartitioning(partial_instance, ILP_assignment);
                if(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND)
                    ILPCost = getSplitCost(dag, ILP_assignment);

                std::vector<unsigned> assignment = ILPCost < heuristicCost ? ILP_assignment : heuristic_assignment;

                // split DAG according to labels
                std::vector<Graph_t> splitDags = create_induced_subgraphs<Graph_t, Graph_t>(dag, assignment);
                /*std::cout<<"SPLIT DONE: "<<dag.numberOfVertices()<<" nodes to ";
                for(auto sdag : splitDags)
                    std::cout<<sdag.numberOfVertices()<<" + ";
                std::cout<<std::endl;*/


                // update labels
                std::vector<vertex_idx> node_idx_in_new_subDag(dag.num_vertices());
                std::vector<unsigned> nr_nodes_in_new_subDag(splitDags.size(), 0);
                for(vertex_idx local_ID = 0; local_ID < dag.num_vertices(); ++local_ID)
                {
                    node_idx_in_new_subDag[local_ID] = nr_nodes_in_new_subDag[assignment[local_ID]];
                    ++nr_nodes_in_new_subDag[assignment[local_ID]];
                }
                
                for(auto next_dag : splitDags)
                    original_id_updated.emplace_back(next_dag.num_vertices());

                for(vertex_idx local_ID = 0; local_ID < dag.num_vertices(); ++local_ID)
                {
                    node_to_subdag_and_index[original_id[idx][local_ID]] = {newDagList.size() + assignment[local_ID], node_idx_in_new_subDag[local_ID]};
                    original_id_updated[newDagList.size() + assignment[local_ID]][node_idx_in_new_subDag[local_ID]] = original_id[idx][local_ID];
                }
                for(auto next_dag : splitDags)
                    newDagList.push_back(next_dag);
            }
        }

        subDags = newDagList;
        original_id = original_id_updated;
    }

    // output final cost
    std::vector<unsigned> final_assignment(N);
    for(vertex_idx node = 0; node < N; ++node)
        final_assignment[node] = node_to_subdag_and_index[node].first;
    std::cout<<"Final cut cost of acyclic DAG divider is "<<getSplitCost(instance.getComputationalDag(), final_assignment)<<std::endl;

    return final_assignment;
}

template<typename Graph_t>
std::vector<unsigned> AcyclicDagDivider<Graph_t>::getTopologicalSplit(const Graph_t &G, std::pair<unsigned, unsigned> min_and_max, const std::vector<bool>& is_original_source) const
{
    std::vector<unsigned> node_to_part(G.num_vertices());

    std::vector<vertex_idx> top_order = GetTopOrder(G);
    std::vector<unsigned> top_order_idx(G.num_vertices());
    for(unsigned idx = 0; idx < G.num_vertices(); ++idx)
        top_order_idx[top_order[idx]] = idx;

    std::vector<unsigned> last_node_idx_in_hyperedge(G.num_vertices());
    for(unsigned node = 0; node < G.num_vertices(); ++node)
    {
        last_node_idx_in_hyperedge[node] = top_order_idx[node];
        for (const auto &succ : G.children(node))
            last_node_idx_in_hyperedge[node] = std::max(last_node_idx_in_hyperedge[node], top_order_idx[succ]);
    }

    unsigned index = 0;
    unsigned current_part_id = 0;

    unsigned nodes_remaining = static_cast<unsigned>(G.num_vertices());
    if(ignore_sources_in_size)
    {
        nodes_remaining = 0;
        for(unsigned node = 0; node < G.num_vertices(); ++node)
            if(!is_original_source[node])
                ++nodes_remaining;
    }

    while(nodes_remaining > min_and_max.second)
    {
        unsigned best_cost = UINT_MAX;
        unsigned best_end = index;

        unsigned end;
        unsigned newly_added_nodes = 0;
        for(end = index + 1; index < G.num_vertices() && newly_added_nodes < min_and_max.first; ++end)
            if(!ignore_sources_in_size || !is_original_source[end])
                ++newly_added_nodes;

        while(end < G.num_vertices() && newly_added_nodes < min_and_max.second)
        {
            unsigned extra_cost = 0;

            // check the extra cut cost of the potential endpoint
            for(unsigned top_order_pos = index; top_order_pos <= end; ++top_order_pos)
            {
                vertex_idx node = top_order[top_order_pos];
                if(last_node_idx_in_hyperedge[node] > end)
                    extra_cost += G.vertex_comm_weight(node);
                
                for (const auto &pred : G.parents(node))
                    if(last_node_idx_in_hyperedge[pred] > end)
                        extra_cost += G.vertex_comm_weight(pred); 
            }

            if(extra_cost < best_cost)
            {
                best_cost = extra_cost;
                best_end = end;
            }

            ++end;
            if(!ignore_sources_in_size || !is_original_source[end])
                ++newly_added_nodes;
        }

        for(vertex_idx idx = index; idx <= best_end; ++idx)
        {
            node_to_part[top_order[idx]] = current_part_id;
            if(!ignore_sources_in_size || !is_original_source[idx])
                --nodes_remaining;
        }
        index = best_end + 1;
        ++current_part_id;
    }

    // remaining nodes go into last part
    for(vertex_idx idx = index; idx < G.num_vertices(); ++idx)
        node_to_part[top_order[idx]] = current_part_id;

    return node_to_part;
}

template<typename Graph_t>
v_commw_t<Graph_t> AcyclicDagDivider<Graph_t>::getSplitCost(const Graph_t &G, const std::vector<unsigned>& node_to_part)
{
    v_commw_t<Graph_t> cost = 0;

    for(vertex_idx node = 0; node < G.num_vertices(); ++node)
    {
        std::set<unsigned> parts_included;
        parts_included.insert(node_to_part[node]);
        for (const auto &succ : G.children(node))
            parts_included.insert(node_to_part[succ]);
        
        cost += static_cast<v_commw_t<Graph_t>>(parts_included.size() -1) * G.vertex_comm_weight(node);
    }

    return cost;
}

}