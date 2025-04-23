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


#include "scheduler/PebblingILP/AuxiliaryForPartialILP/AcyclicDagDivider.hpp"
#include "dag_divider/ConnectedComponentDivider.hpp"
#include <stdexcept>


std::vector<unsigned> AcyclicDagDivider::computePartitioning(const BspInstance &instance)
{
    const unsigned N = instance.numberOfVertices();

    // split to connected components first
    ConnectedComponentDivider connected_comp;
    connected_comp.compute_connected_components(instance.getComputationalDag());

    std::vector<ComputationalDag> subDags = connected_comp.get_sub_dags();
    std::vector<std::pair<unsigned, unsigned> > node_to_subdag_and_index(N);
    std::vector<std::vector<unsigned> > original_id(subDags.size());
    for(unsigned node = 0; node < N; ++node)
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
            const ComputationalDag& dag = subDags[idx];
            if(!ignore_sources_in_size)
            {
                dag_real_size[idx] = dag.numberOfVertices();
                if(dag.numberOfVertices() > maxPartitionSize)
                {
                    dag_is_too_large[idx] = true;
                    exists_too_large = true;
                }
            }
            else
            {
                for(unsigned local_ID = 0; local_ID < dag.numberOfVertices(); ++local_ID)
                    if(instance.getComputationalDag().numberOfParents(original_id[idx][local_ID]) > 0)
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
        
        std::vector<ComputationalDag> newDagList;
        std::vector<std::vector<unsigned> > original_id_updated;

        for(unsigned idx = 0; idx < subDags.size(); ++idx)
        {
            const ComputationalDag& dag = subDags[idx];
            if(!dag_is_too_large[idx])
            {
                for(unsigned local_ID = 0; local_ID < dag.numberOfVertices(); ++local_ID)
                    node_to_subdag_and_index[original_id[idx][local_ID]].first = newDagList.size();

                original_id_updated.push_back(original_id[idx]);
                newDagList.push_back(dag);
            }
            else
            {
                std::vector<ComputationalDag> splitDags;

                std::vector<unsigned> ILP_assignment;
                //unsigned newMin = dag_real_size[idx]/3, minPartitionSize); minimum condition removed - it can cause very strict bisections
                unsigned newMin = dag_real_size[idx]/3;
                unsigned newMax =  dag_real_size[idx] - newMin;

                // mark the source nodes of the original DAG
                std::vector<bool> is_original_source(dag.numberOfVertices());
                for(unsigned local_ID = 0; local_ID < dag.numberOfVertices(); ++local_ID)
                    is_original_source[local_ID] = (instance.getComputationalDag().numberOfParents(original_id[idx][local_ID]) == 0);

                // heuristic splitting
                std::vector<unsigned> heuristic_assignment = getTopologicalSplit(dag, {newMin, newMax}, is_original_source);
                unsigned heuristicCost = getSplitCost(dag, heuristic_assignment);
                unsigned ILPCost = UINT_MAX;

                // ILP-based splitting
                AcyclicPartitioningILP partitioner;
                partitioner.setTimeLimitSeconds(120);
                partitioner.setMinAndMaxSize({newMin, newMax});
                partitioner.setIsOriginalSource(is_original_source);
                partitioner.setNumberOfParts(2); // note - if set to more than 2, ILP is MUCH more inefficient
                BspInstance partial_instance(dag, instance.getArchitecture(), instance.getNodeProcessorCompatibilityMatrix());
                auto result = partitioner.computePartitioning(partial_instance);
                if(result.first == SUCCESS || result.first == BEST_FOUND)
                {
                    ILP_assignment = result.second;
                    ILPCost = getSplitCost(dag, ILP_assignment);
                }

                std::vector<unsigned> assignment = ILPCost < heuristicCost ? ILP_assignment : heuristic_assignment;

                // split DAG according to labels
                splitDags = dag.createInducedSubgraphs(assignment);
                /*std::cout<<"SPLIT DONE: "<<dag.numberOfVertices()<<" nodes to ";
                for(auto sdag : splitDags)
                    std::cout<<sdag.numberOfVertices()<<" + ";
                std::cout<<std::endl;*/


                // update labels
                std::vector<unsigned> node_idx_in_new_subDag(dag.numberOfVertices());
                std::vector<unsigned> nr_nodes_in_new_subDag(splitDags.size(), 0);
                for(unsigned local_ID = 0; local_ID < dag.numberOfVertices(); ++local_ID)
                {
                    node_idx_in_new_subDag[local_ID] = nr_nodes_in_new_subDag[assignment[local_ID]];
                    ++nr_nodes_in_new_subDag[assignment[local_ID]];
                }
                
                for(auto next_dag : splitDags)
                    original_id_updated.emplace_back(next_dag.numberOfVertices());

                for(unsigned local_ID = 0; local_ID < dag.numberOfVertices(); ++local_ID)
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
    for(unsigned node = 0; node < N; ++node)
        final_assignment[node] = node_to_subdag_and_index[node].first;
    std::cout<<"Final cut cost of acyclic DAG divider is "<<getSplitCost(instance.getComputationalDag(), final_assignment)<<std::endl;

    return final_assignment;
}


std::vector<unsigned> AcyclicDagDivider::getTopologicalSplit(const ComputationalDag &G, std::pair<unsigned, unsigned> min_and_max, const std::vector<bool>& is_original_source) const
{
    std::vector<unsigned> node_to_part(G.numberOfVertices());

    auto top_order = G.GetTopOrder();
    std::vector<unsigned> top_order_idx(G.numberOfVertices());
    for(unsigned node = 0; node < G.numberOfVertices(); ++node)
        top_order_idx[top_order[node]] = node;

    std::vector<unsigned> last_node_idx_in_hyperedge(G.numberOfVertices());
    for(unsigned node = 0; node < G.numberOfVertices(); ++node)
    {
        last_node_idx_in_hyperedge[node] = top_order_idx[node];
        for (const auto &succ : G.children(node))
            last_node_idx_in_hyperedge[node] = std::max(last_node_idx_in_hyperedge[node], top_order_idx[succ]);
    }

    unsigned index = 0;
    unsigned current_part_id = 0;

    unsigned nodes_remaining = G.numberOfVertices();
    if(ignore_sources_in_size)
    {
        nodes_remaining = 0;
        for(unsigned node = 0; node < G.numberOfVertices(); ++node)
            if(!is_original_source[node])
                ++nodes_remaining;
    }

    while(nodes_remaining > min_and_max.second)
    {
        unsigned best_cost = UINT_MAX;
        unsigned best_end = index;

        unsigned end;
        unsigned newly_added_nodes = 0;
        for(end = index + 1; index < G.numberOfVertices() && newly_added_nodes < min_and_max.first; ++end)
            if(!ignore_sources_in_size || !is_original_source[end])
                ++newly_added_nodes;

        while(end < G.numberOfVertices() && newly_added_nodes < min_and_max.second)
        {
            unsigned extra_cost = 0;

            // check the extra cut cost of the potential endpoint
            for(unsigned top_order_pos = index; top_order_pos <= end; ++top_order_pos)
            {
                unsigned node = top_order[top_order_pos];
                if(last_node_idx_in_hyperedge[node] > end)
                    extra_cost += G.nodeCommunicationWeight(node);
                
                for (const auto &pred : G.parents(node))
                    if(last_node_idx_in_hyperedge[pred] > end)
                        extra_cost += G.nodeCommunicationWeight(pred); 
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

        for(unsigned idx = index; idx <= best_end; ++idx)
        {
            node_to_part[top_order[idx]] = current_part_id;
            if(!ignore_sources_in_size || !is_original_source[idx])
                --nodes_remaining;
        }
        index = best_end + 1;
        ++current_part_id;
    }

    // remaining nodes go into last part
    for(unsigned idx = index; idx < G.numberOfVertices(); ++idx)
        node_to_part[top_order[idx]] = current_part_id;

    return node_to_part;
}

unsigned AcyclicDagDivider::getSplitCost(const ComputationalDag &G, const std::vector<unsigned>& node_to_part)
{
    unsigned cost = 0;

    for(unsigned node = 0; node < G.numberOfVertices(); ++node)
    {
        std::set<unsigned> parts_included;
        parts_included.insert(node_to_part[node]);
        for (const auto &succ : G.children(node))
            parts_included.insert(node_to_part[succ]);
        
        cost += (parts_included.size() -1) * G.nodeCommunicationWeight(node);
    }

    return cost;
}