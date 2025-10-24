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

#include <queue>
#include <unordered_set>
#include <vector>

#include "osp/partitioning/model/hypergraph.hpp"

/**
 * @file hypergraph_utility.hpp
 * @brief Utility functions and classes for working with hypergraphs graphs.
 *
 * This file provides a collection of simple utility functions for the hypergraph class.
 */

namespace osp {


// summing up weights

template<typename hypergraph_t>
typename hypergraph_t::vertex_work_weight_type compute_total_vertex_work_weight(const hypergraph_t& hgraph)
{
    using index_type = typename hypergraph_t::vertex_idx;
    using workw_type = typename hypergraph_t::vertex_work_weight_type;

    workw_type total = 0;
    for(index_type node = 0; node < hgraph.num_vertices(); ++node)
        total += hgraph.get_vertex_work_weight(node);
    return total;
}

template<typename hypergraph_t>
typename hypergraph_t::vertex_mem_weight_type compute_total_vertex_memory_weight(const hypergraph_t& hgraph)
{
    using index_type = typename hypergraph_t::vertex_idx;
    using memw_type = typename hypergraph_t::vertex_mem_weight_type;

    memw_type total = 0;
    for(index_type node = 0; node < hgraph.num_vertices(); ++node)
        total += hgraph.get_vertex_memory_weight(node);
    return total;
}


// get induced subhypergraph

template<typename hypergraph_t>
hypergraph_t create_induced_hypergraph(const hypergraph_t& hgraph, const std::vector<bool>& include)
{
    if(include.size() != hgraph.num_vertices())
        throw std::invalid_argument("Invalid Argument while extracting induced hypergraph: input bool array has incorrect size.");

    using index_type = typename hypergraph_t::vertex_idx;


    std::vector<index_type> new_index(hgraph.num_vertices());
    unsigned current_index = 0;
    for(index_type node = 0; node < hgraph.num_vertices(); ++node)
        if(include[node])
            new_index[node] = current_index++;
    
    hypergraph_t new_hgraph(current_index, 0);
    for(index_type node = 0; node < hgraph.num_vertices(); ++node)
        if(include[node])
        {
            new_hgraph.set_vertex_work_weight(new_index[node], hgraph.get_vertex_work_weight(node));
            new_hgraph.set_vertex_memory_weight(new_index[node], hgraph.get_vertex_memory_weight(node));
        }

    for(index_type hyperedge = 0; hyperedge < hgraph.num_hyperedges(); ++hyperedge)
    {
        unsigned nr_induced_pins = 0;
        std::vector<index_type> induced_hyperedge;
        for(index_type node : hgraph.get_vertices_in_hyperedge(hyperedge))
            if(include[node])
            {
                induced_hyperedge.push_back(new_index[node]);
                ++nr_induced_pins;
            }
        
        if(nr_induced_pins >= 2)
            new_hgraph.add_hyperedge(induced_hyperedge, hgraph.get_hyperedge_weight(hyperedge));
    }
    return new_hgraph;
}


// conversion

template<typename hypergraph_t, typename Graph_t>
hypergraph_t convert_from_cdag_as_dag(const Graph_t& dag)
{
    using index_type = typename hypergraph_t::vertex_idx;
    using workw_type = typename hypergraph_t::vertex_work_weight_type;
    using memw_type = typename hypergraph_t::vertex_mem_weight_type;
    using commw_type = typename hypergraph_t::vertex_comm_weight_type;

    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, index_type>, "Index type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, workw_type>, "Work weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<v_memw_t<Graph_t>, memw_type>, "Memory weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(!has_edge_weights_v<Graph_t> || std::is_same_v<e_commw_t<Graph_t>, commw_type>, "Communication weight type mismatch, cannot convert DAG to hypergraph.");

    hypergraph_t hgraph(dag.num_vertices(), 0);
    for(const auto &node : dag.vertices())
    {
        hgraph.set_vertex_work_weight(node, dag.vertex_work_weight(node));
        hgraph.set_vertex_memory_weight(node, dag.vertex_mem_weight(node));
        for (const auto &child : dag.children(node))
            if constexpr(has_edge_weights_v<Graph_t>)
                hgraph.add_hyperedge({node, child}, dag.edge_comm_weight(edge_desc(node, child, dag).first));
            else 
                hgraph.add_hyperedge({node, child});
    }
    return hgraph;
}

template<typename hypergraph_t, typename Graph_t>
hypergraph_t convert_from_cdag_as_hyperdag(const Graph_t& dag)
{
    using index_type = typename hypergraph_t::vertex_idx;
    using workw_type = typename hypergraph_t::vertex_work_weight_type;
    using memw_type = typename hypergraph_t::vertex_mem_weight_type;
    using commw_type = typename hypergraph_t::vertex_comm_weight_type;

    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, index_type>, "Index type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, workw_type>, "Work weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<v_memw_t<Graph_t>, memw_type>, "Memory weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<v_commw_t<Graph_t>, commw_type>, "Communication weight type mismatch, cannot convert DAG to hypergraph.");

    hypergraph_t hgraph(dag.num_vertices(), 0);
    for(const auto &node : dag.vertices())
    {
        hgraph.set_vertex_work_weight(node, dag.vertex_work_weight(node));
        hgraph.set_vertex_memory_weight(node, dag.vertex_mem_weight(node));
        if(dag.out_degree(node) == 0)
            continue;
        std::vector<index_type> new_hyperedge({node});
        for (const auto &child : dag.children(node))
            new_hyperedge.push_back(child);
        hgraph.add_hyperedge(new_hyperedge, dag.vertex_comm_weight(node));
    }
    return hgraph;
}

} // namespace osp