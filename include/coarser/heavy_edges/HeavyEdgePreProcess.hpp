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

#include "auxiliary/datastructures/union_find.hpp"
#include "concepts/computational_dag_concept.hpp"

namespace osp {

template<typename Graph_t>
std::vector<std::vector<vertex_idx_t<Graph_t>>>
heavy_edge_preprocess(const Graph_t &graph, const double heavy_is_x_times_median,
                      const double min_percent_components_retained, const double bound_component_weight_percent) {

    static_assert(is_computational_dag_edge_desc_v<Graph_t>,
                  "HeavyEdgePreProcess can only be used with computational DAGs with edge weights.");

    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;

    // Initialising the union find structure
    union_find_universe_t<Graph_t> uf_structure;
    for (const VertexType &vert : graph.vertices()) {
        uf_structure.add_object(vert, graph.vertex_work_weight(vert));
    }

    // Making edge comunications list
    std::vector<e_commw_t<Graph_t>> edge_communications;
    edge_communications.reserve(graph.num_edges());
    for (const auto &edge : graph.edges()) {
        edge_communications.emplace_back(graph.edge_comm_weight(edge));
    }

    // Computing the median and setting it to at least one
    e_commw_t<Graph_t> median_edge_weight = 1;
    if (not edge_communications.empty()) {

        auto median_it = edge_communications.begin();
        std::advance(median_it, edge_communications.size() / 2);
        std::nth_element(edge_communications.begin(), median_it, edge_communications.end());
        median_edge_weight =
            std::max(edge_communications[edge_communications.size() / 2], static_cast<e_commw_t<Graph_t>>(1));
    }

    // Making edge list
    e_commw_t<Graph_t> minimal_edge_weight =
        static_cast<e_commw_t<Graph_t>>(heavy_is_x_times_median * median_edge_weight);
    std::vector<EdgeType> edge_list;
    edge_list.reserve(graph.num_edges());
    for (const auto &edge : graph.edges()) {
        if (graph.edge_comm_weight(edge) > minimal_edge_weight) {
            edge_list.emplace_back(edge);
        }
    }

    // Sorting edge list
    std::sort(edge_list.begin(), edge_list.end(), [graph](const EdgeType &left, const EdgeType &right) {
        return graph.edge_comm_weight(left) > graph.edge_comm_weight(right);
    });

    // Computing max component size
    v_workw_t<Graph_t> max_component_size = 0;
    for (const VertexType &vert : graph.vertices()) {
        max_component_size += graph.vertex_work_weight(vert);
    }

    max_component_size = static_cast<v_workw_t<Graph_t>>(max_component_size * bound_component_weight_percent);

    // Joining heavy edges
    for (const EdgeType &edge : edge_list) {
        if (static_cast<double>(uf_structure.get_number_of_connected_components()) - 1.0 <
            min_percent_components_retained * static_cast<double>(graph.num_vertices()))
            break;

        v_workw_t<Graph_t> weight_comp_a = uf_structure.get_weight_of_component_by_name(source(edge, graph));
        v_workw_t<Graph_t> weight_comp_b = uf_structure.get_weight_of_component_by_name(target(edge, graph));
        if (weight_comp_a + weight_comp_b > max_component_size)
            continue;

        uf_structure.join_by_name(edge.m_source, edge.m_target);
    }

    return uf_structure.get_connected_components();
};

} // namespace osp
