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

#include "coarser/heavy_edges/HeavyEdgePreProcess.hpp"

std::vector<std::vector<VertexType>> heavy_edge_preprocess(const ComputationalDag &graph, const float heavy_is_x_times_median, const float min_percent_components_retained, const float bound_component_weight_percent) {
    
    // Initialising the union find structure
    Union_Find_Universe<VertexType> uf_structure;
    for (const VertexType &vert : graph.vertices()) {
        uf_structure.add_object(vert, graph.nodeWorkWeight(vert));
    }

    // Making edge comunications list
    std::vector<int> edge_communications;
    edge_communications.reserve(graph.numberOfEdges());
    for (const auto &edge : graph.edges()) {
        edge_communications.emplace_back(graph[edge].communicationWeight);
    }

    // Computing the median and setting it to at least one
    int median_edge_weight;
    if (edge_communications.size() == 0) {
        median_edge_weight = 0;
    } else {
        auto median_it = edge_communications.begin() + edge_communications.size() / 2;
        std::nth_element(edge_communications.begin(), median_it, edge_communications.end());
        median_edge_weight = edge_communications[edge_communications.size() / 2];
    }
    median_edge_weight = std::max(median_edge_weight, 1);

    // Making edge list
    float minimal_edge_weight = heavy_is_x_times_median * median_edge_weight;
    std::vector<EdgeType> edge_list;
    edge_list.reserve(graph.numberOfEdges());
    for (const auto &edge : graph.edges()) {
        if (graph[edge].communicationWeight > minimal_edge_weight) {
            edge_list.emplace_back(edge);
        }
    }

    // Sorting edge list
    std::sort(edge_list.begin(), edge_list.end(), [graph](const EdgeType &left, const EdgeType &right) { return graph[left].communicationWeight > graph[right].communicationWeight; });

    // Computing max component size
    unsigned max_component_size = 0;
    for (const VertexType &vert : graph.vertices()) {
        max_component_size += graph.nodeWorkWeight(vert);
    }
    max_component_size *= bound_component_weight_percent;

    // Joining heavy edges
    for (const EdgeType &edge : edge_list) {
        if (uf_structure.get_number_of_connected_components() - 1 < min_percent_components_retained * graph.numberOfVertices()) break;

        unsigned weight_comp_a = uf_structure.get_weight_of_component_by_name(edge.m_source);
        unsigned weight_comp_b = uf_structure.get_weight_of_component_by_name(edge.m_target);
        if ( weight_comp_a + weight_comp_b > max_component_size) continue;

        uf_structure.join_by_name(edge.m_source, edge.m_target);
    }

    return uf_structure.get_connected_components();
};