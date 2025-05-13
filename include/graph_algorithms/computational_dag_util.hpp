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

#include <numeric>

#include "concepts/computational_dag_concept.hpp"
#include "directed_graph_top_sort.hpp"

namespace osp {

template<typename Graph_t>
v_memw_t<Graph_t> max_memory_weight(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(has_vertex_weights_v<Graph_t>, "Graph_t must have vertex weights");

    v_memw_t<Graph_t> max_memory_weight = 0;

    for (const auto &v : graph.vertices()) {
        max_memory_weight = std::max(max_memory_weight, graph.vertex_memory_weight(v));
    }
    return max_memory_weight;
}

template<typename Graph_t>
v_memw_t<Graph_t> max_memory_weight(const v_type_t<Graph_t> &nodeType_, const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(has_vertex_weights_v<Graph_t>, "Graph_t must have vertex weights");
    static_assert(has_typed_vertices_v<Graph_t>, "Graph_t must have typed vertices");

    v_memw_t<Graph_t> max_memory_weight = 0;

    for (const auto &node : graph.vertices()) {
        if (graph.node_type(node) == nodeType_) {
            max_memory_weight = std::max(max_memory_weight, graph.vertex_memory_weight(node));
        }
    }
    return max_memory_weight;
}

template<typename Graph_t, typename VertexIterator>
v_workw_t<Graph_t> sumOfVerticesWorkWeights(VertexIterator begin, VertexIterator end, const Graph_t &graph) {
    static_assert(has_vertex_weights_v<Graph_t>, "Graph_t must have vertex weights");

    return std::accumulate(begin, end, 0, [&](const auto sum, const vertex_idx_t<Graph_t> &v) {
        return sum + graph.vertex_work_weight(v);
    });
};

template<typename Graph_t>
v_workw_t<Graph_t> sumOfVerticesWorkWeights(const Graph_t &graph) {
    static_assert(has_vertex_weights_v<Graph_t>, "Graph_t must have vertex weights");

    return std::accumulate(graph.vertices().begin(), graph.vertices().end(), static_cast<v_workw_t<Graph_t>>(0), [&](const v_workw_t<Graph_t> sum, const vertex_idx_t<Graph_t> &v) {
        return sum + graph.vertex_work_weight(v);
    });
};

template<typename Graph_t>
v_workw_t<Graph_t> sumOfVerticesWorkWeights(const std::initializer_list<vertex_idx_t<Graph_t>> vertices_,
                                            const Graph_t &graph) {
    return sumOfVerticesWorkWeights(vertices_.begin(), vertices_.end(), graph);
};

template<typename VertexIterator, typename Graph_t>
v_commw_t<Graph_t> sumOfVerticesCommunicationWeights(VertexIterator begin, VertexIterator end, const Graph_t &graph) {
    static_assert(has_vertex_weights_v<Graph_t>, "Graph_t must have vertex weights");
    return std::accumulate(begin, end, 0, [&](const auto sum, const vertex_idx_t<Graph_t> &v) {
        return sum + graph.vertex_comm_weight(v);
    });
}

template<typename Graph_t>
v_commw_t<Graph_t> sumOfVerticesCommunicationWeights(const Graph_t &graph) {
    static_assert(has_vertex_weights_v<Graph_t>, "Graph_t must have vertex weights");

    return std::accumulate(graph.vertices().begin(), graph.vertices().end(), static_cast<v_commw_t<Graph_t>>(0), [&](const v_commw_t<Graph_t> sum, const vertex_idx_t<Graph_t> &v) {
        return sum + graph.vertex_comm_weight(v);
    });
};

template<typename Graph_t>
v_commw_t<Graph_t> sumOfVerticesCommunicationWeights(const std::initializer_list<vertex_idx_t<Graph_t>> &vertices_,
                                                     const Graph_t &graph) {
    return sumOfVerticesCommunicationWeights(vertices_.begin(), vertices_.end(), graph);
}

template<typename EdgeIterator, typename Graph_t>
e_commw_t<Graph_t> sumOfEdgesCommunicationWeights(EdgeIterator begin, EdgeIterator end, const Graph_t &graph) {

    static_assert(has_edge_weights_v<Graph_t>, "Graph_t must have edge weights");
    return std::accumulate(
        begin, end, 0, [&](const auto sum, const edge_desc_t<Graph_t> &e) { return sum + graph.edge_comm_weight(e); });
}

template<typename Graph_t>
e_commw_t<Graph_t> sumOfEdgesCommunicationWeights(const std::initializer_list<edge_desc_t<Graph_t>> &edges_,
                                                  const Graph_t &graph) {
    return sumOfEdgesCommunicationWeights(edges_.begin(), edges_.end(), graph);
}

template<typename Graph_t>
v_workw_t<Graph_t> critical_path_weight(const Graph_t &graph) {

    static_assert(is_directed_graph_edge_desc_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(has_vertex_weights_v<Graph_t>, "Graph_t must have vertex weights");

    if (graph.num_vertices() == 0) {
        return 0;
    }

    std::vector<v_workw_t<Graph_t>> top_length(graph.num_vertices(), 0);
    v_workw_t<Graph_t> critical_path_weight = 0;

    // calculating lenght of longest path
    for (const auto &node : GetTopOrder(AS_IT_COMES, graph)) {

        v_workw_t<Graph_t> max_temp = 0;
        for (const auto &parent : graph.parents(node)) {
            max_temp = std::max(max_temp, top_length[parent]);
        }

        top_length[node] = max_temp + graph.vertex_work_weight(node);

        if (top_length[node] > critical_path_weight) {

            critical_path_weight = top_length[node];
        }
    }

    return critical_path_weight;
}

} // namespace osp
