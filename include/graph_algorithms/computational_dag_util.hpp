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

namespace osp {

template<typename Graph_t>
v_memw_t<Graph_t> max_memory_weight(const Graph_t &graph) {

    v_memw_t<Graph_t> max_memory_weight = 0;

    for (const auto &v : graph.vertices()) {
        max_memory_weight = std::max(max_memory_weight, graph.vertex_memory_weight(v));
    }
    return max_memory_weight;
}

template<typename Graph_t>
v_memw_t<Graph_t> max_memory_weight(v_type_t<Graph_t> nodeType_, const Graph_t &graph) {

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
    return std::accumulate(begin, end, 0, [&](const auto sum, const vertex_idx_t<Graph_t> &v) {
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
    return std::accumulate(begin, end, 0, [&](const auto sum, const vertex_idx_t<Graph_t> &v) {
        return sum + graph.vertex_comm_weight(v);
    });
}

template<typename Graph_t>
v_commw_t<Graph_t> sumOfVerticesCommunicationWeights(const std::initializer_list<vertex_idx_t<Graph_t>> &vertices_,
                                                     const Graph_t &graph) {
    return sumOfVerticesCommunicationWeights(vertices_.begin(), vertices_.end(), graph);
}

template<typename EdgeIterator, typename Graph_t>
e_commw_t<Graph_t> sumOfEdgesCommunicationWeights(EdgeIterator begin, EdgeIterator end, const Graph_t &graph) {
    return std::accumulate(begin, end, 0, [&](const auto sum, const edge_desc_t<Graph_t> &e) {
        return sum + graph.edge_comm_weight(e);
    });
}

template<typename Graph_t>
e_commw_t<Graph_t> sumOfEdgesCommunicationWeights(const std::initializer_list<edge_desc_t<Graph_t>> &edges_,
                                                  const Graph_t &graph) {
    return sumOfEdgesCommunicationWeights(edges_.begin(), edges_.end(), graph);
}

} // namespace osp