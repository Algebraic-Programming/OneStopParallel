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

#include <omp.h>

#include <queue>
#include <unordered_set>
#include <vector>

#include "directed_graph_edge_desc_util.hpp"
#include "osp/concepts/directed_graph_concept.hpp"

namespace osp {

template <typename GraphT>
std::unordered_set<edge_desc_t<Graph_t>> LongEdgesInTrianglesParallel(const GraphT &graph) {
    static_assert(IsDirectedGraphEdgeDescV<Graph_t>, "Graph_t must satisfy the directed_graph edge desc concept");
    static_assert(has_hashable_edge_desc_v<Graph_t>, "Graph_t must satisfy the has_hashable_edge_desc concept");

    if (graph.num_edges() < 1000) {
        return long_edges_in_triangles(graph);
    }

    std::unordered_set<edge_desc_t<Graph_t>> longEdges;
    std::vector<std::vector<edge_desc_t<Graph_t>>> deletedEdgesThread(static_cast<size_t>(omp_get_max_threads()));

#pragma omp parallel for schedule(dynamic, 4)
    for (vertex_idx_t<Graph_t> vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        // for (const auto &vertex : graph.vertices()) {

        const unsigned int proc = static_cast<unsigned>(omp_get_thread_num());

        std::unordered_set<vertex_idx_t<Graph_t>> children_set;
        for (const auto &v : graph.children(vertex)) {
            children_set.emplace(v);
        }

        for (const auto &edge : out_edges(vertex, graph)) {
            const auto &child = target(edge, graph);

            for (const auto &parent : graph.parents(child)) {
                if (children_set.find(parent) != children_set.cend()) {
                    deleted_edges_thread[proc].emplace_back(edge);
                    break;
                }
            }
        }
    }

    for (const auto &edges_thread : deleted_edges_thread) {
        for (const auto &edge : edges_thread) {
            long_edges.emplace(edge);
        }
    }

    return long_edges;
}

}    // namespace osp
