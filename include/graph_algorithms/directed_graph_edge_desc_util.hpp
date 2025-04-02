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

#include "concepts/directed_graph_concept.hpp"

namespace osp {

template<typename Graph_t>
std::pair<edge_desc_t<Graph_t>, bool> edge_desc(const vertex_idx_t<Graph_t> &src, const vertex_idx_t<Graph_t> &dest,
                                                const Graph_t &graph) {

    static_assert(is_directed_graph_edge_desc_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    for (const auto &edge : graph.out_edges(src)) {
        if (target(edge, graph) == dest) {
            return {edge, true};
        }
    }
    return {edge_desc_t<Graph_t>(), false};
}

template<typename Graph_t>
std::unordered_set<edge_desc_t<Graph_t>> long_edges_in_triangles(const Graph_t &graph) {

    static_assert(is_directed_graph_edge_desc_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(has_hashable_edge_desc_v<Graph_t>, "Graph_t must satisfy the has_hashable_edge_desc concept");

    std::unordered_set<edge_desc_t<Graph_t>> long_edges;

    for (const auto &vertex : graph.vertices()) {

        std::unordered_set<vertex_idx_t<Graph_t>> children_set;

        for (const auto &v : graph.children(vertex)) {
            children_set.emplace(v);
        }

        for (const auto &edge : graph.out_edges(vertex)) {

            const auto &child = target(edge, graph);

            for (const auto &parent : graph.parents(child)) {

                if (children_set.find(parent) != children_set.cend()) {
                    long_edges.emplace(edge);
                    break;
                }
            }
        }
    }

    return long_edges;
}

} // namespace osp