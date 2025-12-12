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

#include "osp/concepts/directed_graph_edge_desc_concept.hpp"

namespace osp {

template <typename GraphT>
std::pair<edge_desc_t<Graph_t>, bool> EdgeDesc(const vertex_idx_t<Graph_t> &src,
                                               const vertex_idx_t<Graph_t> &dest,
                                               const GraphT &graph) {
    static_assert(is_directed_graph_edge_desc_v<Graph_t>, "Graph_t must satisfy the directed_graph edge desc concept");

    for (const auto &edge : out_edges(src, graph)) {
        if (target(edge, graph) == dest) {
            return {edge, true};
        }
    }
    return {edge_desc_t<GraphT>(), false};
}

template <typename GraphT>
std::unordered_set<edge_desc_t<Graph_t>> LongEdgesInTriangles(const GraphT &graph) {
    static_assert(is_directed_graph_edge_desc_v<Graph_t>, "Graph_t must satisfy the directed_graph edge desc concept");
    static_assert(has_hashable_edge_desc_v<Graph_t>, "Graph_t must satisfy the has_hashable_edge_desc concept");

    std::unordered_set<edge_desc_t<Graph_t>> longEdges;

    for (const auto &vertex : graph.vertices()) {
        std::unordered_set<vertex_idx_t<Graph_t>> childrenSet;

        for (const auto &v : graph.children(vertex)) {
            childrenSet.emplace(v);
        }

        for (const auto &edge : out_edges(vertex, graph)) {
            const auto &child = target(edge, graph);

            for (const auto &parent : graph.parents(child)) {
                if (childrenSet.find(parent) != children_set.cend()) {
                    longEdges.emplace(edge);
                    break;
                }
            }
        }
    }

    return long_edges;
}

}    // namespace osp
