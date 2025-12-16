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
std::pair<EdgeDescT<GraphT>, bool> EdgeDesc(const VertexIdxT<GraphT> &src, const VertexIdxT<GraphT> &dest, const GraphT &graph) {
    static_assert(isDirectedGraphEdgeDescV<GraphT>, "Graph_t must satisfy the directed_graph edge desc concept");

    for (const auto &edge : OutEdges(src, graph)) {
        if (Target(edge, graph) == dest) {
            return {edge, true};
        }
    }
    return {EdgeDescT<GraphT>(), false};
}

template <typename GraphT>
std::unordered_set<EdgeDescT<GraphT>> LongEdgesInTriangles(const GraphT &graph) {
    static_assert(isDirectedGraphEdgeDescV<GraphT>, "Graph_t must satisfy the directed_graph edge desc concept");
    static_assert(hasHashableEdgeDescV<GraphT>, "Graph_t must satisfy the HasHashableEdgeDesc concept");

    std::unordered_set<EdgeDescT<GraphT>> longEdges;

    for (const auto &vertex : graph.Vertices()) {
        std::unordered_set<VertexIdxT<GraphT>> childrenSet;

        for (const auto &v : graph.Children(vertex)) {
            childrenSet.emplace(v);
        }

        for (const auto &edge : OutEdges(vertex, graph)) {
            const auto &child = Target(edge, graph);

            for (const auto &parent : graph.Parents(child)) {
                if (childrenSet.find(parent) != childrenSet.cend()) {
                    longEdges.emplace(edge);
                    break;
                }
            }
        }
    }

    return longEdges;
}

}    // namespace osp
