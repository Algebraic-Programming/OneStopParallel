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

#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "directed_graph_top_sort.hpp"
#include "directed_graph_util.hpp"
#include "osp/auxiliary/Balanced_Coin_Flips.hpp"
#include "osp/concepts/directed_graph_concept.hpp"

namespace osp {

template <typename GraphT>
std::vector<EdgeDescT<GraphT>> GetContractableEdgesFromPosetIntMap(const std::vector<int> &posetIntMap, const GraphT &graph) {
    static_assert(IsDirectedGraphEdgeDescV<GraphT>, "Graph_t must satisfy the directed_graph_edge_desc concept");

    std::vector<EdgeDescT<GraphT>> output;

    for (const auto &edge : Edges(graph)) {
        VertexIdxT<GraphT> src = Source(edge, graph);
        VertexIdxT<GraphT> tgt = Target(edge, graph);

        if (posetIntMap[tgt] == posetIntMap[src] + 1) {
            output.emplace_back(edge);
        }
    }

    return output;
}

}    // namespace osp
