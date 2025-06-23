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

#include "auxiliary/Balanced_Coin_Flips.hpp"
#include "concepts/directed_graph_concept.hpp"
#include "directed_graph_top_sort.hpp"
#include "directed_graph_util.hpp"

namespace osp {



template<typename Graph_t>
std::vector<edge_desc_t<Graph_t>> get_contractable_edges_from_poset_int_map(const std::vector<int> &poset_int_map, const Graph_t &graph) {
    static_assert(is_directed_graph_edge_desc_v<Graph_t>, "Graph_t must satisfy the directed_graph_edge_desc concept");

    std::vector<edge_desc_t<Graph_t>> output;

    for (const auto &edge : graph.edges()) {
        vertex_idx_t<Graph_t> src = source(edge, graph);
        vertex_idx_t<Graph_t> tgt = target(edge, graph);
        
        if (poset_int_map[tgt] == poset_int_map[src] + 1) {
            output.emplace_back(edge);
        }
    }
    
    return output;
}

} // namespace osp