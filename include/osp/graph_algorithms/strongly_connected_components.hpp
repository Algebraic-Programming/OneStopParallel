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

#include "osp/concepts/directed_graph_concept.hpp"
#include <algorithm>
#include <limits>
#include <stack>
#include <vector>

namespace osp {

/**
 * @brief Finds the strongly connected components of a directed graph using Tarjan's algorithm.
 *
 * Tarjan's algorithm performs a single depth-first search to find all strongly connected components.
 * It has a time complexity of O(V + E), where V is the number of vertices and E is the number of edges.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the `directed_graph` concept.
 * @param graph The input directed graph.
 * @return A vector of vectors, where each inner vector contains the vertices of a strongly connected component.
 */
template <typename Graph_t>
std::vector<std::vector<vertex_idx_t<Graph_t>>> strongly_connected_components(const Graph_t &graph) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = vertex_idx_t<Graph_t>;
    const auto num_vertices = graph.num_vertices();
    if (num_vertices == 0) {
        return {};
    }

    const VertexType unvisited = std::numeric_limits<VertexType>::max();
    std::vector<VertexType> ids(num_vertices, unvisited);
    std::vector<VertexType> low(num_vertices, unvisited);
    std::vector<bool> on_stack(num_vertices, false);
    std::stack<VertexType> s;
    VertexType id_counter = 0;
    std::vector<std::vector<VertexType>> sccs;

    using ChildIterator = decltype(graph.children(std::declval<VertexType>()).begin());

    for (VertexType i = 0; i < num_vertices; ++i) {
        if (ids[i] == unvisited) {
            std::vector<std::pair<VertexType, std::pair<ChildIterator, ChildIterator>>> dfs_stack;

            dfs_stack.emplace_back(i, std::make_pair(graph.children(i).begin(), graph.children(i).end()));

            s.push(i);
            on_stack[i] = true;
            ids[i] = low[i] = id_counter++;

            while (!dfs_stack.empty()) {
                auto &[at, iter_pair] = dfs_stack.back();
                auto &child_iter = iter_pair.first;
                const auto &child_end = iter_pair.second;

                if (child_iter != child_end) {
                    VertexType to = *child_iter;
                    ++child_iter;

                    if (ids[to] == unvisited) {
                        dfs_stack.emplace_back(
                            to, std::make_pair(graph.children(to).begin(), graph.children(to).end()));
                        s.push(to);
                        on_stack[to] = true;
                        ids[to] = low[to] = id_counter++;
                    } else if (on_stack[to]) {
                        low[at] = std::min(low[at], ids[to]);
                    }
                } else {
                    if (ids[at] == low[at]) {
                        std::vector<VertexType> scc;
                        while (true) {
                            VertexType node = s.top();
                            s.pop();
                            on_stack[node] = false;
                            scc.push_back(node);
                            if (node == at)
                                break;
                        }
                        sccs.emplace_back(std::move(scc));
                    }

                    if (dfs_stack.size() > 1) {
                        auto &[parent, _] = dfs_stack[dfs_stack.size() - 2];
                        low[parent] = std::min(low[parent], low[at]);
                    }

                    dfs_stack.pop_back();
                }
            }
        }
    }

    return sccs;
}

} // namespace osp