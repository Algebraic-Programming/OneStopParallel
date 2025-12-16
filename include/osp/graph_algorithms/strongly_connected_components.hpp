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

#include <algorithm>
#include <limits>
#include <stack>
#include <vector>

#include "osp/concepts/directed_graph_concept.hpp"

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
template <typename GraphT>
std::vector<std::vector<VertexIdxT<GraphT>>> StronglyConnectedComponents(const GraphT &graph) {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = VertexIdxT<GraphT>;
    const auto numVertices = graph.NumVertices();
    if (numVertices == 0) {
        return {};
    }

    const VertexType unvisited = std::numeric_limits<VertexType>::max();
    std::vector<VertexType> ids(numVertices, unvisited);
    std::vector<VertexType> low(numVertices, unvisited);
    std::vector<bool> onStack(numVertices, false);
    std::stack<VertexType> s;
    VertexType idCounter = 0;
    std::vector<std::vector<VertexType>> sccs;

    using ChildIterator = decltype(graph.Children(std::declval<VertexType>()).begin());

    for (VertexType i = 0; i < numVertices; ++i) {
        if (ids[i] == unvisited) {
            std::vector<std::pair<VertexType, std::pair<ChildIterator, ChildIterator>>> dfsStack;

            dfsStack.emplace_back(i, std::make_pair(graph.Children(i).begin(), graph.Children(i).end()));

            s.push(i);
            onStack[i] = true;
            ids[i] = low[i] = idCounter++;

            while (!dfsStack.empty()) {
                auto &[at, iterPair] = dfsStack.back();
                auto &childIter = iterPair.first;
                const auto &childEnd = iterPair.second;

                if (childIter != childEnd) {
                    VertexType to = *childIter;
                    ++childIter;

                    if (ids[to] == unvisited) {
                        dfsStack.emplace_back(to, std::make_pair(graph.Children(to).begin(), graph.Children(to).end()));
                        s.push(to);
                        onStack[to] = true;
                        ids[to] = low[to] = idCounter++;
                    } else if (onStack[to]) {
                        low[at] = std::min(low[at], ids[to]);
                    }
                } else {
                    if (ids[at] == low[at]) {
                        std::vector<VertexType> scc;
                        while (true) {
                            VertexType node = s.top();
                            s.pop();
                            onStack[node] = false;
                            scc.push_back(node);
                            if (node == at) {
                                break;
                            }
                        }
                        sccs.emplace_back(std::move(scc));
                    }

                    if (dfsStack.size() > 1) {
                        auto &[parent, _] = dfsStack[dfsStack.size() - 2];
                        low[parent] = std::min(low[parent], low[at]);
                    }

                    dfsStack.pop_back();
                }
            }
        }
    }

    return sccs;
}

}    // namespace osp
