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
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "directed_graph_top_sort.hpp"
#include "directed_graph_util.hpp"
#include "osp/auxiliary/Balanced_Coin_Flips.hpp"
#include "osp/concepts/directed_graph_edge_desc_concept.hpp"

namespace osp {

/**
 * @brief Checks if a path exists between two vertices in a directed graph.
 *
 * This function performs a Breadth-First Search (BFS) starting from the `src`
 * vertex to determine if the `dest` vertex is reachable.
 *
 * @tparam Graph_t The type of the graph.
 * @param src The source vertex.
 * @param dest The destination vertex.
 * @param graph The graph to search in.
 * @return true if a path exists from src to dest, false otherwise.
 */
template <typename GraphT>
bool HasPath(const VertexIdxT<GraphT> src, const VertexIdxT<GraphT> dest, const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    for (const auto &child : bfs_view(graph, src)) {
        if (child == dest) {
            return true;
        }
    }

    return false;
}

template <typename GraphT>
std::size_t LongestPath(const std::set<VertexIdxT<GraphT>> &vertices, const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = VertexIdxT<GraphT>;

    std::queue<VertexType> bfsQueue;
    std::map<VertexType, std::size_t> distances, inDegrees, visitCounter;

    // Find source nodes
    for (const VertexType &node : vertices) {
        unsigned indeg = 0;
        for (const VertexType &parent : graph.Parents(node)) {
            if (vertices.count(parent) == 1) {
                ++indeg;
            }
        }

        if (indeg == 0) {
            bfsQueue.push(node);
            distances[node] = 0;
        }
        inDegrees[node] = indeg;
        visitCounter[node] = 0;
    }

    // Execute BFS
    while (!bfsQueue.empty()) {
        const VertexType current = bfsQueue.front();
        bfsQueue.pop();

        for (const VertexType &child : graph.Children(current)) {
            if (vertices.count(child) == 0) {
                continue;
            }

            ++visitCounter[child];
            if (visitCounter[child] == inDegrees[child]) {
                bfsQueue.push(child);
                distances[child] = distances[current] + 1;
            }
        }
    }

    return std::accumulate(vertices.cbegin(), vertices.cend(), 0u, [&](const std::size_t mx, const VertexType &node) {
        return std::max(mx, distances[node]);
    });
}

template <typename GraphT>
std::size_t LongestPath(const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = VertexIdxT<GraphT>;

    std::size_t maxEdgecount = 0;
    std::queue<VertexType> bfsQueue;
    std::vector<VertexType> distances(graph.NumVertices(), 0), visitCounter(graph.NumVertices(), 0);

    // Find source nodes
    for (const auto &node : source_vertices_view(graph)) {
        bfsQueue.push(node);
    }

    // Execute BFS
    while (!bfsQueue.empty()) {
        const VertexType current = bfsQueue.front();
        bfsQueue.pop();

        for (const VertexType &child : graph.Children(current)) {
            ++visitCounter[child];
            if (visitCounter[child] == graph.InDegree(child)) {
                bfsQueue.push(child);
                distances[child] = distances[current] + 1;
                maxEdgecount = std::max(maxEdgecount, distances[child]);
            }
        }
    }

    return maxEdgecount;
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> LongestChain(const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = VertexIdxT<GraphT>;

    std::vector<VertexType> chain;

    if (graph.NumVertices() == 0) {
        return chain;
    }

    std::vector<unsigned> topLength(graph.NumVertices(), 0);
    unsigned runningLongestChain = 0;

    VertexType endLongestChain = 0;

    // calculating lenght of longest path
    for (const VertexType &node : top_sort_view(graph)) {
        unsigned maxTemp = 0;
        for (const auto &parent : graph.Parents(node)) {
            maxTemp = std::max(maxTemp, topLength[parent]);
        }

        topLength[node] = maxTemp + 1;
        if (topLength[node] > runningLongestChain) {
            endLongestChain = node;
            runningLongestChain = topLength[node];
        }
    }

    // reconstructing longest path
    chain.push_back(endLongestChain);
    while (graph.InDegree(endLongestChain) != 0) {
        for (const VertexType &inNode : graph.Parents(endLongestChain)) {
            if (topLength[inNode] != topLength[endLongestChain] - 1) {
                continue;
            }

            endLongestChain = inNode;
            chain.push_back(endLongestChain);
            break;
        }
    }

    std::reverse(chain.begin(), chain.end());
    return chain;
}

template <typename GraphT, typename T = unsigned>
std::vector<T> GetBottomNodeDistance(const GraphT &graph) {
    static_assert(std::is_integral_v<T>, "T must be of integral type");

    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    std::vector<T> bottomDistance(graph.NumVertices(), 0);

    const auto topOrder = GetTopOrder(graph);
    for (std::size_t i = topOrder.size() - 1; i < topOrder.size(); i--) {
        T maxTemp = 0;
        for (const auto &j : graph.Children(topOrder[i])) {
            maxTemp = std::max(maxTemp, bottomDistance[j]);
        }
        bottomDistance[topOrder[i]] = ++maxTemp;
    }
    return bottomDistance;
}

template <typename GraphT, typename T = unsigned>
std::vector<T> GetTopNodeDistance(const GraphT &graph) {
    static_assert(std::is_integral_v<T>, "T must be of integral type");

    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    std::vector<T> topDistance(graph.NumVertices(), 0);

    for (const auto &vertex : bfs_top_sort_view(graph)) {
        T maxTemp = 0;
        for (const auto &j : graph.Parents(vertex)) {
            maxTemp = std::max(maxTemp, topDistance[j]);
        }
        topDistance[vertex] = ++maxTemp;
    }
    return topDistance;
}

template <typename GraphT>
std::vector<std::vector<VertexIdxT<GraphT>>> ComputeWavefronts(const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    std::vector<std::vector<VertexIdxT<GraphT>>> wavefronts;
    std::vector<VertexIdxT<GraphT>> parentsVisited(graph.NumVertices(), 0);

    wavefronts.push_back(std::vector<VertexIdxT<GraphT>>());
    for (const auto &vertex : graph.vertices()) {
        if (graph.InDegree(vertex) == 0) {
            wavefronts.back().push_back(vertex);
        } else {
            parentsVisited[vertex] = static_cast<VertexIdxT<GraphT>>(graph.InDegree(vertex));
        }
    }

    VertexIdxT<GraphT> counter = static_cast<VertexIdxT<GraphT>>(wavefronts.back().size());

    while (counter < graph.NumVertices()) {
        std::vector<VertexIdxT<GraphT>> nextWavefront;
        for (const auto &vPrevWavefront : wavefronts.back()) {
            for (const auto &child : graph.Children(vPrevWavefront)) {
                parentsVisited[child]--;
                if (parentsVisited[child] == 0) {
                    nextWavefront.push_back(child);
                    counter++;
                }
            }
        }

        wavefronts.push_back(nextWavefront);
    }

    return wavefronts;
}

template <typename GraphT>
std::vector<int> GetStrictPosetIntegerMap(unsigned const noise, double const poissonParam, const GraphT &graph) {
    static_assert(IsDirectedGraphEdgeDescV<GraphT>, "Graph_t must satisfy the directed_graph_edge_desc concept");

    if (noise > static_cast<unsigned>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Overflow in get_strict_poset_integer_map");
    }

    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    std::vector<VertexType> topOrder = GetTopOrder(graph);

    RepeatChance repeaterCoin;

    std::unordered_map<EdgeType, bool> upOrDown;

    for (const auto &edge : Edges(graph)) {
        upOrDown.emplace(edge, repeaterCoin.GetFlip());
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<> poissonGen(poissonParam + 1.0e-12);

    std::vector<unsigned> topDistance = GetTopNodeDistance(graph);
    std::vector<unsigned> botDistance = GetBottomNodeDistance(graph);
    std::vector<int> newTop(graph.NumVertices(), 0);
    std::vector<int> newBot(graph.NumVertices(), 0);

    unsigned maxPath = 0;
    for (const auto &vertex : graph.Vertices()) {
        maxPath = std::max(maxPath, topDistance[vertex]);
    }

    for (const auto &source : SourceVertices(graph)) {
        if (maxPath - botDistance[source] + 1U + 2U * noise > static_cast<unsigned>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Overflow in get_strict_poset_integer_map");
        }
        newTop[source] = RandInt(static_cast<int>(maxPath - botDistance[source] + 1 + 2 * noise)) - static_cast<int>(noise);
    }

    for (const auto &sink : SinkVertices(graph)) {
        if (maxPath - topDistance[sink] + 1U + 2U * noise > static_cast<unsigned>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Overflow in get_strict_poset_integer_map");
        }
        newBot[sink] = RandInt(static_cast<int>(maxPath - topDistance[sink] + 1U + 2U * noise)) - static_cast<int>(noise);
    }

    for (const auto &vertex : topOrder) {
        if (IsSource(vertex, graph)) {
            continue;
        }

        int maxTemp = std::numeric_limits<int>::min();

        for (const auto &edge : InEdges(vertex, graph)) {
            int temp = newTop[Source(edge, graph)];
            if (upOrDown.at(edge)) {
                if (poissonParam <= 0.0) {
                    temp += 1;
                } else {
                    temp += 1 + poissonGen(gen);
                }
            }
            maxTemp = std::max(maxTemp, temp);
        }
        newTop[vertex] = maxTemp;
    }

    for (std::reverse_iterator iter = topOrder.crbegin(); iter != topOrder.crend(); ++iter) {
        if (IsSink(*iter, graph)) {
            continue;
        }

        int maxTemp = std::numeric_limits<int>::min();

        for (const auto &edge : OutEdges(*iter, graph)) {
            int temp = newBot[Traget(edge, graph)];
            if (!upOrDown.at(edge)) {
                temp += 1 + poissonGen(gen);
            }
            maxTemp = std::max(maxTemp, temp);
        }
        newBot[*iter] = maxTemp;
    }

    std::vector<int> output(graph.NumVertices());
    for (unsigned i = 0; i < graph.NumVertices(); i++) {
        output[i] = newTop[i] - newBot[i];
    }
    return output;
}

}    // namespace osp
