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
#include <unordered_set>
#include <vector>

#include "concepts/directed_graph_concept.hpp"
#include "directed_graph_top_sort.hpp"

namespace osp {

template<typename Graph_t>
bool has_path(const vertex_idx_t<Graph_t> src, const vertex_idx_t<Graph_t> dest, const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    std::unordered_set<vertex_idx_t<Graph_t>> visited;
    visited.emplace(src);

    std::queue<vertex_idx_t<Graph_t>> next;
    next.push(src);

    while (!next.empty()) {
        vertex_idx_t<Graph_t> v = next.front();
        next.pop();

        for (const vertex_idx_t<Graph_t> &child : graph.children(v)) {

            if (child == dest) {
                return true;
            }

            if (visited.find(child) == visited.end()) {
                visited.emplace(child);
                next.push(child);
            }
        }
    }

    return false;
}

template<typename Graph_t>
std::size_t longestPath(const std::set<vertex_idx_t<Graph_t>> &vertices, const Graph_t &graph) {

    using VertexType = vertex_idx_t<Graph_t>;

    std::queue<VertexType> bfs_queue;
    std::map<VertexType, std::size_t> distances, in_degrees, visit_counter;

    // Find source nodes
    for (const VertexType &node : vertices) {
        unsigned indeg = 0;
        for (const VertexType &parent : graph.parents(node))
            if (vertices.count(parent) == 1)
                ++indeg;

        if (indeg == 0) {
            bfs_queue.push(node);
            distances[node] = 0;
        }
        in_degrees[node] = indeg;
        visit_counter[node] = 0;
    }

    // Execute BFS
    while (!bfs_queue.empty()) {
        const VertexType current = bfs_queue.front();
        bfs_queue.pop();

        for (const VertexType &child : graph.children(current)) {
            if (vertices.count(child) == 0)
                continue;

            ++visit_counter[child];
            if (visit_counter[child] == in_degrees[child]) {
                bfs_queue.push(child);
                distances[child] = distances[current] + 1;
            }
        }
    }

    return std::accumulate(vertices.cbegin(), vertices.cend(), 0,
                           [&](const size_t mx, const VertexType &node) { return std::max(mx, distances[node]); });
}

template<typename Graph_t>
std::size_t longestPath(const Graph_t &graph) {

    using VertexType = vertex_idx_t<Graph_t>;

    std::size_t max_edgecount = 0;
    std::queue<VertexType> bfs_queue;
    std::vector<VertexType> distances(graph.num_vertices(), 0), visit_counter(graph.num_vertices(), 0);

    // Find source nodes
    for (const auto &node : source_vertex_view(graph)) {
        bfs_queue.push(node);
    }

    // Execute BFS
    while (!bfs_queue.empty()) {
        const VertexType current = bfs_queue.front();
        bfs_queue.pop();

        for (const VertexType &child : graph.children(current)) {

            ++visit_counter[child];
            if (visit_counter[child] == graph.in_degree(child)) {
                bfs_queue.push(child);
                distances[child] = distances[current] + 1;
                max_edgecount = std::max(max_edgecount, distances[child]);
            }
        }
    }

    return max_edgecount;
}

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> longestChain(const Graph_t &graph) {
    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<VertexType> chain;

    if (graph.num_vertices() == 0) {
        return chain;
    }

    std::vector<unsigned> top_length(graph.num_vertices(), 0);
    unsigned running_longest_chain = 0;

    VertexType end_longest_chain = 0;

    // calculating lenght of longest path
    for (const VertexType &node : GetTopOrder(AS_IT_COMES, graph)) {

        unsigned max_temp = 0;
        for (const auto &parent : graph.parents(node)) {
            max_temp = std::max(max_temp, top_length[parent]);
        }

        top_length[node] = max_temp + 1;
        if (top_length[node] > running_longest_chain) {
            end_longest_chain = node;
            running_longest_chain = top_length[node];
        }
    }

    // reconstructing longest path
    chain.push_back(end_longest_chain);
    while (graph.in_degree(end_longest_chain) != 0) {

        for (const VertexType &in_node : graph.parents(end_longest_chain)) {
            if (top_length[in_node] != top_length[end_longest_chain] - 1) {
                continue;
            }

            end_longest_chain = in_node;
            chain.push_back(end_longest_chain);
            break;
        }
    }

    std::reverse(chain.begin(), chain.end());
    return chain;
}

} // namespace osp