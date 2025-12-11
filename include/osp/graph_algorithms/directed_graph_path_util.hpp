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
template <typename Graph_t>
bool has_path(const vertex_idx_t<Graph_t> src, const vertex_idx_t<Graph_t> dest, const Graph_t &graph) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    for (const auto &child : bfs_view(graph, src)) {
        if (child == dest) { return true; }
    }

    return false;
}

template <typename Graph_t>
std::size_t longestPath(const std::set<vertex_idx_t<Graph_t>> &vertices, const Graph_t &graph) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = vertex_idx_t<Graph_t>;

    std::queue<VertexType> bfs_queue;
    std::map<VertexType, std::size_t> distances, in_degrees, visit_counter;

    // Find source nodes
    for (const VertexType &node : vertices) {
        unsigned indeg = 0;
        for (const VertexType &parent : graph.parents(node)) {
            if (vertices.count(parent) == 1) { ++indeg; }
        }

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
            if (vertices.count(child) == 0) { continue; }

            ++visit_counter[child];
            if (visit_counter[child] == in_degrees[child]) {
                bfs_queue.push(child);
                distances[child] = distances[current] + 1;
            }
        }
    }

    return std::accumulate(vertices.cbegin(), vertices.cend(), 0u, [&](const std::size_t mx, const VertexType &node) {
        return std::max(mx, distances[node]);
    });
}

template <typename Graph_t>
std::size_t longestPath(const Graph_t &graph) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = vertex_idx_t<Graph_t>;

    std::size_t max_edgecount = 0;
    std::queue<VertexType> bfs_queue;
    std::vector<VertexType> distances(graph.num_vertices(), 0), visit_counter(graph.num_vertices(), 0);

    // Find source nodes
    for (const auto &node : source_vertices_view(graph)) { bfs_queue.push(node); }

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

template <typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> longestChain(const Graph_t &graph) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<VertexType> chain;

    if (graph.num_vertices() == 0) { return chain; }

    std::vector<unsigned> top_length(graph.num_vertices(), 0);
    unsigned running_longest_chain = 0;

    VertexType end_longest_chain = 0;

    // calculating lenght of longest path
    for (const VertexType &node : top_sort_view(graph)) {
        unsigned max_temp = 0;
        for (const auto &parent : graph.parents(node)) { max_temp = std::max(max_temp, top_length[parent]); }

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
            if (top_length[in_node] != top_length[end_longest_chain] - 1) { continue; }

            end_longest_chain = in_node;
            chain.push_back(end_longest_chain);
            break;
        }
    }

    std::reverse(chain.begin(), chain.end());
    return chain;
}

template <typename Graph_t, typename T = unsigned>
std::vector<T> get_bottom_node_distance(const Graph_t &graph) {
    static_assert(std::is_integral_v<T>, "T must be of integral type");

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    std::vector<T> bottom_distance(graph.num_vertices(), 0);

    const auto top_order = GetTopOrder(graph);
    for (std::size_t i = top_order.size() - 1; i < top_order.size(); i--) {
        T max_temp = 0;
        for (const auto &j : graph.children(top_order[i])) { max_temp = std::max(max_temp, bottom_distance[j]); }
        bottom_distance[top_order[i]] = ++max_temp;
    }
    return bottom_distance;
}

template <typename Graph_t, typename T = unsigned>
std::vector<T> get_top_node_distance(const Graph_t &graph) {
    static_assert(std::is_integral_v<T>, "T must be of integral type");

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    std::vector<T> top_distance(graph.num_vertices(), 0);

    for (const auto &vertex : bfs_top_sort_view(graph)) {
        T max_temp = 0;
        for (const auto &j : graph.parents(vertex)) { max_temp = std::max(max_temp, top_distance[j]); }
        top_distance[vertex] = ++max_temp;
    }
    return top_distance;
}

template <typename Graph_t>
std::vector<std::vector<vertex_idx_t<Graph_t>>> compute_wavefronts(const Graph_t &graph) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    std::vector<std::vector<vertex_idx_t<Graph_t>>> wavefronts;
    std::vector<vertex_idx_t<Graph_t>> parents_visited(graph.num_vertices(), 0);

    wavefronts.push_back(std::vector<vertex_idx_t<Graph_t>>());
    for (const auto &vertex : graph.vertices()) {
        if (graph.in_degree(vertex) == 0) {
            wavefronts.back().push_back(vertex);
        } else {
            parents_visited[vertex] = static_cast<vertex_idx_t<Graph_t>>(graph.in_degree(vertex));
        }
    }

    vertex_idx_t<Graph_t> counter = static_cast<vertex_idx_t<Graph_t>>(wavefronts.back().size());

    while (counter < graph.num_vertices()) {
        std::vector<vertex_idx_t<Graph_t>> next_wavefront;
        for (const auto &v_prev_wavefront : wavefronts.back()) {
            for (const auto &child : graph.children(v_prev_wavefront)) {
                parents_visited[child]--;
                if (parents_visited[child] == 0) {
                    next_wavefront.push_back(child);
                    counter++;
                }
            }
        }

        wavefronts.push_back(next_wavefront);
    }

    return wavefronts;
}

template <typename Graph_t>
std::vector<int> get_strict_poset_integer_map(unsigned const noise, double const poisson_param, const Graph_t &graph) {
    static_assert(is_directed_graph_edge_desc_v<Graph_t>, "Graph_t must satisfy the directed_graph_edge_desc concept");

    if (noise > static_cast<unsigned>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Overflow in get_strict_poset_integer_map");
    }

    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;

    std::vector<VertexType> top_order = GetTopOrder(graph);

    Repeat_Chance repeater_coin;

    std::unordered_map<EdgeType, bool> up_or_down;

    for (const auto &edge : edges(graph)) { up_or_down.emplace(edge, repeater_coin.get_flip()); }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<> poisson_gen(poisson_param + 1.0e-12);

    std::vector<unsigned> top_distance = get_top_node_distance(graph);
    std::vector<unsigned> bot_distance = get_bottom_node_distance(graph);
    std::vector<int> new_top(graph.num_vertices(), 0);
    std::vector<int> new_bot(graph.num_vertices(), 0);

    unsigned max_path = 0;
    for (const auto &vertex : graph.vertices()) { max_path = std::max(max_path, top_distance[vertex]); }

    for (const auto &source : source_vertices_view(graph)) {
        if (max_path - bot_distance[source] + 1U + 2U * noise > static_cast<unsigned>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Overflow in get_strict_poset_integer_map");
        }
        new_top[source] = randInt(static_cast<int>(max_path - bot_distance[source] + 1 + 2 * noise)) - static_cast<int>(noise);
    }

    for (const auto &sink : sink_vertices_view(graph)) {
        if (max_path - top_distance[sink] + 1U + 2U * noise > static_cast<unsigned>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Overflow in get_strict_poset_integer_map");
        }
        new_bot[sink] = randInt(static_cast<int>(max_path - top_distance[sink] + 1U + 2U * noise)) - static_cast<int>(noise);
    }

    for (const auto &vertex : top_order) {
        if (is_source(vertex, graph)) { continue; }

        int max_temp = std::numeric_limits<int>::min();

        for (const auto &edge : in_edges(vertex, graph)) {
            int temp = new_top[source(edge, graph)];
            if (up_or_down.at(edge)) {
                if (poisson_param <= 0.0) {
                    temp += 1;
                } else {
                    temp += 1 + poisson_gen(gen);
                }
            }
            max_temp = std::max(max_temp, temp);
        }
        new_top[vertex] = max_temp;
    }

    for (std::reverse_iterator iter = top_order.crbegin(); iter != top_order.crend(); ++iter) {
        if (is_sink(*iter, graph)) { continue; }

        int max_temp = std::numeric_limits<int>::min();

        for (const auto &edge : out_edges(*iter, graph)) {
            int temp = new_bot[target(edge, graph)];
            if (!up_or_down.at(edge)) { temp += 1 + poisson_gen(gen); }
            max_temp = std::max(max_temp, temp);
        }
        new_bot[*iter] = max_temp;
    }

    std::vector<int> output(graph.num_vertices());
    for (unsigned i = 0; i < graph.num_vertices(); i++) { output[i] = new_top[i] - new_bot[i]; }
    return output;
}

}    // namespace osp
