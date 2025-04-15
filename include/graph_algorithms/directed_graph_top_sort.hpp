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

#include <limits>
#include <queue>
#include <random>
#include <vector>

#include "auxiliary/misc.hpp"
#include "concepts/directed_graph_concept.hpp"
#include "directed_graph_util.hpp"

namespace osp {


enum TOP_SORT_ORDER { AS_IT_COMES, MAX_CHILDREN, RANDOM, MINIMAL_NUMBER };

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> GetTopOrder(const TOP_SORT_ORDER q_order, const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<VertexType> predecessors_count(graph.num_vertices(), 0);
    std::vector<VertexType> TopOrder;

    if (q_order == AS_IT_COMES) {
        std::queue<VertexType> next;

        // Find source nodes
        for (const VertexType &v : source_vertices_view(graph))
            next.push(v);

        // Execute BFS
        while (!next.empty()) {
            const VertexType node = next.front();
            next.pop();
            TopOrder.push_back(node);

            for (const VertexType &current : graph.children(node)) {
                ++predecessors_count[current];
                if (predecessors_count[current] == graph.in_degree(current))
                    next.push(current);
            }
        }
    }

    if (q_order == MAX_CHILDREN) {
        const auto q_cmp = [](const std::pair<VertexType, size_t> &left, const std::pair<VertexType, size_t> &right) {
            return (left.second < right.second) || ((left.second < right.second) && (left.first < right.first));
        };
        std::priority_queue<std::pair<VertexType, size_t>, std::vector<std::pair<VertexType, size_t>>, decltype(q_cmp)>
            next(q_cmp);

        // Find source nodes
        for (const VertexType &i : source_vertices_view(graph))
            next.emplace(i, graph.out_degree(i));

        // Execute BFS
        while (!next.empty()) {
            const auto [node, n_chldrn] = next.top();
            next.pop();
            TopOrder.push_back(node);

            for (const VertexType &current : graph.children(node)) {
                ++predecessors_count[current];
                if (predecessors_count[current] == graph.in_degree(current))
                    next.emplace(current, graph.out_degree(current));
            }
        }
    }

    if (q_order == RANDOM) {
        std::vector<VertexType> next;

        // Find source nodes
        for (const VertexType &i : source_vertices_view(graph))
            next.push_back(i);

        std::random_device rd;
        std::mt19937_64 eng(rd());
        std::uniform_int_distribution<unsigned long> distr(0, next.size());

        // Execute BFS
        while (!next.empty()) {
            auto node_it = next.begin();
            std::advance(node_it, distr(eng) % next.size());
            const VertexType node = *node_it;
            next.erase(node_it);
            TopOrder.push_back(node);

            for (const VertexType &current : graph.children(node)) {
                ++predecessors_count[current];
                if (predecessors_count[current] == graph.in_degree(current))
                    next.push_back(current);
            }
        }
    }

    if (q_order == MINIMAL_NUMBER) {
        std::priority_queue<VertexType, std::vector<VertexType>, std::greater<VertexType>> next;

        // Find source nodes
        for (const VertexType &i : source_vertices_view(graph))
            next.emplace(i);

        // Execute BFS
        while (!next.empty()) {
            const VertexType node = next.top();
            next.pop();
            TopOrder.push_back(node);

            for (const VertexType &current : graph.children(node)) {
                ++predecessors_count[current];
                if (predecessors_count[current] == graph.in_degree(current))
                    next.emplace(current);
            }
        }
    }

    if (TopOrder.size() != graph.num_vertices())
        throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.num_vertices() [" +
                                 std::to_string(TopOrder.size()) + " != " + std::to_string(graph.num_vertices()) + "]");

    return TopOrder;
}

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> GetFilteredTopOrder(const std::vector<bool> &valid, const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    std::vector<vertex_idx_t<Graph_t>> filteredOrder;
    for (const auto &node : GetTopOrder(AS_IT_COMES, graph))
        if (valid[node])
            filteredOrder.push_back(node);

    return filteredOrder;
}

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> top_sort_dfs(const Graph_t &dag) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    std::vector<vertex_idx_t<Graph_t>> top_order;
    top_order.reserve(dag.num_vertices());
    std::vector<bool> visited(dag.num_vertices(), false);

    std::function<void(vertex_idx_t<Graph_t>)> dfs_visit = [&](const vertex_idx_t<Graph_t> node) {
        visited[node] = true;
        for (const vertex_idx_t<Graph_t> &child : dag.children(node)) {
            if (!visited[child])
                dfs_visit(child);
        }
        top_order.emplace_back(node);
    };

    for (const vertex_idx_t<Graph_t> &i : source_vertices(dag))
        if (!visited[i])
            dfs_visit(i);

    std::reverse(top_order.begin(), top_order.end());
    return top_order;
}

// template<typename Graph_t>
// std::vector<vertex_idx_t<Graph_t>> top_sort_bfs(const Graph_t &dag) {

//     static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

//     if constexpr (has_typed_vertices_v<Graph_t>) {

//         std::vector<vertex_idx_t<Graph_t>> predecessors_count(dag.num_vertices(), 0);
//         std::vector<vertex_idx_t<Graph_t>> top_order(dag.num_vertices(), 0);
//         std::vector<std::queue<vertex_idx_t<Graph_t>>> next(dag.getNumberOfNodeTypes());

//         for (const vertex_idx_t<Graph_t> &i : dag.sourceVertices()) {
//             next[dag.nodeType(i)].push(i);
//         }

//         size_t idx = 0;
//         unsigned current_node_type = 0;

//         while (idx < dag.numberOfVertices()) {

//             while (not next[current_node_type].empty()) {
//                 const vertex_idx_t<Graph_t> node = next[current_node_type].front();
//                 next[current_node_type].pop();
//                 top_order[idx++] = node;

//                 for (const vertex_idx_t<Graph_t> &current : dag.children(node)) {
//                     ++predecessors_count[current];
//                     if (predecessors_count[current] == dag.numberOfParents(current))
//                         next[dag.nodeType(current)].push(current);
//                 }
//             }

//             current_node_type = (current_node_type + 1) % dag.getNumberOfNodeTypes();
//         }

//         return top_order;

//     } else {

//     }
// }

// template<typename Graph_t>
// std::vector<vertex_idx_t<Graph_t>> top_sort_locality(const Graph_t &dag);

// template<typename Graph_t>
// std::vector<vertex_idx_t<Graph_t>> top_sort_max_children(const Graph_t &dag);

// template<typename Graph_t>
// std::vector<vertex_idx_t<Graph_t>> top_sort_random(const Graph_t &dag);

// template<typename Graph_t>
// std::vector<vertex_idx_t<Graph_t>> top_sort_heavy_edges(const Graph_t &dag, bool sum = false);

// template<typename Graph_t, typename T>
// std::vector<vertex_idx_t<Graph_t>> top_sort_priority_node_type(const Graph_t &dag, const std::vector<T>
// &node_priority) {

//     std::vector<vertex_idx_t<Graph_t>> predecessors_count(dag.numberOfVertices(), 0);
//     std::vector<vertex_idx_t<Graph_t>> top_order(dag.numberOfVertices(), 0);

//     struct heap_node {

//         unsigned node;

//         T priority;

//         heap_node() : node(0), priority(0) {}
//         heap_node(unsigned n, unsigned p) : node(n), priority(p) {}

//         bool operator<(heap_node const &rhs) const {
//             return (priority > rhs.priority) || (priority == rhs.priority and node > rhs.node);
//         }
//     };

//     std::vector<std::vector<heap_node>> heap(dag.getNumberOfNodeTypes());

//     for (const auto &source_vertex : dag.sourceVertices()) {

//         heap[dag.nodeType(source_vertex)].emplace_back(source_vertex, node_priority[source_vertex]);
//         std::push_heap(heap[dag.nodeType(source_vertex)].begin(), heap[dag.nodeType(source_vertex)].end());
//     }

//     unsigned idx = 0;

//     unsigned current_node_type = 0;

//     while (idx < dag.numberOfVertices()) {

//         while (not heap[current_node_type].empty()) { // keep the same node type as long as possible

//             std::pop_heap(heap[current_node_type].begin(), heap[current_node_type].end());
//             const unsigned current_node = heap[current_node_type].back().node;
//             heap[current_node_type].pop_back();

//             top_order[idx++] = current_node;

//             for (const auto &child : dag.children(current_node)) {

//                 predecessors_count[child]++;
//                 if (predecessors_count[child] == dag.numberOfParents(child)) {

//                     heap[dag.nodeType(child)].emplace_back(child, node_priority[child]);
//                     std::push_heap(heap[dag.nodeType(child)].begin(), heap[dag.nodeType(child)].end());
//                 }
//             }
//         }

//         current_node_type = (current_node_type + 1) % dag.getNumberOfNodeTypes();
//     }

//     return top_order;
// };

// template<typename Graph_t, typename T>
// std::vector<vertex_idx_t<Graph_t>> top_sort_priority(const Graph_t &dag, const std::vector<T> &node_priority) {

//     std::vector<vertex_idx_t<Graph_t>> predecessors_count(dag.numberOfVertices(), 0);
//     std::vector<vertex_idx_t<Graph_t>> top_order(dag.numberOfVertices(), 0);

//     struct heap_node {

//         unsigned node;

//         T priority;

//         heap_node() : node(0), priority(0) {}
//         heap_node(unsigned n, unsigned p) : node(n), priority(p) {}

//         bool operator<(heap_node const &rhs) const {
//             return (priority > rhs.priority) || (priority == rhs.priority and node > rhs.node);
//         }
//     };

//     std::vector<heap_node> heap;

//     for (const auto &source_vertex : dag.sourceVertices()) {

//         heap.emplace_back(source_vertex, node_priority[source_vertex]);
//         std::push_heap(heap.begin(), heap.end());
//     }

//     unsigned idx = 0;

//     while (not heap.empty()) {

//         std::pop_heap(heap.begin(), heap.end());
//         const unsigned current_node = heap.back().node;
//         heap.pop_back();

//         top_order[idx++] = current_node;

//         for (const auto &child : dag.children(current_node)) {

//             predecessors_count[child]++;
//             if (predecessors_count[child] == dag.numberOfParents(child)) {

//                 heap.emplace_back(child, node_priority[child]);
//                 std::push_heap(heap.begin(), heap.end());
//             }
//         }
//     }

//     return top_order;
// };

} // namespace osp