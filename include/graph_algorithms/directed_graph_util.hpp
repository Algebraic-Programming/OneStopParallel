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

#include "concepts/directed_graph_concept.hpp"

/**
 * @file directed_graph_util.hpp
 * @brief Utility functions and classes for working with directed graphs.
 *
 * This file provides a collection of utility functions, iterators, and views
 * for performing operations on directed graphs. These utilities include
 * functions for checking graph properties, retrieving specific vertices,
 * and traversing the graph using BFS and DFS.
 */

namespace osp {

/**
 * @brief Checks if there is an edge between two vertices in the graph.
 *
 * @tparam Graph_t The type of the graph.
 * @param src The source vertex.
 * @param dest The destination vertex.
 * @param graph The graph to check.
 * @return true if there is an edge from src to dest, false otherwise.
 */
template<typename Graph_t>
bool edge(const vertex_idx_t<Graph_t> &src, const vertex_idx_t<Graph_t> &dest, const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    for (const auto &child : graph.children(src)) {
        if (child == dest) {
            return true;
        }
    }
    return false;
}



/**
 * @brief Checks if a vertex is a sink (no outgoing edges).
 *
 * @tparam Graph_t The type of the graph.
 * @param v The vertex to check.
 * @param graph The graph to check.
 * @return true if the vertex is a sink, false otherwise.
 */
template<typename Graph_t>
bool is_sink(const vertex_idx_t<Graph_t> &v, const Graph_t &graph) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    return graph.out_degree(v) == 0u;
}

/**
 * @brief Checks if a vertex is a source (no incoming edges).
 *
 * @tparam Graph_t The type of the graph.
 * @param v The vertex to check.
 * @param graph The graph to check.
 * @return true if the vertex is a source, false otherwise.
 */
template<typename Graph_t>
bool is_source(const vertex_idx_t<Graph_t> &v, const Graph_t &graph) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    return graph.in_degree(v) == 0u;
}

/**
 * @brief Helper struct for iterating over vertices with a condition.
 *
 * This struct provides an iterator that filters vertices based on a given condition.
 * It is used to create views for source and sink vertices in a directed graph.
 *
 */
template<typename cond_eval, typename Graph_t, typename iterator_t>
struct vertex_cond_iterator {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    iterator_t current_vertex;
    cond_eval cond;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = vertex_idx_t<Graph_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type *;
    using reference = const value_type &;

    vertex_cond_iterator(const Graph_t &graph_, const iterator_t &start) : graph(graph_), current_vertex(start) {

        while (current_vertex != graph.vertices().end()) {
            if (cond.eval(graph, *current_vertex)) {
                break;
            }
            current_vertex++;
        }
    }

    value_type operator*() const { return current_vertex.operator*(); }

    // Prefix increment
    vertex_cond_iterator &operator++() {
        current_vertex++;

        while (current_vertex != graph.vertices().end()) {
            if (cond.eval(graph, *current_vertex)) {
                break;
            }
            current_vertex++;
        }

        return *this;
    }

    // Postfix increment
    vertex_cond_iterator operator++(int) {
        vertex_cond_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(const vertex_cond_iterator &one, const vertex_cond_iterator &other) {
        return one.current_vertex == other.current_vertex;
    };
    friend bool operator!=(const vertex_cond_iterator &one, const vertex_cond_iterator &other) {
        return one.current_vertex != other.current_vertex;
    };
};

/**
 * @brief Views for source vertices in a directed graph.
 *
 * These classes provide iterators to traverse the source and sink vertices
 * of a directed graph.
 */
template<typename Graph_t>
class source_vertices_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    struct source_eval {
        static bool eval(const Graph_t &graph, const vertex_idx_t<Graph_t> &v) { return graph.in_degree(v) == 0; }
    };

    using source_iterator = vertex_cond_iterator<source_eval, Graph_t, decltype(graph.vertices().begin())>;

  public:
    source_vertices_view(const Graph_t &graph_) : graph(graph_) {}

    auto begin() const { return source_iterator(graph, graph.vertices().begin()); }

    auto end() const { return source_iterator(graph, graph.vertices().end()); }
};

/**
 * @brief Views for sink vertices in a directed graph.
 *
 * These classes provide iterators to traverse the source and sink vertices
 * of a directed graph.
 */
template<typename Graph_t>
class sink_vertices_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    struct sink_eval {
        static bool eval(const Graph_t &graph, const vertex_idx_t<Graph_t> &v) { return graph.out_degree(v) == 0; }
    };

    using sink_iterator = vertex_cond_iterator<sink_eval, Graph_t, decltype(graph.vertices().begin())>;

  public:
    sink_vertices_view(const Graph_t &graph_) : graph(graph_) {}

    auto begin() const { return sink_iterator(graph, graph.vertices().begin()); }

    auto end() const { return sink_iterator(graph, graph.vertices().end()); }
};

/**
 * @brief Returns a collection containing the source vertices of a graph.
 *
 * @tparam Graph_t The type of the graph.
 * @param graph The graph to check.
 * @return A vector containing the indices of the source vertices.
 */
template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> source_vertices(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    std::vector<vertex_idx_t<Graph_t>> vec;
    for (const auto &source : source_vertices_view(graph)) {
        vec.push_back(source);
    }
    return vec;
}

/**
 * @brief Returns a collection containing the sink vertices of a graph.
 *
 * @tparam Graph_t The type of the graph.
 * @param graph The graph to check.
 * @return A vector containing the indices of the sink vertices.
 */
template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> sink_vertices(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    std::vector<vertex_idx_t<Graph_t>> vec;

    for (const auto &sink : sink_vertices_view(graph)) {
        vec.push_back(sink);
    }
    return vec;
}

/**
 * @brief Traversal iterator for directed graphs.
 *
 * This iterator allows traversing the vertices of a directed graph.
 * It uses a container wrapper to manage the traversal order.
 * The adj_iterator can be used to setup the traversal along children or parents.
 */
template<typename Graph_t, typename container_wrapper, typename adj_iterator>
struct traversal_iterator {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;

    adj_iterator adj_iter;

    container_wrapper vertex_container;

    std::unordered_set<vertex_idx_t<Graph_t>> visited;
    vertex_idx_t<Graph_t> current_vertex;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = vertex_idx_t<Graph_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type *;
    using reference = const value_type &;

    traversal_iterator(const Graph_t &graph_, const vertex_idx_t<Graph_t> &start)
        : graph(graph_), adj_iter(graph_), current_vertex(start) {

        if (graph.num_vertices() == start) {
            return;
        }

        visited.insert(start);

        for (const auto &v : adj_iter.iterate(current_vertex)) {
            vertex_container.push(v);
            visited.insert(v);
        }
    }

    value_type operator*() const { return current_vertex; }

    // Prefix increment
    traversal_iterator &operator++() {

        if (vertex_container.empty()) {
            current_vertex = graph.num_vertices();
            return *this;
        }

        current_vertex = vertex_container.pop_next();

        for (const auto &v : adj_iter.iterate(current_vertex)) {
            if (visited.find(v) == visited.end()) {
                vertex_container.push(v);
                visited.insert(v);
            }
        }

        return *this;
    }

    // Postfix increment !! expensive
    traversal_iterator operator++(int) {
        traversal_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(const traversal_iterator &one, const traversal_iterator &other) {
        return one.current_vertex == other.current_vertex;
    };
    friend bool operator!=(const traversal_iterator &one, const traversal_iterator &other) {
        return one.current_vertex != other.current_vertex;
    };
};

template<typename Graph_t>
struct child_iterator {
    const Graph_t &graph;

    child_iterator(const Graph_t &graph_) : graph(graph_) {}

    inline auto iterate(const vertex_idx_t<Graph_t> &v) const { return graph.children(v); }
};

template<typename Graph_t>
struct bfs_queue_wrapper {
    std::queue<vertex_idx_t<Graph_t>> queue;

    void push(const vertex_idx_t<Graph_t> &v) { queue.push(v); }

    vertex_idx_t<Graph_t> pop_next() {
        auto v = queue.front();
        queue.pop();
        return v;
    }

    bool empty() const { return queue.empty(); }
};

/**
 * @brief Views for traversing a directed graph using BFS.
 *
 * These classes provide iterators to traverse the vertices of a directed graph strating from a given vertex
 * using breadth-first search (BFS).
 */
template<typename Graph_t>
class bfs_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    vertex_idx_t<Graph_t> start_vertex;

    using bfs_iterator = traversal_iterator<Graph_t, bfs_queue_wrapper<Graph_t>, child_iterator<Graph_t>>;

  public:
    bfs_view(const Graph_t &graph_, const vertex_idx_t<Graph_t> &start) : graph(graph_), start_vertex(start) {}

    auto begin() const { return bfs_iterator(graph, start_vertex); }

    auto end() const { return bfs_iterator(graph, graph.num_vertices()); }
};

template<typename Graph_t>
struct dfs_stack_wrapper {
    std::vector<vertex_idx_t<Graph_t>> stack;

    void push(const vertex_idx_t<Graph_t> &v) { stack.push_back(v); }

    vertex_idx_t<Graph_t> pop_next() {
        auto v = stack.back();
        stack.pop_back();
        return v;
    }

    bool empty() const { return stack.empty(); }
};

/**
 * @brief Views for traversing a directed graph using DFS.
 *
 * These classes provide iterators to traverse the vertices of a directed graph strating from a given vertex
 * using depth-first search (DFS).
 */
template<typename Graph_t>
class dfs_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    vertex_idx_t<Graph_t> start_vertex;

    using dfs_iterator = traversal_iterator<Graph_t, dfs_stack_wrapper<Graph_t>, child_iterator<Graph_t>>;

  public:
    dfs_view(const Graph_t &graph_, const vertex_idx_t<Graph_t> &start) : graph(graph_), start_vertex(start) {}

    auto begin() const { return dfs_iterator(graph, start_vertex); }

    auto end() const { return dfs_iterator(graph, graph.num_vertices()); }
};

template<typename Graph_t>
struct parents_iterator {
    const Graph_t &graph;

    parents_iterator(const Graph_t &graph_) : graph(graph_) {}

    inline auto iterate(const vertex_idx_t<Graph_t> &v) const { return graph.parents(v); }
};

/**
 * @brief Views for traversing a directed graph using BFS in reverse order.
 *
 * These classes provide iterators to traverse the vertices of a directed graph strating from a given vertex
 * using breadth-first search (BFS) in reverse order.
 */
template<typename Graph_t>
class bfs_reverse_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    vertex_idx_t<Graph_t> start_vertex;

    using bfs_iterator = traversal_iterator<Graph_t, bfs_queue_wrapper<Graph_t>, parents_iterator<Graph_t>>;

  public:
    bfs_reverse_view(const Graph_t &graph_, const vertex_idx_t<Graph_t> &start) : graph(graph_), start_vertex(start) {}

    auto begin() const { return bfs_iterator(graph, start_vertex); }

    auto end() const { return bfs_iterator(graph, graph.num_vertices()); }
};

/**
 * @brief Returns a collection containing the successors of a vertex in a directed graph.
 *
 * @tparam Graph_t The type of the graph.
 * @param v The vertex to check.
 * @param graph The graph to check.
 * @return A vector containing the indices of the successors of the vertex.
 */
template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> successors(const vertex_idx_t<Graph_t> &v, const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    std::vector<vertex_idx_t<Graph_t>> vec;
    for (const auto &suc : bfs_view(graph, v)) {
        vec.push_back(suc);
    }
    return vec;
};

/**
 * @brief Returns a collection containing the ancestors of a vertex in a directed graph.
 *
 * @tparam Graph_t The type of the graph.
 * @param v The vertex to check.
 * @param graph The graph to check.
 * @return A vector containing the indices of the ancestors of the vertex.
 */
template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> ancestors(const vertex_idx_t<Graph_t> &v, const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    std::vector<vertex_idx_t<Graph_t>> vec;
    for (const auto &anc : bfs_reverse_view(graph, v)) {
        vec.push_back(anc);
    }
    return vec;
};

} // namespace osp