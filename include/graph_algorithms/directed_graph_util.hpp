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

namespace osp {

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

template<typename Graph_t>
bool checkNodesInTopologicalOrder(const Graph_t &graph) {
    for (const auto &node : graph.vertices()) {
        for (const auto &child : graph.children(node)) {
            if (child < node) {
                return false;
            }
        }
    }

    return true;
}

template<typename Graph_t>
bool is_sink(const vertex_idx_t<Graph_t> &v, const Graph_t &graph) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    return graph.out_degree(v) == 0u;
}

template<typename Graph_t>
bool is_source(const vertex_idx_t<Graph_t> &v, const Graph_t &graph) {
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    return graph.in_degree(v) == 0u;
}

// Function to get source vertices
template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> source_vertices(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    std::vector<vertex_idx_t<Graph_t>> vec;
    for (const vertex_idx_t<Graph_t> v_idx : graph.vertices()) {
        if (graph.in_degree(v_idx) == 0) {
            vec.push_back(v_idx);
        }
    }
    return vec;
}

// Function to get sink vertices
template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> sink_vertices(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    std::vector<vertex_idx_t<Graph_t>> vec;
    for (const vertex_idx_t<Graph_t> v_idx : graph.vertices()) {
        if (graph.out_degree(v_idx) == 0) {
            vec.push_back(v_idx);
        }
    }
    return vec;
}

template<typename cond_eval, typename Graph_t, typename iterator_t>
struct vertex_cond_iterator {

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

template<typename Graph_t>
class source_vertices_view {

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

template<typename Graph_t>
class sink_vertices_view {
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

template<typename Graph_t, typename container_wrapper>
struct traversal_iterator {

    const Graph_t &graph;

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
        : graph(graph_), current_vertex(start) {

        if (graph.num_vertices() == start) {
            return;
        }

        visited.insert(start);

        for (const auto &child : graph.children(current_vertex)) {
            vertex_container.push(child);
            visited.insert(child);
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

        for (const auto &child : graph.children(current_vertex)) {
            if (visited.find(child) == visited.end()) {
                vertex_container.push(child);
                visited.insert(child);
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
class bfs_view {

    const Graph_t &graph;
    vertex_idx_t<Graph_t> start_vertex;

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

    using bfs_iterator = traversal_iterator<Graph_t, bfs_queue_wrapper>;

  public:
    bfs_view(const Graph_t &graph_, const vertex_idx_t<Graph_t> &start) : graph(graph_), start_vertex(start) {}

    auto begin() const { return bfs_iterator(graph, start_vertex); }

    auto end() const { return bfs_iterator(graph, graph.num_vertices()); }
};

template<typename Graph_t>
class dfs_view {

    const Graph_t &graph;
    vertex_idx_t<Graph_t> start_vertex;

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

    using dfs_iterator = traversal_iterator<Graph_t, dfs_stack_wrapper>;

  public:
    dfs_view(const Graph_t &graph_, const vertex_idx_t<Graph_t> &start) : graph(graph_), start_vertex(start) {}

    auto begin() const { return dfs_iterator(graph, start_vertex); }

    auto end() const { return dfs_iterator(graph, graph.num_vertices()); }
};

} // namespace osp