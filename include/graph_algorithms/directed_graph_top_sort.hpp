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

template<typename Graph_t, typename container_wrapper>
struct top_sort_iterator {

    const Graph_t &graph;
    container_wrapper &next;

    vertex_idx_t<Graph_t> current_vertex;

    std::vector<vertex_idx_t<Graph_t>> predecessors_count;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = vertex_idx_t<Graph_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type *;
    using reference = const value_type &;

    top_sort_iterator(const Graph_t &graph_, container_wrapper &next_, vertex_idx_t<Graph_t> start)
        : graph(graph_), next(next_), current_vertex(start), predecessors_count(graph_.num_vertices(), 0) {

        if (current_vertex == graph.num_vertices()) {
            return;
        }

        for (const auto &v : graph.vertices()) {
            if (is_source(v, graph)) {
                next.push(v);
            } else {
                predecessors_count[v] = graph.in_degree(v);
            }
        }
        current_vertex = next.pop_next();

        for (const auto &child : graph.children(current_vertex)) {
            --predecessors_count[child];
            if (not predecessors_count[child]) {
                next.push(child);
            }
        }
    }

    value_type operator*() const { return current_vertex; }

    // Prefix increment
    top_sort_iterator &operator++() {

        if (next.empty()) {
            current_vertex = graph.num_vertices();
            return *this;
        }

        current_vertex = next.pop_next();

        for (const auto &child : graph.children(current_vertex)) {
            --predecessors_count[child];
            if (not predecessors_count[child]) {
                next.push(child);
            }
        }
        return *this;
    }

    // Postfix increment
    top_sort_iterator operator++(int) {
        top_sort_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(const top_sort_iterator &one, const top_sort_iterator &other) {
        return one.current_vertex == other.current_vertex;
    };
    friend bool operator!=(const top_sort_iterator &one, const top_sort_iterator &other) {
        return one.current_vertex != other.current_vertex;
    };
};

template<typename Graph_t>
class bfs_top_sort_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    bfs_queue_wrapper<Graph_t> vertex_container;

    using ts_iterator = top_sort_iterator<Graph_t, bfs_queue_wrapper<Graph_t>>;

  public:
    bfs_top_sort_view(const Graph_t &graph_) : graph(graph_) {}

    auto begin() { return ts_iterator(graph, vertex_container, 0); }

    auto end() { return ts_iterator(graph, vertex_container, graph.num_vertices()); }
};

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> bfs_top_sort(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    std::vector<vertex_idx_t<Graph_t>> top_sort;

    for (const auto &node : bfs_top_sort_view(graph)) {
        top_sort.push_back(node);
    }
    return top_sort;
}

template<typename Graph_t>
class dfs_top_sort_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    dfs_stack_wrapper<Graph_t> vertex_container;

    using ts_iterator = top_sort_iterator<Graph_t, dfs_stack_wrapper<Graph_t>>;

  public:
    dfs_top_sort_view(const Graph_t &graph_) : graph(graph_) {}

    auto begin() { return ts_iterator(graph, vertex_container, 0); }

    auto end() { return ts_iterator(graph, vertex_container, graph.num_vertices()); }
};

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> dfs_top_sort(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    std::vector<vertex_idx_t<Graph_t>> top_sort;

    for (const auto &node : dfs_top_sort_view(graph)) {
        top_sort.push_back(node);
    }
    return top_sort;
}

template<typename Graph_t, typename priority_eval_f, typename T>
struct priority_queue_wrapper {

    const priority_eval_f &prio_f;

    struct heap_node {

        vertex_idx_t<Graph_t> node;

        T priority;

        heap_node() : node(0), priority(0) {}
        heap_node(vertex_idx_t<Graph_t> n, T p) : node(n), priority(p) {}

        bool operator<(heap_node const &rhs) const {
            return (priority < rhs.priority) || (priority == rhs.priority and node > rhs.node);
        }
    };

    std::vector<heap_node> heap;

  public:
    priority_queue_wrapper(const priority_eval_f &_f) : prio_f(_f) {}

    void push(const vertex_idx_t<Graph_t> &v) {
        heap.emplace_back(v, prio_f.eval(v));
        std::push_heap(heap.begin(), heap.end());
    }

    vertex_idx_t<Graph_t> pop_next() {
        std::pop_heap(heap.begin(), heap.end());
        const auto current_node = heap.back().node;
        heap.pop_back();
        return current_node;
    }

    bool empty() const { return heap.empty(); }
};

template<typename Graph_t, typename priority_eval_f, typename T>
class priority_top_sort_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    priority_queue_wrapper<Graph_t, priority_eval_f, T> vertex_container;

    using ts_iterator = top_sort_iterator<Graph_t, priority_queue_wrapper<Graph_t, priority_eval_f, T>>;

  public:
    priority_top_sort_view(const Graph_t &graph_, const priority_eval_f &f) : graph(graph_), vertex_container(f) {}

    auto begin() const { return ts_iterator(graph, vertex_container, 0); }

    auto end() const { return ts_iterator(graph, vertex_container, graph.num_vertices()); }
};

template<typename Graph_t>
class locality_top_sort_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;

    struct loc_eval_f {
        static auto eval(vertex_idx_t<Graph_t> v) { return std::numeric_limits<vertex_idx_t<Graph_t>>::max() - v; }
    };

    priority_queue_wrapper<Graph_t, loc_eval_f, vertex_idx_t<Graph_t>> vertex_container;

    using ts_iterator = top_sort_iterator<Graph_t, priority_queue_wrapper<Graph_t, loc_eval_f, vertex_idx_t<Graph_t>>>;

  public:
    locality_top_sort_view(const Graph_t &graph_) : graph(graph_), vertex_container(loc_eval_f()) {}

    auto begin() { return ts_iterator(graph, vertex_container, 0); }

    auto end() { return ts_iterator(graph, vertex_container, graph.num_vertices()); }
};

template<typename Graph_t>
class max_children_top_sort_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    struct max_children_eval_f {

        const Graph_t &graph;

        max_children_eval_f(const Graph_t &g) : graph(g) {}

        auto eval(vertex_idx_t<Graph_t> v) const { return graph.out_degree(v); }
    };

    max_children_eval_f eval_f;

    priority_queue_wrapper<Graph_t, max_children_eval_f, vertex_idx_t<Graph_t>> vertex_container;

    using ts_iterator =
        top_sort_iterator<Graph_t, priority_queue_wrapper<Graph_t, max_children_eval_f, vertex_idx_t<Graph_t>>>;

  public:
    max_children_top_sort_view(const Graph_t &graph_) : eval_f(graph_), vertex_container(eval_f) {}

    auto begin() { return ts_iterator(eval_f.graph, vertex_container, 0); }

    auto end() { return ts_iterator(eval_f.graph, vertex_container, eval_f.graph.num_vertices()); }
};

template<typename Graph_t>
class random_top_sort_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;

    struct random_eval_f {

        std::vector<vertex_idx_t<Graph_t>> priority;

        random_eval_f(const std::size_t num) : priority(num, 0) {

            std::iota(priority.begin(), priority.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(priority.begin(), priority.end(), g);
        }

        auto eval(vertex_idx_t<Graph_t> v) const { return priority[v]; }
    };

    random_eval_f eval_f;

    priority_queue_wrapper<Graph_t, random_eval_f, vertex_idx_t<Graph_t>> vertex_container;

    using ts_iterator =
        top_sort_iterator<Graph_t, priority_queue_wrapper<Graph_t, random_eval_f, vertex_idx_t<Graph_t>>>;

  public:
    random_top_sort_view(const Graph_t &graph_)
        : graph(graph_), eval_f(graph.num_vertices()), vertex_container(eval_f) {}

    auto begin() { return ts_iterator(graph, vertex_container, 0); }

    auto end() { return ts_iterator(graph, vertex_container, graph.num_vertices()); }
};

} // namespace osp