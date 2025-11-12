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

#include "osp/auxiliary/math/math_helper.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/concepts/directed_graph_concept.hpp"
#include "directed_graph_util.hpp"

/**
 * @file directed_graph_top_sort.hpp
 * @brief This file contains various algorithms and utilities for topological sorting of directed graphs.
 *
 * The provided functionalities include:
 * - Checking if a given order of vertices is a valid topological order.
 * - Generating topological orders based on different strategies such as:
 *   - Order based on the maximum number of children.
 *   - Randomized order.
 *   - Minimal vertex index order.
 *   - Gorder strategy for optimized graph processing.
 * - Iterators and views for BFS and DFS-based topological sorting.
 * - Priority-based topological sorting with customizable evaluation functions.
 * - Utility traits and concepts to ensure graph and container compatibility.
 *
 * The algorithms are implemented as templates to support various graph representations.
 *
 */
namespace osp {

/**
 * @brief Checks if the natural order of the vertices is a topological order.
 *
 * @tparam Graph_t The type of the graph.
 * @param graph The graph to check.
 * @return true if the vertices are in topological order, false otherwise.
 */
template<typename Graph_t>
bool checkNodesInTopologicalOrder(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

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
std::vector<vertex_idx_t<Graph_t>> GetTopOrder(const Graph_t &graph) {

    if constexpr (has_vertices_in_top_order_v<Graph_t>) {

        std::vector<vertex_idx_t<Graph_t>> topOrd(graph.num_vertices());
        std::iota(topOrd.begin(), topOrd.end(), static_cast<vertex_idx_t<Graph_t>>(0));
        return topOrd;

    } else {

        using VertexType = vertex_idx_t<Graph_t>;

        std::vector<VertexType> predecessors_count(graph.num_vertices(), 0);
        std::vector<VertexType> TopOrder;
        TopOrder.reserve(graph.num_vertices());

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

        if (TopOrder.size() != graph.num_vertices())
            throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.num_vertices() [" +
                                     std::to_string(TopOrder.size()) + " != " + std::to_string(graph.num_vertices()) +
                                     "]");

        return TopOrder;
    }
}

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> GetTopOrderReverse(const Graph_t &graph) {

    std::vector<vertex_idx_t<Graph_t>> TopOrder = GetTopOrder(graph);
    std::reverse(TopOrder.begin(), TopOrder.end());
    return TopOrder;
}

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> GetTopOrderGorder(const Graph_t &graph) {

    // Generating modified Gorder topological order cf. "Speedup Graph Processing by Graph Ordering" by Hao Wei, Jeffrey
    // Xu Yu, Can Lu, and Xuemin Lin

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<VertexType> predecessors_count(graph.num_vertices(), 0);
    std::vector<VertexType> TopOrder;
    TopOrder.reserve(graph.num_vertices());

    const double decay = 8.0;

    std::vector<double> priorities(graph.num_vertices(), 0.0);

    auto v_cmp = [&priorities, &graph](const VertexType &lhs, const VertexType &rhs) {
        return (priorities[lhs] < priorities[rhs]) ||
               ((priorities[lhs] <= priorities[rhs]) && (graph.out_degree(lhs) < graph.out_degree(rhs))) ||
               ((priorities[lhs] <= priorities[rhs]) && (graph.out_degree(lhs) == graph.out_degree(rhs)) &&
                (lhs > rhs));
    };

    std::priority_queue<VertexType, std::vector<VertexType>, decltype(v_cmp)> ready_q(v_cmp);
    for (const VertexType &vert : source_vertices_view(graph)) {
        ready_q.push(vert);
    }

    while (!ready_q.empty()) {
        VertexType vert = ready_q.top();
        ready_q.pop();

        double pos = static_cast<double>(TopOrder.size());
        pos /= decay;

        TopOrder.push_back(vert);

        // update priorities
        for (const VertexType &chld : graph.children(vert)) {
            priorities[chld] = log_sum_exp(priorities[chld], pos);
        }
        for (const VertexType &par : graph.parents(vert)) {
            for (const VertexType &sibling : graph.children(par)) {
                priorities[sibling] = log_sum_exp(priorities[sibling], pos);
            }
        }
        for (const VertexType &chld : graph.children(vert)) {
            for (const VertexType &couple : graph.parents(chld)) {
                priorities[couple] = log_sum_exp(priorities[couple], pos);
            }
        }

        // update constraints and push to queue
        for (const VertexType &chld : graph.children(vert)) {
            ++predecessors_count[chld];
            if (predecessors_count[chld] == graph.in_degree(chld)) {
                ready_q.push(chld);
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
    for (const auto &node : GetTopOrder(graph))
        if (valid[node])
            filteredOrder.push_back(node);

    return filteredOrder;
}

/**
 * @brief Trait to check if a type satisfies the container wrapper requirements.
 *
 * This trait ensures that any container wrapper used in the top_sort_iterator
 * provides the required interface for managing vertices during topological sorting.
 *
 * @tparam T The type of the container wrapper.
 * @tparam Graph_t The type of the graph.
 */
template<typename T, typename Graph_t>
struct is_container_wrapper {
  private:
    template<typename U>
    static auto test(int) -> decltype(std::declval<U>().push(std::declval<vertex_idx_t<Graph_t>>()),
                                      std::declval<U>().pop_next(), std::declval<U>().empty(), std::true_type());

    template<typename>
    static std::false_type test(...);

  public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template<typename T, typename Graph_t>
inline constexpr bool is_container_wrapper_v = is_container_wrapper<T, Graph_t>::value;

template<typename Graph_t, typename container_wrapper>
struct top_sort_iterator {

    static_assert(is_container_wrapper_v<container_wrapper, Graph_t>,
                  "container_wrapper must satisfy the container wrapper concept");

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
                predecessors_count[v] = static_cast<vertex_idx_t<Graph_t>>( graph.in_degree(v) );
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

/**
 * @class top_sort_view
 * @brief Provides a view for iterating over the vertices of a directed graph in topological order.
 *
 * This class supports two modes of iteration:
 * 1. If the graph type `Graph_t` has a predefined topological order (determined by the
 *    `has_vertices_in_top_order_v` trait), the iteration will directly use the graph's vertices.
 * 2. Otherwise, it performs a topological sort using a depth-first search (DFS) stack wrapper.
 *
 * @tparam Graph_t The type of the directed graph. Must satisfy the `is_directed_graph` concept.
 *
 */
template<typename Graph_t>
class top_sort_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    dfs_stack_wrapper<Graph_t> vertex_container;

    using ts_iterator = top_sort_iterator<Graph_t, dfs_stack_wrapper<Graph_t>>;

  public:
    top_sort_view(const Graph_t &graph_) : graph(graph_) {}

    auto begin() {
        if constexpr (has_vertices_in_top_order_v<Graph_t>) {
            return graph.vertices().begin();
        } else {
            return ts_iterator(graph, vertex_container, 0);
        }
    }

    auto end() {
        if constexpr (has_vertices_in_top_order_v<Graph_t>) {
            return graph.vertices().end();
        } else {
            return ts_iterator(graph, vertex_container, graph.num_vertices());
        }
    }
};

/**
 * @class dfs_top_sort_view
 * @brief Provides a view for performing a topological sort on a directed graph using depth-first search (DFS).
 *
 * This class is designed to work with graphs that satisfy the `directed_graph` concept. It uses a DFS-based
 * approach to generate a topological ordering of the vertices in the graph.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the `is_directed_graph` concept.
 *
 */
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

/**
 * @class bfs_top_sort_view
 * @brief Provides a view for performing a topological sort on a directed graph using breadth-first search (BFS).
 *
 * This class is designed to work with graphs that satisfy the `directed_graph` concept. It uses a BFS-based
 * approach to generate a topological ordering of the vertices in the graph.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the `is_directed_graph` concept.
 *
 */
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
std::vector<vertex_idx_t<Graph_t>> dfs_top_sort(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    std::vector<vertex_idx_t<Graph_t>> top_sort;

    for (const auto &node : top_sort_view(graph)) {
        top_sort.push_back(node);
    }
    return top_sort;
}

template<typename Graph_t, typename priority_eval_f, typename T>
struct priority_queue_wrapper {

    priority_eval_f prio_f;

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
    template<typename... Args>
    priority_queue_wrapper(Args &&...args) : prio_f(std::forward<Args>(args)...) {}

    void push(const vertex_idx_t<Graph_t> &v) {
        heap.emplace_back(v, prio_f(v));
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
    using container = priority_queue_wrapper<Graph_t, priority_eval_f, T>;
    container vertex_container;

    using ts_iterator = top_sort_iterator<Graph_t, container>;

  public:
    template<typename... Args>
    priority_top_sort_view(const Graph_t &graph_, Args &&...args)
        : graph(graph_), vertex_container(std::forward<Args>(args)...) {}

    auto begin() const { return ts_iterator(graph, vertex_container, 0); }

    auto end() const { return ts_iterator(graph, vertex_container, graph.num_vertices()); }
};

template<typename Graph_t>
class locality_top_sort_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;

    struct loc_eval_f {
        auto operator()(vertex_idx_t<Graph_t> v) { return std::numeric_limits<vertex_idx_t<Graph_t>>::max() - v; }
    };

    priority_queue_wrapper<Graph_t, loc_eval_f, vertex_idx_t<Graph_t>> vertex_container;

    using ts_iterator = top_sort_iterator<Graph_t, priority_queue_wrapper<Graph_t, loc_eval_f, vertex_idx_t<Graph_t>>>;

  public:
    locality_top_sort_view(const Graph_t &graph_) : graph(graph_), vertex_container() {}

    auto begin() { return ts_iterator(graph, vertex_container, 0); }

    auto end() { return ts_iterator(graph, vertex_container, graph.num_vertices()); }
};

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> GetTopOrderMinIndex(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<VertexType> TopOrder;
    TopOrder.reserve(graph.num_vertices());

    for (const auto &vert : locality_top_sort_view(graph)) {
        TopOrder.push_back(vert);
    }

    if (TopOrder.size() != graph.num_vertices())
        throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.num_vertices() [" +
                                 std::to_string(TopOrder.size()) + " != " + std::to_string(graph.num_vertices()) + "]");

    return TopOrder;
}

template<typename Graph_t>
class max_children_top_sort_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;

    struct max_children_eval_f {

        const Graph_t &graph;

        max_children_eval_f(const Graph_t &g) : graph(g) {}

        auto operator()(vertex_idx_t<Graph_t> v) const { return graph.out_degree(v); }
    };

    priority_queue_wrapper<Graph_t, max_children_eval_f, vertex_idx_t<Graph_t>> vertex_container;

    using ts_iterator =
        top_sort_iterator<Graph_t, priority_queue_wrapper<Graph_t, max_children_eval_f, vertex_idx_t<Graph_t>>>;

  public:
    max_children_top_sort_view(const Graph_t &graph_) : graph(graph_), vertex_container(graph_) {}

    auto begin() { return ts_iterator(graph, vertex_container, 0); }

    auto end() { return ts_iterator(graph, vertex_container, graph.num_vertices()); }
};

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> GetTopOrderMaxChildren(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<VertexType> TopOrder;
    TopOrder.reserve(graph.num_vertices());

    for (const auto &vert : max_children_top_sort_view(graph)) {
        TopOrder.push_back(vert);
    }

    if (TopOrder.size() != graph.num_vertices())
        throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.num_vertices() [" +
                                 std::to_string(TopOrder.size()) + " != " + std::to_string(graph.num_vertices()) + "]");

    return TopOrder;
}

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

        auto operator()(vertex_idx_t<Graph_t> v) const { return priority[v]; }
    };

    priority_queue_wrapper<Graph_t, random_eval_f, vertex_idx_t<Graph_t>> vertex_container;

    using ts_iterator =
        top_sort_iterator<Graph_t, priority_queue_wrapper<Graph_t, random_eval_f, vertex_idx_t<Graph_t>>>;

  public:
    random_top_sort_view(const Graph_t &graph_) : graph(graph_), vertex_container(graph.num_vertices()) {}

    auto begin() { return ts_iterator(graph, vertex_container, 0); }

    auto end() { return ts_iterator(graph, vertex_container, graph.num_vertices()); }
};

template<typename Graph_t>
std::vector<vertex_idx_t<Graph_t>> GetTopOrderRandom(const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<VertexType> TopOrder;
    TopOrder.reserve(graph.num_vertices());

    for (const auto &vert : random_top_sort_view(graph)) {
        TopOrder.push_back(vert);
    }

    if (TopOrder.size() != graph.num_vertices())
        throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.num_vertices() [" +
                                 std::to_string(TopOrder.size()) + " != " + std::to_string(graph.num_vertices()) + "]");

    return TopOrder;
}

template<typename Graph_t, typename prio_t>
class priority_vec_top_sort_view {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;

    struct priority_eval_f {

        const std::vector<prio_t> &priority;

        priority_eval_f(const std::vector<prio_t> &p) : priority(p) {}

        prio_t operator()(vertex_idx_t<Graph_t> v) const { return priority[v]; }
    };

    priority_queue_wrapper<Graph_t, priority_eval_f, prio_t> vertex_container;

    using ts_iterator = top_sort_iterator<Graph_t, priority_queue_wrapper<Graph_t, priority_eval_f, prio_t>>;

  public:
    priority_vec_top_sort_view(const Graph_t &graph_, const std::vector<prio_t> &priorities_vec)
        : graph(graph_), vertex_container(priorities_vec) {}

    auto begin() { return ts_iterator(graph, vertex_container, 0); }

    auto end() { return ts_iterator(graph, vertex_container, graph.num_vertices()); }
};

} // namespace osp