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
#include <unordered_set>
#include <vector>

#include "osp/concepts/directed_graph_concept.hpp"

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
template <typename GraphT>
bool Edge(const VertexIdxT<GraphT> &src, const VertexIdxT<GraphT> &dest, const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    for (const auto &child : graph.Children(src)) {
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
template <typename GraphT>
bool IsSink(const VertexIdxT<GraphT> &v, const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    return graph.OutDegree(v) == 0u;
}

/**
 * @brief Checks if a vertex is a source (no incoming edges).
 *
 * @tparam Graph_t The type of the graph.
 * @param v The vertex to check.
 * @param graph The graph to check.
 * @return true if the vertex is a source, false otherwise.
 */
template <typename GraphT>
bool IsSource(const VertexIdxT<GraphT> &v, const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    return graph.InDegree(v) == 0u;
}

/**
 * @brief Helper struct for iterating over vertices with a condition.
 *
 * This struct provides an iterator that filters vertices based on a given condition.
 * It is used to create views for source and sink vertices in a directed graph.
 *
 */
template <typename CondEval, typename GraphT, typename IteratorT>
struct VertexCondIterator {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    // TODO static_assert(is_callabl_v<cond_eval>;

    const GraphT &graph_;
    IteratorT currentVertex_;
    CondEval cond_;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = VertexIdxT<GraphT>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type *;
    using reference = const value_type &;

    VertexCondIterator(const GraphT &graph, const IteratorT &start) : graph_(graph), currentVertex_(start) {
        while (currentVertex_ != graph_.Vertices().end()) {
            // if (cond.eval(graph, *current_vertex)) {
            if (cond_(graph_, *currentVertex_)) {
                break;
            }
            currentVertex_++;
        }
    }

    value_type operator*() const { return currentVertex_.operator*(); }

    // Prefix increment
    VertexCondIterator &operator++() {
        currentVertex_++;

        while (currentVertex_ != graph_.Vertices().end()) {
            if (cond_(graph_, *currentVertex_)) {
                break;
            }
            currentVertex_++;
        }

        return *this;
    }

    // Postfix increment
    VertexCondIterator operator++(int) {
        VertexCondIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    inline bool operator==(const VertexCondIterator &other) { return currentVertex_ == other.currentVertex_; };

    inline bool operator!=(const VertexCondIterator &other) { return currentVertex_ != other.currentVertex_; };
};

/**
 * @brief Views for source vertices in a directed graph.
 *
 * These classes provide iterators to traverse the source and sink vertices
 * of a directed graph.
 */
template <typename GraphT>
class SourceVerticesView {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;

    struct SourceEval {
        // static bool eval(const Graph_t &graph, const VertexIdxT<GraphT> &v) { return graph.InDegree(v) == 0; }
        bool operator()(const GraphT &graph, const VertexIdxT<GraphT> &v) const { return graph.InDegree(v) == 0; }
    };

    using SourceIterator = VertexCondIterator<SourceEval, GraphT, decltype(graph_.Vertices().begin())>;

  public:
    SourceVerticesView(const GraphT &graph) : graph_(graph) {}

    auto begin() const { return SourceIterator(graph_, graph_.Vertices().begin()); }

    auto end() const { return SourceIterator(graph_, graph_.Vertices().end()); }

    auto size() const { return graph_.NumVertices(); }
};

/**
 * @brief Views for sink vertices in a directed graph.
 *
 * These classes provide iterators to traverse the source and sink vertices
 * of a directed graph.
 */
template <typename GraphT>
class SinkVerticesView {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;

    struct SinkEval {
        // static bool eval(const Graph_t &graph, const VertexIdxT<GraphT> &v) { return graph.OutDegree(v) == 0; }
        bool operator()(const GraphT &graph, const VertexIdxT<GraphT> &v) { return graph.OutDegree(v) == 0; }
    };

    using SinkIterator = VertexCondIterator<SinkEval, GraphT, decltype(graph_.Vertices().begin())>;

  public:
    SinkVerticesView(const GraphT &graph) : graph_(graph) {}

    auto begin() const { return SinkIterator(graph_, graph_.Vertices().begin()); }

    auto end() const { return SinkIterator(graph_, graph_.Vertices().end()); }

    auto size() const { return graph_.NumVertices(); }
};

/**
 * @brief Returns a collection containing the source vertices of a graph.
 *
 * @tparam Graph_t The type of the graph.
 * @param graph The graph to check.
 * @return A vector containing the indices of the source vertices.
 */
template <typename GraphT>
std::vector<VertexIdxT<GraphT>> SourceVertices(const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    std::vector<VertexIdxT<GraphT>> vec;
    for (const auto &source : SourceVerticesView(graph)) {
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
template <typename GraphT>
std::vector<VertexIdxT<GraphT>> SinkVertices(const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    std::vector<VertexIdxT<GraphT>> vec;

    for (const auto &sink : SinkVerticesView(graph)) {
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
template <typename GraphT, typename ContainerWrapper, typename AdjIterator>
struct TraversalIterator {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;

    AdjIterator adjIter_;

    ContainerWrapper vertexContainer_;

    std::unordered_set<VertexIdxT<GraphT>> visited_;
    VertexIdxT<GraphT> currentVertex_;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = VertexIdxT<GraphT>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type *;
    using reference = const value_type &;

    TraversalIterator(const GraphT &graph, const VertexIdxT<GraphT> &start)
        : graph_(graph), adjIter_(graph), currentVertex_(start) {
        if (graph_.NumVertices() == start) {
            return;
        }

        visited_.insert(start);

        for (const auto &v : adjIter_.iterate(currentVertex_)) {
            vertexContainer_.push(v);
            visited_.insert(v);
        }
    }

    value_type operator*() const { return currentVertex_; }

    // Prefix increment
    TraversalIterator &operator++() {
        if (vertexContainer_.empty()) {
            currentVertex_ = graph_.NumVertices();
            return *this;
        }

        currentVertex_ = vertexContainer_.pop_next();

        for (const auto &v : adjIter_.iterate(currentVertex_)) {
            if (visited_.find(v) == visited_.end()) {
                vertexContainer_.push(v);
                visited_.insert(v);
            }
        }

        return *this;
    }

    // Postfix increment !! expensive
    TraversalIterator operator++(int) {
        TraversalIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    inline bool operator==(const TraversalIterator &other) { return currentVertex_ == other.currentVertex_; };

    inline bool operator!=(const TraversalIterator &other) { return currentVertex_ != other.currentVertex_; };
};

template <typename GraphT>
struct ChildIterator {
    const GraphT &graph_;

    ChildIterator(const GraphT &graph) : graph_(graph) {}

    inline auto Iterate(const VertexIdxT<GraphT> &v) const { return graph_.Children(v); }
};

template <typename GraphT>
struct BfsQueueWrapper {
    std::queue<VertexIdxT<GraphT>> queue_;

    void Push(const VertexIdxT<GraphT> &v) { queue_.push(v); }

    VertexIdxT<GraphT> PopNext() {
        auto v = queue_.front();
        queue_.pop();
        return v;
    }

    bool empty() const { return queue_.empty(); }
};

/**
 * @brief Views for traversing a directed graph using BFS.
 *
 * These classes provide iterators to traverse the vertices of a directed graph strating from a given vertex
 * using breadth-first search (BFS).
 */
template <typename GraphT>
class BfsView {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;
    VertexIdxT<GraphT> startVertex_;

    using BfsIterator = TraversalIterator<GraphT, BfsQueueWrapper<GraphT>, ChildIterator<GraphT>>;

  public:
    BfsView(const GraphT &graph, const VertexIdxT<GraphT> &start) : graph_(graph), startVertex_(start) {}

    auto begin() const { return BfsIterator(graph_, startVertex_); }

    auto end() const { return BfsIterator(graph_, graph_.NumVertices()); }

    auto size() const { return graph_.NumVertices(); }
};

template <typename GraphT>
struct DfsStackWrapper {
    std::vector<VertexIdxT<GraphT>> stack_;

    void Push(const VertexIdxT<GraphT> &v) { stack_.push_back(v); }

    VertexIdxT<GraphT> PopNext() {
        auto v = stack_.back();
        stack_.pop_back();
        return v;
    }

    bool empty() const { return stack_.empty(); }
};

/**
 * @brief Views for traversing a directed graph using DFS.
 *
 * These classes provide iterators to traverse the vertices of a directed graph strating from a given vertex
 * using depth-first search (DFS).
 */
template <typename GraphT>
class DfsView {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;
    VertexIdxT<GraphT> startVertex_;

    using DfsIterator = TraversalIterator<GraphT, DfsStackWrapper<GraphT>, ChildIterator<GraphT>>;

  public:
    DfsView(const GraphT &graph, const VertexIdxT<GraphT> &start) : graph_(graph), startVertex_(start) {}

    auto begin() const { return DfsIterator(graph_, startVertex_); }

    auto end() const { return DfsIterator(graph_, graph_.NumVertices()); }

    auto size() const { return graph_.NumVertices(); }
};

template <typename GraphT>
struct ParentsIterator {
    const GraphT &graph_;

    ParentsIterator(const GraphT &graph) : graph_(graph) {}

    inline auto Iterate(const VertexIdxT<GraphT> &v) const { return graph_.Parents(v); }
};

/**
 * @brief Views for traversing a directed graph using BFS in reverse order.
 *
 * These classes provide iterators to traverse the vertices of a directed graph strating from a given vertex
 * using breadth-first search (BFS) in reverse order.
 */
template <typename GraphT>
class BfsReverseView {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;
    VertexIdxT<GraphT> startVertex_;

    using BfsIterator = TraversalIterator<GraphT, BfsQueueWrapper<GraphT>, ParentsIterator<GraphT>>;

  public:
    BfsReverseView(const GraphT &graph, const VertexIdxT<GraphT> &start) : graph_(graph), startVertex_(start) {}

    auto begin() const { return BfsIterator(graph_, startVertex_); }

    auto end() const { return BfsIterator(graph_, graph_.NumVertices()); }

    auto size() const { return graph_.NumVertices(); }
};

/**
 * @brief Returns a collection containing the successors of a vertex in a directed graph.
 *
 * @tparam Graph_t The type of the graph.
 * @param v The vertex to check.
 * @param graph The graph to check.
 * @return A vector containing the indices of the successors of the vertex.
 */
template <typename GraphT>
std::vector<VertexIdxT<GraphT>> Successors(const VertexIdxT<GraphT> &v, const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    std::vector<VertexIdxT<GraphT>> vec;
    for (const auto &suc : BfsView(graph, v)) {
        vec.push_back(suc);
    }
    return vec;
}

/**
 * @brief Returns a collection containing the ancestors of a vertex in a directed graph.
 *
 * @tparam Graph_t The type of the graph.
 * @param v The vertex to check.
 * @param graph The graph to check.
 * @return A vector containing the indices of the ancestors of the vertex.
 */
template <typename GraphT>
std::vector<VertexIdxT<GraphT>> Ancestors(const VertexIdxT<GraphT> &v, const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    std::vector<VertexIdxT<GraphT>> vec;
    for (const auto &anc : BfsReverseView(graph, v)) {
        vec.push_back(anc);
    }
    return vec;
}

template <typename GraphT>
bool IsAcyclic(const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = VertexIdxT<GraphT>;

    if (graph.NumVertices() < 2) {
        return true;
    }

    std::vector<VertexType> predecessorsCount(graph.NumVertices(), 0);

    std::queue<VertexType> next;

    // Find source nodes
    for (const VertexType &v : SourceVerticesView(graph)) {
        next.push(v);
    }

    VertexType nodeCount = 0;
    while (!next.empty()) {
        const VertexType node = next.front();
        next.pop();
        ++nodeCount;

        for (const VertexType &current : graph.Children(node)) {
            ++predecessorsCount[current];
            if (predecessorsCount[current] == graph.InDegree(current)) {
                next.push(current);
            }
        }
    }

    return nodeCount == graph.NumVertices();
}

template <typename GraphT>
bool IsConnected(const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = VertexIdxT<GraphT>;

    if (graph.NumVertices() < 2) {
        return true;
    }

    std::unordered_set<VertexType> visited;

    std::queue<VertexType> next;
    next.push(0);
    visited.insert(0);

    VertexType nodeCount = 0;
    while (!next.empty()) {
        const VertexType node = next.front();
        next.pop();
        ++nodeCount;

        for (const VertexType &current : graph.Children(node)) {
            if (visited.find(current) == visited.end()) {
                next.push(current);
                visited.insert(current);
            }
        }
    }

    return nodeCount == graph.NumVertices();
}

template <typename GraphT>
std::size_t NumCommonParents(const GraphT &graph, VertexIdxT<GraphT> v1, VertexIdxT<GraphT> v2) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    std::unordered_set<VertexIdxT<GraphT>> parents;
    parents.reserve(graph.InDegree(v1));
    for (const auto &par : graph.Parents(v1)) {
        parents.emplace(par);
    }

    std::size_t num = 0;
    for (const auto &par : graph.Parents(v2)) {
        if (parents.find(par) != parents.end()) {
            ++num;
        }
    }

    return num;
}

template <typename GraphT>
std::size_t NumCommonChildren(const GraphT &graph, VertexIdxT<GraphT> v1, VertexIdxT<GraphT> v2) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    std::unordered_set<VertexIdxT<GraphT>> childrn;
    childrn.reserve(graph.OutDegree(v1));
    for (const auto &chld : graph.Children(v1)) {
        childrn.emplace(chld);
    }

    std::size_t num = 0;
    for (const auto &chld : graph.Children(v2)) {
        if (childrn.find(chld) != childrn.end()) {
            ++num;
        }
    }

    return num;
}

/**
 * @brief Computes the weakly connected components of a directed graph.
 *
 * A weakly connected component is a maximal subgraph where for any two vertices
 * u, v in the subgraph, there is a path between u and v in the underlying
 * undirected graph.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the `directed_graph` concept.
 * @param graph The input directed graph.
 * @param[out] components A vector where `components[i]` will be the component ID for vertex `i`.
 * @return The total number of weakly connected components.
 */
template <typename GraphT>
std::size_t ComputeWeaklyConnectedComponents(const GraphT &graph, std::vector<VertexIdxT<GraphT>> &components) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    using VertexType = VertexIdxT<GraphT>;

    if (graph.NumVertices() == 0) {
        components.clear();
        return 0;
    }

    components.assign(graph.NumVertices(), std::numeric_limits<VertexType>::max());
    VertexType componentId = 0;

    for (const auto &v : graph.Vertices()) {
        if (components[v] == std::numeric_limits<VertexType>::max()) {
            std::vector<VertexType> q;
            q.push_back(v);
            components[v] = componentId;
            size_t head = 0;

            while (head < q.size()) {
                VertexType u = q[head++];
                for (const auto &neighbor : graph.Parents(u)) {
                    if (components[neighbor] == std::numeric_limits<VertexType>::max()) {
                        components[neighbor] = componentId;
                        q.push_back(neighbor);
                    }
                }
                for (const auto &neighbor : graph.Children(u)) {
                    if (components[neighbor] == std::numeric_limits<VertexType>::max()) {
                        components[neighbor] = componentId;
                        q.push_back(neighbor);
                    }
                }
            }
            componentId++;
        }
    }
    return componentId;
}

/**
 * @brief Counts the number of weakly connected components in a directed graph.
 * @param graph The input directed graph.
 * @return The number of weakly connected components.
 */
template <typename GraphT>
std::size_t CountWeaklyConnectedComponents(const GraphT &graph) {
    std::vector<VertexIdxT<GraphT>> components;
    return ComputeWeaklyConnectedComponents(graph, components);
}

}    // namespace osp
