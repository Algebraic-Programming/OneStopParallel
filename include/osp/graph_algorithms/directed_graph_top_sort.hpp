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

#include "directed_graph_util.hpp"
#include "osp/auxiliary/math/math_helper.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/concepts/directed_graph_concept.hpp"

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
template <typename GraphT>
bool CheckNodesInTopologicalOrder(const GraphT &graph) {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    for (const auto &node : graph.Vertices()) {
        for (const auto &child : graph.Children(node)) {
            if (child < node) {
                return false;
            }
        }
    }

    return true;
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> GetTopOrder(const GraphT &graph) {
    if constexpr (hasVerticesInTopOrderV<GraphT>) {
        std::vector<VertexIdxT<GraphT>> topOrd(graph.NumVertices());
        std::iota(topOrd.begin(), topOrd.end(), static_cast<VertexIdxT<GraphT>>(0));
        return topOrd;

    } else {
        using VertexType = VertexIdxT<GraphT>;

        std::vector<VertexType> predecessorsCount(graph.NumVertices(), 0);
        std::vector<VertexType> topOrder;
        topOrder.reserve(graph.NumVertices());

        std::queue<VertexType> next;

        // Find source nodes
        for (const VertexType &v : SourceVerticesView(graph)) {
            next.push(v);
        }

        // Execute BFS
        while (!next.empty()) {
            const VertexType node = next.front();
            next.pop();
            topOrder.push_back(node);

            for (const VertexType &current : graph.Children(node)) {
                ++predecessorsCount[current];
                if (predecessorsCount[current] == graph.InDegree(current)) {
                    next.push(current);
                }
            }
        }

        if (static_cast<VertexType>(topOrder.size()) != graph.NumVertices()) {
            throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.NumVertices() ["
                                     + std::to_string(topOrder.size()) + " != " + std::to_string(graph.NumVertices()) + "]");
        }

        return topOrder;
    }
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> GetTopOrderReverse(const GraphT &graph) {
    std::vector<VertexIdxT<GraphT>> topOrder = GetTopOrder(graph);
    std::reverse(topOrder.begin(), topOrder.end());
    return topOrder;
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> GetTopOrderGorder(const GraphT &graph) {
    // Generating modified Gorder topological order cf. "Speedup Graph Processing by Graph Ordering" by Hao Wei, Jeffrey
    // Xu Yu, Can Lu, and Xuemin Lin

    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = VertexIdxT<GraphT>;

    std::vector<VertexType> predecessorsCount(graph.NumVertices(), 0);
    std::vector<VertexType> topOrder;
    topOrder.reserve(graph.NumVertices());

    const double decay = 8.0;

    std::vector<double> priorities(graph.NumVertices(), 0.0);

    auto vCmp = [&priorities, &graph](const VertexType &lhs, const VertexType &rhs) {
        return (priorities[lhs] < priorities[rhs])
               || ((priorities[lhs] <= priorities[rhs]) && (graph.OutDegree(lhs) < graph.OutDegree(rhs)))
               || ((priorities[lhs] <= priorities[rhs]) && (graph.OutDegree(lhs) == graph.OutDegree(rhs)) && (lhs > rhs));
    };

    std::priority_queue<VertexType, std::vector<VertexType>, decltype(vCmp)> readyQ(vCmp);
    for (const VertexType &vert : SourceVerticesView(graph)) {
        readyQ.push(vert);
    }

    while (!readyQ.empty()) {
        VertexType vert = readyQ.top();
        readyQ.pop();

        double pos = static_cast<double>(topOrder.size());
        pos /= decay;

        topOrder.push_back(vert);

        // update priorities
        for (const VertexType &chld : graph.Children(vert)) {
            priorities[chld] = LogSumExp(priorities[chld], pos);
        }
        for (const VertexType &par : graph.Parents(vert)) {
            for (const VertexType &sibling : graph.Children(par)) {
                priorities[sibling] = LogSumExp(priorities[sibling], pos);
            }
        }
        for (const VertexType &chld : graph.Children(vert)) {
            for (const VertexType &couple : graph.Parents(chld)) {
                priorities[couple] = LogSumExp(priorities[couple], pos);
            }
        }

        // update constraints and push to queue
        for (const VertexType &chld : graph.Children(vert)) {
            ++predecessorsCount[chld];
            if (predecessorsCount[chld] == graph.InDegree(chld)) {
                readyQ.push(chld);
            }
        }
    }

    if (topOrder.size() != graph.NumVertices()) {
        throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.NumVertices() ["
                                 + std::to_string(topOrder.size()) + " != " + std::to_string(graph.NumVertices()) + "]");
    }

    return topOrder;
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> GetFilteredTopOrder(const std::vector<bool> &valid, const GraphT &graph) {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    std::vector<VertexIdxT<GraphT>> filteredOrder;
    for (const auto &node : GetTopOrder(graph)) {
        if (valid[node]) {
            filteredOrder.push_back(node);
        }
    }

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
template <typename T, typename GraphT>
struct IsContainerWrapper {
  private:
    template <typename U>
    static auto Test(int) -> decltype(std::declval<U>().Push(std::declval<VertexIdxT<GraphT>>()),
                                      std::declval<U>().PopNext(),
                                      std::declval<U>().Empty(),
                                      std::true_type());

    template <typename>
    static std::false_type Test(...);

  public:
    static constexpr bool value = decltype(Test<T>(0))::value;
};

template <typename T, typename GraphT>
inline constexpr bool isContainerWrapperV = IsContainerWrapper<T, GraphT>::value;

template <typename GraphT, typename ContainerWrapper>
struct TopSortIterator {
    static_assert(isContainerWrapperV<ContainerWrapper, GraphT>, "container_wrapper must satisfy the container wrapper concept");

    const GraphT &graph;
    ContainerWrapper &next;

    VertexIdxT<GraphT> currentVertex;

    std::vector<VertexIdxT<GraphT>> predecessorsCount;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = VertexIdxT<GraphT>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type *;
    using reference = const value_type &;

    TopSortIterator(const GraphT &graph_, ContainerWrapper &next_, VertexIdxT<GraphT> start)
        : graph(graph_), next(next_), currentVertex(start), predecessorsCount(graph.NumVertices(), 0) {
        if (currentVertex == graph.NumVertices()) {
            return;
        }

        for (const auto &v : graph.Vertices()) {
            if (IsSource(v, graph)) {
                next.Push(v);
            } else {
                predecessorsCount[v] = static_cast<VertexIdxT<GraphT>>(graph.InDegree(v));
            }
        }
        currentVertex = next.PopNext();

        for (const auto &child : graph.Children(currentVertex)) {
            --predecessorsCount[child];
            if (not predecessorsCount[child]) {
                next.Push(child);
            }
        }
    }

    value_type operator*() const { return currentVertex; }

    // Prefix increment
    TopSortIterator &operator++() {
        if (next.Empty()) {
            currentVertex = graph.NumVertices();
            return *this;
        }

        currentVertex = next.PopNext();

        for (const auto &child : graph.Children(currentVertex)) {
            --predecessorsCount[child];
            if (not predecessorsCount[child]) {
                next.Push(child);
            }
        }
        return *this;
    }

    // Postfix increment
    TopSortIterator operator++(int) {
        TopSortIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(const TopSortIterator &one, const TopSortIterator &other) {
        return one.currentVertex == other.currentVertex;
    };

    friend bool operator!=(const TopSortIterator &one, const TopSortIterator &other) {
        return one.currentVertex != other.currentVertex;
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
template <typename GraphT>
class TopSortView {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;
    DfsStackWrapper<GraphT> vertexContainer_;

    using TsIterator = TopSortIterator<GraphT, DfsStackWrapper<GraphT>>;

  public:
    TopSortView(const GraphT &graph) : graph_(graph) {}

    auto Begin() {
        if constexpr (hasVerticesInTopOrderV<GraphT>) {
            return graph_.Vertices().begin();
        } else {
            return TsIterator(graph_, vertexContainer_, 0);
        }
    }

    auto End() {
        if constexpr (hasVerticesInTopOrderV<GraphT>) {
            return graph_.Vertices().end();
        } else {
            return TsIterator(graph_, vertexContainer_, graph_.NumVertices());
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
template <typename GraphT>
class DfsTopSortView {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;
    DfsStackWrapper<GraphT> vertexContainer_;

    using TsIterator = TopSortIterator<GraphT, DfsStackWrapper<GraphT>>;

  public:
    DfsTopSortView(const GraphT &graph) : graph_(graph) {}

    auto Begin() { return TsIterator(graph_, vertexContainer_, 0); }

    auto End() { return TsIterator(graph_, vertexContainer_, graph_.NumVertices()); }
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
template <typename GraphT>
class BfsTopSortView {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;
    BfsQueueWrapper<GraphT> vertexContainer_;

    using TsIterator = TopSortIterator<GraphT, BfsQueueWrapper<GraphT>>;

  public:
    BfsTopSortView(const GraphT &graph) : graph_(graph) {}

    auto Begin() { return TsIterator(graph_, vertexContainer_, 0); }

    auto End() { return TsIterator(graph_, vertexContainer_, graph_.NumVertices()); }
};

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> BfsTopSort(const GraphT &graph) {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    std::vector<VertexIdxT<GraphT>> topSort;

    for (const auto &node : BfsTopSortView(graph)) {
        topSort.push_back(node);
    }
    return topSort;
}

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> DfsTopSort(const GraphT &graph) {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    std::vector<VertexIdxT<GraphT>> topSort;

    for (const auto &node : TopSortView(graph)) {
        topSort.push_back(node);
    }
    return topSort;
}

template <typename GraphT, typename PriorityEvalF, typename T>
struct PriorityQueueWrapper {
    PriorityEvalF prioF;

    struct HeapNode {
        VertexIdxT<GraphT> node;

        T priority;

        HeapNode() : node(0), priority(0) {}

        HeapNode(VertexIdxT<GraphT> n, T p) : node(n), priority(p) {}

        bool operator<(HeapNode const &rhs) const {
            return (priority < rhs.priority) || (priority == rhs.priority and node > rhs.node);
        }
    };

    std::vector<HeapNode> heap;

  public:
    template <typename... Args>
    PriorityQueueWrapper(Args &&...args) : prioF(std::forward<Args>(args)...) {}

    void Push(const VertexIdxT<GraphT> &v) {
        heap.emplace_back(v, prioF(v));
        std::push_heap(heap.begin(), heap.end());
    }

    VertexIdxT<GraphT> PopNext() {
        std::pop_heap(heap.begin(), heap.end());
        const auto currentNode = heap.back().node;
        heap.pop_back();
        return currentNode;
    }

    bool Empty() const { return heap.empty(); }
};

template <typename GraphT, typename PriorityEvalF, typename T>
class PriorityTopSortView {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;
    using Container = PriorityQueueWrapper<GraphT, PriorityEvalF, T>;
    Container vertexContainer_;

    using TsIterator = TopSortIterator<GraphT, Container>;

  public:
    template <typename... Args>
    PriorityTopSortView(const GraphT &graph, Args &&...args) : graph_(graph), vertexContainer_(std::forward<Args>(args)...) {}

    auto Begin() const { return TsIterator(graph_, vertexContainer_, 0); }

    auto End() const { return TsIterator(graph_, vertexContainer_, graph_.NumVertices()); }
};

template <typename GraphT>
class LocalityTopSortView {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;

    struct LocEvalF {
        auto operator()(VertexIdxT<GraphT> v) { return std::numeric_limits<VertexIdxT<GraphT>>::max() - v; }
    };

    PriorityQueueWrapper<GraphT, LocEvalF, VertexIdxT<GraphT>> vertexContainer_;

    using TsIterator = TopSortIterator<GraphT, PriorityQueueWrapper<GraphT, LocEvalF, VertexIdxT<GraphT>>>;

  public:
    LocalityTopSortView(const GraphT &graph) : graph_(graph), vertexContainer_() {}

    auto Begin() { return TsIterator(graph_, vertexContainer_, 0); }

    auto End() { return TsIterator(graph_, vertexContainer_, graph_.NumVertices()); }
};

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> GetTopOrderMinIndex(const GraphT &graph) {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = VertexIdxT<GraphT>;

    std::vector<VertexType> topOrder;
    topOrder.reserve(graph.NumVertices());

    for (const auto &vert : LocalityTopSortView(graph)) {
        topOrder.push_back(vert);
    }

    if (topOrder.size() != graph.NumVertices()) {
        throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.NumVertices() ["
                                 + std::to_string(topOrder.size()) + " != " + std::to_string(graph.NumVertices()) + "]");
    }

    return topOrder;
}

template <typename GraphT>
class MaxChildrenTopSortView {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;

    struct MaxChildrenEvalF {
        const GraphT &graph;

        MaxChildrenEvalF(const GraphT &g) : graph(g) {}

        auto operator()(VertexIdxT<GraphT> v) const { return graph.OutDegree(v); }
    };

    PriorityQueueWrapper<GraphT, MaxChildrenEvalF, VertexIdxT<GraphT>> vertexContainer_;

    using TsIterator = TopSortIterator<GraphT, PriorityQueueWrapper<GraphT, MaxChildrenEvalF, VertexIdxT<GraphT>>>;

  public:
    MaxChildrenTopSortView(const GraphT &graph) : graph_(graph), vertexContainer_(graph) {}

    auto Begin() { return TsIterator(graph_, vertexContainer_, 0); }

    auto End() { return TsIterator(graph_, vertexContainer_, graph_.NumVertices()); }
};

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> GetTopOrderMaxChildren(const GraphT &graph) {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = VertexIdxT<GraphT>;

    std::vector<VertexType> topOrder;
    topOrder.reserve(graph.NumVertices());

    for (const auto &vert : MaxChildrenTopSortView(graph)) {
        topOrder.push_back(vert);
    }

    if (topOrder.size() != graph.NumVertices()) {
        throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.NumVertices() ["
                                 + std::to_string(topOrder.size()) + " != " + std::to_string(graph.NumVertices()) + "]");
    }

    return topOrder;
}

template <typename GraphT>
class RandomTopSortView {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;

    struct RandomEvalF {
        std::vector<VertexIdxT<GraphT>> priority;

        RandomEvalF(const std::size_t num) : priority(num, 0) {
            std::iota(priority.begin(), priority.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(priority.begin(), priority.end(), g);
        }

        auto operator()(VertexIdxT<GraphT> v) const { return priority[v]; }
    };

    PriorityQueueWrapper<GraphT, RandomEvalF, VertexIdxT<GraphT>> vertexContainer_;

    using TsIterator = TopSortIterator<GraphT, PriorityQueueWrapper<GraphT, RandomEvalF, VertexIdxT<GraphT>>>;

  public:
    RandomTopSortView(const GraphT &graph) : graph_(graph), vertexContainer_(graph_.NumVertices()) {}

    auto Begin() { return TsIterator(graph_, vertexContainer_, 0); }

    auto End() { return TsIterator(graph_, vertexContainer_, graph_.NumVertices()); }
};

template <typename GraphT>
std::vector<VertexIdxT<GraphT>> GetTopOrderRandom(const GraphT &graph) {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    using VertexType = VertexIdxT<GraphT>;

    std::vector<VertexType> topOrder;
    topOrder.reserve(graph.NumVertices());

    for (const auto &vert : RandomTopSortView(graph)) {
        topOrder.push_back(vert);
    }

    if (topOrder.size() != graph.NumVertices()) {
        throw std::runtime_error("Error during topological ordering: TopOrder.size() != graph.NumVertices() ["
                                 + std::to_string(topOrder.size()) + " != " + std::to_string(graph.NumVertices()) + "]");
    }

    return topOrder;
}

template <typename GraphT, typename PrioT>
class PriorityVecTopSortView {
    static_assert(isDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    const GraphT &graph_;

    struct PriorityEvalF {
        const std::vector<PrioT> &priority;

        PriorityEvalF(const std::vector<PrioT> &p) : priority(p) {}

        PrioT operator()(VertexIdxT<GraphT> v) const { return priority[v]; }
    };

    PriorityQueueWrapper<GraphT, PriorityEvalF, PrioT> vertexContainer_;

    using TsIterator = TopSortIterator<GraphT, PriorityQueueWrapper<GraphT, PriorityEvalF, PrioT>>;

  public:
    PriorityVecTopSortView(const GraphT &graph, const std::vector<PrioT> &prioritiesVec)
        : graph_(graph), vertexContainer_(prioritiesVec) {}

    auto Begin() { return TsIterator(graph_, vertexContainer_, 0); }

    auto End() { return TsIterator(graph_, vertexContainer_, graph_.NumVertices()); }
};

}    // namespace osp
