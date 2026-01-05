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
#include <type_traits>

#include "osp/concepts/directed_graph_concept.hpp"

namespace osp {

/**
 * @brief A view over all edges in a directed graph.
 *
 * This class provides an iterator-based view to iterate over all edges in a directed graph.
 * The iteration order is lexicographical with respect to (source, target) pairs, determined by
 * the order of vertices and their adjacency lists.
 *
 * @tparam GraphT The type of the graph, which must satisfy the `is_directed_graph_v` concept.
 */
template <typename GraphT>
class EdgeView {
  private:
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");

    const GraphT &graph_;

    template <typename ChildIteratorT>
    class DirectedEdgeIterator {
      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = DirectedEdge<GraphT>;
        using pointer = value_type *;
        using reference = value_type &;

        struct ArrowProxy {
            value_type value_;

            const value_type *operator->() const noexcept { return &value_; }
        };

      private:
        const GraphT *graph_;                  // Pointer to the graph
        VertexIdxT<GraphT> currentVertex_;     // Current source vertex
        ChildIteratorT currentChild_;          // Iterator to the current target vertex in current_vertex's adjacency list
        VertexIdxT<GraphT> currentEdgeIdx_;    // Global index of the current edge in the traversal order

        void AdvanceToValid() {
            while (currentVertex_ != graph_->NumVertices()) {
                if (graph_->Children(currentVertex_).begin() != graph_->Children(currentVertex_).end()) {
                    currentChild_ = graph_->Children(currentVertex_).begin();
                    break;
                }
                currentVertex_++;
            }
        }

      public:
        DirectedEdgeIterator() noexcept : graph_(nullptr), currentVertex_(0), currentEdgeIdx_(0) {}

        DirectedEdgeIterator(const DirectedEdgeIterator &other) = default;
        DirectedEdgeIterator(DirectedEdgeIterator &&other) noexcept = default;

        DirectedEdgeIterator &operator=(const DirectedEdgeIterator &other) = default;
        DirectedEdgeIterator &operator=(DirectedEdgeIterator &&other) noexcept = default;

        explicit DirectedEdgeIterator(const GraphT &graph1) : graph_(&graph1), currentVertex_(0), currentEdgeIdx_(0) {
            AdvanceToValid();
        }

        DirectedEdgeIterator(const VertexIdxT<GraphT> edgeIdx, const GraphT &graph1)
            : graph_(&graph1), currentVertex_(0), currentEdgeIdx_(edgeIdx) {
            if (currentEdgeIdx_ >= graph_->NumEdges()) {
                currentEdgeIdx_ = graph_->NumEdges();
                currentVertex_ = graph_->NumVertices();
                return;
            }

            VertexIdxT<GraphT> currentAccumulatedEdges = 0;

            // Optimization: Skip vertices entirely if their degree is small enough
            while (currentVertex_ < graph_->NumVertices()) {
                const auto degree = graph_->OutDegree(currentVertex_);
                if (currentAccumulatedEdges + degree > currentEdgeIdx_) {
                    break;
                }
                currentAccumulatedEdges += degree;
                currentVertex_++;
            }

            // Initialize child iterator and advance within the specific vertex
            if (currentVertex_ < graph_->NumVertices()) {
                currentChild_ = graph_->Children(currentVertex_).begin();
                std::advance(currentChild_, currentEdgeIdx_ - currentAccumulatedEdges);
            }
        }

        [[nodiscard]] value_type operator*() const { return {currentVertex_, *currentChild_}; }

        [[nodiscard]] ArrowProxy operator->() const { return {operator*()}; }

        DirectedEdgeIterator &operator++() {
            currentChild_++;
            currentEdgeIdx_++;

            if (currentChild_ == graph_->Children(currentVertex_).end()) {
                currentVertex_++;
                AdvanceToValid();
            }
            return *this;
        }

        DirectedEdgeIterator operator++(int) {
            DirectedEdgeIterator temp = *this;
            ++(*this);
            return temp;
        }

        [[nodiscard]] bool operator==(const DirectedEdgeIterator &other) const noexcept {
            return currentEdgeIdx_ == other.currentEdgeIdx_;
        }

        [[nodiscard]] bool operator!=(const DirectedEdgeIterator &other) const noexcept { return !(*this == other); }
    };

  public:
    using DirEdgeIterator
        = DirectedEdgeIterator<decltype(std::declval<GraphT>().Children(std::declval<VertexIdxT<GraphT>>()).begin())>;
    using Iterator = DirEdgeIterator;
    using ConstIterator = DirEdgeIterator;

    explicit EdgeView(const GraphT &graph) : graph_(graph) {}

    [[nodiscard]] auto begin() const { return DirEdgeIterator(graph_); }

    [[nodiscard]] auto cbegin() const { return DirEdgeIterator(graph_); }

    [[nodiscard]] auto end() const { return DirEdgeIterator(graph_.NumEdges(), graph_); }

    [[nodiscard]] auto cend() const { return DirEdgeIterator(graph_.NumEdges(), graph_); }

    [[nodiscard]] auto size() const { return graph_.NumEdges(); }

    [[nodiscard]] bool empty() const { return graph_.NumEdges() == 0; }
};

/**
 * @brief A view over the incident edges of a specific vertex in a directed graph.
 *
 * This class provides an iterator-based view to iterate over either outgoing or incoming edges
 * of a given vertex. It is a lightweight, non-owning view.
 *
 * @tparam GraphT The type of the graph, which must satisfy the `is_directed_graph_v` concept.
 * @tparam IsOutgoing If true, iterates over outgoing edges; otherwise, incoming edges.
 */
template <typename GraphT, bool isOutgoing>
class IncidentEdgeView {
  private:
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");

    const GraphT &graph_;
    VertexIdxT<GraphT> anchorVertex_;

    template <typename ChildIteratorT>
    class IncidentEdgeIterator {
      public:
        using iterator_category = typename std::iterator_traits<ChildIteratorT>::iterator_category;
        using difference_type = std::ptrdiff_t;
        using value_type = DirectedEdge<GraphT>;
        using pointer = value_type *;
        using reference = value_type &;

        struct ArrowProxy {
            value_type value_;

            const value_type *operator->() const noexcept { return &value_; }
        };

      private:
        VertexIdxT<GraphT> anchorVertex_;
        ChildIteratorT currentIt_;

      public:
        IncidentEdgeIterator() = default;

        IncidentEdgeIterator(VertexIdxT<GraphT> u, ChildIteratorT it) : anchorVertex_(u), currentIt_(it) {}

        [[nodiscard]] value_type operator*() const {
            if constexpr (isOutgoing) {
                return {anchorVertex_, *currentIt_};
            } else {
                return {*currentIt_, anchorVertex_};
            }
        }

        [[nodiscard]] ArrowProxy operator->() const { return {operator*()}; }

        IncidentEdgeIterator &operator++() {
            ++currentIt_;
            return *this;
        }

        IncidentEdgeIterator operator++(int) {
            IncidentEdgeIterator temp = *this;
            ++(*this);
            return temp;
        }

        IncidentEdgeIterator &operator--() {
            --currentIt_;
            return *this;
        }

        IncidentEdgeIterator operator--(int) {
            IncidentEdgeIterator temp = *this;
            --(*this);
            return temp;
        }

        [[nodiscard]] bool operator==(const IncidentEdgeIterator &other) const noexcept { return currentIt_ == other.currentIt_; }

        [[nodiscard]] bool operator!=(const IncidentEdgeIterator &other) const noexcept { return !(*this == other); }
    };

    // Helper to deduce iterator type based on direction
    using BaseIteratorType
        = std::conditional_t<isOutgoing,
                             decltype(std::declval<GraphT>().Children(std::declval<VertexIdxT<GraphT>>()).begin()),
                             decltype(std::declval<GraphT>().Parents(std::declval<VertexIdxT<GraphT>>()).begin())>;

  public:
    using Iterator = IncidentEdgeIterator<BaseIteratorType>;
    using ConstIterator = Iterator;

    IncidentEdgeView(const GraphT &graph, VertexIdxT<GraphT> u) : graph_(graph), anchorVertex_(u) {}

    [[nodiscard]] auto begin() const {
        if constexpr (isOutgoing) {
            return Iterator(anchorVertex_, graph_.Children(anchorVertex_).begin());
        } else {
            return Iterator(anchorVertex_, graph_.Parents(anchorVertex_).begin());
        }
    }

    [[nodiscard]] auto cbegin() const { return begin(); }

    [[nodiscard]] auto end() const {
        if constexpr (isOutgoing) {
            return Iterator(anchorVertex_, graph_.Children(anchorVertex_).end());
        } else {
            return Iterator(anchorVertex_, graph_.Parents(anchorVertex_).end());
        }
    }

    [[nodiscard]] auto cend() const { return end(); }

    [[nodiscard]] auto size() const {
        if constexpr (isOutgoing) {
            return graph_.OutDegree(anchorVertex_);
        } else {
            return graph_.InDegree(anchorVertex_);
        }
    }

    [[nodiscard]] bool empty() const {
        if constexpr (isOutgoing) {
            return graph_.OutDegree(anchorVertex_) == 0;
        } else {
            return graph_.InDegree(anchorVertex_) == 0;
        }
    }
};

/**
 * @brief A view over the outgoing edges of a specific vertex in a directed graph.
 */
template <typename GraphT>
using OutEdgeView = IncidentEdgeView<GraphT, true>;

/**
 * @brief A view over the incoming edges of a specific vertex in a directed graph.
 */
template <typename GraphT>
using InEdgeView = IncidentEdgeView<GraphT, false>;

}    // namespace osp
