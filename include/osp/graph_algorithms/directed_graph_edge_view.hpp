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
#include "osp/concepts/directed_graph_concept.hpp"
#include <type_traits>

namespace osp {

/**
 * @brief A view over all edges in a directed graph.
 *
 * This class provides an iterator-based view to iterate over all edges in a directed graph.
 * The iteration order is lexicographical with respect to (source, target) pairs, determined by
 * the order of vertices and their adjacency lists.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the `is_directed_graph_v` concept.
 */
template<typename Graph_t>
class edge_view {
  private:
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph_;

    template<typename child_iterator_t>
    class DirectedEdgeIterator {
      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = directed_edge<Graph_t>;
        using pointer = value_type *;
        using reference = value_type &;

        struct arrow_proxy {
            value_type value;
            const value_type *operator->() const noexcept { return &value; }
        };

      private:
        const Graph_t *graph_;                 // Pointer to the graph
        vertex_idx_t<Graph_t> currentVertex_;  // Current source vertex
        child_iterator_t currentChild_;        // Iterator to the current target vertex in current_vertex's adjacency list
        vertex_idx_t<Graph_t> currentEdgeIdx_; // Global index of the current edge in the traversal order

        void advanceToValid() {
            while (currentVertex_ != graph_->num_vertices()) {
                if (graph_->children(currentVertex_).begin() != graph_->children(currentVertex_).end()) {
                    currentChild_ = graph_->children(currentVertex_).begin();
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

        explicit DirectedEdgeIterator(const Graph_t &graph) : graph_(&graph), currentVertex_(0), currentEdgeIdx_(0) {
            advanceToValid();
        }

        DirectedEdgeIterator(const vertex_idx_t<Graph_t> edge_idx, const Graph_t &graph)
            : graph_(&graph), currentVertex_(0), currentEdgeIdx_(edge_idx) {

            if (currentEdgeIdx_ >= graph_->num_edges()) {
                currentEdgeIdx_ = graph_->num_edges();
                currentVertex_ = graph_->num_vertices();
                return;
            }

            vertex_idx_t<Graph_t> currentAccumulatedEdges = 0;

            // Optimization: Skip vertices entirely if their degree is small enough
            while (currentVertex_ < graph_->num_vertices()) {
                const auto degree = graph_->out_degree(currentVertex_);
                if (currentAccumulatedEdges + degree > currentEdgeIdx_) {
                    break;
                }
                currentAccumulatedEdges += degree;
                currentVertex_++;
            }

            // Initialize child iterator and advance within the specific vertex
            if (currentVertex_ < graph_->num_vertices()) {
                currentChild_ = graph_->children(currentVertex_).begin();
                std::advance(currentChild_, currentEdgeIdx_ - currentAccumulatedEdges);
            }
        }

        [[nodiscard]] value_type operator*() const { return {currentVertex_, *currentChild_}; }
        [[nodiscard]] arrow_proxy operator->() const { return {operator*()}; }

        DirectedEdgeIterator &operator++() {
            currentChild_++;
            currentEdgeIdx_++;

            if (currentChild_ == graph_->children(currentVertex_).end()) {
                currentVertex_++;
                advanceToValid();
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
    using DirEdgeIterator = DirectedEdgeIterator<decltype(std::declval<Graph_t>().children(std::declval<vertex_idx_t<Graph_t>>()).begin())>;
    using iterator = DirEdgeIterator;
    using constIterator = DirEdgeIterator;

    explicit edge_view(const Graph_t &graph) : graph_(graph) {}

    [[nodiscard]] auto begin() const { return DirEdgeIterator(graph_); }
    [[nodiscard]] auto cbegin() const { return DirEdgeIterator(graph_); }

    [[nodiscard]] auto end() const { return DirEdgeIterator(graph_.num_edges(), graph_); }
    [[nodiscard]] auto cend() const { return DirEdgeIterator(graph_.num_edges(), graph_); }

    [[nodiscard]] auto size() const { return graph_.num_edges(); }

    [[nodiscard]] bool empty() const { return graph_.num_edges() == 0; }
};

/**
 * @brief A view over the incident edges of a specific vertex in a directed graph.
 *
 * This class provides an iterator-based view to iterate over either outgoing or incoming edges
 * of a given vertex. It is a lightweight, non-owning view.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the `is_directed_graph_v` concept.
 * @tparam IsOutgoing If true, iterates over outgoing edges; otherwise, incoming edges.
 */
template<typename Graph_t, bool IsOutgoing>
class IncidentEdgeView {
  private:
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph_;
    vertex_idx_t<Graph_t> anchorVertex_;

    template<typename child_iterator_t>
    class IncidentEdgeIterator {
      public:
        using iterator_category = typename std::iterator_traits<child_iterator_t>::iterator_category;
        using difference_type = std::ptrdiff_t;
        using value_type = directed_edge<Graph_t>;
        using pointer = value_type *;
        using reference = value_type &;

        struct arrow_proxy {
            value_type value;
            const value_type *operator->() const noexcept { return &value; }
        };

      private:
        vertex_idx_t<Graph_t> anchorVertex_;
        child_iterator_t currentIt_;

      public:
        IncidentEdgeIterator() = default;
        IncidentEdgeIterator(vertex_idx_t<Graph_t> u, child_iterator_t it) : anchorVertex_(u), currentIt_(it) {}

        [[nodiscard]] value_type operator*() const {
            if constexpr (IsOutgoing) {
                return {anchorVertex_, *currentIt_};
            } else {
                return {*currentIt_, anchorVertex_};
            }
        }
        [[nodiscard]] arrow_proxy operator->() const { return {operator*()}; }

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

        [[nodiscard]] bool operator==(const IncidentEdgeIterator &other) const noexcept {
            return currentIt_ == other.currentIt_;
        }

        [[nodiscard]] bool operator!=(const IncidentEdgeIterator &other) const noexcept { return !(*this == other); }
    };

    // Helper to deduce iterator type based on direction
    using base_iterator_type =
        std::conditional_t<IsOutgoing, decltype(std::declval<Graph_t>().children(std::declval<vertex_idx_t<Graph_t>>()).begin()),
                           decltype(std::declval<Graph_t>().parents(std::declval<vertex_idx_t<Graph_t>>()).begin())>;

  public:
    using iterator = IncidentEdgeIterator<base_iterator_type>;
    using constIterator = iterator;

    IncidentEdgeView(const Graph_t &graph, vertex_idx_t<Graph_t> u) : graph_(graph), anchorVertex_(u) {}

    [[nodiscard]] auto begin() const {
        if constexpr (IsOutgoing) {
            return iterator(anchorVertex_, graph_.children(anchorVertex_).begin());
        } else {
            return iterator(anchorVertex_, graph_.parents(anchorVertex_).begin());
        }
    }
    [[nodiscard]] auto cbegin() const { return begin(); }

    [[nodiscard]] auto end() const {
        if constexpr (IsOutgoing) {
            return iterator(anchorVertex_, graph_.children(anchorVertex_).end());
        } else {
            return iterator(anchorVertex_, graph_.parents(anchorVertex_).end());
        }
    }
    [[nodiscard]] auto cend() const { return end(); }

    [[nodiscard]] auto size() const {
        if constexpr (IsOutgoing) {
            return graph_.out_degree(anchorVertex_);
        } else {
            return graph_.in_degree(anchorVertex_);
        }
    }
    [[nodiscard]] bool empty() const {
        if constexpr (IsOutgoing) {
            return graph_.out_degree(anchorVertex_) == 0;
        } else {
            return graph_.in_degree(anchorVertex_) == 0;
        }
    }
};

/**
 * @brief A view over the outgoing edges of a specific vertex in a directed graph.
 */
template<typename Graph_t>
using OutEdgeView = IncidentEdgeView<Graph_t, true>;

/**
 * @brief A view over the incoming edges of a specific vertex in a directed graph.
 */
template<typename Graph_t>
using InEdgeView = IncidentEdgeView<Graph_t, false>;

} // namespace osp