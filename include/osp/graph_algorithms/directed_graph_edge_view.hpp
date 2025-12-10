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
            if (currentEdgeIdx_ < graph_->num_edges()) {
                vertex_idx_t<Graph_t> tmp = 0u;
                advanceToValid();

                while (currentVertex_ != graph_->num_vertices() && tmp < currentEdgeIdx_) {
                    while (currentChild_ != graph_->children(currentVertex_).end()) {
                        if (tmp == currentEdgeIdx_) {
                            return;
                        }
                        currentChild_++;
                        tmp++;
                    }
                    // Move to next vertex
                    currentVertex_++;
                    if (currentVertex_ != graph_->num_vertices()) {
                        currentChild_ = graph_->children(currentVertex_).begin();
                        // Skip empty adjacency lists
                        while (currentVertex_ != graph_->num_vertices() &&
                               graph_->children(currentVertex_).begin() == graph_->children(currentVertex_).end()) {
                            currentVertex_++;
                            if (currentVertex_ != graph_->num_vertices()) {
                                currentChild_ = graph_->children(currentVertex_).begin();
                            }
                        }
                    }
                }
            } else {
                currentEdgeIdx_ = graph_->num_edges();
                currentVertex_ = graph_->num_vertices();
            }
        }

        [[nodiscard]] value_type operator*() const { return {currentVertex_, *currentChild_}; }
        [[nodiscard]] arrow_proxy operator->() const { return {operator*()}; }

        DirectedEdgeIterator &operator++() {
            currentChild_++;
            currentEdgeIdx_++;

            if (currentChild_ == graph_->children(currentVertex_).end()) {
                currentVertex_++;
                // Skip empty vertices
                while (currentVertex_ != graph_->num_vertices()) {
                    if (graph_->children(currentVertex_).begin() != graph_->children(currentVertex_).end()) {
                        currentChild_ = graph_->children(currentVertex_).begin();
                        break;
                    }
                    currentVertex_++;
                }
            }
            return *this;
        }

        [[nodiscard]] DirectedEdgeIterator operator++(int) {
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
 * @brief A view over the outgoing edges of a specific vertex in a directed graph.
 *
 * This class provides an iterator-based view to iterate over the outgoing edges
 * of a given vertex `u`. It is a lightweight, non-owning view.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the `is_directed_graph_v` concept.
 */
template<typename Graph_t>
class out_edge_view {
  private:
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    vertex_idx_t<Graph_t> source_vertex;

    template<typename child_iterator_t>
    class out_edge_iterator {
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
        vertex_idx_t<Graph_t> source_vertex;
        child_iterator_t current_child_it;

      public:
        out_edge_iterator() = default;
        out_edge_iterator(vertex_idx_t<Graph_t> u, child_iterator_t it) : source_vertex(u), current_child_it(it) {}

        [[nodiscard]] value_type operator*() const { return {source_vertex, *current_child_it}; }
        [[nodiscard]] arrow_proxy operator->() const { return {operator*()}; }

        out_edge_iterator &operator++() {
            ++current_child_it;
            return *this;
        }

        out_edge_iterator operator++(int) {
            out_edge_iterator temp = *this;
            ++(*this);
            return temp;
        }

        out_edge_iterator &operator--() {
            --current_child_it;
            return *this;
        }

        out_edge_iterator operator--(int) {
            out_edge_iterator temp = *this;
            --(*this);
            return temp;
        }

        [[nodiscard]] bool operator==(const out_edge_iterator &other) const noexcept {
            return current_child_it == other.current_child_it;
        }

        [[nodiscard]] bool operator!=(const out_edge_iterator &other) const noexcept { return !(*this == other); }
    };

  public:
    using iterator =
        out_edge_iterator<decltype(std::declval<Graph_t>().children(std::declval<vertex_idx_t<Graph_t>>()).begin())>;
    using const_iterator = iterator;

    out_edge_view(const Graph_t &graph_, vertex_idx_t<Graph_t> u) : graph(graph_), source_vertex(u) {}

    [[nodiscard]] auto begin() const { return iterator(source_vertex, graph.children(source_vertex).begin()); }
    [[nodiscard]] auto cbegin() const { return begin(); }

    [[nodiscard]] auto end() const { return iterator(source_vertex, graph.children(source_vertex).end()); }
    [[nodiscard]] auto cend() const { return end(); }

    [[nodiscard]] auto size() const { return graph.out_degree(source_vertex); }
    [[nodiscard]] bool empty() const { return graph.out_degree(source_vertex) == 0; }
};

/**
 * @brief A view over the incoming edges of a specific vertex in a directed graph.
 *
 * This class provides an iterator-based view to iterate over the incoming edges
 * of a given vertex `v`. It is a lightweight, non-owning view.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the `is_directed_graph_v` concept.
 */
template<typename Graph_t>
class in_edge_view {
  private:
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;
    vertex_idx_t<Graph_t> target_vertex;

    template<typename parent_iterator_t>
    class in_edge_iterator {
      public:
        using iterator_category = typename std::iterator_traits<parent_iterator_t>::iterator_category;
        using difference_type = std::ptrdiff_t;
        using value_type = directed_edge<Graph_t>;
        using pointer = value_type *;
        using reference = value_type &;

        struct arrow_proxy {
            value_type value;
            const value_type *operator->() const noexcept { return &value; }
        };

      private:
        vertex_idx_t<Graph_t> target_vertex;
        parent_iterator_t current_parent_it;

      public:
        in_edge_iterator() = default;
        in_edge_iterator(vertex_idx_t<Graph_t> v, parent_iterator_t it) : target_vertex(v), current_parent_it(it) {}

        [[nodiscard]] value_type operator*() const { return {*current_parent_it, target_vertex}; }
        [[nodiscard]] arrow_proxy operator->() const { return {operator*()}; }

        in_edge_iterator &operator++() {
            ++current_parent_it;
            return *this;
        }

        in_edge_iterator operator++(int) {
            in_edge_iterator temp = *this;
            ++(*this);
            return temp;
        }

        in_edge_iterator &operator--() {
            --current_parent_it;
            return *this;
        }

        in_edge_iterator operator--(int) {
            in_edge_iterator temp = *this;
            --(*this);
            return temp;
        }

        [[nodiscard]] bool operator==(const in_edge_iterator &other) const noexcept {
            return current_parent_it == other.current_parent_it;
        }

        [[nodiscard]] bool operator!=(const in_edge_iterator &other) const noexcept { return !(*this == other); }
    };

  public:
    using iterator =
        in_edge_iterator<decltype(std::declval<Graph_t>().parents(std::declval<vertex_idx_t<Graph_t>>()).begin())>;
    using const_iterator = iterator;

    in_edge_view(const Graph_t &graph_, vertex_idx_t<Graph_t> v) : graph(graph_), target_vertex(v) {}

    [[nodiscard]] auto begin() const { return iterator(target_vertex, graph.parents(target_vertex).begin()); }
    [[nodiscard]] auto cbegin() const { return begin(); }

    [[nodiscard]] auto end() const { return iterator(target_vertex, graph.parents(target_vertex).end()); }
    [[nodiscard]] auto cend() const { return end(); }

    [[nodiscard]] auto size() const { return graph.in_degree(target_vertex); }
    [[nodiscard]] bool empty() const { return graph.in_degree(target_vertex) == 0; }
};

} // namespace osp