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

#include <iterator>
#include <vector>

#include "osp/concepts/directed_graph_concept.hpp"

namespace osp {

template <typename Graph_t>
class edge_range_vector_impl {
    using directed_edge_descriptor = typename directed_graph_edge_desc_traits<Graph_t>::directed_edge_descriptor;
    using vertex_idx = typename directed_graph_traits<Graph_t>::vertex_idx;
    using iter = typename Graph_t::out_edges_iterator_t;
    const Graph_t &graph;

    struct edge_iterator {
        vertex_idx current_vertex;
        std::size_t current_edge_idx;
        iter current_edge;

        const Graph_t *graph;

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = directed_edge_descriptor;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        edge_iterator() : current_vertex(0u), current_edge_idx(0u), graph(nullptr) {}

        edge_iterator(const edge_iterator &other)
            : current_vertex(other.current_vertex), current_edge_idx(other.current_edge_idx), graph(other.graph) {}

        edge_iterator &operator=(const edge_iterator &other) {
            if (this != &other) {
                current_vertex = other.current_vertex;
                current_edge_idx = other.current_edge_idx;
                graph = other.graph;
            }
            return *this;
        }

        edge_iterator(const Graph_t &graph_) : current_vertex(0u), current_edge_idx(0u), graph(&graph_) {
            while (current_vertex != graph->num_vertices()) {
                if (graph->out_edges(current_vertex).begin() != graph->out_edges(current_vertex).end()) {
                    current_edge = graph->out_edges(current_vertex).begin();
                    break;
                }
                current_vertex++;
            }
        }

        edge_iterator(std::size_t current_edge_idx_, const Graph_t &graph_)
            : current_vertex(0u), current_edge_idx(current_edge_idx_), graph(&graph_) {
            if (current_edge_idx < graph->num_edges()) {
                std::size_t tmp = 0u;

                if (tmp < current_edge_idx) {
                    while (current_vertex != graph->num_vertices()) {
                        current_edge = graph->out_edges(current_vertex).begin();

                        while (current_edge != graph->out_edges(current_vertex).end()) {
                            if (tmp == current_edge_idx) {
                                break;
                            }

                            current_edge++;
                            tmp++;
                        }

                        current_vertex++;
                    }
                }

            } else {
                current_edge_idx = graph->num_edges();
                current_vertex = graph->num_vertices();
            }
        }

        const value_type &operator*() const { return *current_edge; }

        const value_type *operator->() const { return &(*current_edge); }

        // Prefix increment
        edge_iterator &operator++() {
            current_edge++;
            current_edge_idx++;

            if (current_edge == graph->out_edges(current_vertex).end()) {
                current_vertex++;

                while (current_vertex != graph->num_vertices()) {
                    if (graph->out_edges(current_vertex).begin() != graph->out_edges(current_vertex).end()) {
                        current_edge = graph->out_edges(current_vertex).begin();
                        break;
                    }

                    current_vertex++;
                }
            }

            return *this;
        }

        // Postfix increment
        edge_iterator operator++(int) {
            edge_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        inline bool operator==(const edge_iterator &other) const { return current_edge_idx == other.current_edge_idx; }

        inline bool operator!=(const edge_iterator &other) const { return current_edge_idx != other.current_edge_idx; }
    };

  public:
    edge_range_vector_impl(const Graph_t &graph_) : graph(graph_) {}

    auto begin() const { return edge_iterator(graph); }

    auto end() const { return edge_iterator(graph.num_edges(), graph); }

    auto size() const { return graph.num_edges(); }
};

template <typename Graph_t>
class edge_source_range {
    using directed_edge_descriptor = typename directed_graph_edge_desc_traits<Graph_t>::directed_edge_descriptor;
    using vertex_idx = typename directed_graph_traits<Graph_t>::vertex_idx;
    using iter = typename Graph_t::in_edges_iterator_t;

    const Graph_t &graph;
    const std::vector<directed_edge_descriptor> &edges;

    struct source_iterator {
        const Graph_t *graph;
        iter current_edge;

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = vertex_idx;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        source_iterator() : graph(nullptr) {}

        source_iterator(const source_iterator &other) : graph(other.graph), current_edge(other.current_edge) {}

        source_iterator &operator=(const source_iterator &other) {
            if (this != &other) {
                graph = other.graph;
                current_edge = other.current_edge;
            }
            return *this;
        }

        source_iterator(iter current_edge_, const Graph_t &graph_) : graph(&graph_), current_edge(current_edge_) {}

        value_type operator*() const { return source(*current_edge, *graph); }

        // Prefix increment
        source_iterator &operator++() {
            current_edge++;
            return *this;
        }

        // Postfix increment
        source_iterator operator++(int) {
            source_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        inline bool operator==(const source_iterator &other) const { return current_edge == other.current_edge; }

        inline bool operator!=(const source_iterator &other) const { return current_edge != other.current_edge; }
    };

  public:
    edge_source_range(const std::vector<directed_edge_descriptor> &edges_, const Graph_t &graph_)
        : graph(graph_), edges(edges_) {}

    auto begin() const { return source_iterator(edges.begin(), graph); }

    auto end() const { return source_iterator(edges.end(), graph); }

    auto size() const { return edges.size(); }
};

template <typename Graph_t>
class edge_target_range {
    using directed_edge_descriptor = typename directed_graph_edge_desc_traits<Graph_t>::directed_edge_descriptor;
    using vertex_idx = typename directed_graph_traits<Graph_t>::vertex_idx;
    using iter = typename Graph_t::out_edges_iterator_t;
    const Graph_t &graph;
    const std::vector<directed_edge_descriptor> &edges;

    struct target_iterator {
        const Graph_t *graph;
        iter current_edge;

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = vertex_idx;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        target_iterator() : graph(nullptr) {}

        target_iterator(const target_iterator &other) : graph(other.graph), current_edge(other.current_edge) {}

        target_iterator &operator=(const target_iterator &other) {
            if (this != &other) {
                graph = other.graph;
                current_edge = other.current_edge;
            }
            return *this;
        }

        target_iterator(iter current_edge_, const Graph_t &graph_) : graph(&graph_), current_edge(current_edge_) {}

        value_type operator*() const { return target(*current_edge, *graph); }

        // Prefix increment
        target_iterator &operator++() {
            current_edge++;
            return *this;
        }

        // Postfix increment
        target_iterator operator++(int) {
            target_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        inline bool operator==(const target_iterator &other) const { return current_edge == other.current_edge; }

        inline bool operator!=(const target_iterator &other) const { return current_edge != other.current_edge; }
    };

  public:
    edge_target_range(const std::vector<directed_edge_descriptor> &edges_, const Graph_t &graph_)
        : graph(graph_), edges(edges_) {}

    auto begin() const { return target_iterator(edges.begin(), graph); }

    auto end() const { return target_iterator(edges.end(), graph); }

    auto size() const { return edges.size(); }
};

}    // namespace osp
