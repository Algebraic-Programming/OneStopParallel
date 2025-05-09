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
#include "concepts/directed_graph_concept.hpp"
#include <type_traits>

namespace osp {

template<typename Graph_t>
struct directed_edge {

    vertex_idx_t<Graph_t> source;
    vertex_idx_t<Graph_t> target;

    std::size_t idx;

    directed_edge(vertex_idx_t<Graph_t> src, vertex_idx_t<Graph_t> tgt, std::size_t idx_)
        : source(src), target(tgt), idx(idx_) {}
};

template<typename Graph_t>
class edge_view {
  private:
    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    const Graph_t &graph;

    template<typename child_iterator_t>
    class directed_edge_iterator {
      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = directed_edge<Graph_t>;
        using pointer = value_type *;
        using reference = value_type &;

      private:
        const Graph_t *graph;

        vertex_idx_t<Graph_t> current_vertex;
        child_iterator_t current_child;
        std::size_t current_edge_idx;

      public:
        directed_edge_iterator() : graph(nullptr), current_vertex(0), current_edge_idx(0) {}
        directed_edge_iterator(const directed_edge_iterator &other)
            : graph(other.graph), current_vertex(other.current_vertex), current_child(other.current_child),
              current_edge_idx(other.current_edge_idx) {}

        directed_edge_iterator operator=(const directed_edge_iterator &other) {
            graph = other.graph;
            current_vertex = other.current_vertex;
            current_child = other.current_child;
            current_edge_idx = other.current_edge_idx;
            return *this;
        }

        directed_edge_iterator(const Graph_t &graph_) : graph(&graph_), current_vertex(0), current_edge_idx(0) {

            while (current_vertex != graph->num_vertices()) {
                if (graph->children(current_vertex).begin() != graph->children(current_vertex).end()) {
                    current_child = graph->children(current_vertex).begin();
                    break;
                }
                current_vertex++;
            }
        }

        directed_edge_iterator(const std::size_t edge_idx, const Graph_t &graph_)
            : graph(&graph_), current_vertex(0), current_edge_idx(edge_idx) {

            if (current_edge_idx < graph->num_edges()) {

                std::size_t tmp = 0u;

                if (tmp < current_edge_idx) {

                    while (current_vertex != graph->num_vertices()) {

                        current_child = graph->children(current_vertex).begin();

                        while (current_child != graph->children(current_vertex).end()) {

                            if (tmp == current_edge_idx) {
                                break;
                            }

                            current_child++;
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

        inline value_type operator*() const { return {current_vertex, *current_child, current_edge_idx}; }

        inline directed_edge_iterator &operator++() {

            current_child++;
            current_edge_idx++;

            if (current_child == graph->children(current_vertex).end()) {

                current_vertex++;

                while (current_vertex != graph->num_vertices()) {

                    if (graph->children(current_vertex).begin() != graph->children(current_vertex).end()) {
                        current_child = graph->children(current_vertex).begin();
                        break;
                    }

                    current_vertex++;
                }
            }

            return *this;
        }

        inline directed_edge_iterator operator++(int) {
            directed_edge_iterator temp = *this;
            ++(*this);
            return temp;
        }

        inline bool operator==(const directed_edge_iterator &other) const {
            return current_edge_idx == other.current_edge_idx;
        }

        inline bool operator!=(const directed_edge_iterator &other) const { return !(*this == other); }
    };

  public:
    using dir_edge_iterator = directed_edge_iterator<
        decltype(std::declval<Graph_t>().children(std::declval<vertex_idx_t<Graph_t>>()).begin())>;

    edge_view(const Graph_t &graph_) : graph(graph_) {}

    inline auto begin() const { return dir_edge_iterator(graph); }
    inline auto cbegin() const { return dir_edge_iterator(graph); }

    inline auto end() const { return dir_edge_iterator(graph.num_edges(), graph); }
    inline auto cend() const { return dir_edge_iterator(graph.num_edges(), graph); }

    inline auto size() const { return graph.num_edges(); }
};

} // namespace osp

template<typename Graph_t>
struct std::hash<osp::directed_edge<Graph_t>> {
    std::size_t operator()(const osp::directed_edge<Graph_t> &p) const noexcept { return p.idx; }
};