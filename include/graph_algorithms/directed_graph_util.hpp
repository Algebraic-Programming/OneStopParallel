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

template<typename Graph_t>
bool has_path(const vertex_idx_t<Graph_t> src, const vertex_idx_t<Graph_t> dest, const Graph_t &graph) {

    static_assert(is_directed_graph_v<Graph_t>, "Graph_t must satisfy the directed_graph concept");

    std::unordered_set<vertex_idx_t<Graph_t>> visited;
    visited.emplace(src);

    std::queue<vertex_idx_t<Graph_t>> next;
    next.push(src);

    while (!next.empty()) {
        vertex_idx_t<Graph_t> v = next.front();
        next.pop();

        for (const vertex_idx_t<Graph_t> &child : graph.children(v)) {

            if (child == dest) {
                return true;
            }

            if (visited.find(child) == visited.end()) {
                visited.emplace(child);
                next.push(child);
            }
        }
    }

    return false;
}

template<typename Graph_t>
class source_vertices_view {

    const Graph_t &graph;

    template<typename iterator_t>
    struct sources_iterator {

        const Graph_t &graph;
        iterator_t current_source;

      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = vertex_idx_t<Graph_t>;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        sources_iterator(const Graph_t &graph_, const iterator_t &start)
            : graph(graph_), current_source(start) {

            while (current_source != graph.vertices().end()) {
                if (graph.in_degree(*current_source) == 0) {
                    break;
                }
                current_source++;
            }
        }

        value_type operator*() const { return current_source.operator*(); }

        // Prefix increment
        sources_iterator &operator++() {
            current_source++;

            while (current_source != graph.vertices().end()) {
                if (graph.in_degree(*current_source) == 0) {
                    break;
                }
                current_source++;
            }

            return *this;
        }

        // Postfix increment
        sources_iterator operator++(int) {
            sources_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const sources_iterator &one, const sources_iterator &other) {
            return one.current_source == other.current_source;
        };
        friend bool operator!=(const sources_iterator &one, const sources_iterator &other) {
            return one.current_source != other.current_source;
        };
    };

  public:
    source_vertices_view(const Graph_t &graph_) : graph(graph_) {}

    auto begin() const { return sources_iterator(graph, graph.vertices().begin()); }

    auto end() const { return sources_iterator(graph, graph.vertices().end()); }
};

template<typename Graph_t>
class sink_vertices_view {

    const Graph_t &graph;

    template<typename iterator_t>
    struct sinks_iterator {

        const Graph_t &graph;
        iterator_t current_sink;

      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = vertex_idx_t<Graph_t>;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

        sinks_iterator(const Graph_t &graph_, const iterator_t &start)
            : graph(graph_), current_sink(start) {

            while (current_sink != graph.vertices().end()) {
                if (graph.out_degree(*current_sink) == 0) {
                    break;
                }
                current_sink++;
            }
        }

        value_type operator*() const { return current_sink.operator*(); }

        // Prefix increment
        sinks_iterator &operator++() {
            current_sink++;

            while (current_sink != graph.vertices().end()) {
                if (graph.out_degree(*current_sink) == 0) {
                    break;
                }
                current_sink++;
            }

            return *this;
        }

        // Postfix increment
        sinks_iterator operator++(int) {
            sinks_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const sinks_iterator &one, const sinks_iterator &other) {
            return one.current_sink == other.current_sink;
        };
        friend bool operator!=(const sinks_iterator &one, const sinks_iterator &other) {
            return one.current_sink != other.current_sink;
        };
    };

  public:
    sink_vertices_view(const Graph_t &graph_) : graph(graph_) {}

    auto begin() const { return sinks_iterator(graph, graph.vertices().begin()); }

    auto end() const { return sinks_iterator(graph, graph.vertices().end()); }
};

} // namespace osp