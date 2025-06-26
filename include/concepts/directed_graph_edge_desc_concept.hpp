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

#include "directed_graph_concept.hpp"
#include "graph_algorithms/directed_graph_edge_view.hpp"
#include "graph_traits.hpp"

namespace osp {

// default implementation to get the source of an edge
template<typename Graph_t>
inline vertex_idx_t<Graph_t> source(const directed_edge<Graph_t> &edge, const Graph_t &) {
    return edge.source;
}

// default implementation to get the target of an edge
template<typename Graph_t>
inline vertex_idx_t<Graph_t> target(const directed_edge<Graph_t> &edge, const Graph_t &) {
    return edge.target;
}

template<typename Graph_t>
inline edge_view<Graph_t> edges(const Graph_t &graph) {
    return edge_view(graph);
}

template<typename Graph_t>
inline out_edge_view<Graph_t> out_edges(vertex_idx_t<Graph_t> u, const Graph_t &graph) {
    return out_edge_view(graph, u);
}

template<typename Graph_t>
inline in_edge_view<Graph_t> in_edges(vertex_idx_t<Graph_t> v, const Graph_t &graph) {
    return in_edge_view(graph, v);
}

template<typename T, typename = void>
struct is_other_directed_graph_edge_desc : std::false_type {};

template<typename T>
struct is_other_directed_graph_edge_desc<T,
                                   std::void_t<typename directed_graph_edge_desc_traits<T>::directed_edge_descriptor,
                                               decltype(edges(std::declval<T>())),
                                               decltype(out_edges(std::declval<vertex_idx_t<T>>(), std::declval<T>())),
                                               decltype(in_edges(std::declval<vertex_idx_t<T>>(), std::declval<T>())),
                                               decltype(source(std::declval<edge_desc_t<T>>(), std::declval<T>())),
                                               decltype(target(std::declval<edge_desc_t<T>>(), std::declval<T>()))>>
    : std::conjunction<
          is_directed_graph<T>, std::is_default_constructible<edge_desc_t<T>>,
          std::is_copy_constructible<edge_desc_t<T>>,
          is_input_range_of<decltype(edges(std::declval<T>())), edge_desc_t<T>>,
          is_input_range_of<decltype(out_edges(std::declval<vertex_idx_t<T>>(), std::declval<T>())), edge_desc_t<T>>,
          is_input_range_of<decltype(in_edges(std::declval<vertex_idx_t<T>>(), std::declval<T>())), edge_desc_t<T>>,
          std::is_same<decltype(source(std::declval<edge_desc_t<T>>(), std::declval<T>())), vertex_idx_t<T>>,
          std::is_same<decltype(target(std::declval<edge_desc_t<T>>(), std::declval<T>())), vertex_idx_t<T>>> {};

template<typename T>
inline constexpr bool is_other_directed_graph_edge_desc_v = is_other_directed_graph_edge_desc<T>::value;

} // namespace osp

template<typename Graph_t>
struct std::hash<osp::directed_edge<Graph_t>> {
    std::size_t operator()(const osp::directed_edge<Graph_t> &p) const noexcept {
        // Combine hashes of source and target
        std::size_t h1 = std::hash<osp::vertex_idx_t<Graph_t>>{}(p.source);
        std::size_t h2 = std::hash<osp::vertex_idx_t<Graph_t>>{}(p.target);
        return h1 ^ (h2 << 1); // Simple hash combining
    }
};