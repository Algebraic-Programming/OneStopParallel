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

#include "graph_traits.hpp"
#include "iterator_concepts.hpp"
namespace osp {

// directed_graph concept without explicit edges
template<typename T, typename = void>
struct is_directed_graph : std::false_type {};

template<typename T>
struct is_directed_graph<
    T, std::void_t<typename directed_graph_traits<T>::vertex_idx, decltype(std::declval<T>().vertices()),
                   decltype(std::declval<T>().num_vertices()), decltype(std::declval<T>().num_edges()),
                   decltype(std::declval<T>().parents(std::declval<vertex_idx_t<T>>())),
                   decltype(std::declval<T>().children(std::declval<vertex_idx_t<T>>())),
                   decltype(std::declval<T>().in_degree(std::declval<vertex_idx_t<T>>())),
                   decltype(std::declval<T>().out_degree(std::declval<vertex_idx_t<T>>())), decltype(T())>>
    : std::conjunction<
          is_forward_range_of<decltype(std::declval<T>().vertices()), vertex_idx_t<T>>,
          std::is_unsigned<decltype(std::declval<T>().num_vertices())>,
          std::is_unsigned<decltype(std::declval<T>().num_edges())>,
          is_input_range_of<decltype(std::declval<T>().parents(std::declval<vertex_idx_t<T>>())), vertex_idx_t<T>>,
          is_input_range_of<decltype(std::declval<T>().children(std::declval<vertex_idx_t<T>>())), vertex_idx_t<T>>,
          std::is_unsigned<decltype(std::declval<T>().in_degree(std::declval<vertex_idx_t<T>>()))>,
          std::is_unsigned<decltype(std::declval<T>().out_degree(std::declval<vertex_idx_t<T>>()))>,
          std::is_default_constructible<T>> {};

template<typename T>
inline constexpr bool is_directed_graph_v = is_directed_graph<T>::value;

template<typename Graph_t>
vertex_idx_t<Graph_t> source(const edge_desc_t<Graph_t> &edge, const Graph_t &graph) {
    return graph.source(edge);
}

template<typename Graph_t>
vertex_idx_t<Graph_t> target(const edge_desc_t<Graph_t> &edge, const Graph_t &graph) {
    return graph.target(edge);
}

// directed_graph_edge_idx concept
template<typename T, typename = void>
struct is_directed_graph_edge_desc : std::false_type {};

template<typename T>
struct is_directed_graph_edge_desc<T,
                                   std::void_t<typename directed_graph_edge_desc_traits<T>::directed_edge_descriptor,
                                               decltype(std::declval<T>().edges()),
                                               decltype(std::declval<T>().out_edges(std::declval<vertex_idx_t<T>>())),
                                               decltype(std::declval<T>().in_edges(std::declval<vertex_idx_t<T>>())),
                                               decltype(source(std::declval<edge_desc_t<T>>(), std::declval<T>())),
                                               decltype(target(std::declval<edge_desc_t<T>>(), std::declval<T>()))>>
    : std::conjunction<
          is_directed_graph<T>, std::is_default_constructible<edge_desc_t<T>>,
          std::is_copy_constructible<edge_desc_t<T>>,
          is_input_range_of<decltype(std::declval<T>().edges()), edge_desc_t<T>>,
          is_input_range_of<decltype(std::declval<T>().out_edges(std::declval<vertex_idx_t<T>>())), edge_desc_t<T>>,
          is_input_range_of<decltype(std::declval<T>().in_edges(std::declval<vertex_idx_t<T>>())), edge_desc_t<T>>,
          std::is_same<decltype(source(std::declval<edge_desc_t<T>>(), std::declval<T>())), vertex_idx_t<T>>,
          std::is_same<decltype(target(std::declval<edge_desc_t<T>>(), std::declval<T>())), vertex_idx_t<T>>> {};

template<typename T>
inline constexpr bool is_directed_graph_edge_desc_v = is_directed_graph_edge_desc<T>::value;

template<typename T, typename = void>
struct has_hashable_edge_desc : std::false_type {};

template<typename T>
struct has_hashable_edge_desc<
    T, std::void_t<decltype(std::hash<edge_desc_t<T>>{}(std::declval<edge_desc_t<T>>())),
                   decltype(std::declval<edge_desc_t<T>>() == std::declval<edge_desc_t<T>>(), void()),
                   decltype(std::declval<edge_desc_t<T>>() != std::declval<edge_desc_t<T>>(), void())>>
    : std::conjunction<is_directed_graph_edge_desc<T>, 
    std::is_default_constructible<edge_desc_t<T>>,
    std::is_copy_constructible<edge_desc_t<T>>> {};
;

template<typename T>
inline constexpr bool has_hashable_edge_desc_v = has_hashable_edge_desc<T>::value;

} // namespace osp