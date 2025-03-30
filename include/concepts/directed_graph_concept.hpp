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

#include "iterator_concepts.hpp"

namespace osp {

using vertex_idx = size_t;

// directed_graph concept without explicit edges
template<typename T, typename = void>
struct is_directed_graph : std::false_type {};

template<typename T>
struct is_directed_graph<T, std::void_t<
    decltype(std::declval<T>().vertices()),
    decltype(std::declval<T>().num_vertices()),
    decltype(std::declval<T>().num_edges()),
    decltype(std::declval<T>().parents(std::declval<vertex_idx>())),
    decltype(std::declval<T>().children(std::declval<vertex_idx>())),
    decltype(std::declval<T>().in_degree(std::declval<vertex_idx>())),
    decltype(std::declval<T>().out_degree(std::declval<vertex_idx>())),
    decltype(T())>> 
    : std::conjunction<
    is_forward_range_of<decltype(std::declval<T>().vertices()), vertex_idx>,
    std::is_unsigned<decltype(std::declval<T>().num_vertices())>,
    std::is_unsigned<decltype(std::declval<T>().num_edges())>,
    is_forward_range_of<decltype(std::declval<T>().parents(std::declval<vertex_idx>())), vertex_idx>,
    is_forward_range_of<decltype(std::declval<T>().children(std::declval<vertex_idx>())), vertex_idx>,
    std::is_unsigned<decltype(std::declval<T>().in_degree(std::declval<vertex_idx>()))>,
    std::is_unsigned<decltype(std::declval<T>().out_degree(std::declval<vertex_idx>()))>,
    std::is_default_constructible<T>> {};

template<typename T>
inline constexpr bool is_directed_graph_v = is_directed_graph<T>::value;



using edge_idx = size_t;

// Trait to specify the directed_edge_descriptor type for a graph
template<typename T, typename = void>
struct graph_traits {};

// Specialization for graphs that define a directed_edge_descriptor
template<typename T>
struct graph_traits<T, std::void_t<typename T::directed_edge_descriptor>> {
    using directed_edge_descriptor = typename T::directed_edge_descriptor;
};


// directed_graph_edge_idx concept
template<typename T, typename = void>
struct is_directed_graph_edge_desc : std::false_type {};

template<typename T>
struct is_directed_graph_edge_desc<T, std::void_t<
    typename graph_traits<T>::directed_edge_descriptor,
    decltype(std::declval<T>().edges()),
    decltype(std::declval<T>().out_edges(std::declval<vertex_idx>())),
    decltype(std::declval<T>().in_edges(std::declval<vertex_idx>())),
    decltype(source(std::declval<typename graph_traits<T>::directed_edge_descriptor>(), std::declval<T>())),
    decltype(target(std::declval<typename graph_traits<T>::directed_edge_descriptor>(), std::declval<T>())),
    decltype(edge_id(std::declval<typename graph_traits<T>::directed_edge_descriptor>(), std::declval<T>()))>> : std::conjunction<
    is_directed_graph<T>,
    is_forward_range_of<decltype(std::declval<T>().edges()), typename graph_traits<T>::directed_edge_descriptor>,
    is_forward_range_of<decltype(std::declval<T>().out_edges(std::declval<vertex_idx>())), typename graph_traits<T>::directed_edge_descriptor>,
    is_forward_range_of<decltype(std::declval<T>().in_edges(std::declval<vertex_idx>())), typename graph_traits<T>::directed_edge_descriptor>,
    std::is_same<decltype(source(std::declval<typename graph_traits<T>::directed_edge_descriptor>(), std::declval<T>())), vertex_idx>,
    std::is_same<decltype(target(std::declval<typename graph_traits<T>::directed_edge_descriptor>(), std::declval<T>())), vertex_idx>,
    std::is_same<decltype(edge_id(std::declval<typename graph_traits<T>::directed_edge_descriptor>(), std::declval<T>())), edge_idx>
    > {};

template<typename T>
inline constexpr bool is_directed_graph_edge_desc_v = is_directed_graph_edge_desc<T>::value;


template<typename edge_desc, typename Graph_t>
constexpr vertex_idx source(const edge_desc& edge, const Graph_t& graph) {
    return graph.source(edge);
}

template<typename edge_desc, typename Graph_t>
constexpr vertex_idx target(const edge_desc& edge, const Graph_t& graph) {
    return graph.target(edge);
}

template<typename edge_desc, typename Graph_t>
constexpr edge_idx edge_id(const edge_desc& edge, const Graph_t& graph) {
    return graph.edge_id(edge);
}

} // namespace osp