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
    decltype(std::declval<T>().out_degree(std::declval<vertex_idx>()))>> 
    : std::conjunction<
    is_input_range_of<decltype(std::declval<T>().vertices()), vertex_idx>,
    std::is_unsigned<decltype(std::declval<T>().num_vertices())>,
    std::is_unsigned<decltype(std::declval<T>().num_edges())>,
    is_input_range_of<decltype(std::declval<T>().parents(std::declval<vertex_idx>())), vertex_idx>,
    is_input_range_of<decltype(std::declval<T>().children(std::declval<vertex_idx>())), vertex_idx>,
    std::is_unsigned<decltype(std::declval<T>().in_degree(std::declval<vertex_idx>()))>,
    std::is_unsigned<decltype(std::declval<T>().out_degree(std::declval<vertex_idx>()))>> {};

template<typename T>
inline constexpr bool is_directed_graph_v = is_directed_graph<T>::value;


// template<typename T>
// struct is_constructable_from_directed_graph : std::false_type {};

// template<typename T>
// struct is_constructable_from_directed_graph<std::vector<T>> : std::conjunction<
//     is_directed_graph<T>> {};

// template<typename T>
// inline constexpr bool is_constructable_from_directed_graph_v = is_constructable_from_directed_graph<T>::value;



using edge_idx = size_t;

struct directed_edge_descriptor {

    edge_idx idx;

    vertex_idx source;
    vertex_idx target;

    directed_edge_descriptor() = default;
    directed_edge_descriptor(vertex_idx source, vertex_idx target, edge_idx idx) : idx(idx), source(source), target(target) {}
    ~directed_edge_descriptor() = default;
};

// directed_graph_edge_idx concept
template<typename T, typename = void>
struct is_directed_graph_edge_desc : std::false_type {};

template<typename T>
struct is_directed_graph_edge_desc<T, std::void_t<
    decltype(std::declval<T>().edges()),
    decltype(std::declval<T>().out_edges(std::declval<vertex_idx>())),
    decltype(std::declval<T>().in_edges(std::declval<vertex_idx>()))>> : std::conjunction<
    is_directed_graph<T>,
    is_input_range_of<decltype(std::declval<T>().edges()), directed_edge_descriptor>,
    is_input_range_of<decltype(std::declval<T>().out_edges(std::declval<vertex_idx>())), directed_edge_descriptor>,
    is_input_range_of<decltype(std::declval<T>().in_edges(std::declval<vertex_idx>())), directed_edge_descriptor>
    > {};

template<typename T>
inline constexpr bool is_directed_graph_edge_desc_v = is_directed_graph_edge_desc<T>::value;


} // namespace osp