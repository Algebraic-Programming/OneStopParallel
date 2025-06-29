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
    T, std::void_t<typename directed_graph_traits<T>::vertex_idx, 
                   decltype(std::declval<T>().vertices()),
                   decltype(std::declval<T>().num_vertices()), 
                   decltype(std::declval<T>().num_edges()),
                   decltype(std::declval<T>().parents(std::declval<vertex_idx_t<T>>())),
                   decltype(std::declval<T>().children(std::declval<vertex_idx_t<T>>())),
                   decltype(std::declval<T>().in_degree(std::declval<vertex_idx_t<T>>())),
                   decltype(std::declval<T>().out_degree(std::declval<vertex_idx_t<T>>()))>>
    : std::conjunction<
          is_forward_range_of<decltype(std::declval<T>().vertices()), vertex_idx_t<T>>,
          std::is_integral<decltype(std::declval<T>().num_vertices())>,
          std::is_integral<decltype(std::declval<T>().num_edges())>,
          is_input_range_of<decltype(std::declval<T>().parents(std::declval<vertex_idx_t<T>>())), vertex_idx_t<T>>,
          is_input_range_of<decltype(std::declval<T>().children(std::declval<vertex_idx_t<T>>())), vertex_idx_t<T>>,
          std::is_integral<decltype(std::declval<T>().in_degree(std::declval<vertex_idx_t<T>>()))>,
          std::is_integral<decltype(std::declval<T>().out_degree(std::declval<vertex_idx_t<T>>()))>
          > {};

template<typename T>
inline constexpr bool is_directed_graph_v = is_directed_graph<T>::value;

template<typename T, typename v_type, typename e_type, typename = void>
struct is_edge_list_type : std::false_type {};

template<typename T, typename v_type, typename e_type>
struct is_edge_list_type<
    T, v_type, e_type, std::void_t<decltype(std::declval<T>().begin()),
                   decltype(std::declval<T>().end()),
                   decltype(std::declval<T>().size()),
                   typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type,
                   decltype(std::declval<typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>().source),
                   decltype(std::declval<typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>().target)>>
                //    decltype((*(std::declval<T>().begin())).source())>>
                //    decltype(std::declval<*(std::declval<T>().begin())>().target())>>
    : std::conjunction< std::is_same<decltype(std::declval<typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>().source), v_type>,
                        std::is_same<decltype(std::declval<typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>().target), v_type>,
                        std::is_same<decltype(std::declval<T>().size()), e_type>> {};

template<typename T, typename v_type, typename e_type>
inline constexpr bool is_edge_list_type_v = is_edge_list_type<T, v_type, e_type>::value;


} // namespace osp