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

#include "computational_dag_concept.hpp"
#include "directed_graph_concept.hpp"

namespace osp {

// add vertices
template<typename T, typename = void>
struct is_constructable_cdag_vertex : std::false_type {};

template<typename T>
struct is_constructable_cdag_vertex<T, std::void_t<decltype(std::declval<T>().add_vertex(
                                           std::declval<typename cdag_traits<T>::vertex_work_weight_t>(),
                                           std::declval<typename cdag_traits<T>::vertex_comm_weight_t>(),
                                           std::declval<typename cdag_traits<T>::vertex_mem_weight_t>())), 
                                           decltype(std::declval<T>().set_vertex_work_weight(std::declval<vertex_idx>(), std::declval<typename cdag_traits<T>::vertex_work_weight_t>())),
                                           decltype(std::declval<T>().set_vertex_comm_weight(std::declval<vertex_idx>(), std::declval<typename cdag_traits<T>::vertex_comm_weight_t>())),
                                           decltype(std::declval<T>().set_vertex_mem_weight(std::declval<vertex_idx>(), std::declval<typename cdag_traits<T>::vertex_mem_weight_t>()))>>
    : std::true_type {};

template<typename T>
inline constexpr bool is_constructable_cdag_vertex_v = is_constructable_cdag_vertex<T>::value;

// add vertices with types
template<typename T, typename = void>
struct is_constructable_cdag_typed_vertex : std::false_type {};

template<typename T>
struct is_constructable_cdag_typed_vertex<
    T,
    std::void_t<decltype(std::declval<T>().add_vertex(std::declval<typename cdag_traits<T>::vertex_work_weight_t>(),
                                                      std::declval<typename cdag_traits<T>::vertex_comm_weight_t>(),
                                                      std::declval<typename cdag_traits<T>::vertex_mem_weight_t>(),
                                                      std::declval<typename cdag_traits<T>::vertex_type_t>())),
                                                      decltype(std::declval<T>().set_vertex_type(std::declval<vertex_idx>(), std::declval<typename cdag_traits<T>::vertex_type_t>()))>>
    : is_constructable_cdag_vertex<T> {}; // for default node type

template<typename T>
inline constexpr bool is_constructable_cdag_typed_vertex_v = is_constructable_cdag_typed_vertex<T>::value;

// add edges
template<typename T, typename = void>
struct is_constructable_cdag_edge : std::false_type {};

template<typename T>
struct is_constructable_cdag_edge<
    T, std::void_t<decltype(std::declval<T>().add_edge(std::declval<vertex_idx>(), std::declval<vertex_idx>()))>>
    : std::true_type {};

template<typename T>
inline constexpr bool is_constructable_cdag_edge_v = is_constructable_cdag_edge<T>::value;

// add edges with comm costs
template<typename T, typename = void>
struct is_constructable_cdag_comm_edge : std::false_type {};

template<typename T>
struct is_constructable_cdag_comm_edge<
    T, std::void_t<decltype(std::declval<T>().add_edge(std::declval<vertex_idx>(), std::declval<vertex_idx>(),
                                                       std::declval<typename cdag_traits<T>::edge_comm_weight_t>())),
                                                       decltype(std::declval<T>().set_edge_comm_weight(std::declval<edge_idx>(), std::declval<typename cdag_traits<T>::edge_comm_weight_t>()))>>
     : is_constructable_cdag_edge<T> {}; // for default edge weight

template<typename T>
inline constexpr bool is_constructable_cdag_comm_edge_v = is_constructable_cdag_comm_edge<T>::value;

} // namespace osp