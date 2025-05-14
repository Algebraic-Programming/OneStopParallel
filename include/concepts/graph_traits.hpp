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

#define DEFINE_TYPE_MEMBER_TEST(test_name, member_name)                                                                \
    template<typename T, typename = void>                                                                              \
    struct test_name : std::false_type {};                                                                             \
    template<typename T>                                                                                               \
    struct test_name<T, std::void_t<typename T::member_name>> : std::true_type {};

DEFINE_TYPE_MEMBER_TEST(has_vertex_idx_tmember, vertex_idx)
DEFINE_TYPE_MEMBER_TEST(has_edge_desc_tmember, directed_edge_descriptor)
DEFINE_TYPE_MEMBER_TEST(has_vertex_work_weight_tmember, vertex_work_weight_type)
DEFINE_TYPE_MEMBER_TEST(has_vertex_comm_weight_tmember, vertex_comm_weight_type)
DEFINE_TYPE_MEMBER_TEST(has_vertex_mem_weight_tmember, vertex_mem_weight_type)
DEFINE_TYPE_MEMBER_TEST(has_vertex_type_tmember, vertex_type_type)
DEFINE_TYPE_MEMBER_TEST(has_edge_comm_weight_tmember, edge_comm_weight_type)

// Every directed graph must have a vertex_idx type
template<typename T>
struct directed_graph_traits {
    static_assert(has_vertex_idx_tmember<T>::value, "graph must have vertex_idx");
    using vertex_idx = typename T::vertex_idx;
};

template<typename T>
// Macro to extract the vertex_idx type from the graph
using vertex_idx_t = typename directed_graph_traits<T>::vertex_idx;

// Specialization for graphs that define a directed_edge_descriptor
template<typename T, typename = void>
struct directed_graph_edge_desc_traits : std::false_type {};

template<typename T>
struct directed_graph_edge_desc_traits<T, std::void_t<typename T::directed_edge_descriptor>> {
    static_assert(has_edge_desc_tmember<T>::value, "graph must have edge desc");
    using directed_edge_descriptor = typename T::directed_edge_descriptor;
};

template<typename T>
using edge_desc_t = typename directed_graph_edge_desc_traits<T>::directed_edge_descriptor;

template<typename T>
struct computational_dag_traits {
    static_assert(has_vertex_work_weight_tmember<T>::value, "cdag must have vertex work weight type");
    static_assert(has_vertex_comm_weight_tmember<T>::value, "cdag must have vertex comm weight type");
    static_assert(has_vertex_mem_weight_tmember<T>::value, "cdag must have vertex mem weight type");

    using vertex_work_weight_type = typename T::vertex_work_weight_type;
    using vertex_comm_weight_type = typename T::vertex_comm_weight_type;
    using vertex_mem_weight_type = typename T::vertex_mem_weight_type;
};

template<typename T>
using v_workw_t = typename computational_dag_traits<T>::vertex_work_weight_type;

template<typename T>
using v_commw_t = typename computational_dag_traits<T>::vertex_comm_weight_type;

template<typename T>
using v_memw_t = typename computational_dag_traits<T>::vertex_mem_weight_type;

template<typename T, typename = void>
struct computational_dag_typed_vertices_traits : std::false_type {};

template<typename T>
struct computational_dag_typed_vertices_traits<T, std::void_t<typename T::vertex_type_type>> {
    static_assert(has_vertex_type_tmember<T>::value, "cdag must have vertex type type");

    using vertex_type_type = typename T::vertex_type_type;
};

template<typename T>
using v_type_t = typename computational_dag_typed_vertices_traits<T>::vertex_type_type;

template<typename T, typename = void>
struct computational_dag_edge_desc_traits : std::false_type {};

template<typename T>
struct computational_dag_edge_desc_traits<T, std::void_t<typename T::edge_comm_weight_type>> {
    static_assert(has_edge_comm_weight_tmember<T>::value, "cdag must have edge comm weight type");
    using edge_comm_weight_type = typename T::edge_comm_weight_type;
};

template<typename T>
using e_commw_t = typename computational_dag_edge_desc_traits<T>::edge_comm_weight_type;

template<typename T, typename = void>
struct has_vertices_in_top_order_trait : std::false_type {};

template<typename T>
struct has_vertices_in_top_order_trait<T, std::void_t<decltype(T::vertices_in_top_order)>>
    : std::bool_constant<std::is_same_v<decltype(T::vertices_in_top_order), const bool> && T::vertices_in_top_order> {};

template<typename T>
inline constexpr bool has_vertices_in_top_order_v = has_vertices_in_top_order_trait<T>::value;

template<typename T, typename = void>
struct has_children_in_top_order_trait : std::false_type {};

template<typename T>
struct has_children_in_top_order_trait<T, std::void_t<decltype(T::children_in_top_order)>>
    : std::bool_constant<std::is_same_v<decltype(T::children_in_top_order), const bool> && T::children_in_top_order> {};

template<typename T>
inline constexpr bool has_children_in_top_order_v = has_children_in_top_order_trait<T>::value;

template<typename T, typename = void>
struct has_children_in_vertex_order_trait : std::false_type {};

template<typename T>
struct has_children_in_vertex_order_trait<T, std::void_t<decltype(T::children_in_vertex_order)>>
    : std::bool_constant<std::is_same_v<decltype(T::children_in_vertex_order), const bool> && T::children_in_vertex_order> {};

template<typename T>
inline constexpr bool has_children_in_vertex_order_v = has_children_in_vertex_order_trait<T>::value;

template<typename T, typename = void>
struct has_parents_in_top_order_trait : std::false_type {};

template<typename T>
struct has_parents_in_top_order_trait<T, std::void_t<decltype(T::parents_in_top_order)>>
    : std::bool_constant<std::is_same_v<decltype(T::parents_in_top_order), const bool> && T::parents_in_top_order> {};

template<typename T>
inline constexpr bool has_parents_in_top_order_v = has_parents_in_top_order_trait<T>::value;

template<typename T, typename = void>
struct has_parents_in_vertex_order_trait : std::false_type {};

template<typename T>
struct has_parents_in_vertex_order_trait<T, std::void_t<decltype(T::parents_in_vertex_order)>>
    : std::bool_constant<std::is_same_v<decltype(T::parents_in_vertex_order), const bool> && T::parents_in_vertex_order> {};

template<typename T>
inline constexpr bool has_parents_in_vertex_order_v = has_parents_in_vertex_order_trait<T>::value;

template<typename T>
inline constexpr bool has_parents_and_children_in_top_order_v =
    has_parents_in_top_order_trait<T>::value and has_children_in_top_order_trait<T>::value;

} // namespace osp