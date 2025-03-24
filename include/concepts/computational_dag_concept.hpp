#pragma once

#include <iterator>
#include <type_traits>

#include "directed_graph_concept.hpp"

namespace osp {

// weighted vertices explicit edges
template<typename T, typename = void>
struct has_vertex_weights : std::false_type {};

template<typename T>
struct has_vertex_weights<T, std::void_t<decltype(std::declval<T>().vertex_work_weight(std::declval<vertex_idx>())),
                                         decltype(std::declval<T>().vertex_comm_weight(std::declval<vertex_idx>())),
                                         decltype(std::declval<T>().vertex_mem_weight(std::declval<vertex_idx>()))>>
    : std::conjunction<std::is_arithmetic<decltype(std::declval<T>().vertex_work_weight(std::declval<vertex_idx>()))>,
                       std::is_arithmetic<decltype(std::declval<T>().vertex_comm_weight(std::declval<vertex_idx>()))>,
                       std::is_arithmetic<decltype(std::declval<T>().vertex_mem_weight(std::declval<vertex_idx>()))>> {
};

template<typename T>
inline constexpr bool has_vertex_weights_v = has_vertex_weights<T>::value;

// typed vertices concept
template<typename T, typename = void>
struct has_typed_vertices : std::false_type {};

template<typename T>
struct has_typed_vertices<T, std::void_t<decltype(std::declval<T>().vertex_type(std::declval<vertex_idx>())),
                                         decltype(std::declval<T>().num_vertex_types())>>
    : std::conjunction<std::is_integral<decltype(std::declval<T>().vertex_type(std::declval<vertex_idx>()))>,
                       std::is_integral<decltype(std::declval<T>().num_vertex_types())>> {};

template<typename T>
inline constexpr bool has_typed_vertices_v = has_typed_vertices<T>::value;

// weighted edges concept
template<typename T, typename = void>
struct has_edge_weights : std::false_type {};

template<typename T>
struct has_edge_weights<T, std::void_t<decltype(std::declval<T>().edge_comm_weight(std::declval<edge_idx>()))>>
    : std::conjunction<std::is_arithmetic<decltype(std::declval<T>().edge_comm_weight(std::declval<edge_idx>()))>> {};

template<typename T>
inline constexpr bool has_edge_weights_v = has_edge_weights<T>::value;

// computation dag concept without explicit edges
template<typename T, typename = void>
struct is_computation_dag : std::false_type {};

template<typename T>
struct is_computation_dag<T, std::void_t<>> : std::conjunction<is_directed_graph<T>, has_vertex_weights<T>> {};

template<typename T>
inline constexpr bool is_computation_dag_v = is_computation_dag<T>::value;

// computation dag with typed vertices concept
template<typename T, typename = void>
struct is_computation_dag_typed_vertices : std::false_type {};

template<typename T>
struct is_computation_dag_typed_vertices<T, std::void_t<>>
    : std::conjunction<is_computation_dag<T>, has_typed_vertices<T>> {};

template<typename T>
inline constexpr bool is_computation_dag_typed_vertices_v = is_computation_dag_typed_vertices<T>::value;

// computation dag with explicit edges concept
template<typename T, typename = void>
struct is_computation_dag_edge_desc : std::false_type {};

template<typename T>
struct is_computation_dag_edge_desc<T, std::void_t<>>
    : std::conjunction<is_directed_graph_edge_desc<T>, is_computation_dag<T>> {};

template<typename T>
inline constexpr bool is_computation_dag_edge_desc_v = is_computation_dag_edge_desc<T>::value;

// computation_dag_typed_vertices_edge_idx concept
template<typename T, typename = void>
struct is_computation_dag_typed_vertices_edge_desc : std::false_type {};

template<typename T>
struct is_computation_dag_typed_vertices_edge_desc<T, std::void_t<>>
    : std::conjunction<is_directed_graph_edge_desc<T>, is_computation_dag_typed_vertices<T>> {};

template<typename T>
inline constexpr bool is_computation_dag_typed_vertices_edge_desc_v =
    is_computation_dag_typed_vertices_edge_desc<T>::value;

} // namespace osp