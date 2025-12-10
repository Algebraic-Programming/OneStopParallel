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

/**
 * @file graph_traits.hpp
 * @brief Type traits and concepts for graph structures in OneStopParallel.
 *
 * This file defines the core requirements for types used by graph implementations in the library,
 * specifically for computational DAGs. It provides mechanisms
 * to extract types for vertex indices, edge descriptors, and weights,
 * ensuring that graph implementations conform to the expected interfaces.
 */

namespace osp {

/**
 * @brief Traits to check for the existence of specific type members.
 *
 * These structs inherit from `std::true_type` if the specified member type exists in `T`,
 * otherwise they inherit from `std::false_type`.
 */
template<typename T, typename = void>
struct has_vertex_idx_tmember : std::false_type {};
template<typename T>
struct has_vertex_idx_tmember<T, std::void_t<typename T::vertex_idx>> : std::true_type {};

template<typename T, typename = void>
struct has_edge_desc_tmember : std::false_type {};
template<typename T>
struct has_edge_desc_tmember<T, std::void_t<typename T::directed_edge_descriptor>> : std::true_type {};

template<typename T, typename = void>
struct has_vertex_work_weight_tmember : std::false_type {};
template<typename T>
struct has_vertex_work_weight_tmember<T, std::void_t<typename T::vertex_work_weight_type>> : std::true_type {};

template<typename T, typename = void>
struct has_vertex_comm_weight_tmember : std::false_type {};
template<typename T>
struct has_vertex_comm_weight_tmember<T, std::void_t<typename T::vertex_comm_weight_type>> : std::true_type {};

template<typename T, typename = void>
struct has_vertex_mem_weight_tmember : std::false_type {};
template<typename T>
struct has_vertex_mem_weight_tmember<T, std::void_t<typename T::vertex_mem_weight_type>> : std::true_type {};

template<typename T, typename = void>
struct has_vertex_type_tmember : std::false_type {};
template<typename T>
struct has_vertex_type_tmember<T, std::void_t<typename T::vertex_type_type>> : std::true_type {};

template<typename T, typename = void>
struct has_edge_comm_weight_tmember : std::false_type {};
template<typename T>
struct has_edge_comm_weight_tmember<T, std::void_t<typename T::edge_comm_weight_type>> : std::true_type {};

/**
 * @brief Core traits for any directed graph type.
 *
 * Requires that the graph type `T` defines a `vertex_idx` type member.
 *
 * @tparam T The graph type.
 */
template<typename T>
struct directed_graph_traits {
    static_assert(has_vertex_idx_tmember<T>::value, "graph must have vertex_idx");
    using vertex_idx = typename T::vertex_idx;
};

/**
 * @brief Alias to easily access the vertex index type of a graph.
 */
template<typename T>
using vertex_idx_t = typename directed_graph_traits<T>::vertex_idx;

/**
 * @brief A default edge descriptor for directed graphs.
 *
 * This struct is used when the graph type does not provide its own edge descriptor.
 * It simply holds the source and target vertex indices.
 *
 * @tparam Graph_t The graph type.
 */
template<typename Graph_t>
struct directed_edge {
    vertex_idx_t<Graph_t> source;
    vertex_idx_t<Graph_t> target;

    bool operator==(const directed_edge &other) const { return source == other.source && target == other.target; }
    bool operator!=(const directed_edge &other) const { return !(*this == other); }
    directed_edge() : source(0), target(0) {}
    directed_edge(const directed_edge &other) = default;
    directed_edge(directed_edge &&other) = default;
    directed_edge &operator=(const directed_edge &other) = default;
    directed_edge &operator=(directed_edge &&other) = default;
    ~directed_edge() = default;

    directed_edge(vertex_idx_t<Graph_t> src, vertex_idx_t<Graph_t> tgt) : source(src), target(tgt) {}
};

/**
 * @brief Helper struct to extract the edge descriptor type of a directed graph.
 *
 * If the graph defines `directed_edge_descriptor`, it is extracted; otherwise, `directed_edge` is used as a default implementation.
 */
template<typename T, bool has_edge>
struct directed_graph_edge_desc_traits_helper {
    using directed_edge_descriptor = directed_edge<T>;
};

template<typename T>
struct directed_graph_edge_desc_traits_helper<T, true> {
    using directed_edge_descriptor = typename T::directed_edge_descriptor;
};

template<typename T>
struct directed_graph_edge_desc_traits {
    using directed_edge_descriptor =
        typename directed_graph_edge_desc_traits_helper<T, has_edge_desc_tmember<T>::value>::directed_edge_descriptor;
};

template<typename T>
using edge_desc_t = typename directed_graph_edge_desc_traits<T>::directed_edge_descriptor;

/**
 * @brief Traits for computational Directed Acyclic Graphs (DAGs).
 *
 * Computational DAGs extend basic graphs by adding requirements for weight types:
 * - `vertex_work_weight_type`: Represents computational cost of a task.
 * - `vertex_comm_weight_type`: Represents data size/communication cost.
 * - `vertex_mem_weight_type`: Represents memory usage of a task.
 *
 * @tparam T The computational DAG type.
 */
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

/**
 * @brief Traits to extract the vertex type of a computational DAG, if defined.
 *
 * If the DAG defines `vertex_type_type`, it is extracted; otherwise, `void` is used.
 */
template<typename T, typename = void>
struct computational_dag_typed_vertices_traits {
    using vertex_type_type = void;
};

template<typename T>
struct computational_dag_typed_vertices_traits<T, std::void_t<typename T::vertex_type_type>> {
    using vertex_type_type = typename T::vertex_type_type;
};

template<typename T>
using v_type_t = typename computational_dag_typed_vertices_traits<T>::vertex_type_type;

/**
 * @brief Traits to extract the edge communication weight type of a computational DAG, if defined.
 *
 * If the DAG defines `edge_comm_weight_type`, it is extracted; otherwise, `void` is used.
 */
template<typename T, typename = void>
struct computational_dag_edge_desc_traits {
    using edge_comm_weight_type = void;
};

template<typename T>
struct computational_dag_edge_desc_traits<T, std::void_t<typename T::edge_comm_weight_type>> {
    using edge_comm_weight_type = typename T::edge_comm_weight_type;
};

template<typename T>
using e_commw_t = typename computational_dag_edge_desc_traits<T>::edge_comm_weight_type;

// -----------------------------------------------------------------------------
// Property Traits
// -----------------------------------------------------------------------------

/**
 * @brief Check if a graph guarantees vertices are stored/iterated in topological order.
 * It allows a graph implementation to notify algorithms that vertices are stored/iterated in topological order which can be used to optimize the algorithm.
 */
template<typename T, typename = void>
struct has_vertices_in_top_order_trait : std::false_type {};

template<typename T>
struct has_vertices_in_top_order_trait<T, std::void_t<decltype(T::vertices_in_top_order)>>
    : std::bool_constant<std::is_same_v<decltype(T::vertices_in_top_order), const bool> && T::vertices_in_top_order> {};

template<typename T>
inline constexpr bool has_vertices_in_top_order_v = has_vertices_in_top_order_trait<T>::value;

/**
 * @brief Check if a graph guarantees children of a vertex are stored/iterated in vertex index order.
 */
template<typename T, typename = void>
struct has_children_in_vertex_order_trait : std::false_type {};

template<typename T>
struct has_children_in_vertex_order_trait<T, std::void_t<decltype(T::children_in_vertex_order)>>
    : std::bool_constant<std::is_same_v<decltype(T::children_in_vertex_order), const bool> && T::children_in_vertex_order> {};

template<typename T>
inline constexpr bool has_children_in_vertex_order_v = has_children_in_vertex_order_trait<T>::value;

/**
 * @brief Check if a graph guarantees parents of a vertex are stored/iterated in vertex index order.
 */
template<typename T, typename = void>
struct has_parents_in_vertex_order_trait : std::false_type {};

template<typename T>
struct has_parents_in_vertex_order_trait<T, std::void_t<decltype(T::parents_in_vertex_order)>>
    : std::bool_constant<std::is_same_v<decltype(T::parents_in_vertex_order), const bool> && T::parents_in_vertex_order> {};

template<typename T>
inline constexpr bool has_parents_in_vertex_order_v = has_parents_in_vertex_order_trait<T>::value;

} // namespace osp

/**
 * @brief Specialization of std::hash for osp::directed_edge.
 *
 * This specialization provides a hash function for osp::directed_edge, which is used in hash-based containers like std::unordered_set and std::unordered_map.
 */
template<typename Graph_t>
struct std::hash<osp::directed_edge<Graph_t>> {
    std::size_t operator()(const osp::directed_edge<Graph_t> &p) const noexcept {
        // Combine hashes of source and target
        std::size_t h1 = std::hash<osp::vertex_idx_t<Graph_t>>{}(p.source);
        std::size_t h2 = std::hash<osp::vertex_idx_t<Graph_t>>{}(p.target);
        return h1 ^ (h2 << 1); // Simple hash combining
    }
};