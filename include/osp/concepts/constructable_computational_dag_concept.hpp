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

#include <set>

#include "computational_dag_concept.hpp"

/**
 * @file constructable_computational_dag_concept.hpp
 * @brief Concepts for Constructable and Modifiable Computational DAGs.
 *
 * This file defines concepts that validate whether a graph type supports dynamic construction
 * and modification of its structure and properties. This includes adding vertices and edges,
 * as well as setting weights and types for existing elements.
 *
 * These concepts are useful for algorithms that need to build or transform graphs,
 * such as graph generators or coarsening algorithms.
 */

namespace osp {

/**
 * @brief Concept to check if vertex weights are modifiable.
 *
 * Requires:
 * - `set_vertex_work_weight(v, w)`
 * - `set_vertex_comm_weight(v, w)`
 * - `set_vertex_mem_weight(v, w)`
 *
 * Also requires the graph to be default constructible, copy/move constructible, and assignable.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct is_modifiable_cdag_vertex : std::false_type {};

template <typename T>
struct is_modifiable_cdag_vertex<
    T,
    std::void_t<decltype(std::declval<T>().set_vertex_work_weight(std::declval<vertex_idx_t<T>>(), std::declval<v_workw_t<T>>())),
                decltype(std::declval<T>().set_vertex_comm_weight(std::declval<vertex_idx_t<T>>(), std::declval<v_commw_t<T>>())),
                decltype(std::declval<T>().set_vertex_mem_weight(std::declval<vertex_idx_t<T>>(), std::declval<v_memw_t<T>>()))>>
    : std::conjunction<is_computational_dag<T>,
                       std::is_default_constructible<T>,
                       std::is_copy_constructible<T>,
                       std::is_move_constructible<T>,
                       std::is_copy_assignable<T>,
                       std::is_move_assignable<T>> {};

template <typename T>
inline constexpr bool is_modifiable_cdag_vertex_v = is_modifiable_cdag_vertex<T>::value;

/**
 * @brief Concept to check if vertices can be added to the graph.
 *
 * Requires:
 * - `add_vertex(work_weight, comm_weight, mem_weight)`
 * - Constructibility from `vertex_idx_t` (for reserving size).
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct is_constructable_cdag_vertex : std::false_type {};

template <typename T>
struct is_constructable_cdag_vertex<T,
                                    std::void_t<decltype(std::declval<T>().add_vertex(
                                        std::declval<v_workw_t<T>>(), std::declval<v_commw_t<T>>(), std::declval<v_memw_t<T>>()))>>
    : std::conjunction<is_modifiable_cdag_vertex<T>, std::is_constructible<T, vertex_idx_t<T>>> {};

template <typename T>
inline constexpr bool is_constructable_cdag_vertex_v = is_constructable_cdag_vertex<T>::value;

/**
 * @brief Concept to check if vertex types are modifiable.
 *
 * Requires:
 * - `set_vertex_type(v, type)`
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct is_modifiable_cdag_typed_vertex : std::false_type {};

template <typename T>
struct is_modifiable_cdag_typed_vertex<
    T,
    std::void_t<decltype(std::declval<T>().set_vertex_type(std::declval<vertex_idx_t<T>>(), std::declval<v_type_t<T>>()))>>
    : std::conjunction<is_modifiable_cdag_vertex<T>, is_computational_dag_typed_vertices<T>> {};    // for default node type

template <typename T>
inline constexpr bool is_modifiable_cdag_typed_vertex_v = is_modifiable_cdag_typed_vertex<T>::value;

/**
 * @brief Concept to check if typed vertices can be added.
 *
 * Requires:
 * - `add_vertex(work, comm, mem, type)`
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct is_constructable_cdag_typed_vertex : std::false_type {};

template <typename T>
struct is_constructable_cdag_typed_vertex<
    T,
    std::void_t<decltype(std::declval<T>().add_vertex(
        std::declval<v_workw_t<T>>(), std::declval<v_commw_t<T>>(), std::declval<v_memw_t<T>>(), std::declval<v_type_t<T>>()))>>
    : std::conjunction<is_constructable_cdag_vertex<T>, is_modifiable_cdag_typed_vertex<T>> {};    // for default node type

template <typename T>
inline constexpr bool is_constructable_cdag_typed_vertex_v = is_constructable_cdag_typed_vertex<T>::value;

/**
 * @brief Concept to check if edges can be added (unweighted).
 *
 * Requires:
 * - `add_edge(source, target)`
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct is_constructable_cdag_edge : std::false_type {};

template <typename T>
struct is_constructable_cdag_edge<
    T,
    std::void_t<decltype(std::declval<T>().add_edge(std::declval<vertex_idx_t<T>>(), std::declval<vertex_idx_t<T>>()))>>
    : is_directed_graph<T> {};

template <typename T>
inline constexpr bool is_constructable_cdag_edge_v = is_constructable_cdag_edge<T>::value;

/**
 * @brief Concept to check if edge communication weights are modifiable.
 *
 * Requires:
 * - `set_edge_comm_weight(edge, weight)`
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct is_modifiable_cdag_comm_edge : std::false_type {};

template <typename T>
struct is_modifiable_cdag_comm_edge<
    T,
    std::void_t<decltype(std::declval<T>().set_edge_comm_weight(std::declval<edge_desc_t<T>>(), std::declval<e_commw_t<T>>()))>>
    : std::conjunction<is_computational_dag_edge_desc<T>> {};    // for default edge weight

template <typename T>
inline constexpr bool is_modifiable_cdag_comm_edge_v = is_modifiable_cdag_comm_edge<T>::value;

/**
 * @brief Concept to check if weighted edges can be added.
 *
 * Requires:
 * - `add_edge(source, target, weight)`
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct is_constructable_cdag_comm_edge : std::false_type {};

template <typename T>
struct is_constructable_cdag_comm_edge<
    T,
    std::void_t<decltype(std::declval<T>().add_edge(
        std::declval<vertex_idx_t<T>>(), std::declval<vertex_idx_t<T>>(), std::declval<e_commw_t<T>>()))>>
    : std::conjunction<is_constructable_cdag_edge<T>, is_computational_dag_edge_desc<T>, is_modifiable_cdag_comm_edge<T>> {
};    // for default edge weight

template <typename T>
inline constexpr bool is_constructable_cdag_comm_edge_v = is_constructable_cdag_comm_edge<T>::value;

/**
 * @brief Concept for a fully constructable computational DAG.
 *
 * Combines `is_constructable_cdag_vertex` and `is_constructable_cdag_edge`.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct is_constructable_cdag : std::false_type {};

template <typename T>
struct is_constructable_cdag<T, std::void_t<>>
    : std::conjunction<is_computational_dag<T>, is_constructable_cdag_vertex<T>, is_constructable_cdag_edge<T>> {};

template <typename T>
inline constexpr bool is_constructable_cdag_v = is_constructable_cdag<T>::value;

/**
 * @brief Helper trait to check if a graph can be directly constructed from a vertex count and a set of edges.
 */
template <typename T>
inline constexpr bool is_direct_constructable_cdag_v
    = std::is_constructible<T, vertex_idx_t<T>, std::set<std::pair<vertex_idx_t<T>, vertex_idx_t<T>>>>::value;

}    // namespace osp
