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
 * - `SetVertexWorkWeight(v, w)`
 * - `SetVertexCommWeight(v, w)`
 * - `SetVertexMemWeight(v, w)`
 *
 * Also requires the graph to be default constructible, copy/move constructible, and assignable.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsModifiableCdagVertex : std::false_type {};

template <typename T>
struct IsModifiableCdagVertex<
    T,
    std::void_t<decltype(std::declval<T>().SetVertexWorkWeight(std::declval<VertexIdxT<T>>(), std::declval<VWorkwT<T>>())),
                decltype(std::declval<T>().SetVertexCommWeight(std::declval<VertexIdxT<T>>(), std::declval<VCommwT<T>>())),
                decltype(std::declval<T>().SetVertexMemWeight(std::declval<VertexIdxT<T>>(), std::declval<VMemwT<T>>()))>>
    : std::conjunction<IsComputationalDag<T>,
                       std::is_default_constructible<T>,
                       std::is_copy_constructible<T>,
                       std::is_move_constructible<T>,
                       std::is_copy_assignable<T>,
                       std::is_move_assignable<T>> {};

template <typename T>
inline constexpr bool IsModifiableCdagVertexV = IsModifiableCdagVertex<T>::value;

/**
 * @brief Concept to check if vertices can be added to the graph.
 *
 * Requires:
 * - `AddVertex(work_weight, comm_weight, mem_weight)`
 * - Constructibility from `vertex_idx_t` (for reserving size).
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsConstructableCdagVertex : std::false_type {};

template <typename T>
struct IsConstructableCdagVertex<T,
                                 std::void_t<decltype(std::declval<T>().AddVertex(
                                     std::declval<VWorkwT<T>>(), std::declval<VCommwT<T>>(), std::declval<VMemwT<T>>()))>>
    : std::conjunction<IsModifiableCdagVertex<T>, std::is_constructible<T, VertexIdxT<T>>> {};

template <typename T>
inline constexpr bool IsConstructableCdagVertexV = IsConstructableCdagVertex<T>::value;

/**
 * @brief Concept to check if vertex types are modifiable.
 *
 * Requires:
 * - `SetVertexType(v, type)`
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsModifiableCdagTypedVertex : std::false_type {};

template <typename T>
struct IsModifiableCdagTypedVertex<
    T,
    std::void_t<decltype(std::declval<T>().SetVertexType(std::declval<VertexIdxT<T>>(), std::declval<VTypeT<T>>()))>>
    : std::conjunction<IsModifiableCdagVertex<T>, IsComputationalDagTypedVertices<T>> {};    // for default node type

template <typename T>
inline constexpr bool IsModifiableCdagTypedVertexV = IsModifiableCdagTypedVertex<T>::value;

/**
 * @brief Concept to check if typed vertices can be added.
 *
 * Requires:
 * - `AddVertex(work, comm, mem, type)`
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsConstructableCdagTypedVertex : std::false_type {};

template <typename T>
struct IsConstructableCdagTypedVertex<
    T,
    std::void_t<decltype(std::declval<T>().AddVertex(
        std::declval<VWorkwT<T>>(), std::declval<VCommwT<T>>(), std::declval<VMemwT<T>>(), std::declval<VTypeT<T>>()))>>
    : std::conjunction<IsConstructableCdagVertex<T>, IsModifiableCdagTypedVertex<T>> {};    // for default node type

template <typename T>
inline constexpr bool IsConstructableCdagTypedVertexV = IsConstructableCdagTypedVertex<T>::value;

/**
 * @brief Concept to check if edges can be added (unweighted).
 *
 * Requires:
 * - `add_edge(source, target)`
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsConstructableCdagEdge : std::false_type {};

template <typename T>
struct IsConstructableCdagEdge<
    T,
    std::void_t<decltype(std::declval<T>().AddEdge(std::declval<VertexIdxT<T>>(), std::declval<VertexIdxT<T>>()))>>
    : IsDirectedGraph<T> {};

template <typename T>
inline constexpr bool IsConstructableCdagEdgeV = IsConstructableCdagEdge<T>::value;

/**
 * @brief Concept to check if edge communication weights are modifiable.
 *
 * Requires:
 * - `SetEdgeCommWeight(edge, weight)`
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsModifiableCdagCommEdge : std::false_type {};

template <typename T>
struct IsModifiableCdagCommEdge<
    T,
    std::void_t<decltype(std::declval<T>().SetEdgeCommWeight(std::declval<EdgeDescT<T>>(), std::declval<ECommwT<T>>()))>>
    : std::conjunction<IsComputationalDagEdgeDesc<T>> {};    // for default edge weight

template <typename T>
inline constexpr bool IsModifiableCdagCommEdgeV = IsModifiableCdagCommEdge<T>::value;

/**
 * @brief Concept to check if weighted edges can be added.
 *
 * Requires:
 * - `add_edge(source, target, weight)`
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsConstructableCdagCommEdge : std::false_type {};

template <typename T>
struct IsConstructableCdagCommEdge<T,
                                   std::void_t<decltype(std::declval<T>().AddEdge(
                                       std::declval<VertexIdxT<T>>(), std::declval<VertexIdxT<T>>(), std::declval<ECommwT<T>>()))>>
    : std::conjunction<IsConstructableCdagEdge<T>, IsComputationalDagEdgeDesc<T>, IsModifiableCdagCommEdge<T>> {
};    // for default edge weight

template <typename T>
inline constexpr bool IsConstructableCdagCommEdgeV = IsConstructableCdagCommEdge<T>::value;

/**
 * @brief Concept for a fully constructable computational DAG.
 *
 * Combines `is_constructable_cdag_vertex` and `is_constructable_cdag_edge`.
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct IsConstructableCdag : std::false_type {};

template <typename T>
struct IsConstructableCdag<T, std::void_t<>>
    : std::conjunction<IsComputationalDag<T>, IsConstructableCdagVertex<T>, IsConstructableCdagEdge<T>> {};

template <typename T>
inline constexpr bool IsConstructableCdagV = IsConstructableCdag<T>::value;

/**
 * @brief Helper trait to check if a graph can be directly constructed from a vertex count and a set of edges.
 */
template <typename T>
inline constexpr bool IsDirectConstructableCdagV
    = std::is_constructible<T, VertexIdxT<T>, std::set<std::pair<VertexIdxT<T>, VertexIdxT<T>>>>::value;

}    // namespace osp
