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

#include "directed_graph_concept.hpp"
#include "graph_traits.hpp"
#include "osp/graph_algorithms/directed_graph_edge_view.hpp"

/**
 * @file directed_graph_edge_desc_concept.hpp
 * @brief Concepts and default implementations for edge descriptors in directed graphs.
 *
 * This file extends the basic directed graph concepts to support edge descriptors.
 * It provides default implementations for accessing source/target vertices from edges
 * and defines the `is_directed_graph_edge_desc` concept to check if a graph type
 * properly supports edge descriptors and edge iteration.
 *
 * Note: If a graph implementation satisfies `directed_graph_concept`, OSP automatically
 * adds the edge descriptor API on top using `directed_edge` as the default edge descriptor.
 * The mechanics for this are implemented in `directed_graph_edge_view.hpp`.
 */

namespace osp {

/**
 * @brief Default implementation to get the source vertex of an edge.
 *
 * @tparam Graph_t The graph type.
 * @param edge The edge descriptor.
 * @return The source vertex index.
 */
template <typename Graph_t>
inline vertex_idx_t<Graph_t> source(const directed_edge<Graph_t> &edge, const Graph_t &) {
    return edge.source;
}

/**
 * @brief Default implementation to get the target vertex of an edge.
 *
 * @tparam Graph_t The graph type.
 * @param edge The edge descriptor.
 * @return The target vertex index.
 */
template <typename Graph_t>
inline vertex_idx_t<Graph_t> target(const directed_edge<Graph_t> &edge, const Graph_t &) {
    return edge.target;
}

/**
 * @brief Get a view of all edges in the graph.
 *
 * @tparam Graph_t The graph type.
 * @param graph The graph instance.
 * @return An `edge_view` allowing iteration over all edges.
 */
template <typename Graph_t>
inline edge_view<Graph_t> edges(const Graph_t &graph) {
    return edge_view(graph);
}

/**
 * @brief Get a view of outgoing edges from a vertex.
 *
 * @tparam Graph_t The graph type.
 * @param u The source vertex index.
 * @param graph The graph instance.
 * @return An `out_edge_view` allowing iteration over outgoing edges from `u`.
 */
template <typename Graph_t>
inline OutEdgeView<Graph_t> out_edges(vertex_idx_t<Graph_t> u, const Graph_t &graph) {
    return OutEdgeView<Graph_t>(graph, u);
}

/**
 * @brief Get a view of incoming edges to a vertex.
 *
 * @tparam Graph_t The graph type.
 * @param v The target vertex index.
 * @param graph The graph instance.
 * @return An `in_edge_view` allowing iteration over incoming edges to `v`.
 */
template <typename Graph_t>
inline InEdgeView<Graph_t> in_edges(vertex_idx_t<Graph_t> v, const Graph_t &graph) {
    return InEdgeView<Graph_t>(graph, v);
}

/**
 * @brief Concept check for a directed graph with edge descriptors.
 *
 * Checks if a type `T` satisfies the requirements of a directed graph that also
 * supports edge descriptors, including:
 * - Validity of `directed_graph_edge_desc_traits`.
 * - Existence of `edges()`, `out_edges()`, and `in_edges()` functions returning input ranges of edge descriptors.
 * - Existence of `source()` and `target()` functions mapping edge descriptors to vertex indices.
 * - Default and copy constructibility of the edge descriptor type.
 *
 * @tparam T The graph type to check.
 */
template <typename T, typename = void>
struct is_directed_graph_edge_desc : std::false_type {};

template <typename T>
struct is_directed_graph_edge_desc<T,
                                   std::void_t<typename directed_graph_edge_desc_traits<T>::directed_edge_descriptor,
                                               decltype(edges(std::declval<T>())),
                                               decltype(out_edges(std::declval<vertex_idx_t<T>>(), std::declval<T>())),
                                               decltype(in_edges(std::declval<vertex_idx_t<T>>(), std::declval<T>())),
                                               decltype(source(std::declval<edge_desc_t<T>>(), std::declval<T>())),
                                               decltype(target(std::declval<edge_desc_t<T>>(), std::declval<T>()))>>
    : std::conjunction<is_directed_graph<T>,
                       std::is_default_constructible<edge_desc_t<T>>,
                       std::is_copy_constructible<edge_desc_t<T>>,
                       is_input_range_of<decltype(edges(std::declval<T>())), edge_desc_t<T>>,
                       is_input_range_of<decltype(out_edges(std::declval<vertex_idx_t<T>>(), std::declval<T>())), edge_desc_t<T>>,
                       is_input_range_of<decltype(in_edges(std::declval<vertex_idx_t<T>>(), std::declval<T>())), edge_desc_t<T>>,
                       std::is_same<decltype(source(std::declval<edge_desc_t<T>>(), std::declval<T>())), vertex_idx_t<T>>,
                       std::is_same<decltype(target(std::declval<edge_desc_t<T>>(), std::declval<T>())), vertex_idx_t<T>>> {};

template <typename T>
inline constexpr bool is_directed_graph_edge_desc_v = is_directed_graph_edge_desc<T>::value;

/**
 * @brief Specialization for graphs that define a directed_edge_descriptor that can be used as a key in a hash table.
 *
 * Compatible with STL hash tables (requires `std::hash` specialization and equality operator).
 *
 * @tparam T The graph type.
 */
template <typename T, typename = void>
struct has_hashable_edge_desc : std::false_type {};

template <typename T>
struct has_hashable_edge_desc<T,
                              std::void_t<decltype(std::hash<edge_desc_t<T>>{}(std::declval<edge_desc_t<T>>())),
                                          decltype(std::declval<edge_desc_t<T>>() == std::declval<edge_desc_t<T>>()),
                                          decltype(std::declval<edge_desc_t<T>>() != std::declval<edge_desc_t<T>>())>>
    : std::conjunction<is_directed_graph_edge_desc<T>,
                       std::is_default_constructible<edge_desc_t<T>>,
                       std::is_copy_constructible<edge_desc_t<T>>> {};

template <typename T>
inline constexpr bool has_hashable_edge_desc_v = has_hashable_edge_desc<T>::value;

}    // namespace osp
