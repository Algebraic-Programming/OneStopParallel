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

#include <vector>

#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/concepts/directed_graph_edge_desc_concept.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

/**
 * @brief Computes the transitive reduction of a sparse directed acyclic graph (DAG).
 *
 * The transitive reduction of a DAG is a new graph with the same vertices and reachability,
 * but with the minimum number of edges. It is created by removing every edge (u, v)
 * for which there is an alternative path from u to v of length greater than 1.
 *
 * This implementation iterates through each edge (u, v) of the input graph. For each
 * edge, it checks if any of u's other children, w, can reach v. If such a path
 * u -> w -> ... -> v exists, the original edge (u, v) is considered transitive
 * and is not added to the output graph.
 *
 * This algorithm is efficient for sparse graphs, with a complexity of roughly O(E * (V+E)).
 *
 * @tparam Graph_t_in The type of the input graph. Must satisfy the `is_directed_graph` concept.
 * @tparam Graph_t_out The type of the output graph. Must satisfy the `is_constructable_cdag` concept.
 * @param graph_in The input DAG.
 * @param graph_out The output graph, which will contain the transitive reduction. The graph should be empty.
 */
template <typename Graph_t_in, typename Graph_t_out>
void transitive_reduction_sparse(const Graph_t_in &graph_in, Graph_t_out &graph_out) {
    static_assert(is_directed_graph_v<Graph_t_in>, "Input graph must be a directed graph.");
    static_assert(is_constructable_cdag_v<Graph_t_out>, "Output graph must be a constructable computational DAG.");
    assert(graph_out.num_vertices() == 0 && "Output graph must be empty.");

    if (graph_in.num_vertices() == 0) { return; }

    // 1. Copy vertices and their properties from graph_in to graph_out.
    for (const auto &v_idx : graph_in.vertices()) {
        if constexpr (has_typed_vertices_v<Graph_t_in> && is_constructable_cdag_typed_vertex_v<Graph_t_out>) {
            graph_out.add_vertex(graph_in.vertex_work_weight(v_idx),
                                 graph_in.vertex_comm_weight(v_idx),
                                 graph_in.vertex_mem_weight(v_idx),
                                 graph_in.vertex_type(v_idx));
        } else {
            graph_out.add_vertex(
                graph_in.vertex_work_weight(v_idx), graph_in.vertex_comm_weight(v_idx), graph_in.vertex_mem_weight(v_idx));
        }
    }

    // 2. Add an edge (u, v) to the reduction if it's not transitive.
    // An edge (u, v) is transitive if there exists a child w of u (w != v) that can reach v.
    for (const auto &edge : edges(graph_in)) {
        const auto u = source(edge, graph_in);
        const auto v = target(edge, graph_in);
        bool is_transitive = false;
        for (const auto &w : graph_in.children(u)) {
            if (w != v && has_path(w, v, graph_in)) {
                is_transitive = true;
                break;
            }
        }
        if (!is_transitive) {
            if constexpr (has_edge_weights_v<Graph_t_in> && is_constructable_cdag_comm_edge_v<Graph_t_out>) {
                graph_out.add_edge(u, v, graph_in.edge_comm_weight(edge));
            } else {
                graph_out.add_edge(u, v);
            }
        }
    }
}

/**
 * @brief Computes the transitive reduction of a dense directed acyclic graph (DAG).
 *
 * The transitive reduction of a DAG is a new graph with the same vertices and reachability,
 * but with the minimum number of edges. It is created by removing every edge (u, v)
 * for which there is an alternative path from u to v of length greater than 1.
 *
 * This implementation first computes the transitive closure of the graph using a
 * Floyd-Warshall-like algorithm. Then, for each edge (u, v), it checks if there is an
 * intermediate vertex w such that u can reach w and w can reach v. If so, the edge is
 * transitive and is not included in the output graph.
 *
 * This algorithm is efficient for dense graphs, with a complexity of O(V^3).
 *
 * @tparam Graph_t_in The type of the input graph. Must satisfy the `is_directed_graph_edge_desc` concept.
 * @tparam Graph_t_out The type of the output graph. Must satisfy the `is_constructable_cdag` concept.
 * @param graph_in The input DAG.
 * @param graph_out The output graph, which will contain the transitive reduction. The graph should be empty.
 */
template <typename Graph_t_in, typename Graph_t_out>
void transitive_reduction_dense(const Graph_t_in &graph_in, Graph_t_out &graph_out) {
    static_assert(is_directed_graph_edge_desc_v<Graph_t_in>, "Input graph must be a directed graph with edge descriptors.");
    static_assert(is_constructable_cdag_v<Graph_t_out>, "Output graph must be a constructable computational DAG.");
    assert(graph_out.num_vertices() == 0 && "Output graph must be empty.");

    const auto num_v = graph_in.num_vertices();
    if (num_v == 0) { return; }

    // 1. Copy vertices and their properties from graph_in to graph_out.
    for (const auto &v_idx : graph_in.vertices()) {
        if constexpr (has_typed_vertices_v<Graph_t_in> && is_constructable_cdag_typed_vertex_v<Graph_t_out>) {
            graph_out.add_vertex(graph_in.vertex_work_weight(v_idx),
                                 graph_in.vertex_comm_weight(v_idx),
                                 graph_in.vertex_mem_weight(v_idx),
                                 graph_in.vertex_type(v_idx));
        } else {
            graph_out.add_vertex(
                graph_in.vertex_work_weight(v_idx), graph_in.vertex_comm_weight(v_idx), graph_in.vertex_mem_weight(v_idx));
        }
    }

    // 2. Compute transitive closure (reachability matrix).
    std::vector<std::vector<bool>> reachable(num_v, std::vector<bool>(num_v, false));
    for (const auto &edge : edges(graph_in)) { reachable[source(edge, graph_in)][target(edge, graph_in)] = true; }

    const auto top_order = GetTopOrder(graph_in);
    for (const auto &k : top_order) {
        for (const auto &i : top_order) {
            if (reachable[i][k]) {
                for (const auto &j : top_order) {
                    if (reachable[k][j]) { reachable[i][j] = true; }
                }
            }
        }
    }

    // 3. Add an edge (u, v) to the reduction if it's not transitive.
    for (const auto &edge : edges(graph_in)) {
        const auto u = source(edge, graph_in);
        const auto v = target(edge, graph_in);
        bool is_transitive = false;
        for (const auto &w : graph_in.children(u)) {
            if (w != v && reachable[w][v]) {
                is_transitive = true;
                break;
            }
        }
        if (!is_transitive) {
            if constexpr (has_edge_weights_v<Graph_t_in> && is_constructable_cdag_comm_edge_v<Graph_t_out>) {
                graph_out.add_edge(u, v, graph_in.edge_comm_weight(edge));
            } else {
                graph_out.add_edge(u, v);
            }
        }
    }
}

}    // namespace osp
