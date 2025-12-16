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
 * @tparam GraphTIn The type of the input graph. Must satisfy the `is_directed_graph` concept.
 * @tparam GraphTOut The type of the output graph. Must satisfy the `is_constructable_cdag` concept.
 * @param graph_in The input DAG.
 * @param graph_out The output graph, which will contain the transitive reduction. The graph should be empty.
 */
template <typename GraphTIn, typename GraphTOut>
void TransitiveReductionSparse(const GraphTIn &graphIn, GraphTOut &graphOut) {
    static_assert(IsDirectedGraphV<GraphTIn>, "Input graph must be a directed graph.");
    static_assert(isConstructableCdagV<GraphTOut>, "Output graph must be a constructable computational DAG.");
    assert(graphOut.NumVertices() == 0 && "Output graph must be empty.");

    if (graphIn.NumVertices() == 0) {
        return;
    }

    // 1. Copy vertices and their properties from graph_in to graph_out.
    for (const auto &vIdx : graphIn.Vertices()) {
        if constexpr (hasTypedVerticesV<GraphTIn> && IsConstructableCdagTypedVertexV<GraphTOut>) {
            graphOut.AddVertex(graphIn.VertexWorkWeight(vIdx),
                               graphIn.VertexCommWeight(vIdx),
                               graphIn.VertexMemWeight(vIdx),
                               graphIn.VertexType(vIdx));
        } else {
            graphOut.AddVertex(graphIn.VertexWorkWeight(vIdx), graphIn.VertexCommWeight(vIdx), graphIn.VertexMemWeight(vIdx));
        }
    }

    // 2. Add an edge (u, v) to the reduction if it's not transitive.
    // An edge (u, v) is transitive if there exists a child w of u (w != v) that can reach v.
    for (const auto &edge : Edges(graphIn)) {
        const auto u = Source(edge, graphIn);
        const auto v = Target(edge, graphIn);
        bool isTransitive = false;
        for (const auto &w : graphIn.Children(u)) {
            if (w != v && HasPath(w, v, graphIn)) {
                isTransitive = true;
                break;
            }
        }
        if (!isTransitive) {
            if constexpr (hasEdgeWeightsV<GraphTIn> && isConstructableCdagCommEdgeV<GraphTOut>) {
                graphOut.AddEdge(u, v, graphIn.EdgeCommWeight(edge));
            } else {
                graphOut.AddEdge(u, v);
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
 * @tparam GraphTIn The type of the input graph. Must satisfy the `is_directed_graph_edge_desc` concept.
 * @tparam GraphTOut The type of the output graph. Must satisfy the `is_constructable_cdag` concept.
 * @param graph_in The input DAG.
 * @param graph_out The output graph, which will contain the transitive reduction. The graph should be empty.
 */
template <typename GraphTIn, typename GraphTOut>
void TransitiveReductionDense(const GraphTIn &graphIn, GraphTOut &graphOut) {
    static_assert(IsDirectedGraphEdgeDescV<GraphTIn>, "Input graph must be a directed graph with edge descriptors.");
    static_assert(isConstructableCdagV<GraphTOut>, "Output graph must be a constructable computational DAG.");
    assert(graphOut.NumVertices() == 0 && "Output graph must be empty.");

    const auto numV = graphIn.NumVertices();
    if (numV == 0) {
        return;
    }

    // 1. Copy vertices and their properties from graph_in to graph_out.
    for (const auto &vIdx : graphIn.Vertices()) {
        if constexpr (hasTypedVerticesV<GraphTIn> && IsConstructableCdagTypedVertexV<GraphTOut>) {
            graphOut.AddVertex(graphIn.VertexWorkWeight(vIdx),
                               graphIn.VertexCommWeight(vIdx),
                               graphIn.VertexMemWeight(vIdx),
                               graphIn.VertexType(vIdx));
        } else {
            graphOut.AddVertex(graphIn.VertexWorkWeight(vIdx), graphIn.VertexCommWeight(vIdx), graphIn.VertexMemWeight(vIdx));
        }
    }

    // 2. Compute transitive closure (reachability matrix).
    std::vector<std::vector<bool>> reachable(numV, std::vector<bool>(numV, false));
    for (const auto &edge : Edges(graphIn)) {
        reachable[Source(edge, graphIn)][Target(edge, graphIn)] = true;
    }

    const auto topOrder = GetTopOrder(graphIn);
    for (const auto &k : topOrder) {
        for (const auto &i : topOrder) {
            if (reachable[i][k]) {
                for (const auto &j : topOrder) {
                    if (reachable[k][j]) {
                        reachable[i][j] = true;
                    }
                }
            }
        }
    }

    // 3. Add an edge (u, v) to the reduction if it's not transitive.
    for (const auto &edge : Edges(graphIn)) {
        const auto u = Source(edge, graphIn);
        const auto v = Target(edge, graphIn);
        bool isTransitive = false;
        for (const auto &w : graphIn.Children(u)) {
            if (w != v && reachable[w][v]) {
                isTransitive = true;
                break;
            }
        }
        if (!isTransitive) {
            if constexpr (hasEdgeWeightsV<GraphTIn> && isConstructableCdagCommEdgeV<GraphTOut>) {
                graphOut.AddEdge(u, v, graphIn.EdgeCommWeight(edge));
            } else {
                graphOut.AddEdge(u, v);
            }
        }
    }
}

}    // namespace osp
