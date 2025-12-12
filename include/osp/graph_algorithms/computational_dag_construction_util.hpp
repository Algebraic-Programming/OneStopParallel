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

#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/constructable_computational_dag_concept.hpp"

namespace osp {

/**
 * @brief Constructs a computational DAG from another graph.
 *
 * This function copies the structure and properties of a source graph into a target graph structure.
 * Assumes that the vertices of the source graph are indexed from 0 to N-1. If the target graph is empty, indices are sequentially
 * assigned starting from 0. If the target graph is not empty, new vertices will be added to the target graph and their indices
 * will be sequentially assigned starting from the index N.
 *
 * @tparam GraphFrom The type of the source graph. Must satisfy `is_computational_dag`.
 * @tparam GraphTo The type of the target graph. Must satisfy `is_constructable_cdag_vertex`.
 * @param from The source graph.
 * @param to The target graph.
 */
template <typename GraphFrom, typename GraphTo>
void constructComputationalDag(const GraphFrom &from, GraphTo &to) {
    static_assert(IsComputationalDagV<GraphFrom>, "GraphFrom must satisfy the computational_dag concept");
    static_assert(IsConstructableCdagVertexV<GraphTo>, "GraphTo must satisfy the constructable_cdag_vertex concept");

    std::vector<VertexIdxT<GraphTo>> vertexMap;
    vertexMap.reserve(from.NumVertices());

    for (const auto &vIdx : from.Vertices()) {
        if constexpr (HasTypedVerticesV<GraphFrom> and HasTypedVerticesV<GraphTo>) {
            vertexMap.push_back(to.AddVertex(
                from.VertexWorkWeight(vIdx), from.VertexCommWeight(vIdx), from.VertexMemWeight(vIdx), from.VertexType(vIdx)));
        } else {
            vertexMap.push_back(to.AddVertex(from.VertexWorkWeight(vIdx), from.VertexCommWeight(vIdx), from.VertexMemWeight(vIdx)));
        }
    }

    if constexpr (HasEdgeWeightsV<GraphFrom> and HasEdgeWeightsV<GraphTo>) {
        for (const auto &e : Edges(from)) {
            to.AddEdge(vertexMap[Source(e, from)], vertexMap[Target(e, from)], from.EdgeCommWeight(e));
        }
    } else {
        for (const auto &v : from.Vertices()) {
            for (const auto &child : from.Children(v)) {
                to.AddEdge(vertexMap[v], vertexMap[child]);
            }
        }
    }
}

}    // namespace osp
