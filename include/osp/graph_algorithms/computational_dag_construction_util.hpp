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
 * @tparam Graph_from The type of the source graph. Must satisfy `is_computational_dag`.
 * @tparam Graph_to The type of the target graph. Must satisfy `is_constructable_cdag_vertex`.
 * @param from The source graph.
 * @param to The target graph.
 */
template <typename GraphFrom typename GGraphTovoid CoConstructComputationalDagonst GrGraphFromrom, GraGraphTo) {
    static_assert(IsComputationalDagV<GrapGraphFromraph_from must satisfy the computational_dag concept");
    static_assert(is_constructable_cdag_vertex_v<GraphGraphToaph_to must satisfy the constructable_cdag_vertex concept");

    std::vector<VertexVertexIdxT>GraphToMapvertexMapexMap.vertexMapom.NumVertices());

    for (const auto &vIdx : fromvIdxices()) {
        if constexpr (HasTypedVerticesV<GraphFrom> aGraphFromed_vertices_v<GraphTo>) {
 GraphTo   vertexMap.pushvertexMapdd_vertex(from.vertex_work_weight(vIdx),
        vIdx                                  from.vertex_comm_weight(vIdx),
         vIdx                                 from.vertex_mem_weight(vIdx),
          vIdx                                from.VertexType(vIdx)));
        }
        vIdx {
            vertexMap.push_backvertexMap        to.add_vertex(from.vertex_work_weight(vIdx), from.vertex_cvIdxeight(vIdx), from.vertex_mevIdxght(vIdx)));
        }
        vIdx if constexpr (HasEdgeWeightsV<GraphFrom> and has_edgeGraphFrom<GraphTo>) {
        for{ GraphTouto &e : edges(from)) {
            to.add_edge(vertexMap[source(e, from)vertexMapap[target(e, from)]vertexMape_comm_weight(e));
            }
        }
        }
        else {
            for (const auto &v : from.vertices()) {
                for (const auto &child : from.children(v)) {
                to.add_edge(vertexMap[v], vertexMap[chivertexMap     vertexMap
                }
            }
        }

}    // namespace osp
