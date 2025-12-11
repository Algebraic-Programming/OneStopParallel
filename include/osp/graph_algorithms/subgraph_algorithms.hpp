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

#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/concepts/directed_graph_concept.hpp"

namespace osp {

template <typename GraphTIn, typename GraphTOut>
void CreateInducedSubgraph(const GraphTIn &dag,
                           GraphTOut &dagOut,
                           const std::set<VertexIdxT<GraphTIn>> &selectedNodes,
                           const std::set<VertexIdxT<GraphTIn>> &extraSources = {}) {
    static_assert(std::is_same_v<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>>,
                  "Graph_t_in and out must have the same vertex_idx types");

    static_assert(isConstructableCdagVertexV<GraphTOut>, "Graph_t_out must satisfy the constructable_cdag_vertex concept");

    static_assert(isConstructableCdagEdgeV<GraphTOut>, "Graph_t_out must satisfy the constructable_cdag_edge concept");

    assert(dagOut.NumVertices() == 0);

    std::map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTIn>> localIdx;

    for (const auto &node : extraSources) {
        localIdx[node] = dagOut.NumVertices();
        if constexpr (isConstructableCdagTypedVertexV<GraphTOut> and hasTypedVerticesV<GraphTIn>) {
            // add extra source with type
            dagOut.AddVertex(0, dag.VertexCommWeight(node), dag.VertexMemWeight(node), dag.VertexType(node));
        } else {
            // add extra source without type
            dagOut.add_vertex(0, dag.vertex_comm_weight(node), dag.vertex_mem_weight(node));
        }
    }

    for (const auto &node : selectedNodes) {
        localIdx[node] = dagOut.NumVertices();

        if constexpr (isConstructableCdagTypedVertexV<GraphTOut> and hasTypedVerticesV<GraphTIn>) {
            // add vertex with type
            dagOut.AddVertex(
                dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node), dag.VertexType(node));
        } else {
            // add vertex without type
            dagOut.add_vertex(dag.vertex_work_weight(node), dag.vertex_comm_weight(node), dag.vertex_mem_weight(node));
        }
    }

    if constexpr (hasEdgeWeightsV<GraphTIn> and hasEdgeWeightsV<GraphTOut>) {
        // add edges with edge comm weights
        for (const auto &node : selectedNodes) {
            for (const auto &inEdge : in_edges(node, dag)) {
                const auto &pred = source(inEdge, dag);
                if (selectedNodes.find(pred) != selectedNodes.end() || extraSources.find(pred) != extraSources.end()) {
                    dagOut.add_edge(localIdx[pred], localIdx[node], dag.edge_comm_weight(inEdge));
                }
            }
        }

    } else {
        // add edges without edge comm weights
        for (const auto &node : selectedNodes) {
            for (const auto &pred : dag.Parents(node)) {
                if (selectedNodes.find(pred) != selectedNodes.end() || extraSources.find(pred) != extraSources.end()) {
                    dagOut.AddEdge(localIdx[pred], localIdx[node]);
                }
            }
        }
    }
}

template <typename GraphTIn, typename GraphTOut>
void CreateInducedSubgraph(const GraphTIn &dag, GraphTOut &dagOut, const std::vector<VertexIdxT<GraphTIn>> &selectedNodes) {
    return CreateInducedSubgraph(dag, dagOut, std::set<VertexIdxT<GraphTIn>>(selectedNodes.begin(), selectedNodes.end()));
}

template <typename GraphT>
bool CheckOrderedIsomorphism(const GraphT &first, const GraphT &second) {
    static_assert(is_directed_graph_v<GraphT>, "Graph_t must satisfy the directed_graph concept");

    if (first.num_vertices() != second.num_vertices() || first.num_edges() != second.num_edges()) {
        return false;
    }

    for (const auto &node : first.vertices()) {
        if (first.vertex_work_weight(node) != second.vertex_work_weight(node)
            || first.vertex_mem_weight(node) != second.vertex_mem_weight(node)
            || first.vertex_comm_weight(node) != second.vertex_comm_weight(node)
            || first.vertex_type(node) != second.vertex_type(node)) {
            return false;
        }

        if (first.in_degree(node) != second.in_degree(node) || first.out_degree(node) != second.out_degree(node)) {
            return false;
        }

        if constexpr (has_edge_weights_v<GraphT>) {
            std::set<std::pair<VertexIdxT<GraphT>, ECommwT<GraphT>>> firstChildren, secondChildren;

            for (const auto &outEdge : out_edges(node, first)) {
                firstChildren.emplace(target(outEdge, first), first.edge_comm_weight(outEdge));
            }

            for (const auto &outEdge : out_edges(node, second)) {
                secondChildren.emplace(target(outEdge, second), second.edge_comm_weight(outEdge));
            }

            auto itr = firstChildren.begin(), secondItr = secondChildren.begin();
            for (; itr != firstChildren.end() && secondItr != secondChildren.end(); ++itr) {
                if (*itr != *secondItr) {
                    return false;
                }
                ++secondItr;
            }

        } else {
            std::set<VertexIdxT<GraphT>> firstChildren, secondChildren;

            for (const auto &child : first.children(node)) {
                firstChildren.emplace(child);
            }

            for (const auto &child : second.children(node)) {
                secondChildren.emplace(child);
            }

            auto itr = firstChildren.begin(), secondItr = secondChildren.begin();
            for (; itr != firstChildren.end() && secondItr != secondChildren.end(); ++itr) {
                if (*itr != *secondItr) {
                    return false;
                }
                ++secondItr;
            }
        }
    }

    return true;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<GraphTOut> CreateInducedSubgraphs(const GraphTIn &dagIn, const std::vector<unsigned> &partitionIDs) {
    // assumes that input partition IDs are consecutive and starting from 0

    static_assert(std::is_same_v<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>>,
                  "Graph_t_in and out must have the same vertex_idx types");

    static_assert(isConstructableCdagVertexV<GraphTOut>, "Graph_t_out must satisfy the constructable_cdag_vertex concept");

    static_assert(isConstructableCdagEdgeV<GraphTOut>, "Graph_t_out must satisfy the constructable_cdag_edge concept");

    unsigned numberOfParts = 0;
    for (const auto id : partitionIDs) {
        numberOfParts = std::max(numberOfParts, id + 1);
    }

    std::vector<GraphTOut> splitDags(numberOfParts);

    std::vector<VertexIdxT<GraphTOut>> localIdx(dagIn.NumVertices());

    for (const auto node : dagIn.Vertices()) {
        localIdx[node] = splitDags[partitionIDs[node]].NumVertices();

        if constexpr (isConstructableCdagTypedVertexV<GraphTOut> and hasTypedVerticesV<GraphTIn>) {
            splitDags[partitionIDs[node]].AddVertex(
                dagIn.VertexWorkWeight(node), dagIn.vertex_comm_weight(node), dagIn.VertexMemWeight(node), dagIn.vertex_type(node));
        } else {
            splitDags[partitionIDs[node]].add_vertex(
                dagIn.vertex_work_weight(node), dagIn.vertex_comm_weight(node), dagIn.vertex_mem_weight(node));
        }
    }

    if constexpr (hasEdgeWeightsV<GraphTIn> and hasEdgeWeightsV<GraphTOut>) {
        for (const auto node : dagIn.vertices()) {
            for (const auto &outEdge : out_edges(node, dagIn)) {
                auto succ = target(outEdge, dagIn);

                if (partitionIDs[node] == partitionIDs[succ]) {
                    splitDags[partitionIDs[node]].add_edge(localIdx[node], localIdx[succ], dagIn.edge_comm_weight(outEdge));
                }
            }
        }
    } else {
        for (const auto node : dagIn.Vertices()) {
            for (const auto &child : dagIn.Children(node)) {
                if (partitionIDs[node] == partitionIDs[child]) {
                    splitDags[partitionIDs[node]].AddEdge(localIdx[node], localIdx[child]);
                }
            }
        }
    }

    return splitDags;
}

template <typename GraphTIn, typename GraphTOut>
std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTIn>> CreateInducedSubgraphMap(
    const GraphTIn &dag, GraphTOut &dagOut, const std::vector<VertexIdxT<GraphTIn>> &selectedNodes) {
    static_assert(std::is_same_v<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>>,
                  "Graph_t_in and out must have the same vertex_idx types");

    static_assert(isConstructableCdagVertexV<GraphTOut>, "Graph_t_out must satisfy the constructable_cdag_vertex concept");

    static_assert(isConstructableCdagEdgeV<GraphTOut>, "Graph_t_out must satisfy the constructable_cdag_edge concept");

    assert(dagOut.NumVertices() == 0);

    std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTIn>> localIdx;
    localIdx.reserve(selectedNodes.size());

    for (const auto &node : selectedNodes) {
        localIdx[node] = dagOut.NumVertices();

        if constexpr (isConstructableCdagTypedVertexV<GraphTOut> and hasTypedVerticesV<GraphTIn>) {
            // add vertex with type
            dagOut.AddVertex(
                dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node), dag.VertexType(node));
        } else {
            // add vertex without type
            dagOut.add_vertex(dag.vertex_work_weight(node), dag.vertex_comm_weight(node), dag.vertex_mem_weight(node));
        }
    }

    if constexpr (hasEdgeWeightsV<GraphTIn> and hasEdgeWeightsV<GraphTOut>) {
        // add edges with edge comm weights
        for (const auto &node : selectedNodes) {
            for (const auto &inEdge : InEdges(node, dag)) {
                const auto &pred = Source(inEdge, dag);
                if (localIdx.count(pred)) {
                    dagOut.AddEdge(localIdx[pred], localIdx[node], dag.EdgeCommWeight(inEdge));
                }
            }
        }

    } else {
        // add edges without edge comm weights
        for (const auto &node : selectedNodes) {
            for (const auto &pred : dag.Parents(node)) {
                if (localIdx.count(pred)) {
                    dagOut.AddEdge(localIdx[pred], localIdx[node]);
                }
            }
        }
    }

    return localIdx;
}

}    // end namespace osp
