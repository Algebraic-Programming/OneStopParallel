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
                  "GraphTIn and out must have the same VertexIdx types");

    static_assert(IsConstructableCdagVertexV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_vertex concept");

    static_assert(IsConstructableCdagEdgeV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_edge concept");

    assert(dagOut.NumVertices() == 0);

    std::map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTIn>> localIdx;

    for (const auto &node : extraSources) {
        localIdx[node] = dagOut.NumVertices();
        if constexpr (IsConstructableCdagTypedVertexV<GraphTOut> and HasTypedVerticesV<GraphTIn>) {
            // add extra source with type
            dagOut.AddVertex(0, dag.VertexCommWeight(node), dag.VertexMemWeight(node), dag.VertexType(node));
        } else {
            // add extra source without type
            dagOut.AddVertex(0, dag.VertexCommWeight(node), dag.VertexMemWeight(node));
        }
    }

    for (const auto &node : selectedNodes) {
        localIdx[node] = dagOut.NumVertices();

        if constexpr (IsConstructableCdagTypedVertexV<GraphTOut> and HasTypedVerticesV<GraphTIn>) {
            // add vertex with type
            dagOut.AddVertex(
                dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node), dag.VertexType(node));
        } else {
            // add vertex without type
            dagOut.AddVertex(dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node));
        }
    }

    if constexpr (HasEdgeWeightsV<GraphTIn> and HasEdgeWeightsV<GraphTOut>) {
        // add edges with edge comm weights
        for (const auto &node : selectedNodes) {
            for (const auto &inEdge : InEdges(node, dag)) {
                const auto &pred = Source(inEdge, dag);
                if (selectedNodes.find(pred) != selectedNodes.end() || extraSources.find(pred) != extraSources.end()) {
                    dagOut.AddEdge(localIdx[pred], localIdx[node], dag.EdgeCommWeight(inEdge));
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
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");

    if (first.NumVertices() != second.NumVertices() || first.NumEdges() != second.NumEdges()) {
        return false;
    }

    for (const auto &node : first.Vertices()) {
        if (first.VertexWorkWeight(node) != second.VertexWorkWeight(node)
            || first.VertexMemWeight(node) != second.VertexMemWeight(node)
            || first.VertexCommWeight(node) != second.VertexCommWeight(node) || first.VertexType(node) != second.VertexType(node)) {
            return false;
        }

        if (first.InDegree(node) != second.InDegree(node) || first.OutDegree(node) != second.OutDegree(node)) {
            return false;
        }

        if constexpr (HasEdgeWeightsV<GraphT>) {
            std::set<std::pair<VertexIdxT<GraphT>, ECommwT<GraphT>>> firstChildren, secondChildren;

            for (const auto &outEdge : OutEdges(node, first)) {
                firstChildren.emplace(Target(outEdge, first), first.EdgeCommWeight(outEdge));
            }

            for (const auto &outEdge : OutEdges(node, second)) {
                secondChildren.emplace(Target(outEdge, second), second.EdgeCommWeight(outEdge));
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

            for (const auto &child : first.Children(node)) {
                firstChildren.emplace(child);
            }

            for (const auto &child : second.Children(node)) {
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
                  "GraphTIn and out must have the same VertexIdx types");

    static_assert(IsConstructableCdagVertexV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_vertex concept");

    static_assert(IsConstructableCdagEdgeV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_edge concept");

    unsigned numberOfParts = 0;
    for (const auto id : partitionIDs) {
        numberOfParts = std::max(numberOfParts, id + 1);
    }

    std::vector<GraphTOut> splitDags(numberOfParts);

    std::vector<VertexIdxT<GraphTOut>> localIdx(dagIn.NumVertices());

    for (const auto node : dagIn.Vertices()) {
        localIdx[node] = splitDags[partitionIDs[node]].NumVertices();

        if constexpr (IsConstructableCdagTypedVertexV<GraphTOut> and HasTypedVerticesV<GraphTIn>) {
            splitDags[partitionIDs[node]].AddVertex(
                dagIn.VertexWorkWeight(node), dagIn.VertexCommWeight(node), dagIn.VertexMemWeight(node), dagIn.VertexType(node));
        } else {
            splitDags[partitionIDs[node]].AddVertex(
                dagIn.VertexWorkWeight(node), dagIn.VertexCommWeight(node), dagIn.VertexMemWeight(node));
        }
    }

    if constexpr (HasEdgeWeightsV<GraphTIn> and HasEdgeWeightsV<GraphTOut>) {
        for (const auto node : dagIn.Vertices()) {
            for (const auto &outEdge : OutEdges(node, dagIn)) {
                auto succ = Target(outEdge, dagIn);

                if (partitionIDs[node] == partitionIDs[succ]) {
                    splitDags[partitionIDs[node]].AddEdge(localIdx[node], localIdx[succ], dagIn.EdgeCommWeight(outEdge));
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
std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>> CreateInducedSubgraphMap(
    const GraphTIn &dag, GraphTOut &dagOut, const std::vector<VertexIdxT<GraphTIn>> &selectedNodes) {
    static_assert(std::is_same_v<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>>,
                  "GraphTIn and out must have the same VertexIdx types");

    static_assert(IsConstructableCdagVertexV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_vertex concept");

    static_assert(IsConstructableCdagEdgeV<GraphTOut>, "GraphTOut must satisfy the constructable_cdag_edge concept");

    assert(dagOut.NumVertices() == 0);

    std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTOut>> localIdx;
    localIdx.reserve(selectedNodes.size());

    for (const auto &node : selectedNodes) {
        localIdx[node] = dagOut.NumVertices();

        if constexpr (IsConstructableCdagTypedVertexV<GraphTOut> and HasTypedVerticesV<GraphTIn>) {
            // add vertex with type
            dagOut.AddVertex(
                dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node), dag.VertexType(node));
        } else {
            // add vertex without type
            dagOut.AddVertex(dag.VertexWorkWeight(node), dag.VertexCommWeight(node), dag.VertexMemWeight(node));
        }
    }

    if constexpr (HasEdgeWeightsV<GraphTIn> and HasEdgeWeightsV<GraphTOut>) {
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
