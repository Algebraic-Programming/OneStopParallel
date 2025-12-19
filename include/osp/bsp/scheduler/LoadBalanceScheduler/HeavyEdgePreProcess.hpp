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

#include "osp/auxiliary/datastructures/union_find.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

template <typename GraphT>
std::vector<std::vector<VertexIdxT<GraphT>>> HeavyEdgePreprocess(const GraphT &graph,
                                                                 const double heavyIsXTimesMedian,
                                                                 const double minPercentComponentsRetained,
                                                                 const double boundComponentWeightPercent) {
    static_assert(isComputationalDagEdgeDescV<GraphT>,
                  "HeavyEdgePreProcess can only be used with computational DAGs with edge weights.");

    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

    // Initialising the union find structure
    UnionFindUniverseT<GraphT> ufStructure;
    for (const VertexType &vert : graph.Vertices()) {
        ufStructure.AddObject(vert, graph.VertexWorkWeight(vert));
    }

    // Making edge comunications list
    std::vector<ECommwT<GraphT>> edgeCommunications;
    edgeCommunications.reserve(graph.NumEdges());
    for (const auto &edge : Edges(graph)) {
        if constexpr (hasEdgeWeightsV<GraphT>) {
            edgeCommunications.emplace_back(graph.EdgeCommWeight(edge));
        } else {
            edgeCommunications.emplace_back(graph.VertexCommWeight(Source(edge, graph)));
        }
    }

    // Computing the median and setting it to at least one
    ECommwT<GraphT> medianEdgeWeight = 1;
    if (not edgeCommunications.empty()) {
        auto medianIt = edgeCommunications.begin();
        std::advance(medianIt, edgeCommunications.size() / 2);
        std::nth_element(edgeCommunications.begin(), medianIt, edgeCommunications.end());
        medianEdgeWeight = std::max(edgeCommunications[edgeCommunications.size() / 2], static_cast<ECommwT<GraphT>>(1));
    }

    // Making edge list
    ECommwT<GraphT> minimalEdgeWeight = static_cast<ECommwT<GraphT>>(heavyIsXTimesMedian * medianEdgeWeight);
    std::vector<EdgeType> edgeList;
    edgeList.reserve(graph.NumEdges());
    for (const auto &edge : Edges(graph)) {
        if constexpr (hasEdgeWeightsV<GraphT>) {
            if (graph.EdgeCommWeight(edge) > minimalEdgeWeight) {
                edgeList.emplace_back(edge);
            }
        } else {
            if (graph.VertexCommWeight(Source(edge, graph)) > minimalEdgeWeight) {
                edgeList.emplace_back(edge);
            }
        }
    }

    if constexpr (hasEdgeWeightsV<GraphT>) {
        // Sorting edge list
        std::sort(edgeList.begin(), edgeList.end(), [graph](const EdgeType &left, const EdgeType &right) {
            return graph.EdgeCommWeight(left) > graph.EdgeCommWeight(right);
        });
    } else {
        std::sort(edgeList.begin(), edgeList.end(), [graph](const EdgeType &left, const EdgeType &right) {
            return graph.VertexCommWeight(Source(left, graph)) > graph.VertexCommWeight(Source(right, graph));
        });
    }

    // Computing max component size
    VWorkwT<GraphT> maxComponentSize = 0;
    for (const VertexType &vert : graph.Vertices()) {
        maxComponentSize += graph.VertexWorkWeight(vert);
    }

    maxComponentSize = static_cast<VWorkwT<GraphT>>(maxComponentSize * boundComponentWeightPercent);

    // Joining heavy edges
    for (const EdgeType &edge : edgeList) {
        if (static_cast<double>(ufStructure.GetNumberOfConnectedComponents()) - 1.0
            < minPercentComponentsRetained * static_cast<double>(graph.NumVertices())) {
            break;
        }

        VWorkwT<GraphT> weightCompA = ufStructure.GetWeightOfComponentByName(Source(edge, graph));
        VWorkwT<GraphT> weightCompB = ufStructure.GetWeightOfComponentByName(Target(edge, graph));
        if (weightCompA + weightCompB > maxComponentSize) {
            continue;
        }

        ufStructure.JoinByName(Source(edge, graph), Target(edge, graph));
    }

    return ufStructure.GetConnectedComponents();
}

}    // namespace osp
