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

#include <numeric>

#include "directed_graph_top_sort.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

template <typename GraphT>
VMemwT<GraphT> MaxMemoryWeight(const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    static_assert(hasVertexWeightsV<GraphT>, "Graph_t must have vertex weights");

    VMemwT<GraphT> maxMemoryWeight = 0;

    for (const auto &v : graph.Vertices()) {
        maxMemoryWeight = std::max(maxMemoryWeight, graph.VertexMemWeight(v));
    }
    return maxMemoryWeight;
}

template <typename GraphT>
VMemwT<GraphT> MaxMemoryWeight(const VTypeT<GraphT> &nodeType, const GraphT &graph) {
    static_assert(IsDirectedGraphV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    static_assert(hasVertexWeightsV<GraphT>, "Graph_t must have vertex weights");
    static_assert(hasTypedVerticesV<GraphT>, "Graph_t must have typed vertices");

    VMemwT<GraphT> maxMemoryWeight = 0;

    for (const auto &node : graph.Vertices()) {
        if (graph.VertexType(node) == nodeType) {
            maxMemoryWeight = std::max(maxMemoryWeight, graph.VertexMemWeight(node));
        }
    }
    return maxMemoryWeight;
}

template <typename GraphT, typename VertexIterator>
VWorkwT<GraphT> SumOfVerticesWorkWeights(VertexIterator begin, VertexIterator end, const GraphT &graph) {
    static_assert(hasVertexWeightsV<GraphT>, "Graph_t must have vertex weights");

    return std::accumulate(
        begin, end, 0, [&](const auto sum, const VertexIdxT<GraphT> &v) { return sum + graph.VertexWorkWeight(v); });
}

template <typename GraphT>
VWorkwT<GraphT> SumOfVerticesWorkWeights(const GraphT &graph) {
    static_assert(hasVertexWeightsV<GraphT>, "Graph_t must have vertex weights");

    return std::accumulate(graph.Vertices().begin(),
                           graph.Vertices().end(),
                           static_cast<VWorkwT<GraphT>>(0),
                           [&](const VWorkwT<GraphT> sum, const VertexIdxT<GraphT> &v) { return sum + graph.VertexWorkWeight(v); });
}

template <typename GraphT>
VWorkwT<GraphT> SumOfVerticesWorkWeights(const std::initializer_list<VertexIdxT<GraphT>> vertices, const GraphT &graph) {
    return SumOfVerticesWorkWeights(vertices.begin(), vertices.end(), graph);
}

template <typename VertexIterator, typename GraphT>
VCommwT<GraphT> SumOfVerticesCommunicationWeights(VertexIterator begin, VertexIterator end, const GraphT &graph) {
    static_assert(hasVertexWeightsV<GraphT>, "Graph_t must have vertex weights");
    return std::accumulate(
        begin, end, 0, [&](const auto sum, const VertexIdxT<GraphT> &v) { return sum + graph.VertexCommWeight(v); });
}

/**
 * @brief Calculates the sum of work weights for vertices compatible with a specific processor type.
 * @tparam SubGraph_t The type of the subgraph being analyzed.
 * @tparam Instance_t The type of the instance object (e.g., BspInstance) used for compatibility checks.
 * @tparam VertexIterator An iterator over vertex indices of the subgraph.
 */
template <typename SubGraphT, typename InstanceT, typename VertexIterator>
VWorkwT<SubGraphT> SumOfCompatibleWorkWeights(
    VertexIterator begin, VertexIterator end, const SubGraphT &graph, const InstanceT &mainInstance, unsigned processorType) {
    static_assert(hasVertexWeightsV<SubGraphT>, "SubGraph_t must have vertex weights");
    return std::accumulate(
        begin, end, static_cast<VWorkwT<SubGraphT>>(0), [&](const VWorkwT<SubGraphT> sum, const VertexIdxT<SubGraphT> &v) {
            if (mainInstance.IsCompatibleType(graph.VertexType(v), processorType)) {
                return sum + graph.VertexWorkWeight(v);
            }
            return sum;
        });
}

/**
 * @brief Overload to calculate compatible work weight for all vertices in a graph.
 */
template <typename SubGraphT, typename InstanceT>
VWorkwT<SubGraphT> SumOfCompatibleWorkWeights(const SubGraphT &graph, const InstanceT &mainInstance, unsigned processorType) {
    return SumOfCompatibleWorkWeights(graph.Vertices().begin(), graph.Vertices().end(), graph, mainInstance, processorType);
}

template <typename GraphT>
VCommwT<GraphT> SumOfVerticesCommunicationWeights(const GraphT &graph) {
    static_assert(hasVertexWeightsV<GraphT>, "Graph_t must have vertex weights");

    return std::accumulate(graph.Vertices().begin(),
                           graph.Vertices().end(),
                           static_cast<VCommwT<GraphT>>(0),
                           [&](const VCommwT<GraphT> sum, const VertexIdxT<GraphT> &v) { return sum + graph.VertexCommWeight(v); });
}

template <typename GraphT>
VCommwT<GraphT> SumOfVerticesCommunicationWeights(const std::initializer_list<VertexIdxT<GraphT>> &vertices, const GraphT &graph) {
    return SumOfVerticesCommunicationWeights(vertices.begin(), vertices.end(), graph);
}

template <typename EdgeIterator, typename GraphT>
ECommwT<GraphT> SumOfEdgesCommunicationWeights(EdgeIterator begin, EdgeIterator end, const GraphT &graph) {
    static_assert(hasEdgeWeightsV<GraphT>, "Graph_t must have edge weights");
    return std::accumulate(
        begin, end, 0, [&](const auto sum, const EdgeDescT<GraphT> &e) { return sum + graph.EdgeCommWeight(e); });
}

template <typename GraphT>
ECommwT<GraphT> SumOfEdgesCommunicationWeights(const std::initializer_list<EdgeDescT<GraphT>> &edges, const GraphT &graph) {
    return SumOfEdgesCommunicationWeights(edges.begin(), edges.end(), graph);
}

template <typename GraphT>
VWorkwT<GraphT> CriticalPathWeight(const GraphT &graph) {
    static_assert(IsDirectedGraphEdgeDescV<GraphT>, "Graph_t must satisfy the directed_graph concept");
    static_assert(hasVertexWeightsV<GraphT>, "Graph_t must have vertex weights");

    if (graph.NumVertices() == 0) {
        return 0;
    }

    std::vector<VWorkwT<GraphT>> topLength(graph.NumVertices(), 0);
    VWorkwT<GraphT> criticalPathWeight = 0;

    // calculating lenght of longest path
    for (const auto &node : GetTopOrder(graph)) {
        VWorkwT<GraphT> maxTemp = 0;
        for (const auto &parent : graph.Parents(node)) {
            maxTemp = std::max(maxTemp, topLength[parent]);
        }

        topLength[node] = maxTemp + graph.VertexWorkWeight(node);

        if (topLength[node] > criticalPathWeight) {
            criticalPathWeight = topLength[node];
        }
    }

    return criticalPathWeight;
}

}    // namespace osp
