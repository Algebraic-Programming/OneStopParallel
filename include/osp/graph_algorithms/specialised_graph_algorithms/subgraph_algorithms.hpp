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

#include "osp/concepts/graph_traits.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"

namespace osp {

template <typename GraphTIn, typename VertT, typename EdgeT, typename WorkWeightType, typename CommWeightType, typename MemWeightType, typename VertexTypeTemplateType>
std::unordered_map<VertexIdxT<GraphTIn>, VertexIdxT<GraphTIn>> CreateInducedSubgraphMap(
    const GraphTIn &dag,
    CompactSparseGraph<true, true, true, true, true, VertT, EdgeT, WorkWeightType, CommWeightType, MemWeightType, VertexTypeTemplateType>
        &dagOut,
    const std::vector<VertexIdxT<GraphTIn>> &selectedNodes) {
    using GraphTOut
        = CompactSparseGraph<true, true, true, true, true, VertT, EdgeT, WorkWeightType, CommWeightType, MemWeightType, VertexTypeTemplateType>;

    static_assert(std::is_same_v<vertexIdxT<GraphTIn>, vertexIdxT<GraphTOut>>,
                  "GraphTIn and out must have the same vertex_idx types");

    const std::vector<vertexIdxT<GraphTIn>> topOrder = GetTopOrder(dag);
    std::vector<vertexIdxT<GraphTIn>> topOrderPosition(topOrder.size());
    for (vertexIdxT<GraphTIn> pos = 0; pos < dag.NumVertices(); ++pos) {
        topOrderPosition[topOrder[pos]] = pos;
    }

    auto topCmp = [&topOrderPosition](const vertexIdxT<GraphTIn> &lhs, const vertexIdxT<GraphTIn> &rhs) {
        return topOrderPosition[lhs] < topOrderPosition[rhs];
    };

    std::set<vertexIdxT<GraphTIn>, decltype(topCmp)> selectedVerticesOrdered(selectedNodes.begin(), selectedNodes.end(), topCmp);

    std::unordered_map<vertexIdxT<GraphTIn>, vertexIdxT<GraphTIn>> localIdx;
    localIdx.reserve(selectedNodes.size());

    vertexIdxT<GraphTIn> nodeCntr = 0;
    for (const auto &node : selectedVerticesOrdered) {
        localIdx[node] = nodeCntr++;
    }

    std::vector<std::pair<vertexIdxT<GraphTIn>, vertexIdxT<GraphTIn>>> edges;
    for (const auto &node : selectedVerticesOrdered) {
        for (const auto &chld : dag.Children(node)) {
            if (selectedVerticesOrdered.find(chld) != selectedVerticesOrdered.end()) {
                edges.emplace_back(localIdx.at(node), localIdx.at(chld));
            }
        }
    }

    dagOut = GraphTOut(nodeCntr, edges);

    for (const auto &[oriVert, outVert] : localIdx) {
        dagOut.SetVertexWorkWeight(outVert, dag.VertexWorkWeight(oriVert));
        dagOut.SetVertexCommWeight(outVert, dag.VertexCommWeight(oriVert));
        dagOut.SetVertexMemWeight(outVert, dag.VertexMemWeight(oriVert));
        dagOut.SetVertexType(outVert, dag.VertexType(oriVert));
    }

    return localIdx;
}

}    // end namespace osp
