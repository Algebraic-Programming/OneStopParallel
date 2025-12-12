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

#include <queue>
#include <unordered_set>
#include <vector>

#include "osp/partitioning/model/hypergraph.hpp"

/**
 * @file hypergraph_utility.hpp
 * @brief Utility functions and classes for working with hypergraphs graphs.
 *
 * This file provides a collection of simple utility functions for the hypergraph class.
 */

namespace osp {

// summing up weights

template <typename HypergraphT>
typename HypergraphT::VertexWorkWeightType ComputeTotalVertexWorkWeight(const HypergraphT &hgraph) {
    using IndexType = typename HypergraphT::VertexIdx;
    using WorkwType = typename HypergraphT::VertexWorkWeightType;

    WorkwType total = 0;
    for (IndexType node = 0; node < hgraph.NumVertices(); ++node) {
        total += hgraph.GetVertexWorkWeight(node);
    }
    return total;
}

template <typename HypergraphT>
typename HypergraphT::vertex_mem_weight_type ComputeTotalVertexMemoryWeight(const HypergraphT &hgraph) {
    using IndexType = typename HypergraphT::vertex_idx;
    using MemwType = typename HypergraphT::vertex_mem_weight_type;

    MemwType total = 0;
    for (IndexType node = 0; node < hgraph.NumVertices(); ++node) {
        total += hgraph.get_vertex_memory_weight(node);
    }
    return total;
}

// get induced subhypergraph

template <typename HypergraphT>
HypergraphT CreateInducedHypergraph(const HypergraphT &hgraph, const std::vector<bool> &include) {
    if (include.size() != hgraph.NumVertices()) {
        throw std::invalid_argument("Invalid Argument while extracting induced hypergraph: input bool array has incorrect size.");
    }

    using IndexType = typename HypergraphT::VertexIdx;

    std::vector<IndexType> newIndex(hgraph.NumVertices());
    unsigned currentIndex = 0;
    for (IndexType node = 0; node < hgraph.NumVertices(); ++node) {
        if (include[node]) {
            newIndex[node] = currentIndex++;
        }
    }

    HypergraphT newHgraph(currentIndex, 0);
    for (IndexType node = 0; node < hgraph.NumVertices(); ++node) {
        if (include[node]) {
            newHgraph.SetVertexWorkWeight(newIndex[node], hgraph.GetVertexWorkWeight(node));
            newHgraph.SetVertexMemoryWeight(newIndex[node], hgraph.GetVertexMemoryWeight(node));
        }
    }

    for (IndexType hyperedge = 0; hyperedge < hgraph.NumHyperedges(); ++hyperedge) {
        unsigned nrInducedPins = 0;
        std::vector<IndexType> inducedHyperedge;
        for (IndexType node : hgraph.GetVerticesInHyperedge(hyperedge)) {
            if (include[node]) {
                inducedHyperedge.push_back(newIndex[node]);
                ++nrInducedPins;
            }
        }

        if (nrInducedPins >= 2) {
            newHgraph.AddHyperedge(inducedHyperedge, hgraph.GetHyperedgeWeight(hyperedge));
        }
    }
    return newHgraph;
}

// conversion

template <typename HypergraphT, typename GraphT>
HypergraphT ConvertFromCdagAsDag(const GraphT &dag) {
    using IndexType = typename HypergraphT::vertex_idx;
    using WorkwType = typename HypergraphT::vertex_work_weight_type;
    using MemwType = typename HypergraphT::vertex_mem_weight_type;
    using CommwType = typename HypergraphT::vertex_comm_weight_type;

    static_assert(std::is_same_v<VertexIdxT<GraphT>, index_type>, "Index type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<VWorkwT<GraphT>, workw_type>, "Work weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<VMemwT<GraphT>, memw_type>, "Memory weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(!HasEdgeWeightsV<GraphT> || std::is_same_v<ECommwT<GraphT>, commw_type>,
                  "Communication weight type mismatch, cannot convert DAG to hypergraph.");

    HypergraphT hgraph(dag.NumVertices(), 0);
    for (const auto &node : dag.vertices()) {
        hgraph.SetVertexWorkWeight(node, dag.VertexWorkWeight(node));
        hgraph.set_vertex_memory_weight(node, dag.VertexMemWeight(node));
        for (const auto &child : dag.Children(node)) {
            if constexpr (HasEdgeWeightsV<GraphT>) {
                hgraph.add_hyperedge({node, child}, dag.EdgeCommWeight(edge_desc(node, child, dag).first));
            } else {
                hgraph.add_hyperedge({node, child});
            }
        }
    }
    return hgraph;
}

template <typename HypergraphT, typename GraphT>
HypergraphT ConvertFromCdagAsHyperdag(const GraphT &dag) {
    using IndexType = typename HypergraphT::vertex_idx;
    using WorkwType = typename HypergraphT::vertex_work_weight_type;
    using MemwType = typename HypergraphT::vertex_mem_weight_type;
    using CommwType = typename HypergraphT::vertex_comm_weight_type;

    static_assert(std::is_same_v<VertexIdxT<GraphT>, index_type>, "Index type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<VWorkwT<GraphT>, workw_type>, "Work weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<VMemwT<GraphT>, memw_type>, "Memory weight type mismatch, cannot convert DAG to hypergraph.");
    static_assert(std::is_same_v<VCommwT<GraphT>, commw_type>,
                  "Communication weight type mismatch, cannot convert DAG to hypergraph.");

    HypergraphT hgraph(dag.NumVertices(), 0);
    for (const auto &node : dag.vertices()) {
        hgraph.SetVertexWorkWeight(node, dag.VertexWorkWeight(node));
        hgraph.set_vertex_memory_weight(node, dag.VertexMemWeight(node));
        if (dag.OutDegree(node) == 0) {
            continue;
        }
        std::vector<IndexType> newHyperedge({node});
        for (const auto &child : dag.Children(node)) {
            newHyperedge.push_back(child);
        }
        hgraph.add_hyperedge(newHyperedge, dag.VertexCommWeight(node));
    }
    return hgraph;
}

}    // namespace osp
