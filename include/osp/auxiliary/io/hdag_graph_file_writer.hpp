/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may-obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos K. Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <fstream>
#include <iostream>
#include <vector>

#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {
namespace file_writer {

/**
 * @brief Writes a computational DAG to a stream in the HyperdagDB format.
 *
 * This function converts a given graph into a hypergraph representation where each node
 * with outgoing edges becomes a hyperedge source. The format is compatible with the
 * `ReadComputationalDagHyperdagFormatDB` reader.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the ComputationalDag concept.
 * @param os The output stream to write to.
 * @param graph The computational DAG to write.
 */
template <typename GraphT>
void WriteComputationalDagHyperdagFormatDb(std::ostream &os, const GraphT &graph, const bool writeCommentLines = false) {
    static_assert(IsComputationalDagV<GraphT>, "Graph_t must be a computational DAG");

    const auto numVertices = graph.NumVertices();
    unsigned numHyperedges = 0;
    VertexIdxT<GraphT> numPins = 0;
    std::vector<VertexIdxT<GraphT>> hyperedgeIdxToNode;

    for (const auto &u : graph.Vertices()) {
        if (graph.OutDegree(u) > 0) {
            hyperedgeIdxToNode.push_back(u);
            numHyperedges++;
            numPins += (graph.OutDegree(u) + 1);
        }
    }

    // Header
    os << "%% HyperdagDB format written by OneStopParallel\n";
    os << numHyperedges << " " << numVertices << " " << numPins << "\n";

    // Hyperedges
    if (writeCommentLines) {
        os << "%% Hyperedges: ID comm_weight mem_weight\n";
    }
    for (unsigned i = 0; i < numHyperedges; ++i) {
        const auto u = hyperedgeIdxToNode[i];
        os << i << " " << graph.VertexCommWeight(u) << " " << graph.VertexMemWeight(u) << "\n";
    }

    // Vertices
    if (writeCommentLines) {
        os << "%% Vertices: ID work_weight type\n";
    }
    for (const auto &u : graph.Vertices()) {
        os << u << " " << graph.VertexWorkWeight(u);
        if constexpr (HasTypedVerticesV<GraphT>) {
            os << " " << graph.VertexType(u);
        } else {
            os << " " << 0;
        }
        os << "\n";
    }

    // Pins
    if (writeCommentLines) {
        os << "%% Pins: HyperedgeID NodeID\n";
    }
    for (unsigned i = 0; i < numHyperedges; ++i) {
        const auto u = hyperedgeIdxToNode[i];
        os << i << " " << u << "\n";    // Source pin
        for (const auto &v : graph.Children(u)) {
            os << i << " " << v << "\n";    // Target pins
        }
    }
}

/**
 * @brief Writes a computational DAG to a file in the HyperdagDB format.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the ComputationalDag concept.
 * @param filename The path to the output file.
 * @param graph The computational DAG to write.
 * @return true if writing was successful, false otherwise.
 */
template <typename GraphT>
bool WriteComputationalDagHyperdagFormatDb(const std::string &filename, const GraphT &graph, const bool writeCommentLines = false) {
    std::ofstream os(filename);
    if (!os.is_open()) {
        std::cerr << "Error: Failed to open file for writing: " << filename << "\n";
        return false;
    }
    WriteComputationalDagHyperdagFormatDb(os, graph, writeCommentLines);
    return true;
}

}    // namespace file_writer
}    // namespace osp
