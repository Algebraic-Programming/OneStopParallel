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
 * `readComputationalDagHyperdagFormatDB` reader.
 *
 * @tparam Graph_t The type of the graph, which must satisfy the ComputationalDag concept.
 * @param os The output stream to write to.
 * @param graph The computational DAG to write.
 */
template <typename Graph_t>
void writeComputationalDagHyperdagFormatDB(std::ostream &os, const Graph_t &graph, const bool write_comment_lines = false) {
    static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

    const auto num_vertices = graph.num_vertices();
    unsigned num_hyperedges = 0;
    vertex_idx_t<Graph_t> num_pins = 0;
    std::vector<vertex_idx_t<Graph_t>> hyperedge_idx_to_node;

    for (const auto &u : graph.vertices()) {
        if (graph.out_degree(u) > 0) {
            hyperedge_idx_to_node.push_back(u);
            num_hyperedges++;
            num_pins += (graph.out_degree(u) + 1);
        }
    }

    // Header
    os << "%% HyperdagDB format written by OneStopParallel\n";
    os << num_hyperedges << " " << num_vertices << " " << num_pins << "\n";

    // Hyperedges
    if (write_comment_lines) { os << "%% Hyperedges: ID comm_weight mem_weight\n"; }
    for (unsigned i = 0; i < num_hyperedges; ++i) {
        const auto u = hyperedge_idx_to_node[i];
        os << i << " " << graph.vertex_comm_weight(u) << " " << graph.vertex_mem_weight(u) << "\n";
    }

    // Vertices
    if (write_comment_lines) { os << "%% Vertices: ID work_weight type\n"; }
    for (const auto &u : graph.vertices()) {
        os << u << " " << graph.vertex_work_weight(u);
        if constexpr (has_typed_vertices_v<Graph_t>) {
            os << " " << graph.vertex_type(u);
        } else {
            os << " " << 0;
        }
        os << "\n";
    }

    // Pins
    if (write_comment_lines) { os << "%% Pins: HyperedgeID NodeID\n"; }
    for (unsigned i = 0; i < num_hyperedges; ++i) {
        const auto u = hyperedge_idx_to_node[i];
        os << i << " " << u << "\n";    // Source pin
        for (const auto &v : graph.children(u)) {
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
template <typename Graph_t>
bool writeComputationalDagHyperdagFormatDB(const std::string &filename,
                                           const Graph_t &graph,
                                           const bool write_comment_lines = false) {
    std::ofstream os(filename);
    if (!os.is_open()) {
        std::cerr << "Error: Failed to open file for writing: " << filename << "\n";
        return false;
    }
    writeComputationalDagHyperdagFormatDB(os, graph, write_comment_lines);
    return true;
}

}    // namespace file_writer
}    // namespace osp
