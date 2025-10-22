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

@author Toni Boehnlein, Christos Matzoros, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <filesystem>

#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/auxiliary/io/filepath_checker.hpp"

namespace osp {
namespace file_reader {

template<typename Graph_t>
bool readComputationalDagHyperdagFormat(std::ifstream& infile, Graph_t& graph) {
    std::string line;

    // Skip comment lines starting with '%'
    while (std::getline(infile, line) && line[0] == '%') {}

    if (line.length() > MAX_LINE_LENGTH) {
        std::cerr << "Error: Input line too long.\n";
        return false;
    }

    int hEdges, pins, N;
    std::istringstream headerStream(line);
    if (!(headerStream >> hEdges >> N >> pins) || N <= 0 || hEdges <= 0 || pins <= 0) {
        std::cerr << "Incorrect input file format (invalid or non-positive sizes).\n";
        return false;
    }

    const vertex_idx_t<Graph_t> num_nodes = static_cast<vertex_idx_t<Graph_t>>(N);
    for (vertex_idx_t<Graph_t> i = 0; i < num_nodes; i++) {
        graph.add_vertex(1, 1, 1);
    }

    std::vector<int> edgeSource(static_cast<std::size_t>(hEdges), -1);

    // Read pins
    for (int i = 0; i < pins; ++i) {
        while (std::getline(infile, line) && line[0] == '%') {}
        if (line.empty() || line.length() > MAX_LINE_LENGTH) {
            std::cerr << "Incorrect input file format (invalid or long line).\n";
            return false;
        }

        std::istringstream pinStream(line);
        int hEdge, node;
        if (!(pinStream >> hEdge >> node) || hEdge < 0 || node < 0 || hEdge >= hEdges || node >= N) {
            std::cerr << "Incorrect input file format (invalid pin line or out-of-range index).\n";
            return false;
        }

        const std::size_t edgeIdx = static_cast<vertex_idx_t<Graph_t>>(hEdge);
        if (edgeIdx >= edgeSource.size()) {
            std::cerr << "Error: hEdge out of bounds.\n";
            return false;
        }

        if (edgeSource[edgeIdx] == -1) {
            edgeSource[edgeIdx] = node;
        } else {
            graph.add_edge(static_cast<vertex_idx_t<Graph_t>>(edgeSource[edgeIdx]),
                        static_cast<vertex_idx_t<Graph_t>>(node));
        }
    }

    // Read node weights
    for (int i = 0; i < N; ++i) {
        while (std::getline(infile, line) && line[0] == '%') {}
        if (line.empty() || line.length() > MAX_LINE_LENGTH) {
            std::cerr << "Incorrect input file format (invalid or long line).\n";
            return false;
        }

        std::istringstream weightStream(line);
        int node;
        v_workw_t<Graph_t> work;
        v_commw_t<Graph_t> comm;

        if (!(weightStream >> node >> work >> comm) || node < 0 || node >= N) {
            std::cerr << "Incorrect input file format (invalid node or weights).\n";
            return false;
        }

        graph.set_vertex_comm_weight(static_cast<vertex_idx_t<Graph_t>>(node), comm);
        graph.set_vertex_work_weight(static_cast<vertex_idx_t<Graph_t>>(node), work);
    }

    // Check for unexpected trailing lines
    /*
    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') {
            std::cerr << "Incorrect input file format (file has unexpected trailing lines).\n";
            return false;
        }
    }
    */
    
    if (!is_acyclic(graph)) {
        std::cerr << "Error: DAG is not acyclic.\n";
        return false;
    }

    return true;
}

template<typename Graph_t>
bool readComputationalDagHyperdagFormat(const std::string& filename, Graph_t& graph) {
    if (!isPathSafe(filename)) {
        std::cerr << "Error: Unsafe file path (possible traversal or invalid type).\n";
        return false;
    }

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Failed to open file.\n";
        return false;
    }

    return readComputationalDagHyperdagFormat(infile, graph);
}


template<typename Graph_t>
bool readComputationalDagHyperdagFormatDB(std::ifstream& infile, Graph_t& graph) {
    std::string line;

    // Skip comment lines
    while (std::getline(infile, line) && line[0] == '%') {}

    if (line.empty() || line.length() > MAX_LINE_LENGTH) {
        std::cerr << "Error: Invalid or excessively long header line.\n";
        return false;
    }

    int hEdges = 0, pins = 0, N = 0;
    std::istringstream headerStream(line);
    if (!(headerStream >> hEdges >> N >> pins) || N <= 0 || hEdges <= 0 || pins <= 0) {
        std::cerr << "Incorrect input file format (invalid or non-positive sizes).\n";
        return false;
    }

    std::vector<v_commw_t<Graph_t>> hyperedge_comm_weights(static_cast<size_t>(hEdges), 1);
    std::vector<v_memw_t<Graph_t>> hyperedge_mem_weights(static_cast<size_t>(hEdges), 1);

    // Read hyperedges
    for (int i = 0; i < hEdges; ++i) {
        while (std::getline(infile, line) && line[0] == '%') {}
        if (line.empty() || line.length() > MAX_LINE_LENGTH) {
            std::cerr << "Warning: Skipping invalid or overly long line for hyperedge " << i << ".\n";
            continue;
        }

        std::istringstream edgeStream(line);
        int hEdge = -1, comm_weight = 1, mem_weight = 1;
        if (!(edgeStream >> hEdge)) {
            std::cerr << "Warning: Could not read hyperedge ID for hyperedge " << i << ".\n";
            continue;
        }
        edgeStream >> comm_weight >> mem_weight; // optional

        if (hEdge < 0 || hEdge >= hEdges) {
            std::cerr << "Error: Hyperedge ID " << hEdge << " is out of range (0 to " << hEdges - 1 << ").\n";
            continue;
        }
        hyperedge_comm_weights[static_cast<size_t>(hEdge)] = static_cast<v_commw_t<Graph_t>>(comm_weight);
        hyperedge_mem_weights[static_cast<size_t>(hEdge)] = static_cast<v_memw_t<Graph_t>>(mem_weight);
    }

    graph = Graph_t(static_cast<vertex_idx_t<Graph_t>>(N));

    // Read vertices
    for (int i = 0; i < N; ++i) {
        while (std::getline(infile, line) && line[0] == '%') {}
        if (line.empty() || line.length() > MAX_LINE_LENGTH) {
            std::cerr << "Warning: Skipping invalid or overly long line for vertex " << i << ".\n";
            continue;
        }

        std::istringstream vertexStream(line);
        int node = -1, work = 1, type = 0;
        if (!(vertexStream >> node)) {
            std::cerr << "Warning: Could not read vertex ID for vertex " << i << ".\n";
            continue;
        }
        vertexStream >> work >> type;

        if (node < 0 || node >= N) {
            std::cerr << "Error: Vertex ID " << node << " is out of range (0 to " << N - 1 << ").\n";
            continue;
        }

        graph.set_vertex_work_weight(static_cast<vertex_idx_t<Graph_t>>(node), static_cast<v_workw_t<Graph_t>>(work));

        if constexpr (has_typed_vertices_v<Graph_t>) {
            graph.set_vertex_type(static_cast<vertex_idx_t<Graph_t>>(node), static_cast<v_type_t<Graph_t>>(type));
        }  
    }

    // Resize(N);
    std::vector<int> edgeSource(static_cast<std::size_t>(hEdges), -1);

    // Read pins
    for (int i = 0; i < pins; ++i) {
        while (std::getline(infile, line) && line[0] == '%') {}
        if (line.empty() || line.length() > MAX_LINE_LENGTH) {
            std::cerr << "Warning: Skipping invalid or overly long line for pin " << i << ".\n";
            continue;
        }

        std::istringstream pinStream(line);
        int hEdge = -1, node = -1;
        if (!(pinStream >> hEdge >> node)) {
            std::cerr << "Warning: Could not read both hyperedge and node IDs for pin " << i << ".\n";
            continue;
        }

        if (hEdge < 0 || hEdge >= hEdges || node < 0 || node >= N) {
            std::cerr << "Error: Invalid pin indices at line " << i << ".\n";
            continue;
        }

        const std::size_t edgeIdx = static_cast<std::size_t>(hEdge);
        const std::size_t nodeIdx = static_cast<std::size_t>(node);

        if (edgeSource[edgeIdx] == -1) {
            edgeSource[edgeIdx] = node;
            graph.set_vertex_comm_weight(static_cast<vertex_idx_t<Graph_t>>(node), hyperedge_comm_weights[edgeIdx]);
            graph.set_vertex_mem_weight(static_cast<vertex_idx_t<Graph_t>>(node), hyperedge_mem_weights[edgeIdx]);
        } else {
            if constexpr (is_modifiable_cdag_comm_edge_v<Graph_t>) {

                auto edge = graph.add_edge(static_cast<vertex_idx_t<Graph_t>>(edgeSource[edgeIdx]),
                                    static_cast<vertex_idx_t<Graph_t>>(nodeIdx));

                graph.set_edge_comm_weight(edge.first,
                    static_cast<e_commw_t<Graph_t>>(hyperedge_comm_weights[edgeIdx]));

            } else {
                graph.add_edge(static_cast<vertex_idx_t<Graph_t>>(edgeSource[edgeIdx]),
                                    static_cast<vertex_idx_t<Graph_t>>(nodeIdx));
            }
        }
    }

    if (!is_acyclic(graph)) {
        std::cerr << "Error: Constructed DAG is not acyclic.\n";
        return false;
    }

    return true;
}

template<typename Graph_t>
bool readComputationalDagHyperdagFormatDB(const std::string& filename, Graph_t& graph) {
    // Optional: limit file extension for safety
    if (std::filesystem::path(filename).extension() != ".hdag") {
        std::cerr << "Error: Only .hdag files are accepted.\n";
        return false;
    }

    if (!isPathSafe(filename)) {
        std::cerr << "Error: Unsafe file path (potential traversal or invalid type).\n";
        return false;
    }

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Failed to open file.\n";
        return false;
    }

    return readComputationalDagHyperdagFormatDB(infile, graph);
}

}} // namespace osp::file_reader