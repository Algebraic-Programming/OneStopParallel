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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "osp/auxiliary/io/filepath_checker.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/concepts/constructable_computational_dag_concept.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {
namespace file_reader {

template <typename GraphT>
bool ReadComputationalDagHyperdagFormat(std::ifstream &infile, GraphT &graph) {
    std::string line;

    // Skip comment lines starting with '%'
    while (std::getline(infile, line) && line[0] == '%') {}

    if (line.length() > MAX_LINE_LENGTH) {
        std::cerr << "Error: Input line too long.\n";
        return false;
    }

    int hEdges, pins, n;
    std::istringstream headerStream(line);
    if (!(headerStream >> hEdges >> n >> pins) || n <= 0 || hEdges <= 0 || pins <= 0) {
        std::cerr << "Incorrect input file format (invalid or non-positive sizes).\n";
        return false;
    }

    const VertexIdxT<GraphT> numNodes = static_cast<VertexIdxT<GraphT>>(n);
    for (VertexIdxT<GraphT> i = 0; i < numNodes; i++) {
        graph.AddVertex(1, 1, 1);
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
        if (!(pinStream >> hEdge >> node) || hEdge < 0 || node < 0 || hEdge >= hEdges || node >= n) {
            std::cerr << "Incorrect input file format (invalid pin line or out-of-range index).\n";
            return false;
        }

        const std::size_t edgeIdx = static_cast<VertexIdxT<GraphT>>(hEdge);
        if (edgeIdx >= edgeSource.size()) {
            std::cerr << "Error: hEdge out of bounds.\n";
            return false;
        }

        if (edgeSource[edgeIdx] == -1) {
            edgeSource[edgeIdx] = node;
        } else {
            graph.AddEdge(static_cast<VertexIdxT<GraphT>>(edgeSource[edgeIdx]), static_cast<VertexIdxT<GraphT>>(node));
        }
    }

    // Read node weights
    for (int i = 0; i < n; ++i) {
        while (std::getline(infile, line) && line[0] == '%') {}
        if (line.empty() || line.length() > MAX_LINE_LENGTH) {
            std::cerr << "Incorrect input file format (invalid or long line).\n";
            return false;
        }

        std::istringstream weightStream(line);
        int node;
        VWorkwT<GraphT> work;
        VCommwT<GraphT> comm;

        if (!(weightStream >> node >> work >> comm) || node < 0 || node >= n) {
            std::cerr << "Incorrect input file format (invalid node or weights).\n";
            return false;
        }

        graph.SetVertexCommWeight(static_cast<VertexIdxT<GraphT>>(node), comm);
        graph.SetVertexWorkWeight(static_cast<VertexIdxT<GraphT>>(node), work);
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

    if (!IsAcyclic(graph)) {
        std::cerr << "Error: DAG is not acyclic.\n";
        return false;
    }

    return true;
}

template <typename GraphT>
bool ReadComputationalDagHyperdagFormat(const std::string &filename, GraphT &graph) {
    if (!IsPathSafe(filename)) {
        std::cerr << "Error: Unsafe file path (possible traversal or invalid type).\n";
        return false;
    }

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Failed to open file.\n";
        return false;
    }

    return ReadComputationalDagHyperdagFormat(infile, graph);
}

template <typename GraphT>
bool ReadComputationalDagHyperdagFormatDb(std::ifstream &infile, GraphT &graph) {
    std::string line;

    // Skip comment lines
    while (std::getline(infile, line) && line[0] == '%') {}

    if (line.empty() || line.length() > MAX_LINE_LENGTH) {
        std::cerr << "Error: Invalid or excessively long header line.\n";
        return false;
    }

    int hEdges = 0, pins = 0, n = 0;
    std::istringstream headerStream(line);
    if (!(headerStream >> hEdges >> n >> pins) || n <= 0 || hEdges <= 0 || pins <= 0) {
        std::cerr << "Incorrect input file format (invalid or non-positive sizes).\n";
        return false;
    }

    std::vector<VCommwT<GraphT>> hyperedgeCommWeights(static_cast<size_t>(hEdges), 1);
    std::vector<VMemwT<GraphT>> hyperedgeMemWeights(static_cast<size_t>(hEdges), 1);

    // Read hyperedges
    for (int i = 0; i < hEdges; ++i) {
        while (std::getline(infile, line) && line[0] == '%') {}
        if (line.empty() || line.length() > MAX_LINE_LENGTH) {
            std::cerr << "Warning: Skipping invalid or overly long line for hyperedge " << i << ".\n";
            continue;
        }

        std::istringstream edgeStream(line);
        int hEdge = -1, commWeight = 1, memWeight = 1;
        if (!(edgeStream >> hEdge)) {
            std::cerr << "Warning: Could not read hyperedge ID for hyperedge " << i << ".\n";
            continue;
        }
        edgeStream >> commWeight >> memWeight;    // optional

        if (hEdge < 0 || hEdge >= hEdges) {
            std::cerr << "Error: Hyperedge ID " << hEdge << " is out of range (0 to " << hEdges - 1 << ").\n";
            continue;
        }
        hyperedgeCommWeights[static_cast<size_t>(hEdge)] = static_cast<VCommwT<GraphT>>(commWeight);
        hyperedgeMemWeights[static_cast<size_t>(hEdge)] = static_cast<VMemwT<GraphT>>(memWeight);
    }

    graph = GraphT(static_cast<VertexIdxT<GraphT>>(n));

    // Read vertices
    for (int i = 0; i < n; ++i) {
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

        if (node < 0 || node >= n) {
            std::cerr << "Error: Vertex ID " << node << " is out of range (0 to " << n - 1 << ").\n";
            continue;
        }

        graph.SetVertexWorkWeight(static_cast<VertexIdxT<GraphT>>(node), static_cast<VWorkwT<GraphT>>(work));

        if constexpr (HasTypedVerticesV<GraphT>) {
            graph.SetVertexType(static_cast<VertexIdxT<GraphT>>(node), static_cast<VTypeT<GraphT>>(type));
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

        if (hEdge < 0 || hEdge >= hEdges || node < 0 || node >= n) {
            std::cerr << "Error: Invalid pin indices at line " << i << ".\n";
            continue;
        }

        const std::size_t edgeIdx = static_cast<std::size_t>(hEdge);
        const std::size_t nodeIdx = static_cast<std::size_t>(node);

        if (edgeSource[edgeIdx] == -1) {
            edgeSource[edgeIdx] = node;
            graph.SetVertexCommWeight(static_cast<VertexIdxT<GraphT>>(node), hyperedgeCommWeights[edgeIdx]);
            graph.SetVertexMemWeight(static_cast<VertexIdxT<GraphT>>(node), hyperedgeMemWeights[edgeIdx]);
        } else {
            if constexpr (IsModifiableCdagCommEdgeV<GraphT>) {
                auto edge = graph.AddEdge(static_cast<VertexIdxT<GraphT>>(edgeSource[edgeIdx]),
                                          static_cast<VertexIdxT<GraphT>>(nodeIdx));

                graph.SetEdgeCommWeight(edge.first, static_cast<ECommwT<GraphT>>(hyperedgeCommWeights[edgeIdx]));

            } else {
                graph.AddEdge(static_cast<VertexIdxT<GraphT>>(edgeSource[edgeIdx]), static_cast<VertexIdxT<GraphT>>(nodeIdx));
            }
        }
    }

    if (!IsAcyclic(graph)) {
        std::cerr << "Error: Constructed DAG is not acyclic.\n";
        return false;
    }

    return true;
}

template <typename GraphT>
bool ReadComputationalDagHyperdagFormatDb(const std::string &filename, GraphT &graph) {
    // Optional: limit file extension for safety
    if (std::filesystem::path(filename).extension() != ".hdag") {
        std::cerr << "Error: Only .hdag files are accepted.\n";
        return false;
    }

    if (!IsPathSafe(filename)) {
        std::cerr << "Error: Unsafe file path (potential traversal or invalid type).\n";
        return false;
    }

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Failed to open file.\n";
        return false;
    }

    return ReadComputationalDagHyperdagFormatDb(infile, graph);
}

}    // namespace file_reader
}    // namespace osp
