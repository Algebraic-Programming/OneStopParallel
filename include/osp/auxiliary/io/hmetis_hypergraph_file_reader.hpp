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
#include "osp/partitioning/model/hypergraph.hpp"

namespace osp {
namespace file_reader {

// reads a Hypergraph from a file in hMetis format
template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
bool ReadHypergraphMetisFormat(std::ifstream &infile,
                                      Hypergraph<IndexType, WorkwType, MemwType, CommwType> &hgraph) {
    std::string line;

    // Skip comments or empty lines (robustly)
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }

        // Null byte check
        if (line.find('\0') != std::string::npos) {
            std::cerr << "Error: Null byte detected in header line.\n";
            return false;
        }

        if (line.size() > MAX_LINE_LENGTH) {
            std::cerr << "Error: Line too long, possible malformed or malicious file.\n";
            return false;
        }
        break;    // We found the actual header line
    }

    if (infile.eof()) {
        std::cerr << "Error: Unexpected end of file while reading header.\n";
        return false;
    }

    IndexType numVertices, numHyperedges, weightCode = 0;

    std::istringstream headerStream(line);
    if (!(headerStream >> numHyperedges >> numVertices) || numVertices <= 0 || numHyperedges <= 0) {
        std::cerr << "Error: Invalid header.\n";
        return false;
    }

    headerStream >> weightCode;
    bool hasVertexWeights = false, hasHyperedgeWeights = false;

    switch(weightCode) {
    case 0:
        break;
    case 1:
        hasHyperedgeWeights = true;
        break;
    case 10:
        hasVertexWeights = true;
        break;
    case 11:
        hasHyperedgeWeights = true;
        hasVertexWeights = true;
        break;
    default:
        std::cerr << "Error: Invalid weight code in header (must be 0, 1, 10 or 11).\n";
        return false;
    }

    hgraph.Reset(numVertices, 0);

    IndexType edgesRead = 0;
    while (edgesRead < numHyperedges && std::getline(infile, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        if (line.size() > MAX_LINE_LENGTH) {
            std::cerr << "Error: Line too long.\n";
            return false;
        }

        std::istringstream edgeStream(line);

        CommwType weight = 1;
        if (hasHyperedgeWeights && !(edgeStream >> weight)) {
            std::cerr << "Error: Malformed hyperedge row (weight).\n";
            return false;
        }

        std::vector<IndexType> verticesInHyperedge;
        IndexType vertex;
        while (edgeStream >> vertex) {
            --vertex; // Convert to 0-based
            if (vertex < 0 || vertex >= numVertices) {
                std::cerr << "Error: Malformed hyperedge row (vertex entry).\n";
                return false;
            }
            verticesInHyperedge.push_back(vertex);
        }

        if (verticesInHyperedge.empty()) {
            std::cerr << "Error: Empty hyperedge in file.\n";
            return false;
        }

        std::sort(verticesInHyperedge.begin(), verticesInHyperedge.end());
        for (unsigned index = 0; index < verticesInHyperedge.size() - 1; ++index) {
            if (verticesInHyperedge[index] == verticesInHyperedge[index + 1]) {
                std::cerr << "Error: Malformed hyperedge row (same vertex appears multiple times).\n";
                return false;
            }
        }

        hgraph.AddHyperedge(verticesInHyperedge, weight);
        ++edgesRead;
    }

    if (edgesRead != numHyperedges) {
        std::cerr << "Error: Incomplete hyperedge entries.\n";
        return false;
    }

    if (hasVertexWeights) {
        IndexType vertexWeightsRead = 0;
        while (vertexWeightsRead < numVertices && std::getline(infile, line)) {
            if (line.empty() || line[0] == '%') {
                continue;
            }
            if (line.size() > MAX_LINE_LENGTH) {
                std::cerr << "Error: Line too long.\n";
                return false;
            }

            std::istringstream weightStream(line);
            WorkwType weight = 1;
            if (!(weightStream >> weight)) {
                std::cerr << "Error: Malformed vertex weight row.\n";
                return false;
            }

            hgraph.SetVertexWorkWeight(vertexWeightsRead, weight);

            ++vertexWeightsRead;
        }

        if (vertexWeightsRead != numVertices) {
            std::cerr << "Error: Incomplete vertex weight lines.\n";
            return false;
        }

    }

    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') {
            std::cerr << "Error: Extra data after hypergraph content.\n";
            return false;
        }
    }

    return true;
}

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
bool ReadHypergraphMetisFormat(const std::string &filename,
                                      Hypergraph<IndexType, WorkwType, MemwType, CommwType> &hgraph) {
    // Ensure the file is .hmetis format
    if (std::filesystem::path(filename).extension() != ".hmetis") {
        std::cerr << "Error: Only .hmetis files are accepted.\n";
        return false;
    }

    if (!IsPathSafe(filename)) {
        std::cerr << "Error: Unsafe file path (potential traversal attack).\n";
        return false;
    }

    if (std::filesystem::is_symlink(filename)) {
        std::cerr << "Error: Symbolic links are not allowed.\n";
        return false;
    }

    if (!std::filesystem::is_regular_file(filename)) {
        std::cerr << "Error: Input is not a regular file.\n";
        return false;
    }

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Failed to open file.\n";
        return false;
    }

    return ReadHypergraphMetisFormat(infile, hgraph);
}

}    // namespace file_reader

}    // namespace osp
