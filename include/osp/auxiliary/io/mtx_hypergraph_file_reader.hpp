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

// reads a matrix into Hypergraph format, where nonzeros are vertices, and rows/columns are hyperedges
template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
bool ReadHypergraphMartixMarketFormat(std::ifstream &infile, Hypergraph<IndexType, WorkwType, MemwType, CommwType> &hgraph) {
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

    int mRow = 0, mCol = 0, nEntries = 0;

    std::istringstream headerStream(line);
    if (!(headerStream >> mRow >> mCol >> nEntries) || mRow <= 0 || mCol <= 0) {
        std::cerr << "Error: Invalid header.\n";
        return false;
    }

    const IndexType numNodes = static_cast<IndexType>(nEntries);

    hgraph.reset(numNodes, 0);
    for (IndexType node = 0; node < numNodes; ++node) {
        hgraph.SetVertexWorkWeight(node, static_cast<WorkwType>(1));
        hgraph.SetVertexMemoryWeight(node, static_cast<MemwType>(1));
    }

    std::vector<std::vector<IndexType>> rowHyperedges(static_cast<IndexType>(mRow));
    std::vector<std::vector<IndexType>> columnHyperedges(static_cast<IndexType>(mCol));

    int entriesRead = 0;
    while (entriesRead < nEntries && std::getline(infile, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        if (line.size() > MAX_LINE_LENGTH) {
            std::cerr << "Error: Line too long.\n";
            return false;
        }

        std::istringstream entryStream(line);
        int row = -1, col = -1;
        double val = 0.0;

        if (!(entryStream >> row >> col >> val)) {
            std::cerr << "Error: Malformed matrix entry.\n";
            return false;
        }

        row -= 1;
        col -= 1;    // Convert to 0-based

        if (row < 0 || col < 0 || row >= mRow || col >= mCol) {
            std::cerr << "Error: Matrix entry out of bounds.\n";
            return false;
        }

        if (static_cast<IndexType>(row) >= numNodes || static_cast<IndexType>(col) >= numNodes) {
            std::cerr << "Error: Index exceeds vertex type limit.\n";
            return false;
        }

        rowHyperedges[static_cast<IndexType>(row)].push_back(static_cast<IndexType>(entriesRead));
        columnHyperedges[static_cast<IndexType>(col)].push_back(static_cast<IndexType>(entriesRead));

        ++entriesRead;
    }

    if (entriesRead != nEntries) {
        std::cerr << "Error: Incomplete matrix entries.\n";
        return false;
    }

    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') {
            std::cerr << "Error: Extra data after matrix content.\n";
            return false;
        }
    }

    for (IndexType row = 0; row < static_cast<IndexType>(mRow); ++row) {
        if (!rowHyperedges[row].empty()) {
            hgraph.AddHyperedge(rowHyperedges[row]);
        }
    }

    for (IndexType col = 0; col < static_cast<IndexType>(mCol); ++col) {
        if (!columnHyperedges[col].empty()) {
            hgraph.AddHyperedge(columnHyperedges[col]);
        }
    }

    return true;
}

template <typename IndexType, typename WorkwType, typename MemwType, typename CommwType>
bool ReadHypergraphMartixMarketFormat(const std::string &filename, Hypergraph<IndexType, WorkwType, MemwType, CommwType> &hgraph) {
    // Ensure the file is .mtx format
    if (std::filesystem::path(filename).extension() != ".mtx") {
        std::cerr << "Error: Only .mtx files are accepted.\n";
        return false;
    }

    if (!isPathSafe(filename)) {
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

    return ReadHypergraphMartixMarketFormat(infile, hgraph);
}

}    // namespace file_reader

}    // namespace osp
