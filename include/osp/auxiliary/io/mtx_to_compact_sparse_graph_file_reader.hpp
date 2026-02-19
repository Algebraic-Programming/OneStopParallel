/*
Copyright 2026 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <tuple>
#include <type_traits>

#include "osp/auxiliary/io/mtx_graph_file_reader.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"

namespace osp {
namespace file_reader {

template <>
bool ReadComputationalDagMartixMarketFormat<
    CompactSparseGraph<true, false, false, false, false, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
    unsigned>>( std::ifstream &infile, CompactSparseGraph<true, false, false, false, false, std::size_t, std::size_t,
    std::size_t, std::size_t, std::size_t, unsigned>
        &graph) {
    using GraphT
        = CompactSparseGraph<true, false, false, false, false, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
        unsigned>;
    using VertexT = VertexIdxT<GraphT>;

    std::vector<std::pair<VertexT, VertexT>> edges;
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

    VertexT mRow = 0;
    VertexT mCol = 0;
    std::size_t nEntries = 0;

    std::istringstream headerStream(line);
    if (!(headerStream >> mRow >> mCol >> nEntries) || mRow <= 0 || mCol <= 0 || mRow != mCol) {
        std::cerr << "Error: Invalid header or non-square matrix.\n";
        return false;
    }

    const VertexT numNodes = mRow;

    std::size_t entriesRead = 0;
    while (entriesRead < nEntries && std::getline(infile, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        if (line.size() > MAX_LINE_LENGTH) {
            std::cerr << "Error: Line too long.\n";
            return false;
        }

        std::istringstream entryStream(line);
        VertexT row = std::numeric_limits<VertexT>::max();
        VertexT col = std::numeric_limits<VertexT>::max();
        double val = 0.0;

        if (!(entryStream >> row >> col >> val)) {
            std::cerr << "Error: Malformed matrix entry.\n";
            return false;
        }

        row -= 1;
        col -= 1;    // Convert to 0-based

        if (row >= mRow || col >= mCol) {
            std::cerr << "Error: Matrix entry out of bounds.\n";
            return false;
        }

        if (row < col) {
            std::cerr << "Error: Expected lower-triangular matrix.\n";
            return false;
        }

        if (row != col) {
            edges.emplace_back(col, row);
        }

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

    graph = GraphT(numNodes, edges);

    return true;
}

}    // namespace file_reader
}    // namespace osp
