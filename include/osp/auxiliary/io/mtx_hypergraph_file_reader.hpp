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

#include "osp/partitioning/model/hypergraph.hpp"
#include "osp/auxiliary/io/filepath_checker.hpp"

namespace osp {
namespace file_reader {

// reads a matrix into Hypergraph format, where nonzeros are vertices, and rows/columns are hyperedges
bool readHypergraphMartixMarketFormat(std::ifstream& infile, Hypergraph& hgraph) {

    std::string line;

    // Skip comments or empty lines (robustly)
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '%') continue;

        // Null byte check
        if (line.find('\0') != std::string::npos) {
            std::cerr << "Error: Null byte detected in header line.\n";
            return false;
        }

        if (line.size() > MAX_LINE_LENGTH) {
            std::cerr << "Error: Line too long, possible malformed or malicious file.\n";
            return false;
        }
        break; // We found the actual header line
    }

    if (infile.eof()) {
        std::cerr << "Error: Unexpected end of file while reading header.\n";
        return false;
    }

    int M_row = 0, M_col = 0, nEntries = 0;

    std::istringstream header_stream(line);
    if (!(header_stream >> M_row >> M_col >> nEntries) ||
        M_row <= 0 || M_col <= 0) {
        std::cerr << "Error: Invalid header.\n";
        return false;
    }

    const unsigned num_nodes = static_cast<unsigned>(nEntries);
    if (num_nodes > std::numeric_limits<unsigned>::max()) {
        std::cerr << "Error: Matrix dimension too large for vertex type.\n";
        return false;
    }

    std::vector<int> node_work_wts(num_nodes, 0);
    std::vector<int> node_comm_wts(num_nodes, 1);

    hgraph.reset(num_nodes, 0);
    for (unsigned node = 0; node < num_nodes; ++node) {
        hgraph.set_vertex_weight(node, 1);
    }

    std::vector<std::vector<unsigned>> row_hyperedges(static_cast<unsigned>(M_row));
    std::vector<std::vector<unsigned>> column_hyperedges(static_cast<unsigned>(M_col));

    int entries_read = 0;
    while (entries_read < nEntries && std::getline(infile, line)) {
        if (line.empty() || line[0] == '%') continue;
        if (line.size() > MAX_LINE_LENGTH) {
            std::cerr << "Error: Line too long.\n";
            return false;
        }

        std::istringstream entry_stream(line);
        int row = -1, col = -1;
        double val = 0.0;

        if (!(entry_stream >> row >> col >> val)) {
            std::cerr << "Error: Malformed matrix entry.\n";
            return false;
        }

        row -= 1; col -= 1; // Convert to 0-based

        if (row < 0 || col < 0 || row >= M_row || col >= M_col) {
            std::cerr << "Error: Matrix entry out of bounds.\n";
            return false;
        }

        if (static_cast<unsigned>(row) >= num_nodes || static_cast<unsigned>(col) >= num_nodes) {
            std::cerr << "Error: Index exceeds vertex type limit.\n";
            return false;
        }

        row_hyperedges[static_cast<unsigned>(row)].push_back(static_cast<unsigned>(entries_read));
        column_hyperedges[static_cast<unsigned>(col)].push_back(static_cast<unsigned>(entries_read));

        ++entries_read;
    }

    if (entries_read != nEntries) {
        std::cerr << "Error: Incomplete matrix entries.\n";
        return false;
    }

    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') {
            std::cerr << "Error: Extra data after matrix content.\n";
            return false;
        }
    }

    for(unsigned row = 0; row < static_cast<unsigned>(M_row); ++row)
        if(!row_hyperedges[row].empty())
            hgraph.add_hyperedge(row_hyperedges[row]);

    for(unsigned col = 0; col < static_cast<unsigned>(M_col); ++col)
        if(!column_hyperedges[col].empty())
            hgraph.add_hyperedge(column_hyperedges[col]);

    return true;
}

bool readHypergraphMartixMarketFormat(const std::string& filename, Hypergraph& hgraph) {
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

    return readHypergraphMartixMarketFormat(infile, hgraph);
}

} // namespace FileReader

} // namespace osp