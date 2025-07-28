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
#define MAX_LINE_LENGTH 1024         // Prevents memory abuse via long lines

namespace osp {
namespace file_reader {

// Validates the path is within the current working directory (avoids path traversal)
bool isArchPathSafe(const std::string& path) {
    try {
        std::filesystem::path resolved = std::filesystem::weakly_canonical(path);

        // Block symlinks and non-regular files
        if (std::filesystem::is_symlink(resolved)) return false;
        if (!std::filesystem::is_regular_file(resolved)) return false;

        // Optional: basic name sanity (no nulls)
        if (resolved.string().find('\0') != std::string::npos) return false;

        return true; //  File exists, is canonical, is not a symlink, and is regular
    } catch (...) {
        return false;
    }
}


template<typename Graph_t>
bool readComputationalDagMartixMarketFormat(std::ifstream& infile, Graph_t& graph) {
    using vertex_t = vertex_idx_t<Graph_t>;

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
        M_row <= 0 || M_col <= 0 || M_row != M_col) {
        std::cerr << "Error: Invalid header or non-square matrix.\n";
        return false;
    }

    const vertex_t num_nodes = static_cast<vertex_t>(M_row);
    if (num_nodes > std::numeric_limits<vertex_t>::max()) {
        std::cerr << "Error: Matrix dimension too large for vertex type.\n";
        return false;
    }

    std::vector<int> node_work_wts(num_nodes, 0);
    std::vector<int> node_comm_wts(num_nodes, 1);

    for (vertex_t i = 0; i < num_nodes; ++i) {
        graph.add_vertex(1, 1, 1);
    }

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

        if (static_cast<vertex_t>(row) >= num_nodes || static_cast<vertex_t>(col) >= num_nodes) {
            std::cerr << "Error: Index exceeds vertex type limit.\n";
            return false;
        }

        if (row < col) {
            std::cerr << "Error: Expected lower-triangular matrix.\n";
            return false;
        }

        if (row != col) {
            graph.add_edge(static_cast<vertex_t>(col), static_cast<vertex_t>(row));
            node_work_wts[static_cast<vertex_t>(row)] += 1;
        }

        ++entries_read;
    }

    if (entries_read != nEntries) {
        std::cerr << "Error: Incomplete matrix entries.\n";
        return false;
    }

    for (vertex_t i = 0; i < num_nodes; ++i) {
        graph.set_vertex_work_weight(i, static_cast<v_workw_t<Graph_t>>(node_work_wts[i]));
        graph.set_vertex_comm_weight(i, static_cast<v_commw_t<Graph_t>>(node_comm_wts[i]));
        graph.set_vertex_mem_weight(i, static_cast<v_memw_t<Graph_t>>(node_work_wts[i]));
    }

    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') {
            std::cerr << "Error: Extra data after matrix content.\n";
            return false;
        }
    }

    return true;
}

template<typename Graph_t>
bool readComputationalDagMartixMarketFormat(const std::string& filename, Graph_t& graph) {
    // Ensure the file is .mtx format
    if (std::filesystem::path(filename).extension() != ".mtx") {
        std::cerr << "Error: Only .mtx files are accepted.\n";
        return false;
    }

    if (!isArchPathSafe(filename)) {
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

    return readComputationalDagMartixMarketFormat(infile, graph);
}


// bool readProblem(const std::string &filename, DAG &G, BSPproblem &params, bool NoNUMA = true);

// std::pair<bool, BspInstance> readBspInstance(const std::string &filename);



// std::pair<bool, ComputationalDag>
// readComputationalDagMartixMarketFormat(const std::string &filename,
//                                        std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> &mtx);

// std::pair<bool, ComputationalDag>
// readComputationalDagMartixMarketFormat(std::ifstream &infile,
//                                        std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> &mtx);

// std::pair<bool, ComputationalDag> readComputationalDagMartixMarketFormat(const std::string &filename);

// std::pair<bool, ComputationalDag> readComputationalDagMartixMarketFormat(std::ifstream &infile);

// std::pair<bool, ComputationalDag> readCombinedSptrsvSpmvDagMartixMarket(const std::string &firstFilename, const std::string &secondFilename);

// std::pair<bool, csr_graph> readComputationalDagMartixMarketFormat_csr(const std::string &filename);

// std::pair<bool, csr_graph> readComputationalDagMartixMarketFormat_csr(std::ifstream &infile);

// std::pair<bool, BspArchitecture> readBspArchitecture(const std::string &filename);

// std::pair<bool, BspArchitecture> readBspArchitecture(std::ifstream &infile);


// std::pair<bool, BspSchedule> readBspSchdeuleTxtFormat(const BspInstance &instance, const std::string &filename);

// std::pair<bool, BspSchedule> readBspSchdeuleTxtFormat(const BspInstance &instance, std::ifstream &infile);

// /**
//  * Reads a BspSchedule AND Instance in Dot format from a file. The parameter BspInstance is set as the instance of the
//  * schedule. The ComputationalDag of the intance is supposed to be empty. Vertices are added as specified in the Dot
//  * file.
//  *
//  *
//  */
// std::tuple<bool, BspSchedule> readBspScheduleDotFormat(const std::string &filename, BspInstance &instance);

// /**
//  * Reads a BspSchedule AND Instance in Dot format from a file. The parameter BspInstance is set as the instance of the
//  * schedule. The ComputationalDag of the intance is supposed to be empty. Vertices are added as specified in the Dot
//  * file.
//  *
//  *
//  */
// std::tuple<bool, BspSchedule> readBspScheduleDotFormat(std::ifstream &infile, BspInstance &instance);

// /**
//  * Reads a BspSchedule in Dot format from a file. Does not read an Instance form the DOT file. An appropriate instance
//  * is meant to be passed as an agument and is set as the BspInstance of the schedule.
//  *
//  */
// std::pair<bool, BspScheduleRecomp> extractBspScheduleRecomp(const std::string &filename, const BspInstance &instance);

// /**
//  * Reads a BspSchedule in Dot format from a file. Does not read an Instance form the DOT file. An appropriate instance
//  * is meant to be passed as an agument and is set as the BspInstance of the schedule.
//  *
//  */
// std::pair<bool, BspScheduleRecomp> extractBspScheduleRecomp(std::ifstream &infile, const BspInstance &instance);

} // namespace FileReader

} // namespace osp