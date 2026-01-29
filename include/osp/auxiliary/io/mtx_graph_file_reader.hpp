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

namespace osp {
namespace file_reader {

template <typename GraphT>
bool ReadComputationalDagMartixMarketFormat(std::ifstream &infile, GraphT &graph) {
    using VertexT = VertexIdxT<GraphT>;

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
    if (!(headerStream >> mRow >> mCol >> nEntries) || mRow <= 0 || mCol <= 0 || mRow != mCol) {
        std::cerr << "Error: Invalid header or non-square matrix.\n";
        return false;
    }

    if (static_cast<unsigned long long>(mRow) > std::numeric_limits<VertexT>::max()) {
        std::cerr << "Error: Matrix dimension too large for vertex type.\n";
        return false;
    }

    const VertexT numNodes = static_cast<VertexT>(mRow);
    std::vector<int> nodeWorkWts(numNodes, 0);
    std::vector<int> nodeCommWts(numNodes, 1);

    for (VertexT i = 0; i < numNodes; ++i) {
        graph.AddVertex(1, 1, 1);
    }

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

        std::cout << "row: " << row << " col: " << col << " val: " << val << std::endl;

        row -= 1;
        col -= 1;    // Convert to 0-based

        if (row < 0 || col < 0 || row >= mRow || col >= mCol) {
            std::cerr << "Error: Matrix entry out of bounds.\n";
            return false;
        }

        if (static_cast<VertexT>(row) >= numNodes || static_cast<VertexT>(col) >= numNodes) {
            std::cerr << "Error: Index exceeds vertex type limit.\n";
            return false;
        }

        if (row < col) {
            std::cerr << "Error: Expected lower-triangular matrix.\n";
            return false;
        }

        if (row != col) {
            graph.AddEdge(static_cast<VertexT>(col), static_cast<VertexT>(row));
            nodeWorkWts[static_cast<VertexT>(row)] += 1;
        }

        ++entriesRead;
    }

    if (entriesRead != nEntries) {
        std::cerr << "Error: Incomplete matrix entries.\n";
        return false;
    }

    for (VertexT i = 0; i < numNodes; ++i) {
        graph.SetVertexWorkWeight(i, static_cast<VWorkwT<GraphT>>(nodeWorkWts[i]));
        graph.SetVertexCommWeight(i, static_cast<VCommwT<GraphT>>(nodeCommWts[i]));
        graph.SetVertexMemWeight(i, static_cast<VMemwT<GraphT>>(nodeWorkWts[i]));
    }

    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') {
            std::cerr << "Error: Extra data after matrix content.\n";
            return false;
        }
    }

    return true;
}

template <typename GraphT>
bool ReadComputationalDagMartixMarketFormat(const std::string &filename, GraphT &graph) {
    // Ensure the file is .mtx format
    if (std::filesystem::path(filename).extension() != ".mtx" && std::filesystem::path(filename).extension() != ".mtx2") {
        std::cerr << "Error: Only .mtx files are accepted.\n";
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

    return ReadComputationalDagMartixMarketFormat(infile, graph);
}

// bool readProblem(const std::string &filename, DAG &G, BSPproblem &params, bool NoNUMA = true);

// std::pair<bool, BspInstance> readBspInstance(const std::string &filename);

// std::pair<bool, ComputationalDag>
// ReadComputationalDagMartixMarketFormat(const std::string &filename,
//                                        std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> &mtx);

// std::pair<bool, ComputationalDag>
// ReadComputationalDagMartixMarketFormat(std::ifstream &infile,
//                                        std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> &mtx);

// std::pair<bool, ComputationalDag> ReadComputationalDagMartixMarketFormat(const std::string &filename);

// std::pair<bool, ComputationalDag> ReadComputationalDagMartixMarketFormat(std::ifstream &infile);

// std::pair<bool, ComputationalDag> readCombinedSptrsvSpmvDagMartixMarket(const std::string &firstFilename, const std::string &secondFilename);

// std::pair<bool, csr_graph> readComputationalDagMartixMarketFormat_csr(const std::string &filename);

// std::pair<bool, csr_graph> readComputationalDagMartixMarketFormat_csr(std::ifstream &infile);

// std::pair<bool, BspArchitecture> ReadBspArchitecture(const std::string &filename);

// std::pair<bool, BspArchitecture> ReadBspArchitecture(std::ifstream &infile);

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

}    // namespace file_reader

}    // namespace osp
