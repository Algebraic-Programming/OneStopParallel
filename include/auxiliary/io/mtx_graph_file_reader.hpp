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

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include "concepts/computational_dag_concept.hpp"

namespace osp {

namespace file_reader {


template<typename Graph_t>
bool readComputationalDagMartixMarketFormat(std::ifstream& infile, Graph_t& graph) {
    using vertex_t = vertex_idx_t<Graph_t>;

    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    int nEntries, M_row, M_col;
    sscanf(line.c_str(), "%d %d %d", &M_row, &M_col, &nEntries);

    if (M_row <= 0 || M_col <= 0 || M_row != M_col) {
        std::cout << "Incorrect input file format (No rows/columns or not a square matrix).\n";
        return false;
    }

    const vertex_t num_nodes = static_cast<vertex_t>(M_row);
    std::vector<int> node_work_wts(num_nodes, 0);
    std::vector<int> node_comm_wts(num_nodes, 1);  // default communication weight = 1

    // Add vertices with placeholder weights
    for (vertex_t i = 0; i < num_nodes; ++i) {
        graph.add_vertex(1, 1, 1);  // work, comm, mem
    }

    // Read entries
    for (int i = 0; i < nEntries; ++i) {
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return false;
        }

        int row, col;
        double val; 
        sscanf(line.c_str(), "%d %d %lf", &row, &col, &val);
        row -= 1;
        col -= 1;

        if (row < 0 || col < 0 || row >= M_row || col >= M_col) {
            std::cout << "Incorrect input file format (index out of range).\n";
            return false;
        }

        if (row < col) {
            std::cout << "Incorrect input file format (matrix is not lower triangular).\n";
            return false;
        } else if (col != row) {
            graph.add_edge(static_cast<vertex_t>(col), static_cast<vertex_t>(row));
            node_work_wts[static_cast<vertex_t>(row)] += 1;
        }

    }

    // Update vertex weights
    for (vertex_t i = 0; i < num_nodes; ++i) {
        graph.set_vertex_work_weight(i, static_cast<v_workw_t<Graph_t>>(node_work_wts[i]));
        graph.set_vertex_comm_weight(i, static_cast<v_commw_t<Graph_t>>(node_comm_wts[i]));
        graph.set_vertex_mem_weight(i, static_cast<v_memw_t<Graph_t>>(node_work_wts[i]));
    }

    // Check for trailing non-comment lines
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);
    if (!infile.eof()) {
        std::cout << "Incorrect input file format (file has remaining lines).\n";
        return false;
    }

    return true;
}


template<typename Graph_t>
bool readComputationalDagMartixMarketFormat(const std::string &filename, Graph_t& graph) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input dag file: " << filename << "\n";

        return false;
    }

    return file_reader::readComputationalDagMartixMarketFormat(infile, graph);
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