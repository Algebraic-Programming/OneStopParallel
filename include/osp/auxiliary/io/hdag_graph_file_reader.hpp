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

#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp { namespace file_reader {

template<typename Graph_t>
bool readComputationalDagHyperdagFormat(std::ifstream &infile, Graph_t &graph) {

    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    // the reader cannot read graphs that have more than INT_MAX vertices
    int hEdges, pins, N;
    sscanf(line.c_str(), "%d %d %d", &hEdges, &N, &pins);

    if (N <= 0 || hEdges <= 0 || pins <= 0) {
        std::cout << "Incorrect input file format (number of nodes/hyperedges/pins is not positive).\n";
        return false;
    }

    const vertex_idx_t<Graph_t> num_nodes = static_cast<vertex_idx_t<Graph_t>>(N);

    for (vertex_idx_t<Graph_t> i = 0; i < num_nodes; i++) {
        graph.add_vertex(1, 1, 1);
    }

    // Resize(N);
    std::vector<int> edgeSource(static_cast<std::size_t>(hEdges), -1);
    // read edges
    for (int i = 0; i < pins; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return false;
        }
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int hEdge, node;
        sscanf(line.c_str(), "%d %d", &hEdge, &node);

        if (hEdge < 0 || node < 0 || hEdge >= hEdges || node >= N) {
            std::cout << "Incorrect input file format (index out of range).\n";
            return false;
        }

        if (edgeSource[static_cast<vertex_idx_t<Graph_t>>(hEdge)] == -1)
            edgeSource[static_cast<vertex_idx_t<Graph_t>>(hEdge)] = node;
        else
            graph.add_edge(static_cast<vertex_idx_t<Graph_t>>(edgeSource[static_cast<vertex_idx_t<Graph_t>>(hEdge)]),
                           static_cast<vertex_idx_t<Graph_t>>(node));
    }

    for (int i = 0; i < N; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return false;
        }

        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int node;
        v_workw_t<Graph_t> work;
        v_commw_t<Graph_t> comm; //,  mem;
        // unsigned type;
        sscanf(line.c_str(), "%d %d %d", &node, &work, &comm);

        if (node < 0 || node >= N) {
            std::cout << "Incorrect input file format, node index out of range.\n";
            return false;
        }

        graph.set_vertex_comm_weight(static_cast<vertex_idx_t<Graph_t>>(node), comm);
        graph.set_vertex_work_weight(static_cast<vertex_idx_t<Graph_t>>(node), work);
    }

    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);
    if (!infile.eof()) {
        std::cout << "Incorrect input file format (file has remaining lines).\n";
        return false;
    }

    assert(is_acyclic(graph));

    return true;
};

template<typename Graph_t>
bool readComputationalDagHyperdagFormat(const std::string &filename, Graph_t &graph) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input dag file: " << filename << "\n";

        return false;
    }

    return file_reader::readComputationalDagHyperdagFormat(infile, graph);
}

template<typename Graph_t>
bool readComputationalDagHyperdagFormatDB(std::ifstream &infile, Graph_t &graph) {

    std::string line;
    getline(infile, line);
    while (!infile.eof() && line.at(0) == '%')
        getline(infile, line);

    // the reader cannot read graphs that have more than INT_MAX vertices
    int hEdges, pins, N;
    sscanf(line.c_str(), "%d %d %d", &hEdges, &N, &pins);

    if (N <= 0 || hEdges <= 0 || pins <= 0) {
        std::cout << "Incorrect input file format (number of nodes/hyperedges/pins is not positive).\n";
        return false;
    }

    std::vector<v_commw_t<Graph_t>> hyperedge_weights(static_cast<size_t>(hEdges), 1);
    // read hyperedges and vertices, create them
    for (int i = 0; i < hEdges; ++i) {
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int hEdge = -1;
        int weight = 1; // Default hyperedge weight if not specified
        int items_read = sscanf(line.c_str(), "%d %d", &hEdge, &weight);

        if (items_read < 1) {
            std::cout << "Warning: Could not read hyperedge ID for hyperedge " << i << ". This line will be ignored."
                      << std::endl;
            continue;
        }

        if (hEdge < 0 || hEdge >= hEdges) {
            std::cout << "Error: Hyperedge ID " << hEdge << " is out of range (0 to " << hEdges - 1
                      << "). Ignoring hyperedge." << std::endl;
            continue;
        }
        hyperedge_weights[static_cast<size_t>(hEdge)] = weight;
    }

    graph = Graph_t(static_cast<vertex_idx_t<Graph_t>>(N));

    for (int i = 0; i < N; ++i) {
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int node = -1;
        int work = 1;
        int comm = 1;
        int items_read = sscanf(line.c_str(), "%d %d %d", &node, &work, &comm);

        if (items_read < 1) {
            std::cout << "Warning: Could not read vertex ID for vertex " << i << ". This line will be ignored."
                      << std::endl;
            continue;
        }

        if (node < 0 || node >= N) {
            std::cout << "Error: Vertex ID " << node << " is out of range (0 to " << N - 1 << "). Ignoring vertex."
                      << std::endl;
            continue;
        }

        graph.set_vertex_work_weight(static_cast<vertex_idx_t<Graph_t>>(node), static_cast<v_workw_t<Graph_t>>(work));
        graph.set_vertex_comm_weight(static_cast<vertex_idx_t<Graph_t>>(node), static_cast<v_commw_t<Graph_t>>(comm));
    }

    std::vector<int> edgeSource(static_cast<std::size_t>(hEdges), -1);
    for (int i = 0; i < pins; ++i) {
        if (infile.eof()) {
            std::cout << "Incorrect input file format (file terminated too early).\n";
            return false;
        }
        getline(infile, line);
        while (!infile.eof() && line.at(0) == '%')
            getline(infile, line);

        int hEdge = -1;
        int node = -1;

        if (sscanf(line.c_str(), "%d %d", &hEdge, &node) != 2) {
            std::cout << "Warning: Could not read both hyperedge and node IDs for pin " << i
                      << ". This line will be ignored." << std::endl;
            continue;
        }
        if (hEdge < 0 || hEdge >= hEdges) {
            std::cout << "Error: Hyperedge ID " << hEdge << " is out of range (0 to " << hEdges - 1
                      << "). Ignoring pin." << std::endl;
            continue;
        }
        if (node < 0 || node >= N) {
            std::cout << "Error: Node ID " << node << " is out of range (0 to " << N - 1 << "). Ignoring pin."
                      << std::endl;
            continue;
        }

        if (edgeSource[static_cast<size_t>(hEdge)] == -1) {
            edgeSource[static_cast<size_t>(hEdge)] = node;
        } else {
            auto edge = graph.add_edge(static_cast<vertex_idx_t<Graph_t>>(edgeSource[static_cast<size_t>(hEdge)]), static_cast<vertex_idx_t<Graph_t>>(node));

            if constexpr (is_modifiable_cdag_comm_edge_v<Graph_t>) {
                graph.set_edge_comm_weight(edge.first, static_cast<e_commw_t<Graph_t>>(hyperedge_weights[static_cast<size_t>(hEdge)]));
            }
        }
    }
    assert(is_acyclic(graph));
    return true;
};

template<typename Graph_t>
bool readComputationalDagHyperdagFormatDB(const std::string &filename, Graph_t &graph) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Unable to find/open input dag file: " << filename << "\n";

        return false;
    }

    return file_reader::readComputationalDagHyperdagFormatDB(infile, graph);
}

// bool readProblem(const std::string &filename, DAG &G, BSPproblem &params, bool NoNUMA = true);

// std::pair<bool, BspInstance> readBspInstance(const std::string &filename);

// std::pair<bool, ComputationalDag>
// readComputationalDagMartixMarketFormat(const std::string &filename,
//                                        std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash>
//                                        &mtx);

// std::pair<bool, ComputationalDag>
// readComputationalDagMartixMarketFormat(std::ifstream &infile,
//                                        std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash>
//                                        &mtx);

// std::pair<bool, ComputationalDag> readComputationalDagMartixMarketFormat(const std::string &filename);

// std::pair<bool, ComputationalDag> readComputationalDagMartixMarketFormat(std::ifstream &infile);

// std::pair<bool, ComputationalDag> readCombinedSptrsvSpmvDagMartixMarket(const std::string &firstFilename, const
// std::string &secondFilename);

// std::pair<bool, BspArchitecture> readBspArchitecture(const std::string &filename);

// std::pair<bool, BspArchitecture> readBspArchitecture(std::ifstream &infile);

// std::pair<bool, BspSchedule> readBspSchdeuleTxtFormat(const BspInstance &instance, const std::string &filename);

// std::pair<bool, BspSchedule> readBspSchdeuleTxtFormat(const BspInstance &instance, std::ifstream &infile);

// /**
//  * Reads a BspSchedule AND Instance in Dot format from a file. The parameter BspInstance is set as the instance of
//  the
//  * schedule. The ComputationalDag of the intance is supposed to be empty. Vertices are added as specified in the Dot
//  * file.
//  *
//  *
//  */
// std::tuple<bool, BspSchedule> readBspScheduleDotFormat(const std::string &filename, BspInstance &instance);

// /**
//  * Reads a BspSchedule AND Instance in Dot format from a file. The parameter BspInstance is set as the instance of
//  the
//  * schedule. The ComputationalDag of the intance is supposed to be empty. Vertices are added as specified in the Dot
//  * file.
//  *
//  *
//  */
// std::tuple<bool, BspSchedule> readBspScheduleDotFormat(std::ifstream &infile, BspInstance &instance);

// /**
//  * Reads a BspSchedule in Dot format from a file. Does not read an Instance form the DOT file. An appropriate
//  instance
//  * is meant to be passed as an agument and is set as the BspInstance of the schedule.
//  *
//  */
// std::pair<bool, BspScheduleRecomp> extractBspScheduleRecomp(const std::string &filename, const BspInstance
// &instance);

// /**
//  * Reads a BspSchedule in Dot format from a file. Does not read an Instance form the DOT file. An appropriate
//  instance
//  * is meant to be passed as an agument and is set as the BspInstance of the schedule.
//  *
//  */
// std::pair<bool, BspScheduleRecomp> extractBspScheduleRecomp(std::ifstream &infile, const BspInstance &instance);

}} // namespace osp::file_reader