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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "auxiliary/auxiliary.hpp"
#include "file_interactions/FileReader.hpp"

void add_graph_stats(const ComputationalDag &graph, std::ofstream &outfile) {
    // Short and Average Edges
    unsigned short_edges = 0;
    float avg_edge_length = 0;
    size_t sum_edge_length = 0;

    std::vector<unsigned> top_level = graph.get_top_node_distance();
    std::multiset<unsigned> edge_lengths;
    for (const auto &edge : graph.edges()) {
        unsigned diff = top_level[edge.m_target] - top_level[edge.m_source];

        edge_lengths.emplace(diff);
        sum_edge_length += diff;
        if (diff == 1) {
            short_edges += 1;
        }
    }
    unsigned median_edge_length = 0;
    if (!edge_lengths.empty()) {
        median_edge_length = Get_Median(edge_lengths);
    }

    Get_Median(edge_lengths);

    if (graph.numberOfEdges() != 0) {
        avg_edge_length = (float)sum_edge_length / graph.numberOfEdges();
    }

    // Longest Path
    unsigned longest_path = 1;
    // std::map<unsigned, unsigned> wavefront;
    for (size_t i = 0; i < top_level.size(); i++) {
        longest_path = std::max(longest_path, top_level[i]);
        // if (wavefront.find(top_level[i]) != wavefront.cend()) {
        //     wavefront[top_level[i]] += 1;
        // } else {
        //     wavefront[top_level[i]] = 1;
        // }
    }
    float avg_wavefront = (float)graph.numberOfVertices() / longest_path;

    // Average bottom distance
    std::vector<unsigned> bot_level = graph.get_bottom_node_distance();
    size_t bot_level_sum = 0;
    for (size_t i = 0; i < bot_level.size(); i++) {
        bot_level_sum += bot_level[i];
    }
    float avg_bot_level = (float)bot_level_sum / bot_level.size();

    // // Number of Triangles
    // size_t number_triangles = 0;
    // for (const auto& edge : graph.edges()) {
    //     std::set<int> neighbour_src;
    //     std::set<int> neighbour_tgt;

    //     for (const auto& in_edge : graph.in_edges(edge.m_source)) {
    //         neighbour_src.emplace(in_edge.m_source);
    //     }
    //     for (const auto& in_edge : graph.out_edges(edge.m_source)) {
    //         neighbour_src.emplace(in_edge.m_target);
    //     }
    //     for (const auto& in_edge : graph.in_edges(edge.m_target)) {
    //         neighbour_tgt.emplace(in_edge.m_source);
    //     }
    //     for (const auto& in_edge : graph.out_edges(edge.m_target)) {
    //         neighbour_tgt.emplace(in_edge.m_target);
    //     }

    //     auto it_src = neighbour_src.begin();
    //     auto it_tgt = neighbour_tgt.begin();

    //     while( it_src != neighbour_src.cend() && it_tgt != neighbour_tgt.cend() ) {
    //     if ( *it_src == *it_tgt )
    //     {
    //         number_triangles++;
    //         std::advance(it_src, 1);
    //         std::advance(it_tgt, 1);
    //     }
    //     else if ( *it_src < *it_tgt )
    //         std::advance(it_src, 1);
    //     else
    //         std::advance(it_tgt,1);
    // }

    // }
    // number_triangles /= 3;

    // Adding statistics
    outfile << graph.numberOfVertices() << ",";
    outfile << graph.numberOfEdges() << ",";
    outfile << longest_path << ",";
    outfile << avg_wavefront << ",";
    outfile << short_edges << ",";
    outfile << median_edge_length << ",";
    outfile << avg_edge_length << ",";
    outfile << avg_bot_level;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <directory of graphs> <output file>\n" << std::endl;
        return 1;
    }

    std::filesystem::path graph_dir = argv[1];
    std::ofstream graph_stats_stream(argv[2]);

    if (!graph_stats_stream.is_open()) {
        std::cout << "Unable to write/open output file.\n";
        return 1;
    }

    // Generating Header
    graph_stats_stream << "Graph,Vertices,Edges,Longest_Path,Average_Wavefront_Size,Short_Edges,Median_Edge_Length,"
                          "Average_Edge_Length,Average_Bottom_Level"
                       << std::endl;

    for (const auto &dirEntry : std::filesystem::recursive_directory_iterator(graph_dir)) {
        if (std::filesystem::is_directory(dirEntry))
            continue;

        std::cout << "Processing: " << dirEntry << std::endl;

        std::string path_str = dirEntry.path();

        std::pair<bool, ComputationalDag> read_graph(false, ComputationalDag());
        if (path_str.substr(path_str.rfind(".") + 1) == "txt") {
            read_graph = FileReader::readComputationalDagHyperdagFormat(dirEntry.path());
        } else if (path_str.substr(path_str.rfind(".") + 1) == "mtx") {
            read_graph = FileReader::readComputationalDagMartixMarketFormat(dirEntry.path());
        }
        bool status_graph = read_graph.first;
        ComputationalDag &graph = read_graph.second;

        if (!status_graph)
            continue;

        std::string graph_name = path_str.substr(path_str.rfind("/") + 1);
        graph_name = graph_name.substr(0, graph_name.rfind("."));

        graph_stats_stream << graph_name << ",";
        add_graph_stats(graph, graph_stats_stream);
        graph_stats_stream << std::endl;
    }

    return 0;
}