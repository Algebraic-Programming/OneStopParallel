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

#include "osp/auxiliary/io/bsp_schedule_file_writer.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

using ComputationalDag = ComputationalDagEdgeIdxVectorImplDefIntT;

void AddGraphStats(const ComputationalDag &graph, std::ofstream &outfile) {
    // Short and Average Edges
    unsigned shortEdges = 0;
    float avgEdgeLength = 0;
    size_t sumEdgeLength = 0;

    std::vector<unsigned> topLevel = GetTopNodeDistance(graph);
    std::multiset<unsigned> edgeLengths;
    for (const auto &edge : Edges(graph)) {
        unsigned diff = topLevel[Target(edge, graph)] - topLevel[Source(edge, graph)];

        edgeLengths.emplace(diff);
        sumEdgeLength += diff;
        if (diff == 1) {
            shortEdges += 1;
        }
    }
    unsigned medianEdgeLength = 0;
    if (!edgeLengths.empty()) {
        medianEdgeLength = GetMedian(edgeLengths);
    }

    GetMedian(edgeLengths);

    if (graph.NumEdges() != 0) {
        avgEdgeLength = static_cast<float>(sumEdgeLength) / static_cast<float>(graph.NumEdges());
    }

    // Longest Path
    unsigned longestPath = 1;
    // std::map<unsigned, unsigned> wavefront;
    for (size_t i = 0; i < topLevel.size(); i++) {
        longestPath = std::max(longestPath, topLevel[i]);
        // if (wavefront.find(top_level[i]) != wavefront.cend()) {
        //     wavefront[top_level[i]] += 1;
        // } else {
        //     wavefront[top_level[i]] = 1;
        // }
    }
    float avgWavefront = static_cast<float>(graph.NumVertices()) / static_cast<float>(longestPath);

    // Average bottom distance
    std::vector<unsigned> botLevel = GetBottomNodeDistance(graph);
    size_t botLevelSum = 0;
    for (size_t i = 0; i < botLevel.size(); i++) {
        botLevelSum += botLevel[i];
    }
    float avgBotLevel = static_cast<float>(botLevelSum) / static_cast<float>(botLevel.size());

    // // Number of Triangles
    // size_t number_triangles = 0;
    // for (const auto& edge : edges(graph)) {
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
    outfile << graph.NumVertices() << ",";
    outfile << graph.NumEdges() << ",";
    outfile << longestPath << ",";
    outfile << avgWavefront << ",";
    outfile << shortEdges << ",";
    outfile << medianEdgeLength << ",";
    outfile << avgEdgeLength << ",";
    outfile << avgBotLevel;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <directory of graphs> <output file>\n" << std::endl;
        return 1;
    }

    std::filesystem::path graphDir = argv[1];
    std::ofstream graphStatsStream(argv[2]);

    if (!graphStatsStream.is_open()) {
        std::cout << "Unable to write/open output file.\n";
        return 1;
    }

    // Generating Header
    graphStatsStream << "Graph,Vertices,Edges,Longest_Path,Average_Wavefront_Size,Short_Edges,Median_Edge_Length,"
                        "Average_Edge_Length,Average_Bottom_Level"
                     << std::endl;

    for (const auto &dirEntry : std::filesystem::recursive_directory_iterator(graphDir)) {
        if (std::filesystem::is_directory(dirEntry)) {
            continue;
        }

        std::cout << "Processing: " << dirEntry << std::endl;

        std::string pathStr = dirEntry.path();

        ComputationalDag graph;
        bool status = file_reader::ReadGraph(dirEntry.path(), graph);
        if (!status) {
            std::cout << "Failed to read graph\n";
            return 1;
        }

        if (!status) {
            continue;
        }

        std::string graphName = pathStr.substr(pathStr.rfind("/") + 1);
        graphName = graphName.substr(0, graphName.rfind("."));

        graphStatsStream << graphName << ",";
        AddGraphStats(graph, graphStatsStream);
        graphStatsStream << std::endl;
    }

    return 0;
}
