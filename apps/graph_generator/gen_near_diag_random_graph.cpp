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

#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "osp/auxiliary/misc.hpp"
#include "osp/auxiliary/random_graph_generator/near_diagonal_random_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

using ComputationalDag = computational_dag_vector_impl_def_int_t;
using VertexType = VertexIdxT<ComputationalDag>;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <number of vertices> <probability> <bandwidth> (optional:) <number of graphs>\n"
                  << std::endl;
        return 1;
    }

    size_t numVert = static_cast<size_t>(std::stoul(argv[1]));
    double prob = std::atof(argv[2]);
    double bandwidth = std::atof(argv[3]);
    unsigned numGraphs = 1;
    if (argc > 4) {
        numGraphs = static_cast<unsigned>(std::stoul(argv[3]));
    }

    // std::cout << "Vert: " << num_vert << " prob: " << prob << " bandwidth: " << bandwidth << " graphs: " <<
    // num_graphs << std::endl;

    //  Initiating random values
    double lowerBound = -2;
    double upperBound = 2;
    std::uniform_real_distribution<double> unif(lowerBound, upperBound);

    std::uniform_real_distribution<double> unifLog(-std::log(upperBound), std::log(upperBound));
    std::default_random_engine re;

    for (size_t i = 0; i < numGraphs; i++) {
        // Generating the graph
        ComputationalDag graph;
        near_diag_random_graph(graph, numVert, bandwidth, prob);

        //  Generating graph name
        std::string graphName = "RandomBand_";
        graphName += "p" + std::to_string(static_cast<int>(100 * prob)) + "_";
        graphName += "b" + std::to_string(static_cast<int>(bandwidth)) + "_";
        std::string graphSizeName;
        if (graph.NumVertices() < 1000) {
            graphSizeName = std::to_string(graph.NumVertices()) + "_";
        } else {
            graphSizeName = std::to_string(graph.NumVertices() / 1000) + "k_";
        }
        graphName += graphSizeName;

        std::string graphEdgeSize;
        if (graph.NumEdges() < 1000) {
            graphEdgeSize = std::to_string(graph.NumEdges()) + "_";
        } else if (graph.NumEdges() < 1000000) {
            graphEdgeSize = std::to_string(graph.NumEdges() / 1000) + "k_";
        } else {
            graphEdgeSize = std::to_string(graph.NumEdges() / 1000000) + "m_";
        }
        graphName += graphEdgeSize;

        graphName += std::to_string(i);

        graphName += ".mtx";

        // Graph header
        std::string header = "%"
                             "%"
                             "MatrixMarket matrix coordinate real symmetric\n"
                             "%-------------------------------------------------------------------------------\n"
                             "%"
                             " A random sparse lower trianguler matrix\n"
                             "% with uniform values from -2 to 2 on the off-diagonal\n"
                             "% and log-uniform values between 1/2 and 2 with random sign.\n"
                             "%-------------------------------------------------------------------------------\n";

        // Writing the graph to file
        std::ofstream graphWrite;
        graphWrite.open(graphName);
        graphWrite << header;
        graphWrite << std::to_string(graph.NumVertices()) + " " + std::to_string(graph.NumVertices()) + " "
                          + std::to_string(graph.NumEdges() + graph.NumVertices()) + "\n";
        for (VertexType j = 0; j < numVert; j++) {
            double val = (1 - 2 * randInt(2)) * std::exp(unifLog(re));
            graphWrite << std::to_string(j + 1) + " " + std::to_string(j + 1) + " " + std::to_string(val) + "\n";
            for (const auto &chld : graph.Children(j)) {
                val = unif(re);
                graphWrite << std::to_string(chld + 1) + " " + std::to_string(j + 1) + " " + std::to_string(val) + "\n";
            }
        }
        graphWrite.close();
    }

    return 0;
}
