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

#include "auxiliary/misc.hpp"
#include "auxiliary/random_graph_generator/near_diagonal_random_graph.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace osp;

using ComputationalDag = computational_dag_vector_impl_def_int_t;
using VertexType = vertex_idx_t<ComputationalDag>;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <number of vertices> <probability> <bandwidth> (optional:) <number of graphs>\n"
                  << std::endl;
        return 1;
    }

    size_t num_vert = static_cast<size_t>(std::stoul(argv[1]));
    double prob = std::atof(argv[2]);
    double bandwidth = std::atof(argv[3]);
    unsigned num_graphs = 1;
    if (argc > 4) {
        num_graphs = static_cast<unsigned>(std::stoul(argv[3]));
    }

    // std::cout << "Vert: " << num_vert << " prob: " << prob << " bandwidth: " << bandwidth << " graphs: " <<
    // num_graphs << std::endl;

    //  Initiating random values
    double lower_bound = -2;
    double upper_bound = 2;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);

    std::uniform_real_distribution<double> unif_log(-std::log(upper_bound), std::log(upper_bound));
    std::default_random_engine re;

    for (size_t i = 0; i < num_graphs; i++) {
        // Generating the graph
        ComputationalDag graph;
        near_diag_random_graph(graph, num_vert, bandwidth, prob);

        //  Generating graph name
        std::string graph_name = "RandomBand_";
        graph_name += "p" + std::to_string((int)(100 * prob)) + "_";
        graph_name += "b" + std::to_string((int)bandwidth) + "_";
        std::string graph_size_name;
        if (graph.num_vertices() < 1000) {
            graph_size_name = std::to_string(graph.num_vertices()) + "_";
        } else {
            graph_size_name = std::to_string(graph.num_vertices() / 1000) + "k_";
        }
        graph_name += graph_size_name;

        std::string graph_edge_size;
        if (graph.num_edges() < 1000) {
            graph_edge_size = std::to_string(graph.num_edges()) + "_";
        } else if (graph.num_edges() < 1000000) {
            graph_edge_size = std::to_string(graph.num_edges() / 1000) + "k_";
        } else {
            graph_edge_size = std::to_string(graph.num_edges() / 1000000) + "m_";
        }
        graph_name += graph_edge_size;

        graph_name += std::to_string(i);

        graph_name += ".mtx";

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
        std::ofstream graph_write;
        graph_write.open(graph_name);
        graph_write << header;
        graph_write << std::to_string(graph.num_vertices()) + " " + std::to_string(graph.num_vertices()) + " " +
                           std::to_string(graph.num_edges() + graph.num_vertices()) + "\n";
        for (VertexType i = 0; i < num_vert; i++) {
            double val = (1 - 2 * randInt(2)) * std::exp(unif_log(re));
            graph_write << std::to_string(i + 1) + " " + std::to_string(i + 1) + " " + std::to_string(val) + "\n";
            for (const auto &chld : graph.children(i)) {
                val = unif(re);
                graph_write << std::to_string(chld + 1) + " " + std::to_string(i + 1) + " " + std::to_string(val) +
                                   "\n";
            }
        }
        graph_write.close();
    }

    return 0;
}