/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos K. Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#include <filesystem>
#include <fstream>
#include <iostream>

#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/coarser/Sarkar/SarkarMul.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;
using Graph_t = computational_dag_edge_idx_vector_impl_def_int_t;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph> <output file>\n" << std::endl;
        return 1;
    }

    std::string graph_file = argv[1];
    std::string graph_name = graph_file.substr(graph_file.rfind("/") + 1, graph_file.rfind(".") - graph_file.rfind("/") - 1);

    Graph_t graph;
    bool status = file_reader::readGraph(graph_file, graph);
    if (!status) {
        std::cout << "Failed to read graph\n";
        return 1;
    }

    SarkarParams::MulParameters<v_workw_t<Graph_t>> params;
    params.commCostVec = std::vector<v_workw_t<Graph_t>>({1, 2, 5, 10, 20, 50, 100, 200, 500, 1000});
    params.max_num_iteration_without_changes = 3;
    params.leniency = 0.005;
    params.maxWeight = 15000;
    params.smallWeightThreshold = 4000;
    params.buffer_merge_mode = SarkarParams::BufferMergeMode::FULL;

    SarkarMul<Graph_t, Graph_t> coarser;
    coarser.setParameters(params);

    Graph_t coarse_graph;
    std::vector<vertex_idx_t<Graph_t>> contraction_map;

    Graph_t graph_copy = graph;
    bool ignore_vertex_types = false;

    if (ignore_vertex_types) {
        for (const auto &vert : graph_copy.vertices()) {
            graph_copy.set_vertex_type(vert, 0);
        }
    }

    coarser.coarsenDag(graph_copy, coarse_graph, contraction_map);

    std::vector<unsigned> colours(contraction_map.size());
    for (std::size_t i = 0; i < contraction_map.size(); ++i) {
        colours[i] = static_cast<unsigned>(contraction_map[i]);
    }

    std::ofstream out_dot(argv[2]);
    if (!out_dot.is_open()) {
        std::cout << "Unable to write/open output file.\n";
        return 1;
    }

    DotFileWriter writer;
    writer.write_colored_graph(out_dot, graph, colours);

    if (argc >= 4) {
        std::ofstream coarse_out_dot(argv[3]);
        if (!coarse_out_dot.is_open()) {
            std::cout << "Unable to write/open output file.\n";
            return 1;
        }

        std::vector<unsigned> coarse_colours(coarse_graph.num_vertices());
        std::iota(coarse_colours.begin(), coarse_colours.end(), 0);

        writer.write_colored_graph(coarse_out_dot, coarse_graph, coarse_colours);
    }

    return 0;
}
