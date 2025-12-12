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
using GraphT = computational_dag_edge_idx_vector_impl_def_int_t;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph> <output file>\n" << std::endl;
        return 1;
    }

    std::string graphFile = argv[1];
    std::string graphName = graphFile.substr(graphFile.rfind("/") + 1, graphFile.rfind(".") - graphFile.rfind("/") - 1);

    GraphT graph;
    bool status = file_reader::readGraph(graphFile, graph);
    if (!status) {
        std::cout << "Failed to read graph\n";
        return 1;
    }

    SarkarParams::MulParameters<VWorkwT<GraphT>> params;
    params.commCostVec = std::vector<VWorkwT<GraphT>>({1, 2, 5, 10, 20, 50, 100, 200, 500, 1000});
    params.max_num_iteration_without_changes = 3;
    params.leniency = 0.005;
    params.maxWeight = 15000;
    params.smallWeightThreshold = 4000;
    params.buffer_merge_mode = SarkarParams::BufferMergeMode::FULL;

    SarkarMul<GraphT, GraphT> coarser;
    coarser.setParameters(params);

    GraphT coarseGraph;
    std::vector<vertex_idx_t<GraphT>> contractionMap;

    GraphT graphCopy = graph;
    bool ignoreVertexTypes = false;

    if (ignoreVertexTypes) {
        for (const auto &vert : graphCopy.vertices()) {
            graphCopy.SetVertexType(vert, 0);
        }
    }

    coarser.coarsenDag(graphCopy, coarseGraph, contractionMap);

    std::vector<unsigned> colours(contractionMap.size());
    for (std::size_t i = 0; i < contractionMap.size(); ++i) {
        colours[i] = static_cast<unsigned>(contractionMap[i]);
    }

    std::ofstream outDot(argv[2]);
    if (!outDot.is_open()) {
        std::cout << "Unable to write/open output file.\n";
        return 1;
    }

    DotFileWriter writer;
    writer.write_colored_graph(outDot, graph, colours);

    if (argc >= 4) {
        std::ofstream coarseOutDot(argv[3]);
        if (!coarseOutDot.is_open()) {
            std::cout << "Unable to write/open output file.\n";
            return 1;
        }

        std::vector<unsigned> coarseColours(coarseGraph.NumVertices());
        std::iota(coarseColours.begin(), coarseColours.end(), 0);

        writer.write_colored_graph(coarseOutDot, coarseGraph, coarseColours);
    }

    return 0;
}
