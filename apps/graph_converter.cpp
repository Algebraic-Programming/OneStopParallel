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
#include <iostream>
#include <string>

#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_writer.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

using ComputationalDag = computational_dag_edge_idx_vector_impl_def_int_t;

void PrintUsage(const char *progName) {
    std::cerr << "Graph Format Converter" << std::endl;
    std::cerr << "----------------------" << std::endl;
    std::cerr << "This tool converts a directed graph from one file format to another. The desired output" << std::endl;
    std::cerr << "format is determined by the file extension of the output file." << std::endl << std::endl;
    std::cerr << "Usage: " << progName << " <input_file> <output_file>" << std::endl << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "  <input_file>   Path to the input graph file." << std::endl << std::endl;
    std::cerr << "  <output_file>  Path for the output graph file. Special values of '.dot' or '.hdag' can be" << std::endl;
    std::cerr << "                 used to automatically generate the output filename by replacing the input" << std::endl;
    std::cerr << "                 file's extension with the specified one." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Supported Formats:" << std::endl;
    std::cerr << "  Input (by extension):  .hdag, .mtx, .dot" << std::endl;
    std::cerr << "  Output (by extension): .hdag, .dot" << std::endl << std::endl;
    std::cerr << "The .hdag format is the HyperdagDB format. A detailed description can be found at:" << std::endl;
    std::cerr << "https://github.com/Algebraic-Programming/HyperDAG_DB" << std::endl << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  " << progName << " my_graph.mtx my_graph.hdag" << std::endl;
    std::cerr << "  " << progName << " my_graph.hdag my_graph.dot" << std::endl;
    std::cerr << "  " << progName << " my_graph.mtx .dot           # Creates my_graph.dot" << std::endl;
    std::cerr << "  " << progName << " my_graph.dot .hdag          # Creates my_graph.hdag" << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::string inputFilename = argv[1];
    std::string outputFilenameArg = argv[2];

    std::filesystem::path inputPath(inputFilename);
    std::string inputExt = inputPath.extension().string();
    std::string outputFilename;

    if (outputFilenameArg == ".dot") {
        if (inputExt == ".dot") {
            std::cerr << "Error: Input file is already a .dot file. Cannot use '.dot' as the output file argument in "
                         "this case."
                      << std::endl;
            return 1;
        }
        outputFilename = std::filesystem::path(inputFilename).replace_extension(".dot").string();
    } else if (outputFilenameArg == ".hdag") {
        if (inputExt == ".hdag") {
            std::cerr << "Error: Input file is already a .hdag file. Cannot use '.hdag' as the output file argument in "
                         "this case."
                      << std::endl;
            return 1;
        }
        outputFilename = std::filesystem::path(inputFilename).replace_extension(".hdag").string();
    } else {
        outputFilename = outputFilenameArg;
    }

    ComputationalDag graph;
    std::cout << "Attempting to read graph from " << inputFilename << "..." << std::endl;
    bool status = file_reader::readGraph(inputFilename, graph);
    if (!status) {
        std::cout << "Failed to read graph\n";
        return 1;
    }

    std::cout << "Successfully read graph with " << graph.NumVertices() << " vertices and " << graph.NumEdges() << " edges."
              << std::endl;

    std::filesystem::path outputPath(outputFilename);
    std::string outputExt = outputPath.extension().string();

    if (outputExt == ".dot") {
        DotFileWriter writer;
        writer.write_graph(outputFilename, graph);
    } else if (outputExt == ".hdag") {
        file_writer::writeComputationalDagHyperdagFormatDB(outputFilename, graph);
    } else {
        std::cerr << "Error: Unsupported output file format: " << outputExt << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    std::cout << "Successfully wrote graph to " << outputFilename << std::endl;

    return 0;
}
