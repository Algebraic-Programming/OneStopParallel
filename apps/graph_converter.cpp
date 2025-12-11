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

void print_usage(const char *prog_name) {
    std::cerr << "Graph Format Converter" << std::endl;
    std::cerr << "----------------------" << std::endl;
    std::cerr << "This tool converts a directed graph from one file format to another. The desired output" << std::endl;
    std::cerr << "format is determined by the file extension of the output file." << std::endl << std::endl;
    std::cerr << "Usage: " << prog_name << " <input_file> <output_file>" << std::endl << std::endl;
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
    std::cerr << "  " << prog_name << " my_graph.mtx my_graph.hdag" << std::endl;
    std::cerr << "  " << prog_name << " my_graph.hdag my_graph.dot" << std::endl;
    std::cerr << "  " << prog_name << " my_graph.mtx .dot           # Creates my_graph.dot" << std::endl;
    std::cerr << "  " << prog_name << " my_graph.dot .hdag          # Creates my_graph.hdag" << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_filename = argv[1];
    std::string output_filename_arg = argv[2];

    std::filesystem::path input_path(input_filename);
    std::string input_ext = input_path.extension().string();
    std::string output_filename;

    if (output_filename_arg == ".dot") {
        if (input_ext == ".dot") {
            std::cerr << "Error: Input file is already a .dot file. Cannot use '.dot' as the output file argument in "
                         "this case."
                      << std::endl;
            return 1;
        }
        output_filename = std::filesystem::path(input_filename).replace_extension(".dot").string();
    } else if (output_filename_arg == ".hdag") {
        if (input_ext == ".hdag") {
            std::cerr << "Error: Input file is already a .hdag file. Cannot use '.hdag' as the output file argument in "
                         "this case."
                      << std::endl;
            return 1;
        }
        output_filename = std::filesystem::path(input_filename).replace_extension(".hdag").string();
    } else {
        output_filename = output_filename_arg;
    }

    ComputationalDag graph;
    std::cout << "Attempting to read graph from " << input_filename << "..." << std::endl;
    bool status = file_reader::readGraph(input_filename, graph);
    if (!status) {
        std::cout << "Failed to read graph\n";
        return 1;
    }

    std::cout << "Successfully read graph with " << graph.num_vertices() << " vertices and " << graph.num_edges() << " edges."
              << std::endl;

    std::filesystem::path output_path(output_filename);
    std::string output_ext = output_path.extension().string();

    if (output_ext == ".dot") {
        DotFileWriter writer;
        writer.write_graph(output_filename, graph);
    } else if (output_ext == ".hdag") {
        file_writer::writeComputationalDagHyperdagFormatDB(output_filename, graph);
    } else {
        std::cerr << "Error: Unsupported output file format: " << output_ext << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Successfully wrote graph to " << output_filename << std::endl;

    return 0;
}
