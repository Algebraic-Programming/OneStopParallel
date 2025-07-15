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
#include "osp/auxiliary/io/dot_graph_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/mtx_graph_file_reader.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

using ComputationalDag = computational_dag_edge_idx_vector_impl_def_int_t;

void print_usage(const char *prog_name) {
    std::cerr << "Usage: " << prog_name << " <input_file> <output_file>" << std::endl;
    std::cerr << "Converts a graph from one file format to another." << std::endl;
    std::cerr << "If <output_file> is '.dot', the output file will be named after the input file with a .dot extension." << std::endl;
    std::cerr << "Supported input formats (by extension): .hdag, .mtx, .dot" << std::endl;
    std::cerr << "Supported output formats (by extension): .dot" << std::endl;
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
    } else {
        output_filename = output_filename_arg;
    }

    ComputationalDag graph;
    bool read_success = false;

    std::cout << "Attempting to read graph from " << input_filename << "..." << std::endl;

    if (input_ext == ".hdag") {
        read_success = file_reader::readComputationalDagHyperdagFormat(input_filename, graph);
    } else if (input_ext == ".txt") {
        read_success = file_reader::readComputationalDagHyperdagFormatDB(input_filename, graph);
    } else if (input_ext == ".mtx") {
        read_success = file_reader::readComputationalDagMartixMarketFormat(input_filename, graph);
    } else if (input_ext == ".dot") {
        read_success = file_reader::readComputationalDagDotFormat(input_filename, graph);
    } else {
        std::cerr << "Unknown input file extension: " << input_ext << ". Assuming .hdag format." << std::endl;
        read_success = file_reader::readComputationalDagHyperdagFormat(input_filename, graph);
    }

    if (!read_success) {
        std::cerr << "Error: Failed to read graph from " << input_filename << std::endl;
        return 1;
    }

    std::cout << "Successfully read graph with " << graph.num_vertices() << " vertices and " << graph.num_edges()
              << " edges." << std::endl;

    std::filesystem::path output_path(output_filename);
    std::string output_ext = output_path.extension().string();

    if (output_ext == ".dot") {
        DotFileWriter writer;
        writer.write_graph(output_filename, graph);
    } else {
        std::cerr << "Error: Unsupported output file format: " << output_ext << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Successfully wrote graph to " << output_filename << std::endl;

    return 0;
}