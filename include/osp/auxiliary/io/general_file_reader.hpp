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

#pragma once

#include "osp/auxiliary/io/dot_graph_file_reader.hpp"
#include "osp/auxiliary/io/filepath_checker.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/mtx_graph_file_reader.hpp"

namespace osp {
namespace file_reader {

template<typename Graph_t>
bool readGraph(const std::string& filename, Graph_t& graph) {
    if (!isPathSafe(filename)) {
        std::cerr << "Error: Unsafe file path (possible traversal or invalid type).\n";
        return false;
    }

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Failed to open file.\n";
        return false;
    }

    bool status;
    std::string file_ending = filename.substr(filename.rfind(".") + 1);
    if (file_ending == "hdag") {
        status = file_reader::readComputationalDagHyperdagFormat(infile, graph);
    } else if (file_ending == "mtx") {
        status = file_reader::readComputationalDagMartixMarketFormat(infile, graph);
    } else if (file_ending == "dot") {
        status = file_reader::readComputationalDagDotFormat(infile, graph);
    } else {
        std::cout << "Unknown file ending: ." << file_ending
                    << " ...assuming hyperDag format." << std::endl;
        status = file_reader::readComputationalDagHyperdagFormat(infile, graph);
    }

    return status;
}

}} // namespace osp::file_reader