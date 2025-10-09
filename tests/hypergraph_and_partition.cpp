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

#define BOOST_TEST_MODULE HYPERGRAPH_AND_PARTITION
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "osp/partitioning/model/hypergraph.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(Hypergraph_and_Partition_test) {

    using graph = boost_graph_uint_t;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    graph DAG;

    bool status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), DAG);

    BOOST_CHECK(status);

    hypergraph Hgraph;
    
    Hgraph.convert_from_cdag_as_dag(DAG);
    BOOST_CHECK_EQUAL(DAG.num_vertices(), Hgraph.num_vertices());
    BOOST_CHECK_EQUAL(DAG.num_edges(), Hgraph.num_hyperedges());
    BOOST_CHECK_EQUAL(DAG.num_edges()*2, Hgraph.num_pins());

    Hgraph.convert_from_cdag_as_hyperdag(DAG);

};