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

#define BOOST_TEST_MODULE File_Reader
#include <boost/test/unit_test.hpp>

#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "io/arch_file_reader.hpp"
#include "io/hdag_graph_file_reader.hpp"
#include "io/dot_graph_file_reader.hpp"
#include <filesystem>
#include <iostream>

using namespace osp;

BOOST_AUTO_TEST_CASE(test_bicgstab) {

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    computational_dag_vector_impl_def_t graph;

    bool status =
        file_reader::readComputationalDagHyperdagFormat((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), graph);

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 54);
};

BOOST_AUTO_TEST_CASE(test_arch_smpl) {

    std::filesystem::path cwd = std::filesystem::current_path();

    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
    }

    BspArchitecture<computational_dag_vector_impl_def_t> arch;

    bool status = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), arch);

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(arch.numberOfProcessors(), 3);
    BOOST_CHECK_EQUAL(arch.communicationCosts(), 3);
    BOOST_CHECK_EQUAL(arch.synchronisationCosts(), 5);
    BOOST_CHECK_EQUAL(arch.getMemoryConstraintType(), NONE);

}

BOOST_AUTO_TEST_CASE(test_k_means) {


    std::filesystem::path cwd = std::filesystem::current_path();

    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
    }

    std::vector<int> work{1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3,
                          3, 3, 2, 1, 1, 1, 1, 1, 3, 3, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1};
    std::vector<int> comm{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    computational_dag_vector_impl_def_t graph;

    bool status =
        file_reader::readComputationalDagHyperdagFormat((cwd / "data/spaa/tiny/instance_k-means.hdag").string(), graph);

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 40);
    BOOST_CHECK_EQUAL(graph.num_edges(), 45);

    for (const auto &v : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.vertex_work_weight(v), work[v]);
        BOOST_CHECK_EQUAL(graph.vertex_comm_weight(v), comm[v]);
    }

    computational_dag_edge_idx_vector_impl_def_t graph2;

    status =
        file_reader::readComputationalDagHyperdagFormat((cwd / "data/spaa/tiny/instance_k-means.hdag").string(), graph2);

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(graph2.num_vertices(), 40);
    BOOST_CHECK_EQUAL(graph2.num_edges(), 45);

    for (const auto &v : graph2.vertices()) {
        BOOST_CHECK_EQUAL(graph2.vertex_work_weight(v), work[v]);
        BOOST_CHECK_EQUAL(graph2.vertex_comm_weight(v), comm[v]);
    }
};

BOOST_AUTO_TEST_CASE(test_dot_graph) {


    std::filesystem::path cwd = std::filesystem::current_path();

    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
    }

    std::vector<unsigned> work{5, 2, 4, 5, 1, 8, 12, 8, 2, 9, 3};
    std::vector<unsigned> comm{4, 3, 2, 4, 3, 2, 4, 3, 2, 2, 2};
    std::vector<unsigned> mem{3, 5, 5, 3, 5, 5, 3, 5, 5, 5, 5};
    std::vector<unsigned> type{0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0};

    computational_dag_vector_impl_def_t graph;

    bool status =
        file_reader::readComputationalDagDotFormat((cwd / "data/dot/smpl_dot_graph_1.dot").string(), graph);

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 11);
    BOOST_CHECK_EQUAL(graph.num_edges(), 10);
    BOOST_CHECK_EQUAL(graph.num_vertex_types(), 2);

    for (const auto &v : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.vertex_work_weight(v), work[v]);
        BOOST_CHECK_EQUAL(graph.vertex_comm_weight(v), comm[v]);
        BOOST_CHECK_EQUAL(graph.vertex_mem_weight(v), mem[v]);
        BOOST_CHECK_EQUAL(graph.vertex_type(v), type[v]);
    }


};