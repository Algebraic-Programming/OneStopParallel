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

#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/dot_graph_file_reader.hpp"
#include "osp/auxiliary/io/mtx_graph_file_reader.hpp"
#include <filesystem>
#include <iostream>

using namespace osp;


BOOST_AUTO_TEST_CASE(test_mtx_computational_dag_vector_impl) {

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    computational_dag_vector_impl_def_t graph;

    bool status =
        file_reader::readComputationalDagMartixMarketFormat((cwd / "data/mtx_tests/ErdosRenyi_8_19_A.mtx").string(), graph);

    std::cout << "STATUS:" << status << std::endl;
    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 8);
    BOOST_CHECK_EQUAL(graph.num_edges(),  19);

    // ---- Node 0
    std::vector<int> p0{ };
    std::vector<int> c0{4, 6, 3, 5, 2};

    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(0).begin(), graph.parents(0).end(), p0.begin(), p0.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(0).begin(), graph.children(0).end(), c0.begin(), c0.end());

    // ---- Node 1
    std::vector<int> p1{ };
    std::vector<int> c1{3, 5, 2, 6};
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(1).begin(), graph.parents(1).end(), p1.begin(), p1.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(1).begin(), graph.children(1).end(), c1.begin(), c1.end());

    // ---- Node 2
    std::vector<int> p2{0, 1};
    std::vector<int> c2{3, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(2).begin(), graph.parents(2).end(), p2.begin(), p2.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(2).begin(), graph.children(2).end(), c2.begin(), c2.end());

    // ---- Node 3
    std::vector<int> p3{0, 1, 2};
    std::vector<int> c3{5, 4, 6, 7};
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(3).begin(), graph.parents(3).end(), p3.begin(), p3.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(3).begin(), graph.children(3).end(), c3.begin(), c3.end());

    // ---- Node 4
    std::vector<int> p4{0, 3};
    std::vector<int> c4{5, 6, 7};
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(4).begin(), graph.parents(4).end(), p4.begin(), p4.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(4).begin(), graph.children(4).end(), c4.begin(), c4.end());

    // ---- Node 5
    std::vector<int> p5{0, 1, 2, 3, 4};
    std::vector<int> c5{ };
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(5).begin(), graph.parents(5).end(), p5.begin(), p5.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(5).begin(), graph.children(5).end(), c5.begin(), c5.end());

    // ---- Node 6
    std::vector<int> p6{0, 1, 3, 4};
    std::vector<int> c6{7};
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(6).begin(), graph.parents(6).end(), p6.begin(), p6.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(6).begin(), graph.children(6).end(), c6.begin(), c6.end());

    // ---- Node 7
    std::vector<int> p7{3, 4, 6};
    std::vector<int> c7{ };
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(7).begin(), graph.parents(7).end(), p7.begin(), p7.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(7).begin(), graph.children(7).end(), c7.begin(), c7.end());

};


BOOST_AUTO_TEST_CASE(test_mtx_boost_graph) {

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    boost_graph_int_t graph;

    bool status =
        file_reader::readComputationalDagMartixMarketFormat((cwd / "data/mtx_tests/ErdosRenyi_8_19_A.mtx").string(), graph);

    std::cout << "STATUS:" << status << std::endl;
    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 8);
    BOOST_CHECK_EQUAL(graph.num_edges(),  19);

    // ---- Node 0
    std::vector<int> p0{ };
    std::vector<int> c0{4, 6, 3, 5, 2};

    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(0).begin(), graph.parents(0).end(), p0.begin(), p0.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(0).begin(), graph.children(0).end(), c0.begin(), c0.end());

    // ---- Node 1
    std::vector<int> p1{ };
    std::vector<int> c1{3, 5, 2, 6};
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(1).begin(), graph.parents(1).end(), p1.begin(), p1.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(1).begin(), graph.children(1).end(), c1.begin(), c1.end());

    // ---- Node 2
    std::vector<int> p2{0, 1};
    std::vector<int> c2{3, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(2).begin(), graph.parents(2).end(), p2.begin(), p2.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(2).begin(), graph.children(2).end(), c2.begin(), c2.end());

    // ---- Node 3
    std::vector<int> p3{0, 1, 2};
    std::vector<int> c3{5, 4, 6, 7};
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(3).begin(), graph.parents(3).end(), p3.begin(), p3.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(3).begin(), graph.children(3).end(), c3.begin(), c3.end());

    // ---- Node 4
    std::vector<int> p4{0, 3};
    std::vector<int> c4{5, 6, 7};
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(4).begin(), graph.parents(4).end(), p4.begin(), p4.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(4).begin(), graph.children(4).end(), c4.begin(), c4.end());

    // ---- Node 5
    std::vector<int> p5{0, 1, 2, 3, 4};
    std::vector<int> c5{ };
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(5).begin(), graph.parents(5).end(), p5.begin(), p5.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(5).begin(), graph.children(5).end(), c5.begin(), c5.end());

    // ---- Node 6
    std::vector<int> p6{0, 1, 3, 4};
    std::vector<int> c6{7};
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(6).begin(), graph.parents(6).end(), p6.begin(), p6.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(6).begin(), graph.children(6).end(), c6.begin(), c6.end());

    // ---- Node 7
    std::vector<int> p7{3, 4, 6};
    std::vector<int> c7{ };
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.parents(7).begin(), graph.parents(7).end(), p7.begin(), p7.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(graph.children(7).begin(), graph.children(7).end(), c7.begin(), c7.end());

};



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
        file_reader::readComputationalDagHyperdagFormat((cwd / "data/spaa/tiny/instance_bicgstab.lhdag").string(), graph);

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 54);
};

BOOST_AUTO_TEST_CASE(test_hdag_boost) {

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    boost_graph_int_t graph;

    bool status =
        file_reader::readComputationalDagHyperdagFormat((cwd / "data/spaa/tiny/instance_bicgstab.lhdag").string(), graph);

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
    BOOST_CHECK_EQUAL(arch.getMemoryConstraintType(), MEMORY_CONSTRAINT_TYPE::NONE);

}

BOOST_AUTO_TEST_CASE(test_arch_smpl_signed) {

    std::filesystem::path cwd = std::filesystem::current_path();

    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
    }

    BspArchitecture<computational_dag_vector_impl_def_int_t> arch;

    bool status = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), arch);

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(arch.numberOfProcessors(), 3);
    BOOST_CHECK_EQUAL(arch.communicationCosts(), 3);
    BOOST_CHECK_EQUAL(arch.synchronisationCosts(), 5);
    BOOST_CHECK_EQUAL(arch.getMemoryConstraintType(), MEMORY_CONSTRAINT_TYPE::NONE);

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
        file_reader::readComputationalDagHyperdagFormat((cwd / "data/spaa/tiny/instance_k-means.lhdag").string(), graph);

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 40);
    BOOST_CHECK_EQUAL(graph.num_edges(), 45);

    for (const auto &v : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.vertex_work_weight(v), work[v]);
        BOOST_CHECK_EQUAL(graph.vertex_comm_weight(v), comm[v]);
    }

    computational_dag_edge_idx_vector_impl_def_t graph2;

    status =
        file_reader::readComputationalDagHyperdagFormat((cwd / "data/spaa/tiny/instance_k-means.lhdag").string(), graph2);

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

BOOST_AUTO_TEST_CASE(test_dot_graph_boost) {


    std::filesystem::path cwd = std::filesystem::current_path();

    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
    }

    std::vector<unsigned> work{5, 2, 4, 5, 1, 8, 12, 8, 2, 9, 3};
    std::vector<unsigned> comm{4, 3, 2, 4, 3, 2, 4, 3, 2, 2, 2};
    std::vector<unsigned> mem{3, 5, 5, 3, 5, 5, 3, 5, 5, 5, 5};
    std::vector<unsigned> type{0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0};

    boost_graph_int_t graph;

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