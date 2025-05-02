#define BOOST_TEST_MODULE File_Reader
#include <boost/test/unit_test.hpp>

#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "io/arch_file_reader.hpp"
#include "io/graph_file_reader.hpp"
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