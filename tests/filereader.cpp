#define BOOST_TEST_MODULE File_Reader
#include <boost/test/unit_test.hpp>

#include "graph_implementations/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_implementations/computational_dag_vector_impl.hpp"
#include "io/FileReader.hpp"
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

    computational_dag_vector_impl graph;

    bool status =
        FileReader::readComputationalDagHyperdagFormat((cwd / "data/spaa/tiny/instance_bicgstab.txt").string(), graph);

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 54);
    //BOOST_CHECK_EQUAL(graph.num_edges(), 101);
};