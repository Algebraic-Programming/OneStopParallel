#define BOOST_TEST_MODULE File_Reader
#include <boost/test/unit_test.hpp>

#include "file_interactions/FileReader.hpp"
#include "model/ComputationalDag.hpp"
#include <filesystem>
#include <iostream>


std::vector<std::string> test_graphs_dag() {
    return {"data/spaa/tiny/instance_bicgstab.txt", "data/spaa/small/instance_CG_N7_K7_nzP0d2.txt"};
}

std::vector<unsigned> test_graphs_dag_num_nodes() { return {54, 597}; }

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt", "data/machine_params/p16_g1_l5_numa2.txt"};
}

std::vector<std::string> test_p3_architecture() {
    return {"data/machine_params/p3.txt"};
}



BOOST_AUTO_TEST_CASE(Dag) {

    std::vector<std::string> filenames_graph = test_graphs_dag();
    std::vector<unsigned> num_nodes = test_graphs_dag_num_nodes();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (size_t i = 0; i < filenames_graph.size(); i++) {
        std::pair<bool, ComputationalDag> result =
            FileReader::readComputationalDagHyperdagFormat((cwd / filenames_graph[i]).string());
        const bool &status_graph = result.first;
        const ComputationalDag &graph = result.second;

        BOOST_CHECK(status_graph);
        BOOST_CHECK_EQUAL(graph.numberOfVertices(), num_nodes[i]);
    }
};

