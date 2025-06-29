#define BOOST_TEST_MODULE heavy_edge_partitioning
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "coarser/heavy_edges/HeavyEdgePreProcess.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"
#include "auxiliary/io/hdag_graph_file_reader.hpp"

std::vector<std::string> test_graphs() {
    return {"data/spaa/tiny/instance_bicgstab.hdag", "data/spaa/tiny/instance_CG_N2_K2_nzP0d75.hdag"};
}

using namespace osp;

BOOST_AUTO_TEST_CASE(HeavyEdgePartitioning) {

    using Graph_t = boost_graph_int_t;

    std::vector<std::string> filenames_graph = test_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        std::string name_graph =
            filename_graph.substr(filename_graph.find_last_of("/\\") + 1, filename_graph.find_last_of("."));

        std::cout << std::endl << "Graph: " << name_graph << std::endl;

        Graph_t graph;

        bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(), graph);

        if (!status_graph) {

            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        int weight = 0;
        for (const auto &e : edges(graph)) {
            graph.set_edge_comm_weight(e, 1 + (weight + 100 % 500));
        }

        auto partition = heavy_edge_preprocess(graph, 5.0, 0.7f, 0.34f);
        std::vector<bool> vertex_in_partition(graph.num_vertices(), false);
        for (const auto &part : partition) {
            for (const auto &vert : part) {
                BOOST_CHECK(!vertex_in_partition[vert]);
                vertex_in_partition[vert] = true;
            }
        }
        for (const bool value : vertex_in_partition) {
            BOOST_CHECK(value);
        }
    }
}