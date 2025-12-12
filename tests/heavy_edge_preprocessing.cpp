#define BOOST_TEST_MODULE heavy_edge_partitioning
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <string>
#include <vector>

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/LoadBalanceScheduler/HeavyEdgePreProcess.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "test_graphs.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(HeavyEdgePartitioning) {
    using GraphT = boost_graph_int_t;

    std::vector<std::string> filenamesGraph = TestGraphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        std::string nameGraph = filenameGraph.substr(filenameGraph.find_last_of("/\\") + 1, filenameGraph.find_last_of("."));

        std::cout << std::endl << "Graph: " << nameGraph << std::endl;

        GraphT graph;

        bool statusGraph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(), graph);

        if (!statusGraph) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        int weight = 0;
        for (const auto &e : Edges(graph)) {
            graph.SetEdgeCommWeight(e, 1 + (weight + 100 % 500));
        }

        auto partition = heavy_edge_preprocess(graph, 5.0, 0.7f, 0.34f);
        std::vector<bool> vertexInPartition(graph.NumVertices(), false);
        for (const auto &part : partition) {
            for (const auto &vert : part) {
                BOOST_CHECK(!vertexInPartition[vert]);
                vertexInPartition[vert] = true;
            }
        }
        for (const bool value : vertexInPartition) {
            BOOST_CHECK(value);
        }
    }
}
