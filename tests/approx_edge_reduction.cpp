#define BOOST_TEST_MODULE ApproxEdgeReduction
#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "model/ComputationalDag.hpp"
#include "scheduler/Coarsers/TransitiveEdgeReductor.hpp"
#include "file_interactions/FileReader.hpp"


std::vector<std::string> test_graphs() {
    return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt", "data/spaa/test/test1.txt"};
}

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt"};
}


BOOST_AUTO_TEST_CASE(approx_edge_reduction) {
    
    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph =
                filename_graph.substr(filename_machine.find_last_of("/\\") + 1, filename_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            auto [status_graph, graph] =
                FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspInstance instance(graph, architecture);


            AppTransEdgeReductor reductor;

            auto ret = reductor.reduce(instance);

            auto edges = ret.getComputationalDag().long_edges_in_triangles_parallel();

            BOOST_CHECK(edges.size() == 0   );
    
        }
    }



};
