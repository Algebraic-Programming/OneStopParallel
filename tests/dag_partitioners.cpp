#define BOOST_TEST_MODULE dag_partitioners
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "dag_partitioners/Partitioner.hpp"
#include "dag_partitioners/VariancePartitioner.hpp"
#include "dag_partitioners/LightEdgeVariancePartitioner.hpp"

#include "file_interactions/FileReader.hpp"

std::vector<std::string> test_graphs() {
    return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt", "data/spaa/test/test1.txt" };
}

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt", "data/machine_params/p3_local.txt", "data/machine_params/p3_global.txt", "data/machine_params/p3_persistent_transient.txt"};
}

void print_dag_partition(DAGPartition dag_partition) {
    std::vector<std::vector<unsigned>> partition(
        dag_partition.getInstance().numberOfProcessors(), std::vector<unsigned>());

    for (size_t node = 0; node < dag_partition.getInstance().numberOfVertices(); node++) {
        partition[dag_partition.assignedProcessor(node)].push_back(node);
    }
    std::cout << std::endl << "Partition Imbalance: " << dag_partition.computeWorkImbalance() << std::endl;
    std::cout << "Cut Communication Ratio: " << dag_partition.computeCommunicationRatio() << std::endl;
    std::cout << "Cut Edge Ratio: " << dag_partition.computeCutEdgesRatio() << std::endl;
    for (size_t j = 0; j < partition.size(); j++) {
        std::cout << "Processor " << j << ": ";
        for (auto &node : partition[j]) {
            std::cout << node << ", ";
        }
        std::cout << std::endl;
    }
}

void run_test(Partitioner *test_partitioner) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
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

            std::pair<RETURN_STATUS, DAGPartition> result = test_partitioner->computePartition(instance);

            print_dag_partition(result.second);

            BOOST_CHECK(SUCCESS == result.first || BEST_FOUND == result.first);
            BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
            for (const VertexType &node : instance.getComputationalDag().vertices() ) {
                BOOST_CHECK_LE( 0, result.second.assignedProcessor(node) );
                BOOST_CHECK_LE( result.second.assignedProcessor(node), instance.numberOfProcessors() - 1 );
            }
        }
    }
};


BOOST_AUTO_TEST_CASE(VariancePartitioner_linear_test) {
    VariancePartitioner test( IListPartitioner::LINEAR, false );
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(VariancePartitioner_flatspline_test) {
    VariancePartitioner test( IListPartitioner::FLATSPLINE, false );
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(VariancePartitioner_memory_test) {
    VariancePartitioner test( IListPartitioner::FLATSPLINE, true);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(VariancePartitioner_superstep_only_test) {
    VariancePartitioner test( IListPartitioner::SUPERSTEP_ONLY, false);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(VariancePartitioner_global_only_test) {
    VariancePartitioner test( IListPartitioner::GLOBAL_ONLY, false);
    run_test(&test);
}




BOOST_AUTO_TEST_CASE(LightEdgeVariancePartitioner_linear_test) {
    LightEdgeVariancePartitioner test( IListPartitioner::LINEAR, false );
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(LightEdgeVariancePartitioner_flatspline_test) {
    LightEdgeVariancePartitioner test( IListPartitioner::FLATSPLINE, false );
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(LightEdgeVariancePartitioner_memory_test) {
    LightEdgeVariancePartitioner test( IListPartitioner::FLATSPLINE, true);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(LightEdgeVariancePartitioner_superstep_only_test) {
    LightEdgeVariancePartitioner test( IListPartitioner::SUPERSTEP_ONLY, false);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(LightEdgeVariancePartitioner_global_only_test) {
    LightEdgeVariancePartitioner test( IListPartitioner::GLOBAL_ONLY, false);
    run_test(&test);
}