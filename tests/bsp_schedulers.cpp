#define BOOST_TEST_MODULE BSP_SCHEDULERS
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "graph_implementations/computational_dag_vector_impl.hpp"
#include "io/arch_file_reader.hpp"
#include "io/graph_file_reader.hpp"


using namespace osp;


std::vector<std::string> test_graphs() {
    return {"data/spaa/tiny/instance_bicgstab.txt", "data/spaa/tiny/instance_CG_N2_K2_nzP0d75.txt" };
}

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt"};
}


template <typename Graph_t>
void run_test(Scheduler<Graph_t> *test_scheduler) {
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
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Scheduler: " << test_scheduler->getScheduleName() << std::endl; 
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;


            BspInstance<Graph_t> instance;

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(), instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.txt").string(), instance.getArchitecture());

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }
            
            std::pair<RETURN_STATUS, BspSchedule<Graph_t>> result = test_scheduler->computeSchedule(instance);


            BOOST_CHECK_EQUAL(SUCCESS, result.first);
            BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
            BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
            BOOST_CHECK(result.second.hasValidCommSchedule());
        }
    }
};



BOOST_AUTO_TEST_CASE(GreedyBspScheduler_test) {

    GreedyBspScheduler<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}
