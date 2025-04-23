#define BOOST_TEST_MODULE BSP_MEM_SCHEDULERS
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "model/BspMemSchedule.hpp"
#include "scheduler/Scheduler.hpp"
#include "file_interactions/FileReader.hpp"

std::vector<std::string> test_graphs() {
    return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt", "data/spaa/test/test1.txt"};
}

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt", "data/machine_params/p16_g1_l5_numa2.txt"};
}

void run_test(Scheduler *test_scheduler) {
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

            std::pair<RETURN_STATUS, BspSchedule> result = test_scheduler->computeSchedule(instance);
            BOOST_CHECK_EQUAL(SUCCESS, result.first);

            std::vector<unsigned> minimum_memory_required_vector = BspMemSchedule::minimumMemoryRequiredPerNodeType(instance);
            unsigned max_required = *std::max_element(minimum_memory_required_vector.begin(), minimum_memory_required_vector.end());
            instance.getArchitecture().setMemoryBound(max_required);

            BspMemSchedule memSchedule1(result.second, BspMemSchedule::CACHE_EVICTION_STRATEGY::LARGEST_ID);
            BOOST_CHECK_EQUAL(&memSchedule1.getInstance(), &instance);
            BOOST_CHECK(memSchedule1.isValid());            

            BspMemSchedule memSchedule3(result.second, BspMemSchedule::CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED);
            BOOST_CHECK(memSchedule3.isValid());

            BspMemSchedule memSchedule5(result.second, BspMemSchedule::CACHE_EVICTION_STRATEGY::FORESIGHT);
            BOOST_CHECK(memSchedule5.isValid());

            instance.getArchitecture().setMemoryBound(2 * max_required);

            BspMemSchedule memSchedule2(result.second, BspMemSchedule::CACHE_EVICTION_STRATEGY::LARGEST_ID);
            BOOST_CHECK(memSchedule2.isValid());

            BspMemSchedule memSchedule4(result.second, BspMemSchedule::CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED);
            BOOST_CHECK(memSchedule4.isValid());

            BspMemSchedule memSchedule6(result.second, BspMemSchedule::CACHE_EVICTION_STRATEGY::FORESIGHT);
            BOOST_CHECK(memSchedule6.isValid());
        }
    }
};


BOOST_AUTO_TEST_CASE(GreedyBspScheduler_test) {
    GreedyBspScheduler test;
    run_test(&test);
}