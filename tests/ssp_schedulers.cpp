#define BOOST_TEST_MODULE SSP_SCHEDULERS
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "scheduler/Coarsers/FunnelSSP.hpp"
#include "scheduler/Coarsers/FunnelBfsSSP.hpp"
#include "scheduler/SSPScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceSspScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedySspLocking.hpp"
#include "model/SspSchedule.hpp"
#include "file_interactions/FileReader.hpp"

std::vector<std::string> test_graphs() {
    return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt", "data/spaa/test/test1.txt"};
}

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt", "data/machine_params/p16_g1_l5_numa2.txt"};
};

void print_bsp_schedule(const BspSchedule &bsp_schedule) {
    std::vector<std::vector<std::vector<unsigned>>> schedule(
        bsp_schedule.numberOfSupersteps(),
        std::vector<std::vector<unsigned>>(bsp_schedule.getInstance().numberOfProcessors(), std::vector<unsigned>()));

    for (size_t node = 0; node < bsp_schedule.getInstance().numberOfVertices(); node++) {
        schedule[bsp_schedule.assignedSuperstep(node)][bsp_schedule.assignedProcessor(node)].push_back(node);
    }

    std::cout << std::endl << "Schedule:" << std::endl;
    for (size_t i = 0; i < schedule.size(); i++) {
        std::cout << std::endl << "Superstep " << i << std::endl;
        for (size_t j = 0; j < schedule[i].size(); j++) {
            std::cout << "Processor " << j << ": ";
            for (auto &node : schedule[i][j]) {
                std::cout << node << ", ";
            }
            std::cout << std::endl;
        }
    }
};

void print_ssp_procReady_usage(const SspSchedule &schedule) {
    const ComputationalDag &graph = schedule.getInstance().getComputationalDag();

    std::vector<unsigned> superstep_distance_stats(schedule.numberOfSupersteps(), 0);

    for (VertexType vert = 0; vert < graph.numberOfVertices(); vert++) {
        unsigned min_superstep_distance = (graph.numberOfParents(vert) > 0) ? UINT_MAX : 0;
        for (const VertexType &parent : graph.parents(vert)) {
            unsigned diff = schedule.assignedSuperstep(vert) - schedule.assignedSuperstep(parent);
            min_superstep_distance = std::min(min_superstep_distance, diff);
        }
        superstep_distance_stats[min_superstep_distance]++;
    }

    for (size_t dist = 0; dist < superstep_distance_stats.size(); dist++) {
        if (superstep_distance_stats[dist] == 0) continue;

        float relative_amount = static_cast<float>(superstep_distance_stats[dist]) / static_cast<float>(graph.numberOfVertices());
        std::cout << "Distance: " << dist << " Percentage: " << relative_amount << std::endl;
    }
}

void run_test(SSPScheduler *test_scheduler) {
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

            architecture.setMemoryConstraintType(LOCAL);
            architecture.setMemoryBound(8);

            BspInstance instance(graph, architecture);



            std::vector<unsigned> staleness_tests = {1, 2, 3};
            for (const unsigned staleness : staleness_tests) {
                std::pair<RETURN_STATUS, SspSchedule> result = test_scheduler->computeSspSchedule(instance, staleness);

                print_bsp_schedule(result.second);
                std::cout << std::endl;
                print_ssp_procReady_usage(result.second);
                std::cout << std::endl;

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
                BOOST_CHECK(result.second.hasValidCommSchedule());
            }
        }
    }
};

// BOOST_AUTO_TEST_CASE(Serial_test) {
//     Serial test;
//     run_test(&test);
// }

BOOST_AUTO_TEST_CASE(GreedyVarianceSspScheduler_test) {
   GreedyVarianceSspScheduler test;
   run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyVarianceSspScheduler_memory_test) {
    GreedyVarianceSspScheduler test;
    test.setUseMemoryConstraint(true);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test111) {
    Funnel_parameters params(15.0, true, true, true, true);
    GreedyVarianceSspScheduler test_sched;
    FunnelSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test101) {
    Funnel_parameters params(15.0, true, false, true, true);
    GreedyVarianceSspScheduler test_sched;
    FunnelSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test011) {
    Funnel_parameters params(15.0, false, true, true, true);
    GreedyVarianceSspScheduler test_sched;
    FunnelSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test001) {
    Funnel_parameters params(15.0, false, false, true, true);
    GreedyVarianceSspScheduler test_sched;
    FunnelSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test110) {
    Funnel_parameters params(15.0, true, true, true, false);
    GreedyVarianceSspScheduler test_sched;
    FunnelSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test100) {
    Funnel_parameters params(15.0, true, false, true, false);
    GreedyVarianceSspScheduler test_sched;
    FunnelSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test010) {
    Funnel_parameters params(15.0, false, true, true, false);
    GreedyVarianceSspScheduler test_sched;
    FunnelSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test000) {
    Funnel_parameters params(15.0, false, false, true, false);
    GreedyVarianceSspScheduler test_sched;
    FunnelSSP test(&test_sched, params);
    run_test(&test);
}



BOOST_AUTO_TEST_CASE(FunnelBfs_test111) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, true, true, true);
    GreedyVarianceSspScheduler test_sched;
    FunnelBfsSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test101) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, false, true, true);
    GreedyVarianceSspScheduler test_sched;
    FunnelBfsSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test011) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, true, true, true);
    GreedyVarianceSspScheduler test_sched;
    FunnelBfsSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test001) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, false, true, true);
    GreedyVarianceSspScheduler test_sched;
    FunnelBfsSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test110) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, true, true, false);
    GreedyVarianceSspScheduler test_sched;
    FunnelBfsSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test100) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, false, true, false);
    GreedyVarianceSspScheduler test_sched;
    FunnelBfsSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test010) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, true, true, false);
    GreedyVarianceSspScheduler test_sched;
    FunnelBfsSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test000) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, false, true, false);
    GreedyVarianceSspScheduler test_sched;
    FunnelBfsSSP test(&test_sched, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedySspLocking_test) {
   GreedySspLocking test;
   run_test(&test);
}

