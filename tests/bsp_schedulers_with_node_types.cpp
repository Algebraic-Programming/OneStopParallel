#define BOOST_TEST_MODULE BSP_SCHEDULERS_NODE_TYPE
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "auxiliary/auxiliary.hpp"
#include "scheduler/ReverseScheduler.hpp"
#include "scheduler/Coarsers/Funnel.hpp"
#include "scheduler/Coarsers/FunnelBfs.hpp"
#include "scheduler/Coarsers/Sarkar.hpp"
#include "scheduler/Coarsers/ScheduleClumps.hpp"
#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyBspFillupScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyBspLocking.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceFillupScheduler.hpp"
#include "scheduler/HDagg/HCoreHDagg.hpp"
#include "scheduler/LocalSearchSchedulers/HillClimbingScheduler.hpp"
#include "scheduler/Scheduler.hpp"
#include "file_interactions/FileReader.hpp"

std::vector<std::string> test_graphs() {
    return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt",  "data/spaa/test/test1.txt" }; //, "data/spaa/huge/instance_CG_N60_K55_nzP0d12.txt" };
}

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt", "data/machine_params/p16_g1_l5_numa2.txt"};
}

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
}

void run_test(Scheduler *test_scheduler) {
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

            BspInstance instance(graph, architecture);

            for(int i=0; i < instance.getComputationalDag().numberOfVertices(); i++)
                instance.getComputationalDag().setNodeType(i, i%2);

            for(int i=0; i < instance.getArchitecture().numberOfProcessors(); i++)
                instance.getArchitecture().setProcessorType(i, i%2);

            instance.setDiagonalCompatibilityMatrix(2);

            std::pair<RETURN_STATUS, BspSchedule> result = test_scheduler->computeSchedule(instance);

            print_bsp_schedule(result.second);

            BOOST_CHECK_EQUAL(SUCCESS, result.first);
            BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
            BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
            BOOST_CHECK(result.second.satisfiesNodeTypeConstraints());
            BOOST_CHECK(result.second.hasValidCommSchedule());
        }
    }
};

void run_test2(Scheduler *test_scheduler) {
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

            BspInstance instance(graph, architecture);

            for(VertexType i=0; i < instance.getComputationalDag().numberOfVertices(); i++)
                instance.getComputationalDag().setNodeType(i, randInt(7));

            for(unsigned i=0; i < instance.getArchitecture().numberOfProcessors(); i++)
                instance.getArchitecture().setProcessorType(i, i % 3);

            std::vector<std::vector<bool>> nodeType_procType_compat_matrix({
                {true, false, false},
                {false, true, false},
                {false, false, true},
                {true, true, false},
                {true, false, true},
                {false, true, true},
                {true, true, true}
            });

            instance.setNodeProcessorCompatibility(nodeType_procType_compat_matrix);

            std::pair<RETURN_STATUS, BspSchedule> result = test_scheduler->computeSchedule(instance);

            print_bsp_schedule(result.second);

            BOOST_CHECK_EQUAL(SUCCESS, result.first);
            BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
            BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
            BOOST_CHECK(result.second.satisfiesNodeTypeConstraints());
            BOOST_CHECK(result.second.hasValidCommSchedule());
        }
    }
};


BOOST_AUTO_TEST_CASE(GreedyBspScheduler_test) {
    GreedyBspScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_test2) {
    GreedyBspScheduler test;
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBspFillupScheduler_test) {
    GreedyBspFillupScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBspFillupScheduler_test2) {
    GreedyBspFillupScheduler test;
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBSPLocking_test) {
    GreedyBspLocking test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBSPLocking_test2) {
    GreedyBspLocking test;
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(GreedyVarianceScheduler_test) {
    GreedyVarianceScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyVarianceScheduler_test2) {
    GreedyVarianceScheduler test;
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(GreedyVarianceFillupScheduler_test) {
    GreedyVarianceFillupScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyVarianceFillupScheduler_test2) {
    GreedyVarianceFillupScheduler test;
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBspHC_test) {
    GreedyBspScheduler greedy;
    HillClimbingScheduler hill_climbing;
    ComboScheduler test(greedy, hill_climbing);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBspHC_test2) {
    GreedyBspScheduler greedy;
    HillClimbingScheduler hill_climbing;
    ComboScheduler test(greedy, hill_climbing);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(GreedyVarianceHC_test) {
    GreedyVarianceFillupScheduler greedy;
    HillClimbingScheduler hill_climbing;
    ComboScheduler test(greedy, hill_climbing);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyVarianceHC_test2) {
    GreedyVarianceFillupScheduler greedy;
    HillClimbingScheduler hill_climbing;
    ComboScheduler test(greedy, hill_climbing);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test111) {
    Funnel_parameters params(15.0, true, true, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test101) {
    Funnel_parameters params(15.0, true, false, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test011) {
    Funnel_parameters params(15.0, false, true, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test001) {
    Funnel_parameters params(15.0, false, false, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test110) {
    Funnel_parameters params(15.0, true, true, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test100) {
    Funnel_parameters params(15.0, true, false, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test010) {
    Funnel_parameters params(15.0, false, true, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test000) {
    Funnel_parameters params(15.0, false, false, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test111_2) {
    Funnel_parameters params(15.0, true, true, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test101_2) {
    Funnel_parameters params(15.0, true, false, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test011_2) {
    Funnel_parameters params(15.0, false, true, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test001_2) {
    Funnel_parameters params(15.0, false, false, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test110_2) {
    Funnel_parameters params(15.0, true, true, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test100_2) {
    Funnel_parameters params(15.0, true, false, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test010_2) {
    Funnel_parameters params(15.0, false, true, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_test000_2) {
    Funnel_parameters params(15.0, false, false, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    Funnel test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_Locking_test) {
    GreedyBspLocking test_sched;
    Funnel test(&test_sched);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Funnel_Locking_test2) {
    GreedyBspLocking test_sched;
    Funnel test(&test_sched);
    run_test2(&test);
}


BOOST_AUTO_TEST_CASE(FunnelBfs_test111) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, true, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test101) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, false, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test011) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, true, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test001) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, false, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test110) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, true, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test100) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, false, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test010) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, true, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test000) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, false, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test(&test);
}


BOOST_AUTO_TEST_CASE(FunnelBfs_test111_2) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, true, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test101_2) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, false, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test011_2) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, true, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test001_2) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, false, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test110_2) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, true, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test100_2) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, true, false, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test010_2) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, true, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(FunnelBfs_test000_2) {
    FunnelBfs_parameters params(ULONG_MAX, ULONG_MAX, UINT_MAX, false, false, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FunnelBfs test(&test_sched, &test_improv, params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_test) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::WEIGHT_BALANCE;
    params.consider_future_score = false;
    HCoreHDagg test(params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_typed_test) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = false;
    HCoreHDagg test(params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_typed_test_future) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = true;
    HCoreHDagg test(params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_mem_test) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::WEIGHT_BALANCE;
    params.consider_future_score = false;
    HCoreHDagg test(params);
    test.setUseMemoryConstraint(true);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_typed_mem_test) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = false;
    HCoreHDagg test(params);
    test.setUseMemoryConstraint(true);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_typed_mem_test_future) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = true;
    HCoreHDagg test(params);
    test.setUseMemoryConstraint(true);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_test2) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::WEIGHT_BALANCE;
    params.consider_future_score = false;
    HCoreHDagg test(params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_typed_test2) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = false;
    HCoreHDagg test(params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_typed_test2_future) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = true;
    HCoreHDagg test(params);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_mem_test2) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::WEIGHT_BALANCE;
    params.consider_future_score = false;
    HCoreHDagg test(params);
    test.setUseMemoryConstraint(true);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_typed_mem_test2) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = false;
    HCoreHDagg test(params);
    test.setUseMemoryConstraint(true);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(HCoreHDagg_typed_mem_test2_future) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = true;
    HCoreHDagg test(params);
    test.setUseMemoryConstraint(true);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(ScheduleClumps_test) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = true;
    HCoreHDagg clumper(params);

    GreedyBspLocking sched;
    ScheduleClumps test(&clumper, &sched);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(ScheduleClumps_test2) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = true;
    HCoreHDagg clumper(params);

    GreedyBspLocking sched;
    ScheduleClumps test(&clumper, &sched);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(ScheduleClumps_HillClimb_test) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = true;
    HCoreHDagg clumper(params);

    GreedyBspLocking sched;
    HillClimbingScheduler improver;
    ScheduleClumps test(&clumper, &sched, &improver);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(ScheduleClumps_HillClimb_test2) {
    HCoreHDagg_parameters params = HCoreHDagg_parameters();
    params.front_type = HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE;
    params.score_func = HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST;
    params.consider_future_score = true;
    HCoreHDagg clumper(params);

    GreedyBspLocking sched;
    HillClimbingScheduler improver;
    ScheduleClumps test(&clumper, &sched, &improver);
    run_test2(&test);
}

BOOST_AUTO_TEST_CASE(ReverseGreedyVarianceSchedulerFillup_test) {
    GreedyVarianceFillupScheduler var_sched;
    ReverseScheduler test(&var_sched);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(ReverseGreedyBSPLocking_test) {
    GreedyBspLocking lock_sched;
    ReverseScheduler test(&lock_sched);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Sarkar_test) {
    GreedyBspLocking sched;
    Sarkar test(&sched);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Sarkar_Improver_test) {
    GreedyBspLocking sched;
    HillClimbingScheduler improver;
    Sarkar test(&sched, &improver);
    run_test(&test);
}