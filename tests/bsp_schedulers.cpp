#define BOOST_TEST_MODULE BSP_SCHEDULERS
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "scheduler/ReverseScheduler.hpp"
#include "scheduler/Coarsers/CacheLineGluer.hpp"
#include "scheduler/Coarsers/HDaggCoarser.hpp"
#include "scheduler/Coarsers/SquashA.hpp"
#include "scheduler/Coarsers/WavefrontCoarser.hpp"
#include "scheduler/Coarsers/TreesUnited.hpp"
#include "scheduler/Coarsers/FastLane.hpp"
#include "scheduler/Coarsers/Funnel.hpp"
#include "scheduler/Coarsers/FunnelBfs.hpp"
#include "scheduler/Coarsers/ScheduleClumps.hpp"
#include "scheduler/Coarsers/Sarkar.hpp"
#include "scheduler/ContractRefineScheduler/BalDMixR.hpp"
#include "scheduler/ContractRefineScheduler/CoBalDMixR.hpp"
#include "scheduler/ContractRefineScheduler/MultiLevelHillClimbing.hpp"
#include "scheduler/GreedySchedulers/ClassicSchedule.hpp"
#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyBspFillupScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyBspLocking.hpp"
#include "scheduler/GreedySchedulers/GreedyBspStoneAge.hpp"
#include "scheduler/GreedySchedulers/GreedyBspGrowLocal.hpp"
#include "scheduler/GreedySchedulers/GreedyBspGrowLocalAutoCores.hpp"
#include "scheduler/GreedySchedulers/GreedyBspGrowLocalParallel.hpp"
#include "scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "scheduler/GreedySchedulers/GreedyCilkScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyEtfScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyLayers.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceFillupScheduler.hpp"
#include "scheduler/GreedySchedulers/MetaGreedyScheduler.hpp"
#include "scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "scheduler/HDagg/HDagg_simple.hpp"
#include "scheduler/HDagg/HCoreHDagg.hpp"
#include "scheduler/LocalSearchSchedulers/HillClimbingScheduler.hpp"
#include "scheduler/Scheduler.hpp"
#include "scheduler/Serial/Serial.hpp"
#include "scheduler/SubArchitectureSchedulers/SubArchitectures.hpp"
#include "scheduler/Wavefront/Wavefront.hpp"
#include "file_interactions/FileReader.hpp"

std::vector<std::string> test_graphs() {
    return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt", "data/spaa/test/test1.txt", "data/spaa/test/test2.txt" };
}

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt", "data/machine_params/p16_g1_l5_numa2.txt", "data/machine_params/p3_g2_l100.txt"};
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


void add_mem_weights(ComputationalDag &dag) {
    int weight = 1;
    for (const auto &v : dag.vertices()) {
        dag.setNodeMemoryWeight(v, weight++ % 3 + 1);
        dag.setNodeCommunicationWeight(v, weight++ % 3 + 1);
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

            std::cout << std::endl << "Scheduler: " << test_scheduler->getScheduleName() << std::endl; 
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            auto [status_graph, graph] =
                FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            std::cout << "Vertices " << graph.numberOfVertices() << std::endl;
            std::cout << "Edges " << graph.numberOfEdges() << std::endl;

            BspInstance instance(graph, architecture);

            std::pair<RETURN_STATUS, BspSchedule> result = test_scheduler->computeSchedule(instance);

            print_bsp_schedule(result.second);

            BOOST_CHECK_EQUAL(SUCCESS, result.first);
            BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
            BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
            BOOST_CHECK(result.second.hasValidCommSchedule());
        }
    }
};

void run_test_memory(Scheduler *test_scheduler) {
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

            auto [status_graph, graph] =
                FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            add_mem_weights(graph);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            architecture.setMemoryConstraintType(LOCAL);

            BspInstance instance(graph, architecture);

            test_scheduler->setUseMemoryConstraint(true);

            const std::vector<unsigned> bounds_to_test = {5, 10, 20, 50, 100};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                std::pair<RETURN_STATUS, BspSchedule> result = test_scheduler->computeSchedule(instance);

                std::cout << "Memory bound: " << bound << std::endl; 
                print_bsp_schedule(result.second);

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
                BOOST_CHECK(result.second.hasValidCommSchedule());
                BOOST_CHECK(result.second.satisfiesMemoryConstraints());
            }
        }
    }
};


void run_test_memory_local_inc(Scheduler *test_scheduler) {
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

            auto [status_graph, graph] =
                FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            add_mem_weights(graph);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            architecture.setMemoryConstraintType(LOCAL_IN_OUT);

            BspInstance instance(graph, architecture);

            test_scheduler->setUseMemoryConstraint(true);

            unsigned max_in_degree = 0;
            for(const auto &node : graph.vertices()) {
             
                max_in_degree = std::max(max_in_degree, graph.numberOfParents(node));

            }

            max_in_degree *= 3;

            const std::vector<unsigned> bounds_to_test = {max_in_degree, 2 * max_in_degree, 3 * max_in_degree};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                std::pair<RETURN_STATUS, BspSchedule> result = test_scheduler->computeSchedule(instance);

                std::cout << "Memory bound: " << bound << std::endl; 
                print_bsp_schedule(result.second);

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
                BOOST_CHECK(result.second.hasValidCommSchedule());
                BOOST_CHECK(result.second.satisfiesMemoryConstraints());
            }
        }
    }
};

void run_test_memory_local_comm(Scheduler *test_scheduler) {
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

            auto [status_graph, graph] =
                FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            add_mem_weights(graph);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            architecture.setMemoryConstraintType(LOCAL_INC_EDGES);

            BspInstance instance(graph, architecture);

            test_scheduler->setUseMemoryConstraint(true);

            unsigned max_in_degree = 0;
            for(const auto &node : graph.vertices()) {
             
                max_in_degree = std::max(max_in_degree, graph.numberOfParents(node));

            }
  
            max_in_degree++;
            max_in_degree *= 3;

            const std::vector<unsigned> bounds_to_test = {max_in_degree, 2 * max_in_degree, 3 * max_in_degree};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                std::pair<RETURN_STATUS, BspSchedule> result = test_scheduler->computeSchedule(instance);

                std::cout << "Memory bound: " << bound << std::endl; 
                print_bsp_schedule(result.second);

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
                BOOST_CHECK(result.second.hasValidCommSchedule());
                BOOST_CHECK(result.second.satisfiesMemoryConstraints());
            }
        }
    }
};

void run_test_memory_local_comm_2(Scheduler *test_scheduler) {
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

            auto [status_graph, graph] =
                FileReader::readComputationalDagHyperdagFormat((cwd / filename_graph).string());
            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            add_mem_weights(graph);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            architecture.setMemoryConstraintType(LOCAL_INC_EDGES_2);

            BspInstance instance(graph, architecture);

            test_scheduler->setUseMemoryConstraint(true);

            unsigned max_in_degree = 0;
            for(const auto &node : graph.vertices()) {
             
                max_in_degree = std::max(max_in_degree, graph.numberOfParents(node));

            }
  
            max_in_degree;
            max_in_degree *= 3;

            const std::vector<unsigned> bounds_to_test = {max_in_degree, 2 * max_in_degree, 3 * max_in_degree};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                std::pair<RETURN_STATUS, BspSchedule> result = test_scheduler->computeSchedule(instance);

                std::cout << "Memory bound: " << bound << std::endl; 
                print_bsp_schedule(result.second);

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
                BOOST_CHECK(result.second.hasValidCommSchedule());
                BOOST_CHECK(result.second.satisfiesMemoryConstraints());
            }
        }
    }
};

BOOST_AUTO_TEST_CASE(GreedyBspLocking_memory_test_inc_inc_all) {
    GreedyBspLocking test;
    run_test_memory(&test);
    run_test_memory_local_inc(&test);
    run_test_memory_local_comm(&test);
    run_test_memory_local_comm_2(&test);
}


BOOST_AUTO_TEST_CASE(GreedyBspScheduler_memory_test) {
    GreedyBspScheduler test;
    run_test_memory(&test);
}

BOOST_AUTO_TEST_CASE(GreedyVarianceScheduler_memory_test) {
    GreedyVarianceScheduler test;
    run_test_memory(&test);
}

BOOST_AUTO_TEST_CASE(Serial_test) {
    Serial test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(BalDMixR_test) {
    BalDMixR test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(CoBalDMixR_test) {
    CoBalDMixR test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_test) {
    GreedyBspScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBspFillupScheduler_test) {
    GreedyBspFillupScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyCilkScheduler_test) {
    GreedyCilkScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyEtfScheduler_test) {
    GreedyEtfScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(MetaGreedyScheduler_test) {
    MetaGreedyScheduler test;
    run_test(&test);
}


BOOST_AUTO_TEST_CASE(RandomGreedy_test) {
    RandomGreedy test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyChildren_test) {
    GreedyChildren test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyLayers_test) {
    GreedyLayers test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(CacheLineGluer_test) {
    BalDMixR sched;
    unsigned cacheline_size_ = 2;
    CacheLineGluer test(&sched, cacheline_size_);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(SquashA_test) {
    BalDMixR sched;
    SquashA test(&sched);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(SquashA_HillClimb_test) {
    BalDMixR sched;
    HillClimbingScheduler improver;
    SquashA test(&sched, &improver);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HDagg_wo_Hungarian_test) {
    HDagg_parameters para(1.1, 5, false);
    HDagg_simple test(para);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HDagg_w_Hungarian_test) {
    HDagg_parameters para(1.1, 0, true);
    HDagg_simple test(para);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HDagg_original) {
    HDagg_parameters para(1.1, 0, false);
    HDagg_simple hdagg(para);
    HDaggCoarser test(&hdagg);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HDagg_original_xlogx) {
    HDagg_parameters para(0.003, 0, false, HDagg_parameters::XLOGX);
    HDagg_simple hdagg(para);
    HDaggCoarser test(&hdagg);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(SubArch0_HDagg_w_Hungarian_test) {
    HDagg_parameters para(1.1, 0, true);
    HDagg_simple hdag_sched(para);
    SubArchitectureScheduler test(&hdag_sched, 2, true);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(SubArch1_HDagg_w_Hungarian_test) {
    HDagg_parameters para(1.1, 0, true);
    HDagg_simple hdag_sched(para);
    SubArchitectureScheduler test(&hdag_sched, 0, true);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(SubArch2_HDagg_w_Hungarian_test) {
    HDagg_parameters para(1.1, 0, true);
    HDagg_simple hdag_sched(para);
    SubArchitectureScheduler test(&hdag_sched, 0, false);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(MultiLevelHillClimbingScheduler_test) {
    MultiLevelHillClimbingScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyVarianceScheduler_test) {
    GreedyVarianceScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyVarianceSchedulerFillup_test) {
    GreedyVarianceFillupScheduler test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBSPLocking_test) {
    GreedyBspLocking test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Wavefront_test) {
    Wavefront test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(WavefrontCoarser_test) {
    Wavefront test_sched;
    HillClimbingScheduler test_improv;
    WavefrontCoarser test(&test_sched, &test_improv);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(HDaggCoarser_BSP_test) {
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    HDaggCoarser test(&test_sched, &test_improv);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(TreesUnited_test111) {
    TreesUnited_parameters  params(true, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    TreesUnited test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(TreesUnited_test110) {
    TreesUnited_parameters  params(true, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    TreesUnited test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(TreesUnited_test101) {
    TreesUnited_parameters  params(true, false, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    TreesUnited test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(TreesUnited_test100) {
    TreesUnited_parameters  params(true, false, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    TreesUnited test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(TreesUnited_test011) {
    TreesUnited_parameters  params(false, true, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    TreesUnited test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(TreesUnited_test010) {
    TreesUnited_parameters  params(false, true, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    TreesUnited test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(TreesUnited_test001) {
    TreesUnited_parameters  params(false, false, true);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    TreesUnited test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(TreesUnited_test000) {
    TreesUnited_parameters  params(false, false, false);
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    TreesUnited test(&test_sched, &test_improv, params);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(FastLane_test) {
    GreedyBspScheduler test_sched;
    HillClimbingScheduler test_improv;
    FastLane test(&test_sched, &test_improv);
    run_test(&test);
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

BOOST_AUTO_TEST_CASE(GreedyBspStoneAge_test) {
    GreedyBspStoneAge test;
    run_test(&test);
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

BOOST_AUTO_TEST_CASE(GreedyBspGrowLocal_test) {
    GreedyBspGrowLocal test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Sarkar_test) {
    Wavefront sched;
    Sarkar test(&sched);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Sarkar_Improver_test) {
    Wavefront sched;
    HillClimbingScheduler improver;
    Sarkar test(&sched, &improver);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GrowLocalParallel_test) {
    GreedyBspGrowLocalParallel test;
    run_test(&test);
} 


BOOST_AUTO_TEST_CASE(GrowLocalAutoCores_test) {
    GreedyBspGrowLocalAutoCores test;
    run_test(&test);
}
