#define BOOST_TEST_MODULE kl
#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "file_interactions/FileReader.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceFillupScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyBspLocking.hpp"
#include "scheduler/Coarsers/FunnelBfs.hpp"
#include "scheduler/LocalSearchSchedulers/KernighanLin/kl_base.hpp"
#include "scheduler/LocalSearchSchedulers/KernighanLin/kl_total_comm.hpp"
#include "scheduler/LocalSearchSchedulers/KernighanLin/kl_total_cut.hpp"


std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt", "data/machine_params/p4_g3_l5.txt"};
}

std::vector<std::string> test_graphs_2() {
    return {"data/spaa/small/instance_exp_N20_K4_nzP0d2.txt", "data/spaa/test/test1.txt"};
}

std::vector<std::string> test_graphs() {
    return {
        "data/spaa/tiny/instance_k-means.txt", 
            "data/spaa/tiny/instance_bicgstab.txt",
            "data/spaa/tiny/instance_CG_N3_K1_nzP0d5.txt",
           "data/spaa/small/instance_CG_N7_K2_nzP0d6.txt"
            };
}


void add_mem_weights(ComputationalDag &dag) {
    int weight = 1;
    for (const auto &v : dag.vertices()) {
        dag.setNodeMemoryWeight(v, weight++ % 3 + 1);
        dag.setNodeCommunicationWeight(v, weight++ % 3 + 1);
    }
}


BOOST_AUTO_TEST_CASE(kl_memory_test_local_inc) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = test_graphs_2();
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

            add_mem_weights(graph);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            architecture.setSynchronisationCosts(80);
            architecture.setMemoryConstraintType(LOCAL_INC_EDGES);

            BspInstance instance(graph, architecture);

            unsigned max_in_degree = 0;
            for(const auto &node : graph.vertices()) {
             
                max_in_degree = std::max(max_in_degree, graph.numberOfParents(node));

            }
  
            max_in_degree *= 3;

            const std::vector<unsigned> bounds_to_test = {max_in_degree + 1, max_in_degree + 4};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                GreedyBspLocking greedy_scheduler;
                greedy_scheduler.setUseMemoryConstraint(true);

                auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

                BOOST_CHECK_EQUAL(status, SUCCESS);
                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());

                kl_total_comm_test kl;
                kl.setUseMemoryConstraint(true);

                kl.improve_schedule_test_2(schedule);

                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());

                auto [status_2, schedule_2] = greedy_scheduler.computeSchedule(instance);

                BOOST_CHECK_EQUAL(status_2, SUCCESS);
                BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

                kl_total_cut_test kl_cut;
                kl_cut.setUseMemoryConstraint(true);

                kl_cut.improve_schedule_test_2(schedule_2);

                BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

                
            }
        }
    }
};


BOOST_AUTO_TEST_CASE(kl_memory_test_local_inc_2) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = test_graphs_2();
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

            add_mem_weights(graph);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            architecture.setSynchronisationCosts(80);
            architecture.setMemoryConstraintType(LOCAL_INC_EDGES_2);

            BspInstance instance(graph, architecture);

            unsigned max_in_degree = 0;
            for(const auto &node : graph.vertices()) {
             
                max_in_degree = std::max(max_in_degree, graph.numberOfParents(node));

            }
  
            max_in_degree *= 3;

            const std::vector<unsigned> bounds_to_test = {max_in_degree + 1, max_in_degree + 4};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                GreedyBspLocking greedy_scheduler;
                greedy_scheduler.setUseMemoryConstraint(true);

                auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

                BOOST_CHECK_EQUAL(status, SUCCESS);
                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());

                kl_total_comm_test kl;
                kl.setUseMemoryConstraint(true);

                kl.improve_schedule_test_2(schedule);

                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());

                auto [status_2, schedule_2] = greedy_scheduler.computeSchedule(instance);

                BOOST_CHECK_EQUAL(status_2, SUCCESS);
                BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

                kl_total_cut_test kl_cut;
                kl_cut.setUseMemoryConstraint(true);

                kl_cut.improve_schedule_test_2(schedule_2);

                BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

                
            }
        }
    }
};



BOOST_AUTO_TEST_CASE(kl_memory_test_local) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = test_graphs_2();
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

            add_mem_weights(graph);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            architecture.setMemoryConstraintType(LOCAL);
            architecture.setSynchronisationCosts(80);

            BspInstance instance(graph, architecture);

            const std::vector<unsigned> bounds_to_test = {10, 20};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                GreedyBspLocking greedy_scheduler;
                greedy_scheduler.setUseMemoryConstraint(true);

                auto [status, schedule] = greedy_scheduler.computeSchedule(instance);

                BOOST_CHECK_EQUAL(status, SUCCESS);
                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());

                kl_total_comm_test kl;
                kl.setUseMemoryConstraint(true);

                kl.improve_schedule_test_2(schedule);

                BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule.satisfiesMemoryConstraints());

                auto [status_2, schedule_2] = greedy_scheduler.computeSchedule(instance);

                BOOST_CHECK_EQUAL(status_2, SUCCESS);
                BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());

                kl_total_cut_test kl_cut;
                kl_cut.setUseMemoryConstraint(true);

                kl_cut.improve_schedule_test_2(schedule_2);

                BOOST_CHECK(schedule_2.satisfiesPrecedenceConstraints());
                BOOST_CHECK(schedule_2.satisfiesMemoryConstraints());
               
            }
        }
    }
};

