#define BOOST_TEST_MODULE BSP_SCHEDULERS
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/Serial.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "io/arch_file_reader.hpp"
#include "io/graph_file_reader.hpp"

using namespace osp;

std::vector<std::string> test_graphs() {
    return {"data/spaa/tiny/instance_bicgstab.txt", "data/spaa/tiny/instance_CG_N2_K2_nzP0d75.txt"};
}

std::vector<std::string> test_architectures() { return {"data/machine_params/p3.txt"}; }

template<typename Graph_t>
void add_mem_weights(Graph_t &dag) {
    int weight = 1;
    for (const auto &v : dag.vertices()) {
        dag.set_vertex_mem_weight(v, weight++ % 3 + 1);
        dag.set_vertex_comm_weight(v, weight++ % 3 + 1);
    }
}

template<typename Graph_t>
void run_test_local_memory(Scheduler<Graph_t> *test_scheduler) {
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

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                                instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.txt").string(),
                                                                        instance.getArchitecture());

            add_mem_weights(instance.getComputationalDag());
            instance.getArchitecture().setMemoryConstraintType(LOCAL);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<v_memw_t<Graph_t>> bounds_to_test = {5, 10, 20, 50, 100};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                std::pair<RETURN_STATUS, BspSchedule<Graph_t>> result = test_scheduler->computeSchedule(instance);

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
                BOOST_CHECK(result.second.hasValidCommSchedule());
                BOOST_CHECK(result.second.satisfiesMemoryConstraints());
            }
        }
    }
};

template<typename Graph_t>
void run_test_persistent_transient_memory(Scheduler<Graph_t> *test_scheduler) {
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

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                                instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.txt").string(),
                                                                        instance.getArchitecture());

            add_mem_weights(instance.getComputationalDag());
            instance.getArchitecture().setMemoryConstraintType(PERSISTENT_AND_TRANSIENT);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<v_memw_t<Graph_t>> bounds_to_test = {50, 100};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                std::pair<RETURN_STATUS, BspSchedule<Graph_t>> result = test_scheduler->computeSchedule(instance);

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
                BOOST_CHECK(result.second.hasValidCommSchedule());
                BOOST_CHECK(result.second.satisfiesMemoryConstraints());
            }
        }
    }
};

template<typename Graph_t>
void run_test_local_in_out_memory(Scheduler<Graph_t> *test_scheduler) {
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

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                                instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.txt").string(),
                                                                        instance.getArchitecture());

            add_mem_weights(instance.getComputationalDag());
            instance.getArchitecture().setMemoryConstraintType(LOCAL_IN_OUT);

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            const std::vector<v_memw_t<Graph_t>> bounds_to_test = {5, 10, 20, 50, 100};

            for (const auto &bound : bounds_to_test) {

                instance.getArchitecture().setMemoryBound(bound);

                std::pair<RETURN_STATUS, BspSchedule<Graph_t>> result = test_scheduler->computeSchedule(instance);

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
                BOOST_CHECK(result.second.hasValidCommSchedule());
                BOOST_CHECK(result.second.satisfiesMemoryConstraints());
            }
        }
    }
};

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_local_test) {

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t,
                       local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>>
        test;
    run_test_local_memory(&test);
};

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_local_in_out_test) {

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t,
                       local_in_out_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>>
        test;
    run_test_local_in_out_memory(&test);
};

BOOST_AUTO_TEST_CASE(BspLocking_local_in_out_test) {

    BspLocking<computational_dag_edge_idx_vector_impl_def_t,
               local_in_out_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>>
        test;
    run_test_local_in_out_memory(&test);
};

BOOST_AUTO_TEST_CASE(BspLocking_local_test) {

    BspLocking<computational_dag_edge_idx_vector_impl_def_t,
               local_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>>
        test;
    run_test_local_memory(&test);
};

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_persistent_transient_test) {

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t,
                       persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>>
        test;
    run_test_persistent_transient_memory(&test);
};

BOOST_AUTO_TEST_CASE(EtfScheduler_persistent_transient_test) {

    EtfScheduler<computational_dag_edge_idx_vector_impl_def_t,
                 persistent_transient_memory_constraint<computational_dag_edge_idx_vector_impl_def_t>>
        test;
    run_test_persistent_transient_memory(&test);
};