/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#define BOOST_TEST_MODULE BSP_SCHEDULERS
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>


#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyVarianceSspScheduler.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/bsp/scheduler/MaxBspScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "test_graphs.hpp"

using namespace osp;

std::vector<std::string> test_architectures() { return {"data/machine_params/p3.arch"}; }

template<typename Graph_t>
void run_test(Scheduler<Graph_t> *test_scheduler) {
    // static_assert(std::is_base_of<Scheduler, T>::value, "Class is not a scheduler!");
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();
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

            bool status_graph = file_reader::readGraph((cwd / filename_graph).string(),
                                                                                instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                        instance.getArchitecture());

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspSchedule<Graph_t> schedule(instance);
            const auto result = test_scheduler->computeSchedule(schedule);

            BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
            BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
        }
    }
};

template<typename Graph_t>
void run_test_max_bsp(MaxBspScheduler<Graph_t>* test_scheduler) {
    std::vector<std::string> filenames_graph = tiny_spaa_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Locate project root
    std::filesystem::path cwd = std::filesystem::current_path();
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
    }

    for (auto& filename_graph : filenames_graph) {
        for (auto& filename_machine : filenames_architectures) {
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl
                      << "Scheduler (MaxBsp): " << test_scheduler->getScheduleName() << std::endl
                      << "Graph: " << name_graph << std::endl
                      << "Architecture: " << name_machine << std::endl;

            computational_dag_edge_idx_vector_impl_def_int_t graph;
            BspArchitecture<Graph_t> arch;

            bool status_graph = file_reader::readGraph((cwd / filename_graph).string(), graph);
            bool status_architecture =
                file_reader::readBspArchitecture((cwd / filename_machine).string(), arch);

            BOOST_REQUIRE_MESSAGE(status_graph, "Failed to read graph: " << filename_graph);
            BOOST_REQUIRE_MESSAGE(status_architecture, "Failed to read architecture: " << filename_machine);

            BspInstance<Graph_t> instance(graph, arch);

            MaxBspSchedule<Graph_t> schedule(instance);

            const auto result = test_scheduler->computeSchedule(schedule);

            BOOST_CHECK_EQUAL(result, RETURN_STATUS::OSP_SUCCESS);
            BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
        }
    }
}

// Tests computeSchedule(BspSchedule&) → staleness = 1
BOOST_AUTO_TEST_CASE(GreedyVarianceSspScheduler_test_vector_impl) {
    GreedyVarianceSspScheduler<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

// Tests computeSchedule(BspSchedule&) → staleness = 1 (different graph impl)
BOOST_AUTO_TEST_CASE(GreedyVarianceSspScheduler_test_edge_idx_impl) {
    GreedyVarianceSspScheduler<computational_dag_edge_idx_vector_impl_def_t> test;
    run_test(&test);
}

// Tests computeSchedule(MaxBspSchedule&) → staleness = 2
BOOST_AUTO_TEST_CASE(GreedyVarianceSspScheduler_MaxBspSchedule_large_test) {
    GreedyVarianceSspScheduler<computational_dag_edge_idx_vector_impl_def_int_t> test;
    run_test_max_bsp(&test);
}
