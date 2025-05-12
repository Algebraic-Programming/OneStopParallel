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

#include "bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "bsp/scheduler/GreedySchedulers/CilkScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "bsp/scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "bsp/scheduler/GreedySchedulers/VarianceFillup.hpp"
#include "bsp/scheduler/Serial.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "io/arch_file_reader.hpp"
#include "io/hdag_graph_file_reader.hpp"

using namespace osp;

std::vector<std::string> tiny_spaa_graphs() {
    return {"data/spaa/tiny/instance_bicgstab.hdag",
            "data/spaa/tiny/instance_CG_N2_K2_nzP0d75.hdag",
            "data/spaa/tiny/instance_CG_N3_K1_nzP0d5.hdag",
            "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag",
            "data/spaa/tiny/instance_exp_N4_K2_nzP0d5.hdag",
            "data/spaa/tiny/instance_exp_N5_K3_nzP0d4.hdag",
            "data/spaa/tiny/instance_exp_N6_K4_nzP0d25.hdag",
            "data/spaa/tiny/instance_k-means.hdag",
            "data/spaa/tiny/instance_k-NN_3_gyro_m.hdag",
            "data/spaa/tiny/instance_kNN_N4_K3_nzP0d5.hdag",
            "data/spaa/tiny/instance_kNN_N5_K3_nzP0d3.hdag",
            "data/spaa/tiny/instance_kNN_N6_K4_nzP0d2.hdag",
            "data/spaa/tiny/instance_pregel.hdag",
            "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag",
            "data/spaa/tiny/instance_spmv_N7_nzP0d35.hdag",
            "data/spaa/tiny/instance_spmv_N10_nzP0d25.hdag"};
}

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

            bool status_graph = file_reader::readComputationalDagHyperdagFormat((cwd / filename_graph).string(),
                                                                                instance.getComputationalDag());
            bool status_architecture = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(),
                                                                        instance.getArchitecture());

            if (!status_graph || !status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspSchedule<Graph_t> schedule(instance);
            const auto result = test_scheduler->computeSchedule(schedule);

            BOOST_CHECK_EQUAL(SUCCESS, result);
            BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        }
    }
};

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_test) {

    GreedyBspScheduler<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_test_2) {

    GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(Serial_test) {

    Serial<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(cilk_test_1) {

    CilkScheduler<computational_dag_vector_impl_def_t> test;
    test.setMode(CILK);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(cilk_test_2) {

    CilkScheduler<computational_dag_vector_impl_def_t> test;
    test.setMode(SJF);
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(etf_test) {

    EtfScheduler<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(random_test) {

    RandomGreedy<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(children_test) {

    GreedyChildren<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(locking_test) {

    BspLocking<computational_dag_vector_impl_def_int_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(variancefillup_test) {

    VarianceFillup<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(etf_test_edge_desc_impl) {

    EtfScheduler<computational_dag_edge_idx_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(grow_local_auto_test_edge_desc_impl) {

    GrowLocalAutoCores<computational_dag_edge_idx_vector_impl_def_t> test;
    run_test(&test);
}