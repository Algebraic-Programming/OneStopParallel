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

#define BOOST_TEST_MODULE BSP_MEM_SCHEDULERS
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <string>
#include <vector>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/pebbling_schedule_file_writer.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/pebbling/PebblingSchedule.hpp"

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

template <typename Graph_t>
void run_test(Scheduler<Graph_t> *test_scheduler) {
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
            std::string name_graph
                = filename_graph.substr(filename_machine.find_last_of("/\\") + 1, filename_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            BspInstance<Graph_t> instance;

            bool status_graph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filename_graph).string(),
                                                                                  instance.getComputationalDag());

            bool status_architecture
                = file_reader::readBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.getArchitecture());

            if (!status_graph || !status_architecture) {
                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspSchedule bsp_schedule(instance);

            RETURN_STATUS result = test_scheduler->computeSchedule(bsp_schedule);
            BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);

            std::vector<v_memw_t<Graph_t> > minimum_memory_required_vector
                = PebblingSchedule<Graph_t>::minimumMemoryRequiredPerNodeType(instance);
            v_memw_t<Graph_t> max_required
                = *std::max_element(minimum_memory_required_vector.begin(), minimum_memory_required_vector.end());
            instance.getArchitecture().setMemoryBound(max_required);

            PebblingSchedule<Graph_t> memSchedule1(bsp_schedule, PebblingSchedule<Graph_t>::CACHE_EVICTION_STRATEGY::LARGEST_ID);
            BOOST_CHECK_EQUAL(&memSchedule1.getInstance(), &instance);
            BOOST_CHECK(memSchedule1.isValid());

            PebblingSchedule<Graph_t> memSchedule3(bsp_schedule,
                                                   PebblingSchedule<Graph_t>::CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED);
            BOOST_CHECK(memSchedule3.isValid());

            PebblingSchedule<Graph_t> memSchedule5(bsp_schedule, PebblingSchedule<Graph_t>::CACHE_EVICTION_STRATEGY::FORESIGHT);
            BOOST_CHECK(memSchedule5.isValid());

            instance.getArchitecture().setMemoryBound(2 * max_required);

            PebblingSchedule<Graph_t> memSchedule2(bsp_schedule, PebblingSchedule<Graph_t>::CACHE_EVICTION_STRATEGY::LARGEST_ID);
            BOOST_CHECK(memSchedule2.isValid());

            PebblingSchedule<Graph_t> memSchedule4(bsp_schedule,
                                                   PebblingSchedule<Graph_t>::CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED);
            BOOST_CHECK(memSchedule4.isValid());

            PebblingSchedule<Graph_t> memSchedule6(bsp_schedule, PebblingSchedule<Graph_t>::CACHE_EVICTION_STRATEGY::FORESIGHT);
            BOOST_CHECK(memSchedule6.isValid());
        }
    }
}

BOOST_AUTO_TEST_CASE(GreedyBspScheduler_test) {
    GreedyBspScheduler<computational_dag_vector_impl_def_t> test;
    run_test(&test);
}

BOOST_AUTO_TEST_CASE(test_pebbling_schedule_writer) {
    using graph = computational_dag_vector_impl_def_int_t;

    BspInstance<graph> instance;
    instance.setNumberOfProcessors(3);
    instance.setCommunicationCosts(3);
    instance.setSynchronisationCosts(5);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(),
                                                                    instance.getComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertices(), 54);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertex_types(), 1);

    BspSchedule bsp_schedule(instance);
    GreedyBspScheduler<graph> scheduler;

    RETURN_STATUS result = scheduler.computeSchedule(bsp_schedule);
    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);

    std::vector<v_memw_t<graph> > minimum_memory_required_vector
        = PebblingSchedule<graph>::minimumMemoryRequiredPerNodeType(instance);
    v_memw_t<graph> max_required = *std::max_element(minimum_memory_required_vector.begin(), minimum_memory_required_vector.end());
    instance.getArchitecture().setMemoryBound(max_required + 3);

    PebblingSchedule<graph> memSchedule(bsp_schedule, PebblingSchedule<graph>::CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED);
    BOOST_CHECK(memSchedule.isValid());

    std::cout << "Writing pebbling schedule" << std::endl;
    file_writer::write_txt(std::cout, memSchedule);
}
