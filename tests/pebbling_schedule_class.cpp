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

std::vector<std::string> TinySpaaGraphs() {
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

std::vector<std::string> TestArchitectures() { return {"data/machine_params/p3.arch"}; }

template <typename GraphT>
void RunTest(Scheduler<GraphT> *testScheduler) {
    std::vector<std::string> filenamesGraph = TinySpaaGraphs();
    std::vector<std::string> filenamesArchitectures = TestArchitectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filenameGraph : filenamesGraph) {
        for (auto &filenameMachine : filenamesArchitectures) {
            std::string nameGraph = filenameGraph.substr(filenameMachine.find_last_of("/\\") + 1, filenameGraph.find_last_of("."));
            std::string nameMachine = filenameMachine.substr(filenameMachine.find_last_of("/\\") + 1);
            nameMachine = nameMachine.substr(0, nameMachine.rfind("."));

            std::cout << std::endl << "Graph: " << nameGraph << std::endl;
            std::cout << "Architecture: " << nameMachine << std::endl;

            BspInstance<GraphT> instance;

            bool statusGraph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(),
                                                                                 instance.GetComputationalDag());

            bool statusArchitecture
                = file_reader::ReadBspArchitecture((cwd / "data/machine_params/p3.arch").string(), instance.GetArchitecture());

            if (!statusGraph || !statusArchitecture) {
                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            BspSchedule bspSchedule(instance);

            ReturnStatus result = testScheduler->ComputeSchedule(bspSchedule);
            BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);

            std::vector<VMemwT<GraphT> > minimumMemoryRequiredVector
                = PebblingSchedule<GraphT>::minimumMemoryRequiredPerNodeType(instance);
            VMemwT<GraphT> maxRequired = *std::max_element(minimumMemoryRequiredVector.begin(), minimumMemoryRequiredVector.end());
            instance.GetArchitecture().setMemoryBound(maxRequired);

            PebblingSchedule<GraphT> memSchedule1(bspSchedule, PebblingSchedule<GraphT>::CACHE_EVICTION_STRATEGY::LARGEST_ID);
            BOOST_CHECK_EQUAL(&memSchedule1.GetInstance(), &instance);
            BOOST_CHECK(memSchedule1.isValid());

            PebblingSchedule<GraphT> memSchedule3(bspSchedule,
                                                  PebblingSchedule<GraphT>::CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED);
            BOOST_CHECK(memSchedule3.isValid());

            PebblingSchedule<GraphT> memSchedule5(bspSchedule, PebblingSchedule<GraphT>::CACHE_EVICTION_STRATEGY::FORESIGHT);
            BOOST_CHECK(memSchedule5.isValid());

            instance.GetArchitecture().setMemoryBound(2 * maxRequired);

            PebblingSchedule<GraphT> memSchedule2(bspSchedule, PebblingSchedule<GraphT>::CACHE_EVICTION_STRATEGY::LARGEST_ID);
            BOOST_CHECK(memSchedule2.isValid());

            PebblingSchedule<GraphT> memSchedule4(bspSchedule,
                                                  PebblingSchedule<GraphT>::CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED);
            BOOST_CHECK(memSchedule4.isValid());

            PebblingSchedule<GraphT> memSchedule6(bspSchedule, PebblingSchedule<GraphT>::CACHE_EVICTION_STRATEGY::FORESIGHT);
            BOOST_CHECK(memSchedule6.isValid());
        }
    }
}

BOOST_AUTO_TEST_CASE(GreedyBspSchedulerTest) {
    GreedyBspScheduler<ComputationalDagVectorImplDefT> test;
    RunTest(&test);
}

BOOST_AUTO_TEST_CASE(TestPebblingScheduleWriter) {
    using Graph = computational_dag_vector_impl_def_int_t;

    BspInstance<Graph> instance;
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
                                                                    instance.GetComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.GetComputationalDag().NumVertices(), 54);
    BOOST_CHECK_EQUAL(instance.GetComputationalDag().NumVertexTypes(), 1);

    BspSchedule bspSchedule(instance);
    GreedyBspScheduler<Graph> scheduler;

    ReturnStatus result = scheduler.ComputeSchedule(bspSchedule);
    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);

    std::vector<VMemwT<Graph> > minimumMemoryRequiredVector = PebblingSchedule<Graph>::minimumMemoryRequiredPerNodeType(instance);
    VMemwT<Graph> maxRequired = *std::max_element(minimumMemoryRequiredVector.begin(), minimumMemoryRequiredVector.end());
    instance.GetArchitecture().setMemoryBound(maxRequired + 3);

    PebblingSchedule<Graph> memSchedule(bspSchedule, PebblingSchedule<Graph>::CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED);
    BOOST_CHECK(memSchedule.isValid());

    std::cout << "Writing pebbling schedule" << std::endl;
    file_writer::write_txt(std::cout, memSchedule);
}
