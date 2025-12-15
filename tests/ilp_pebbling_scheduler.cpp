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

#define BOOST_TEST_MODULE PEBBLING_ILP
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <iostream>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "osp/pebbling/pebblers/pebblingILP/MultiProcessorPebbling.hpp"
#include "osp/pebbling/pebblers/pebblingILP/PebblingPartialILP.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(TestFull) {
    using graph = ComputationalDagVectorImplDefUnsignedT;

    BspInstance<graph> instance;
    instance.SetNumberOfProcessors(4);
    instance.SetCommunicationCosts(3);
    instance.SetSynchronisationCosts(5);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::ReadComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag").string(), instance.GetComputationalDag());

    BOOST_CHECK(status);

    GreedyBspScheduler<graph> greedy;
    BspSchedule<graph> bspInitial(instance);
    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, greedy.ComputeSchedule(bsp_initial));

    std::vector<VMemwT<graph> > minimumMemoryRequiredVector = PebblingSchedule<graph>::minimumMemoryRequiredPerNodeType(instance);
    VMemwT<graph> maxRequired = *std::max_element(minimum_memory_required_vector.begin(), minimum_memory_required_vector.end());
    instance.GetArchitecture().setMemoryBound(max_required);

    PebblingSchedule<graph> initialSol(bsp_initial, PebblingSchedule<graph>::CACHE_EVICTION_STRATEGY::FORESIGHT);
    BOOST_CHECK(initialSol.isValid());

    MultiProcessorPebbling<graph> mpp;
    mpp.SetTimeLimitSeconds(10);
    PebblingSchedule<graph> schedule(instance);
    mpp.computePebblingWithInitialSolution(initial_sol, schedule);
    schedule.cleanSchedule();
    BOOST_CHECK(schedule.isValid());
};

BOOST_AUTO_TEST_CASE(TestPartial) {
    using graph = ComputationalDagVectorImplDefUnsignedT;

    BspInstance<graph> instance;
    instance.SetNumberOfProcessors(2);
    instance.SetCommunicationCosts(3);
    instance.SetSynchronisationCosts(5);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::ReadComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_spmv_N10_nzP0d25.hdag").string(), instance.GetComputationalDag());

    BOOST_CHECK(status);

    std::vector<VMemwT<graph> > minimumMemoryRequiredVector = PebblingSchedule<graph>::minimumMemoryRequiredPerNodeType(instance);
    VMemwT<graph> maxRequired = *std::max_element(minimum_memory_required_vector.begin(), minimum_memory_required_vector.end());
    instance.GetArchitecture().setMemoryBound(max_required);

    PebblingPartialILP<graph> mpp;
    mpp.setMinSize(15);
    mpp.setSecondsForSubILP(5);
    PebblingSchedule<graph> schedule(instance);
    mpp.computePebbling(schedule);
    BOOST_CHECK(schedule.isValid());
};
