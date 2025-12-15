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

#define BOOST_TEST_MODULE MAXBSP_SCHEDULERS
#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspToMaxBspConverter.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing_for_comm_schedule.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(MaxbspScheduling) {
    using Graph = ComputationalDagVectorImplDefT;

    BspInstance<Graph> instance;
    instance.setNumberOfProcessors(4);
    instance.setCommunicationCosts(3);
    instance.setSynchronisationCosts(3);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag").string(), instance.GetComputationalDag());

    BOOST_CHECK(status);

    GreedyBspScheduler<Graph> greedy;
    BspSchedule<Graph> bspInitial(instance);
    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, greedy.ComputeSchedule(bspInitial));
    BOOST_CHECK(bspInitial.SatisfiesPrecedenceConstraints());

    // PART I: from BspSchedule to MaxBspSchedule conversion

    std::cout << "Original Bsp Cost: " << bspInitial.computeCosts() << std::endl;
    GreedyBspToMaxBspConverter<Graph> converter;
    MaxBspSchedule<Graph> maxbsp = converter.Convert(bspInitial);
    BOOST_CHECK(maxbsp.SatisfiesPrecedenceConstraints());
    auto costConversion = maxbsp.computeCosts();
    std::cout << "Cost after maxBsp conversion: " << costConversion << std::endl;

    // hill climbing

    HillClimbingScheduler<Graph> hc;
    hc.ImproveSchedule(maxbsp);
    BOOST_CHECK(maxbsp.SatisfiesPrecedenceConstraints());
    auto costHc = maxbsp.computeCosts();
    std::cout << "Cost after Hill Climbing: " << costHc << std::endl;
    BOOST_CHECK(costHc <= costConversion);

    // PART II: from BspScheduleCS to MaxBspScheduleCS conversion

    BspScheduleCS<Graph> bspInitialCs(bspInitial);
    BOOST_CHECK(bspInitialCs.hasValidCommSchedule());
    std::cout << "Original BspCS Cost: " << bspInitialCs.computeCosts() << std::endl;

    MaxBspScheduleCS<Graph> maxbspCs = converter.Convert(bspInitialCs);
    BOOST_CHECK(maxbspCs.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(maxbspCs.hasValidCommSchedule());
    auto costConversionCs = maxbspCs.computeCosts();
    std::cout << "Cost after maxBsp(CS) conversion: " << costConversionCs << std::endl;

    // hill climbing for comm. schedule

    HillClimbingForCommSteps<Graph> hCcs;
    hCcs.ImproveSchedule(maxbspCs);
    BOOST_CHECK(maxbspCs.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(maxbspCs.hasValidCommSchedule());
    auto costHccs = maxbspCs.computeCosts();
    std::cout << "Cost after comm. sched. hill climbing: " << costHccs << std::endl;
    BOOST_CHECK(costHccs <= costConversionCs);

    // PART III: same for larger DAG

    status = file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/large/instance_CG_N24_K22_nzP0d2.hdag").string(),
                                                               instance.GetComputationalDag());

    BOOST_CHECK(status);
    instance.setSynchronisationCosts(7);

    BspSchedule<Graph> bspInitialLarge(instance);
    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, greedy.ComputeSchedule(bspInitialLarge));

    BspScheduleCS<Graph> bspInitialLargeCs(bspInitialLarge);
    BOOST_CHECK(bspInitialLargeCs.hasValidCommSchedule());
    std::cout << "Original Bsp Cost on large DAG: " << bspInitialLargeCs.computeCosts() << std::endl;

    MaxBspScheduleCS<Graph> maxbspCsLarge = converter.Convert(bspInitialLargeCs);
    BOOST_CHECK(maxbspCsLarge.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(maxbspCsLarge.hasValidCommSchedule());
    auto costMaxbspCsLarge = maxbspCsLarge.computeCosts();
    std::cout << "Cost after maxBsp conversion on large DAG: " << costMaxbspCsLarge << std::endl;

    hCcs.ImproveSchedule(maxbspCsLarge);
    BOOST_CHECK(maxbspCsLarge.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(maxbspCsLarge.hasValidCommSchedule());
    auto costHccsLarge = maxbspCsLarge.computeCosts();
    std::cout << "Cost after comm. sched. hill climbing on large DAG: " << costHccsLarge << std::endl;
    BOOST_CHECK(costHccsLarge <= costMaxbspCsLarge);
}
