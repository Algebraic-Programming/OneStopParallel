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

#define BOOST_TEST_MODULE HILL_CLIMBING
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"

#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing_for_comm_schedule.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(HillClimbing) {
    using Graph = computational_dag_vector_impl_def_t;

    BspInstance<Graph> instance;
    instance.setNumberOfProcessors(4);
    instance.setCommunicationCosts(3);
    instance.setSynchronisationCosts(5);

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
    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, greedy.computeSchedule(bspInitial));
    BOOST_CHECK_EQUAL(bspInitial.satisfiesPrecedenceConstraints(), true);

    HillClimbingScheduler<Graph> scheduler;
    BspSchedule<Graph> schedule1 = bspInitial;
    scheduler.improveSchedule(schedule1);
    BOOST_CHECK_EQUAL(schedule1.satisfiesPrecedenceConstraints(), true);

    scheduler.setSteepestAscend(true);
    BspSchedule<Graph> schedule2 = bspInitial;
    scheduler.improveSchedule(schedule2);
    BOOST_CHECK_EQUAL(schedule2.satisfiesPrecedenceConstraints(), true);

    BspSchedule<Graph> schedule3 = bspInitial;
    scheduler.setTimeLimitSeconds(1U);
    scheduler.improveScheduleWithTimeLimit(schedule3);
    BOOST_CHECK_EQUAL(schedule3.satisfiesPrecedenceConstraints(), true);

    BspSchedule<Graph> schedule4 = bspInitial;
    scheduler.improveScheduleWithStepLimit(schedule4, 5);
    BOOST_CHECK_EQUAL(schedule4.satisfiesPrecedenceConstraints(), true);
}

BOOST_AUTO_TEST_CASE(HillClimbingForCommSchedule) {
    using Graph = computational_dag_vector_impl_def_t;

    BspInstance<Graph> instance;
    instance.setNumberOfProcessors(2);
    instance.setCommunicationCosts(3);
    instance.setSynchronisationCosts(5);

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
    BspSchedule<Graph> initial(instance);
    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, greedy.computeSchedule(initial));
    BOOST_CHECK_EQUAL(initial.satisfiesPrecedenceConstraints(), true);

    HillClimbingScheduler<Graph> hc;
    hc.improveSchedule(initial);
    BOOST_CHECK_EQUAL(initial.satisfiesPrecedenceConstraints(), true);

    BspSchedule<Graph> schedule = initial;
    BspScheduleCS<Graph> initialCs(std::move(initial));
    // initial_cs.setAutoCommunicationSchedule();
    initialCs.setEagerCommunicationSchedule();
    BOOST_CHECK_EQUAL(initialCs.hasValidCommSchedule(), true);

    HillClimbingForCommSteps<Graph> hcCs;
    BspScheduleCS<Graph> schedule1 = initialCs;
    hcCs.improveSchedule(schedule1);
    BOOST_CHECK_EQUAL(schedule1.hasValidCommSchedule(), true);

    BspScheduleCS<Graph> schedule2 = initialCs;
    hcCs.setSteepestAscend(true);
    hcCs.improveSchedule(schedule2);
    BOOST_CHECK_EQUAL(schedule2.hasValidCommSchedule(), true);
}
