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
#include <boost/test/unit_test.hpp>

#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"
#include "bsp/scheduler/LocalSearch/HillClimbing/hill_climbing_for_comm_schedule.hpp"
#include "auxiliary/io/hdag_graph_file_reader.hpp"
#include <filesystem>

#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

std::vector<std::string> test_graphs() {
    return {"data/spaa/tiny/instance_k-means.hdag", "data/spaa/tiny/instance_bicgstab.hdag",
            "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag"};
}

BOOST_AUTO_TEST_CASE(hill_climbing) {

    using graph = computational_dag_vector_impl_def_t;

    BspInstance<graph> instance;
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

    bool status = file_reader::readComputationalDagHyperdagFormat(
        (cwd / "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    GreedyBspScheduler<graph> greedy;
    BspSchedule<graph> bsp_initial(instance);
    BOOST_CHECK_EQUAL(RETURN_STATUS::SUCCESS, greedy.computeSchedule(bsp_initial));
    BOOST_CHECK_EQUAL(bsp_initial.satisfiesPrecedenceConstraints(), true);

    HillClimbingScheduler<graph> scheduler;
    BspSchedule<graph> schedule1 = bsp_initial;
    scheduler.improveSchedule(schedule1);
    BOOST_CHECK_EQUAL(schedule1.satisfiesPrecedenceConstraints(), true);

    scheduler.setSteepestAscend(true);
    BspSchedule<graph> schedule2 = bsp_initial;
    scheduler.improveSchedule(schedule2);
    BOOST_CHECK_EQUAL(schedule2.satisfiesPrecedenceConstraints(), true);

    BspSchedule<graph> schedule3 = bsp_initial;
    scheduler.setTimeLimitSeconds(1U);
    scheduler.improveScheduleWithTimeLimit(schedule3);
    BOOST_CHECK_EQUAL(schedule3.satisfiesPrecedenceConstraints(), true);

    BspSchedule<graph> schedule4 = bsp_initial;
    scheduler.improveScheduleWithStepLimit(schedule4, 5);
    BOOST_CHECK_EQUAL(schedule4.satisfiesPrecedenceConstraints(), true);

};

BOOST_AUTO_TEST_CASE(hill_climbing_for_comm_schedule) {

    using graph = computational_dag_vector_impl_def_t;

    BspInstance<graph> instance;
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

    bool status = file_reader::readComputationalDagHyperdagFormat(
        (cwd / "data/spaa/tiny/instance_CG_N4_K1_nzP0d35.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    GreedyBspScheduler<graph> greedy;
    BspSchedule<graph> initial(instance);
    BOOST_CHECK_EQUAL(RETURN_STATUS::SUCCESS, greedy.computeSchedule(initial));
    BOOST_CHECK_EQUAL(initial.satisfiesPrecedenceConstraints(), true);

    HillClimbingScheduler<graph> hc;
    hc.improveSchedule(initial);
    BOOST_CHECK_EQUAL(initial.satisfiesPrecedenceConstraints(), true);

    BspSchedule<graph> schedule = initial;
    BspScheduleCS<graph> initial_cs(std::move(initial));
    //initial_cs.setAutoCommunicationSchedule();
    initial_cs.setEagerCommunicationSchedule();
    BOOST_CHECK_EQUAL(initial_cs.hasValidCommSchedule(), true);

    HillClimbingForCommSteps<graph> hc_cs;
    BspScheduleCS<graph> schedule1 = initial_cs;
    hc_cs.improveSchedule(schedule1);
    BOOST_CHECK_EQUAL(schedule1.hasValidCommSchedule(), true);

    BspScheduleCS<graph> schedule2 = initial_cs;
    hc_cs.setSteepestAscend(true);
    hc_cs.improveSchedule(schedule2);
    BOOST_CHECK_EQUAL(schedule2.hasValidCommSchedule(), true);

};