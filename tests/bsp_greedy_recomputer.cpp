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

#define BOOST_TEST_MODULE BSP_GREEDY_RECOMPUTER
#include <boost/test/unit_test.hpp>

#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyRecomputer.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include <filesystem>
#include <iostream>

using namespace osp;

BOOST_AUTO_TEST_CASE(test_recomputer) {

    using graph = computational_dag_vector_impl_def_t;

    BspInstance<graph> instance1;
    instance1.setNumberOfProcessors(2);
    instance1.setCommunicationCosts(1);
    instance1.setSynchronisationCosts(1);

    instance1.getComputationalDag().add_vertex(10, 1, 0);
    instance1.getComputationalDag().add_vertex(10, 1, 0);
    instance1.getComputationalDag().add_vertex(10, 1, 0);
    instance1.getComputationalDag().add_edge(0, 1);
    instance1.getComputationalDag().add_edge(0, 2);

    BspSchedule<graph> schedule_init1(instance1);
    schedule_init1.setAssignedProcessor(0, 0);
    schedule_init1.setAssignedSuperstep(0, 0);
    schedule_init1.setAssignedProcessor(1, 0);
    schedule_init1.setAssignedSuperstep(1, 1);
    schedule_init1.setAssignedProcessor(2, 1);
    schedule_init1.setAssignedSuperstep(2, 1);
    BOOST_CHECK(schedule_init1.satisfiesPrecedenceConstraints());
    BspScheduleCS<graph> schedule_init_cs1(schedule_init1);
    BOOST_CHECK(schedule_init_cs1.hasValidCommSchedule());

    BspScheduleRecomp<graph> schedule(instance1);
    GreedyRecomputer<graph> scheduler;
    scheduler.computeRecompSchedule(schedule_init_cs1, schedule);
    BOOST_CHECK(schedule.satisfiesConstraints());
    BOOST_CHECK(schedule.computeCosts() < schedule_init_cs1.computeCosts());
    std::cout<<"Cost decrease by greedy recomp: "<<schedule_init_cs1.computeCosts()<<" -> "<<schedule.computeCosts()<<std::endl;


    // non-toy instance

    BspInstance<graph> instance2;
    instance2.setNumberOfProcessors(4);
    instance2.setCommunicationCosts(5);
    instance2.setSynchronisationCosts(20);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), instance2.getComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> schedule_init2(instance2);
    BspLocking<graph> greedy;
    greedy.computeSchedule(schedule_init2);
    BOOST_CHECK(schedule_init2.satisfiesPrecedenceConstraints());
    BspScheduleCS<graph> schedule_init_cs2(schedule_init2);
    BOOST_CHECK(schedule_init_cs2.hasValidCommSchedule());

    scheduler.computeRecompSchedule(schedule_init_cs2, schedule);
    BOOST_CHECK(schedule.satisfiesConstraints());
    BOOST_CHECK(schedule.computeCosts() < schedule_init_cs2.computeCosts());
    std::cout<<"Cost decrease by greedy recomp: "<<schedule_init_cs2.computeCosts()<<" -> "<<schedule.computeCosts()<<std::endl;
};