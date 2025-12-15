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
#include <filesystem>
#include <iostream>

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyRecomputer.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(TestRecomputer) {
    using Graph = computational_dag_vector_impl_def_t;

    BspInstance<Graph> instance1;
    instance1.setNumberOfProcessors(2);
    instance1.setCommunicationCosts(1);
    instance1.setSynchronisationCosts(1);

    instance1.GetComputationalDag().AddVertex(10, 1, 0);
    instance1.GetComputationalDag().AddVertex(10, 1, 0);
    instance1.GetComputationalDag().AddVertex(10, 1, 0);
    instance1.GetComputationalDag().AddEdge(0, 1);
    instance1.GetComputationalDag().AddEdge(0, 2);

    BspSchedule<Graph> scheduleInit1(instance1);
    scheduleInit1.SetAssignedProcessor(0, 0);
    scheduleInit1.SetAssignedSuperstep(0, 0);
    scheduleInit1.SetAssignedProcessor(1, 0);
    scheduleInit1.SetAssignedSuperstep(1, 1);
    scheduleInit1.SetAssignedProcessor(2, 1);
    scheduleInit1.SetAssignedSuperstep(2, 1);
    BOOST_CHECK(scheduleInit1.SatisfiesPrecedenceConstraints());
    BspScheduleCS<Graph> scheduleInitCs1(scheduleInit1);
    BOOST_CHECK(scheduleInitCs1.hasValidCommSchedule());

    BspScheduleRecomp<Graph> schedule(instance1);
    GreedyRecomputer<Graph> scheduler;
    scheduler.computeRecompSchedule(scheduleInitCs1, schedule);
    BOOST_CHECK(schedule.satisfiesConstraints());
    BOOST_CHECK(schedule.computeCosts() < scheduleInitCs1.computeCosts());
    std::cout << "Cost decrease by greedy recomp: " << scheduleInitCs1.computeCosts() << " -> " << schedule.computeCosts()
              << std::endl;

    // non-toy instance

    BspInstance<Graph> instance2;
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

    bool status = file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(),
                                                                    instance2.GetComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<Graph> scheduleInit2(instance2);
    BspLocking<Graph> greedy;
    greedy.ComputeSchedule(scheduleInit2);
    BOOST_CHECK(scheduleInit2.SatisfiesPrecedenceConstraints());
    BspScheduleCS<Graph> scheduleInitCs2(scheduleInit2);
    BOOST_CHECK(scheduleInitCs2.hasValidCommSchedule());

    scheduler.computeRecompSchedule(scheduleInitCs2, schedule);
    BOOST_CHECK(schedule.satisfiesConstraints());
    BOOST_CHECK(schedule.computeCosts() < scheduleInitCs2.computeCosts());
    std::cout << "Cost decrease by greedy recomp: " << scheduleInitCs2.computeCosts() << " -> " << schedule.computeCosts()
              << std::endl;
}
