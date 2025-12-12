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

#define BOOST_TEST_MODULE BSP_SCHEDULE_RECOMP
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <iostream>

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/model/BspScheduleRecomp.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(BspScheduleRecompTest) {
    using Graph = computational_dag_vector_impl_def_t;

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

    file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(),
                                                      instance.GetComputationalDag());

    BspSchedule<Graph> schedule(instance);
    GreedyBspScheduler<Graph> scheduler;
    const auto result = scheduler.ComputeSchedule(schedule);

    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
    BOOST_CHECK_EQUAL(&schedule.GetInstance(), &instance);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    BspScheduleCS scheduleCS(schedule);

    BspScheduleRecomp<Graph> scheduleRecomp(schedule);

    BOOST_CHECK(scheduleRecomp.satisfiesConstraints());
    BOOST_CHECK_EQUAL(scheduleRecomp.getTotalAssignments(), instance.NumberOfVertices());
    BOOST_CHECK_EQUAL(scheduleRecomp.computeWorkCosts(), schedule.computeWorkCosts());
    BOOST_CHECK_EQUAL(scheduleRecomp.computeCosts(), scheduleCS.computeCosts());

    BspScheduleRecomp<Graph> scheduleRecompFromCs(scheduleCS);
    BOOST_CHECK(scheduleRecompFromCs.satisfiesConstraints());
    BOOST_CHECK_EQUAL(scheduleRecompFromCs.computeCosts(), scheduleCS.computeCosts());
}
