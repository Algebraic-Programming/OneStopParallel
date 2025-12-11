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

BOOST_AUTO_TEST_CASE(BspScheduleRecomp_test) {
    using graph = computational_dag_vector_impl_def_t;

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

    file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(),
                                                      instance.getComputationalDag());

    BspSchedule<graph> schedule(instance);
    GreedyBspScheduler<graph> scheduler;
    const auto result = scheduler.computeSchedule(schedule);

    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
    BOOST_CHECK_EQUAL(&schedule.getInstance(), &instance);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

    BspScheduleCS scheduleCS(schedule);

    BspScheduleRecomp<graph> schedule_recomp(schedule);

    BOOST_CHECK(schedule_recomp.satisfiesConstraints());
    BOOST_CHECK_EQUAL(schedule_recomp.getTotalAssignments(), instance.numberOfVertices());
    BOOST_CHECK_EQUAL(schedule_recomp.computeWorkCosts(), schedule.computeWorkCosts());
    BOOST_CHECK_EQUAL(schedule_recomp.computeCosts(), scheduleCS.computeCosts());

    BspScheduleRecomp<graph> schedule_recomp_from_cs(scheduleCS);
    BOOST_CHECK(schedule_recomp_from_cs.satisfiesConstraints());
    BOOST_CHECK_EQUAL(schedule_recomp_from_cs.computeCosts(), scheduleCS.computeCosts());
}
