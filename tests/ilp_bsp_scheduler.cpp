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

#define BOOST_TEST_MODULE Bsp_Architecture
#include <boost/test/unit_test.hpp>

#include "bsp/model/BspInstance.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"
#include "io/arch_file_reader.hpp"
#include "io/graph_file_reader.hpp"
#include <filesystem>
#include <iostream>

#include "bsp/scheduler/IlpSchedulers/CoptFullScheduler.hpp"
#include "bsp/scheduler/IlpSchedulers/TotalCommunicationScheduler.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(test_total) {

    using graph = computational_dag_edge_idx_vector_impl_def_t;

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
        (cwd / "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> schedule_to(instance);

    TotalCommunicationScheduler<graph> scheduler_to;
    scheduler_to.setTimeLimitSeconds(10);

    const auto result_to = scheduler_to.computeSchedule(schedule_to);
    BOOST_CHECK_EQUAL(BEST_FOUND, result_to);
    BOOST_CHECK(schedule_to.satisfiesPrecedenceConstraints());

    BspSchedule<graph> schedule(instance);

    TotalCommunicationScheduler<graph> scheduler;
    scheduler.setTimeLimitSeconds(3600);
    const auto result = scheduler.computeSchedule(schedule);

    BOOST_CHECK_EQUAL(SUCCESS, result);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
};

BOOST_AUTO_TEST_CASE(test_full) {

    using graph = computational_dag_edge_idx_vector_impl_def_t;

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
        (cwd / "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    BspScheduleCS<graph> schedule_to(instance);

    CoptFullScheduler<graph> scheduler_to;
    scheduler_to.setTimeLimitSeconds(10);

    const auto result_to = scheduler_to.computeScheduleCS(schedule_to);
    BOOST_CHECK_EQUAL(BEST_FOUND, result_to);
    BOOST_CHECK(schedule_to.satisfiesPrecedenceConstraints());

    BspScheduleCS<graph> schedule(instance);

    CoptFullScheduler<graph> scheduler;
    scheduler.setTimeLimitSeconds(3600);
    const auto result = scheduler.computeScheduleCS(schedule);

    BOOST_CHECK_EQUAL(SUCCESS, result);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
};