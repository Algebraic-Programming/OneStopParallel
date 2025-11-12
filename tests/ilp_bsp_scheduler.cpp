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

#define BOOST_TEST_MODULE COPT_ILP_SCHEDULING
#include <boost/test/unit_test.hpp>

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include <filesystem>
#include <iostream>

#include "osp/bsp/scheduler/IlpSchedulers/CoptFullScheduler.hpp"
#include "osp/bsp/scheduler/IlpSchedulers/TotalCommunicationScheduler.hpp"
#include "osp/bsp/scheduler/IlpSchedulers/CoptCommScheduleOptimizer.hpp"
#include "osp/bsp/scheduler/IlpSchedulers/CoptPartialScheduler.hpp"

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

    bool status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> schedule_to(instance);

    TotalCommunicationScheduler<graph> scheduler_to;
    scheduler_to.setTimeLimitSeconds(10);

    const auto result_to = scheduler_to.computeSchedule(schedule_to);
    BOOST_CHECK_EQUAL(RETURN_STATUS::BEST_FOUND, result_to);
    BOOST_CHECK(schedule_to.satisfiesPrecedenceConstraints());

    BspSchedule<graph> schedule(instance);

    TotalCommunicationScheduler<graph> scheduler;
    scheduler.setTimeLimitSeconds(3600);
    const auto result = scheduler.computeSchedule(schedule);

    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
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

    bool status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    BspScheduleCS<graph> schedule_to(instance);

    CoptFullScheduler<graph> scheduler_to;
    scheduler_to.setTimeLimitSeconds(10);

    const auto result_to = scheduler_to.computeScheduleCS(schedule_to);
    BOOST_CHECK_EQUAL(RETURN_STATUS::BEST_FOUND, result_to);
    BOOST_CHECK(schedule_to.satisfiesPrecedenceConstraints());

    CoptFullScheduler<graph> scheduler_recomp;
    BspScheduleRecomp<graph> schedule_recomp(instance);
    scheduler_recomp.setTimeLimitSeconds(10);
    scheduler_recomp.computeScheduleRecomp(schedule_recomp);
    BOOST_CHECK(schedule_recomp.satisfiesConstraints());

    // WITH INITIALIZATION

    BspSchedule<graph> schedule_init(instance);
    GreedyBspScheduler<graph> greedy;
    greedy.computeSchedule(schedule_init);
    BOOST_CHECK(schedule_init.satisfiesPrecedenceConstraints());
    BspScheduleCS<graph> schedule_init_cs(schedule_init);
    BOOST_CHECK(schedule_init_cs.hasValidCommSchedule());

    // initialize with standard schedule, return standard schedule
    CoptFullScheduler<graph> scheduler_init;
    BspScheduleCS<graph> schedule_improved(instance);
    scheduler_init.setTimeLimitSeconds(10);
    scheduler_init.setInitialSolutionFromBspSchedule(schedule_init_cs);
    const auto result_init = scheduler_init.computeScheduleCS(schedule_improved);
    BOOST_CHECK_EQUAL(RETURN_STATUS::BEST_FOUND, result_init);
    BOOST_CHECK(schedule_improved.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule_improved.hasValidCommSchedule());

    // initialize with standard schedule, return recomputing schedule
    CoptFullScheduler<graph> scheduler_init2(schedule_init_cs);
    BspScheduleRecomp<graph> schedule_improved2(instance);
    scheduler_init2.setTimeLimitSeconds(10);
    const auto result_init2 = scheduler_init2.computeScheduleRecomp(schedule_improved2);
    BOOST_CHECK_EQUAL(RETURN_STATUS::BEST_FOUND, result_init2);
    BOOST_CHECK(schedule_improved2.satisfiesConstraints());

    // initialize with recomputing schedule, return recomputing schedule
    BspScheduleRecomp<graph> schedule_improved3(instance),schedule_init3(schedule_init_cs);
    CoptFullScheduler<graph> scheduler_init3(schedule_init3);
    scheduler_init3.setTimeLimitSeconds(10);
    const auto result_init3 = scheduler_init3.computeScheduleRecomp(schedule_improved3);
    BOOST_CHECK_EQUAL(RETURN_STATUS::BEST_FOUND, result_init3);
    BOOST_CHECK(schedule_improved3.satisfiesConstraints());

    // with vertex types
    BspInstance<graph> instance_typed = instance;
    instance_typed.getArchitecture().setProcessorType(0, 1);
    instance_typed.getArchitecture().setProcessorType(1, 1);
    for(vertex_idx_t<graph> node = 0; node < static_cast<vertex_idx_t<graph> >(instance_typed.numberOfVertices()); ++node)
        instance_typed.getComputationalDag().set_vertex_type(node, node%2);
    instance_typed.setDiagonalCompatibilityMatrix(2);

    BspSchedule<graph> schedule_typed(instance_typed);
    greedy.computeSchedule(schedule_typed);
    BOOST_CHECK(schedule_typed.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule_typed.satisfiesNodeTypeConstraints());

    CoptFullScheduler<graph> scheduler_typed;
    BspScheduleCS<graph> schedule_typed_cs(schedule_typed);
    scheduler_typed.setTimeLimitSeconds(10);
    scheduler_typed.setInitialSolutionFromBspSchedule(schedule_typed_cs);
    const auto result_typed = scheduler_typed.computeSchedule(schedule_typed);
    BOOST_CHECK_EQUAL(RETURN_STATUS::BEST_FOUND, result_typed);
    BOOST_CHECK(schedule_typed.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule_typed.satisfiesNodeTypeConstraints());

    // with MaxBSP schedule
    CoptFullScheduler<graph> scheduler_max;
    MaxBspScheduleCS<graph> schedule_max(instance);
    scheduler_max.setTimeLimitSeconds(10);
    const auto result_max = scheduler_max.computeMaxBspScheduleCS(schedule_max);
    BOOST_CHECK(result_max == RETURN_STATUS::OSP_SUCCESS || result_max == RETURN_STATUS::BEST_FOUND);
    BOOST_CHECK(schedule_max.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule_max.hasValidCommSchedule());

    scheduler_max.setInitialSolutionFromBspSchedule(schedule_max);
    const auto result_max2 = scheduler_max.computeMaxBspScheduleCS(schedule_max);
    BOOST_CHECK(result_max2 == RETURN_STATUS::OSP_SUCCESS || result_max2 == RETURN_STATUS::BEST_FOUND);
    BOOST_CHECK(schedule_max.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule_max.hasValidCommSchedule());

    // longer time
    BspScheduleCS<graph> schedule(instance);

    CoptFullScheduler<graph> scheduler;
    scheduler.setTimeLimitSeconds(3600);
    const auto result = scheduler.computeScheduleCS(schedule);

    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
};

BOOST_AUTO_TEST_CASE(test_cs) {

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

    bool status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_pregel.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> schedule(instance);
    GreedyBspScheduler<graph> greedy;
    greedy.computeSchedule(schedule);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
    BspScheduleCS<graph> schedule_cs(schedule);
    BOOST_CHECK(schedule_cs.hasValidCommSchedule());

    CoptCommScheduleOptimizer<graph> scheduler;
    scheduler.setTimeLimitSeconds(10);
    const auto before = schedule_cs.compute_cs_communication_costs();
    const auto result = scheduler.improveSchedule(schedule_cs);
    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
    const auto after = schedule_cs.compute_cs_communication_costs();
    std::cout<<before<<" --cs--> "<<after<<std::endl;

    BOOST_CHECK(schedule_cs.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule_cs.hasValidCommSchedule());
    BOOST_CHECK(before >= after);
};

BOOST_AUTO_TEST_CASE(test_partial) {

    using graph = computational_dag_edge_idx_vector_impl_def_t;

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

    bool status = file_reader::readComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_pregel.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> schedule_init(instance);
    GreedyBspScheduler<graph> greedy;
    greedy.computeSchedule(schedule_init);
    BOOST_CHECK(schedule_init.satisfiesPrecedenceConstraints());
    BspScheduleCS<graph> schedule(schedule_init);
    BOOST_CHECK(schedule.hasValidCommSchedule());

    CoptPartialScheduler<graph> scheduler;
    scheduler.setTimeLimitSeconds(10);
    scheduler.setStartAndEndSuperstep(0, 2);
    auto cost_before = schedule.computeCosts();
    auto result = scheduler.improveSchedule(schedule);
    BOOST_CHECK(result == RETURN_STATUS::OSP_SUCCESS || result == RETURN_STATUS::BEST_FOUND);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule.hasValidCommSchedule());
    auto cost_mid = schedule.computeCosts();
    BOOST_CHECK(cost_mid <= cost_before);
    scheduler.setStartAndEndSuperstep(2, 5);
    result = scheduler.improveSchedule(schedule);
    BOOST_CHECK(result == RETURN_STATUS::OSP_SUCCESS || result == RETURN_STATUS::BEST_FOUND);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule.hasValidCommSchedule());
    auto cost_after = schedule.computeCosts();
    BOOST_CHECK(cost_after <= cost_mid);

};

