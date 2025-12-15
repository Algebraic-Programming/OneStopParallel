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
#include <filesystem>
#include <iostream>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/IlpSchedulers/CoptCommScheduleOptimizer.hpp"
#include "osp/bsp/scheduler/IlpSchedulers/CoptFullScheduler.hpp"
#include "osp/bsp/scheduler/IlpSchedulers/CoptPartialScheduler.hpp"
#include "osp/bsp/scheduler/IlpSchedulers/TotalCommunicationScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(TestTotal) {
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
        (cwd / "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag").string(), instance.GetComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> scheduleTo(instance);

    TotalCommunicationScheduler<graph> schedulerTo;
    schedulerTo.setTimeLimitSeconds(10);

    const auto resultTo = scheduler_to.ComputeSchedule(schedule_to);
    BOOST_CHECK(result_to == ReturnStatus::OSP_SUCCESS || result_to == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(scheduleTo.SatisfiesPrecedenceConstraints());

    BspSchedule<graph> schedule(instance);

    TotalCommunicationScheduler<graph> scheduler;
    scheduler.setTimeLimitSeconds(3600);
    const auto result = scheduler.ComputeSchedule(schedule);

    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
};

BOOST_AUTO_TEST_CASE(TestFull) {
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
        (cwd / "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag").string(), instance.GetComputationalDag());

    BOOST_CHECK(status);

    BspScheduleCS<graph> scheduleTo(instance);

    CoptFullScheduler<graph> schedulerTo;
    schedulerTo.setTimeLimitSeconds(10);

    const auto resultTo = scheduler_to.ComputeScheduleCS(schedule_to);
    BOOST_CHECK_EQUAL(ReturnStatus::BEST_FOUND, result_to);
    BOOST_CHECK(scheduleTo.SatisfiesPrecedenceConstraints());

    CoptFullScheduler<graph> schedulerRecomp;
    BspScheduleRecomp<graph> scheduleRecomp(instance);
    schedulerRecomp.setTimeLimitSeconds(10);
    schedulerRecomp.computeScheduleRecomp(schedule_recomp);
    BOOST_CHECK(scheduleRecomp.satisfiesConstraints());

    // WITH INITIALIZATION

    BspSchedule<graph> scheduleInit(instance);
    GreedyBspScheduler<graph> greedy;
    greedy.ComputeSchedule(schedule_init);
    BOOST_CHECK(scheduleInit.SatisfiesPrecedenceConstraints());
    BspScheduleCS<graph> scheduleInitCs(scheduleInit);
    BOOST_CHECK(scheduleInitCs.hasValidCommSchedule());

    // initialize with standard schedule, return standard schedule
    CoptFullScheduler<graph> schedulerInit;
    BspScheduleCS<graph> scheduleImproved(instance);
    schedulerInit.setTimeLimitSeconds(10);
    schedulerInit.setInitialSolutionFromBspSchedule(schedule_init_cs);
    const auto resultInit = scheduler_init.ComputeScheduleCS(schedule_improved);
    BOOST_CHECK_EQUAL(ReturnStatus::BEST_FOUND, result_init);
    BOOST_CHECK(scheduleImproved.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleImproved.hasValidCommSchedule());

    // initialize with standard schedule, return recomputing schedule
    CoptFullScheduler<graph> schedulerInit2(scheduleInitCs);
    BspScheduleRecomp<graph> scheduleImproved2(instance);
    schedulerInit2.setTimeLimitSeconds(10);
    const auto resultInit2 = scheduler_init2.computeScheduleRecomp(schedule_improved2);
    BOOST_CHECK_EQUAL(ReturnStatus::BEST_FOUND, result_init2);
    BOOST_CHECK(scheduleImproved2.satisfiesConstraints());

    // initialize with recomputing schedule, return recomputing schedule
    BspScheduleRecomp<graph> scheduleImproved3(instance), schedule_init3(schedule_init_cs);
    CoptFullScheduler<graph> SchedulerInit3(schedule_init3);
    SchedulerInit3.setTimeLimitSeconds(10);
    const auto resultInit3 = scheduler_init3.computeScheduleRecomp(schedule_improved3);
    BOOST_CHECK_EQUAL(ReturnStatus::BEST_FOUND, result_init3);
    BOOST_CHECK(scheduleImproved3.satisfiesConstraints());

    // with vertex types
    BspInstance<graph> instanceTyped = instance;
    instanceTyped.GetArchitecture().setProcessorType(0, 1);
    instanceTyped.GetArchitecture().setProcessorType(1, 1);
    for (VertexIdxT<graph> node = 0; node < static_cast<VertexIdxT<graph> >(instance_typed.NumberOfVertices()); ++node) {
        instanceTyped.GetComputationalDag().SetVertexType(node, node % 2);
    }
    instanceTyped.setDiagonalCompatibilityMatrix(2);

    BspSchedule<graph> scheduleTyped(instanceTyped);
    greedy.ComputeSchedule(schedule_typed);
    BOOST_CHECK(scheduleTyped.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleTyped.satisfiesNodeTypeConstraints());

    CoptFullScheduler<graph> schedulerTyped;
    BspScheduleCS<graph> scheduleTypedCs(scheduleTyped);
    schedulerTyped.setTimeLimitSeconds(10);
    schedulerTyped.setInitialSolutionFromBspSchedule(schedule_typed_cs);
    const auto resultTyped = scheduler_typed.ComputeSchedule(schedule_typed);
    BOOST_CHECK_EQUAL(ReturnStatus::BEST_FOUND, result_typed);
    BOOST_CHECK(scheduleTyped.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleTyped.satisfiesNodeTypeConstraints());

    // with MaxBSP schedule
    CoptFullScheduler<graph> schedulerMax;
    MaxBspScheduleCS<graph> scheduleMax(instance);
    schedulerMax.setTimeLimitSeconds(10);
    const auto resultMax = scheduler_max.computeMaxBspScheduleCS(schedule_max);
    BOOST_CHECK(result_max == ReturnStatus::OSP_SUCCESS || result_max == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(scheduleMax.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleMax.hasValidCommSchedule());

    schedulerMax.setInitialSolutionFromBspSchedule(schedule_max);
    const auto resultMax2 = scheduler_max.computeMaxBspScheduleCS(schedule_max);
    BOOST_CHECK(result_max2 == ReturnStatus::OSP_SUCCESS || result_max2 == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(scheduleMax.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleMax.hasValidCommSchedule());

    // longer time
    BspScheduleCS<graph> schedule(instance);

    CoptFullScheduler<graph> scheduler;
    scheduler.setTimeLimitSeconds(3600);
    const auto result = scheduler.ComputeScheduleCS(schedule);

    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
};

BOOST_AUTO_TEST_CASE(TestCs) {
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

    bool status = file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_pregel.hdag").string(),
                                                                    instance.GetComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> schedule(instance);
    GreedyBspScheduler<graph> greedy;
    greedy.ComputeSchedule(schedule);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    BspScheduleCS<graph> scheduleCs(schedule);
    BOOST_CHECK(scheduleCs.hasValidCommSchedule());

    CoptCommScheduleOptimizer<graph> scheduler;
    scheduler.setTimeLimitSeconds(10);
    const auto before = schedule_cs.compute_cs_communication_costs();
    const auto result = scheduler.ImproveSchedule(schedule_cs);
    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
    const auto after = schedule_cs.compute_cs_communication_costs();
    std::cout << before << " --cs--> " << after << std::endl;

    BOOST_CHECK(scheduleCs.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleCs.hasValidCommSchedule());
    BOOST_CHECK(before >= after);
};

BOOST_AUTO_TEST_CASE(TestPartial) {
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

    bool status = file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_pregel.hdag").string(),
                                                                    instance.GetComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> scheduleInit(instance);
    GreedyBspScheduler<graph> greedy;
    greedy.ComputeSchedule(schedule_init);
    BOOST_CHECK(scheduleInit.SatisfiesPrecedenceConstraints());
    BspScheduleCS<graph> schedule(scheduleInit);
    BOOST_CHECK(schedule.hasValidCommSchedule());

    CoptPartialScheduler<graph> scheduler;
    scheduler.setTimeLimitSeconds(10);
    scheduler.setStartAndEndSuperstep(0, 2);
    auto costBefore = schedule.computeCosts();
    auto result = scheduler.ImproveSchedule(schedule);
    BOOST_CHECK(result == ReturnStatus::OSP_SUCCESS || result == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule.hasValidCommSchedule());
    auto costMid = schedule.computeCosts();
    BOOST_CHECK(costMid <= cost_before);
    scheduler.setStartAndEndSuperstep(2, 5);
    result = scheduler.ImproveSchedule(schedule);
    BOOST_CHECK(result == ReturnStatus::OSP_SUCCESS || result == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule.hasValidCommSchedule());
    auto costAfter = schedule.computeCosts();
    BOOST_CHECK(costAfter <= cost_mid);
};
