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
    using graph = ComputationalDagEdgeIdxVectorImplDefT;

    BspInstance<graph> instance;
    instance.SetNumberOfProcessors(4);
    instance.SetCommunicationCosts(3);
    instance.SetSynchronisationCosts(5);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::ReadComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag").string(), instance.GetComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> scheduleTo(instance);

    TotalCommunicationScheduler<graph> schedulerTo;
    schedulerTo.SetTimeLimitSeconds(10);

    const auto resultTo = schedulerTo.ComputeSchedule(scheduleTo);
    BOOST_CHECK(resultTo == ReturnStatus::OSP_SUCCESS || resultTo == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(scheduleTo.SatisfiesPrecedenceConstraints());

    BspSchedule<graph> schedule(instance);

    TotalCommunicationScheduler<graph> scheduler;
    scheduler.SetTimeLimitSeconds(3600);
    const auto result = scheduler.ComputeSchedule(schedule);

    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
};

BOOST_AUTO_TEST_CASE(TestFull) {
    using graph = ComputationalDagEdgeIdxVectorImplDefT;

    BspInstance<graph> instance;
    instance.SetNumberOfProcessors(4);
    instance.SetCommunicationCosts(3);
    instance.SetSynchronisationCosts(5);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::ReadComputationalDagHyperdagFormatDB(
        (cwd / "data/spaa/tiny/instance_spmv_N6_nzP0d4.hdag").string(), instance.GetComputationalDag());

    BOOST_CHECK(status);

    BspScheduleCS<graph> scheduleTo(instance);

    CoptFullScheduler<graph> schedulerTo;
    schedulerTo.SetTimeLimitSeconds(10);

    const auto resultTo = schedulerTo.ComputeScheduleCS(scheduleTo);
    BOOST_CHECK_EQUAL(ReturnStatus::BEST_FOUND, resultTo);
    BOOST_CHECK(scheduleTo.SatisfiesPrecedenceConstraints());

    CoptFullScheduler<graph> schedulerRecomp;
    BspScheduleRecomp<graph> scheduleRecomp(instance);
    schedulerRecomp.SetTimeLimitSeconds(10);
    schedulerRecomp.ComputeScheduleRecomp(scheduleRecomp);
    BOOST_CHECK(scheduleRecomp.SatisfiesConstraints());

    // WITH INITIALIZATION

    BspSchedule<graph> scheduleInit(instance);
    GreedyBspScheduler<graph> greedy;
    greedy.ComputeSchedule(scheduleInit);
    BOOST_CHECK(scheduleInit.SatisfiesPrecedenceConstraints());
    BspScheduleCS<graph> scheduleInitCs(scheduleInit);
    BOOST_CHECK(scheduleInitCs.HasValidCommSchedule());

    // initialize with standard schedule, return standard schedule
    CoptFullScheduler<graph> schedulerInit;
    BspScheduleCS<graph> scheduleImproved(instance);
    schedulerInit.SetTimeLimitSeconds(10);
    schedulerInit.SetInitialSolutionFromBspSchedule(scheduleInitCs);
    const auto resultInit = schedulerInit.ComputeScheduleCS(scheduleImproved);
    BOOST_CHECK_EQUAL(ReturnStatus::BEST_FOUND, resultInit);
    BOOST_CHECK(scheduleImproved.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleImproved.HasValidCommSchedule());

    // initialize with standard schedule, return recomputing schedule
    CoptFullScheduler<graph> schedulerInit2(scheduleInitCs);
    BspScheduleRecomp<graph> scheduleImproved2(instance);
    schedulerInit2.SetTimeLimitSeconds(10);
    const auto resultInit2 = schedulerInit2.ComputeScheduleRecomp(scheduleImproved2);
    BOOST_CHECK_EQUAL(ReturnStatus::BEST_FOUND, resultInit2);
    BOOST_CHECK(scheduleImproved2.SatisfiesConstraints());

    // initialize with recomputing schedule, return recomputing schedule
    BspScheduleRecomp<graph> scheduleImproved3(instance), scheduleInit3(scheduleInitCs);
    CoptFullScheduler<graph> schedulerInit3(scheduleInit3);
    schedulerInit3.SetTimeLimitSeconds(10);
    const auto resultInit3 = schedulerInit3.ComputeScheduleRecomp(scheduleImproved3);
    BOOST_CHECK_EQUAL(ReturnStatus::BEST_FOUND, resultInit3);
    BOOST_CHECK(scheduleImproved3.SatisfiesConstraints());

    // with vertex types
    BspInstance<graph> instanceTyped = instance;
    instanceTyped.GetArchitecture().SetProcessorType(0, 1);
    instanceTyped.GetArchitecture().SetProcessorType(1, 1);
    for (VertexIdxT<graph> node = 0; node < static_cast<VertexIdxT<graph> >(instanceTyped.NumberOfVertices()); ++node) {
        instanceTyped.GetComputationalDag().SetVertexType(node, node % 2);
    }
    instanceTyped.SetDiagonalCompatibilityMatrix(2);

    BspSchedule<graph> scheduleTyped(instanceTyped);
    greedy.ComputeSchedule(scheduleTyped);
    BOOST_CHECK(scheduleTyped.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleTyped.SatisfiesNodeTypeConstraints());

    CoptFullScheduler<graph> schedulerTyped;
    BspScheduleCS<graph> scheduleTypedCs(scheduleTyped);
    schedulerTyped.SetTimeLimitSeconds(10);
    schedulerTyped.SetInitialSolutionFromBspSchedule(scheduleTypedCs);
    const auto resultTyped = schedulerTyped.ComputeSchedule(scheduleTyped);
    BOOST_CHECK_EQUAL(ReturnStatus::BEST_FOUND, resultTyped);
    BOOST_CHECK(scheduleTyped.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleTyped.SatisfiesNodeTypeConstraints());

    // with MaxBSP schedule
    CoptFullScheduler<graph> schedulerMax;
    MaxBspScheduleCS<graph> scheduleMax(instance);
    schedulerMax.SetTimeLimitSeconds(10);
    const auto resultMax = schedulerMax.ComputeMaxBspScheduleCs(scheduleMax);
    BOOST_CHECK(resultMax == ReturnStatus::OSP_SUCCESS || resultMax == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(scheduleMax.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleMax.HasValidCommSchedule());

    schedulerMax.SetInitialSolutionFromBspSchedule(scheduleMax);
    const auto resultMax2 = schedulerMax.ComputeMaxBspScheduleCs(scheduleMax);
    BOOST_CHECK(resultMax2 == ReturnStatus::OSP_SUCCESS || resultMax2 == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(scheduleMax.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleMax.HasValidCommSchedule());

    // longer time
    BspScheduleCS<graph> schedule(instance);

    CoptFullScheduler<graph> scheduler;
    scheduler.SetTimeLimitSeconds(3600);
    const auto result = scheduler.ComputeScheduleCS(schedule);

    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
};

BOOST_AUTO_TEST_CASE(TestCs) {
    using graph = ComputationalDagEdgeIdxVectorImplDefT;

    BspInstance<graph> instance;
    instance.SetNumberOfProcessors(4);
    instance.SetCommunicationCosts(3);
    instance.SetSynchronisationCosts(5);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_pregel.hdag").string(),
                                                                    instance.GetComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> schedule(instance);
    GreedyBspScheduler<graph> greedy;
    greedy.ComputeSchedule(schedule);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    BspScheduleCS<graph> scheduleCs(schedule);
    BOOST_CHECK(scheduleCs.HasValidCommSchedule());

    CoptCommScheduleOptimizer<graph> scheduler;
    scheduler.SetTimeLimitSeconds(10);
    const auto before = scheduleCs.ComputeCsCommunicationCosts();
    const auto result = scheduler.ImproveSchedule(scheduleCs);
    BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
    const auto after = scheduleCs.ComputeCsCommunicationCosts();
    std::cout << before << " --cs--> " << after << std::endl;

    BOOST_CHECK(scheduleCs.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleCs.HasValidCommSchedule());
    BOOST_CHECK(before >= after);
};

BOOST_AUTO_TEST_CASE(TestPartial) {
    using graph = ComputationalDagEdgeIdxVectorImplDefT;

    BspInstance<graph> instance;
    instance.SetNumberOfProcessors(3);
    instance.SetCommunicationCosts(3);
    instance.SetSynchronisationCosts(5);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_pregel.hdag").string(),
                                                                    instance.GetComputationalDag());

    BOOST_CHECK(status);

    BspSchedule<graph> scheduleInit(instance);
    GreedyBspScheduler<graph> greedy;
    greedy.ComputeSchedule(scheduleInit);
    BOOST_CHECK(scheduleInit.SatisfiesPrecedenceConstraints());
    BspScheduleCS<graph> schedule(scheduleInit);
    BOOST_CHECK(schedule.HasValidCommSchedule());

    CoptPartialScheduler<graph> scheduler;
    scheduler.SetTimeLimitSeconds(10);
    scheduler.SetStartAndEndSuperstep(0, 2);
    auto costBefore = schedule.ComputeCosts();
    auto result = scheduler.ImproveSchedule(schedule);
    BOOST_CHECK(result == ReturnStatus::OSP_SUCCESS || result == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule.HasValidCommSchedule());
    auto costMid = schedule.ComputeCosts();
    BOOST_CHECK(costMid <= costBefore);
    scheduler.SetStartAndEndSuperstep(2, 5);
    result = scheduler.ImproveSchedule(schedule);
    BOOST_CHECK(result == ReturnStatus::OSP_SUCCESS || result == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule.HasValidCommSchedule());
    auto costAfter = schedule.ComputeCosts();
    BOOST_CHECK(costAfter <= costMid);
};
