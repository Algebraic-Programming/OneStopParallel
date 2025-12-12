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
#include <filesystem>
#include <iostream>

#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/BspScheduleCS.hpp"
#include "osp/bsp/model/BspScheduleRecomp.hpp"
#include "osp/bsp/model/MaxBspSchedule.hpp"
#include "osp/bsp/model/MaxBspScheduleCS.hpp"
#include "osp/bsp/model/cost/BufferedSendingCost.hpp"
#include "osp/bsp/model/cost/LazyCommunicationCost.hpp"
#include "osp/bsp/model/cost/TotalCommunicationCost.hpp"
#include "osp/bsp/model/cost/TotalLambdaCommunicationCost.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/CilkScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/VarianceFillup.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(TestInstanceBicgstab) {
    using Graph = computational_dag_edge_idx_vector_impl_def_int_t;

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

    bool status = file_reader::readGraph((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().NumVertices(), 54);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().NumVertexTypes(), 1);

    std::vector<Scheduler<Graph> *> schedulers = {new BspLocking<Graph>(),
                                                  new EtfScheduler<Graph>(),
                                                  new GreedyBspScheduler<Graph>(),
                                                  new GreedyChildren<Graph>(),
                                                  new GrowLocalAutoCores<Graph>(),
                                                  new VarianceFillup<Graph>()};

    std::vector<int> expectedBspCosts = {92, 108, 100, 108, 102, 110};
    std::vector<double> expectedTotalCosts = {74, 87, 84.25, 80.25, 91.25, 86.75};
    std::vector<int> expectedBufferedSendingCosts = {92, 111, 103, 105, 102, 113};
    std::vector<unsigned> expectedSupersteps = {6, 7, 7, 5, 3, 7};

    std::vector<int> expectedBspCsCosts = {86, 99, 97, 99, 102, 107};

    size_t i = 0;
    for (auto &scheduler : schedulers) {
        BspSchedule<Graph> schedule(instance);

        const auto result = scheduler->computeSchedule(schedule);

        BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
        BOOST_CHECK_EQUAL(&schedule.getInstance(), &instance);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BOOST_CHECK_EQUAL(schedule.computeCosts(), expectedBspCosts[i]);
        BOOST_CHECK_EQUAL(TotalCommunicationCost<Graph>()(schedule), expectedTotalCosts[i]);
        BOOST_CHECK_EQUAL(BufferedSendingCost<Graph>()(schedule), expectedBufferedSendingCosts[i]);
        BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), expectedSupersteps[i]);

        BspScheduleCS<Graph> scheduleCs(instance);

        const auto resultCs = scheduler->computeScheduleCS(scheduleCs);

        BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, resultCs);

        BOOST_CHECK(scheduleCs.hasValidCommSchedule());

        BOOST_CHECK_EQUAL(scheduleCs.computeCosts(), expectedBspCsCosts[i]);

        i++;

        delete scheduler;
    }

    BspSchedule<Graph> schedule(instance);
    Serial<Graph> serial;
    const auto result = serial.computeSchedule(schedule);
    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), 1);
}

BOOST_AUTO_TEST_CASE(TestScheduleWriter) {
    using GraphT1 = computational_dag_edge_idx_vector_impl_def_int_t;
    using GraphT2 = computational_dag_vector_impl_def_int_t;

    BspInstance<GraphT1> instance;
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

    bool status = file_reader::readGraph((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().NumVertices(), 54);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().NumVertexTypes(), 1);

    BspLocking<GraphT1> scheduler;
    BspSchedule<GraphT1> schedule(instance);
    const auto result = scheduler.computeSchedule(schedule);
    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

    DotFileWriter schedWriter;

    std::cout << "Writing Graph" << std::endl;
    schedWriter.write_graph(std::cout, instance.getComputationalDag());

    std::cout << "Writing schedule_t1" << std::endl;
    schedWriter.write_schedule(std::cout, schedule);

    BspInstance<GraphT2> instanceT2(instance);
    BspSchedule<GraphT2> scheduleT2(instanceT2);

    BOOST_CHECK_EQUAL(scheduleT2.getInstance().getComputationalDag().NumVertices(), instance.getComputationalDag().NumVertices());
    BOOST_CHECK(scheduleT2.satisfiesPrecedenceConstraints());

    BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().NumVertices(), instance.getComputationalDag().NumVertices());
    BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().NumVertexTypes(), instance.getComputationalDag().NumVertexTypes());
    BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().NumEdges(), instance.getComputationalDag().NumEdges());

    for (const auto &v : instance.getComputationalDag().vertices()) {
        BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().VertexWorkWeight(v), instance.getComputationalDag().VertexWorkWeight(v));
        BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().VertexCommWeight(v), instance.getComputationalDag().VertexCommWeight(v));

        BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().VertexMemWeight(v), instance.getComputationalDag().VertexMemWeight(v));

        BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().VertexType(v), instance.getComputationalDag().VertexType(v));

        BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().out_degree(v), instance.getComputationalDag().out_degree(v));

        BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().in_degree(v), instance.getComputationalDag().in_degree(v));
    }

    std::cout << "Writing schedule_t2" << std::endl;

    schedWriter.write_schedule(std::cout, scheduleT2);

    BspScheduleRecomp<GraphT2> scheduleRecomp(scheduleT2);

    scheduleRecomp.assignments(0).emplace_back(1, 0);
    scheduleRecomp.assignments(0).emplace_back(2, 0);
    scheduleRecomp.assignments(0).emplace_back(3, 0);

    std::cout << "Writing schedule_recomp" << std::endl;
    schedWriter.write_schedule_recomp(std::cout, scheduleRecomp);

    std::cout << "Writing schedule_recomp_duplicate" << std::endl;
    schedWriter.write_schedule_recomp_duplicate(std::cout, scheduleRecomp);

    std::cout << "Writing schedule_t2 CS" << std::endl;
    BspScheduleCS<GraphT2> scheduleCs(scheduleT2);
    schedWriter.write_schedule_cs(std::cout, scheduleCs);
}

BOOST_AUTO_TEST_CASE(TestBspScheduleCs) {
    using Graph = computational_dag_edge_idx_vector_impl_def_int_t;

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

    file_reader::readGraph((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), instance.getComputationalDag());

    BspSchedule<Graph> schedule(instance);
    BspLocking<Graph> scheduler;

    const auto result = scheduler.computeSchedule(schedule);

    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
    BOOST_CHECK_EQUAL(&schedule.getInstance(), &instance);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

    BspSchedule<Graph> scheduleT2(schedule);

    BOOST_CHECK_EQUAL(scheduleT2.getInstance().getComputationalDag().NumVertices(), instance.getComputationalDag().NumVertices());
    BOOST_CHECK(scheduleT2.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(scheduleT2.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {
        BOOST_CHECK_EQUAL(scheduleT2.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(scheduleT2.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    BspSchedule<Graph> scheduleT3(instance);
    scheduleT3 = scheduleT2;
    BOOST_CHECK_EQUAL(scheduleT3.getInstance().getComputationalDag().NumVertices(), instance.getComputationalDag().NumVertices());
    BOOST_CHECK(scheduleT3.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(scheduleT3.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {
        BOOST_CHECK_EQUAL(scheduleT3.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(scheduleT3.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    BspSchedule<Graph> scheduleT4(instance);
    scheduleT4 = std::move(scheduleT3);

    BOOST_CHECK_EQUAL(scheduleT4.getInstance().getComputationalDag().NumVertices(), instance.getComputationalDag().NumVertices());
    BOOST_CHECK(scheduleT4.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(scheduleT4.numberOfSupersteps(), schedule.numberOfSupersteps());
    for (const auto &v : instance.getComputationalDag().vertices()) {
        BOOST_CHECK_EQUAL(scheduleT4.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(scheduleT4.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    BspSchedule<Graph> scheduleT5(std::move(scheduleT4));
    BOOST_CHECK_EQUAL(scheduleT5.getInstance().getComputationalDag().NumVertices(), instance.getComputationalDag().NumVertices());
    BOOST_CHECK(scheduleT5.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(scheduleT5.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {
        BOOST_CHECK_EQUAL(scheduleT5.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(scheduleT5.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    BspScheduleCS<Graph> scheduleCs(scheduleT5);
    BOOST_CHECK_EQUAL(scheduleCs.getInstance().getComputationalDag().NumVertices(), instance.getComputationalDag().NumVertices());
    BOOST_CHECK(scheduleCs.satisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleCs.hasValidCommSchedule());
    BOOST_CHECK_EQUAL(scheduleCs.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {
        BOOST_CHECK_EQUAL(scheduleCs.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(scheduleCs.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    // schedule_t5 is still valid
    BOOST_CHECK_EQUAL(scheduleT5.getInstance().getComputationalDag().NumVertices(), instance.getComputationalDag().NumVertices());
    BOOST_CHECK(scheduleT5.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(scheduleT5.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {
        BOOST_CHECK_EQUAL(scheduleT5.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(scheduleT5.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    BspScheduleCS<Graph> scheduleCsT2(std::move(scheduleT5));
    BOOST_CHECK_EQUAL(scheduleCsT2.getInstance().getComputationalDag().NumVertices(), instance.getComputationalDag().NumVertices());
    BOOST_CHECK(scheduleCsT2.satisfiesPrecedenceConstraints());
    BOOST_CHECK(scheduleCsT2.hasValidCommSchedule());
    BOOST_CHECK_EQUAL(scheduleCsT2.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {
        BOOST_CHECK_EQUAL(scheduleCsT2.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(scheduleCsT2.assignedProcessor(v), schedule.assignedProcessor(v));
    }
}

BOOST_AUTO_TEST_CASE(TestMaxBspSchedule) {
    using Graph = computational_dag_edge_idx_vector_impl_def_int_t;

    BspInstance<Graph> instance;
    instance.setNumberOfProcessors(2);
    instance.setCommunicationCosts(10);       // g=10
    instance.setSynchronisationCosts(100);    // l=100 (not used in MaxBspSchedule cost model)

    auto &dag = instance.getComputationalDag();
    dag.add_vertex(10, 1, 0);    // Node 0
    dag.add_vertex(5, 2, 0);     // Node 1
    dag.add_vertex(5, 3, 0);     // Node 2
    dag.add_vertex(10, 4, 0);    // Node 3
    dag.add_edge(0, 1);
    dag.add_edge(0, 2);
    dag.add_edge(1, 3);
    dag.add_edge(2, 3);

    // Test a valid schedule with staleness = 2
    {
        MaxBspSchedule<Graph> schedule(instance);
        schedule.setAssignedProcessor(0, 0);
        schedule.setAssignedSuperstep(0, 0);
        schedule.setAssignedProcessor(1, 0);
        schedule.setAssignedSuperstep(1, 1);
        schedule.setAssignedProcessor(2, 1);
        schedule.setAssignedSuperstep(2, 2);    // 0->2 is cross-proc, 2 >= 0+2
        schedule.setAssignedProcessor(3, 0);
        schedule.setAssignedSuperstep(3, 4);    // 2->3 is cross-proc, 4 >= 2+2
        schedule.updateNumberOfSupersteps();

        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        // Manual cost calculation:
        // Superstep 0: work = {10, 0} -> max_work = 10. comm = 0. Cost = max(10, 0) = 10.
        // Superstep 1: work = {5, 0} -> max_work = 5. comm from SS0: 0->2 (P0->P1) needed at SS2, comm sent in SS0. comm=1*10=10.
        // Cost = max(5,l+10) = 110. Superstep 2: work = {0, 5} -> max_work = 5. comm = 0. Cost = max(5, 0) = 5. Superstep 3: work
        // = {0, 0} -> max_work = 0. comm from SS2: 2->3 (P1->P0) needed at SS4, comm sent in SS2. comm=3*10=30. Cost = max(0,l+30) = 130.
        // Superstep 4: work = {10, 0} -> max_work = 10. comm = 0. Cost = max(10, 0) = 10.
        // Total cost = 10 + 110 + 5 + 130 + 10 = 265
        BOOST_CHECK_EQUAL(schedule.computeCosts(), 265);
    }

    // Test another valid schedule
    {
        MaxBspSchedule<Graph> schedule(instance);
        schedule.setAssignedProcessor(0, 0);
        schedule.setAssignedSuperstep(0, 0);
        schedule.setAssignedProcessor(1, 1);
        schedule.setAssignedSuperstep(1, 2);    // 0->1 is cross-proc, 2 >= 0+2
        schedule.setAssignedProcessor(2, 1);
        schedule.setAssignedSuperstep(2, 2);    // 0->2 is cross-proc, 2 >= 0+2
        schedule.setAssignedProcessor(3, 0);
        schedule.setAssignedSuperstep(3, 4);    // 1->3, 2->3 are cross-proc, 4 >= 2+2
        schedule.updateNumberOfSupersteps();

        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        // Manual cost calculation:
        // Superstep 0: work = {10, 0} -> max_work = 10. comm = 0. Cost = max(10, 0) = 10.
        // Superstep 1: work = {0, 0} -> max_work = 0. comm from SS0: 0->1, 0->2 (P0->P1) needed at SS2, comm sent in SS0.
        // comm=1*10=10. Cost = max(0,l+10)=110. Superstep 2: work = {0, 10} -> max_work = 10. comm = 0. Cost = max(10, 0) = 10.
        // Superstep 3: work = {0, 0} -> max_work = 0. comm from SS2: 1->3, 2->3 (P1->P0) needed at SS4, comm sent in SS2.
        // comm=(2+3)*10=50. Cost = max(0,l+50)=150. Superstep 4: work = {10, 0} -> max_work = 10. Cost = max(10, 0) = 10. Total
        // cost = 10 + 110 + 10 + 150 + 10 = 290
        BOOST_CHECK_EQUAL(schedule.computeCosts(), 290);
    }

    // Test an invalid schedule (violates staleness=2)
    {
        MaxBspSchedule<Graph> schedule(instance);
        schedule.setAssignedProcessor(0, 0);
        schedule.setAssignedSuperstep(0, 0);
        schedule.setAssignedProcessor(1, 1);    // 0->1 on different procs
        schedule.setAssignedSuperstep(1, 1);    // step(0)+2 > step(1) is FALSE (0+2 > 1)
        schedule.updateNumberOfSupersteps();

        BOOST_CHECK(!schedule.satisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(TestMaxBspScheduleCs) {
    using Graph = computational_dag_edge_idx_vector_impl_def_int_t;

    BspInstance<Graph> instance;
    instance.setNumberOfProcessors(2);
    instance.setCommunicationCosts(10);       // g=10
    instance.setSynchronisationCosts(100);    // l=100

    auto &dag = instance.getComputationalDag();
    dag.add_vertex(10, 1, 0);    // Node 0
    dag.add_vertex(5, 2, 0);     // Node 1
    dag.add_vertex(5, 3, 0);     // Node 2
    dag.add_vertex(10, 4, 0);    // Node 3
    dag.add_edge(0, 1);
    dag.add_edge(0, 2);
    dag.add_edge(1, 3);
    dag.add_edge(2, 3);

    // Test a valid schedule with staleness = 2
    {
        MaxBspScheduleCS<Graph> schedule(instance);
        schedule.setAssignedProcessor(0, 0);
        schedule.setAssignedSuperstep(0, 0);
        schedule.setAssignedProcessor(1, 0);
        schedule.setAssignedSuperstep(1, 1);
        schedule.setAssignedProcessor(2, 1);
        schedule.setAssignedSuperstep(2, 2);    // 0->2 is cross-proc, 2 >= 0+2
        schedule.setAssignedProcessor(3, 0);
        schedule.setAssignedSuperstep(3, 4);    // 2->3 is cross-proc, 4 >= 2+2
        schedule.updateNumberOfSupersteps();

        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        // Set communication schedule (eager)
        schedule.addCommunicationScheduleEntry(0, 0, 1, 0);    // 0->2 (P0->P1) sent in SS0
        schedule.addCommunicationScheduleEntry(2, 1, 0, 2);    // 2->3 (P1->P0) sent in SS2

        BOOST_CHECK(schedule.hasValidCommSchedule());

        // Manual cost calculation:
        // SS0: work={10,0}, max_work=10. comm_send(P0)=1, comm_rec(P1)=0. max_comm_h=1. Cost=max(10, 0)=10.
        // SS1: work={5,0}, max_work=5. comm from SS0: h=1, cost=1*10=10. Cost=max(5,10)+l=10+100=110.
        // SS2: work={0,5}, max_work=5. comm from SS1: h=0, cost=0. Cost=max(5,0)=5.
        // SS3: work={0,0}, max_work=0. comm from SS2: h=3, cost=3*10=30. Cost=max(0,30)+l=30+100=130.
        // SS4: work={10,0}, max_work=10. comm from SS3: h=0, cost=0. Cost=max(10,0)=10.
        // Total cost = 10 + 110 + 5 + 130 + 10 = 265
        BOOST_CHECK_EQUAL(schedule.computeCosts(), 265);
    }

    // Test an invalid schedule (violates staleness=2)
    {
        MaxBspScheduleCS<Graph> schedule(instance);
        schedule.setAssignedProcessor(0, 0);
        schedule.setAssignedSuperstep(0, 0);
        schedule.setAssignedProcessor(1, 1);    // 0->1 on different procs
        schedule.setAssignedSuperstep(1, 1);    // step(0)+2 > step(1) is FALSE (0+2 > 1)
        schedule.updateNumberOfSupersteps();

        BOOST_CHECK(!schedule.satisfiesPrecedenceConstraints());
    }
}
