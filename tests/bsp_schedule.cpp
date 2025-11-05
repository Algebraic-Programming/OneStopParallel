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

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/BspScheduleCS.hpp"
#include "osp/bsp/model/MaxBspSchedule.hpp"
#include "osp/bsp/model/MaxBspScheduleCS.hpp"
#include "osp/bsp/model/BspScheduleRecomp.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include <filesystem>
#include <iostream>

#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/CilkScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/VarianceFillup.hpp"
#include "osp/bsp/scheduler/Serial.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(test_instance_bicgstab) {

    using graph = computational_dag_edge_idx_vector_impl_def_int_t;

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

    bool status = file_reader::readGraph(
        (cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertices(), 54);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertex_types(), 1);

    std::vector<Scheduler<graph> *> schedulers = {new BspLocking<graph>(),         new EtfScheduler<graph>(),
                                                  new GreedyBspScheduler<graph>(), new GreedyChildren<graph>(),
                                                  new GrowLocalAutoCores<graph>(), new VarianceFillup<graph>()};

    std::vector<int> expected_bsp_costs = {92, 108, 100, 108, 102, 110};
    std::vector<double> expected_total_costs = {74, 87, 84.25, 80.25, 91.25, 86.75};
    std::vector<int> expected_buffered_sending_costs = {92, 111, 103, 105, 102, 113};
    std::vector<unsigned> expected_supersteps = {6, 7, 7, 5, 3, 7};

    std::vector<int> expected_bsp_cs_costs = {86, 99, 97, 99, 102, 107};

    size_t i = 0;
    for (auto &scheduler : schedulers) {

        BspSchedule<graph> schedule(instance);

        const auto result = scheduler->computeSchedule(schedule);

        BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
        BOOST_CHECK_EQUAL(&schedule.getInstance(), &instance);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BOOST_CHECK_EQUAL(schedule.computeCosts(), expected_bsp_costs[i]);
        BOOST_CHECK_EQUAL(schedule.computeTotalCosts(), expected_total_costs[i]);
        BOOST_CHECK_EQUAL(schedule.computeBufferedSendingCosts(), expected_buffered_sending_costs[i]);
        BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), expected_supersteps[i]);

        BspScheduleCS<graph> schedule_cs(instance);

        const auto result_cs = scheduler->computeScheduleCS(schedule_cs);

        BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result_cs);

        BOOST_CHECK(schedule_cs.hasValidCommSchedule());

        BOOST_CHECK_EQUAL(schedule_cs.computeCosts(), expected_bsp_cs_costs[i]);

        i++;

        delete scheduler;
    }

    BspSchedule<graph> schedule(instance);
    Serial<graph> serial;
    const auto result = serial.computeSchedule(schedule);
    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), 1);

};

BOOST_AUTO_TEST_CASE(test_schedule_writer) {

    using graph_t1 = computational_dag_edge_idx_vector_impl_def_int_t;
    using graph_t2 = computational_dag_vector_impl_def_int_t;

    BspInstance<graph_t1> instance;
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

    bool status = file_reader::readGraph(
        (cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertices(), 54);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertex_types(), 1);

    BspLocking<graph_t1> scheduler;
    BspSchedule<graph_t1> schedule(instance);
    const auto result = scheduler.computeSchedule(schedule);
    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

    DotFileWriter sched_writer;

    std::cout << "Writing Graph" << std::endl;
    sched_writer.write_graph(std::cout, instance.getComputationalDag());

    std::cout << "Writing schedule_t1" << std::endl;
    sched_writer.write_schedule(std::cout, schedule);

    BspInstance<graph_t2> instance_t2(instance);
    BspSchedule<graph_t2> schedule_t2(instance_t2);

    BOOST_CHECK_EQUAL(schedule_t2.getInstance().getComputationalDag().num_vertices(),
                      instance.getComputationalDag().num_vertices());
    BOOST_CHECK(schedule_t2.satisfiesPrecedenceConstraints());

    BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().num_vertices(), instance.getComputationalDag().num_vertices());
    BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().num_vertex_types(),
                      instance.getComputationalDag().num_vertex_types());
    BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().num_edges(), instance.getComputationalDag().num_edges());

    for (const auto &v : instance.getComputationalDag().vertices()) {

        BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().vertex_work_weight(v),
                          instance.getComputationalDag().vertex_work_weight(v));
        BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().vertex_comm_weight(v),
                          instance.getComputationalDag().vertex_comm_weight(v));

        BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().vertex_mem_weight(v),
                          instance.getComputationalDag().vertex_mem_weight(v));

        BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().vertex_type(v),
                          instance.getComputationalDag().vertex_type(v));

        BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().out_degree(v),
                          instance.getComputationalDag().out_degree(v));

        BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().in_degree(v), instance.getComputationalDag().in_degree(v));
    }

    std::cout << "Writing schedule_t2" << std::endl;

    sched_writer.write_schedule(std::cout, schedule_t2);

    BspScheduleRecomp<graph_t2> schedule_recomp(schedule_t2);

    schedule_recomp.assignments(0).emplace_back(1, 0);
    schedule_recomp.assignments(0).emplace_back(2, 0);
    schedule_recomp.assignments(0).emplace_back(3, 0);

    std::cout << "Writing schedule_recomp" << std::endl;
    sched_writer.write_schedule_recomp(std::cout, schedule_recomp);

    std::cout << "Writing schedule_recomp_duplicate" << std::endl;
    sched_writer.write_schedule_recomp_duplicate(std::cout, schedule_recomp);

    std::cout << "Writing schedule_t2 CS" << std::endl;
    BspScheduleCS<graph_t2> schedule_cs(schedule_t2);
    sched_writer.write_schedule_cs(std::cout, schedule_cs);
};

BOOST_AUTO_TEST_CASE(test_bsp_schedule_cs) {

    using graph = computational_dag_edge_idx_vector_impl_def_int_t;

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

    file_reader::readGraph((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(),
                                                    instance.getComputationalDag());

    BspSchedule<graph> schedule(instance);
    BspLocking<graph> scheduler;

    const auto result = scheduler.computeSchedule(schedule);

    BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
    BOOST_CHECK_EQUAL(&schedule.getInstance(), &instance);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

    BspSchedule<graph> schedule_t2(schedule);

    BOOST_CHECK_EQUAL(schedule_t2.getInstance().getComputationalDag().num_vertices(),
                      instance.getComputationalDag().num_vertices());
    BOOST_CHECK(schedule_t2.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(schedule_t2.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {

        BOOST_CHECK_EQUAL(schedule_t2.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(schedule_t2.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    BspSchedule<graph> schedule_t3(instance);
    schedule_t3 = schedule_t2;
    BOOST_CHECK_EQUAL(schedule_t3.getInstance().getComputationalDag().num_vertices(),
                      instance.getComputationalDag().num_vertices());
    BOOST_CHECK(schedule_t3.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(schedule_t3.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {

        BOOST_CHECK_EQUAL(schedule_t3.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(schedule_t3.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    BspSchedule<graph> schedule_t4(instance);
    schedule_t4 = std::move(schedule_t3);

    BOOST_CHECK_EQUAL(schedule_t4.getInstance().getComputationalDag().num_vertices(),
                      instance.getComputationalDag().num_vertices());
    BOOST_CHECK(schedule_t4.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(schedule_t4.numberOfSupersteps(), schedule.numberOfSupersteps());
    for (const auto &v : instance.getComputationalDag().vertices()) {

        BOOST_CHECK_EQUAL(schedule_t4.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(schedule_t4.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    BspSchedule<graph> schedule_t5(std::move(schedule_t4));
    BOOST_CHECK_EQUAL(schedule_t5.getInstance().getComputationalDag().num_vertices(),
                      instance.getComputationalDag().num_vertices());
    BOOST_CHECK(schedule_t5.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(schedule_t5.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {

        BOOST_CHECK_EQUAL(schedule_t5.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(schedule_t5.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    BspScheduleCS<graph> schedule_cs(schedule_t5);
    BOOST_CHECK_EQUAL(schedule_cs.getInstance().getComputationalDag().num_vertices(),
                      instance.getComputationalDag().num_vertices());
    BOOST_CHECK(schedule_cs.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule_cs.hasValidCommSchedule());
    BOOST_CHECK_EQUAL(schedule_cs.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {

        BOOST_CHECK_EQUAL(schedule_cs.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(schedule_cs.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    // schedule_t5 is still valid
    BOOST_CHECK_EQUAL(schedule_t5.getInstance().getComputationalDag().num_vertices(),
                      instance.getComputationalDag().num_vertices());
    BOOST_CHECK(schedule_t5.satisfiesPrecedenceConstraints());
    BOOST_CHECK_EQUAL(schedule_t5.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {

        BOOST_CHECK_EQUAL(schedule_t5.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(schedule_t5.assignedProcessor(v), schedule.assignedProcessor(v));
    }

    BspScheduleCS<graph> schedule_cs_t2(std::move(schedule_t5));
    BOOST_CHECK_EQUAL(schedule_cs_t2.getInstance().getComputationalDag().num_vertices(),
                      instance.getComputationalDag().num_vertices());
    BOOST_CHECK(schedule_cs_t2.satisfiesPrecedenceConstraints());
    BOOST_CHECK(schedule_cs_t2.hasValidCommSchedule());
    BOOST_CHECK_EQUAL(schedule_cs_t2.numberOfSupersteps(), schedule.numberOfSupersteps());

    for (const auto &v : instance.getComputationalDag().vertices()) {

        BOOST_CHECK_EQUAL(schedule_cs_t2.assignedSuperstep(v), schedule.assignedSuperstep(v));
        BOOST_CHECK_EQUAL(schedule_cs_t2.assignedProcessor(v), schedule.assignedProcessor(v));
    }
};

BOOST_AUTO_TEST_CASE(test_max_bsp_schedule) {

    using graph = computational_dag_edge_idx_vector_impl_def_int_t;

    BspInstance<graph> instance;
    instance.setNumberOfProcessors(2);
    instance.setCommunicationCosts(10); // g=10
    instance.setSynchronisationCosts(100); // l=100 (not used in MaxBspSchedule cost model)

    auto &dag = instance.getComputationalDag();
    dag.add_vertex(10, 1, 0); // Node 0
    dag.add_vertex(5, 2, 0);  // Node 1
    dag.add_vertex(5, 3, 0);  // Node 2
    dag.add_vertex(10, 4, 0); // Node 3
    dag.add_edge(0, 1);
    dag.add_edge(0, 2);
    dag.add_edge(1, 3);
    dag.add_edge(2, 3);

    // Test a valid schedule with staleness = 2
    {
        MaxBspSchedule<graph> schedule(instance);
        schedule.setAssignedProcessor(0, 0);
        schedule.setAssignedSuperstep(0, 0);
        schedule.setAssignedProcessor(1, 0);
        schedule.setAssignedSuperstep(1, 1);
        schedule.setAssignedProcessor(2, 1);
        schedule.setAssignedSuperstep(2, 2); // 0->2 is cross-proc, 2 >= 0+2
        schedule.setAssignedProcessor(3, 0);
        schedule.setAssignedSuperstep(3, 4); // 2->3 is cross-proc, 4 >= 2+2
        schedule.updateNumberOfSupersteps();

        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        // Manual cost calculation:
        // Superstep 0: work = {10, 0} -> max_work = 10. comm = 0. Cost = max(10, 0) = 10.
        // Superstep 1: work = {5, 0} -> max_work = 5. comm from SS0: 0->2 (P0->P1) needed at SS2, comm sent in SS1. comm=1*10=10. Cost = max(5,0) = 5.
        // Superstep 2: work = {0, 5} -> max_work = 5. comm from SS1: 10. Cost = max(5, 10) + l = 10 + 100 = 110.
        // Superstep 3: work = {0, 0} -> max_work = 0. comm from SS2: 2->3 (P1->P0) needed at SS4, comm sent in SS3. comm=3*10=30. Cost = max(0,0) = 0.
        // Superstep 4: work = {10, 0} -> max_work = 10. comm from SS3: 30. Cost = max(10, 30) + l = 30 + 100 = 130.
        // Total cost = 10 + 5 + 110 + 0 + 130 = 255
        BOOST_CHECK_EQUAL(schedule.computeCosts(), 255);
    }

    // Test another valid schedule
    {
        MaxBspSchedule<graph> schedule(instance);
        schedule.setAssignedProcessor(0, 0);
        schedule.setAssignedSuperstep(0, 0);
        schedule.setAssignedProcessor(1, 1);
        schedule.setAssignedSuperstep(1, 2); // 0->1 is cross-proc, 2 >= 0+2
        schedule.setAssignedProcessor(2, 1);
        schedule.setAssignedSuperstep(2, 2); // 0->2 is cross-proc, 2 >= 0+2
        schedule.setAssignedProcessor(3, 0);
        schedule.setAssignedSuperstep(3, 4); // 1->3, 2->3 are cross-proc, 4 >= 2+2
        schedule.updateNumberOfSupersteps();

        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        // Manual cost calculation:
        // Superstep 0: work = {10, 0} -> max_work = 10. comm = 0. Cost = max(10, 0) = 10.
        // Superstep 1: work = {0, 0} -> max_work = 0. comm from SS0: 0->1, 0->2 (P0->P1) needed at SS2, comm sent in SS1. comm=1*10=10. Cost = max(0,0)=0.
        // Superstep 2: work = {0, 10} -> max_work = 10. comm from SS1: 10. Cost = max(10, 10) + l = 10 + 100 = 110.
        // Superstep 3: work = {0, 0} -> max_work = 0. comm from SS2: 1->3, 2->3 (P1->P0) needed at SS4, comm sent in SS3. comm=(2+3)*10=50. Cost = max(0,0)=0.
        // Superstep 4: work = {10, 0} -> max_work = 10. comm from SS3: 50. Cost = max(10, 50) + l = 50 + 100 = 150.
        // Total cost = 10 + 0 + 110 + 0 + 150 = 270
        BOOST_CHECK_EQUAL(schedule.computeCosts(), 270);
    }

    // Test an invalid schedule (violates staleness=2)
    {
        MaxBspSchedule<graph> schedule(instance);
        schedule.setAssignedProcessor(0, 0);
        schedule.setAssignedSuperstep(0, 0);
        schedule.setAssignedProcessor(1, 1); // 0->1 on different procs
        schedule.setAssignedSuperstep(1, 1); // step(0)+2 > step(1) is FALSE (0+2 > 1)
        schedule.updateNumberOfSupersteps();

        BOOST_CHECK(!schedule.satisfiesPrecedenceConstraints());
    }
}

BOOST_AUTO_TEST_CASE(test_max_bsp_schedule_cs) {

    using graph = computational_dag_edge_idx_vector_impl_def_int_t;

    BspInstance<graph> instance;
    instance.setNumberOfProcessors(2);
    instance.setCommunicationCosts(10); // g=10
    instance.setSynchronisationCosts(100); // l=100

    auto &dag = instance.getComputationalDag();
    dag.add_vertex(10, 1, 0); // Node 0
    dag.add_vertex(5, 2, 0);  // Node 1
    dag.add_vertex(5, 3, 0);  // Node 2
    dag.add_vertex(10, 4, 0); // Node 3
    dag.add_edge(0, 1);
    dag.add_edge(0, 2);
    dag.add_edge(1, 3);
    dag.add_edge(2, 3);

    // Test a valid schedule with staleness = 2
    {
        MaxBspScheduleCS<graph> schedule(instance);
        schedule.setAssignedProcessor(0, 0);
        schedule.setAssignedSuperstep(0, 0);
        schedule.setAssignedProcessor(1, 0);
        schedule.setAssignedSuperstep(1, 1);
        schedule.setAssignedProcessor(2, 1);
        schedule.setAssignedSuperstep(2, 2); // 0->2 is cross-proc, 2 >= 0+2
        schedule.setAssignedProcessor(3, 0);
        schedule.setAssignedSuperstep(3, 4); // 2->3 is cross-proc, 4 >= 2+2
        schedule.updateNumberOfSupersteps();

        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        // Set communication schedule (eager)
        schedule.addCommunicationScheduleEntry(0, 0, 1, 0); // 0->2 (P0->P1) sent in SS0
        schedule.addCommunicationScheduleEntry(2, 1, 0, 2); // 2->3 (P1->P0) sent in SS2

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
        MaxBspScheduleCS<graph> schedule(instance);
        schedule.setAssignedProcessor(0, 0);
        schedule.setAssignedSuperstep(0, 0);
        schedule.setAssignedProcessor(1, 1); // 0->1 on different procs
        schedule.setAssignedSuperstep(1, 1); // step(0)+2 > step(1) is FALSE (0+2 > 1)
        schedule.updateNumberOfSupersteps();

        BOOST_CHECK(!schedule.satisfiesPrecedenceConstraints());
    }
}