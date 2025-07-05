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
#include "bsp/model/BspScheduleCS.hpp"
#include "bsp/model/BspScheduleRecomp.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "auxiliary/io/DotFileWriter.hpp"
#include "auxiliary/io/arch_file_reader.hpp"
#include "auxiliary/io/hdag_graph_file_reader.hpp"
#include <filesystem>
#include <iostream>

#include "bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "bsp/scheduler/GreedySchedulers/CilkScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "bsp/scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "bsp/scheduler/GreedySchedulers/VarianceFillup.hpp"

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

    bool status = file_reader::readComputationalDagHyperdagFormat(
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

        BOOST_CHECK_EQUAL(RETURN_STATUS::SUCCESS, result);
        BOOST_CHECK_EQUAL(&schedule.getInstance(), &instance);
        BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

        BOOST_CHECK_EQUAL(schedule.computeCosts(), expected_bsp_costs[i]);
        BOOST_CHECK_EQUAL(schedule.computeTotalCosts(), expected_total_costs[i]);
        BOOST_CHECK_EQUAL(schedule.computeBufferedSendingCosts(), expected_buffered_sending_costs[i]);
        BOOST_CHECK_EQUAL(schedule.numberOfSupersteps(), expected_supersteps[i]);

        BspScheduleCS<graph> schedule_cs(instance);

        const auto result_cs = scheduler->computeScheduleCS(schedule_cs);

        BOOST_CHECK_EQUAL(RETURN_STATUS::SUCCESS, result_cs);

        BOOST_CHECK(schedule_cs.hasValidCommSchedule());

        BOOST_CHECK_EQUAL(schedule_cs.computeCosts(), expected_bsp_cs_costs[i]);

        i++;

        delete scheduler;
    }
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

    bool status = file_reader::readComputationalDagHyperdagFormat(
        (cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertices(), 54);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertex_types(), 1);

    BspLocking<graph_t1> scheduler;
    BspSchedule<graph_t1> schedule(instance);
    const auto result = scheduler.computeSchedule(schedule);
    BOOST_CHECK_EQUAL(RETURN_STATUS::SUCCESS, result);
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

    file_reader::readComputationalDagHyperdagFormat((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(),
                                                    instance.getComputationalDag());

    BspSchedule<graph> schedule(instance);
    BspLocking<graph> scheduler;

    const auto result = scheduler.computeSchedule(schedule);

    BOOST_CHECK_EQUAL(RETURN_STATUS::SUCCESS, result);
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