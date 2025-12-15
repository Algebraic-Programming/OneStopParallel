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

#define BOOST_TEST_MODULE EftSubgraphScheduler
#include <boost/test/unit_test.hpp>

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/dag_divider/isomorphism_divider/EftSubgraphScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(EftSubgraphSchedulerSimpleChain) {
    using GraphT = computational_dag_vector_impl_def_t;

    // 1. Setup Instance
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    // Create a simple coarse-grained DAG: 0 -> 1 -> 2
    dag.AddVertex(100, 1, 0);    // node 0
    dag.AddVertex(200, 1, 0);    // node 1
    dag.AddVertex(300, 1, 0);    // node 2
    dag.AddEdge(0, 1);
    dag.AddEdge(1, 2);

    // Setup Architecture: 2 processors of type 0, 2 of type 1
    instance.GetArchitecture().setProcessorsWithTypes({0, 0, 1, 1});
    instance.setDiagonalCompatibilityMatrix(2);

    // 2. Setup Scheduler Inputs
    std::vector<unsigned> multiplicities = {1, 2, 1};
    std::vector<unsigned> maxProcs = {100, 100, 100};
    std::vector<std::vector<VWorkwT<GraphT>>> requiredProcTypes(3);

    // Node 0: work 100, mult 1. Needs type 0.
    requiredProcTypes[0] = {100, 0};
    // Node 1: work 200, mult 2. Needs type 0 and 1.
    requiredProcTypes[1] = {100, 100};
    // Node 2: work 300, mult 1. Needs type 1.
    requiredProcTypes[2] = {0, 300};

    // 3. Run Scheduler
    EftSubgraphScheduler<GraphT> scheduler;
    scheduler.setMinWorkPerProcessor(1);
    SubgraphSchedule schedule = scheduler.run(instance, multiplicities, requiredProcTypes, maxProcs);

    // 4. Assertions
    BOOST_CHECK_CLOSE(schedule.makespan_, 250.0, 1e-9);
    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_.size(), 3);
    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_[0].size(), 2);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[0][0], 2);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[0][1], 0);
    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_[1].size(), 2);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[1][0], 1);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[1][1], 1);
    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_[2].size(), 2);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[2][0], 0);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[2][1], 2);
}

BOOST_AUTO_TEST_CASE(EftSubgraphSchedulerForkJoin) {
    using GraphT = computational_dag_vector_impl_def_t;

    // 1. Setup Instance
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    // Create a fork-join DAG: 0 -> {1,2} -> 3
    dag.AddVertex(100, 1, 0);    // node 0
    dag.AddVertex(200, 1, 0);    // node 1
    dag.AddVertex(300, 1, 0);    // node 2
    dag.AddVertex(100, 1, 0);    // node 3
    dag.AddEdge(0, 1);
    dag.AddEdge(0, 2);
    dag.AddEdge(1, 3);
    dag.AddEdge(2, 3);

    // Setup Architecture: 4 processors of type 0
    instance.GetArchitecture().setProcessorsWithTypes({0, 0, 0, 0});
    instance.setDiagonalCompatibilityMatrix(1);

    // 2. Setup Scheduler Inputs
    std::vector<unsigned> multiplicities = {1, 2, 1, 4};
    std::vector<unsigned> maxProcs = {100, 100, 100, 100};
    std::vector<std::vector<VWorkwT<GraphT>>> requiredProcTypes(4);

    // All nodes need type 0
    requiredProcTypes[0] = {100};
    requiredProcTypes[1] = {200};
    requiredProcTypes[2] = {300};
    requiredProcTypes[3] = {100};

    // 3. Run Scheduler
    EftSubgraphScheduler<GraphT> scheduler;
    scheduler.setMinWorkPerProcessor(1);
    SubgraphSchedule schedule = scheduler.run(instance, multiplicities, requiredProcTypes, maxProcs);

    // 4. Assertions
    // Manual calculation:
    // Ranks: 0:500, 1:300, 2:400, 3:100. Prio order: 0,2,1,3
    // T=0: Start 0 (4w). Finishes at 100/4=25.
    // T=25: ReadyQ {1,2}. Avail=4. Prio order {2,1}.
    //       Phase 1: Job 2 (mult 1) gets 1w. Avail=3. Job 1 (mult 2) gets 2w. Avail=1.
    //       Phase 2 (proportional on 1w): Job 2 (prio 400) gets floor(1*400/700)=0. Job 1 (prio 300) gets floor(1*300/700)=0.
    //       Phase 2.5 (greedy on 1w): Job 2 (higher prio) gets the remaining 1 worker.
    //       Final allocation: Job 2 gets 1+1=2 workers. Job 1 gets 2 workers.
    //       Job 2 (work 300, 2w) duration 150. Finishes at 25 + 150 = 175.
    //       Job 1 (work 200, 2w) duration 100. Finishes at 25 + 100 = 125.
    // T=125: Job 1 finishes.
    // T=175: Job 2 finishes. Job 3 becomes ready. Starts with 4w. Duration 100/4=25. Ends 200.
    BOOST_CHECK_CLOSE(schedule.makespan_, 200.0, 1e-9);

    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_.size(), 4);
    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_[0].size(), 1);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[0][0], 4);
    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_[1].size(), 1);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[1][0], 1);
    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_[2].size(), 1);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[2][0], 2);
    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_[3].size(), 1);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[3][0], 1);
}

BOOST_AUTO_TEST_CASE(EftSubgraphSchedulerDeadlock) {
    using GraphT = computational_dag_vector_impl_def_t;

    // 1. Setup Instance
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    // Create a single-node DAG
    dag.AddVertex(100, 1, 0);    // node 0

    // Setup Architecture: 1 processor of type 0
    instance.GetArchitecture().setProcessorsWithTypes({0});
    instance.setDiagonalCompatibilityMatrix(1);

    // 2. Setup Scheduler Inputs
    // Job needs 2 workers (multiplicity), but only 1 is available
    std::vector<unsigned> multiplicities = {2};
    std::vector<unsigned> maxProcs = {100};
    std::vector<std::vector<VWorkwT<GraphT>>> requiredProcTypes(1);
    requiredProcTypes[0] = {100};

    // 3. Run Scheduler
    EftSubgraphScheduler<GraphT> scheduler;
    scheduler.setMinWorkPerProcessor(1);
    SubgraphSchedule schedule = scheduler.run(instance, multiplicities, requiredProcTypes, maxProcs);

    // 4. Assertions
    // Expect a deadlock, indicated by a negative makespan
    BOOST_CHECK_LT(schedule.makespan_, 0.0);
}

BOOST_AUTO_TEST_CASE(EftSubgraphSchedulerComplexDag) {
    using GraphT = computational_dag_vector_impl_def_t;

    // 1. Setup Instance
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    dag.AddVertex(50, 1, 0);     // 0
    dag.AddVertex(100, 1, 0);    // 1
    dag.AddVertex(150, 1, 0);    // 2
    dag.AddVertex(80, 1, 0);     // 3
    dag.AddVertex(120, 1, 0);    // 4
    dag.AddVertex(60, 1, 0);     // 5
    dag.AddEdge(0, 1);
    dag.AddEdge(0, 2);
    dag.AddEdge(1, 3);
    dag.AddEdge(2, 3);
    dag.AddEdge(2, 4);
    dag.AddEdge(3, 5);
    dag.AddEdge(4, 5);

    // Setup Architecture: 4 processors of type 0, 4 of type 1
    instance.GetArchitecture().setProcessorsWithTypes({0, 0, 0, 0, 1, 1, 1, 1});
    instance.setDiagonalCompatibilityMatrix(2);

    // 2. Setup Scheduler Inputs
    std::vector<unsigned> multiplicities = {1, 2, 1, 4, 2, 1};
    std::vector<unsigned> maxProcs = {100, 100, 100, 100, 100, 100};
    std::vector<std::vector<VWorkwT<GraphT>>> requiredProcTypes(6);
    requiredProcTypes[0] = {50, 0};     // Job 0: needs T0
    requiredProcTypes[1] = {100, 0};    // Job 1: needs T0
    requiredProcTypes[2] = {0, 150};    // Job 2: needs T1
    requiredProcTypes[3] = {40, 40};    // Job 3: needs T0 & T1
    requiredProcTypes[4] = {0, 120};    // Job 4: needs T1
    requiredProcTypes[5] = {60, 0};     // Job 5: needs T0

    // 3. Run Scheduler
    EftSubgraphScheduler<GraphT> scheduler;
    scheduler.setMinWorkPerProcessor(1);
    SubgraphSchedule schedule = scheduler.run(instance, multiplicities, requiredProcTypes, maxProcs);

    BOOST_CHECK_CLOSE(schedule.makespan_, 105.0, 1e-9);

    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_.size(), 6);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[0][0], 4);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[1][0], 2);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[2][1], 4);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[3][0], 1);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[3][1], 1);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[4][1], 2);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[5][0], 4);
}

BOOST_AUTO_TEST_CASE(EftSubgraphSchedulerResourceContention) {
    using GraphT = computational_dag_vector_impl_def_t;

    // 1. Setup Instance
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    // Create a fork-join DAG: 0 -> {1,2,3} -> 4
    dag.AddVertex(10, 1, 0);     // 0
    dag.AddVertex(100, 1, 0);    // 1 (high rank)
    dag.AddVertex(50, 1, 0);     // 2 (mid rank)
    dag.AddVertex(20, 1, 0);     // 3 (low rank)
    dag.AddVertex(10, 1, 0);     // 4
    dag.AddEdge(0, 1);
    dag.AddEdge(0, 2);
    dag.AddEdge(0, 3);
    dag.AddEdge(1, 4);
    dag.AddEdge(2, 4);
    dag.AddEdge(3, 4);

    // Setup Architecture: 4 processors of type 0
    instance.GetArchitecture().setProcessorsWithTypes({0, 0, 0, 0});
    instance.setDiagonalCompatibilityMatrix(1);

    // 2. Setup Scheduler Inputs
    std::vector<unsigned> multiplicities = {1, 2, 2, 2, 1};
    std::vector<unsigned> maxProcs = {100, 100, 100, 100, 100};
    std::vector<std::vector<VWorkwT<GraphT>>> requiredProcTypes(5);
    requiredProcTypes[0] = {10};
    requiredProcTypes[1] = {100};
    requiredProcTypes[2] = {50};
    requiredProcTypes[3] = {20};
    requiredProcTypes[4] = {10};

    // 3. Run Scheduler
    EftSubgraphScheduler<GraphT> scheduler;
    scheduler.setMinWorkPerProcessor(1);
    SubgraphSchedule schedule = scheduler.run(instance, multiplicities, requiredProcTypes, maxProcs);

    // 4. Assertions
    // Manual calculation:
    // Ranks: 0:120, 1:110, 2:60, 3:30, 4:10. Prio order: 0,1,2,3,4
    // T=0: Start 0 (4w). Finishes at 10/4=2.5.
    // T=2.5: ReadyQ {1,2,3}. Avail=4. Runnable check: {1,2} can run (each needs mult=2).
    //        Guarantee phase: Job 1 gets 2w, Job 2 gets 2w. Remaining=0.
    //        Job 1 (work 100, 2w) duration 50. Finishes at 2.5 + 50 = 52.5.
    //        Job 2 (work 50, 2w) duration 25. Finishes at 2.5 + 25 = 27.5.
    // T=27.5: Job 2 finishes. 2 workers free. Job 3 starts. Duration 20/2=10 (ends 37.5).
    // T=37.5: Job 3 finishes.
    // T=52.5: Job 1 finishes. Job 4 becomes ready. Starts with 4 workers. Duration 10/4=2.5 (ends 55.0).
    BOOST_CHECK_CLOSE(schedule.makespan_, 55.0, 1e-9);

    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_.size(), 5);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[0][0], 4);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[1][0], 1);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[2][0], 1);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[3][0], 1);
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[4][0], 4);
}

BOOST_AUTO_TEST_CASE(EftSubgraphSchedulerProportionalAllocation) {
    using GraphT = computational_dag_vector_impl_def_t;

    // 1. Setup Instance
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    // Create a fork DAG: 0 -> {1,2}
    dag.AddVertex(10, 1, 0);     // 0
    dag.AddVertex(300, 1, 0);    // 1 (high rank)
    dag.AddVertex(100, 1, 0);    // 2 (low rank)
    dag.AddEdge(0, 1);
    dag.AddEdge(0, 2);

    // Setup Architecture: 10 processors of type 0
    instance.GetArchitecture().setProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    instance.setDiagonalCompatibilityMatrix(1);

    // 2. Setup Scheduler Inputs
    std::vector<unsigned> multiplicities = {1, 1, 1};
    std::vector<unsigned> maxProcs = {100, 100, 100};
    std::vector<std::vector<VWorkwT<GraphT>>> requiredProcTypes(3);
    requiredProcTypes[0] = {10};
    requiredProcTypes[1] = {300};
    requiredProcTypes[2] = {100};

    // 3. Run Scheduler
    EftSubgraphScheduler<GraphT> scheduler;
    scheduler.setMinWorkPerProcessor(1);
    SubgraphSchedule schedule = scheduler.run(instance, multiplicities, requiredProcTypes, maxProcs);

    // 4. Assertions
    // Manual calculation:
    // Ranks: 0:310, 1:300, 2:100. Prio: 0,1,2
    // T=0: Start 0 (10 workers). Finishes at 10/10=1.0
    // T=1.0: ReadyQ: {1,2}. Available: 10. All mult=1.
    //        Guarantee phase: Job 1 gets 1w, Job 2 gets 1w. Remaining=8.
    //        Proportional phase (on remaining 8): Total prio = 300+100=400.
    //        Job 1 gets floor(8 * 300/400) = 6 additional workers. Total = 1+6=7.
    //        Job 2 gets floor(8 * 100/400) = 2 additional workers. Total = 1+2=3.
    //        Job 1 finishes at 1 + 300/7 = 1 + 42.857... = 43.857...
    //        Job 2 finishes at 1 + 100/3 = 1 + 33.333... = 34.333...
    //        Makespan is 43.857...
    BOOST_CHECK_CLOSE(schedule.makespan_, 1.0 + 300.0 / 7.0, 1e-9);

    BOOST_REQUIRE_EQUAL(schedule.nodeAssignedWorkerPerType_.size(), 3);
    // Job 0: 10 workers
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[0][0], 10);
    // Job 1 (high rank): gets 7 workers (75% of 10, floored)
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[1][0], 7);
    // Job 2 (low rank): gets 3 workers
    BOOST_CHECK_EQUAL(schedule.nodeAssignedWorkerPerType_[2][0], 3);
}
