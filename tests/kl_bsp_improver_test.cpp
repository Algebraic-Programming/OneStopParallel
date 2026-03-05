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

/// End-to-end tests for KlBspCommCostFunction under all three
/// communication-cost policies (Eager, Lazy, Buffered).
///
/// Uses standard BspSchedule (staleness=0) for all tests.
///
/// Tests verify:
///   1. Inner-loop cost-tracking consistency (ComputeScheduleCostTest ==
///      GetCurrentCost after each RunInnerIterationTest).
///   2. Full ImproveSchedule: precedence constraints are satisfied.
///   3. All three policies produce valid results on varied topologies.
///
/// The inner-loop check is the key test: it validates that
/// UpdateDatastructureAfterMove + ComputeCommAffinity together maintain
/// consistent incremental cost tracking across all three policies.

#define BOOST_TEST_MODULE kl_bsp_improver
#include <boost/test/unit_test.hpp>

#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_test.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using VertexType = Graph::VertexIdx;
using CostT = double;

// ============================================================================
//  Type aliases
// ============================================================================

template <typename CommPolicy, unsigned WindowSize = 1>
using BspCommCostF = KlBspCommCostFunction<Graph, CostT, NoLocalSearchMemoryConstraint, CommPolicy, WindowSize>;

template <typename CommPolicy, unsigned WindowSize = 1>
using BspImprover = KlImprover<Graph, BspCommCostF<CommPolicy, WindowSize>, NoLocalSearchMemoryConstraint, WindowSize, CostT>;

// ============================================================================
//  Helper: run up to maxIter inner iterations and check cost consistency
//
// Uses no penalty (InsertGainHeapTest) so GetCurrentCost() equals the pure
// comm+work+sync cost — identical to ComputeScheduleCostTest().
// ============================================================================
template <typename TestT>
static void RunInnerLoopAndCheckCost(TestT &kl, int maxIter, const std::string &label) {
    for (int iter = 0; iter < maxIter; ++iter) {
        kl.RunInnerIterationTest();

        CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
        CostT tracked = kl.GetCurrentCost();

        BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);
        if (std::abs(recomputed - tracked) > 0.00001 * std::max(1.0, std::abs(recomputed))) {
            BOOST_TEST_MESSAGE("Cost mismatch at " << label << " iteration " << iter << ": recomputed=" << recomputed
                                                   << " tracked=" << tracked);
            break;
        }
    }
}

// ============================================================================
//  Graph fixtures
// ============================================================================

/// Fan-out / fan-in graph (6 nodes, 2 procs).
/// 0 -> {1,2,3}, {1,2,3} -> 4, 4 -> 5
/// Steps: 0:s0p0, 1:s1p1, 2:s1p0, 3:s1p1, 4:s2p0, 5:s3p1
///
struct SmallFanGraph {
    Graph dag;
    BspArchitecture<Graph> arch;
    BspInstance<Graph> *instance = nullptr;
    BspSchedule<Graph> *schedule = nullptr;

    SmallFanGraph() {
        //                          work  mem  comm
        dag.AddVertex(/* 0 */ 3, 1, 5);
        dag.AddVertex(/* 1 */ 4, 1, 3);
        dag.AddVertex(/* 2 */ 2, 1, 4);
        dag.AddVertex(/* 3 */ 5, 1, 2);
        dag.AddVertex(/* 4 */ 3, 1, 6);
        dag.AddVertex(/* 5 */ 4, 1, 2);

        dag.AddEdge(0, 1, 1);
        dag.AddEdge(0, 2, 1);
        dag.AddEdge(0, 3, 1);
        dag.AddEdge(1, 4, 1);
        dag.AddEdge(2, 4, 1);
        dag.AddEdge(3, 4, 1);
        dag.AddEdge(4, 5, 1);

        arch.SetNumberOfProcessors(2);
        arch.SetCommunicationCosts(2);
        arch.SetSynchronisationCosts(3);
    }

    BspSchedule<Graph> &Build() {
        instance = new BspInstance<Graph>(dag, arch);
        schedule = new BspSchedule<Graph>(*instance);

        schedule->SetAssignedProcessors({0, 1, 0, 1, 0, 1});
        schedule->SetAssignedSupersteps({0, 1, 1, 1, 2, 3});
        schedule->UpdateNumberOfSupersteps();
        return *schedule;
    }

    ~SmallFanGraph() {
        delete schedule;
        delete instance;
    }
};

/// 8-node graph with cross-processor edges.
///
///   0->1, 0->2, 0->3, 1->4, 2->4, 2->5, 4->7, 3->7
///
struct EightNodeGraph {
    Graph dag;
    BspArchitecture<Graph> arch;
    BspInstance<Graph> *instance = nullptr;
    BspSchedule<Graph> *schedule = nullptr;

    EightNodeGraph() {
        dag.AddVertex(2, 9, 2);    // 0
        dag.AddVertex(3, 8, 4);    // 1
        dag.AddVertex(4, 7, 3);    // 2
        dag.AddVertex(5, 6, 2);    // 3
        dag.AddVertex(6, 5, 6);    // 4
        dag.AddVertex(7, 4, 2);    // 5
        dag.AddVertex(8, 3, 4);    // 6
        dag.AddVertex(9, 2, 1);    // 7

        dag.AddEdge(0, 1, 2);
        dag.AddEdge(0, 2, 2);
        dag.AddEdge(0, 3, 2);
        dag.AddEdge(1, 4, 12);
        dag.AddEdge(2, 4, 6);
        dag.AddEdge(2, 5, 7);
        dag.AddEdge(4, 7, 9);
        dag.AddEdge(3, 7, 9);

        arch.SetNumberOfProcessors(2);
        arch.SetCommunicationCosts(1);
        arch.SetSynchronisationCosts(1);
    }

    BspSchedule<Graph> &Build() {
        instance = new BspInstance<Graph>(dag, arch);
        schedule = new BspSchedule<Graph>(*instance);

        //   0(p1,s0)->1(p1,s1) same proc ok
        //   0(p1,s0)->2(p0,s1) cross ok
        //   0(p1,s0)->3(p0,s1) cross ok
        //   1(p1,s1)->4(p1,s3) same proc ok
        //   2(p0,s1)->4(p1,s3) cross ok
        //   2(p0,s1)->5(p0,s3) same proc ok
        //   4(p1,s3)->7(p1,s4) same proc ok
        //   3(p0,s1)->7(p1,s4) cross ok
        schedule->SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
        schedule->SetAssignedSupersteps({0, 1, 1, 1, 3, 3, 4, 4});
        schedule->UpdateNumberOfSupersteps();
        return *schedule;
    }

    ~EightNodeGraph() {
        delete schedule;
        delete instance;
    }
};

// ============================================================================
// TEST 1: Inner-loop cost-tracking consistency (Eager)
//
// Uses KlImproverTest to run individual inner iterations and checks that
// the incremental cost tracking matches full recomputation at each step.
// This validates both ComputeCommAffinity gain prediction and
// UpdateDatastructureAfterMove correctness.
// ============================================================================

BOOST_AUTO_TEST_CASE(InnerLoopCostConsistencyEager) {
    EightNodeGraph g;
    auto &schedule = g.Build();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    using CommCostT = BspCommCostF<EagerCommCostPolicy>;
    using TestT = KlImproverTest<Graph, CommCostT>;

    TestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK_EQUAL(kl.GetActiveSchedule().IsFeasible(), true);

    CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    CostT tracked = kl.GetCurrentCost();
    BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);

    auto nodeSelection = kl.InsertGainHeapTest({0, 7});

    RunInnerLoopAndCheckCost(kl, 4, "Eager");
}

// ============================================================================
// TEST 2: Inner-loop cost-tracking consistency (Lazy)
// ============================================================================

BOOST_AUTO_TEST_CASE(InnerLoopCostConsistencyLazy) {
    EightNodeGraph g;
    auto &schedule = g.Build();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    using CommCostT = BspCommCostF<LazyCommCostPolicy>;
    using TestT = KlImproverTest<Graph, CommCostT>;

    TestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK_EQUAL(kl.GetActiveSchedule().IsFeasible(), true);

    CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    CostT tracked = kl.GetCurrentCost();
    BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);

    auto nodeSelection = kl.InsertGainHeapTest({0, 7});

    RunInnerLoopAndCheckCost(kl, 4, "Lazy");
}

// ============================================================================
// TEST 3: Inner-loop cost-tracking consistency (Buffered)
// ============================================================================

BOOST_AUTO_TEST_CASE(InnerLoopCostConsistencyBuffered) {
    EightNodeGraph g;
    auto &schedule = g.Build();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    using CommCostT = BspCommCostF<BufferedCommCostPolicy>;
    using TestT = KlImproverTest<Graph, CommCostT>;

    TestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK_EQUAL(kl.GetActiveSchedule().IsFeasible(), true);

    CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    CostT tracked = kl.GetCurrentCost();
    BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);

    auto nodeSelection = kl.InsertGainHeapTest({0, 7});

    RunInnerLoopAndCheckCost(kl, 4, "Buffered");
}

// ============================================================================
// TEST 4: Inner-loop consistency on SmallFanGraph (all policies)
// ============================================================================

BOOST_AUTO_TEST_CASE(InnerLoopSmallFanAllPolicies) {
    auto RunPolicyTest = [](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);
        using CommCostT = BspCommCostF<Policy>;
        using TestT = KlImproverTest<Graph, CommCostT>;

        SmallFanGraph g;
        auto &schedule = g.Build();

        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": initial schedule violates precedence");

        TestT kl;
        kl.SetupSchedule(schedule);

        BOOST_CHECK_MESSAGE(kl.GetActiveSchedule().IsFeasible(), name + ": initial schedule must be feasible");

        CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
        CostT tracked = kl.GetCurrentCost();
        BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);

        auto nodeSelection = kl.InsertGainHeapTest({0, 5});

        RunInnerLoopAndCheckCost(kl, 4, name);
    };

    RunPolicyTest(EagerCommCostPolicy{}, "EagerFan");
    RunPolicyTest(LazyCommCostPolicy{}, "LazyFan");
    RunPolicyTest(BufferedCommCostPolicy{}, "BufferedFan");
}

// ============================================================================
// TEST 5: Inner-loop consistency on 3-processor graph (all policies)
// ============================================================================

BOOST_AUTO_TEST_CASE(InnerLoopThreeProcsAllPolicies) {
    Graph dag;
    dag.AddVertex(5, 1, 4);    // 0
    dag.AddVertex(3, 1, 6);    // 1
    dag.AddVertex(4, 1, 3);    // 2
    dag.AddVertex(2, 1, 5);    // 3
    dag.AddVertex(6, 1, 2);    // 4
    dag.AddVertex(3, 1, 4);    // 5

    dag.AddEdge(0, 2, 1);
    dag.AddEdge(0, 3, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(1, 4, 1);
    dag.AddEdge(2, 5, 1);
    dag.AddEdge(3, 5, 1);
    dag.AddEdge(4, 5, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(3);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    //   0(p0,s0)->2(p2,s1) cross ok
    //   0(p0,s0)->3(p0,s1) same proc ok
    //   1(p1,s0)->3(p0,s1) cross ok
    //   1(p1,s0)->4(p1,s1) same proc ok
    //   2(p2,s1)->5(p2,s2) same proc ok
    //   3(p0,s1)->5(p2,s2) cross ok
    //   4(p1,s1)->5(p2,s2) cross ok
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    auto RunPolicyTest = [&](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);
        using CommCostT = BspCommCostF<Policy>;
        using TestT = KlImproverTest<Graph, CommCostT>;

        TestT kl;
        kl.SetupSchedule(schedule);

        BOOST_CHECK_MESSAGE(kl.GetActiveSchedule().IsFeasible(), name + ": initial schedule must be feasible");

        CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
        CostT tracked = kl.GetCurrentCost();
        BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);

        auto nodeSelection = kl.InsertGainHeapTest({0, 5});

        RunInnerLoopAndCheckCost(kl, 4, name);
    };

    RunPolicyTest(EagerCommCostPolicy{}, "Eager3P");
    RunPolicyTest(LazyCommCostPolicy{}, "Lazy3P");
    RunPolicyTest(BufferedCommCostPolicy{}, "Buffered3P");
}

// ============================================================================
// TEST 6: Full ImproveSchedule - Eager policy
// ============================================================================

BOOST_AUTO_TEST_CASE(FullImproveScheduleEager) {
    SmallFanGraph g;
    auto &schedule = g.Build();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    BspImprover<EagerCommCostPolicy> kl(42);
    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    BOOST_TEST_MESSAGE("Eager ImproveSchedule: status=" << (status == ReturnStatus::OSP_SUCCESS ? "SUCCESS" : "BEST_FOUND")
                                                        << " steps=" << schedule.NumberOfSupersteps());
}

// ============================================================================
// TEST 7: Full ImproveSchedule - Lazy policy
// ============================================================================

BOOST_AUTO_TEST_CASE(FullImproveScheduleLazy) {
    SmallFanGraph g;
    auto &schedule = g.Build();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    BspImprover<LazyCommCostPolicy> kl(42);
    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    BOOST_TEST_MESSAGE("Lazy ImproveSchedule: steps=" << schedule.NumberOfSupersteps());
}

// ============================================================================
// TEST 8: Full ImproveSchedule - Buffered policy
// ============================================================================

BOOST_AUTO_TEST_CASE(FullImproveScheduleBuffered) {
    SmallFanGraph g;
    auto &schedule = g.Build();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    BspImprover<BufferedCommCostPolicy> kl(42);
    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    BOOST_TEST_MESSAGE("Buffered ImproveSchedule: steps=" << schedule.NumberOfSupersteps());
}

// ============================================================================
// TEST 9: Full ImproveSchedule on EightNodeGraph - all policies
// ============================================================================

BOOST_AUTO_TEST_CASE(FullImproveScheduleEightNode) {
    auto RunForPolicy = [](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);

        EightNodeGraph g;
        auto &schedule = g.Build();

        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": initial schedule violates precedence");

        BspImprover<Policy> kl(42);
        auto status = kl.ImproveSchedule(schedule);

        BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
                            name + ": unexpected return status");
        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": precedence violated after improvement");

        BOOST_TEST_MESSAGE(name << ": completed, steps=" << schedule.NumberOfSupersteps());
    };

    RunForPolicy(EagerCommCostPolicy{}, "Eager8");
    RunForPolicy(LazyCommCostPolicy{}, "Lazy8");
    RunForPolicy(BufferedCommCostPolicy{}, "Buffered8");
}

// ============================================================================
// TEST 10: ImproveSchedule with 3 processors and non-uniform send costs
// ============================================================================

BOOST_AUTO_TEST_CASE(FullImproveScheduleThreeProcs) {
    Graph dag;
    dag.AddVertex(3, 1, 5);    // 0
    dag.AddVertex(4, 1, 3);    // 1
    dag.AddVertex(2, 1, 4);    // 2
    dag.AddVertex(5, 1, 6);    // 3
    dag.AddVertex(3, 1, 2);    // 4
    dag.AddVertex(6, 1, 3);    // 5
    dag.AddVertex(4, 1, 5);    // 6

    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 3, 1);
    dag.AddEdge(3, 4, 1);
    dag.AddEdge(3, 5, 1);
    dag.AddEdge(4, 6, 1);
    dag.AddEdge(5, 6, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(4);

    // Non-uniform send costs
    std::vector<std::vector<int>> sendCosts = {
        {0, 1, 3},
        {1, 0, 2},
        {3, 2, 0}
    };
    arch.SetSendCosts(sendCosts);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    //   0(p0,s0)->1(p1,s1) cross ok
    //   0(p0,s0)->2(p2,s1) cross ok
    //   1(p1,s1)->3(p0,s2) cross ok
    //   2(p2,s1)->3(p0,s2) cross ok
    //   3(p0,s2)->4(p1,s3) cross ok
    //   3(p0,s2)->5(p2,s3) cross ok
    //   4(p1,s3)->6(p0,s4) cross ok
    //   5(p2,s3)->6(p0,s4) cross ok
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 1, 2, 3, 3, 4});
    schedule.UpdateNumberOfSupersteps();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    auto RunForPolicy = [&](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);

        BspSchedule<Graph> sched(schedule);

        BspImprover<Policy> kl(42);
        auto status = kl.ImproveSchedule(sched);

        BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
                            name + ": unexpected status");
        BOOST_CHECK_MESSAGE(sched.SatisfiesPrecedenceConstraints(), name + ": precedence violated");

        BOOST_TEST_MESSAGE(name << ": completed, steps=" << sched.NumberOfSupersteps());
    };

    RunForPolicy(EagerCommCostPolicy{}, "Eager3P");
    RunForPolicy(LazyCommCostPolicy{}, "Lazy3P");
    RunForPolicy(BufferedCommCostPolicy{}, "Buffered3P");
}

// ============================================================================
// TEST 11: Single-processor chain (no comm cost, should not regress)
// ============================================================================

BOOST_AUTO_TEST_CASE(SingleProcChainNoRegression) {
    Graph dag;
    dag.AddVertex(3, 1, 5);
    dag.AddVertex(4, 1, 3);
    dag.AddVertex(2, 1, 4);

    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(3);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    schedule.SetAssignedProcessors({0, 0, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    BspImprover<EagerCommCostPolicy> kl(42);
    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
}

// ============================================================================
// TEST 12: Window size 2 (wider search window)
// ============================================================================

BOOST_AUTO_TEST_CASE(WindowSize2ImproveSchedule) {
    EightNodeGraph g;
    auto &schedule = g.Build();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    BspImprover<EagerCommCostPolicy, 2> kl(42);
    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    BOOST_TEST_MESSAGE("Window2: completed, steps=" << schedule.NumberOfSupersteps());
}

// ============================================================================
// TEST 13: Dense diamond DAG with 3 processors (all policies)
// 0->{1,2}, {1,2}->3
// Steps: 0:s0p0, 1:s1p1, 2:s1p2, 3:s2p0
// ============================================================================

BOOST_AUTO_TEST_CASE(FullImproveScheduleDiamondThreeProcs) {
    Graph dag;
    dag.AddVertex(4, 1, 8);    // 0
    dag.AddVertex(3, 1, 5);    // 1
    dag.AddVertex(5, 1, 7);    // 2
    dag.AddVertex(6, 1, 3);    // 3

    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(3);
    arch.SetSynchronisationCosts(2);

    BspInstance<Graph> instance(dag, arch);

    auto RunForPolicy = [&](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);

        BspSchedule<Graph> schedule(instance);
        schedule.SetAssignedProcessors({0, 1, 2, 0});
        schedule.SetAssignedSupersteps({0, 1, 1, 2});
        schedule.UpdateNumberOfSupersteps();

        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

        BspImprover<Policy> kl(42);
        auto status = kl.ImproveSchedule(schedule);

        BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
                            name + ": unexpected status");
        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": precedence violated");

        BOOST_TEST_MESSAGE(name << ": completed, steps=" << schedule.NumberOfSupersteps());
    };

    RunForPolicy(EagerCommCostPolicy{}, "EagerDiamond");
    RunForPolicy(LazyCommCostPolicy{}, "LazyDiamond");
    RunForPolicy(BufferedCommCostPolicy{}, "BufferedDiamond");
}
