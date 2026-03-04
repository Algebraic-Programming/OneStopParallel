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

/// End-to-end tests for KlMaxBspCommCostFunction under all three
/// communication-cost policies (Eager, Lazy, Buffered).
///
/// Uses MaxBspSchedule (staleness=2) for all tests.
///
/// Tests verify:
///   1. Inner-loop cost-tracking consistency (ComputeScheduleCostTest ==
///      GetCurrentCost after each RunInnerIterationTest).
///   2. Full ImproveSchedule: precedence constraints are satisfied and
///      staleness=2 cross-processor gaps are respected.
///   3. All three policies produce valid results on varied topologies.
///   4. Cost monotonicity: ImproveSchedule never increases cost.
///   5. Large end-to-end runs on real graphs from LargeSpaaGraphs().
///
/// Test structure:
///   Tests 1-5:   Inner-loop consistency (small hand-crafted graphs)
///   Tests 6-9:   Full ImproveSchedule (SmallFanGraph, EightNodeGraph)
///   Tests 10-13: Full ImproveSchedule (3 procs, single proc, window 2, diamond)
///   Tests 14-17: Medium programmatic graphs (wide DAG, pipeline, random, send costs)
///   Test 18:     Cost monotonicity verification
///   Large suite: LargeSpaaGraphs() × {Eager, Lazy, Buffered, Window2}
///               (uses GreedyVarianceSspScheduler for initial MaxBspSchedule)
///   MT suite:    Multi-threaded KlSyncParallelImprover tests (MT-1 to MT-5)
///               Verifies correctness, no-regression, and large-graph support
///
/// NOTE: With staleness=2 and WindowSize=1, superstep gaps are wide
/// (0,0,2,2,4,4,...), so few nodes have valid moves within the window.

#define BOOST_TEST_MODULE kl_max_bsp_improver
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/model/MaxBspSchedule.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyVarianceSspScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_max_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_mt.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_test.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using VertexType = Graph::VertexIdx;
using CostT = double;

// ============================================================================
//  Type aliases
// ============================================================================

template <typename CommPolicy, unsigned WindowSize = 1>
using MaxCommCostF = KlMaxBspCommCostFunction<Graph, CostT, NoLocalSearchMemoryConstraint, CommPolicy, WindowSize>;

template <typename CommPolicy, unsigned WindowSize = 1>
using MaxBspImprover = KlImprover<Graph, MaxCommCostF<CommPolicy, WindowSize>, NoLocalSearchMemoryConstraint, WindowSize, CostT>;

template <typename CommPolicy, unsigned WindowSize = 1>
using MaxBspImproverMt
    = KlSyncParallelImprover<Graph, MaxCommCostF<CommPolicy, WindowSize>, NoLocalSearchMemoryConstraint, WindowSize, CostT>;

// ============================================================================
//  Helper: verify staleness constraints in a schedule
// ============================================================================
static void VerifyStalenessConstraints(const BspSchedule<Graph> &schedule) {
    const unsigned staleness = schedule.GetStaleness();
    const auto &dag = schedule.GetInstance().GetComputationalDag();
    for (const auto &u : dag.Vertices()) {
        for (const auto &v : dag.Children(u)) {
            const unsigned uStep = schedule.AssignedSuperstep(u);
            const unsigned vStep = schedule.AssignedSuperstep(v);
            const unsigned uProc = schedule.AssignedProcessor(u);
            const unsigned vProc = schedule.AssignedProcessor(v);

            const unsigned gap = (uProc != vProc) ? staleness : 0;

            BOOST_CHECK_GE(vStep, uStep + gap);
            if (vStep < uStep + gap) {
                BOOST_TEST_MESSAGE("Staleness violation: edge " << u << "->" << v << " uStep=" << uStep << " vStep=" << vStep
                                                                << " uProc=" << uProc << " vProc=" << vProc
                                                                << " staleness=" << staleness << " required gap=" << gap);
            }
        }
    }
}

// ============================================================================
//  Inner-loop helper: run up to maxIter iterations, break on cost mismatch
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
///
///     0           step 0, proc 0
///    /|
///   1  2  3       step 2, proc 1/0/1
///    \|/
///     4           step 4, proc 0
///     |
///     5           step 6, proc 1
///
struct SmallFanGraph {
    Graph dag;
    BspArchitecture<Graph> arch;
    BspInstance<Graph> *instance = nullptr;
    MaxBspSchedule<Graph> *schedule = nullptr;

    SmallFanGraph() {
        //                          work  comm  mem
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

    MaxBspSchedule<Graph> &Build() {
        instance = new BspInstance<Graph>(dag, arch);
        schedule = new MaxBspSchedule<Graph>(*instance);

        // All cross-processor edges have gap >= 2 (staleness=2).
        //   0(p0,s0)->1(p1,s2) cross gap=2 ok
        //   0(p0,s0)->2(p0,s2) same proc ok
        //   0(p0,s0)->3(p1,s2) cross gap=2 ok
        //   1(p1,s2)->4(p0,s4) cross gap=2 ok
        //   2(p0,s2)->4(p0,s4) same proc ok
        //   3(p1,s2)->4(p0,s4) cross gap=2 ok
        //   4(p0,s4)->5(p1,s6) cross gap=2 ok
        schedule->SetAssignedProcessors({0, 1, 0, 1, 0, 1});
        schedule->SetAssignedSupersteps({0, 2, 2, 2, 4, 6});
        schedule->UpdateNumberOfSupersteps();
        return *schedule;
    }

    ~SmallFanGraph() {
        delete schedule;
        delete instance;
    }
};

/// 8-node graph (adapted from kl_bsp_improver_test.cpp for staleness=2).
///
///   0->1, 0->2, 0->3, 1->4, 2->4, 2->5, 4->7, 3->7
///
struct EightNodeGraph {
    Graph dag;
    BspArchitecture<Graph> arch;
    BspInstance<Graph> *instance = nullptr;
    MaxBspSchedule<Graph> *schedule = nullptr;

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

    MaxBspSchedule<Graph> &Build() {
        instance = new BspInstance<Graph>(dag, arch);
        schedule = new MaxBspSchedule<Graph>(*instance);

        //   0(p1,s0)->1(p1,s0) same proc ok
        //   0(p1,s0)->2(p0,s2) cross gap=2 ok
        //   0(p1,s0)->3(p0,s2) cross gap=2 ok
        //   1(p1,s0)->4(p1,s4) same proc ok
        //   2(p0,s2)->4(p1,s4) cross gap=2 ok
        //   2(p0,s2)->5(p0,s4) same proc ok
        //   4(p1,s4)->7(p1,s6) same proc ok
        //   3(p0,s2)->7(p1,s6) cross gap=4 ok
        schedule->SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
        schedule->SetAssignedSupersteps({0, 0, 2, 2, 4, 4, 6, 6});
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
// ============================================================================

BOOST_AUTO_TEST_CASE(InnerLoopCostConsistencyEager) {
    EightNodeGraph g;
    auto &schedule = g.Build();

    BOOST_CHECK_EQUAL(schedule.GetStaleness(), 2u);

    using CommCostT = MaxCommCostF<EagerCommCostPolicy>;
    using TestT = KlImproverTest<Graph, CommCostT>;

    TestT kl;
    kl.SetupSchedule(schedule);

    auto &klSched = kl.GetActiveSchedule();
    BOOST_CHECK_EQUAL(klSched.NumSteps(), 7);
    BOOST_CHECK_EQUAL(klSched.IsFeasible(), true);
    BOOST_CHECK_EQUAL(klSched.GetStaleness(), 2u);

    // Initial cost consistency
    CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    CostT tracked = kl.GetCurrentCost();
    BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);

    // Insert nodes into gain heap and iterate.
    // Insert nodes into gain heap with penalty/reward.
    auto nodeSelection = kl.InsertGainHeapTestPenalty({0, 7});

    RunInnerLoopAndCheckCost(kl, 2, "Eager");
}

// ============================================================================
// TEST 2: Inner-loop cost-tracking consistency (Lazy)
// ============================================================================

BOOST_AUTO_TEST_CASE(InnerLoopCostConsistencyLazy) {
    EightNodeGraph g;
    auto &schedule = g.Build();

    using CommCostT = MaxCommCostF<LazyCommCostPolicy>;
    using TestT = KlImproverTest<Graph, CommCostT>;

    TestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK_EQUAL(kl.GetActiveSchedule().IsFeasible(), true);

    CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    CostT tracked = kl.GetCurrentCost();
    BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);

    auto nodeSelection = kl.InsertGainHeapTestPenalty({0, 7});

    RunInnerLoopAndCheckCost(kl, 2, "Lazy");
}

// ============================================================================
// TEST 3: Inner-loop cost-tracking consistency (Buffered)
// ============================================================================

BOOST_AUTO_TEST_CASE(InnerLoopCostConsistencyBuffered) {
    EightNodeGraph g;
    auto &schedule = g.Build();

    using CommCostT = MaxCommCostF<BufferedCommCostPolicy>;
    using TestT = KlImproverTest<Graph, CommCostT>;

    TestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK_EQUAL(kl.GetActiveSchedule().IsFeasible(), true);

    CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    CostT tracked = kl.GetCurrentCost();
    BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);

    auto nodeSelection = kl.InsertGainHeapTestPenalty({0, 7});

    RunInnerLoopAndCheckCost(kl, 2, "Buffered");
}

// ============================================================================
// TEST 4: Inner-loop consistency on SmallFanGraph (all policies)
// ============================================================================

BOOST_AUTO_TEST_CASE(InnerLoopSmallFanAllPolicies) {
    auto RunPolicyTest = [](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);
        using CommCostT = MaxCommCostF<Policy>;
        using TestT = KlImproverTest<Graph, CommCostT>;

        SmallFanGraph g;
        auto &schedule = g.Build();

        TestT kl;
        kl.SetupSchedule(schedule);

        BOOST_CHECK_MESSAGE(kl.GetActiveSchedule().IsFeasible(), name + ": initial schedule must be feasible");

        CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
        CostT tracked = kl.GetCurrentCost();
        BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);

        auto nodeSelection = kl.InsertGainHeapTestPenalty({0, 5});

        RunInnerLoopAndCheckCost(kl, 2, name);
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
    MaxBspSchedule<Graph> schedule(instance);

    //   0(p0,s0)->2(p2,s2) cross gap=2 ok
    //   0(p0,s0)->3(p0,s2) same proc ok
    //   1(p1,s0)->3(p0,s2) cross gap=2 ok
    //   1(p1,s0)->4(p1,s2) same proc ok
    //   2(p2,s2)->5(p2,s4) same proc ok
    //   3(p0,s2)->5(p2,s4) cross gap=2 ok
    //   4(p1,s2)->5(p2,s4) cross gap=2 ok
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 2, 2, 2, 4});
    schedule.UpdateNumberOfSupersteps();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    auto RunPolicyTest = [&](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);
        using CommCostT = MaxCommCostF<Policy>;
        using TestT = KlImproverTest<Graph, CommCostT>;

        TestT kl;
        kl.SetupSchedule(schedule);

        BOOST_CHECK_MESSAGE(kl.GetActiveSchedule().IsFeasible(), name + ": initial schedule must be feasible");

        CostT recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
        CostT tracked = kl.GetCurrentCost();
        BOOST_CHECK_CLOSE(recomputed, tracked, 0.00001);

        auto nodeSelection = kl.InsertGainHeapTestPenalty({0, 5});

        RunInnerLoopAndCheckCost(kl, 2, name);
    };

    RunPolicyTest(EagerCommCostPolicy{}, "Eager3P");
    RunPolicyTest(LazyCommCostPolicy{}, "Lazy3P");
    RunPolicyTest(BufferedCommCostPolicy{}, "Buffered3P");
}

// ============================================================================
// TEST 6: Full ImproveSchedule - Eager policy
//
// Runs the complete KL improver and verifies precedence + staleness.
// ============================================================================

BOOST_AUTO_TEST_CASE(FullImproveScheduleEager) {
    SmallFanGraph g;
    auto &schedule = g.Build();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    VerifyStalenessConstraints(schedule);

    MaxBspImprover<EagerCommCostPolicy> kl(42);
    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    VerifyStalenessConstraints(schedule);

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
    VerifyStalenessConstraints(schedule);

    MaxBspImprover<LazyCommCostPolicy> kl(42);
    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    VerifyStalenessConstraints(schedule);

    BOOST_TEST_MESSAGE("Lazy ImproveSchedule: steps=" << schedule.NumberOfSupersteps());
}

// ============================================================================
// TEST 8: Full ImproveSchedule - Buffered policy
// ============================================================================

BOOST_AUTO_TEST_CASE(FullImproveScheduleBuffered) {
    SmallFanGraph g;
    auto &schedule = g.Build();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    VerifyStalenessConstraints(schedule);

    MaxBspImprover<BufferedCommCostPolicy> kl(42);
    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    VerifyStalenessConstraints(schedule);

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
        VerifyStalenessConstraints(schedule);

        MaxBspImprover<Policy> kl(42);
        auto status = kl.ImproveSchedule(schedule);

        BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
                            name + ": unexpected return status");
        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": precedence violated after improvement");
        VerifyStalenessConstraints(schedule);

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
    MaxBspSchedule<Graph> schedule(instance);

    // Chain with cross-proc edges, all gaps >= 2:
    //   0(p0,s0)->1(p1,s2) cross gap=2
    //   0(p0,s0)->2(p2,s2) cross gap=2
    //   1(p1,s2)->3(p0,s4) cross gap=2
    //   2(p2,s2)->3(p0,s4) cross gap=2
    //   3(p0,s4)->4(p1,s6) cross gap=2
    //   3(p0,s4)->5(p2,s6) cross gap=2
    //   4(p1,s6)->6(p0,s8) cross gap=2
    //   5(p2,s6)->6(p0,s8) cross gap=2
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 2, 2, 4, 6, 6, 8});
    schedule.UpdateNumberOfSupersteps();

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    VerifyStalenessConstraints(schedule);

    auto RunForPolicy = [&](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);

        // Copy schedule so each policy starts fresh
        MaxBspSchedule<Graph> sched(schedule);

        MaxBspImprover<Policy> kl(42);
        auto status = kl.ImproveSchedule(sched);

        BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
                            name + ": unexpected status");
        BOOST_CHECK_MESSAGE(sched.SatisfiesPrecedenceConstraints(), name + ": precedence violated");
        VerifyStalenessConstraints(sched);

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
    MaxBspSchedule<Graph> schedule(instance);

    // All nodes on same proc, sequential steps -> no comm cost
    schedule.SetAssignedProcessors({0, 0, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    MaxBspImprover<EagerCommCostPolicy> kl(42);
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
    VerifyStalenessConstraints(schedule);

    MaxBspImprover<EagerCommCostPolicy, 2> kl(42);
    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    VerifyStalenessConstraints(schedule);

    BOOST_TEST_MESSAGE("Window2: completed, steps=" << schedule.NumberOfSupersteps());
}

// ============================================================================
// TEST 13: Dense diamond DAG with 3 processors
//
//     0       step 0, proc 0
//    /
//   1   2     step 2, proc 1/2
//    \ /
//     3       step 4, proc 0
//
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

        MaxBspSchedule<Graph> schedule(instance);
        schedule.SetAssignedProcessors({0, 1, 2, 0});
        schedule.SetAssignedSupersteps({0, 2, 2, 4});
        schedule.UpdateNumberOfSupersteps();

        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
        VerifyStalenessConstraints(schedule);

        MaxBspImprover<Policy> kl(42);
        auto status = kl.ImproveSchedule(schedule);

        BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
                            name + ": unexpected status");
        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": precedence violated");
        VerifyStalenessConstraints(schedule);

        BOOST_TEST_MESSAGE(name << ": completed, steps=" << schedule.NumberOfSupersteps());
    };

    RunForPolicy(EagerCommCostPolicy{}, "EagerDiamond");
    RunForPolicy(LazyCommCostPolicy{}, "LazyDiamond");
    RunForPolicy(BufferedCommCostPolicy{}, "BufferedDiamond");
}

// ============================================================================
// Helpers for large end-to-end tests
// ============================================================================

/// Assign varied work/mem/comm weights to every vertex in a graph that was
/// loaded from a file (which typically only has unit weights).
template <typename GraphT>
static void AddMemWeights(GraphT &dag) {
    int memWeight = 1;
    int commWeight = 7;
    for (const auto &v : dag.Vertices()) {
        dag.SetVertexWorkWeight(v, static_cast<VWorkwT<GraphT>>(memWeight++ % 10 + 2));
        dag.SetVertexMemWeight(v, static_cast<VMemwT<GraphT>>(memWeight++ % 10 + 2));
        dag.SetVertexCommWeight(v, static_cast<VCommwT<GraphT>>(commWeight++ % 10 + 2));
    }
}

// ============================================================================
// TEST 14: Programmatic medium graph — wide layer DAG (20 nodes, 3 procs)
//
// Topology:    source (S0) → 6 middle nodes (S2) → sink (S4)
//              + lateral edges between middle nodes
//
// Tests that MaxBSP handles many-to-many communication patterns.
// ============================================================================

BOOST_AUTO_TEST_CASE(MediumWideDagAllPolicies) {
    Graph dag;

    // Source node
    dag.AddVertex(5, 8, 1);    // v0

    // 6 middle-layer nodes with varied weights
    for (int i = 1; i <= 6; ++i) {
        dag.AddVertex(3 + i, 2 + i, 1);
    }

    // Sink node
    dag.AddVertex(6, 4, 1);    // v7

    // Source → all middle
    for (int i = 1; i <= 6; ++i) {
        dag.AddEdge(0, i, 1);
    }
    // All middle → sink
    for (int i = 1; i <= 6; ++i) {
        dag.AddEdge(i, 7, 1);
    }
    // Lateral edges: chain through middle layer (same step, tests same-step moves)
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(3, 4, 1);
    dag.AddEdge(5, 6, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(3);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);

    auto RunForPolicy = [&](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);

        MaxBspSchedule<Graph> schedule(instance);
        // Source P0/S0. Lateral pairs on same proc: (1,2)→P1, (3,4)→P2, (5,6)→P0.
        // Sink on P1/S4.
        //   0→1: P0→P1 S0→S2 gap=2 ok     0→2: P0→P1 S0→S2 gap=2 ok
        //   0→3: P0→P2 S0→S2 gap=2 ok     0→4: P0→P2 S0→S2 gap=2 ok
        //   0→5: P0→P0 same ok             0→6: P0→P0 same ok
        //   1→2: P1→P1 same ok             3→4: P2→P2 same ok     5→6: P0→P0 same ok
        //   1→7: P1→P1 same ok             3→7: P2→P1 S2→S4 gap=2 ok
        //   5→7: P0→P1 S2→S4 gap=2 ok
        schedule.SetAssignedProcessors({0, 1, 1, 2, 2, 0, 0, 1});
        schedule.SetAssignedSupersteps({0, 2, 2, 2, 2, 2, 2, 4});
        schedule.UpdateNumberOfSupersteps();

        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": initial precedence violated");
        VerifyStalenessConstraints(schedule);

        MaxBspImprover<Policy> kl(42);
        auto status = kl.ImproveSchedule(schedule);

        BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
                            name + ": unexpected status");
        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": precedence violated after improvement");
        VerifyStalenessConstraints(schedule);

        BOOST_TEST_MESSAGE(name << ": completed, steps=" << schedule.NumberOfSupersteps());
    };

    RunForPolicy(EagerCommCostPolicy{}, "EagerWide");
    RunForPolicy(LazyCommCostPolicy{}, "LazyWide");
    RunForPolicy(BufferedCommCostPolicy{}, "BufferedWide");
}

// ============================================================================
// TEST 15: Programmatic medium graph — multi-level pipeline (5 layers)
//
// Tests that step removal works correctly with many supersteps and the
// coupled cost formula's analytical delta is accurate.
// ============================================================================

BOOST_AUTO_TEST_CASE(MediumPipelineFiveLayers) {
    Graph dag;

    // 5 layers × 2 nodes each = 10 nodes
    for (int i = 0; i < 10; ++i) {
        dag.AddVertex(4 + (i % 3), 3 + (i % 5), 1);
    }

    // Edges: each layer connects to next layer (full bipartite)
    for (int layer = 0; layer < 4; ++layer) {
        for (int src = 0; src < 2; ++src) {
            for (int dst = 0; dst < 2; ++dst) {
                dag.AddEdge(layer * 2 + src, (layer + 1) * 2 + dst, 1);
            }
        }
    }

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(4);

    BspInstance<Graph> instance(dag, arch);

    auto RunForPolicy = [&](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);

        MaxBspSchedule<Graph> schedule(instance);
        // Alternating procs, steps spaced by 2 (staleness=2)
        schedule.SetAssignedProcessors({0, 1, 1, 0, 0, 1, 1, 0, 0, 1});
        schedule.SetAssignedSupersteps({0, 0, 2, 2, 4, 4, 6, 6, 8, 8});
        schedule.UpdateNumberOfSupersteps();

        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": initial precedence violated");
        VerifyStalenessConstraints(schedule);

        MaxBspImprover<Policy> kl(42);
        auto status = kl.ImproveSchedule(schedule);

        BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
                            name + ": unexpected status");
        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": precedence violated");
        VerifyStalenessConstraints(schedule);

        BOOST_TEST_MESSAGE(name << ": completed, steps=" << schedule.NumberOfSupersteps());
    };

    RunForPolicy(EagerCommCostPolicy{}, "EagerPipeline");
    RunForPolicy(LazyCommCostPolicy{}, "LazyPipeline");
    RunForPolicy(BufferedCommCostPolicy{}, "BufferedPipeline");
}

// ============================================================================
// TEST 16: Randomly generated DAG (30 nodes, 4 procs)
//
// Generates a layered random DAG to stress-test the improver with non-trivial
// topology. Verifies the improver doesn't crash and constraints hold.
// ============================================================================

BOOST_AUTO_TEST_CASE(RandomLayeredDag30Nodes) {
    Graph dag;
    std::mt19937 rng(12345);

    constexpr unsigned kNumLayers = 5;
    constexpr unsigned kNodesPerLayer = 6;
    constexpr unsigned kTotalNodes = kNumLayers * kNodesPerLayer;

    // Create nodes with random weights
    for (unsigned i = 0; i < kTotalNodes; ++i) {
        unsigned work = 2 + (rng() % 10);
        unsigned comm = 2 + (rng() % 8);
        dag.AddVertex(work, comm, 1);
    }

    // Add edges: each node in layer L connects to 1-3 random nodes in layer L+1
    std::uniform_int_distribution<unsigned> fanDist(1, 3);
    for (unsigned layer = 0; layer < kNumLayers - 1; ++layer) {
        for (unsigned src = 0; src < kNodesPerLayer; ++src) {
            unsigned srcIdx = layer * kNodesPerLayer + src;
            unsigned numChildren = fanDist(rng);
            // Shuffle destination indices for random selection
            std::vector<unsigned> dstIndices(kNodesPerLayer);
            std::iota(dstIndices.begin(), dstIndices.end(), (layer + 1) * kNodesPerLayer);
            std::shuffle(dstIndices.begin(), dstIndices.end(), rng);
            for (unsigned c = 0; c < std::min(numChildren, kNodesPerLayer); ++c) {
                dag.AddEdge(srcIdx, dstIndices[c], 1);
            }
        }
    }

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(3);
    arch.SetSynchronisationCosts(6);

    BspInstance<Graph> instance(dag, arch);

    auto RunForPolicy = [&](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);

        MaxBspSchedule<Graph> schedule(instance);

        // Assign procs round-robin within layer, steps = layer * 2
        std::vector<unsigned> procs(kTotalNodes), steps(kTotalNodes);
        for (unsigned i = 0; i < kTotalNodes; ++i) {
            unsigned layer = i / kNodesPerLayer;
            procs[i] = i % 4;
            steps[i] = layer * 2;
        }
        schedule.SetAssignedProcessors(procs);
        schedule.SetAssignedSupersteps(steps);
        schedule.UpdateNumberOfSupersteps();

        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": initial precedence violated");
        VerifyStalenessConstraints(schedule);

        MaxBspImprover<Policy> kl(42);
        auto status = kl.ImproveSchedule(schedule);

        BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
                            name + ": unexpected status");
        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": precedence violated");
        VerifyStalenessConstraints(schedule);

        BOOST_TEST_MESSAGE(name << ": completed, steps=" << schedule.NumberOfSupersteps());
    };

    RunForPolicy(EagerCommCostPolicy{}, "EagerRandom30");
    RunForPolicy(LazyCommCostPolicy{}, "LazyRandom30");
    RunForPolicy(BufferedCommCostPolicy{}, "BufferedRandom30");
}

// ============================================================================
// TEST 17: Non-uniform send costs (4 procs, asymmetric)
//
// Tests that the coupled cost formula handles non-uniform send costs
// correctly. Different (proc_from, proc_to) pairs have different costs.
// ============================================================================

BOOST_AUTO_TEST_CASE(NonUniformSendCostsFourProcs) {
    Graph dag;
    // 8-node diamond-chain
    dag.AddVertex(5, 6, 1);    // 0
    dag.AddVertex(3, 4, 1);    // 1
    dag.AddVertex(4, 5, 1);    // 2
    dag.AddVertex(3, 3, 1);    // 3
    dag.AddVertex(6, 7, 1);    // 4
    dag.AddVertex(4, 3, 1);    // 5
    dag.AddVertex(5, 8, 1);    // 6
    dag.AddVertex(3, 2, 1);    // 7

    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 3, 1);
    dag.AddEdge(3, 4, 1);
    dag.AddEdge(3, 5, 1);
    dag.AddEdge(4, 6, 1);
    dag.AddEdge(5, 6, 1);
    dag.AddEdge(6, 7, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(5);

    std::vector<std::vector<int>> sendCosts = {
        {0, 1, 3, 4},
        {1, 0, 2, 3},
        {3, 2, 0, 1},
        {4, 3, 1, 0}
    };
    arch.SetSendCosts(sendCosts);

    BspInstance<Graph> instance(dag, arch);

    auto RunForPolicy = [&](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);

        MaxBspSchedule<Graph> schedule(instance);
        //   0→1: P0→P1 S0→S2 gap=2 ok     0→2: P0→P2 S0→S2 gap=2 ok
        //   1→3: P1→P3 S2→S4 gap=2 ok     2→3: P2→P3 S2→S4 gap=2 ok
        //   3→4: P3→P0 S4→S6 gap=2 ok     3→5: P3→P1 S4→S6 gap=2 ok
        //   4→6: P0→P2 S6→S8 gap=2 ok     5→6: P1→P2 S6→S8 gap=2 ok
        //   6→7: P2→P3 S8→S10 gap=2 ok
        schedule.SetAssignedProcessors({0, 1, 2, 3, 0, 1, 2, 3});
        schedule.SetAssignedSupersteps({0, 2, 2, 4, 6, 6, 8, 10});
        schedule.UpdateNumberOfSupersteps();

        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": initial precedence violated");
        VerifyStalenessConstraints(schedule);

        MaxBspImprover<Policy> kl(42);
        auto status = kl.ImproveSchedule(schedule);

        BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
                            name + ": unexpected status");
        BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": precedence violated");
        VerifyStalenessConstraints(schedule);

        BOOST_TEST_MESSAGE(name << ": completed, steps=" << schedule.NumberOfSupersteps());
    };

    RunForPolicy(EagerCommCostPolicy{}, "EagerSendCosts");
    RunForPolicy(LazyCommCostPolicy{}, "LazySendCosts");
    RunForPolicy(BufferedCommCostPolicy{}, "BufferedSendCosts");
}

// ============================================================================
// TEST 18: Cost monotonicity — verify ImproveSchedule doesn't increase cost
//
// Computes MaxBSP cost before and after improvement using the KL cost
// function's own ComputeScheduleCost, and checks the result is non-increasing.
// ============================================================================

BOOST_AUTO_TEST_CASE(CostMonotonicity) {
    Graph dag;
    // 12-node layered graph
    for (int i = 0; i < 12; ++i) {
        dag.AddVertex(3 + (i % 4), 4 + (i % 3), 1);
    }
    // Layer 0 (0-2) → Layer 1 (3-5) → Layer 2 (6-8) → Layer 3 (9-11)
    for (int layer = 0; layer < 3; ++layer) {
        for (int s = 0; s < 3; ++s) {
            for (int d = 0; d < 3; ++d) {
                if ((s + d) % 2 == 0) {    // sparse connectivity
                    dag.AddEdge(layer * 3 + s, (layer + 1) * 3 + d, 1);
                }
            }
        }
    }

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(4);

    BspInstance<Graph> instance(dag, arch);

    auto RunForPolicy = [&](auto policyTag, const std::string &name) {
        using Policy = decltype(policyTag);
        using CommCostT = MaxCommCostF<Policy>;
        using TestT = KlImproverTest<Graph, CommCostT>;

        MaxBspSchedule<Graph> schedule(instance);
        std::vector<unsigned> procs(12), steps(12);
        for (int i = 0; i < 12; ++i) {
            procs[i] = i % 3;
            steps[i] = (i / 3) * 2;
        }
        schedule.SetAssignedProcessors(procs);
        schedule.SetAssignedSupersteps(steps);
        schedule.UpdateNumberOfSupersteps();

        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
        VerifyStalenessConstraints(schedule);

        // Compute initial cost
        TestT kl;
        kl.SetupSchedule(schedule);
        CostT initialCost = kl.GetCommCostF().ComputeScheduleCostTest();

        // Run full improvement
        MaxBspImprover<Policy> klFull(42);
        auto status = klFull.ImproveSchedule(schedule);

        BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
        VerifyStalenessConstraints(schedule);

        // Compute final cost via fresh KL datastructures
        TestT klAfter;
        klAfter.SetupSchedule(schedule);
        CostT finalCost = klAfter.GetCommCostF().ComputeScheduleCostTest();

        BOOST_CHECK_MESSAGE(
            finalCost <= initialCost + 1e-6,
            name + ": cost increased! initial=" + std::to_string(initialCost) + " final=" + std::to_string(finalCost));

        BOOST_TEST_MESSAGE(name << ": initial=" << initialCost << " final=" << finalCost
                                << " steps=" << schedule.NumberOfSupersteps());
    };

    RunForPolicy(EagerCommCostPolicy{}, "EagerMonotone");
    RunForPolicy(LazyCommCostPolicy{}, "LazyMonotone");
    RunForPolicy(BufferedCommCostPolicy{}, "BufferedMonotone");
}

// ============================================================================
// SUITE: Large end-to-end tests — KlMaxBspCommImprover on real graphs

// Mirrors Suite 6 from kl_bsp_cost_policies.cpp but for the MaxBSP cost.
// Uses GreedyVarianceSspScheduler to produce initial MaxBspSchedule
// (staleness=2) directly, then runs the MaxBSP KL improver.
// ============================================================================

// BOOST_AUTO_TEST_CASE(kl_max_bsp_comm_large_test_graphs_eager) {
//     std::vector<std::string> filenames_graph = LargeSpaaGraphs();
//     using graph = ComputationalDagEdgeIdxVectorImplDefIntT;

//     std::filesystem::path cwd = std::filesystem::current_path();
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//     }

//     for (auto &filename_graph : filenames_graph) {
//         GreedyVarianceSspScheduler<graph> test_scheduler;
//         BspInstance<graph> instance;
//         bool status_graph
//             = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / filename_graph).string(), instance.GetComputationalDag());

//         instance.GetArchitecture().SetSynchronisationCosts(500);
//         instance.GetArchitecture().SetCommunicationCosts(5);
//         instance.GetArchitecture().SetNumberOfProcessors(4);

//         std::vector<std::vector<int>> send_cost = {
//             {0, 1, 4, 4},
//             {1, 0, 4, 4},
//             {4, 4, 0, 1},
//             {4, 4, 1, 0}
//         };
//         instance.GetArchitecture().SetSendCosts(send_cost);

//         if (!status_graph) {
//             std::cout << "Reading files failed: " << filename_graph << std::endl;
//             BOOST_CHECK(false);
//             continue;
//         }

//         AddMemWeights(instance.GetComputationalDag());

//         // Create initial MaxBspSchedule via greedy SSP scheduler (staleness=2)
//         MaxBspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.ComputeSchedule(schedule);
//         schedule.UpdateNumberOfSupersteps();

//         BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
//         VerifyStalenessConstraints(schedule);

//         std::cout << "[MaxBSP Eager] " << filename_graph << ": initial steps=" << schedule.NumberOfSupersteps()
//                   << ", cost=" << schedule.ComputeCosts() << std::endl;

//         KlMaxBspCommImprover<graph> kl;

//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl.ImproveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();

//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "[MaxBSP Eager] " << filename_graph << ": finished in " << duration
//                   << "s, steps=" << schedule.NumberOfSupersteps() << ", cost=" << schedule.ComputeCosts() << std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), "Precedence violated: " + filename_graph);
//         VerifyStalenessConstraints(schedule);
//     }
// }

// BOOST_AUTO_TEST_CASE(kl_max_bsp_comm_large_test_graphs_lazy) {
//     std::vector<std::string> filenames_graph = LargeSpaaGraphs();
//     using graph = ComputationalDagEdgeIdxVectorImplDefIntT;

//     std::filesystem::path cwd = std::filesystem::current_path();
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//     }

//     for (auto &filename_graph : filenames_graph) {
//         GreedyVarianceSspScheduler<graph> test_scheduler;
//         BspInstance<graph> instance;
//         bool status_graph
//             = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / filename_graph).string(), instance.GetComputationalDag());

//         instance.GetArchitecture().SetSynchronisationCosts(500);
//         instance.GetArchitecture().SetCommunicationCosts(5);
//         instance.GetArchitecture().SetNumberOfProcessors(4);

//         if (!status_graph) {
//             std::cout << "Reading files failed: " << filename_graph << std::endl;
//             BOOST_CHECK(false);
//             continue;
//         }

//         AddMemWeights(instance.GetComputationalDag());

//         MaxBspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.ComputeSchedule(schedule);
//         schedule.UpdateNumberOfSupersteps();

//         BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
//         VerifyStalenessConstraints(schedule);

//         std::cout << "[MaxBSP Lazy] " << filename_graph << ": initial steps=" << schedule.NumberOfSupersteps()
//                   << ", cost=" << schedule.ComputeCosts() << std::endl;

//         KlMaxBspCommImprover<graph, NoLocalSearchMemoryConstraint, LazyCommCostPolicy> kl;

//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl.ImproveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();

//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "[MaxBSP Lazy] " << filename_graph << ": finished in " << duration
//                   << "s, steps=" << schedule.NumberOfSupersteps() << ", cost=" << schedule.ComputeCosts() << std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), "Precedence violated: " + filename_graph);
//         VerifyStalenessConstraints(schedule);
//     }
// }

// BOOST_AUTO_TEST_CASE(kl_max_bsp_comm_large_test_graphs_buffered) {
//     std::vector<std::string> filenames_graph = LargeSpaaGraphs();
//     using graph = ComputationalDagEdgeIdxVectorImplDefIntT;

//     std::filesystem::path cwd = std::filesystem::current_path();
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//     }

//     for (auto &filename_graph : filenames_graph) {
//         GreedyVarianceSspScheduler<graph> test_scheduler;
//         BspInstance<graph> instance;
//         bool status_graph
//             = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / filename_graph).string(), instance.GetComputationalDag());

//         instance.GetArchitecture().SetSynchronisationCosts(500);
//         instance.GetArchitecture().SetCommunicationCosts(5);
//         instance.GetArchitecture().SetNumberOfProcessors(4);

//         if (!status_graph) {
//             std::cout << "Reading files failed: " << filename_graph << std::endl;
//             BOOST_CHECK(false);
//             continue;
//         }

//         AddMemWeights(instance.GetComputationalDag());

//         MaxBspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.ComputeSchedule(schedule);
//         schedule.UpdateNumberOfSupersteps();

//         BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
//         VerifyStalenessConstraints(schedule);

//         std::cout << "[MaxBSP Buffered] " << filename_graph << ": initial steps=" << schedule.NumberOfSupersteps()
//                   << ", cost=" << schedule.ComputeCosts() << std::endl;

//         KlMaxBspCommImprover<graph, NoLocalSearchMemoryConstraint, BufferedCommCostPolicy> kl;

//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl.ImproveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();

//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "[MaxBSP Buffered] " << filename_graph << ": finished in " << duration
//                   << "s, steps=" << schedule.NumberOfSupersteps() << ", cost=" << schedule.ComputeCosts() << std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), "Precedence violated: " + filename_graph);
//         VerifyStalenessConstraints(schedule);
//     }
// }

// // ============================================================================
// // TEST: Large graph with window size 2
// //
// // Wider search window exercises more candidate placements per node,
// // stressing the coupled work+comm evaluation across a larger step range.
// // ============================================================================

// BOOST_AUTO_TEST_CASE(kl_max_bsp_comm_large_test_graphs_window2) {
//     std::vector<std::string> filenames_graph = LargeSpaaGraphs();
//     using graph = ComputationalDagEdgeIdxVectorImplDefIntT;

//     std::filesystem::path cwd = std::filesystem::current_path();
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//     }

//     for (auto &filename_graph : filenames_graph) {
//         GreedyVarianceSspScheduler<graph> test_scheduler;
//         BspInstance<graph> instance;
//         bool status_graph
//             = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / filename_graph).string(), instance.GetComputationalDag());

//         instance.GetArchitecture().SetSynchronisationCosts(500);
//         instance.GetArchitecture().SetCommunicationCosts(5);
//         instance.GetArchitecture().SetNumberOfProcessors(4);

//         std::vector<std::vector<int>> send_cost = {
//             {0, 1, 4, 4},
//             {1, 0, 4, 4},
//             {4, 4, 0, 1},
//             {4, 4, 1, 0}
//         };
//         instance.GetArchitecture().SetSendCosts(send_cost);

//         if (!status_graph) {
//             std::cout << "Reading files failed: " << filename_graph << std::endl;
//             BOOST_CHECK(false);
//             continue;
//         }

//         AddMemWeights(instance.GetComputationalDag());

//         MaxBspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.ComputeSchedule(schedule);
//         schedule.UpdateNumberOfSupersteps();

//         BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
//         VerifyStalenessConstraints(schedule);

//         std::cout << "[MaxBSP W2] " << filename_graph << ": initial steps=" << schedule.NumberOfSupersteps()
//                   << ", cost=" << schedule.ComputeCosts() << std::endl;

//         KlMaxBspCommImprover<graph, NoLocalSearchMemoryConstraint, EagerCommCostPolicy, 2> kl;

//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl.ImproveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();

//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "[MaxBSP W2] " << filename_graph << ": finished in " << duration
//                   << "s, steps=" << schedule.NumberOfSupersteps() << ", cost=" << schedule.ComputeCosts() << std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), "Precedence violated: " + filename_graph);
//         VerifyStalenessConstraints(schedule);
//     }
// }

// // ============================================================================
// // SUITE: Multi-threaded KlSyncParallelImprover tests

// // Verifies the synchronized parallel wrapper produces valid results:
// //   - Precedence and staleness constraints are respected
// //   - Cost does not regress (the regression guard works)
// //   - Compatible with all comm-cost policies

// // NOTE: MaxBSP/BSP cost functions are NOT thread-safe for true
// // multi-threaded sync parallel use (shared commDs_ data races).
// // These tests use SetMaxNumThreads(2) which may race on BSP/MaxBSP.
// // The regression guard + constraint checks catch incorrect results.
// // For production use, BSP/MaxBSP should use numThreads=1 or the
// // async parallel improver.
// // ============================================================================

// // ============================================================================
// // TEST MT-1: MT ImproveSchedule on SmallFanGraph — all policies
// // ============================================================================

// BOOST_AUTO_TEST_CASE(MtImproveScheduleSmallFan) {
//     auto RunForPolicy = [](auto policyTag, const std::string &name) {
//         using Policy = decltype(policyTag);

//         SmallFanGraph g;
//         auto &schedule = g.Build();
//         const auto initialCost = schedule.ComputeCosts();

//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
//         VerifyStalenessConstraints(schedule);

//         MaxBspImproverMt<Policy> kl(42);
//         kl.SetMaxNumThreads(2);
//         auto status = kl.ImproveSchedule(schedule);

//         BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
//                             name + ": unexpected return status");
//         BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": precedence violated");
//         VerifyStalenessConstraints(schedule);

//         const auto finalCost = schedule.ComputeCosts();
//         BOOST_CHECK_MESSAGE(finalCost <= initialCost,
//                             name + ": cost regressed from " + std::to_string(initialCost) + " to " + std::to_string(finalCost));

//         BOOST_TEST_MESSAGE(name << ": steps=" << schedule.NumberOfSupersteps() << " cost=" << finalCost);
//     };

//     RunForPolicy(EagerCommCostPolicy{}, "MtEagerFan");
//     RunForPolicy(LazyCommCostPolicy{}, "MtLazyFan");
//     RunForPolicy(BufferedCommCostPolicy{}, "MtBufferedFan");
// }

// // ============================================================================
// // TEST MT-2: MT ImproveSchedule on EightNodeGraph — all policies
// // ============================================================================

// BOOST_AUTO_TEST_CASE(MtImproveScheduleEightNode) {
//     auto RunForPolicy = [](auto policyTag, const std::string &name) {
//         using Policy = decltype(policyTag);

//         EightNodeGraph g;
//         auto &schedule = g.Build();
//         const auto initialCost = schedule.ComputeCosts();

//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
//         VerifyStalenessConstraints(schedule);

//         MaxBspImproverMt<Policy> kl(42);
//         kl.SetMaxNumThreads(2);
//         auto status = kl.ImproveSchedule(schedule);

//         BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
//                             name + ": unexpected return status");
//         BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), name + ": precedence violated");
//         VerifyStalenessConstraints(schedule);

//         const auto finalCost = schedule.ComputeCosts();
//         BOOST_CHECK_MESSAGE(finalCost <= initialCost,
//                             name + ": cost regressed from " + std::to_string(initialCost) + " to " + std::to_string(finalCost));

//         BOOST_TEST_MESSAGE(name << ": steps=" << schedule.NumberOfSupersteps() << " cost=" << finalCost);
//     };

//     RunForPolicy(EagerCommCostPolicy{}, "MtEager8");
//     RunForPolicy(LazyCommCostPolicy{}, "MtLazy8");
//     RunForPolicy(BufferedCommCostPolicy{}, "MtBuffered8");
// }

// // ============================================================================
// // TEST MT-3: MT ImproveSchedule with 3 procs and non-uniform send costs
// // ============================================================================

// BOOST_AUTO_TEST_CASE(MtImproveScheduleThreeProcs) {
//     Graph dag;
//     dag.AddVertex(3, 1, 5);    // 0
//     dag.AddVertex(4, 1, 3);    // 1
//     dag.AddVertex(2, 1, 4);    // 2
//     dag.AddVertex(5, 1, 6);    // 3
//     dag.AddVertex(3, 1, 2);    // 4
//     dag.AddVertex(6, 1, 3);    // 5
//     dag.AddVertex(4, 1, 5);    // 6

//     dag.AddEdge(0, 1, 1);
//     dag.AddEdge(0, 2, 1);
//     dag.AddEdge(1, 3, 1);
//     dag.AddEdge(2, 3, 1);
//     dag.AddEdge(3, 4, 1);
//     dag.AddEdge(3, 5, 1);
//     dag.AddEdge(4, 6, 1);
//     dag.AddEdge(5, 6, 1);

//     BspArchitecture<Graph> arch;
//     arch.SetNumberOfProcessors(3);
//     arch.SetCommunicationCosts(2);
//     arch.SetSynchronisationCosts(4);

//     std::vector<std::vector<int>> sendCosts = {
//         {0, 1, 3},
//         {1, 0, 2},
//         {3, 2, 0}
//     };
//     arch.SetSendCosts(sendCosts);

//     BspInstance<Graph> instance(dag, arch);
//     MaxBspSchedule<Graph> schedule(instance);

//     schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2, 0});
//     schedule.SetAssignedSupersteps({0, 2, 2, 4, 6, 6, 8});
//     schedule.UpdateNumberOfSupersteps();

//     BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
//     VerifyStalenessConstraints(schedule);

//     auto RunForPolicy = [&](auto policyTag, const std::string &name) {
//         using Policy = decltype(policyTag);

//         MaxBspSchedule<Graph> sched(schedule);
//         const auto initialCost = sched.ComputeCosts();

//         MaxBspImproverMt<Policy> kl(42);
//         kl.SetMaxNumThreads(2);
//         auto status = kl.ImproveSchedule(sched);

//         BOOST_CHECK_MESSAGE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND,
//                             name + ": unexpected status");
//         BOOST_CHECK_MESSAGE(sched.SatisfiesPrecedenceConstraints(), name + ": precedence violated");
//         VerifyStalenessConstraints(sched);

//         const auto finalCost = sched.ComputeCosts();
//         BOOST_CHECK_MESSAGE(finalCost <= initialCost, name + ": cost regressed");

//         BOOST_TEST_MESSAGE(name << ": steps=" << sched.NumberOfSupersteps() << " cost=" << finalCost);
//     };

//     RunForPolicy(EagerCommCostPolicy{}, "MtEager3P");
//     RunForPolicy(LazyCommCostPolicy{}, "MtLazy3P");
//     RunForPolicy(BufferedCommCostPolicy{}, "MtBuffered3P");
// }

// // ============================================================================
// // TEST MT-4: MT cost monotonicity on 12-node layered graph
// //
// // Same graph as the single-threaded CostMonotonicity test.
// // Verifies the MT regression guard prevents cost from increasing.
// // ============================================================================

// BOOST_AUTO_TEST_CASE(MtCostMonotonicity) {
//     Graph dag;
//     for (int i = 0; i < 12; ++i) {
//         dag.AddVertex(3 + (i % 4), 4 + (i % 3), 1);
//     }
//     for (int layer = 0; layer < 3; ++layer) {
//         for (int s = 0; s < 3; ++s) {
//             for (int d = 0; d < 3; ++d) {
//                 if ((s + d) % 2 == 0) {
//                     dag.AddEdge(layer * 3 + s, (layer + 1) * 3 + d, 1);
//                 }
//             }
//         }
//     }

//     BspArchitecture<Graph> arch;
//     arch.SetNumberOfProcessors(3);
//     arch.SetCommunicationCosts(2);
//     arch.SetSynchronisationCosts(4);

//     BspInstance<Graph> instance(dag, arch);

//     auto RunForPolicy = [&](auto policyTag, const std::string &name) {
//         using Policy = decltype(policyTag);

//         MaxBspSchedule<Graph> schedule(instance);
//         std::vector<unsigned> procs(12), steps(12);
//         for (int i = 0; i < 12; ++i) {
//             procs[i] = i % 3;
//             steps[i] = (i / 3) * 2;
//         }
//         schedule.SetAssignedProcessors(procs);
//         schedule.SetAssignedSupersteps(steps);
//         schedule.UpdateNumberOfSupersteps();

//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
//         VerifyStalenessConstraints(schedule);

//         const auto initialCost = schedule.ComputeCosts();

//         MaxBspImproverMt<Policy> kl(42);
//         kl.SetMaxNumThreads(3);
//         auto status = kl.ImproveSchedule(schedule);

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
//         VerifyStalenessConstraints(schedule);

//         const auto finalCost = schedule.ComputeCosts();
//         BOOST_CHECK_MESSAGE(
//             finalCost <= initialCost,
//             name + ": cost increased! initial=" + std::to_string(initialCost) + " final=" + std::to_string(finalCost));

//         BOOST_TEST_MESSAGE(name << ": initial=" << initialCost << " final=" << finalCost
//                                 << " steps=" << schedule.NumberOfSupersteps());
//     };

//     RunForPolicy(EagerCommCostPolicy{}, "MtEagerMonotone");
//     RunForPolicy(LazyCommCostPolicy{}, "MtLazyMonotone");
//     RunForPolicy(BufferedCommCostPolicy{}, "MtBufferedMonotone");
// }

// // ============================================================================
// // TEST MT-5: MT on large SPAA graphs (Eager policy)
// // ============================================================================

// BOOST_AUTO_TEST_CASE(kl_max_bsp_comm_large_test_graphs_mt_eager) {
//     std::vector<std::string> filenames_graph = LargeSpaaGraphs();
//     using graph = ComputationalDagEdgeIdxVectorImplDefIntT;

//     std::filesystem::path cwd = std::filesystem::current_path();
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//     }

//     for (auto &filename_graph : filenames_graph) {
//         GreedyVarianceSspScheduler<graph> test_scheduler;
//         BspInstance<graph> instance;
//         bool status_graph
//             = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / filename_graph).string(), instance.GetComputationalDag());

//         instance.GetArchitecture().SetSynchronisationCosts(500);
//         instance.GetArchitecture().SetCommunicationCosts(5);
//         instance.GetArchitecture().SetNumberOfProcessors(4);

//         std::vector<std::vector<int>> send_cost = {
//             {0, 1, 4, 4},
//             {1, 0, 4, 4},
//             {4, 4, 0, 1},
//             {4, 4, 1, 0}
//         };
//         instance.GetArchitecture().SetSendCosts(send_cost);

//         if (!status_graph) {
//             std::cout << "Reading files failed: " << filename_graph << std::endl;
//             BOOST_CHECK(false);
//             continue;
//         }

//         AddMemWeights(instance.GetComputationalDag());

//         MaxBspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.ComputeSchedule(schedule);
//         schedule.UpdateNumberOfSupersteps();

//         BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
//         VerifyStalenessConstraints(schedule);

//         const auto initialCost = schedule.ComputeCosts();
//         std::cout << "[MaxBSP MT-Eager] " << filename_graph << ": initial steps=" << schedule.NumberOfSupersteps()
//                   << ", cost=" << initialCost << std::endl;

//         using MtImprover
//             = KlSyncParallelImprover<graph,
//                                      KlMaxBspCommCostFunction<graph, double, NoLocalSearchMemoryConstraint,
//                                      EagerCommCostPolicy, 1>, NoLocalSearchMemoryConstraint, 1, double>;
//         MtImprover kl;
//         kl.SetMaxNumThreads(4);

//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl.ImproveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();

//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         const auto finalCost = schedule.ComputeCosts();
//         std::cout << "[MaxBSP MT-Eager] " << filename_graph << ": finished in " << duration
//                   << "s, steps=" << schedule.NumberOfSupersteps() << ", cost=" << finalCost << std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_MESSAGE(schedule.SatisfiesPrecedenceConstraints(), "Precedence violated: " + filename_graph);
//         VerifyStalenessConstraints(schedule);
//         BOOST_CHECK_MESSAGE(
//             finalCost <= initialCost,
//             "MT cost regressed for " + filename_graph + ": " + std::to_string(initialCost) + " -> " + std::to_string(finalCost));
//     }
// }
