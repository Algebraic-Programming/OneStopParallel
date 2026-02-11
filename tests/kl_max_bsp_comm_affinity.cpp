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

/**
 * @file kl_max_bsp_comm_affinity_test.cpp
 * @brief Direct ComputeCommAffinity tests for KlMaxBspCommCostFunction under
 *        Eager, Lazy, and Buffered communication policies.
 *
 * Each test calls ComputeCommAffinity with penalty=0, reward=0 (isolating the
 * comm-delta logic from the staleness component) and verifies every entry of
 * the affinity table against a brute-force recomputation of the MaxComm sum.
 *
 * The brute-force helper computes MaxComm from scratch for each hypothetical
 * candidate placement and compares the delta with what the affinity predicted.
 *
 * NOTE: These tests require the CalculateStepCostChange bug fix (stale
 * second_max fallback). Without the fix, tests with ≥3 processors will fail
 * because CalculateStepCostChange uses second_max from the unmodified state,
 * which can be stale when multiple (proc, send/recv) values at second_max
 * are also dirty and reduced at the same step.
 *
 * All suites are instantiated under all three comm policies (Eager, Lazy,
 * Buffered) via INSTANTIATE_ALL. The brute-force oracle uses policy-aware
 * MaxCommDatastructure, so it computes correct expected values for each
 * policy. If ComputeCommAffinity's delta placement does not match the
 * policy's send/recv step attribution, the tests will fail.
 *
 * Structure:
 *   Suite 1 – ComputeCommAffinity brute-force verification (all policies)
 *   Suite 2 – Staleness penalty/reward tests (all policies)
 *   Suite 3 – Smoke tests (all policies, redundant but kept)
 */

#define BOOST_TEST_MODULE kl_max_bsp_comm_affinity
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/comm_cost_policies.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_max_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/max_comm_datastructure.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_active_schedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_test.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;
using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using KlActiveScheduleT = KlActiveSchedule<Graph, double, NoLocalSearchMemoryConstraint>;
using KlMove = KlMoveStruct<double, Graph::VertexIdx>;

// ============================================================================
// Policy-parameterised type aliases
// ============================================================================

template <typename Policy>
using MaxBspCostFnT = KlMaxBspCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, Policy>;

template <typename Policy>
using KlTestT = KlImproverTest<Graph, MaxBspCostFnT<Policy>>;

// ============================================================================
// INSTANTIATE macros
// ============================================================================

#define INSTANTIATE_ALL(FuncName)                                                     \
    BOOST_AUTO_TEST_CASE(FuncName##_Eager) { FuncName<EagerCommCostPolicy>(); }       \
    BOOST_AUTO_TEST_CASE(FuncName##_Lazy) { FuncName<LazyCommCostPolicy>(); }         \
    BOOST_AUTO_TEST_CASE(FuncName##_Buffered) { FuncName<BufferedCommCostPolicy>(); }

// ============================================================================
// Policy name helper (for diagnostic messages)
// ============================================================================

template <typename P>
const char *PolicyName();

template <>
const char *PolicyName<EagerCommCostPolicy>() {
    return "Eager";
}

template <>
const char *PolicyName<LazyCommCostPolicy>() {
    return "Lazy";
}

template <>
const char *PolicyName<BufferedCommCostPolicy>() {
    return "Buffered";
}

// ============================================================================
// Brute-force MaxComm sum computation
//
// Given a node assignment (procs, steps), computes:
//   Σ_{s=0}^{numSteps-1} MaxComm(s)
//
// This uses a fresh MaxCommDatastructure with the specified policy.
// ============================================================================

template <typename Policy>
static double ComputeMaxCommSum(const BspInstance<Graph> &inst,
                                const std::vector<unsigned> &procs,
                                const std::vector<unsigned> &steps,
                                unsigned fixedNumSteps) {
    BspSchedule<Graph> sched(inst);
    sched.SetAssignedProcessors(procs);
    sched.SetAssignedSupersteps(steps);
    sched.UpdateNumberOfSupersteps();

    KlActiveScheduleT klSched;
    klSched.Initialize(sched);

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, Policy> ds;
    ds.Initialize(klSched);
    unsigned maxStep = sched.NumberOfSupersteps();
    if (maxStep > 0) {
        ds.ComputeCommDatastructures(0, maxStep - 1);
    }

    double sum = 0;
    for (unsigned s = 0; s < std::min(maxStep, fixedNumSteps); ++s) {
        sum += static_cast<double>(ds.StepMaxComm(s));
    }
    return sum;
}

// ============================================================================
// VerifyAffinityBruteForce
//
// For a given node, calls ComputeCommAffinity (with penalty=0, reward=0 so
// only the comm-delta component is active) and checks every affinity entry
// against a brute-force recomputation.
//
// Expected value for affinity[p][sIdx]:
//   (maxCommSum_after_move - maxCommSum_before_move) * g
//
// Returns true if all entries match within tolerance.
// ============================================================================

template <typename Policy>
static bool VerifyAffinityBruteForce(KlTestT<Policy> &kl, unsigned node, const std::string &context) {
    auto &costF = kl.GetCommCostF();
    auto *activeSched = costF.active_schedule;
    const auto *inst = costF.instance;
    const auto &dag = *costF.graph;
    const unsigned numProcs = inst->NumberOfProcessors();
    const unsigned numSteps = activeSched->NumSteps();
    const unsigned nodeStep = activeSched->AssignedSuperstep(node);

    constexpr unsigned WS = 1;    // WindowSize (template default)
    constexpr unsigned WR = 3;    // WindowRange = 2*WS + 1

    unsigned startStep = nodeStep > WS ? nodeStep - WS : 0;
    unsigned endStep = std::min(nodeStep + WS, numSteps - 1);
    unsigned startIdx = costF.StartIdx(nodeStep, startStep);
    unsigned endIdx = costF.EndIdx(nodeStep, endStep);

    // Ensure comm datastructures are computed
    costF.ComputeSendReceiveDatastructures();

    // Create affinity table: [proc][window_idx], zero-initialised
    std::vector<std::vector<double>> affinity(numProcs, std::vector<double>(WR, 0.0));

    // Call ComputeCommAffinity with zero penalty/reward (isolate comm deltas)
    costF.ComputeCommAffinity(node, affinity, 0.0, 0.0, startStep, endStep);

    // Current assignment
    std::vector<unsigned> origProcs, origSteps;
    for (auto v : dag.Vertices()) {
        origProcs.push_back(activeSched->AssignedProcessor(v));
        origSteps.push_back(activeSched->AssignedSuperstep(v));
    }

    double oldCommSum = ComputeMaxCommSum<Policy>(*inst, origProcs, origSteps, numSteps);
    double g = inst->CommunicationCosts();

    bool allMatch = true;
    for (unsigned p = 0; p < numProcs; ++p) {
        for (unsigned sIdx = startIdx; sIdx < endIdx; ++sIdx) {
            unsigned sTo = nodeStep + sIdx - WS;

            // Skip out-of-range (sTo >= numSteps handled by EndIdx already)
            if (sTo >= numSteps) {
                continue;
            }

            // Build modified assignment
            auto newProcs = origProcs;
            auto newSteps = origSteps;
            newProcs[node] = p;
            newSteps[node] = sTo;

            double newCommSum = ComputeMaxCommSum<Policy>(*inst, newProcs, newSteps, numSteps);
            double expected = (newCommSum - oldCommSum) * g;
            double actual = affinity[p][sIdx];

            if (std::abs(expected - actual) > 1e-6) {
                allMatch = false;
                BOOST_TEST_MESSAGE(context << " [" << PolicyName<Policy>() << "]"
                                           << ": node=" << node << " → (P" << p << ",S" << sTo << ")"
                                           << " expected=" << expected << " actual=" << actual << " diff=" << (actual - expected));
            }
        }
    }

    // Also verify self-move has affinity 0
    unsigned nodeProc = activeSched->AssignedProcessor(node);
    unsigned selfIdx = WS;    // nodeStep maps to index WS in the window
    if (selfIdx >= startIdx && selfIdx < endIdx) {
        double selfAffinity = affinity[nodeProc][selfIdx];
        if (std::abs(selfAffinity) > 1e-6) {
            allMatch = false;
            BOOST_TEST_MESSAGE(context << " [" << PolicyName<Policy>() << "]"
                                       << ": self-move affinity != 0: " << selfAffinity);
        }
    }

    return allMatch;
}

// ============================================================================
// Suite 1: ComputeCommAffinity brute-force verification (all policies)
//
// Exact brute-force comparison against fresh MaxComm recomputation for every
// affinity table entry. Requires the CalculateStepCostChange bug fix.
// ============================================================================

BOOST_AUTO_TEST_SUITE(AffinityBruteForce)

// Simple edge v0→v1. Test affinity of the child (v1).
template <typename Policy>
void TestAffinityChild() {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0: CW=5
    dag.AddVertex(10, 3, 1);    // v1: CW=3
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0 → v1@P1,S1. Comm[0] = 5 (v0 sends to P1).
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 1, "AffinityChild"));
}
INSTANTIATE_ALL(TestAffinityChild)

// Simple edge v0→v1. Test affinity of the parent (v0).
template <typename Policy>
void TestAffinityParent() {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0: CW=5
    dag.AddVertex(10, 3, 1);    // v1: CW=3
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 0, "AffinityParent"));
}
INSTANTIATE_ALL(TestAffinityParent)

// Chain: v0→v1→v2. Test affinity of the middle node (has both parents & children).
template <typename Policy>
void TestAffinityMiddle() {
    Graph dag;
    dag.AddVertex(10, 5, 1);     // v0: CW=5
    dag.AddVertex(10, 10, 1);    // v1: CW=10
    dag.AddVertex(10, 3, 1);     // v2: CW=3
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0, v1@P1,S1, v2@P0,S2
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 1, "AffinityMiddle"));
}
INSTANTIATE_ALL(TestAffinityMiddle)

// Fan-out: v0→{v1, v2} on 3 procs. Test affinity of the parent.
template <typename Policy>
void TestAffinityFanOutParent() {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0: CW=5
    dag.AddVertex(8, 1, 1);     // v1
    dag.AddVertex(12, 1, 1);    // v2
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0, v1@P1,S1, v2@P2,S1
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 0, "AffinityFanOutParent"));
}
INSTANTIATE_ALL(TestAffinityFanOutParent)

// Fan-out: v0→{v1, v2} both on P1. Test affinity of one child (v2).
// Moving v2 to P0 should NOT remove comm from v0 (v1 still on P1, lambda=1).
template <typename Policy>
void TestAffinityFanOutChild() {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0: CW=5
    dag.AddVertex(8, 1, 1);     // v1
    dag.AddVertex(12, 1, 1);    // v2
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0, v1@P1,S1, v2@P1,S1. Lambda[v0][P1] = 2.
    schedule.SetAssignedProcessors({0, 1, 1});
    schedule.SetAssignedSupersteps({0, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 2, "AffinityFanOutChild"));
}
INSTANTIATE_ALL(TestAffinityFanOutChild)

// Fan-in: {v0, v1}→v2 on 3 procs. Test affinity of the child.
template <typename Policy>
void TestAffinityFanInChild() {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0: CW=5
    dag.AddVertex(10, 3, 1);    // v1: CW=3
    dag.AddVertex(20, 1, 1);    // v2
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0, v1@P1,S0, v2@P2,S1
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 2, "AffinityFanInChild"));
}
INSTANTIATE_ALL(TestAffinityFanInChild)

// Diamond: v0→{v1, v2}→v3 on 3 procs. Test affinity for interior nodes.
template <typename Policy>
void TestAffinityDiamond() {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(8, 3, 1);     // v1
    dag.AddVertex(12, 4, 1);    // v2
    dag.AddVertex(15, 1, 1);    // v3
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0, v1@P1,S1, v2@P2,S1, v3@P0,S2
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 1, "AffinityDiamond v1"));
    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 2, "AffinityDiamond v2"));
}
INSTANTIATE_ALL(TestAffinityDiamond)

// Butterfly: v0,v1→v2,v3 with all cross-edges. Test v2.
template <typename Policy>
void TestAffinityButterfly() {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(10, 3, 1);    // v1
    dag.AddVertex(10, 1, 1);    // v2
    dag.AddVertex(10, 1, 1);    // v3
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(0, 3, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(1, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0, v1@P1,S0, v2@P1,S1, v3@P0,S1
    schedule.SetAssignedProcessors({0, 1, 1, 0});
    schedule.SetAssignedSupersteps({0, 0, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 2, "AffinityButterfly v2"));
    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 3, "AffinityButterfly v3"));
}
INSTANTIATE_ALL(TestAffinityButterfly)

// Same proc, no comm: affinity should be zero for all same-step candidates.
template <typename Policy>
void TestAffinitySameProcNoComm() {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // Both on P0. No comm at all.
    schedule.SetAssignedProcessors({0, 0});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 0, "SameProcNoComm v0"));
    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 1, "SameProcNoComm v1"));
}
INSTANTIATE_ALL(TestAffinitySameProcNoComm)

// Wider graph: 6 nodes, 3 procs, 3 steps.
template <typename Policy>
void TestAffinityWiderGraph() {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(10, 3, 1);    // v1
    dag.AddVertex(10, 4, 1);    // v2
    dag.AddVertex(10, 2, 1);    // v3
    dag.AddVertex(10, 6, 1);    // v4
    dag.AddVertex(10, 1, 1);    // v5
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(0, 3, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(1, 4, 1);
    dag.AddEdge(2, 5, 1);
    dag.AddEdge(3, 5, 1);
    dag.AddEdge(4, 5, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    // Test all middle-layer nodes (step 1)
    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 2, "WiderGraph v2"));
    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 3, "WiderGraph v3"));
    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 4, "WiderGraph v4"));
}
INSTANTIATE_ALL(TestAffinityWiderGraph)

// g multiplier > 1: affinity should scale with CommunicationCosts.
template <typename Policy>
void TestAffinityWithGMultiplier() {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(3);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 1, "WithGMultiplier"));
}
INSTANTIATE_ALL(TestAffinityWithGMultiplier)

// No edges: affinity should be zero everywhere.
template <typename Policy>
void TestAffinityNoEdges() {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 0, "NoEdges v0"));
    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 1, "NoEdges v1"));
}
INSTANTIATE_ALL(TestAffinityNoEdges)

// 4-step chain with node at step 2 (full window coverage).
template <typename Policy>
void TestAffinityFullWindow() {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(10, 5, 1);    // v1
    dag.AddVertex(10, 5, 1);    // v2
    dag.AddVertex(10, 5, 1);    // v3
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // Alternating procs across 4 steps
    schedule.SetAssignedProcessors({0, 1, 0, 1});
    schedule.SetAssignedSupersteps({0, 1, 2, 3});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);

    // Test v2 at step 2: window covers steps 1, 2, 3 (all 3 entries valid)
    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 2, "FullWindow v2"));

    // Test v1 at step 1: window covers steps 0, 1, 2 (all 3 entries valid)
    BOOST_CHECK(VerifyAffinityBruteForce<Policy>(kl, 1, "FullWindow v1"));
}
INSTANTIATE_ALL(TestAffinityFullWindow)

BOOST_AUTO_TEST_SUITE_END()    // AffinityBruteForce

// ============================================================================
// Suite 2: Staleness penalty/reward
//
// Tests that the staleness component of ComputeCommAffinity has the expected
// relative behaviour: same-proc candidates are unaffected, cross-proc
// candidates at violating positions are affected, and the magnitude scales
// with penalty/reward values.
// ============================================================================

BOOST_AUTO_TEST_SUITE(StalenessTests)

// Same-proc candidates should have zero staleness contribution regardless of
// step, because staleness constraints only apply across processors.
template <typename Policy>
void TestStalenessZeroForSameProc() {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    const unsigned numProcs = 2;

    std::vector<std::vector<double>> withStale(numProcs, std::vector<double>(WR, 0.0));
    std::vector<std::vector<double>> noStale(numProcs, std::vector<double>(WR, 0.0));

    kl.GetCommCostF().ComputeCommAffinity(1, withStale, 100.0, 100.0, 0, 1);
    kl.GetCommCostF().ComputeCommAffinity(1, noStale, 0.0, 0.0, 0, 1);

    // For each window index, the staleness contribution on the PARENT'S proc
    // (P0, same proc as source) should be zero (no staleness on same proc).
    unsigned sourceProc = 0;    // v0 is on P0
    for (unsigned sIdx = 0; sIdx < WR; ++sIdx) {
        double staleContrib = withStale[sourceProc][sIdx] - noStale[sourceProc][sIdx];
        BOOST_CHECK_SMALL(staleContrib, 1e-6);
    }
}
INSTANTIATE_ALL(TestStalenessZeroForSameProc)

// Non-zero penalty/reward should produce non-zero staleness contribution
// on cross-proc candidates.
template <typename Policy>
void TestStalenessNonZeroCrossProc() {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0 → v1@P1,S1. Staleness=1 is exactly satisfied.
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    const unsigned numProcs = 2;

    std::vector<std::vector<double>> withStale(numProcs, std::vector<double>(WR, 0.0));
    std::vector<std::vector<double>> noStale(numProcs, std::vector<double>(WR, 0.0));

    kl.GetCommCostF().ComputeCommAffinity(1, withStale, 100.0, 100.0, 0, 1);
    kl.GetCommCostF().ComputeCommAffinity(1, noStale, 0.0, 0.0, 0, 1);

    // v1's current proc is P1. The staleness contribution for (P1, S0) — moving
    // v1 to S0 while staying on P1 (cross-proc from parent v0@P0) — should be
    // non-zero since this would violate the staleness constraint.
    double staleContribP1_S0 = withStale[1][0] - noStale[1][0];
    BOOST_CHECK_NE(staleContribP1_S0, 0.0);
}
INSTANTIATE_ALL(TestStalenessNonZeroCrossProc)

// Staleness contribution should scale linearly with penalty/reward magnitude.
template <typename Policy>
void TestStalenessScalesWithMagnitude() {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    const unsigned numProcs = 2;

    std::vector<std::vector<double>> aff1(numProcs, std::vector<double>(WR, 0.0));
    std::vector<std::vector<double>> aff2(numProcs, std::vector<double>(WR, 0.0));
    std::vector<std::vector<double>> noStale(numProcs, std::vector<double>(WR, 0.0));

    kl.GetCommCostF().ComputeCommAffinity(1, aff1, 50.0, 50.0, 0, 1);
    kl.GetCommCostF().ComputeCommAffinity(1, aff2, 100.0, 100.0, 0, 1);
    kl.GetCommCostF().ComputeCommAffinity(1, noStale, 0.0, 0.0, 0, 1);

    // Staleness with magnitude 100 should be 2× staleness with magnitude 50.
    double stale1 = aff1[1][0] - noStale[1][0];
    double stale2 = aff2[1][0] - noStale[1][0];

    if (std::abs(stale1) > 1e-6) {
        BOOST_CHECK_CLOSE(stale2 / stale1, 2.0, 1e-5);
    }
}
INSTANTIATE_ALL(TestStalenessScalesWithMagnitude)

// Self-move (node stays at current position) should have zero TOTAL affinity
// (both staleness and comm delta contribute zero).
template <typename Policy>
void TestStalenessZeroAtSelfMove() {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    constexpr unsigned WS = 1;
    std::vector<std::vector<double>> aff(2, std::vector<double>(WR, 0.0));

    kl.GetCommCostF().ComputeCommAffinity(1, aff, 100.0, 100.0, 0, 1);

    // Self-move: v1 at (P1, S1), window index = WS = 1.
    BOOST_CHECK_SMALL(aff[1][WS], 1e-6);
}
INSTANTIATE_ALL(TestStalenessZeroAtSelfMove)

BOOST_AUTO_TEST_SUITE_END()    // StalenessTests

// ============================================================================
// Suite 3: Smoke tests (all three policies)
//
// Basic sanity checks. Now that Suite 1 runs brute-force under all policies,
// these are redundant but kept as lightweight regression guards.
// ============================================================================

BOOST_AUTO_TEST_SUITE(PolicySmoke)

// Compiles and runs. Self-move affinity should be 0 under all policies.
template <typename Policy>
void TestSmokeSelfMoveZero() {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    constexpr unsigned WS = 1;
    std::vector<std::vector<double>> affinity(2, std::vector<double>(WR, 0.0));

    kl.GetCommCostF().ComputeCommAffinity(1, affinity, 0.0, 0.0, 0, 1);

    // Self-move: v1 stays at (P1, S1). Window index = WS = 1.
    double selfAffinity = affinity[1][WS];
    BOOST_CHECK_SMALL(selfAffinity, 1e-6);
}
INSTANTIATE_ALL(TestSmokeSelfMoveZero)

// Runs on a larger graph without crashing.
template <typename Policy>
void TestSmokeLargerGraph() {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(10, 4, 1);
    dag.AddVertex(10, 2, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    std::vector<std::vector<double>> affinity(3, std::vector<double>(WR, 0.0));

    // Just verify it doesn't crash for each node
    for (unsigned v = 0; v < 4; ++v) {
        // Reset affinity
        for (auto &row : affinity) {
            std::fill(row.begin(), row.end(), 0.0);
        }

        unsigned nodeStep = kl.GetCommCostF().active_schedule->AssignedSuperstep(v);
        unsigned startStep = nodeStep > 0 ? nodeStep - 1 : 0;
        unsigned endStep = std::min(nodeStep + 1, kl.GetCommCostF().active_schedule->NumSteps() - 1);

        kl.GetCommCostF().ComputeCommAffinity(v, affinity, 50.0, 50.0, startStep, endStep);

        // Self-move affinity at current (proc, step) should be 0
        unsigned selfProc = kl.GetCommCostF().active_schedule->AssignedProcessor(v);
        constexpr unsigned WS = 1;
        double selfAffinity = affinity[selfProc][WS];
        BOOST_CHECK_SMALL(selfAffinity, 1e-6);
    }
}
INSTANTIATE_ALL(TestSmokeLargerGraph)

// Same proc edges: no comm regardless of policy.
// Affinity should be 0 for candidates that keep node on same proc.
template <typename Policy>
void TestSmokeSameProcNoComm() {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // Both on P0. No comm.
    schedule.SetAssignedProcessors({0, 0});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    constexpr unsigned WS = 1;
    std::vector<std::vector<double>> affinity(2, std::vector<double>(WR, 0.0));

    // v1 at (P0, S1). Moving to (P0, S0) = same proc, no comm either way.
    kl.GetCommCostF().ComputeCommAffinity(1, affinity, 0.0, 0.0, 0, 1);

    // Same proc candidates: affinity should be 0 (no comm change)
    BOOST_CHECK_SMALL(affinity[0][0], 1e-6);     // (P0, S0)
    BOOST_CHECK_SMALL(affinity[0][WS], 1e-6);    // (P0, S1) = self-move
}
INSTANTIATE_ALL(TestSmokeSameProcNoComm)

// Isolated node (no edges). Affinity must be 0 everywhere.
template <typename Policy>
void TestSmokeIsolated() {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0 isolated
    dag.AddVertex(20, 3, 1);    // v1 anchor at S1

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT<Policy> kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    std::vector<std::vector<double>> affinity(2, std::vector<double>(WR, 0.0));

    kl.GetCommCostF().ComputeCommAffinity(0, affinity, 0.0, 0.0, 0, 1);

    // No edges → all entries should be 0
    for (unsigned p = 0; p < 2; ++p) {
        for (unsigned i = 0; i < WR; ++i) {
            BOOST_CHECK_SMALL(affinity[p][i], 1e-6);
        }
    }
}
INSTANTIATE_ALL(TestSmokeIsolated)

BOOST_AUTO_TEST_SUITE_END()    // PolicySmoke
