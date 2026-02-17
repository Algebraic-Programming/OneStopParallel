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
 * @file kl_bsp_comm_affinity_test.cpp
 * @brief Direct ComputeCommAffinity tests for KlBspCommCostFunction.
 *
 * Each test calls ComputeCommAffinity with penalty=0, reward=0 (isolating the
 * comm-delta logic from the staleness component) and verifies every entry of
 * the affinity table against a brute-force recomputation of the MaxComm sum.
 *
 * The brute-force helper builds a fresh MaxCommDatastructure for each
 * hypothetical candidate placement, computes Σ MaxComm(s), and compares
 * (newSum - oldSum) * g against what the affinity table reported.
 *
 * NOTE: These tests require the CalculateStepCostChange bug fix (stale
 * second_max fallback). Without the fix, tests with ≥3 processors will fail.
 *
 * The regular BSP cost function always uses EagerCommCostPolicy (no policy
 * template parameter) and hardcodes staleness to 1.
 *
 * Structure:
 *   Suite 1 – ComputeCommAffinity brute-force verification
 *   Suite 2 – Staleness penalty/reward tests
 */

#define BOOST_TEST_MODULE kl_bsp_comm_affinity
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/comm_cost_policies.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/max_comm_datastructure.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_active_schedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_test.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;
using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using KlActiveScheduleT = KlActiveSchedule<Graph, double, NoLocalSearchMemoryConstraint>;
using KlMove = KlMoveStruct<double, Graph::VertexIdx>;
using BspCostFnT = KlBspCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint>;
using KlTestT = KlImproverTest<Graph, BspCostFnT>;

using BspCostFnLazyT = KlBspCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, LazyCommCostPolicy>;
using KlTestLazyT = KlImproverTest<Graph, BspCostFnLazyT>;

using BspCostFnBufferedT = KlBspCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, BufferedCommCostPolicy>;
using KlTestBufferedT = KlImproverTest<Graph, BspCostFnBufferedT>;

// ============================================================================
// Brute-force MaxComm sum computation
//
// Given a node assignment (procs, steps), computes:
//   Σ_{s=0}^{numSteps-1} MaxComm(s)
//
// Uses a fresh MaxCommDatastructure with default EagerCommCostPolicy.
// ============================================================================

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

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> ds;
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

// Policy-aware brute-force MaxComm sum computation.
// Uses a fresh MaxCommDatastructure with CommPolicyT.
template <typename CommPolicyT>
static double ComputeMaxCommSumPolicy(const BspInstance<Graph> &inst,
                                      const std::vector<unsigned> &procs,
                                      const std::vector<unsigned> &steps,
                                      unsigned fixedNumSteps) {
    BspSchedule<Graph> sched(inst);
    sched.SetAssignedProcessors(procs);
    sched.SetAssignedSupersteps(steps);
    sched.UpdateNumberOfSupersteps();

    KlActiveScheduleT klSched;
    klSched.Initialize(sched);

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, CommPolicyT> ds;
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

static bool VerifyAffinityBruteForce(KlTestT &kl, unsigned node, const std::string &context) {
    auto &costF = kl.GetCommCostF();
    auto *activeSched = costF.activeSchedule_;
    const auto *inst = costF.instance_;
    const auto &dag = *costF.graph_;
    const unsigned numProcs = inst->NumberOfProcessors();
    const unsigned numSteps = activeSched->NumSteps();
    const unsigned nodeStep = activeSched->AssignedSuperstep(node);

    constexpr unsigned WS = 1;    // windowSize (template default)
    constexpr unsigned WR = 3;    // windowRange = 2*WS + 1

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

    double oldCommSum = ComputeMaxCommSum(*inst, origProcs, origSteps, numSteps);
    double g = inst->CommunicationCosts();

    bool allMatch = true;
    for (unsigned p = 0; p < numProcs; ++p) {
        for (unsigned sIdx = startIdx; sIdx < endIdx; ++sIdx) {
            unsigned sTo = nodeStep + sIdx - WS;

            if (sTo >= numSteps) {
                continue;
            }

            // Build modified assignment
            auto newProcs = origProcs;
            auto newSteps = origSteps;
            newProcs[node] = p;
            newSteps[node] = sTo;

            double newCommSum = ComputeMaxCommSum(*inst, newProcs, newSteps, numSteps);
            double expected = (newCommSum - oldCommSum) * g;
            double actual = affinity[p][sIdx];

            if (std::abs(expected - actual) > 1e-6) {
                allMatch = false;
                BOOST_TEST_MESSAGE(context << ": node=" << node << " -> (P" << p << ",S" << sTo << ")"
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
            BOOST_TEST_MESSAGE(context << ": self-move affinity != 0: " << selfAffinity);
        }
    }

    return allMatch;
}

// Policy-aware brute-force verification.
// KlTestType must expose GetCommCostF() for the appropriate policy.
// CommPolicyT drives the fresh MaxComm recomputation for ground truth.
template <typename KlTestType, typename CommPolicyT>
static bool VerifyAffinityBruteForcePolicy(KlTestType &kl, unsigned node, const std::string &context) {
    auto &costF = kl.GetCommCostF();
    auto *activeSched = costF.activeSchedule_;
    const auto *inst = costF.instance_;
    const auto &dag = *costF.graph_;
    const unsigned numProcs = inst->NumberOfProcessors();
    const unsigned numSteps = activeSched->NumSteps();
    const unsigned nodeStep = activeSched->AssignedSuperstep(node);

    constexpr unsigned WS = 1;
    constexpr unsigned WR = 3;

    unsigned startStep = nodeStep > WS ? nodeStep - WS : 0;
    unsigned endStep = std::min(nodeStep + WS, numSteps - 1);
    unsigned startIdx = costF.StartIdx(nodeStep, startStep);
    unsigned endIdx = costF.EndIdx(nodeStep, endStep);

    costF.ComputeSendReceiveDatastructures();

    std::vector<std::vector<double>> affinity(numProcs, std::vector<double>(WR, 0.0));
    costF.ComputeCommAffinity(node, affinity, 0.0, 0.0, startStep, endStep);

    std::vector<unsigned> origProcs, origSteps;
    for (auto v : dag.Vertices()) {
        origProcs.push_back(activeSched->AssignedProcessor(v));
        origSteps.push_back(activeSched->AssignedSuperstep(v));
    }

    double oldCommSum = ComputeMaxCommSumPolicy<CommPolicyT>(*inst, origProcs, origSteps, numSteps);
    double g = inst->CommunicationCosts();

    bool allMatch = true;
    for (unsigned p = 0; p < numProcs; ++p) {
        for (unsigned sIdx = startIdx; sIdx < endIdx; ++sIdx) {
            unsigned sTo = nodeStep + sIdx - WS;

            if (sTo >= numSteps) {
                continue;
            }

            auto newProcs = origProcs;
            auto newSteps = origSteps;
            newProcs[node] = p;
            newSteps[node] = sTo;

            double newCommSum = ComputeMaxCommSumPolicy<CommPolicyT>(*inst, newProcs, newSteps, numSteps);
            double expected = (newCommSum - oldCommSum) * g;
            double actual = affinity[p][sIdx];

            if (std::abs(expected - actual) > 1e-6) {
                allMatch = false;
                BOOST_TEST_MESSAGE(context << ": node=" << node << " -> (P" << p << ",S" << sTo << ")"
                                           << " expected=" << expected << " actual=" << actual << " diff=" << (actual - expected));
            }
        }
    }

    unsigned nodeProc = activeSched->AssignedProcessor(node);
    unsigned selfIdx = WS;
    if (selfIdx >= startIdx && selfIdx < endIdx) {
        double selfAffinity = affinity[nodeProc][selfIdx];
        if (std::abs(selfAffinity) > 1e-6) {
            allMatch = false;
            BOOST_TEST_MESSAGE(context << ": self-move affinity != 0: " << selfAffinity);
        }
    }

    return allMatch;
}

// ============================================================================
// Suite 1: ComputeCommAffinity brute-force verification
//
// Exact brute-force comparison against fresh MaxComm recomputation for every
// affinity table entry. Requires the CalculateStepCostChange bug fix.
// ============================================================================

BOOST_AUTO_TEST_SUITE(AffinityBruteForce)

// ---------------------------------------------------------------------------
// 2-processor topologies (exact even without CalculateStepCostChange fix)
// ---------------------------------------------------------------------------

// Simple edge v0->v1. Test affinity of the child (v1).
BOOST_AUTO_TEST_CASE(AffinityChild) {
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 1, "AffinityChild"));
}

// Simple edge v0->v1. Test affinity of the parent (v0).
BOOST_AUTO_TEST_CASE(AffinityParent) {
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 0, "AffinityParent"));
}

// Chain: v0->v1->v2. Test affinity of the middle node.
BOOST_AUTO_TEST_CASE(AffinityMiddle) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 10, 1);
    dag.AddVertex(10, 3, 1);
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 1, "AffinityMiddle"));
}

// Fan-out: v0->{v1, v2} both on P1. Test affinity of one child (v2).
// Moving v2 to P0 should NOT remove comm from v0 (v1 still on P1, lambda=1).
BOOST_AUTO_TEST_CASE(AffinityFanOutChild) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(8, 1, 1);
    dag.AddVertex(12, 1, 1);
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 2, "AffinityFanOutChild"));
}

// Butterfly: v0,v1->v2,v3 with all cross-edges.
BOOST_AUTO_TEST_CASE(AffinityButterfly) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(10, 1, 1);
    dag.AddVertex(10, 1, 1);
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 2, "AffinityButterfly v2"));
    BOOST_CHECK(VerifyAffinityBruteForce(kl, 3, "AffinityButterfly v3"));
}

// Same proc, no comm: affinity should be zero everywhere.
BOOST_AUTO_TEST_CASE(AffinitySameProcNoComm) {
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
    schedule.SetAssignedProcessors({0, 0});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 0, "SameProcNoComm v0"));
    BOOST_CHECK(VerifyAffinityBruteForce(kl, 1, "SameProcNoComm v1"));
}

// g multiplier > 1: affinity should scale with CommunicationCosts.
BOOST_AUTO_TEST_CASE(AffinityWithGMultiplier) {
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 1, "WithGMultiplier"));
}

// No edges: affinity should be zero everywhere.
BOOST_AUTO_TEST_CASE(AffinityNoEdges) {
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 0, "NoEdges v0"));
    BOOST_CHECK(VerifyAffinityBruteForce(kl, 1, "NoEdges v1"));
}

// 4-step chain with node at step 2 (full window coverage).
BOOST_AUTO_TEST_CASE(AffinityFullWindow) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 5, 1);
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    // v2 at step 2: window covers steps 1, 2, 3
    BOOST_CHECK(VerifyAffinityBruteForce(kl, 2, "FullWindow v2"));
    // v1 at step 1: window covers steps 0, 1, 2
    BOOST_CHECK(VerifyAffinityBruteForce(kl, 1, "FullWindow v1"));
}

// ---------------------------------------------------------------------------
// 3-processor topologies (require CalculateStepCostChange bug fix)
// ---------------------------------------------------------------------------

// Fan-out: v0->{v1, v2} on 3 procs.
BOOST_AUTO_TEST_CASE(AffinityFanOutParent) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(8, 1, 1);
    dag.AddVertex(12, 1, 1);
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 0, "AffinityFanOutParent"));
}

// Fan-in: {v0, v1}->v2 on 3 procs.
BOOST_AUTO_TEST_CASE(AffinityFanInChild) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(20, 1, 1);
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 2, "AffinityFanInChild"));
}

// Diamond: v0->{v1, v2}->v3 on 3 procs.
BOOST_AUTO_TEST_CASE(AffinityDiamond) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(8, 3, 1);
    dag.AddVertex(12, 4, 1);
    dag.AddVertex(15, 1, 1);
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 1, "AffinityDiamond v1"));
    BOOST_CHECK(VerifyAffinityBruteForce(kl, 2, "AffinityDiamond v2"));
}

// Wider graph: 6 nodes, 3 procs, 3 steps.
BOOST_AUTO_TEST_CASE(AffinityWiderGraph) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(10, 4, 1);
    dag.AddVertex(10, 2, 1);
    dag.AddVertex(10, 6, 1);
    dag.AddVertex(10, 1, 1);
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

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 2, "WiderGraph v2"));
    BOOST_CHECK(VerifyAffinityBruteForce(kl, 3, "WiderGraph v3"));
    BOOST_CHECK(VerifyAffinityBruteForce(kl, 4, "WiderGraph v4"));
}

// Large fan-out: v0 with 4 children on 4 different procs (5 procs total).
// Stress-tests the CalculateStepCostChange fix: 5 dirty (proc,type) pairs at step 0.
BOOST_AUTO_TEST_CASE(AffinityLargeFanOut) {
    Graph dag;
    dag.AddVertex(10, 8, 1);    // v0: CW=8
    dag.AddVertex(10, 1, 1);    // v1
    dag.AddVertex(10, 1, 1);    // v2
    dag.AddVertex(10, 1, 1);    // v3
    dag.AddVertex(10, 1, 1);    // v4
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(0, 3, 1);
    dag.AddEdge(0, 4, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(5);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0, v1@P1,S1, v2@P2,S1, v3@P3,S1, v4@P4,S1
    schedule.SetAssignedProcessors({0, 1, 2, 3, 4});
    schedule.SetAssignedSupersteps({0, 1, 1, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 0, "LargeFanOut v0"));
}

// Large fan-in: 4 parents on 4 procs -> v4.
BOOST_AUTO_TEST_CASE(AffinityLargeFanIn) {
    Graph dag;
    dag.AddVertex(10, 3, 1);    // v0: CW=3
    dag.AddVertex(10, 5, 1);    // v1: CW=5
    dag.AddVertex(10, 4, 1);    // v2: CW=4
    dag.AddVertex(10, 2, 1);    // v3: CW=2
    dag.AddVertex(10, 1, 1);    // v4
    dag.AddEdge(0, 4, 1);
    dag.AddEdge(1, 4, 1);
    dag.AddEdge(2, 4, 1);
    dag.AddEdge(3, 4, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(5);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0, v1@P1,S0, v2@P2,S0, v3@P3,S0, v4@P4,S1
    schedule.SetAssignedProcessors({0, 1, 2, 3, 4});
    schedule.SetAssignedSupersteps({0, 0, 0, 0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(VerifyAffinityBruteForce(kl, 4, "LargeFanIn v4"));
}

BOOST_AUTO_TEST_SUITE_END()    // AffinityBruteForce

// ============================================================================
// Suite 1b: ComputeCommAffinity brute-force verification for Lazy/Buffered policies
//
// Same topologies as Suite 1 but instantiated with LazyCommCostPolicy and
// BufferedCommCostPolicy. These tests validate that ComputeCommAffinity
// correctly predicts the actual comm cost change for all comm policies.
//
// Additional "MultiChildren" cases stress the min-shift logic that is unique
// to Lazy/Buffered (when a parent has multiple children on the same proc at
// different steps, the comm is attributed at min(child_steps)-1 and moves when
// that minimum changes).
// ============================================================================

BOOST_AUTO_TEST_SUITE(AffinityBruteForceLazyBuffered)

// ---------------------------------------------------------------------------
// Lazy policy tests
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(Lazy_AffinityChild) {
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

    KlTestLazyT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 1, "Lazy AffinityChild")));
}

BOOST_AUTO_TEST_CASE(Lazy_AffinityParent) {
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

    KlTestLazyT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 0, "Lazy AffinityParent")));
}

// Two children on same proc at different steps: min-shift logic.
// v0@P0,S0 -> {v1@P1,S1, v2@P1,S2}
// Lazy comm for v0->P1 is at min(1,2)-1 = S0.
// Moving v1 to S2 shifts min from 1 to 2, comm moves to S1.
BOOST_AUTO_TEST_CASE(Lazy_MultiChildrenSameProc) {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(10, 3, 1);    // v1
    dag.AddVertex(10, 2, 1);    // v2
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0  v1@P1,S1  v2@P1,S2
    schedule.SetAssignedProcessors({0, 1, 1});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestLazyT kl;
    kl.SetupSchedule(schedule);

    // Moving v1 (the current-min child) tests min-shift
    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 1, "Lazy MultiChildrenSameProc v1")));
    // Moving v2 (a non-min child) should not change the min
    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 2, "Lazy MultiChildrenSameProc v2")));
}

// Fan-in: {v0@P0, v1@P1} -> v2@P2. Moving v2.
BOOST_AUTO_TEST_CASE(Lazy_FanInChild) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(20, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestLazyT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 2, "Lazy FanInChild")));
}

// Diamond: v0->{v1,v2}->v3 on 3 procs.
BOOST_AUTO_TEST_CASE(Lazy_Diamond) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(8, 3, 1);
    dag.AddVertex(12, 4, 1);
    dag.AddVertex(15, 1, 1);
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

    KlTestLazyT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 1, "Lazy Diamond v1")));
    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 2, "Lazy Diamond v2")));
}

// Wider: 6 nodes, 3 procs, 3 steps.
BOOST_AUTO_TEST_CASE(Lazy_WiderGraph) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(10, 4, 1);
    dag.AddVertex(10, 2, 1);
    dag.AddVertex(10, 6, 1);
    dag.AddVertex(10, 1, 1);
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

    KlTestLazyT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 2, "Lazy WiderGraph v2")));
    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 3, "Lazy WiderGraph v3")));
    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 4, "Lazy WiderGraph v4")));
}

// Large fan-in: 4 parents -> v4. Tests multi-parent Lazy comm attribution.
BOOST_AUTO_TEST_CASE(Lazy_LargeFanIn) {
    Graph dag;
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 4, 1);
    dag.AddVertex(10, 2, 1);
    dag.AddVertex(10, 1, 1);
    dag.AddEdge(0, 4, 1);
    dag.AddEdge(1, 4, 1);
    dag.AddEdge(2, 4, 1);
    dag.AddEdge(3, 4, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(5);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 3, 4});
    schedule.SetAssignedSupersteps({0, 0, 0, 0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestLazyT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestLazyT, LazyCommCostPolicy>(kl, 4, "Lazy LargeFanIn v4")));
}

// ---------------------------------------------------------------------------
// Buffered policy tests
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(Buffered_AffinityChild) {
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

    KlTestBufferedT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 1, "Buffered AffinityChild")));
}

BOOST_AUTO_TEST_CASE(Buffered_AffinityParent) {
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

    KlTestBufferedT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 0, "Buffered AffinityParent")));
}

// Two children on same proc at different steps.
// v0@P0,S0 -> {v1@P1,S1, v2@P1,S2}
// Buffered: send at S0 from P0. Recv at min(1,2)-1 = S0 on P1.
// Moving v1 shifts recv from S0 to S1.
BOOST_AUTO_TEST_CASE(Buffered_MultiChildrenSameProc) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(10, 2, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 1});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestBufferedT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(
        (VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 1, "Buffered MultiChildrenSameProc v1")));
    BOOST_CHECK(
        (VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 2, "Buffered MultiChildrenSameProc v2")));
}

// Fan-in: {v0@P0, v1@P1} -> v2@P2.
BOOST_AUTO_TEST_CASE(Buffered_FanInChild) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(20, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestBufferedT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 2, "Buffered FanInChild")));
}

// Diamond: v0->{v1,v2}->v3 on 3 procs.
BOOST_AUTO_TEST_CASE(Buffered_Diamond) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(8, 3, 1);
    dag.AddVertex(12, 4, 1);
    dag.AddVertex(15, 1, 1);
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

    KlTestBufferedT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 1, "Buffered Diamond v1")));
    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 2, "Buffered Diamond v2")));
}

// Wider: 6 nodes, 3 procs, 3 steps.
BOOST_AUTO_TEST_CASE(Buffered_WiderGraph) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(10, 4, 1);
    dag.AddVertex(10, 2, 1);
    dag.AddVertex(10, 6, 1);
    dag.AddVertex(10, 1, 1);
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

    KlTestBufferedT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 2, "Buffered WiderGraph v2")));
    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 3, "Buffered WiderGraph v3")));
    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 4, "Buffered WiderGraph v4")));
}

// Large fan-in: 4 parents -> v4.
BOOST_AUTO_TEST_CASE(Buffered_LargeFanIn) {
    Graph dag;
    dag.AddVertex(10, 3, 1);
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 4, 1);
    dag.AddVertex(10, 2, 1);
    dag.AddVertex(10, 1, 1);
    dag.AddEdge(0, 4, 1);
    dag.AddEdge(1, 4, 1);
    dag.AddEdge(2, 4, 1);
    dag.AddEdge(3, 4, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(5);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 3, 4});
    schedule.SetAssignedSupersteps({0, 0, 0, 0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestBufferedT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK((VerifyAffinityBruteForcePolicy<KlTestBufferedT, BufferedCommCostPolicy>(kl, 4, "Buffered LargeFanIn v4")));
}

BOOST_AUTO_TEST_SUITE_END()    // AffinityBruteForceLazyBuffered

// ============================================================================
// Suite 2: Staleness penalty/reward
//
// The regular BSP cost function hardcodes staleness to 1 (cross-proc edges
// require target_step >= source_step + 1). Tests verify the penalty/reward
// component of ComputeCommAffinity by differencing (with vs without penalty).
// ============================================================================

BOOST_AUTO_TEST_SUITE(StalenessTests)

// Same-proc candidates should have zero staleness contribution.
BOOST_AUTO_TEST_CASE(StalenessZeroForSameProc) {
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

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    const unsigned numProcs = 2;

    std::vector<std::vector<double>> withStale(numProcs, std::vector<double>(WR, 0.0));
    std::vector<std::vector<double>> noStale(numProcs, std::vector<double>(WR, 0.0));

    kl.GetCommCostF().ComputeCommAffinity(1, withStale, 100.0, 100.0, 0, 1);
    kl.GetCommCostF().ComputeCommAffinity(1, noStale, 0.0, 0.0, 0, 1);

    // Staleness contribution on parent's proc (P0) should be zero
    // (same-proc → no staleness constraint)
    unsigned sourceProc = 0;
    for (unsigned sIdx = 0; sIdx < WR; ++sIdx) {
        double staleContrib = withStale[sourceProc][sIdx] - noStale[sourceProc][sIdx];
        BOOST_CHECK_SMALL(staleContrib, 1e-6);
    }
}

// Cross-proc candidates at violating positions should have non-zero staleness.
BOOST_AUTO_TEST_CASE(StalenessNonZeroCrossProc) {
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
    // v0@P0,S0 -> v1@P1,S1. Staleness=1 is exactly satisfied.
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    const unsigned numProcs = 2;

    std::vector<std::vector<double>> withStale(numProcs, std::vector<double>(WR, 0.0));
    std::vector<std::vector<double>> noStale(numProcs, std::vector<double>(WR, 0.0));

    kl.GetCommCostF().ComputeCommAffinity(1, withStale, 100.0, 100.0, 0, 1);
    kl.GetCommCostF().ComputeCommAffinity(1, noStale, 0.0, 0.0, 0, 1);

    // Moving v1 to S0 on P1 (cross-proc from parent v0@P0) violates staleness.
    double staleContribP1_S0 = withStale[1][0] - noStale[1][0];
    BOOST_CHECK_NE(staleContribP1_S0, 0.0);
}

// Staleness contribution should scale linearly with penalty/reward magnitude.
BOOST_AUTO_TEST_CASE(StalenessScalesWithMagnitude) {
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

    KlTestT kl;
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

    double stale1 = aff1[1][0] - noStale[1][0];
    double stale2 = aff2[1][0] - noStale[1][0];

    if (std::abs(stale1) > 1e-6) {
        BOOST_CHECK_CLOSE(stale2 / stale1, 2.0, 1e-5);
    }
}

// Self-move should have zero TOTAL affinity (staleness + comm delta = 0).
BOOST_AUTO_TEST_CASE(StalenessZeroAtSelfMove) {
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

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.GetCommCostF().ComputeSendReceiveDatastructures();

    constexpr unsigned WR = 3;
    constexpr unsigned WS = 1;
    std::vector<std::vector<double>> aff(2, std::vector<double>(WR, 0.0));

    kl.GetCommCostF().ComputeCommAffinity(1, aff, 100.0, 100.0, 0, 1);

    // Self-move: v1 at (P1, S1), window index = WS = 1.
    BOOST_CHECK_SMALL(aff[1][WS], 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()    // StalenessTests
