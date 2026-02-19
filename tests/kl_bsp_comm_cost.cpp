/*
 * @file kl_bsp_cost_policies_stress.cpp
 * @brief Stress tests designed to expose the Lazy/Buffered incremental update bug.
 *
 * ============================================================================
 * TEST COVERAGE GAP ANALYSIS
 * ============================================================================
 *
 * The existing test suites (1-5) thoroughly cover:
 *   ✓ Initial placement correctness for all three policies
 *   ✓ Single moves with ValidateCommDs (all policies)
 *   ✓ Forward-only multi-move sequences (3-5 moves)
 *   ✓ Full rollback of 5-move sequences
 *   ✓ Partial rollback (forward, partial reverse, forward again)
 *   ✓ Asymmetric send costs with rollback (4 moves + reverse)
 *   ✓ Edge cases: step-0 children, duplicate steps, isolated nodes, etc.
 *   ✓ Min-step shift on same-proc children
 *   ✓ Zigzag proc changes
 *
 * Identified coverage gaps that could hide the bug:
 *
 * GAP 1: Moving a PARENT node (outgoing section) when fromStep != toStep
 *         AND fromProc != toProc, combined with Lazy/Buffered + asymmetric costs.
 *         Existing TestMoveParent (4f) only does proc-change-only and step-change-only
 *         moves, never both simultaneously. The integration test regularly produces
 *         combined step+proc changes.
 *
 * GAP 2: Long move sequences (>10 moves) with accumulation effects.
 *         After many incremental updates, small numerical drifts or state
 *         corruption can accumulate. Existing tests cap at 5-10 moves.
 *
 * GAP 3: Moving a parent node AFTER its children have been rearranged by
 *         previous moves. The outgoing section uses the CURRENT lambda (which
 *         may have been modified by the parent section of earlier moves),
 *         creating a subtle ordering dependency.
 *
 * GAP 4: Same-proc step changes where the moved node is both a parent (has
 *         children, processed in outgoing section) AND a child (has parents,
 *         processed in parent section), with asymmetric costs.
 *
 * GAP 5: Rollback sequences where the reversed move undoes a min-step shift,
 *         combined with asymmetric costs creating different cost magnitudes
 *         for the removal vs. addition sides.
 *
 * GAP 6: Fan-out hub nodes with children on 3+ different procs, where
 *         sequential moves to/from these procs exercise all combinations
 *         of the `if (proc != fromProc)` / `if (proc != toProc)` guards.
 *
 * GAP 7: Randomized stress testing — the integration test effectively does
 *         this (the KL optimizer makes data-driven moves on complex graphs).
 *         No existing unit test uses randomization.
 *
 * ============================================================================
 * BUG HYPOTHESIS
 * ============================================================================
 *
 * After exhaustive analytical tracing of every code path in the Lazy and
 * Buffered policies, every individual operation checks out as correct in
 * isolation. The bug therefore manifests only through specific multi-move
 * SEQUENCES that create a state unreachable by the existing tests.
 *
 * Most likely root cause candidates:
 *
 * (A) Outgoing section: simultaneous step+proc change with asymmetric costs.
 *     When a parent node moves from (P0,S0) to (P2,S3), and has children on
 *     P1 (cost SendCosts(P0,P1)=1) and P3 (cost SendCosts(P0,P3)=4), the
 *     removal uses old costs and the addition uses new costs. For Lazy, both
 *     removal and addition target min(val)-1, so recv side should cancel.
 *     But with asymmetric costs, the AMOUNTS differ, so recv doesn't cancel.
 *     This is CORRECT behavior, but if the from-scratch computation computes
 *     a different amount, there's a discrepancy.
 *
 * (B) Parent section aliasing when fromProc == toProc: The same lambda vector
 *     is modified by both RemoveChild and AddChild. After RemoveChild, the
 *     vector state fed into UnattributeCommunication might interact badly
 *     with the subsequent AddChild+AttributeCommunication when there are
 *     duplicate step values.
 *
 * (C) Interaction between outgoing and parent sections within the SAME
 *     UpdateDatastructureAfterMove call, when both sections modify the
 *     same (step, proc) cell in stepProcSend_/stepProcReceive_ (different
 *     edges, but same cell). The net result should be additive (commutative),
 *     but a missed MarkStep or cache invalidation could cause ArrangeSuperstepCommData
 *     to miss a dirty step.
 *
 * These tests systematically exercise all three candidates.
 * ============================================================================
 */

#define BOOST_TEST_MODULE kl_bsp_cost_policies_stress
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_bsp_comm_cost.hpp"

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/comm_cost_policies.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/max_comm_datastructure.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_active_schedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;
using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using KlActiveScheduleT = KlActiveSchedule<Graph, double, NoLocalSearchMemoryConstraint>;
using KlMove = KlMoveStruct<double, Graph::VertexIdx>;

// ============================================================================
// Reuse ValidateCommDs and helpers from the main test file
// ============================================================================

template <typename T>
bool LambdaValuesEqual(const T &a, const T &b) {
    return a == b;
}

template <>
bool LambdaValuesEqual<std::vector<unsigned>>(const std::vector<unsigned> &a, const std::vector<unsigned> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    auto sa = a, sb = b;
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    return sa == sb;
}

template <typename T>
std::string LambdaValueStr(const T &val) {
    return std::to_string(val);
}

template <>
std::string LambdaValueStr<std::vector<unsigned>>(const std::vector<unsigned> &val) {
    std::string s = "[";
    for (size_t i = 0; i < val.size(); ++i) {
        if (i) {
            s += ",";
        }
        s += std::to_string(val[i]);
    }
    return s + "]";
}

template <typename CommPolicy>
bool ValidateCommDs(const MaxCommDatastructure<Graph, double, KlActiveScheduleT, CommPolicy> &commDsInc,
                    KlActiveScheduleT &activeSched,
                    const BspInstance<Graph> &instance,
                    const std::string &context) {
    BspSchedule<Graph> currentSchedule(instance);
    activeSched.WriteSchedule(currentSchedule);

    KlActiveScheduleT klSchedFresh;
    klSchedFresh.Initialize(currentSchedule);

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, CommPolicy> commDsFresh;
    commDsFresh.Initialize(klSchedFresh);
    unsigned maxStep = currentSchedule.NumberOfSupersteps();
    commDsFresh.ComputeCommDatastructures(0, maxStep > 0 ? maxStep - 1 : 0);

    bool ok = true;

    for (unsigned step = 0; step < maxStep; ++step) {
        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            auto sI = commDsInc.StepProcSend(step, p);
            auto sF = commDsFresh.StepProcSend(step, p);
            auto rI = commDsInc.StepProcReceive(step, p);
            auto rF = commDsFresh.StepProcReceive(step, p);
            if (std::abs(sI - sF) > 1e-6 || std::abs(rI - rF) > 1e-6) {
                ok = false;
                std::cout << "  [" << context << "] SEND/RECV mismatch step=" << step << " proc=" << p << "  inc(s=" << sI
                          << ",r=" << rI << ")  fresh(s=" << sF << ",r=" << rF << ")\n";
            }
        }
    }

    for (unsigned step = 0; step < maxStep; ++step) {
        if (commDsInc.StepMaxComm(step) != commDsFresh.StepMaxComm(step)) {
            ok = false;
            std::cout << "  [" << context << "] MAX mismatch step=" << step << "  inc=" << commDsInc.StepMaxComm(step)
                      << "  fresh=" << commDsFresh.StepMaxComm(step) << "\n";
        }
    }

    using ValT = typename CommPolicy::ValueType;
    for (const auto v : instance.Vertices()) {
        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            bool hasI = commDsInc.nodeLambdaMap_.HasProcEntry(v, p);
            bool hasF = commDsFresh.nodeLambdaMap_.HasProcEntry(v, p);
            if (hasI != hasF) {
                ok = false;
                std::cout << "  [" << context << "] LAMBDA presence mismatch node=" << v << " proc=" << p << "\n";
            } else if (hasI) {
                const ValT &valI = commDsInc.nodeLambdaMap_.GetProcEntry(v, p);
                const ValT &valF = commDsFresh.nodeLambdaMap_.GetProcEntry(v, p);
                if (!LambdaValuesEqual(valI, valF)) {
                    ok = false;
                    std::cout << "  [" << context << "] LAMBDA value mismatch node=" << v << " proc=" << p
                              << "  inc=" << LambdaValueStr(valI) << "  fresh=" << LambdaValueStr(valF) << "\n";
                }
            }
        }
    }

    return ok;
}

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

struct TestSetup {
    Graph dag;
    BspArchitecture<Graph> arch;
    std::unique_ptr<BspInstance<Graph>> instance;
    std::unique_ptr<BspSchedule<Graph>> schedule;
    std::unique_ptr<KlActiveScheduleT> klSched;
    ThreadLocalActiveScheduleData<Graph, double> asd;

    void Build(const std::vector<unsigned> &procs, const std::vector<unsigned> &steps) {
        instance = std::make_unique<BspInstance<Graph>>(dag, arch);
        schedule = std::make_unique<BspSchedule<Graph>>(*instance);
        schedule->SetAssignedProcessors(procs);
        schedule->SetAssignedSupersteps(steps);
        schedule->UpdateNumberOfSupersteps();
        klSched = std::make_unique<KlActiveScheduleT>();
        klSched->Initialize(*schedule);
        asd.InitializeCost(0.0);
    }

    void Apply(KlMove &m) { klSched->ApplyMove(m, asd); }
};

#define INSTANTIATE_ALL(FuncName)                                                     \
    BOOST_AUTO_TEST_CASE(FuncName##_Eager) { FuncName<EagerCommCostPolicy>(); }       \
    BOOST_AUTO_TEST_CASE(FuncName##_Lazy) { FuncName<LazyCommCostPolicy>(); }         \
    BOOST_AUTO_TEST_CASE(FuncName##_Buffered) { FuncName<BufferedCommCostPolicy>(); }

// Only Lazy/Buffered (Eager is known-correct for incremental updates)
#define INSTANTIATE_LAZY_BUFFERED(FuncName)                                           \
    BOOST_AUTO_TEST_CASE(FuncName##_Lazy) { FuncName<LazyCommCostPolicy>(); }         \
    BOOST_AUTO_TEST_CASE(FuncName##_Buffered) { FuncName<BufferedCommCostPolicy>(); }

// ============================================================================
// TEST 1: GAP 1 — Simultaneous step+proc change for parent node
//
// A parent with children on multiple procs moves both step AND proc.
// This exercises the outgoing section's `if (fromStep != toStep)` branch
// with `fromProc != toProc`, which existing tests don't cover for
// Lazy/Buffered with asymmetric costs.
// ============================================================================

template <typename P>
void TestParentMoveStepAndProc() {
    TestSetup t;
    // Node 0: parent, high comm weight. Nodes 1-4: children.
    t.dag.AddVertex(1, 10, 1);    // 0: parent
    t.dag.AddVertex(1, 3, 1);     // 1
    t.dag.AddVertex(1, 3, 1);     // 2
    t.dag.AddVertex(1, 3, 1);     // 3
    t.dag.AddVertex(1, 3, 1);     // 4
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);
    t.dag.AddEdge(0, 3, 1);
    t.dag.AddEdge(0, 4, 1);

    t.arch.SetNumberOfProcessors(4);
    std::vector<std::vector<int>> sendCosts = {
        {0, 1, 4, 4},
        {1, 0, 4, 4},
        {4, 4, 0, 1},
        {4, 4, 1, 0}
    };
    t.arch.SetSendCosts(sendCosts);
    t.arch.SetSynchronisationCosts(0);

    // Parent 0@(P0,S0), children: 1@(P1,S2), 2@(P2,S3), 3@(P3,S2), 4@(P1,S4)
    t.Build({0, 1, 2, 3, 1}, {0, 2, 3, 2, 4});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 4);
    const std::string tag = PolicyName<P>();
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_psp_init"));

    // Move parent 0 from (P0,S0) to (P2,S1) — BOTH step AND proc change
    // This is the critical pattern from GAP 1.
    KlMove m1(0, 0.0, 0, 0, 2, 1);
    t.Apply(m1);
    ds.UpdateDatastructureAfterMove(m1, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_psp_m1"));

    // Move parent 0 from (P2,S1) to (P3,S3) — another combined change
    KlMove m2(0, 0.0, 2, 1, 3, 3);
    t.Apply(m2);
    ds.UpdateDatastructureAfterMove(m2, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_psp_m2"));

    // Rollback m2
    KlMove r2 = m2.ReverseMove();
    t.Apply(r2);
    ds.UpdateDatastructureAfterMove(r2, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_psp_r2"));

    // Rollback m1
    KlMove r1 = m1.ReverseMove();
    t.Apply(r1);
    ds.UpdateDatastructureAfterMove(r1, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_psp_r1"));
}
INSTANTIATE_ALL(TestParentMoveStepAndProc)

// ============================================================================
// TEST 2: GAP 3 — Move parent AFTER children rearranged
//
// First rearrange children (modifying lambda entries via parent section),
// then move the parent (outgoing section reads modified lambda).
// ============================================================================

template <typename P>
void TestMoveParentAfterChildRearrange() {
    TestSetup t;
    t.dag.AddVertex(1, 12, 1);    // 0: parent
    t.dag.AddVertex(1, 5, 1);     // 1
    t.dag.AddVertex(1, 7, 1);     // 2
    t.dag.AddVertex(1, 3, 1);     // 3
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);
    t.dag.AddEdge(0, 3, 1);

    t.arch.SetNumberOfProcessors(4);
    std::vector<std::vector<int>> sendCosts = {
        {0, 1, 4, 4},
        {1, 0, 4, 4},
        {4, 4, 0, 1},
        {4, 4, 1, 0}
    };
    t.arch.SetSendCosts(sendCosts);
    t.arch.SetSynchronisationCosts(0);

    // Parent 0@(P0,S0). Children: 1@(P1,S2), 2@(P1,S4), 3@(P2,S3)
    t.Build({0, 1, 1, 2}, {0, 2, 4, 3});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 4);
    const std::string tag = PolicyName<P>();
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_mpar_init"));

    // Phase 1: Rearrange children (modifies lambda for parent 0)
    // Move child 1 from (P1,S2) to (P3,S2) — changes which procs have children
    KlMove m1(1, 0.0, 1, 2, 3, 2);
    t.Apply(m1);
    ds.UpdateDatastructureAfterMove(m1, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_mpar_c1"));

    // Move child 2 from (P1,S4) to (P1,S1) — changes min step on P1
    KlMove m2(2, 0.0, 1, 4, 1, 1);
    t.Apply(m2);
    ds.UpdateDatastructureAfterMove(m2, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_mpar_c2"));

    // Phase 2: NOW move parent — outgoing section reads modified lambda
    // Move parent 0 from (P0,S0) to (P2,S1) — step+proc change
    KlMove m3(0, 0.0, 0, 0, 2, 1);
    t.Apply(m3);
    ds.UpdateDatastructureAfterMove(m3, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_mpar_p1"));

    // Phase 3: Rollback all in reverse
    auto r3 = m3.ReverseMove();
    t.Apply(r3);
    ds.UpdateDatastructureAfterMove(r3, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_mpar_r3"));
    auto r2 = m2.ReverseMove();
    t.Apply(r2);
    ds.UpdateDatastructureAfterMove(r2, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_mpar_r2"));
    auto r1 = m1.ReverseMove();
    t.Apply(r1);
    ds.UpdateDatastructureAfterMove(r1, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_mpar_r1"));
}
INSTANTIATE_ALL(TestMoveParentAfterChildRearrange)

// ============================================================================
// TEST 3: GAP 4 — Hub node (both parent AND child) with step+proc change
//
// Move a node that has BOTH parents and children, exercising both
// outgoing and parent sections in the same UpdateDatastructureAfterMove call,
// with asymmetric costs.
// ============================================================================

template <typename P>
void TestHubNodeMove() {
    TestSetup t;
    t.dag.AddVertex(1, 8, 1);     // 0: grandparent
    t.dag.AddVertex(1, 10, 1);    // 1: hub (parent of 3,4; child of 0)
    t.dag.AddVertex(1, 6, 1);     // 2: another parent of hub
    t.dag.AddVertex(1, 3, 1);     // 3: child of hub
    t.dag.AddVertex(1, 4, 1);     // 4: child of hub
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(2, 1, 1);
    t.dag.AddEdge(1, 3, 1);
    t.dag.AddEdge(1, 4, 1);

    t.arch.SetNumberOfProcessors(4);
    std::vector<std::vector<int>> sendCosts = {
        {0, 1, 4, 4},
        {1, 0, 4, 4},
        {4, 4, 0, 1},
        {4, 4, 1, 0}
    };
    t.arch.SetSendCosts(sendCosts);
    t.arch.SetSynchronisationCosts(0);

    // 0@(P0,S0), 1@(P1,S1), 2@(P2,S0), 3@(P3,S2), 4@(P0,S3)
    t.Build({0, 1, 2, 3, 0}, {0, 1, 0, 2, 3});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 3);
    const std::string tag = PolicyName<P>();
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_hub_init"));

    // Move hub node 1 from (P1,S1) to (P3,S2) — combined step+proc change
    // Both outgoing (node1→{3,4}) and parent ({0,2}→node1) sections fire
    KlMove m1(1, 0.0, 1, 1, 3, 2);
    t.Apply(m1);
    ds.UpdateDatastructureAfterMove(m1, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_hub_m1"));

    // Move hub again: (P3,S2) to (P0,S0)
    KlMove m2(1, 0.0, 3, 2, 0, 0);
    t.Apply(m2);
    ds.UpdateDatastructureAfterMove(m2, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_hub_m2"));

    // Rollback
    auto r2 = m2.ReverseMove();
    t.Apply(r2);
    ds.UpdateDatastructureAfterMove(r2, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_hub_r2"));
    auto r1 = m1.ReverseMove();
    t.Apply(r1);
    ds.UpdateDatastructureAfterMove(r1, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_hub_r1"));
}
INSTANTIATE_ALL(TestHubNodeMove)

// ============================================================================
// TEST 4: GAP 2 — Long sequential move sequence (20+ moves)
//
// Multi-layer DAG with 4 procs, asymmetric costs. Apply 20 moves with
// ValidateCommDs after each one. Tests accumulation effects.
// ============================================================================

template <typename P>
void TestLongSequenceAsymmetric() {
    TestSetup t;

    // 3-layer DAG: 4 sources → 4 middle → 4 sinks
    for (int i = 0; i < 12; ++i) {
        t.dag.AddVertex(1 + i % 3, 5 + (i * 7) % 11, 1);
    }
    // Layer 0→1 edges (fan-out)
    for (unsigned s = 0; s < 4; ++s) {
        for (unsigned m = 4; m < 8; ++m) {
            if ((s + m) % 3 != 0) {
                t.dag.AddEdge(s, m, 1);
            }
        }
    }
    // Layer 1→2 edges
    for (unsigned m = 4; m < 8; ++m) {
        for (unsigned d = 8; d < 12; ++d) {
            if ((m + d) % 3 != 0) {
                t.dag.AddEdge(m, d, 1);
            }
        }
    }

    t.arch.SetNumberOfProcessors(4);
    std::vector<std::vector<int>> sendCosts = {
        {0, 1, 4, 4},
        {1, 0, 4, 4},
        {4, 4, 0, 1},
        {4, 4, 1, 0}
    };
    t.arch.SetSendCosts(sendCosts);
    t.arch.SetSynchronisationCosts(0);

    // Initial: sources@S0, middle@S1, sinks@S2; distributed across procs
    std::vector<unsigned> procs = {0, 1, 2, 3, 1, 2, 3, 0, 2, 3, 0, 1};
    std::vector<unsigned> steps = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    t.Build(procs, steps);

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 2);
    const std::string tag = PolicyName<P>();
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_long_init"));

    // Deterministic sequence of 24 moves covering multiple patterns
    struct MoveSpec {
        unsigned node, fromProc, fromStep, toProc, toStep;
    };

    std::vector<MoveSpec> moves = {
        // Same-step proc changes
        { 4, 1, 1, 0, 1},
        { 5, 2, 1, 3, 1},
        { 8, 2, 2, 1, 2},
        // Step changes (same proc)
        { 0, 0, 0, 0, 1},
        { 4, 0, 1, 0, 0},
        // Combined step+proc changes
        { 6, 3, 1, 0, 2},
        { 1, 1, 0, 3, 1},
        { 9, 3, 2, 1, 1},
        // Move middle nodes around (hub nodes)
        { 7, 0, 1, 2, 1},
        { 5, 3, 1, 1, 1},
        // Move sinks
        {10, 0, 2, 3, 2},
        {11, 1, 2, 0, 2},
        // Move sources
        { 2, 2, 0, 0, 0},
        { 3, 3, 0, 1, 0},
        // More combined changes
        { 6, 0, 2, 1, 0},
        { 9, 1, 1, 2, 2},
        // Return some nodes closer to original positions
        { 4, 0, 0, 1, 1},
        { 1, 3, 1, 1, 0},
        // Stress: move same node multiple times
        { 7, 2, 1, 3, 0},
        { 7, 3, 0, 0, 2},
        { 7, 0, 2, 1, 1},
        // Final moves
        { 8, 1, 2, 0, 0},
        {10, 3, 2, 2, 1},
        {11, 0, 2, 3, 0},
    };

    std::vector<KlMove> applied;
    for (size_t i = 0; i < moves.size(); ++i) {
        const auto &ms = moves[i];
        KlMove m(ms.node, 0.0, ms.fromProc, ms.fromStep, ms.toProc, ms.toStep);
        t.Apply(m);
        ds.UpdateDatastructureAfterMove(m, 0, 2);
        applied.push_back(m);
        bool ok = ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_long_m" + std::to_string(i));
        if (!ok) {
            BOOST_CHECK_MESSAGE(false,
                                tag << " diverged at move " << i << ": node=" << ms.node << " (" << ms.fromProc << ","
                                    << ms.fromStep << ")->(" << ms.toProc << "," << ms.toStep << ")");
            return;    // Stop at first failure for diagnostic clarity
        }
    }

    // Full rollback
    for (size_t i = applied.size(); i-- > 0;) {
        auto rev = applied[i].ReverseMove();
        t.Apply(rev);
        ds.UpdateDatastructureAfterMove(rev, 0, 2);
        bool ok = ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_long_r" + std::to_string(i));
        if (!ok) {
            BOOST_CHECK_MESSAGE(false, tag << " diverged at rollback of move " << i);
            return;
        }
    }
}
INSTANTIATE_ALL(TestLongSequenceAsymmetric)

// ============================================================================
// TEST 5: GAP 5 — Rollback of min-step shift with asymmetric costs
//
// Parent with children on same proc at different steps. Move the min-step
// child to a later step (shifting min), then rollback. With asymmetric costs,
// the cost changes create different magnitudes.
// ============================================================================

template <typename P>
void TestMinStepRollbackAsymmetric() {
    TestSetup t;
    t.dag.AddVertex(1, 10, 1);    // 0: parent
    t.dag.AddVertex(1, 3, 1);     // 1: min child
    t.dag.AddVertex(1, 3, 1);     // 2: other child same proc
    t.dag.AddVertex(1, 3, 1);     // 3: child on different proc
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);
    t.dag.AddEdge(0, 3, 1);

    t.arch.SetNumberOfProcessors(4);
    std::vector<std::vector<int>> sendCosts = {
        {0, 1, 4, 4},
        {1, 0, 4, 4},
        {4, 4, 0, 1},
        {4, 4, 1, 0}
    };
    t.arch.SetSendCosts(sendCosts);
    t.arch.SetSynchronisationCosts(0);

    // Parent 0@(P0,S0). Children: 1@(P2,S1), 2@(P2,S4), 3@(P3,S2)
    // lambda[0][P2] = [1, 4] → min=1
    // lambda[0][P3] = [2]
    t.Build({0, 2, 2, 3}, {0, 1, 4, 2});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 4);
    const std::string tag = PolicyName<P>();
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_msr_init"));

    // Move min child (1) from (P2,S1) to (P2,S5) — same proc, step change
    // lambda[0][P2] shifts from min=1 to min=4
    KlMove m1(1, 0.0, 2, 1, 2, 4);
    t.Apply(m1);
    ds.UpdateDatastructureAfterMove(m1, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_msr_fwd1"));

    // Now move child 3 from (P3,S2) to (P2,S2) — changes which procs have children
    KlMove m2(3, 0.0, 3, 2, 2, 2);
    t.Apply(m2);
    ds.UpdateDatastructureAfterMove(m2, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_msr_fwd2"));

    // Rollback m2
    auto r2 = m2.ReverseMove();
    t.Apply(r2);
    ds.UpdateDatastructureAfterMove(r2, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_msr_rev2"));

    // Rollback m1
    auto r1 = m1.ReverseMove();
    t.Apply(r1);
    ds.UpdateDatastructureAfterMove(r1, 0, 4);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_msr_rev1"));
}
INSTANTIATE_ALL(TestMinStepRollbackAsymmetric)

// ============================================================================
// TEST 6: GAP 6 — Fan-out hub to 3+ procs with sequential moves
//
// Exercises all combinations of the `if (proc != fromProc) / (proc != toProc)`
// guards by moving the parent across all 4 procs.
// ============================================================================

template <typename P>
void TestFanOutAllProcCombinations() {
    TestSetup t;
    t.dag.AddVertex(1, 15, 1);    // 0: parent
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddVertex(1, 4, 1);
    t.dag.AddVertex(1, 5, 1);
    t.dag.AddVertex(1, 6, 1);
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);
    t.dag.AddEdge(0, 3, 1);
    t.dag.AddEdge(0, 4, 1);

    t.arch.SetNumberOfProcessors(4);
    std::vector<std::vector<int>> sendCosts = {
        {0, 2, 3, 5},
        {2, 0, 7, 1},
        {3, 7, 0, 4},
        {5, 1, 4, 0}
    };
    t.arch.SetSendCosts(sendCosts);
    t.arch.SetSynchronisationCosts(0);

    // Parent 0@(P0,S0). Children on ALL 4 procs:
    // 1@(P0,S1)-local, 2@(P1,S2), 3@(P2,S3), 4@(P3,S2)
    t.Build({0, 0, 1, 2, 3}, {0, 1, 2, 3, 2});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 3);
    const std::string tag = PolicyName<P>();
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_fapc_init"));

    // Move parent through all procs: P0→P1, P1→P2, P2→P3, P3→P0
    // Each move changes which children are local vs remote
    KlMove m1(0, 0.0, 0, 0, 1, 0);
    t.Apply(m1);
    ds.UpdateDatastructureAfterMove(m1, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_fapc_m1"));

    KlMove m2(0, 0.0, 1, 0, 2, 1);
    t.Apply(m2);
    ds.UpdateDatastructureAfterMove(m2, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_fapc_m2"));

    KlMove m3(0, 0.0, 2, 1, 3, 0);
    t.Apply(m3);
    ds.UpdateDatastructureAfterMove(m3, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_fapc_m3"));

    KlMove m4(0, 0.0, 3, 0, 0, 0);
    t.Apply(m4);
    ds.UpdateDatastructureAfterMove(m4, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_fapc_m4"));

    // Full rollback
    auto r4 = m4.ReverseMove();
    t.Apply(r4);
    ds.UpdateDatastructureAfterMove(r4, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_fapc_r4"));
    auto r3 = m3.ReverseMove();
    t.Apply(r3);
    ds.UpdateDatastructureAfterMove(r3, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_fapc_r3"));
    auto r2 = m2.ReverseMove();
    t.Apply(r2);
    ds.UpdateDatastructureAfterMove(r2, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_fapc_r2"));
    auto r1 = m1.ReverseMove();
    t.Apply(r1);
    ds.UpdateDatastructureAfterMove(r1, 0, 3);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_fapc_r1"));
}
INSTANTIATE_ALL(TestFanOutAllProcCombinations)

// ============================================================================
// TEST 7: GAP 7 — Randomized stress test
//
// Generates random valid moves on a medium-sized DAG with asymmetric costs.
// Validates after every single move. This mimics what the KL optimizer does
// on large graphs.
// ============================================================================

template <typename P>
void TestRandomizedStress() {
    // Build a 20-node 3-layer DAG with varying comm weights
    TestSetup t;
    for (int i = 0; i < 20; ++i) {
        t.dag.AddVertex(1 + i % 4, 3 + (i * 7) % 13, 1);
    }
    // Layer 0 (nodes 0-5) → Layer 1 (nodes 6-12) → Layer 2 (nodes 13-19)
    std::mt19937 edgeRng(12345);
    for (unsigned s = 0; s < 6; ++s) {
        for (unsigned m = 6; m < 13; ++m) {
            if (edgeRng() % 3 != 0) {
                t.dag.AddEdge(s, m, 1);
            }
        }
    }
    for (unsigned m = 6; m < 13; ++m) {
        for (unsigned d = 13; d < 20; ++d) {
            if (edgeRng() % 3 != 0) {
                t.dag.AddEdge(m, d, 1);
            }
        }
    }

    t.arch.SetNumberOfProcessors(4);
    std::vector<std::vector<int>> sendCosts = {
        {0, 1, 4, 4},
        {1, 0, 4, 4},
        {4, 4, 0, 1},
        {4, 4, 1, 0}
    };
    t.arch.SetSendCosts(sendCosts);
    t.arch.SetSynchronisationCosts(0);

    // Initial assignment: spread across 4 procs and 3 steps
    std::vector<unsigned> procs(20), steps(20);
    for (unsigned i = 0; i < 20; ++i) {
        procs[i] = i % 4;
        if (i < 6) {
            steps[i] = 0;
        } else if (i < 13) {
            steps[i] = 1;
        } else {
            steps[i] = 2;
        }
    }
    t.Build(procs, steps);

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 2);
    const std::string tag = PolicyName<P>();
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_rand_init"));

    // Generate 60 random moves and validate after each
    std::mt19937 rng(42);
    std::vector<KlMove> applied;
    unsigned numSteps = 3;
    unsigned numProcs = 4;

    for (unsigned iter = 0; iter < 60; ++iter) {
        // Pick a random node
        unsigned node = static_cast<unsigned>(rng() % 20);
        unsigned curProc = t.klSched->GetVectorSchedule().AssignedProcessor(node);
        unsigned curStep = t.klSched->GetVectorSchedule().AssignedSuperstep(node);

        // Pick random target (different from current)
        unsigned newProc, newStep;
        do {
            newProc = static_cast<unsigned>(rng() % numProcs);
            newStep = static_cast<unsigned>(rng() % numSteps);
        } while (newProc == curProc && newStep == curStep);

        KlMove m(node, 0.0, curProc, curStep, newProc, newStep);
        t.Apply(m);
        ds.UpdateDatastructureAfterMove(m, 0, numSteps - 1);
        applied.push_back(m);

        bool ok = ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_rand_m" + std::to_string(iter));
        if (!ok) {
            BOOST_CHECK_MESSAGE(false,
                                tag << " DIVERGED at random move " << iter << ": node=" << node << " (" << curProc << ","
                                    << curStep << ")->(" << newProc << "," << newStep << ")");
            return;
        }

        // Every 15 moves, do a partial rollback of the last 5
        if ((iter + 1) % 15 == 0 && applied.size() >= 5) {
            for (int r = 0; r < 5; ++r) {
                auto rev = applied.back().ReverseMove();
                t.Apply(rev);
                ds.UpdateDatastructureAfterMove(rev, 0, numSteps - 1);
                applied.pop_back();

                ok = ValidateCommDs<P>(
                    ds, *t.klSched, *t.instance, tag + "_rand_rb" + std::to_string(iter) + "_" + std::to_string(r));
                if (!ok) {
                    BOOST_CHECK_MESSAGE(false, tag << " DIVERGED at rollback " << r << " within iteration " << iter);
                    return;
                }
            }
        }
    }

    // Full rollback of remaining moves
    for (size_t i = applied.size(); i-- > 0;) {
        auto rev = applied[i].ReverseMove();
        t.Apply(rev);
        ds.UpdateDatastructureAfterMove(rev, 0, numSteps - 1);
        bool ok = ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_rand_final_r" + std::to_string(i));
        if (!ok) {
            BOOST_CHECK_MESSAGE(false, tag << " DIVERGED at final rollback of move " << i);
            return;
        }
    }
}
INSTANTIATE_ALL(TestRandomizedStress)

// ============================================================================
// TEST 8: Interleaved forward/reverse on same node — stress lambda aliasing
//
// Rapidly move the same node back and forth between two procs, with a second
// child on the destination proc creating the aliased-reference condition.
// ============================================================================

template <typename P>
void TestRapidSameNodeOscillation() {
    TestSetup t;
    t.dag.AddVertex(1, 10, 1);    // 0: parent
    t.dag.AddVertex(1, 5, 1);     // 1: child A (will oscillate)
    t.dag.AddVertex(1, 7, 1);     // 2: child B (stays on P1)
    t.dag.AddVertex(1, 3, 1);     // 3: child C (stays on P2)
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);
    t.dag.AddEdge(0, 3, 1);

    t.arch.SetNumberOfProcessors(3);
    std::vector<std::vector<int>> sendCosts = {
        {0, 2, 5},
        {2, 0, 3},
        {5, 3, 0}
    };
    t.arch.SetSendCosts(sendCosts);
    t.arch.SetSynchronisationCosts(0);

    // Parent 0@(P0,S0). A@(P1,S1), B@(P1,S3), C@(P2,S2)
    // lambda[0][P1] = [1, 3] → aliased when A moves within P1
    t.Build({0, 1, 1, 2}, {0, 1, 3, 2});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 3);
    const std::string tag = PolicyName<P>();

    // Oscillate A between P1 and P2 10 times, with step changes
    for (unsigned round = 0; round < 10; ++round) {
        // A: P1→P2 with step change
        unsigned fromProc = (round % 2 == 0) ? 1 : 2;
        unsigned toProc = (round % 2 == 0) ? 2 : 1;
        unsigned fromStep = 1 + (round % 3);
        unsigned toStep = 1 + ((round + 1) % 3);

        unsigned curProc = t.klSched->GetVectorSchedule().AssignedProcessor(1);
        unsigned curStep = t.klSched->GetVectorSchedule().AssignedSuperstep(1);

        KlMove m(1, 0.0, curProc, curStep, toProc, toStep);
        t.Apply(m);
        ds.UpdateDatastructureAfterMove(m, 0, 3);
        BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_osc_" + std::to_string(round)));
    }
}
INSTANTIATE_ALL(TestRapidSameNodeOscillation)

// ============================================================================
// TEST 9: Deep chain with propagating min-step shifts
//
// Chain: 0→1→2→3→4. All children on different procs spread across many steps.
// Moving the earliest child shifts min for grandparent's lambda, then moving
// a middle node shifts min for another lambda, creating cascading effects
// that test Lazy/Buffered min-tracking across multiple levels.
// ============================================================================

template <typename P>
void TestDeepChainMinShift() {
    TestSetup t;
    for (int i = 0; i < 5; ++i) {
        t.dag.AddVertex(1, 10 - i * 2, 1);
    }
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(1, 2, 1);
    t.dag.AddEdge(2, 3, 1);
    t.dag.AddEdge(3, 4, 1);

    t.arch.SetNumberOfProcessors(3);
    std::vector<std::vector<int>> sendCosts = {
        {0, 2, 5},
        {2, 0, 3},
        {5, 3, 0}
    };
    t.arch.SetSendCosts(sendCosts);
    t.arch.SetSynchronisationCosts(0);

    // 0@(P0,S0), 1@(P1,S2), 2@(P2,S4), 3@(P0,S6), 4@(P1,S8)
    t.Build({0, 1, 2, 0, 1}, {0, 2, 4, 6, 8});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 8);
    const std::string tag = PolicyName<P>();
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_deep_init"));

    // Move node 1 from (P1,S2) to (P1,S5) — same proc, step change
    KlMove m1(1, 0.0, 1, 2, 1, 5);
    t.Apply(m1);
    ds.UpdateDatastructureAfterMove(m1, 0, 8);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_deep_m1"));

    // Move node 2 from (P2,S4) to (P0,S3) — proc+step change (hub: parent of 3, child of 1)
    KlMove m2(2, 0.0, 2, 4, 0, 3);
    t.Apply(m2);
    ds.UpdateDatastructureAfterMove(m2, 0, 8);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_deep_m2"));

    // Move node 3 from (P0,S6) to (P2,S7) — becomes remote from parent (now node 2@P0,S3)
    KlMove m3(3, 0.0, 0, 6, 2, 7);
    t.Apply(m3);
    ds.UpdateDatastructureAfterMove(m3, 0, 8);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_deep_m3"));

    // Move node 0 from (P0,S0) to (P2,S1) — parent of node 1
    KlMove m4(0, 0.0, 0, 0, 2, 1);
    t.Apply(m4);
    ds.UpdateDatastructureAfterMove(m4, 0, 8);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_deep_m4"));

    // Full rollback
    auto r4 = m4.ReverseMove();
    t.Apply(r4);
    ds.UpdateDatastructureAfterMove(r4, 0, 8);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_deep_r4"));
    auto r3 = m3.ReverseMove();
    t.Apply(r3);
    ds.UpdateDatastructureAfterMove(r3, 0, 8);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_deep_r3"));
    auto r2 = m2.ReverseMove();
    t.Apply(r2);
    ds.UpdateDatastructureAfterMove(r2, 0, 8);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_deep_r2"));
    auto r1 = m1.ReverseMove();
    t.Apply(r1);
    ds.UpdateDatastructureAfterMove(r1, 0, 8);
    BOOST_CHECK(ValidateCommDs<P>(ds, *t.klSched, *t.instance, tag + "_deep_r1"));
}
INSTANTIATE_ALL(TestDeepChainMinShift)
