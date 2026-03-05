/*
 * @file kl_step_removal.cpp
 * @brief Unit tests for superstep removal/insertion in KL local search.
 *
 * Tests the step removal path:
 *   SwapEmptyStepFwd → SwapCommSteps → endStep--
 *   → UpdateLambdaAfterStepRemoval → FixupSendRecvAfterStepRemoval
 *
 * And the reversal path:
 *   SwapEmptyStepBwd → SwapCommSteps (reverse) → endStep++
 *   → UpdateLambdaAfterStepInsertion → FixupSendRecvAfterStepInsertion
 *
 * Key invariant: after removal (and after reinsertion), incremental
 * data structures must match a from-scratch recomputation.
 *
 * For lazy/buffered: an empty step can carry receive comm (placed at
 * min(child_steps)-1). When removed, this comm merges into the
 * preceding step. The cost delta must reflect this.
 */

#define BOOST_TEST_MODULE kl_step_removal
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/comm_cost_policies.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/max_comm_datastructure.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_active_schedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;
using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using KlActiveScheduleT = KlActiveSchedule<Graph, double, NoLocalSearchMemoryConstraint>;
using KlMove = KlMoveStruct<double, Graph::VertexIdx>;

// ============================================================================
// Helpers
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

/// Validate incremental comm ds against from-scratch over [0, numActiveSteps).
template <typename CommPolicy>
bool ValidateCommDs(MaxCommDatastructure<Graph, double, KlActiveScheduleT, CommPolicy> &commDsInc,
                    KlActiveScheduleT &activeSched,
                    const BspInstance<Graph> &instance,
                    unsigned numActiveSteps,
                    const std::string &context) {
    BspSchedule<Graph> snap(instance);
    activeSched.WriteSchedule(snap);

    KlActiveScheduleT fresh;
    fresh.Initialize(snap);

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, CommPolicy> freshDs;
    freshDs.Initialize(fresh);
    freshDs.ComputeCommDatastructures(0, numActiveSteps > 0 ? numActiveSteps - 1 : 0);

    bool ok = true;
    for (unsigned s = 0; s < numActiveSteps; ++s) {
        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            if (std::abs(commDsInc.StepProcSend(s, p) - freshDs.StepProcSend(s, p)) > 1e-6
                || std::abs(commDsInc.StepProcReceive(s, p) - freshDs.StepProcReceive(s, p)) > 1e-6) {
                ok = false;
                std::cout << "  [" << context << "] SEND/RECV mismatch step=" << s << " proc=" << p
                          << "  inc(s=" << commDsInc.StepProcSend(s, p) << ",r=" << commDsInc.StepProcReceive(s, p) << ")"
                          << "  fresh(s=" << freshDs.StepProcSend(s, p) << ",r=" << freshDs.StepProcReceive(s, p) << ")\n";
            }
        }
        if (std::abs(commDsInc.StepMaxComm(s) - freshDs.StepMaxComm(s)) > 1e-6) {
            ok = false;
            std::cout << "  [" << context << "] MAX mismatch step=" << s << "  inc=" << commDsInc.StepMaxComm(s)
                      << "  fresh=" << freshDs.StepMaxComm(s) << "\n";
        }
    }

    using ValT = typename CommPolicy::ValueType;
    for (const auto v : instance.Vertices()) {
        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            bool hasI = commDsInc.nodeLambdaMap_.HasProcEntry(v, p);
            bool hasF = freshDs.nodeLambdaMap_.HasProcEntry(v, p);
            if (hasI != hasF) {
                ok = false;
                std::cout << "  [" << context << "] LAMBDA presence mismatch node=" << v << " proc=" << p << "\n";
            } else if (hasI) {
                const ValT &vi = commDsInc.nodeLambdaMap_.GetProcEntry(v, p);
                const ValT &vf = freshDs.nodeLambdaMap_.GetProcEntry(v, p);
                if (!LambdaValuesEqual(vi, vf)) {
                    ok = false;
                    std::cout << "  [" << context << "] LAMBDA mismatch node=" << v << " proc=" << p
                              << "  inc=" << LambdaValueStr(vi) << "  fresh=" << LambdaValueStr(vf) << "\n";
                }
            }
        }
    }
    return ok;
}

/// Perform step removal: SwapEmptyStepFwd + swap comm steps + fixups.
template <typename CommPolicy>
void DoRemoveStep(unsigned removedStep,
                  unsigned &endStep,
                  KlActiveScheduleT &sched,
                  MaxCommDatastructure<Graph, double, KlActiveScheduleT, CommPolicy> &ds) {
    sched.SwapEmptyStepFwd(removedStep, endStep);
    const unsigned oldEndStep = endStep;
    for (unsigned i = removedStep; i < endStep; i++) {
        ds.SwapSteps(i, i + 1);
    }
    endStep--;
    ds.UpdateLambdaAfterStepRemoval(removedStep);
    ds.FixupSendRecvAfterStepRemoval(removedStep, oldEndStep);
    sched.GetVectorSchedule().numberOfSupersteps_ = endStep + 1;
}

/// Perform step reinsertion (reversal of removal).
template <typename CommPolicy>
void DoInsertStep(unsigned insertedStep,
                  unsigned &endStep,
                  unsigned startStep,
                  KlActiveScheduleT &sched,
                  MaxCommDatastructure<Graph, double, KlActiveScheduleT, CommPolicy> &ds) {
    sched.SwapEmptyStepBwd(++endStep, insertedStep);
    for (unsigned i = endStep; i > insertedStep; i--) {
        ds.SwapSteps(i - 1, i);
    }
    ds.UpdateLambdaAfterStepInsertion(insertedStep);
    ds.FixupSendRecvAfterStepInsertion(insertedStep, startStep, endStep);
    sched.GetVectorSchedule().numberOfSupersteps_ = endStep + 1;
}

#define INSTANTIATE_ALL(FuncName)                                                     \
    BOOST_AUTO_TEST_CASE(FuncName##_Eager) { FuncName<EagerCommCostPolicy>(); }       \
    BOOST_AUTO_TEST_CASE(FuncName##_Lazy) { FuncName<LazyCommCostPolicy>(); }         \
    BOOST_AUTO_TEST_CASE(FuncName##_Buffered) { FuncName<BufferedCommCostPolicy>(); }

// ============================================================================
// TEST 1: Scatter → remove step → validate data structures.
//
// 0→1, 0→2.  0@(P0,S0), 1@(P1,S1), 2@(P1,S2).
// Scatter 1: S1→S2, remove empty S1.
// For lazy/buffered: S1 had receive comm, must merge into S0.
// ============================================================================

template <typename P>
void TestRemoveStep_DataStructures() {
    TestSetup t;
    t.dag.AddVertex(1, 10, 5);
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);

    t.arch.SetNumberOfProcessors(2);
    t.arch.SetSendCosts({
        {0, 7},
        {7, 0}
    });
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(100);

    t.Build({0, 1, 1}, {0, 1, 2});
    unsigned endStep = 2;

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, endStep);

    KlMove scatter(1, 0.0, 1, 1, 1, 2);
    t.Apply(scatter);
    ds.UpdateDatastructureAfterMove(scatter, 0, endStep);

    DoRemoveStep(1, endStep, *t.klSched, ds);

    BOOST_CHECK_MESSAGE(ValidateCommDs<P>(ds, *t.klSched, *t.instance, endStep + 1, PolicyName<P>() + std::string("_remove")),
                        PolicyName<P>() << " diverged after step removal");
}
INSTANTIATE_ALL(TestRemoveStep_DataStructures)

// ============================================================================
// TEST 2: Scatter → remove → re-insert → undo scatter = initial state.
//
// This is exactly what RevertToBestSchedule does when bestScheduleIdx_
// is before the scatter (i.e. the removal didn't help).
// ============================================================================

template <typename P>
void TestRemoveStep_Roundtrip() {
    TestSetup t;
    t.dag.AddVertex(1, 10, 5);
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);

    t.arch.SetNumberOfProcessors(2);
    t.arch.SetSendCosts({
        {0, 7},
        {7, 0}
    });
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(100);

    t.Build({0, 1, 1}, {0, 1, 2});
    unsigned endStep = 2;

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, endStep);

    // Scatter + remove
    KlMove scatter(1, 0.0, 1, 1, 1, 2);
    t.Apply(scatter);
    ds.UpdateDatastructureAfterMove(scatter, 0, endStep);
    DoRemoveStep(1, endStep, *t.klSched, ds);

    // Re-insert + undo scatter (= RevertMoves back to bound 0)
    DoInsertStep(1, endStep, 0, *t.klSched, ds);
    auto revScatter = scatter.ReverseMove();
    t.Apply(revScatter);
    ds.UpdateDatastructureAfterMove(revScatter, 0, endStep);

    BOOST_CHECK_MESSAGE(ValidateCommDs<P>(ds, *t.klSched, *t.instance, endStep + 1, PolicyName<P>() + std::string("_roundtrip")),
                        PolicyName<P>() << " diverged after full roundtrip back to initial state");
}
INSTANTIATE_ALL(TestRemoveStep_Roundtrip)

// ============================================================================
// TEST 3: Cost delta formula matches actual cost change.
//
// Verifies: costBefore + (-syncCost + commDelta) == costAfter
// where commDelta = (-removedMax + (newPrevMax - oldPrevMax)) * commMul
// ============================================================================

template <typename P>
void TestRemoveStep_CostDelta() {
    TestSetup t;
    t.dag.AddVertex(1, 10, 5);
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);

    t.arch.SetNumberOfProcessors(2);
    t.arch.SetSendCosts({
        {0, 7},
        {7, 0}
    });
    t.arch.SetCommunicationCosts(3);
    t.arch.SetSynchronisationCosts(50);

    t.Build({0, 1, 1}, {0, 1, 2});
    unsigned endStep = 2;
    const double commMul = t.instance->CommunicationCosts();
    const double syncVal = t.instance->SynchronisationCosts();

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, endStep);

    // Scatter
    KlMove scatter(1, 0.0, 1, 1, 1, 2);
    t.Apply(scatter);
    ds.UpdateDatastructureAfterMove(scatter, 0, endStep);

    // Cost before removal
    double costBefore = 0;
    for (unsigned s = 0; s <= endStep; ++s) {
        costBefore += t.klSched->GetStepMaxWork(s) + ds.StepMaxComm(s) * commMul;
    }
    costBefore += endStep * syncVal;

    const double removedMax = ds.StepMaxComm(1);
    const double prevMax = ds.StepMaxComm(0);

    DoRemoveStep(1, endStep, *t.klSched, ds);

    // Cost after removal
    double costAfter = 0;
    for (unsigned s = 0; s <= endStep; ++s) {
        costAfter += t.klSched->GetStepMaxWork(s) + ds.StepMaxComm(s) * commMul;
    }
    if (endStep > 0) {
        costAfter += endStep * syncVal;
    }

    // Tracked
    double commDelta = -removedMax * commMul + (ds.StepMaxComm(0) - prevMax) * commMul;
    double tracked = costBefore - syncVal + commDelta;

    BOOST_CHECK_MESSAGE(std::abs(costAfter - tracked) < 1e-6, PolicyName<P>() << " cost delta error=" << (costAfter - tracked));
}
INSTANTIATE_ALL(TestRemoveStep_CostDelta)

// ============================================================================
// TEST 4: Removing step 0 (no R-1 to merge into).
// ============================================================================

template <typename P>
void TestRemoveStep_StepZero() {
    TestSetup t;
    t.dag.AddVertex(1, 10, 5);
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddEdge(0, 1, 1);

    t.arch.SetNumberOfProcessors(2);
    t.arch.SetSendCosts({
        {0, 4},
        {4, 0}
    });
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(50);

    t.Build({0, 1}, {0, 1});
    unsigned endStep = 1;

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, endStep);

    KlMove scatter(0, 0.0, 0, 0, 0, 1);
    t.Apply(scatter);
    ds.UpdateDatastructureAfterMove(scatter, 0, endStep);

    DoRemoveStep(0, endStep, *t.klSched, ds);

    BOOST_CHECK_MESSAGE(ValidateCommDs<P>(ds, *t.klSched, *t.instance, endStep + 1, PolicyName<P>() + std::string("_step0")),
                        PolicyName<P>() << " diverged after step 0 removal");
}
INSTANTIATE_ALL(TestRemoveStep_StepZero)

// ============================================================================
// TEST 5: Multi-proc with asymmetric send costs.
// ============================================================================

template <typename P>
void TestRemoveStep_MultiProc() {
    TestSetup t;
    t.dag.AddVertex(1, 10, 8);
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddVertex(1, 3, 1);
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);
    t.dag.AddEdge(0, 3, 1);

    t.arch.SetNumberOfProcessors(4);
    t.arch.SetSendCosts({
        {0, 2, 5, 3},
        {2, 0, 4, 6},
        {5, 4, 0, 1},
        {3, 6, 1, 0}
    });
    t.arch.SetCommunicationCosts(2);
    t.arch.SetSynchronisationCosts(50);

    // 0@(P0,S0), children all at S1 on different procs
    t.Build({0, 1, 2, 3}, {0, 1, 1, 1});
    unsigned endStep = 1;

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, endStep);

    // Scatter all to S0
    KlMove s1(1, 0.0, 1, 1, 1, 0);
    t.Apply(s1);
    ds.UpdateDatastructureAfterMove(s1, 0, endStep);
    KlMove s2(2, 0.0, 2, 1, 2, 0);
    t.Apply(s2);
    ds.UpdateDatastructureAfterMove(s2, 0, endStep);
    KlMove s3(3, 0.0, 3, 1, 3, 0);
    t.Apply(s3);
    ds.UpdateDatastructureAfterMove(s3, 0, endStep);

    DoRemoveStep(1, endStep, *t.klSched, ds);

    BOOST_CHECK_MESSAGE(ValidateCommDs<P>(ds, *t.klSched, *t.instance, endStep + 1, PolicyName<P>() + std::string("_multiproc")),
                        PolicyName<P>() << " diverged after multi-proc step removal");
}
INSTANTIATE_ALL(TestRemoveStep_MultiProc)

// ============================================================================
// TEST 6: Scatter → remove → move → full revert.
// ============================================================================

template <typename P>
void TestRemoveStep_ScatterRemoveRevert() {
    TestSetup t;
    t.dag.AddVertex(1, 10, 5);
    t.dag.AddVertex(1, 4, 1);
    t.dag.AddVertex(1, 4, 1);
    t.dag.AddVertex(1, 4, 1);
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);
    t.dag.AddEdge(0, 3, 1);

    t.arch.SetNumberOfProcessors(2);
    t.arch.SetSendCosts({
        {0, 7},
        {7, 0}
    });
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(50);

    t.Build({0, 1, 1, 1}, {0, 1, 2, 3});
    const unsigned startStep = 0;
    unsigned endStep = 3;
    const double syncVal = t.instance->SynchronisationCosts();

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, endStep);

    std::vector<KlMove> moves;

    // Scatter + remove
    KlMove scatter(1, 0.0, 1, 1, 1, 2);
    t.Apply(scatter);
    ds.UpdateDatastructureAfterMove(scatter, startStep, endStep);
    moves.push_back(scatter);

    DoRemoveStep(1, endStep, *t.klSched, ds);
    moves.push_back(KlMove::MakeRemoveStep(1, syncVal));

    // A regular move
    KlMove m1(2, 0.0, 1, 1, 0, 1);
    t.Apply(m1);
    ds.UpdateDatastructureAfterMove(m1, startStep, endStep);
    moves.push_back(m1);

    // Full revert
    while (!moves.empty()) {
        const auto move = moves.back();
        moves.pop_back();
        if (move.type_ == KlMoveType::REMOVE_STEP) {
            DoInsertStep(move.fromStep_, endStep, startStep, *t.klSched, ds);
        } else {
            auto rev = move.ReverseMove();
            t.Apply(rev);
            ds.UpdateDatastructureAfterMove(rev, startStep, endStep);
        }
    }

    BOOST_CHECK_MESSAGE(ValidateCommDs<P>(ds, *t.klSched, *t.instance, endStep + 1, PolicyName<P>() + std::string("_revert")),
                        PolicyName<P>() << " diverged after scatter→remove→move→revert");
}
INSTANTIATE_ALL(TestRemoveStep_ScatterRemoveRevert)

// ============================================================================
// TEST 7: Cost delta with asymmetric send costs and 3 procs.
// ============================================================================

template <typename P>
void TestRemoveStep_CostDeltaAsymmetric() {
    TestSetup t;
    t.dag.AddVertex(1, 10, 3);
    t.dag.AddVertex(1, 5, 2);
    t.dag.AddVertex(1, 5, 2);
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);

    t.arch.SetNumberOfProcessors(3);
    t.arch.SetSendCosts({
        {0, 3, 5},
        {3, 0, 2},
        {5, 2, 0}
    });
    t.arch.SetCommunicationCosts(2);
    t.arch.SetSynchronisationCosts(80);

    t.Build({0, 1, 2}, {0, 1, 1});
    unsigned endStep = 1;
    const double commMul = t.instance->CommunicationCosts();
    const double syncVal = t.instance->SynchronisationCosts();

    MaxCommDatastructure<Graph, double, KlActiveScheduleT, P> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, endStep);

    // Scatter children to S0
    KlMove s1(1, 0.0, 1, 1, 1, 0);
    t.Apply(s1);
    ds.UpdateDatastructureAfterMove(s1, 0, endStep);
    KlMove s2(2, 0.0, 2, 1, 2, 0);
    t.Apply(s2);
    ds.UpdateDatastructureAfterMove(s2, 0, endStep);

    double costBefore = 0;
    for (unsigned s = 0; s <= endStep; ++s) {
        costBefore += t.klSched->GetStepMaxWork(s) + ds.StepMaxComm(s) * commMul;
    }
    costBefore += endStep * syncVal;

    const double removedMax = ds.StepMaxComm(1);
    const double prevMax = ds.StepMaxComm(0);

    DoRemoveStep(1, endStep, *t.klSched, ds);

    double costAfter = 0;
    for (unsigned s = 0; s <= endStep; ++s) {
        costAfter += t.klSched->GetStepMaxWork(s) + ds.StepMaxComm(s) * commMul;
    }
    if (endStep > 0) {
        costAfter += endStep * syncVal;
    }

    double commDelta = -removedMax * commMul + (ds.StepMaxComm(0) - prevMax) * commMul;
    double tracked = costBefore - syncVal + commDelta;

    BOOST_CHECK_MESSAGE(std::abs(costAfter - tracked) < 1e-6,
                        PolicyName<P>() << " asymmetric cost delta error=" << (costAfter - tracked));
}
INSTANTIATE_ALL(TestRemoveStep_CostDeltaAsymmetric)
