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
 * @file kl_max_bsp_affinity_test.cpp
 * @brief Comprehensive tests for KlMaxBspCommCostFunction.
 *
 * BSP max-cost formula:
 *   cost = Work[0] + Σ_{s=1}^{S-1} max(Work[s], MaxComm[s-1]) * g + (S-1) * L
 *
 * where:
 *   Work[s]    = max processor work weight at superstep s
 *   MaxComm[s] = max over all procs of max(send(s,p), recv(s,p))
 *   g          = CommunicationCosts()  (bandwidth parameter)
 *   L          = SynchronisationCosts() (latency parameter)
 *
 * Communication at step s, proc p (Eager policy):
 *   For each parent u at step s on proc p with children on proc q != p:
 *     send(s, p) += VertexCommWeight(u) * SendCosts(p, q)
 *     recv(s, q) += VertexCommWeight(u) * SendCosts(p, q)
 *   Comm counted ONCE per distinct destination proc (AddChild returns true only
 *   when lambda count goes 0→1 for Eager).
 *
 * Note: Edge weights are NOT used — BSP comm cost uses VertexCommWeight.
 *
 * Structure:
 *   Suite 1 – ComputeScheduleCost formula verification (direct computation)
 *   Suite 2 – Incremental datastructure validation (manual moves)
 *   Suite 3 – KL integration (InsertGainHeapTest + RunInnerIterationTest)
 *   Suite 4 – Larger and multi-topology graphs
 */

#define BOOST_TEST_MODULE kl_max_bsp_affinity
#include <boost/test/unit_test.hpp>
#include <cmath>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_max_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_test.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;
using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using KlActiveScheduleT = KlActiveSchedule<Graph, double, NoLocalSearchMemoryConstraint>;
using CommCostT = KlMaxBspCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint>;
using KlImproverTestT = KlImproverTest<Graph, CommCostT>;
using KlMove = KlMoveStruct<double, Graph::VertexIdx>;

// =============================================================================
// Helpers
// =============================================================================

/// Compute cost with fresh datastructure recomputation.
static double FreshCost(KlImproverTestT &kl) { return kl.GetCommCostF().ComputeScheduleCost<true>(); }

/// Compute cost without recomputation (uses incremental state).
static double IncrementalCost(KlImproverTestT &kl) { return kl.GetCommCostF().ComputeScheduleCostTest(); }

/// Validate that incremental comm datastructures match a fresh computation.
/// Returns true if all send/recv values match within tolerance.
static bool ValidateCommDs(KlImproverTestT &kl, const std::string &context) {
    auto &costF = kl.GetCommCostF();
    auto *activeSched = costF.active_schedule;
    const auto *inst = costF.instance;
    const auto &dsInc = costF.commDs_;

    // Clone current schedule and recompute from scratch
    BspSchedule<Graph> currentSchedule(*inst);
    activeSched->WriteSchedule(currentSchedule);

    KlActiveScheduleT klSchedFresh;
    klSchedFresh.Initialize(currentSchedule);

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> dsFresh;
    dsFresh.Initialize(klSchedFresh);
    unsigned maxStep = currentSchedule.NumberOfSupersteps();
    dsFresh.ComputeCommDatastructures(0, maxStep > 0 ? maxStep - 1 : 0);

    bool allMatch = true;
    for (unsigned step = 0; step < maxStep; ++step) {
        for (unsigned p = 0; p < inst->NumberOfProcessors(); ++p) {
            auto sendInc = dsInc.StepProcSend(step, p);
            auto sendFresh = dsFresh.StepProcSend(step, p);
            auto recvInc = dsInc.StepProcReceive(step, p);
            auto recvFresh = dsFresh.StepProcReceive(step, p);

            if (std::abs(sendInc - sendFresh) > 1e-6 || std::abs(recvInc - recvFresh) > 1e-6) {
                allMatch = false;
                BOOST_TEST_MESSAGE(context << ": MISMATCH step=" << step << " proc=" << p << " send(inc=" << sendInc
                                           << ", fresh=" << sendFresh << ")"
                                           << " recv(inc=" << recvInc << ", fresh=" << recvFresh << ")");
            }
        }
    }
    return allMatch;
}

// =============================================================================
// Suite 1: ComputeScheduleCost formula verification
//
// Each test builds a schedule, calls FreshCost(), and checks the exact value.
// Uses simple graphs where comm behavior is unambiguous (single child per
// destination proc).
// =============================================================================

BOOST_AUTO_TEST_SUITE(CostFormula)

// All nodes same step → no comm or sync terms. cost = Work[0].
BOOST_AUTO_TEST_CASE(SingleStepSameProc) {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(20, 5, 1);    // v1
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(10);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // Both on P0, step 0. Same proc → no comm.
    schedule.SetAssignedProcessors({0, 0});
    schedule.SetAssignedSupersteps({0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Work[0] = 10 + 20 = 30 on P0. Only 1 step → no comm/sync terms.
    BOOST_CHECK_CLOSE(FreshCost(kl), 30.0, 1e-5);
}

// Two nodes on different procs, same step → comm placed at that step,
// but no subsequent step to pay for it. cost = max work at step 0.
BOOST_AUTO_TEST_CASE(SingleStepDiffProc) {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(20, 5, 1);    // v1
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0 and v1@P1,S0. Comm at S0, but only 1 step.
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Work[0] = max(10, 20) = 20. No step 1 → comm doesn't contribute.
    BOOST_CHECK_CLOSE(FreshCost(kl), 20.0, 1e-5);
}

// Two steps, work dominates comm. g=1, L=0.
BOOST_AUTO_TEST_CASE(TwoStepsWorkDominates) {
    Graph dag;
    dag.AddVertex(10, 1, 1);    // v0: W=10, CW=1
    dag.AddVertex(10, 1, 1);    // v1: W=10, CW=1
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0 → v1@P1,S1.
    // Comm[0] = CW(v0)*SendCosts(0,1) = 1*1 = 1. MaxComm[0] = 1.
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // cost = Work[0] + max(Work[1], MaxComm[0]) * g
    //      = 10 + max(10, 1) * 1 = 20
    BOOST_CHECK_CLOSE(FreshCost(kl), 20.0, 1e-5);
}

// Two steps, comm dominates work. Large VertexCommWeight.
BOOST_AUTO_TEST_CASE(TwoStepsCommDominates) {
    Graph dag;
    dag.AddVertex(10, 100, 1);    // v0: W=10, CW=100
    dag.AddVertex(20, 1, 1);      // v1: W=20, CW=1
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Comm[0] = 100. MaxComm[0] = 100.
    // cost = 10 + max(20, 100) * 1 = 110
    BOOST_CHECK_CLOSE(FreshCost(kl), 110.0, 1e-5);
}

// Two steps, same proc → no communication.
BOOST_AUTO_TEST_CASE(TwoStepsSameProc) {
    Graph dag;
    dag.AddVertex(10, 100, 1);    // v0
    dag.AddVertex(20, 1, 1);      // v1
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // Same proc, different steps → no comm.
    schedule.SetAssignedProcessors({0, 0});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // cost = 10 + max(20, 0) * 1 = 30
    BOOST_CHECK_CLOSE(FreshCost(kl), 30.0, 1e-5);
}

// Test g multiplier. g=3.
BOOST_AUTO_TEST_CASE(TwoStepsWithGMultiplier) {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(20, 1, 1);    // v1
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Comm[0] = 5. MaxComm[0] = 5.
    // cost = 10 + max(20, 5) * 3 = 10 + 60 = 70
    BOOST_CHECK_CLOSE(FreshCost(kl), 70.0, 1e-5);
}

// Test synchronisation costs. L=7.
BOOST_AUTO_TEST_CASE(TwoStepsWithSync) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 1, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(7);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Comm[0] = 5. cost = 10 + max(20, 5)*2 + 1*7 = 10 + 40 + 7 = 57
    BOOST_CHECK_CLOSE(FreshCost(kl), 57.0, 1e-5);
}

// Three-step chain alternating procs. g=1, L=0.
BOOST_AUTO_TEST_CASE(ThreeStepChain) {
    Graph dag;
    dag.AddVertex(10, 5, 1);     // v0: CW=5
    dag.AddVertex(20, 10, 1);    // v1: CW=10
    dag.AddVertex(15, 1, 1);     // v2: CW=1
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0 → v1@P1,S1 → v2@P0,S2
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Comm[0] = 5 (v0→P1). Comm[1] = 10 (v1→P0).
    // cost = 10 + max(20, 5)*1 + max(15, 10)*1 = 10 + 20 + 15 = 45
    BOOST_CHECK_CLOSE(FreshCost(kl), 45.0, 1e-5);
}

// Three steps with sync. g=2, L=5.
BOOST_AUTO_TEST_CASE(ThreeStepChainWithSync) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 10, 1);
    dag.AddVertex(15, 1, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Comm[0]=5, Comm[1]=10.
    // cost = 10 + max(20,5)*2 + max(15,10)*2 + 2*5
    //      = 10 + 40 + 30 + 10 = 90
    BOOST_CHECK_CLOSE(FreshCost(kl), 90.0, 1e-5);
}

// Fan-out: v0 sends to v1 and v2 on same proc P1.
// Eager counts comm once per destination proc.
BOOST_AUTO_TEST_CASE(FanOutSameDestProc) {
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
    // v0@P0,S0 → v1@P1,S1, v2@P1,S1
    schedule.SetAssignedProcessors({0, 1, 1});
    schedule.SetAssignedSupersteps({0, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Eager: v0 has 2 children on P1, lambda count = 2.
    //   AddChild(0, S1) → val=1, returns true → AttributeComm(5)
    //   AddChild(1, S1) → val=2, returns false → no additional comm
    // Comm[0] = send(0,P0) = 5, recv(0,P1) = 5. MaxComm[0] = 5.

    auto &ds = kl.GetCommCostF().commDs_;
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 5.0, 1e-5);

    // Work[0] = 10, Work[1] = max(8, 12) = 12.
    // cost = 10 + max(12, 5)*1 = 22
    BOOST_CHECK_CLOSE(FreshCost(kl), 22.0, 1e-5);
}

// Fan-out to different destination procs.
BOOST_AUTO_TEST_CASE(FanOutDiffDestProc) {
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
    // v0@P0,S0 → v1@P1,S1, v2@P2,S1 (3 procs)
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    auto &ds = kl.GetCommCostF().commDs_;
    // v0 sends to P1 (for v1) and P2 (for v2). Each is a distinct proc.
    // send(0,P0) = 5 + 5 = 10, recv(0,P1) = 5, recv(0,P2) = 5.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 10.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 2), 5.0, 1e-5);

    // MaxComm[0] = 10 (send on P0 dominates).
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 10.0, 1e-5);

    // Work[0]=10, Work[1]=max(8,12)=12.
    // cost = 10 + max(12, 10)*1 = 22
    BOOST_CHECK_CLOSE(FreshCost(kl), 22.0, 1e-5);
}

// Fan-in: two parents on different procs send to one child.
BOOST_AUTO_TEST_CASE(FanIn) {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0: CW=5 on P0
    dag.AddVertex(10, 3, 1);    // v1: CW=3 on P1
    dag.AddVertex(20, 1, 1);    // v2: child
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    auto &ds = kl.GetCommCostF().commDs_;
    // v0@P0 sends to P2: send(0,P0) += 5, recv(0,P2) += 5.
    // v1@P1 sends to P2: send(0,P1) += 3, recv(0,P2) += 3.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 1), 3.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 2), 8.0, 1e-5);

    // MaxComm[0] = max(5, 3, 0, 0, 8) = 8 (recv on P2).
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 8.0, 1e-5);

    // Work[0] = max(10, 10, 0) = 10. Work[1] = 20.
    // cost = 10 + max(20, 8)*1 = 30
    BOOST_CHECK_CLOSE(FreshCost(kl), 30.0, 1e-5);
}

// Diamond: v0 → {v1, v2} → v3. All on different procs.
BOOST_AUTO_TEST_CASE(Diamond) {
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    auto &ds = kl.GetCommCostF().commDs_;
    // Step 0: v0@P0 sends to P1 (v1) and P2 (v2).
    //   send(0,P0) = 5+5 = 10, recv(0,P1) = 5, recv(0,P2) = 5.
    // Step 1: v1@P1 sends to P0 (v3): send(1,P1) = 3, recv(1,P0) = 3.
    //         v2@P2 sends to P0 (v3): send(1,P2) = 4, recv(1,P0) += 4 = 7.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 10.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcSend(1, 1), 3.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcSend(1, 2), 4.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(1, 0), 7.0, 1e-5);

    // MaxComm[0] = 10 (send on P0), MaxComm[1] = 7 (recv on P0).
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 10.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(1), 7.0, 1e-5);

    // Work[0]=10, Work[1]=max(8,12)=12, Work[2]=15.
    // cost = 10 + max(12,10)*1 + max(15,7)*1 = 10 + 12 + 15 = 37
    BOOST_CHECK_CLOSE(FreshCost(kl), 37.0, 1e-5);
}

// Empty step in the middle.
BOOST_AUTO_TEST_CASE(EmptyMiddleStep) {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(20, 1, 1);    // v1
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0, v1@P1,S2. Step 1 is empty.
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 2});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Comm[0] = 5 (v0→P1). Work[0]=10, Work[1]=0, Work[2]=20.
    // cost = 10 + max(0, 5)*1 + max(20, 0)*1 = 10 + 5 + 20 = 35
    BOOST_CHECK_CLOSE(FreshCost(kl), 35.0, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()    // CostFormula

// =============================================================================
// Suite 2: Incremental datastructure validation
//
// After manual moves (applied via KlImproverTest internal mechanisms),
// verify that incremental state matches fresh computation.
// =============================================================================

BOOST_AUTO_TEST_SUITE(IncrementalUpdates)

// Move the only comm-generating node to same proc → comm disappears.
BOOST_AUTO_TEST_CASE(MoveEliminatesComm) {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(20, 3, 1);    // v1
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Initial: cost = 10 + max(20, 5)*1 = 30
    BOOST_CHECK_CLOSE(FreshCost(kl), 30.0, 1e-5);

    // Move v1 from (P1,S1) to (P0,S1). Same proc → no comm.
    kl.InsertGainHeapTest({1});
    kl.RunInnerIterationTest();

    // After KL, validate datastructures match fresh
    BOOST_CHECK(ValidateCommDs(kl, "MoveEliminatesComm"));

    // Incremental cost should match fresh
    double costInc = IncrementalCost(kl);
    double costFresh = FreshCost(kl);
    BOOST_CHECK_CLOSE(costInc, costFresh, 1e-5);
}

// Chain: apply KL to middle node. Verify ds consistency.
BOOST_AUTO_TEST_CASE(ChainMoveMiddle) {
    Graph dag;
    dag.AddVertex(10, 5, 1);     // v0
    dag.AddVertex(20, 10, 1);    // v1
    dag.AddVertex(15, 3, 1);     // v2
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0 → v1@P1,S1 → v2@P0,S2
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Initial cost = 10 + max(20,5)*1 + max(15,10)*1 = 10 + 20 + 15 = 45
    BOOST_CHECK_CLOSE(FreshCost(kl), 45.0, 1e-5);

    // Let KL pick a move for v1
    kl.InsertGainHeapTest({1});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "ChainMoveMiddle"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// Multiple KL iterations sequentially.
BOOST_AUTO_TEST_CASE(SequentialMoves) {
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
    schedule.SetAssignedProcessors({0, 1, 0, 1});
    schedule.SetAssignedSupersteps({0, 1, 2, 3});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Apply several KL iterations, checking consistency after each
    for (unsigned iter = 0; iter < 3; ++iter) {
        // Pick the first unlocked movable node
        std::vector<Graph::VertexIdx> candidates;
        for (unsigned v = 0; v < 4; ++v) {
            candidates.push_back(v);
        }

        kl.InsertGainHeapTest(candidates);
        kl.RunInnerIterationTest();

        BOOST_CHECK_MESSAGE(ValidateCommDs(kl, "SequentialMoves iter " + std::to_string(iter)),
                            "DS mismatch after iteration " << iter);
        BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
    }
}

BOOST_AUTO_TEST_SUITE_END()    // IncrementalUpdates

// =============================================================================
// Suite 3: KL integration – full InsertGainHeapTest + RunInnerIterationTest
//
// These tests verify that KL makes consistent moves: cost_incremental ==
// cost_fresh after each iteration. We don't prescribe which move KL picks.
// =============================================================================

BOOST_AUTO_TEST_SUITE(KlIntegration)

// Corrected version of the original test_max_cost_logic.
// v0(W=10,CW=1)→v1(W=10,CW=1). g=1, L=0.
BOOST_AUTO_TEST_CASE(BasicTwoNodeKl) {
    Graph dag;
    dag.AddVertex(10, 1, 1);
    dag.AddVertex(10, 1, 1);
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Initial: Comm[0]=1, cost = 10 + max(10,1)*1 = 20
    BOOST_CHECK_CLOSE(FreshCost(kl), 20.0, 1e-5);

    kl.InsertGainHeapTest({1});
    kl.RunInnerIterationTest();

    // After KL: verify consistency (don't assume which move was picked)
    BOOST_CHECK(ValidateCommDs(kl, "BasicTwoNodeKl"));
    double afterCost = IncrementalCost(kl);
    BOOST_CHECK_CLOSE(afterCost, FreshCost(kl), 1e-5);
    // Cost should not increase (KL picks best gain)
    BOOST_CHECK_LE(afterCost, 20.0 + 1e-6);
}

// Larger comm weight makes the comm cost significant.
BOOST_AUTO_TEST_CASE(LargeCommWeight) {
    Graph dag;
    dag.AddVertex(10, 50, 1);    // v0: CW=50
    dag.AddVertex(10, 1, 1);     // v1
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Initial: Comm[0]=50. cost = 10 + max(10, 50) = 60
    BOOST_CHECK_CLOSE(FreshCost(kl), 60.0, 1e-5);

    kl.InsertGainHeapTest({1});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "LargeCommWeight"));
    double afterCost = FreshCost(kl);
    // Moving v1 to P0 eliminates comm: cost should decrease
    BOOST_CHECK_LE(afterCost, 60.0 + 1e-6);
    BOOST_CHECK_CLOSE(IncrementalCost(kl), afterCost, 1e-5);
}

// KL on fan-out graph
BOOST_AUTO_TEST_CASE(FanOutKl) {
    Graph dag;
    dag.AddVertex(10, 20, 1);    // v0: CW=20
    dag.AddVertex(8, 1, 1);      // v1
    dag.AddVertex(12, 1, 1);     // v2
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    double initialCost = FreshCost(kl);

    // Move one child
    kl.InsertGainHeapTest({1, 2});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "FanOutKl"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// KL on diamond graph
BOOST_AUTO_TEST_CASE(DiamondKl) {
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    double initialCost = FreshCost(kl);
    // cost = 37 (from CostFormula/Diamond test)
    BOOST_CHECK_CLOSE(initialCost, 37.0, 1e-5);

    // Run KL on all nodes
    kl.InsertGainHeapTest({0, 1, 2, 3});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "DiamondKl"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// Test with 3 procs and 4 steps.
BOOST_AUTO_TEST_CASE(ThreeProcsChain) {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0
    dag.AddVertex(10, 5, 1);    // v1
    dag.AddVertex(10, 5, 1);    // v2
    dag.AddVertex(10, 5, 1);    // v3
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // Spread across procs: v0@P0, v1@P1, v2@P2, v3@P0
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 2, 3});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Comm[0]=5, Comm[1]=5, Comm[2]=5.
    // cost = 10 + max(10,5) + max(10,5) + max(10,5) = 10+10+10+10 = 40
    BOOST_CHECK_CLOSE(FreshCost(kl), 40.0, 1e-5);

    // Run two KL iterations
    for (int i = 0; i < 2; ++i) {
        kl.InsertGainHeapTest({0, 1, 2, 3});
        kl.RunInnerIterationTest();

        BOOST_CHECK(ValidateCommDs(kl, "ThreeProcsChain iter " + std::to_string(i)));
        BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
    }
}

BOOST_AUTO_TEST_SUITE_END()    // KlIntegration

// =============================================================================
// Suite 4: Larger graphs and edge cases
// =============================================================================

BOOST_AUTO_TEST_SUITE(LargerGraphs)

// Butterfly/bipartite: 2 sources, 2 sinks, all cross-edges.
BOOST_AUTO_TEST_CASE(ButterflyKl) {
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    auto &ds = kl.GetCommCostF().commDs_;
    // v0@P0 → v2@P1: lambda(v0,P1). AddChild(0,S1)→val=1, true. cost=5.
    // v0@P0 → v3@P0: same proc, no comm.
    // v1@P1 → v2@P1: same proc, no comm.
    // v1@P1 → v3@P0: lambda(v1,P0). AddChild(0,S1)→val=1, true. cost=3.
    // send(0,P0) = 5, recv(0,P1) = 5.
    // send(0,P1) = 3, recv(0,P0) = 3.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 1), 3.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 0), 3.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 5.0, 1e-5);

    // MaxComm[0] = max(5, 3, 3, 5) = 5.
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 5.0, 1e-5);

    double initialCost = FreshCost(kl);
    // Work[0]=max(10,10)=10, Work[1]=max(10,10)=10.
    // cost = 10 + max(10, 5)*1 = 20
    BOOST_CHECK_CLOSE(initialCost, 20.0, 1e-5);

    // Run KL
    kl.InsertGainHeapTest({0, 1, 2, 3});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "ButterflyKl"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// Wider graph: 6 nodes, 3 procs, 3 steps.
BOOST_AUTO_TEST_CASE(WiderGraph) {
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    double initialCost = FreshCost(kl);

    // Run 3 KL iterations
    for (int i = 0; i < 3; ++i) {
        kl.InsertGainHeapTest({0, 1, 2, 3, 4, 5});
        kl.RunInnerIterationTest();

        BOOST_CHECK_MESSAGE(ValidateCommDs(kl, "WiderGraph iter " + std::to_string(i)), "DS mismatch at iteration " << i);
        BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
    }
}

// No communication edges at all.
BOOST_AUTO_TEST_CASE(NoCommunication) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 3, 1);
    // No edges!

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // No comm. cost = 10 + max(20, 0)*1 + 1*5 = 35
    BOOST_CHECK_CLOSE(FreshCost(kl), 35.0, 1e-5);

    kl.InsertGainHeapTest({0, 1});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "NoCommunication"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// Consistency: ComputeScheduleCostTest == ComputeScheduleCost<true> at init.
BOOST_AUTO_TEST_CASE(CostConsistentAtInit) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 10, 1);
    dag.AddVertex(15, 3, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Both should agree right after initialization
    double costTest = IncrementalCost(kl);
    double costFresh = FreshCost(kl);
    BOOST_CHECK_CLOSE(costTest, costFresh, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()    // LargerGraphs
