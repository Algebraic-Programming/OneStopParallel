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
 *   Work[s]    = max over processors of (sum of VertexWorkWeight for nodes
 *                assigned to that (proc, step))
 *   MaxComm[s] = max over all procs of max(send(s,p), recv(s,p))
 *   g          = CommunicationCosts()  (bandwidth parameter)
 *   L          = SynchronisationCosts() (latency parameter)
 *
 * Communication at step s, proc p (Eager policy):
 *   For each parent u at step s on proc p, comm is counted ONCE per distinct
 *   destination proc q (AddChild returns true only when lambda count goes 0→1):
 *     send(s, p) += VertexCommWeight(u) * SendCosts(p, q)
 *     recv(s, q) += VertexCommWeight(u) * SendCosts(p, q)
 *
 * Note: Edge weights are NOT used — BSP comm cost uses VertexCommWeight.
 *
 * Structure:
 *   Suite 1 – CostFormula: verify ComputeScheduleCost against hand-computed values
 *   Suite 2 – DirectDatastructureTests: manual moves on MaxCommDatastructure,
 *             compare incremental vs fresh
 *   Suite 3 – KlIntegration: through KlImproverTest, verify consistency
 *   Suite 4 – LargerGraphs: bigger topologies, single-iteration KL
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

/// Validate incremental comm datastructures match a fresh computation.
static bool ValidateCommDs(KlImproverTestT &kl, const std::string &context) {
    auto &costF = kl.GetCommCostF();
    auto *activeSched = costF.active_schedule;
    const auto *inst = costF.instance;
    const auto &dsInc = costF.commDs_;

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

/// TestSetup for direct MaxCommDatastructure tests (no KL overhead).
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

/// Validate a MaxCommDatastructure against fresh computation.
static bool ValidateDirectDs(const MaxCommDatastructure<Graph, double, KlActiveScheduleT> &dsInc,
                             TestSetup &t,
                             const std::string &context) {
    BspSchedule<Graph> currentSchedule(*t.instance);
    t.klSched->WriteSchedule(currentSchedule);

    KlActiveScheduleT klSchedFresh;
    klSchedFresh.Initialize(currentSchedule);

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> dsFresh;
    dsFresh.Initialize(klSchedFresh);
    unsigned maxStep = currentSchedule.NumberOfSupersteps();
    dsFresh.ComputeCommDatastructures(0, maxStep > 0 ? maxStep - 1 : 0);

    bool allMatch = true;
    for (unsigned step = 0; step < maxStep; ++step) {
        for (unsigned p = 0; p < t.instance->NumberOfProcessors(); ++p) {
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
// Suite 1: CostFormula – verify ComputeScheduleCost against hand-computed values
// =============================================================================

BOOST_AUTO_TEST_SUITE(CostFormula)

// All nodes same step, same proc → no comm or sync. cost = Work[0].
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
    schedule.SetAssignedProcessors({0, 0});
    schedule.SetAssignedSupersteps({0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Work[0] = 10+20 = 30 on P0. 1 step → no comm/sync.
    BOOST_CHECK_CLOSE(FreshCost(kl), 30.0, 1e-5);
}

// Two nodes on different procs, same step → comm exists but no later step to
// pay for it.
BOOST_AUTO_TEST_CASE(SingleStepDiffProc) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 5, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Work[0] = max(10, 20) = 20. No step 1 → comm doesn't contribute.
    BOOST_CHECK_CLOSE(FreshCost(kl), 20.0, 1e-5);
}

// Two steps. Work dominates comm. g=1, L=0.
BOOST_AUTO_TEST_CASE(TwoStepsWorkDominates) {
    Graph dag;
    dag.AddVertex(10, 1, 1);    // v0: CW=1
    dag.AddVertex(10, 1, 1);    // v1: CW=1
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

    // Comm[0] = CW(v0)*SendCosts(0,1) = 1. MaxComm[0]=1.
    // cost = 10 + max(10, 1)*1 = 20
    BOOST_CHECK_CLOSE(FreshCost(kl), 20.0, 1e-5);
}

// Two steps. Comm dominates work. Large VertexCommWeight.
BOOST_AUTO_TEST_CASE(TwoStepsCommDominates) {
    Graph dag;
    dag.AddVertex(10, 100, 1);    // v0: CW=100
    dag.AddVertex(20, 1, 1);
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

    // Comm[0] = 100. cost = 10 + max(20, 100)*1 = 110
    BOOST_CHECK_CLOSE(FreshCost(kl), 110.0, 1e-5);
}

// Two steps, same proc → no communication.
BOOST_AUTO_TEST_CASE(TwoStepsSameProc) {
    Graph dag;
    dag.AddVertex(10, 100, 1);
    dag.AddVertex(20, 1, 1);
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // No comm. cost = 10 + max(20, 0)*1 = 30
    BOOST_CHECK_CLOSE(FreshCost(kl), 30.0, 1e-5);
}

// g multiplier (g=3).
BOOST_AUTO_TEST_CASE(TwoStepsWithGMultiplier) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 1, 1);
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

    // Comm[0]=5. cost = 10 + max(20, 5)*3 = 10 + 60 = 70
    BOOST_CHECK_CLOSE(FreshCost(kl), 70.0, 1e-5);
}

// Synchronisation costs (L=7).
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

    // Comm[0]=5. cost = 10 + max(20, 5)*2 + 1*7 = 57
    BOOST_CHECK_CLOSE(FreshCost(kl), 57.0, 1e-5);
}

// Three-step chain, alternating procs.
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
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Comm[0]=5 (v0→P1). Comm[1]=10 (v1→P0).
    // cost = 10 + max(20, 5) + max(15, 10) = 10 + 20 + 15 = 45
    BOOST_CHECK_CLOSE(FreshCost(kl), 45.0, 1e-5);
}

// Three steps with g=2, L=5.
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

    // cost = 10 + max(20,5)*2 + max(15,10)*2 + 2*5 = 10 + 40 + 30 + 10 = 90
    BOOST_CHECK_CLOSE(FreshCost(kl), 90.0, 1e-5);
}

// Fan-out to SAME destination proc → Eager counts comm ONCE (AddChild returns
// true only on first child for a given dest proc).
BOOST_AUTO_TEST_CASE(FanOutSameDestProc) {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0: CW=5
    dag.AddVertex(8, 1, 1);     // v1: W=8
    dag.AddVertex(12, 1, 1);    // v2: W=12
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

    auto &ds = kl.GetCommCostF().commDs_;
    // Eager: AddChild returns true on first child (0→1), false on second (1→2).
    // Comm counted once: send(0,P0) = 5, recv(0,P1) = 5.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 5.0, 1e-5);

    // Work[0] = 10 (v0 on P0).
    // Work[1] = 8 + 12 = 20 (v1 + v2 BOTH on P1, work is summed per proc).
    // cost = 10 + max(20, 5)*1 = 30
    BOOST_CHECK_CLOSE(FreshCost(kl), 30.0, 1e-5);
}

// Fan-out to DIFFERENT destination procs → comm counted per dest.
BOOST_AUTO_TEST_CASE(FanOutDiffDestProc) {
    Graph dag;
    dag.AddVertex(10, 5, 1);    // v0: CW=5
    dag.AddVertex(8, 1, 1);     // v1: W=8
    dag.AddVertex(12, 1, 1);    // v2: W=12
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
    // Two distinct dest procs → AddChild returns true twice.
    // send(0,P0) = 5 + 5 = 10, recv(0,P1) = 5, recv(0,P2) = 5.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 10.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 2), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 10.0, 1e-5);

    // Work[0] = 10, Work[1] = max(8, 12) = 12 (v1 on P1, v2 on P2, separate procs).
    // cost = 10 + max(12, 10)*1 = 22
    BOOST_CHECK_CLOSE(FreshCost(kl), 22.0, 1e-5);
}

// Fan-in: two parents from different procs send to one child.
BOOST_AUTO_TEST_CASE(FanIn) {
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

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    auto &ds = kl.GetCommCostF().commDs_;
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 1), 3.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 2), 8.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 8.0, 1e-5);

    // Work[0] = max(10, 10) = 10. Work[1] = 20.
    // cost = 10 + max(20, 8)*1 = 30
    BOOST_CHECK_CLOSE(FreshCost(kl), 30.0, 1e-5);
}

// Diamond: v0 → {v1, v2} → v3. 3 procs.
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
    // S0: v0@P0→P1(v1): 5, v0@P0→P2(v2): 5. send(0,P0)=10.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 10.0, 1e-5);
    // S1: v1@P1→P0(v3): 3, v2@P2→P0(v3): 4.
    BOOST_CHECK_CLOSE(ds.StepProcSend(1, 1), 3.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcSend(1, 2), 4.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(1, 0), 7.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 10.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(1), 7.0, 1e-5);

    // Work[0]=10, Work[1]=max(8,12)=12 (diff procs), Work[2]=15.
    // cost = 10 + max(12,10) + max(15,7) = 10+12+15 = 37
    BOOST_CHECK_CLOSE(FreshCost(kl), 37.0, 1e-5);
}

// Empty step in the middle.
BOOST_AUTO_TEST_CASE(EmptyMiddleStep) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 1, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // v0@P0,S0, v1@P1,S2. S1 is empty.
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 2});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Comm[0]=5. Work[0]=10, Work[1]=0, Work[2]=20.
    // cost = 10 + max(0, 5) + max(20, 0) = 35
    BOOST_CHECK_CLOSE(FreshCost(kl), 35.0, 1e-5);
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

    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()    // CostFormula

// =============================================================================
// Suite 2: DirectDatastructureTests – manual moves on MaxCommDatastructure
//
// Uses TestSetup to construct a schedule + MaxCommDatastructure directly, applies
// moves via Apply(), then validates incremental vs fresh.
// =============================================================================

BOOST_AUTO_TEST_SUITE(DirectDatastructureTests)

// Initial state: chain v0→v1→v2 across procs.
BOOST_AUTO_TEST_CASE(InitialCommDataChain) {
    TestSetup t;
    t.dag.AddVertex(10, 5, 1);     // v0: CW=5
    t.dag.AddVertex(20, 10, 1);    // v1: CW=10
    t.dag.AddVertex(15, 3, 1);     // v2: CW=3
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(1, 2, 1);
    t.arch.SetNumberOfProcessors(2);
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(0);
    t.Build({0, 1, 0}, {0, 1, 2});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 2);

    // S0: v0@P0→v1@P1. send(0,P0)=5, recv(0,P1)=5.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 5.0, 1e-5);

    // S1: v1@P1→v2@P0. send(1,P1)=10, recv(1,P0)=10.
    BOOST_CHECK_CLOSE(ds.StepProcSend(1, 1), 10.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(1, 0), 10.0, 1e-5);

    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(1), 10.0, 1e-5);
}

// Manual move: v1 from (P1,S1) → (P0,S1). Comm disappears from S0.
BOOST_AUTO_TEST_CASE(ManualMoveRemovesComm) {
    TestSetup t;
    t.dag.AddVertex(10, 5, 1);     // v0: CW=5
    t.dag.AddVertex(20, 10, 1);    // v1: CW=10
    t.dag.AddEdge(0, 1, 1);
    t.arch.SetNumberOfProcessors(2);
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(0);
    t.Build({0, 1}, {0, 1});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 1);

    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 5.0, 1e-5);

    // Move v1 from (P1,S1) to (P0,S1).
    KlMove m(1, 0.0, 1, 1, 0, 1);
    t.Apply(m);
    ds.UpdateDatastructureAfterMove(m, 0, 1);

    // All comm gone.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 0.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 0.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 0.0, 1e-5);
    BOOST_CHECK(ValidateDirectDs(ds, t, "ManualMoveRemovesComm"));
}

// Manual move: v1 from (P0,S1) → (P1,S1). Introduces cross-proc comm.
BOOST_AUTO_TEST_CASE(ManualMoveIntroducesComm) {
    TestSetup t;
    t.dag.AddVertex(10, 5, 1);     // v0: CW=5
    t.dag.AddVertex(20, 10, 1);    // v1: CW=10
    t.dag.AddEdge(0, 1, 1);
    t.arch.SetNumberOfProcessors(2);
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(0);
    t.Build({0, 0}, {0, 1});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 1);

    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 0.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 0.0, 1e-5);

    // Move v1 from (P0,S1) to (P1,S1).
    KlMove m(1, 0.0, 0, 1, 1, 1);
    t.Apply(m);
    ds.UpdateDatastructureAfterMove(m, 0, 1);

    // send(0,P0) = 5, recv(0,P1) = 5.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 5.0, 1e-5);
    BOOST_CHECK(ValidateDirectDs(ds, t, "ManualMoveIntroducesComm"));
}

// Manual move on chain: two sequential moves.
BOOST_AUTO_TEST_CASE(ManualMoveChain) {
    TestSetup t;
    t.dag.AddVertex(10, 5, 1);     // v0: CW=5
    t.dag.AddVertex(20, 10, 1);    // v1: CW=10
    t.dag.AddVertex(15, 3, 1);     // v2: CW=3
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(1, 2, 1);
    t.arch.SetNumberOfProcessors(2);
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(0);
    t.Build({0, 1, 0}, {0, 1, 2});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 2);

    // Move 1: v1 from (P1,S1) to (P0,S1). Kills comm on S0 (v0→v1 now same proc).
    // S1: v1@P0→v2@P0: same proc → no comm either. All comm goes to 0.
    KlMove m1(1, 0.0, 1, 1, 0, 1);
    t.Apply(m1);
    ds.UpdateDatastructureAfterMove(m1, 0, 2);

    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 0.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcSend(1, 1), 0.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 0.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(1), 0.0, 1e-5);
    BOOST_CHECK(ValidateDirectDs(ds, t, "ManualMoveChain after m1"));

    // Move 2: v2 from (P0,S2) to (P1,S2). Adds comm on S1 (v1@P0→v2@P1).
    KlMove m2(2, 0.0, 0, 2, 1, 2);
    t.Apply(m2);
    ds.UpdateDatastructureAfterMove(m2, 0, 2);

    // send(1,P0) = 10 (v1@P0 → v2@P1). recv(1,P1) = 10.
    BOOST_CHECK_CLOSE(ds.StepProcSend(1, 0), 10.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(1, 1), 10.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(1), 10.0, 1e-5);
    BOOST_CHECK(ValidateDirectDs(ds, t, "ManualMoveChain after m2"));
}

// Manual move on diamond graph.
BOOST_AUTO_TEST_CASE(ManualMoveDiamond) {
    TestSetup t;
    t.dag.AddVertex(10, 5, 1);    // v0
    t.dag.AddVertex(8, 3, 1);     // v1
    t.dag.AddVertex(12, 4, 1);    // v2
    t.dag.AddVertex(15, 1, 1);    // v3
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);
    t.dag.AddEdge(1, 3, 1);
    t.dag.AddEdge(2, 3, 1);
    t.arch.SetNumberOfProcessors(3);
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(0);
    // v0@P0,S0, v1@P1,S1, v2@P2,S1, v3@P0,S2
    t.Build({0, 1, 2, 0}, {0, 1, 1, 2});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 2);

    // Initial state verified in CostFormula/Diamond.
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 10.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(1), 7.0, 1e-5);

    // Move v1 from (P1,S1) to (P0,S1).
    // S0: v0@P0→P2(v2) still sends. v0@P0→P0(v1) now same proc → remove.
    //   send(0,P0) = 5 (only to P2). recv(0,P2)=5.
    // S1: v1@P0→P0(v3) same proc → no comm. v2@P2→P0(v3) still sends.
    //   send(1,P2) = 4, recv(1,P0) = 4.
    KlMove m(1, 0.0, 1, 1, 0, 1);
    t.Apply(m);
    ds.UpdateDatastructureAfterMove(m, 0, 2);

    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);       // v0→P2 only
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 0.0, 1e-5);    // no recv on P1
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 2), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcSend(1, 2), 4.0, 1e-5);       // v2→P0
    BOOST_CHECK_CLOSE(ds.StepProcReceive(1, 0), 4.0, 1e-5);    // was 7, now 4
    BOOST_CHECK(ValidateDirectDs(ds, t, "ManualMoveDiamond"));
}

// Fan-out: partial move (one of two children). Lambda count goes 2→1,
// Eager: no comm change (RemoveChild returns false when val != 0).
BOOST_AUTO_TEST_CASE(ManualMoveFanOutPartial) {
    TestSetup t;
    t.dag.AddVertex(10, 5, 1);    // v0: CW=5
    t.dag.AddVertex(8, 1, 1);     // v1
    t.dag.AddVertex(12, 1, 1);    // v2
    t.dag.AddEdge(0, 1, 1);
    t.dag.AddEdge(0, 2, 1);
    t.arch.SetNumberOfProcessors(2);
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(0);
    // v0@P0,S0, v1@P1,S1, v2@P1,S1
    t.Build({0, 1, 1}, {0, 1, 1});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 1);

    // Initial: send(0,P0)=5 (once to P1). Lambda[v0][P1]=2.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);

    // Move v2 from (P1,S1) to (P0,S1).
    // Lambda[v0][P1] goes 2→1. RemoveChild returns false (val != 0).
    // No UnattributeCommunication → send(0,P0) still = 5.
    // v0→P0 for v2: same proc → no new comm.
    KlMove m(2, 0.0, 1, 1, 0, 1);
    t.Apply(m);
    ds.UpdateDatastructureAfterMove(m, 0, 1);

    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 5.0, 1e-5);
    BOOST_CHECK(ValidateDirectDs(ds, t, "ManualMoveFanOutPartial"));
}

// Move LAST child off dest proc → comm disappears.
BOOST_AUTO_TEST_CASE(ManualMoveFanOutLast) {
    TestSetup t;
    t.dag.AddVertex(10, 5, 1);    // v0: CW=5
    t.dag.AddVertex(8, 1, 1);     // v1
    t.dag.AddEdge(0, 1, 1);
    t.arch.SetNumberOfProcessors(2);
    t.arch.SetCommunicationCosts(1);
    t.arch.SetSynchronisationCosts(0);
    // v0@P0,S0, v1@P1,S1. Lambda[v0][P1]=1.
    t.Build({0, 1}, {0, 1});

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> ds;
    ds.Initialize(*t.klSched);
    ds.ComputeCommDatastructures(0, 1);

    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);

    // Move v1 from (P1,S1) to (P0,S1). Lambda[v0][P1]=1→0. RemoveChild true.
    KlMove m(1, 0.0, 1, 1, 0, 1);
    t.Apply(m);
    ds.UpdateDatastructureAfterMove(m, 0, 1);

    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 0.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcReceive(0, 1), 0.0, 1e-5);
    BOOST_CHECK(ValidateDirectDs(ds, t, "ManualMoveFanOutLast"));
}

BOOST_AUTO_TEST_SUITE_END()    // DirectDatastructureTests

// =============================================================================
// Suite 3: KL Integration – through KlImproverTest
//
// InsertGainHeapTest + RunInnerIterationTest. After each iteration, verify
// incremental cost equals fresh cost. We don't prescribe which move KL picks.
// =============================================================================

BOOST_AUTO_TEST_SUITE(KlIntegration)

// Basic two-node graph. Corrected version of original test.
BOOST_AUTO_TEST_CASE(BasicTwoNode) {
    Graph dag;
    dag.AddVertex(10, 1, 1);    // CW=1
    dag.AddVertex(10, 1, 1);    // CW=1
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

    // Initial: Comm[0]=1. cost = 10 + max(10, 1) = 20
    BOOST_CHECK_CLOSE(FreshCost(kl), 20.0, 1e-5);

    kl.InsertGainHeapTest({1});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "BasicTwoNode"));
    double afterCost = IncrementalCost(kl);
    BOOST_CHECK_CLOSE(afterCost, FreshCost(kl), 1e-5);
    BOOST_CHECK_LE(afterCost, 20.0 + 1e-6);
}

// Large comm weight makes comm dominate → KL should improve.
BOOST_AUTO_TEST_CASE(LargeCommWeight) {
    Graph dag;
    dag.AddVertex(10, 50, 1);    // CW=50
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

    // Initial: Comm[0]=50. cost = 10 + max(10, 50) = 60
    BOOST_CHECK_CLOSE(FreshCost(kl), 60.0, 1e-5);

    kl.InsertGainHeapTest({1});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "LargeCommWeight"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// Chain with middle node move.
BOOST_AUTO_TEST_CASE(ChainMoveMiddle) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 10, 1);
    dag.AddVertex(15, 3, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK_CLOSE(FreshCost(kl), 45.0, 1e-5);

    kl.InsertGainHeapTest({1});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "ChainMoveMiddle"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// Fan-out KL: insert both children.
BOOST_AUTO_TEST_CASE(FanOutKl) {
    Graph dag;
    dag.AddVertex(10, 20, 1);    // CW=20
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
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    kl.InsertGainHeapTest({1, 2});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "FanOutKl"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// Diamond KL.
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

    BOOST_CHECK_CLOSE(FreshCost(kl), 37.0, 1e-5);

    // Insert all, run one iteration
    kl.InsertGainHeapTest({0, 1, 2, 3});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "DiamondKl"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// Insert all nodes once, run TWO iterations to test consecutive pops from heap.
BOOST_AUTO_TEST_CASE(MultipleInnerIterations) {
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
    schedule.SetAssignedProcessors({0, 1, 0, 1});
    schedule.SetAssignedSupersteps({0, 1, 2, 3});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK_CLOSE(FreshCost(kl), 40.0, 1e-5);

    // Insert all once, then pop multiple times (stale gains OK in KL).
    kl.InsertGainHeapTest({0, 1, 2, 3});

    kl.RunInnerIterationTest();
    BOOST_CHECK(ValidateCommDs(kl, "MultipleInnerIterations iter 0"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);

    kl.RunInnerIterationTest();
    BOOST_CHECK(ValidateCommDs(kl, "MultipleInnerIterations iter 1"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()    // KlIntegration

// =============================================================================
// Suite 4: Larger graphs and edge cases
// =============================================================================

BOOST_AUTO_TEST_SUITE(LargerGraphs)

// Butterfly (bipartite): v0,v1 → v2,v3 with cross-edges.
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
    // v0@P0→v2@P1: send(0,P0)+=5, recv(0,P1)+=5. v0@P0→v3@P0: same proc.
    // v1@P1→v2@P1: same proc. v1@P1→v3@P0: send(0,P1)+=3, recv(0,P0)+=3.
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 0), 5.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepProcSend(0, 1), 3.0, 1e-5);
    BOOST_CHECK_CLOSE(ds.StepMaxComm(0), 5.0, 1e-5);

    // Work[0]=max(10,10)=10, Work[1]=max(10,10)=10.
    // cost = 10 + max(10, 5)*1 = 20
    BOOST_CHECK_CLOSE(FreshCost(kl), 20.0, 1e-5);

    kl.InsertGainHeapTest({0, 1, 2, 3});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "ButterflyKl"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// 6-node wider graph, 3 procs, 3 steps. Single KL iteration.
BOOST_AUTO_TEST_CASE(WiderGraphSingleIteration) {
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

    kl.InsertGainHeapTest({0, 1, 2, 3, 4, 5});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "WiderGraphSingleIteration"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

// No edges at all.
BOOST_AUTO_TEST_CASE(NoCommunication) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 3, 1);

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

// 3 procs chain. Single iteration only.
BOOST_AUTO_TEST_CASE(ThreeProcsChainSingleIteration) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(10, 5, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(0);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 2, 3});
    schedule.UpdateNumberOfSupersteps();

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    // Comm[0]=5, Comm[1]=5, Comm[2]=5.
    // cost = 10 + max(10,5) + max(10,5) + max(10,5) = 40
    BOOST_CHECK_CLOSE(FreshCost(kl), 40.0, 1e-5);

    kl.InsertGainHeapTest({0, 1, 2, 3});
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDs(kl, "ThreeProcsChainSingleIteration"));
    BOOST_CHECK_CLOSE(IncrementalCost(kl), FreshCost(kl), 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()    // LargerGraphs
