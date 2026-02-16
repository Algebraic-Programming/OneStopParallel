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

#define BOOST_TEST_MODULE kl_cross_step_work_affinity
#include <boost/test/unit_test.hpp>
#include <set>

#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_test.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

template <typename TableT>
void CheckEqualAffinityTable(TableT &table1, TableT &table2, const std::set<size_t> &nodes) {
    for (auto i : nodes) {
        BOOST_CHECK_EQUAL(table1[i].size(), table2[i].size());
        if (table1[i].size() != table2[i].size()) {
            continue;
        }
        for (size_t j = 0; j < table1[i].size(); ++j) {
            BOOST_CHECK_EQUAL(table1[i][j].size(), table2[i][j].size());
            if (table1[i][j].size() != table2[i][j].size()) {
                continue;
            }
            for (size_t k = 0; k < table1[i][j].size(); ++k) {
                BOOST_CHECK(std::abs(table1[i][j][k] - table2[i][j][k]) < 0.000001);

                if (std::abs(table1[i][j][k] - table2[i][j][k]) > 0.000001) {
                    std::cout << "Mismatch at node[" << i << "] proc[" << j << "] step_idx[" << k
                              << "]: incremental=" << table1[i][j][k] << ", fresh=" << table2[i][j][k] << std::endl;
                }
            }
        }
    }
}

/**
 * Test that ProcessWorkUpdateStep handles cross-step moves correctly.
 *
 * This test applies a move where fromStep != toStep AND fromProc != toProc,
 * then verifies that the incrementally updated work affinities match
 * freshly computed ones for all active nodes.
 *
 * The key concern is whether nodes on the toProc at the fromStep (or fromProc
 * at the toStep) have their affinities updated correctly, since
 * ProcessWorkUpdateStep only receives one (moveStep, moveProc) pair per call.
 */
BOOST_AUTO_TEST_CASE(CrossStepCrossProcMoveWorkAffinityTest) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;
    using KlMove = KlMoveStruct<double, VertexType>;

    Graph dag;

    // Create 8 nodes with varying work weights to create interesting max-work scenarios
    const VertexType v0 = dag.AddVertex(10, 1, 2);    // work=10
    const VertexType v1 = dag.AddVertex(8, 1, 3);     // work=8
    const VertexType v2 = dag.AddVertex(6, 1, 4);     // work=6
    const VertexType v3 = dag.AddVertex(12, 1, 2);    // work=12
    const VertexType v4 = dag.AddVertex(5, 1, 5);     // work=5
    const VertexType v5 = dag.AddVertex(7, 1, 3);     // work=7
    const VertexType v6 = dag.AddVertex(9, 1, 2);     // work=9
    const VertexType v7 = dag.AddVertex(4, 1, 1);     // work=4

    // Simple edges
    dag.AddEdge(v0, v2, 2);
    dag.AddEdge(v0, v3, 3);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v1, v5, 4);
    dag.AddEdge(v2, v6, 5);
    dag.AddEdge(v3, v7, 3);
    dag.AddEdge(v4, v6, 2);
    dag.AddEdge(v5, v7, 6);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);

    // Spread nodes across 3 supersteps on different processors:
    //   Step 0: v0(p0, w=10), v1(p1, w=8)
    //   Step 1: v2(p0, w=6),  v3(p1, w=12), v4(p2, w=5)
    //   Step 2: v5(p0, w=7),  v6(p1, w=9),  v7(p2, w=4)
    schedule.SetAssignedProcessors({0, 1, 0, 1, 2, 0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 1, 2, 2, 2});
    schedule.UpdateNumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    using KlImproverTestT = KlImproverTest<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double>;

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    std::vector<VertexType> allNodes = {0, 1, 2, 3, 4, 5, 6, 7};
    auto nodeSelection = kl.InsertGainHeapTest(allNodes);
    auto &affinity = kl.GetAffinityTable();

    // Move v3 (work=12) from proc 1, step 1 to proc 2, step 2
    // This is a cross-step, cross-proc move.
    // At fromStep (step 1): proc 1 loses 12 work, changing max from 12 to 6
    // At toStep (step 2): proc 2 gains 12 work, changing max from 9 to 16
    KlMove move1(v3, 0.0, 1, 1, 2, 2);
    kl.UpdateAffinityTableTest(move1, nodeSelection);

    // Build a fresh KlImproverTest from the current schedule state, compute all affinities from scratch
    BspSchedule<Graph> postMoveSchedule(instance);
    kl.GetActiveScheduleTest(postMoveSchedule);
    KlImproverTestT klFresh;
    klFresh.SetupSchedule(postMoveSchedule);
    klFresh.InsertGainHeapTest(allNodes);

    std::set<size_t> nodesToCheck = {0, 1, 2, 4, 5, 6, 7};    // exclude v3 (moved node)

    CheckEqualAffinityTable(affinity, klFresh.GetAffinityTable(), nodesToCheck);
}

/**
 * Test cross-step move where the moved node's toProc already has nodes at fromStep.
 *
 * This specifically targets the scenario where at fromStep, toProc has existing
 * nodes whose affinity needs correct handling through the generic (non-moveProc)
 * branch of ProcessWorkUpdateStep.
 */
BOOST_AUTO_TEST_CASE(CrossStepMoveWithToProcAtFromStepTest) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;
    using KlMove = KlMoveStruct<double, VertexType>;

    Graph dag;

    const VertexType v0 = dag.AddVertex(10, 1, 2);
    const VertexType v1 = dag.AddVertex(15, 1, 3);
    const VertexType v2 = dag.AddVertex(8, 1, 4);
    const VertexType v3 = dag.AddVertex(12, 1, 2);
    const VertexType v4 = dag.AddVertex(7, 1, 5);
    const VertexType v5 = dag.AddVertex(20, 1, 3);

    dag.AddEdge(v0, v2, 2);
    dag.AddEdge(v0, v3, 3);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v1, v5, 4);
    dag.AddEdge(v2, v4, 5);
    dag.AddEdge(v3, v5, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule schedule(instance);

    // Step 0: v0(p0, w=10), v1(p2, w=15)
    // Step 1: v2(p0, w=8),  v3(p2, w=12)
    // Step 2: v4(p1, w=7),  v5(p0, w=20)
    //
    // Move v2 from (p0, step1) to (p2, step2).
    // At fromStep=1, toProc=p2 has v3 -- this tests that v3's affinity
    // is updated correctly via the generic branch.
    schedule.SetAssignedProcessors({0, 2, 0, 2, 1, 0});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2});
    schedule.UpdateNumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    using KlImproverTestT = KlImproverTest<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double>;

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    std::vector<VertexType> allNodes = {0, 1, 2, 3, 4, 5};
    auto nodeSelection = kl.InsertGainHeapTest(allNodes);
    auto &affinity = kl.GetAffinityTable();

    KlMove move(v2, 0.0, 0, 1, 2, 2);
    kl.UpdateAffinityTableTest(move, nodeSelection);

    BspSchedule<Graph> postMoveSchedule(instance);
    kl.GetActiveScheduleTest(postMoveSchedule);
    KlImproverTestT klFresh;
    klFresh.SetupSchedule(postMoveSchedule);
    klFresh.InsertGainHeapTest(allNodes);

    std::set<size_t> nodesToCheck = {0, 1, 3, 4, 5};    // exclude v2 (moved node)

    CheckEqualAffinityTable(affinity, klFresh.GetAffinityTable(), nodesToCheck);
}

/**
 * Test a sequence of cross-step moves to verify accumulation of incremental updates.
 */
BOOST_AUTO_TEST_CASE(SequentialCrossStepMovesTest) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;
    using KlMove = KlMoveStruct<double, VertexType>;

    Graph dag;

    const VertexType v0 = dag.AddVertex(10, 1, 2);
    const VertexType v1 = dag.AddVertex(8, 1, 3);
    const VertexType v2 = dag.AddVertex(6, 1, 4);
    const VertexType v3 = dag.AddVertex(12, 1, 2);
    const VertexType v4 = dag.AddVertex(5, 1, 5);
    const VertexType v5 = dag.AddVertex(7, 1, 3);
    const VertexType v6 = dag.AddVertex(9, 1, 2);
    const VertexType v7 = dag.AddVertex(4, 1, 1);

    dag.AddEdge(v0, v2, 2);
    dag.AddEdge(v0, v3, 3);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v1, v5, 4);
    dag.AddEdge(v2, v6, 5);
    dag.AddEdge(v3, v7, 3);
    dag.AddEdge(v4, v6, 2);
    dag.AddEdge(v5, v7, 6);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule schedule(instance);

    schedule.SetAssignedProcessors({0, 1, 0, 1, 2, 0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 1, 2, 2, 2});
    schedule.UpdateNumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    using KlImproverTestT = KlImproverTest<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double>;

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    std::vector<VertexType> allNodes = {0, 1, 2, 3, 4, 5, 6, 7};
    auto nodeSelection = kl.InsertGainHeapTest(allNodes);
    auto &affinity = kl.GetAffinityTable();
    std::set<size_t> nodesToCheck = {0, 1, 2, 3, 4, 5, 6, 7};

    // Move 1: v3 from (p1, step1) to (p2, step2) -- cross-step, cross-proc
    KlMove move1(v3, 0.0, 1, 1, 2, 2);
    kl.UpdateAffinityTableTest(move1, nodeSelection);
    nodesToCheck.erase(v3);

    {
        BspSchedule<Graph> tmpSchedule(instance);
        kl.GetActiveScheduleTest(tmpSchedule);
        KlImproverTestT klFresh;
        klFresh.SetupSchedule(tmpSchedule);
        klFresh.InsertGainHeapTest(allNodes);
        CheckEqualAffinityTable(affinity, klFresh.GetAffinityTable(), nodesToCheck);
    }

    // Move 2: v1 from (p1, step0) to (p0, step1) -- cross-step, cross-proc
    KlMove move2(v1, 0.0, 1, 0, 0, 1);
    kl.UpdateAffinityTableTest(move2, nodeSelection);
    nodesToCheck.erase(v1);

    {
        BspSchedule<Graph> tmpSchedule(instance);
        kl.GetActiveScheduleTest(tmpSchedule);
        KlImproverTestT klFresh;
        klFresh.SetupSchedule(tmpSchedule);
        klFresh.InsertGainHeapTest(allNodes);
        CheckEqualAffinityTable(affinity, klFresh.GetAffinityTable(), nodesToCheck);
    }

    // Move 3: v6 from (p1, step2) to (p2, step1) -- cross-step, cross-proc (moving backward)
    KlMove move3(v6, 0.0, 1, 2, 2, 1);
    kl.UpdateAffinityTableTest(move3, nodeSelection);
    nodesToCheck.erase(v6);

    {
        BspSchedule<Graph> tmpSchedule(instance);
        kl.GetActiveScheduleTest(tmpSchedule);
        KlImproverTestT klFresh;
        klFresh.SetupSchedule(tmpSchedule);
        klFresh.InsertGainHeapTest(allNodes);
        CheckEqualAffinityTable(affinity, klFresh.GetAffinityTable(), nodesToCheck);
    }
}

/**
 * Test cross-step move where the moved node changes the max-work processor at both steps.
 * This exercises the updateEntireStep path in ProcessWorkUpdateStep.
 */
BOOST_AUTO_TEST_CASE(CrossStepMoveChangingMaxWorkBothStepsTest) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;
    using KlMove = KlMoveStruct<double, VertexType>;

    Graph dag;

    // Set up so moving a heavy node changes max work at both steps
    const VertexType v0 = dag.AddVertex(3, 1, 1);     // work=3
    const VertexType v1 = dag.AddVertex(3, 1, 1);     // work=3
    const VertexType v2 = dag.AddVertex(20, 1, 1);    // work=20 (will be moved, is max at fromStep)
    const VertexType v3 = dag.AddVertex(5, 1, 1);     // work=5
    const VertexType v4 = dag.AddVertex(4, 1, 1);     // work=4 (current max at toStep)
    const VertexType v5 = dag.AddVertex(2, 1, 1);     // work=2

    dag.AddEdge(v0, v3, 1);
    dag.AddEdge(v1, v4, 1);
    dag.AddEdge(v2, v5, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule schedule(instance);

    // Step 0: v0(p0, w=3), v1(p1, w=3), v2(p2, w=20) -- max=20 on p2
    // Step 1: v3(p0, w=5), v4(p1, w=4), v5(p2, w=2) -- max=5 on p0
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 0, 1, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    using KlImproverTestT = KlImproverTest<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double>;

    KlImproverTestT kl;
    kl.SetupSchedule(schedule);

    std::vector<VertexType> allNodes = {0, 1, 2, 3, 4, 5};
    auto nodeSelection = kl.InsertGainHeapTest(allNodes);
    auto &affinity = kl.GetAffinityTable();

    // Move v2 (work=20) from (p2, step0) to (p1, step1)
    // fromStep: max drops from 20 to 3
    // toStep: max rises from 5 to 24 (4+20)
    KlMove move(v2, 0.0, 2, 0, 1, 1);
    kl.UpdateAffinityTableTest(move, nodeSelection);

    BspSchedule<Graph> postMoveSchedule(instance);
    kl.GetActiveScheduleTest(postMoveSchedule);
    KlImproverTestT klFresh;
    klFresh.SetupSchedule(postMoveSchedule);
    klFresh.InsertGainHeapTest(allNodes);

    std::set<size_t> nodesToCheck = {0, 1, 3, 4, 5};

    CheckEqualAffinityTable(affinity, klFresh.GetAffinityTable(), nodesToCheck);
}
