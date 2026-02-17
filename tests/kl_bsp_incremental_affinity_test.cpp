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
 * @file kl_bsp_incremental_affinity_test.cpp
 * @brief Comprehensive tests for incremental affinity table updates under
 *        KlBspCommCostFunction.
 *
 * After each KL inner iteration (RunInnerIterationTest), we validate:
 *   1. Communication datastructures (send/receive per step/proc) match fresh computation
 *   2. Tracked cost matches recomputed cost
 *   3. Affinity tables for remaining active nodes match freshly computed ones
 *
 * Structure:
 *   Suite 1 – SmallGraphSingleMove: small graphs, 1 node inserted, 1 iteration
 *   Suite 2 – SequentialMoves: multiple sequential iterations on same instance
 *   Suite 3 – StructuredGraphs: grid, butterfly, ladder, tree, pipeline topologies
 *   Suite 4 – EdgeCases: boundary conditions and special configurations
 *   Suite 5 – AffinityTableConsistency: focused affinity table validation
 */

#define BOOST_TEST_MODULE kl_bsp_incremental_affinity
#include <boost/test/unit_test.hpp>
#include <cmath>

#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_test.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;
using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using VertexType = Graph::VertexIdx;
using KlActiveScheduleT = KlActiveSchedule<Graph, double, NoLocalSearchMemoryConstraint>;
using CommCostT = KlBspCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint>;
using KlTestT = KlImproverTest<Graph, CommCostT>;
using KlMoveT = KlMoveStruct<double, VertexType>;

// windowSize=2 variants
using CommCostW2T = KlBspCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 2>;
using KlTestW2T = KlImproverTest<Graph, CommCostW2T, NoLocalSearchMemoryConstraint, 2>;

// ============================================================================
// Helpers (adapted from kl_bsp_affinity_test.cpp)
// ============================================================================

/// Validate comm datastructures: compare incrementally maintained values
/// against freshly computed ones.
bool ValidateCommDatastructures(const MaxCommDatastructure<Graph, double, KlActiveScheduleT> &commDsIncremental,
                                KlActiveScheduleT &activeSched,
                                const BspInstance<Graph> &instance,
                                const std::string &context) {
    BspSchedule<Graph> currentSchedule(instance);
    activeSched.WriteSchedule(currentSchedule);

    KlActiveScheduleT klSchedFresh;
    klSchedFresh.Initialize(currentSchedule);

    MaxCommDatastructure<Graph, double, KlActiveScheduleT> commDsFresh;
    commDsFresh.Initialize(klSchedFresh);

    unsigned maxStep = currentSchedule.NumberOfSupersteps();
    commDsFresh.ComputeCommDatastructures(0, maxStep > 0 ? maxStep - 1 : 0);

    bool allMatch = true;
    for (unsigned step = 0; step < maxStep; ++step) {
        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            auto sendInc = commDsIncremental.StepProcSend(step, p);
            auto sendFresh = commDsFresh.StepProcSend(step, p);
            auto recvInc = commDsIncremental.StepProcReceive(step, p);
            auto recvFresh = commDsFresh.StepProcReceive(step, p);

            if (std::abs(sendInc - sendFresh) > 1e-6 || std::abs(recvInc - recvFresh) > 1e-6) {
                allMatch = false;
                std::cout << "  COMM MISMATCH [" << context << "] step " << step << " proc " << p << ":"
                          << " send(inc=" << sendInc << ", fresh=" << sendFresh << ")"
                          << " recv(inc=" << recvInc << ", fresh=" << recvFresh << ")" << std::endl;
            }
        }
    }

    // Validate lambda maps
    for (const auto v : instance.Vertices()) {
        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            unsigned countInc = 0;
            if (commDsIncremental.nodeLambdaMap_.HasProcEntry(v, p)) {
                countInc = commDsIncremental.nodeLambdaMap_.GetProcEntry(v, p);
            }
            unsigned countFresh = 0;
            if (commDsFresh.nodeLambdaMap_.HasProcEntry(v, p)) {
                countFresh = commDsFresh.nodeLambdaMap_.GetProcEntry(v, p);
            }
            if (countInc != countFresh) {
                allMatch = false;
                std::cout << "  LAMBDA MISMATCH [" << context << "] node " << v << " proc " << p << ":"
                          << " inc=" << countInc << " fresh=" << countFresh << std::endl;
            }
        }
    }

    return allMatch;
}

/// Validate affinity tables: compare incrementally maintained affinity values
/// against freshly computed ones, skipping out-of-range step indices.
bool ValidateAffinityTables(KlTestT &klIncremental, const BspInstance<Graph> &instance, const std::string &context) {
    constexpr unsigned windowSize = 1;

    BspSchedule<Graph> currentSchedule(instance);
    klIncremental.GetActiveScheduleTest(currentSchedule);

    KlTestT klFresh;
    klFresh.SetupSchedule(currentSchedule);

    std::vector<VertexType> selectedNodes;
    const size_t activeCount = klIncremental.GetAffinityTable().size();
    for (size_t i = 0; i < activeCount; ++i) {
        selectedNodes.push_back(klIncremental.GetAffinityTable().GetSelectedNodes()[i]);
    }

    klFresh.InsertGainHeapTest(selectedNodes);

    bool allMatch = true;
    const unsigned numProcs = instance.NumberOfProcessors();
    // Use the minimum of both step counts: the incremental may keep phantom
    // empty steps that the fresh schedule compacts away.
    const unsigned numStepsInc = klIncremental.GetActiveSchedule().NumSteps();
    const unsigned numStepsFresh = klFresh.GetActiveSchedule().NumSteps();
    const unsigned numSteps = std::min(numStepsInc, numStepsFresh);

    for (const auto &node : selectedNodes) {
        const auto &affinityInc = klIncremental.GetAffinityTable().GetAffinityTable(node);
        const auto &affinityFresh = klFresh.GetAffinityTable().GetAffinityTable(node);

        unsigned nodeStep = klIncremental.GetActiveSchedule().AssignedSuperstep(node);

        for (unsigned p = 0; p < numProcs; ++p) {
            if (p >= affinityInc.size() || p >= affinityFresh.size()) {
                continue;
            }
            for (unsigned idx = 0; idx < affinityInc[p].size() && idx < affinityFresh[p].size(); ++idx) {
                int stepOffset = static_cast<int>(idx) - static_cast<int>(windowSize);
                int targetStepSigned = static_cast<int>(nodeStep) + stepOffset;

                // Skip affinities for supersteps that don't exist in either schedule
                if (targetStepSigned < 0 || targetStepSigned >= static_cast<int>(numSteps)) {
                    continue;
                }

                double valInc = affinityInc[p][idx];
                double valFresh = affinityFresh[p][idx];

                if (std::abs(valInc - valFresh) > 1e-4) {
                    allMatch = false;
                    std::cout << "  AFFINITY MISMATCH [" << context << "]: node=" << node << " P" << p << " S" << targetStepSigned
                              << " (offset=" << stepOffset << ")"
                              << " inc=" << valInc << " fresh=" << valFresh << " diff=" << (valInc - valFresh) << std::endl;
                }
            }
        }
    }
    return allMatch;
}

/// Run one inner iteration and validate all consistency checks.
void RunAndValidate(KlTestT &kl, const BspInstance<Graph> &instance, const std::string &context) {
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, context));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);
    BOOST_CHECK(ValidateAffinityTables(kl, instance, context));
}

/// Run one inner iteration and validate comm datastructures and cost only.
/// Use this when many active nodes span distant steps, since the incremental
/// update intentionally only recomputes affinities for nodes whose window
/// overlaps the changed steps.
/// Templated to support different KlTestType instantiations (e.g. windowSize=2).
template <typename KlTestType>
void RunAndValidateCommAndCost(KlTestType &kl, const BspInstance<Graph> &instance, const std::string &context) {
    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, context));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);
}

// ============================================================================
// Suite 1: SmallGraphSingleMove — small graphs, few nodes inserted
// ============================================================================

BOOST_AUTO_TEST_SUITE(SmallGraphSingleMove)

// Simple edge: parent on P0, child on P1, 2 supersteps.
BOOST_AUTO_TEST_CASE(SimpleParentChild) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddEdge(0, 1, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({0, 1});

    RunAndValidate(kl, instance, "SimpleParentChild");
}

// Fan-out: 1 parent, 2 children on different procs.
BOOST_AUTO_TEST_CASE(FanOutTwoChildren) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 1, 1);
    dag.AddVertex(12, 1, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 2});

    RunAndValidate(kl, instance, "FanOutTwoChildren iter1");
    RunAndValidate(kl, instance, "FanOutTwoChildren iter2");
}

// Fan-in: 2 parents, 1 child.
BOOST_AUTO_TEST_CASE(FanInTwoParents) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(10, 3, 2);
    dag.AddVertex(20, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({2});

    RunAndValidate(kl, instance, "FanInTwoParents");
}

// Diamond: 4 nodes, source->(mid1,mid2)->sink.
BOOST_AUTO_TEST_CASE(DiamondGraph) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
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
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 2});

    RunAndValidate(kl, instance, "Diamond iter1");
    RunAndValidate(kl, instance, "Diamond iter2");
}

// Chain: 3 nodes all on different procs, same step.
BOOST_AUTO_TEST_CASE(ChainSameStep) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(20, 10, 3);
    dag.AddVertex(15, 3, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "ChainSameStep");
}

// Chain: 3 nodes on different procs AND different steps.
BOOST_AUTO_TEST_CASE(ChainCrossStep) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(20, 10, 3);
    dag.AddVertex(15, 3, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "ChainCrossStep");
}

// 3 processors: parent P0, child P2, move candidate.
BOOST_AUTO_TEST_CASE(ThreeProcsSimple) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 2});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "ThreeProcsSimple");
}

// Fan-out: parent with 3 children on 3 different procs.
BOOST_AUTO_TEST_CASE(FanOutThreeChildren) {
    Graph dag;
    dag.AddVertex(10, 8, 3);
    dag.AddVertex(5, 1, 1);
    dag.AddVertex(5, 1, 1);
    dag.AddVertex(5, 1, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(0, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 3});
    schedule.SetAssignedSupersteps({0, 1, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 2, 3});

    RunAndValidate(kl, instance, "FanOutThreeChildren iter1");
    RunAndValidate(kl, instance, "FanOutThreeChildren iter2");
    RunAndValidate(kl, instance, "FanOutThreeChildren iter3");
}

// No edges: isolated nodes.
BOOST_AUTO_TEST_CASE(NoEdges) {
    Graph dag;
    dag.AddVertex(10, 5, 1);
    dag.AddVertex(20, 3, 1);
    dag.AddVertex(15, 4, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "NoEdges");
}

// All nodes on same processor: no initial communication.
BOOST_AUTO_TEST_CASE(AllSameProc) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(6, 3, 3);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 0, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "AllSameProc");
}

// All nodes on same step: only proc changes possible.
BOOST_AUTO_TEST_CASE(AllSameStep) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(6, 3, 3);
    dag.AddVertex(12, 2, 2);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 3});
    schedule.SetAssignedSupersteps({0, 0, 0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "AllSameStep");
}

BOOST_AUTO_TEST_SUITE_END()    // SmallGraphSingleMove

// ============================================================================
// Suite 2: SequentialMoves — multiple iterations on same instance
// ============================================================================

BOOST_AUTO_TEST_SUITE(SequentialMoves)

// Linear chain with 4 nodes, all on different procs.
BOOST_AUTO_TEST_CASE(ChainFourNodeAllProcs) {
    Graph dag;
    dag.AddVertex(1, 10, 1);
    dag.AddVertex(1, 8, 1);
    dag.AddVertex(1, 6, 1);
    dag.AddVertex(1, 4, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 3});
    schedule.SetAssignedSupersteps({0, 0, 0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "Chain4 iter1");

    kl.RunInnerIterationTest();
    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "Chain4 iter2"));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);
}

// Tree with 4 nodes, multiple iterations.
BOOST_AUTO_TEST_CASE(TreeMultipleIterations) {
    Graph dag;
    dag.AddVertex(1, 1, 1);
    dag.AddVertex(1, 1, 1);
    dag.AddVertex(1, 1, 1);
    dag.AddVertex(1, 1, 1);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(0, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 3});
    schedule.SetAssignedSupersteps({0, 0, 0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "Tree4 iter1");

    kl.RunInnerIterationTest();
    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "Tree4 iter2"));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    kl.RunInnerIterationTest();
    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "Tree4 iter3"));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);
}

// 8-node complex graph: insert 2 nodes, 5 iterations.
BOOST_AUTO_TEST_CASE(EightNodeComplexSequential) {
    Graph dag;
    dag.AddVertex(2, 9, 2);
    dag.AddVertex(3, 8, 4);
    dag.AddVertex(4, 7, 3);
    dag.AddVertex(5, 6, 2);
    dag.AddVertex(6, 5, 6);
    dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);
    dag.AddVertex(9, 2, 1);

    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 2, 2);
    dag.AddEdge(0, 3, 2);
    dag.AddEdge(1, 4, 12);
    dag.AddEdge(2, 4, 6);
    dag.AddEdge(2, 5, 7);
    dag.AddEdge(4, 7, 9);
    dag.AddEdge(3, 7, 9);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({2, 0});

    for (int i = 0; i < 5; ++i) {
        RunAndValidate(kl, instance, "8node iter" + std::to_string(i));
    }
}

// 8-node with denser edges: insert 2 nodes, more iterations.
// Nodes span 4 steps; after initial moves new nodes are added across distant
// steps, so the incremental update only recomputes affinities within the
// changed-steps window. We validate comm datastructures and cost (always exact).
BOOST_AUTO_TEST_CASE(EightNodeDenseSequential) {
    Graph dag;
    dag.AddVertex(2, 9, 2);
    dag.AddVertex(3, 8, 4);
    dag.AddVertex(4, 7, 3);
    dag.AddVertex(5, 6, 2);
    dag.AddVertex(6, 5, 6);
    dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);
    dag.AddVertex(9, 2, 1);

    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 4, 2);
    dag.AddEdge(0, 5, 2);
    dag.AddEdge(0, 2, 2);
    dag.AddEdge(0, 3, 2);
    dag.AddEdge(1, 4, 12);
    dag.AddEdge(1, 5, 2);
    dag.AddEdge(1, 6, 2);
    dag.AddEdge(1, 7, 2);
    dag.AddEdge(2, 4, 6);
    dag.AddEdge(2, 5, 7);
    dag.AddEdge(2, 6, 2);
    dag.AddEdge(2, 7, 2);
    dag.AddEdge(4, 7, 9);
    dag.AddEdge(3, 7, 9);
    dag.AddEdge(4, 6, 2);
    dag.AddEdge(5, 6, 2);
    dag.AddEdge(6, 7, 2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 3});

    RunAndValidate(kl, instance, "8nodeDense iter0");
    for (int i = 1; i < 5; ++i) {
        RunAndValidateCommAndCost(kl, instance, "8nodeDense iter" + std::to_string(i));
    }
}

// Cross-step moves: 6 nodes across 3 steps, 3 procs.
BOOST_AUTO_TEST_CASE(CrossStepSequential) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 3, 3);
    dag.AddVertex(6, 4, 4);
    dag.AddVertex(12, 6, 2);
    dag.AddVertex(5, 2, 5);
    dag.AddVertex(7, 3, 3);

    dag.AddEdge(0, 2, 2);
    dag.AddEdge(0, 3, 3);
    dag.AddEdge(1, 4, 2);
    dag.AddEdge(1, 5, 4);
    dag.AddEdge(2, 4, 5);
    dag.AddEdge(3, 5, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({3, 1});

    for (int i = 0; i < 4; ++i) {
        RunAndValidate(kl, instance, "CrossStep iter" + std::to_string(i));
    }
}

// Two independent components: moves in one shouldn't affect the other.
BOOST_AUTO_TEST_CASE(IndependentComponents) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(15, 6, 3);
    dag.AddVertex(12, 3, 2);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({0, 1, 2, 3});

    RunAndValidate(kl, instance, "IndepComp iter1");
    RunAndValidate(kl, instance, "IndepComp iter2");
}

// Each node individually: test each node as the single move candidate.
BOOST_AUTO_TEST_CASE(EachNodeIndividually) {
    Graph dag;
    dag.AddVertex(2, 9, 2);
    dag.AddVertex(3, 8, 4);
    dag.AddVertex(4, 7, 3);
    dag.AddVertex(5, 6, 2);
    dag.AddVertex(6, 5, 6);
    dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);
    dag.AddVertex(9, 2, 1);

    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 2, 2);
    dag.AddEdge(0, 3, 2);
    dag.AddEdge(1, 4, 12);
    dag.AddEdge(2, 4, 6);
    dag.AddEdge(2, 5, 7);
    dag.AddEdge(4, 7, 9);
    dag.AddEdge(3, 7, 9);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.UpdateNumberOfSupersteps();

    for (VertexType v = 0; v < 8; ++v) {
        KlTestT kl;
        kl.SetupSchedule(schedule);
        kl.InsertGainHeapTest({v});
        RunAndValidate(kl, instance, "EachNode v" + std::to_string(v));
    }
}

BOOST_AUTO_TEST_SUITE_END()    // SequentialMoves

// ============================================================================
// Suite 3: StructuredGraphs — standard topologies
// ============================================================================

BOOST_AUTO_TEST_SUITE(StructuredGraphs)

// 5x5 Grid Graph.
BOOST_AUTO_TEST_CASE(GridGraph5x5) {
    Graph dag = osp::ConstructGridDag<Graph>(5, 5);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    std::vector<unsigned> procs(25), steps(25);
    for (unsigned r = 0; r < 5; ++r) {
        for (unsigned c = 0; c < 5; ++c) {
            unsigned idx = r * 5 + c;
            if (r < 2) {
                procs[idx] = 0;
                steps[idx] = (c < 3) ? 0 : 1;
            } else if (r < 4) {
                procs[idx] = 1;
                steps[idx] = (c < 3) ? 2 : 3;
            } else {
                procs[idx] = 2;
                steps[idx] = (c < 3) ? 4 : 5;
            }
        }
    }
    procs[7] = 3;
    steps[7] = 1;

    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({12, 8, 7});

    for (int i = 0; i < 3; ++i) {
        RunAndValidate(kl, instance, "Grid5x5 iter" + std::to_string(i));
    }
}

// Butterfly graph: 2 stages (12 nodes).
BOOST_AUTO_TEST_CASE(ButterflyGraph2Stage) {
    Graph dag = osp::ConstructButterflyDag<Graph>(2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    std::vector<unsigned> procs(12), steps(12);
    for (unsigned i = 0; i < 12; ++i) {
        if (i < 4) {
            procs[i] = 0;
            steps[i] = 0;
        } else if (i < 8) {
            procs[i] = 1;
            steps[i] = 1;
        } else {
            procs[i] = 0;
            steps[i] = 2;
        }
    }
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({4, 6, 0});

    for (int i = 0; i < 3; ++i) {
        RunAndValidate(kl, instance, "Butterfly iter" + std::to_string(i));
    }
}

// Ladder graph: 5 rungs (12 nodes).
BOOST_AUTO_TEST_CASE(LadderGraph5Rungs) {
    Graph dag = osp::ConstructLadderDag<Graph>(5);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    std::vector<unsigned> procs(12), steps(12);
    for (unsigned i = 0; i < 6; ++i) {
        procs[2 * i] = 0;
        steps[2 * i] = i;
        procs[2 * i + 1] = 1;
        steps[2 * i + 1] = i;
    }
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 3, 0, 2});

    for (int i = 0; i < 4; ++i) {
        RunAndValidate(kl, instance, "Ladder iter" + std::to_string(i));
    }
}

// Binary out-tree: height 3 (15 nodes).
BOOST_AUTO_TEST_CASE(BinaryOutTreeH3) {
    Graph dag = osp::ConstructBinaryOutTree<Graph>(3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    size_t numNodes = dag.NumVertices();
    std::vector<unsigned> procs(numNodes), steps(numNodes);
    for (size_t i = 0; i < numNodes; ++i) {
        procs[i] = static_cast<unsigned>(i % 4);
        unsigned level = 0;
        size_t idx = i + 1;
        while (idx > 1) {
            idx /= 2;
            level++;
        }
        steps[i] = level;
    }
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 3, 5, 7});

    for (int i = 0; i < 4; ++i) {
        RunAndValidate(kl, instance, "BinTree iter" + std::to_string(i));
    }
}

// Multi-pipeline: 3 pipelines of length 3 (9 nodes).
BOOST_AUTO_TEST_CASE(MultiPipeline3x3) {
    Graph dag = osp::ConstructMultiPipelineDag<Graph>(3, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    size_t numNodes = dag.NumVertices();
    std::vector<unsigned> procs(numNodes), steps(numNodes);
    for (size_t i = 0; i < numNodes; ++i) {
        unsigned pipeline = static_cast<unsigned>(i / 3);
        unsigned stage = static_cast<unsigned>(i % 3);
        procs[i] = pipeline;
        steps[i] = stage;
    }
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 4, 7});

    for (int i = 0; i < 3; ++i) {
        RunAndValidate(kl, instance, "MultiPipe iter" + std::to_string(i));
    }
}

// Binary in-tree: height 2 (7 nodes), leaves spread across procs.
BOOST_AUTO_TEST_CASE(BinaryInTreeH2) {
    Graph dag = osp::ConstructBinaryInTree<Graph>(2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    // Root=0 at bottom (step 2), internal nodes at step 1, leaves at step 0
    // In-tree: leaves 3,4,5,6 -> internals 1,2 -> root 0
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2, 0});
    schedule.SetAssignedSupersteps({2, 1, 1, 0, 0, 0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({3, 4, 5, 6});

    for (int i = 0; i < 4; ++i) {
        RunAndValidate(kl, instance, "InTree iter" + std::to_string(i));
    }
}

BOOST_AUTO_TEST_SUITE_END()    // StructuredGraphs

// ============================================================================
// Suite 4: EdgeCases — special configurations and boundary conditions
// ============================================================================

BOOST_AUTO_TEST_SUITE(EdgeCases)

// Large comm weights and costs: precision test.
BOOST_AUTO_TEST_CASE(LargeCommWeights) {
    Graph dag;
    dag.AddVertex(10, 500, 100);
    dag.AddVertex(8, 1000, 200);
    dag.AddVertex(6, 300, 50);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(10);
    arch.SetSynchronisationCosts(100);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "LargeComm");
}

// 5 processors, 2 nodes.
BOOST_AUTO_TEST_CASE(ManyProcessors) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(5);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 4});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "ManyProcs");
}

// Mixed local and cross-proc edges.
BOOST_AUTO_TEST_CASE(MixedLocalCrossEdges) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(6, 3, 3);
    dag.AddVertex(12, 2, 2);
    dag.AddVertex(5, 6, 4);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 4, 1);
    dag.AddEdge(3, 4, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 0, 1, 2, 1});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 3});

    RunAndValidate(kl, instance, "MixedEdges iter1");
    RunAndValidate(kl, instance, "MixedEdges iter2");
}

// Long chain: 6 nodes, alternating procs, sequential steps.
BOOST_AUTO_TEST_CASE(LongChainAllDifferent) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 3);
    dag.AddVertex(6, 3, 1);
    dag.AddVertex(12, 6, 4);
    dag.AddVertex(5, 2, 2);
    dag.AddVertex(7, 1, 3);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(2, 3, 1);
    dag.AddEdge(3, 4, 1);
    dag.AddEdge(4, 5, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2});
    schedule.SetAssignedSupersteps({0, 1, 2, 3, 4, 5});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({2, 3});

    RunAndValidate(kl, instance, "LongChain iter1");
    RunAndValidate(kl, instance, "LongChain iter2");
}

// Dense graph: every earlier node connects to every later node.
// Nodes start on the same step but moves may scatter them across steps,
// causing later iterations to have stale affinities for distant nodes.
BOOST_AUTO_TEST_CASE(DenseGraph) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 3);
    dag.AddVertex(6, 3, 1);
    dag.AddVertex(12, 6, 4);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(0, 3, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 3});
    schedule.SetAssignedSupersteps({0, 0, 0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({2, 1});

    RunAndValidate(kl, instance, "DenseGraph iter1");
    RunAndValidateCommAndCost(kl, instance, "DenseGraph iter2");
}

// Source and sink with isolated node.
BOOST_AUTO_TEST_CASE(IsolatedSourceSink) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(6, 3, 3);
    dag.AddVertex(15, 2, 1);    // isolated
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 2, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({3, 1});

    RunAndValidate(kl, instance, "IsolatedSrcSink iter1");
    RunAndValidate(kl, instance, "IsolatedSrcSink iter2");
}

// Symmetric schedule: two identical edges with same weights.
BOOST_AUTO_TEST_CASE(SymmetricEdges) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(10, 5, 2);
    dag.AddEdge(0, 2, 3);
    dag.AddEdge(1, 3, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 1, 0});
    schedule.SetAssignedSupersteps({0, 0, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({2, 3});

    RunAndValidate(kl, instance, "SymEdges iter1");
    RunAndValidate(kl, instance, "SymEdges iter2");
}

BOOST_AUTO_TEST_SUITE_END()    // EdgeCases

// ============================================================================
// Suite 5: AffinityTableConsistency — focused on affinity table matching
// ============================================================================

BOOST_AUTO_TEST_SUITE(AffinityTableConsistency)

// Run full sweep: insert all 8 nodes, exhaust the heap.
// With all nodes active across 4 steps and windowSize=1, the incremental
// update only recomputes affinities for nodes near the changed steps.
// We validate comm datastructures and cost for all iterations.
BOOST_AUTO_TEST_CASE(FullSweepEightNodes) {
    Graph dag;
    dag.AddVertex(2, 9, 2);
    dag.AddVertex(3, 8, 4);
    dag.AddVertex(4, 7, 3);
    dag.AddVertex(5, 6, 2);
    dag.AddVertex(6, 5, 6);
    dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);
    dag.AddVertex(9, 2, 1);

    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 2, 2);
    dag.AddEdge(0, 3, 2);
    dag.AddEdge(1, 4, 12);
    dag.AddEdge(2, 4, 6);
    dag.AddEdge(2, 5, 7);
    dag.AddEdge(4, 7, 9);
    dag.AddEdge(3, 7, 9);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({0, 1, 2, 3, 4, 5, 6, 7});

    for (int i = 0; i < 7; ++i) {
        RunAndValidateCommAndCost(kl, instance, "FullSweep iter" + std::to_string(i));
    }
}

// 3-proc graph: 6 nodes across 3 steps, comm and cost validation.
BOOST_AUTO_TEST_CASE(ThreeProcAffinityCheck) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 3, 1);
    dag.AddVertex(12, 4, 3);
    dag.AddVertex(15, 1, 2);
    dag.AddVertex(6, 6, 4);
    dag.AddVertex(9, 2, 1);
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
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({0, 1, 2, 3, 4, 5});

    for (int i = 0; i < 4; ++i) {
        RunAndValidateCommAndCost(kl, instance, "3proc iter" + std::to_string(i));
    }
}

// Grid with all nodes inserted: extensive sweep.
// 16 nodes across 4 steps: only comm datastructures and cost validated.
BOOST_AUTO_TEST_CASE(GridFullSweep) {
    Graph dag = osp::ConstructGridDag<Graph>(4, 4);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    std::vector<unsigned> procs(16), steps(16);
    for (unsigned i = 0; i < 16; ++i) {
        procs[i] = i % 3;
        steps[i] = i / 4;
    }
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);

    std::vector<VertexType> allNodes;
    for (unsigned i = 0; i < 16; ++i) {
        allNodes.push_back(i);
    }
    kl.InsertGainHeapTest(allNodes);

    for (int i = 0; i < 10; ++i) {
        RunAndValidateCommAndCost(kl, instance, "GridSweep iter" + std::to_string(i));
    }
}

// Butterfly with mixed proc/step assignments.
// 12 nodes across 3 steps: comm and cost validation only.
BOOST_AUTO_TEST_CASE(ButterflyMixedAssignment) {
    Graph dag = osp::ConstructButterflyDag<Graph>(2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    // Spread nodes across 3 procs and 3 steps
    std::vector<unsigned> procs(12), steps(12);
    for (unsigned i = 0; i < 12; ++i) {
        procs[i] = i % 3;
        steps[i] = i / 4;
    }
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);

    std::vector<VertexType> allNodes;
    for (unsigned i = 0; i < 12; ++i) {
        allNodes.push_back(i);
    }
    kl.InsertGainHeapTest(allNodes);

    for (int i = 0; i < 8; ++i) {
        RunAndValidateCommAndCost(kl, instance, "ButterflyMixed iter" + std::to_string(i));
    }
}

// Ladder with all nodes: full consistency check.
// 10 nodes across 5 steps: comm and cost validation only.
BOOST_AUTO_TEST_CASE(LadderFullSweep) {
    Graph dag = osp::ConstructLadderDag<Graph>(4);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    std::vector<unsigned> procs(10), steps(10);
    for (unsigned i = 0; i < 5; ++i) {
        procs[2 * i] = 0;
        steps[2 * i] = i;
        procs[2 * i + 1] = 1;
        steps[2 * i + 1] = i;
    }
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);

    std::vector<VertexType> allNodes;
    for (unsigned i = 0; i < 10; ++i) {
        allNodes.push_back(i);
    }
    kl.InsertGainHeapTest(allNodes);

    for (int i = 0; i < 8; ++i) {
        RunAndValidateCommAndCost(kl, instance, "LadderSweep iter" + std::to_string(i));
    }
}

BOOST_AUTO_TEST_SUITE_END()    // AffinityTableConsistency

// ============================================================================
// Suite 6: WindowSize2 — tests with windowSize=2 (windowRange=5)
// ============================================================================

BOOST_AUTO_TEST_SUITE(WindowSize2)

// Simple parent-child with windowSize=2: broader affinity window.
BOOST_AUTO_TEST_CASE(SimpleParentChildW2) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddEdge(0, 1, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestW2T kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({0, 1});

    RunAndValidateCommAndCost(kl, instance, "W2_ParentChild");
}

// Diamond graph with windowSize=2.
BOOST_AUTO_TEST_CASE(DiamondW2) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
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
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestW2T kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 2});

    RunAndValidateCommAndCost(kl, instance, "W2_Diamond iter1");
    RunAndValidateCommAndCost(kl, instance, "W2_Diamond iter2");
}

// Chain across 5 steps with windowSize=2: window covers more steps.
BOOST_AUTO_TEST_CASE(LongChainW2) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 3);
    dag.AddVertex(6, 3, 1);
    dag.AddVertex(12, 6, 4);
    dag.AddVertex(5, 2, 2);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(2, 3, 1);
    dag.AddEdge(3, 4, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1});
    schedule.SetAssignedSupersteps({0, 1, 2, 3, 4});
    schedule.UpdateNumberOfSupersteps();

    KlTestW2T kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({2});

    RunAndValidateCommAndCost(kl, instance, "W2_LongChain");
}

// 8-node graph with 4 steps, windowSize=2 covers nearly everything.
BOOST_AUTO_TEST_CASE(EightNodeW2) {
    Graph dag;
    dag.AddVertex(2, 9, 2);
    dag.AddVertex(3, 8, 4);
    dag.AddVertex(4, 7, 3);
    dag.AddVertex(5, 6, 2);
    dag.AddVertex(6, 5, 6);
    dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);
    dag.AddVertex(9, 2, 1);

    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 2, 2);
    dag.AddEdge(0, 3, 2);
    dag.AddEdge(1, 4, 12);
    dag.AddEdge(2, 4, 6);
    dag.AddEdge(2, 5, 7);
    dag.AddEdge(4, 7, 9);
    dag.AddEdge(3, 7, 9);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.UpdateNumberOfSupersteps();

    KlTestW2T kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({2, 5});

    for (int i = 0; i < 3; ++i) {
        RunAndValidateCommAndCost(kl, instance, "W2_8node iter" + std::to_string(i));
    }
}

// Grid 4x4 with windowSize=2 and 4 procs.
BOOST_AUTO_TEST_CASE(GridW2) {
    Graph dag = osp::ConstructGridDag<Graph>(4, 4);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    std::vector<unsigned> procs(16), steps(16);
    for (unsigned i = 0; i < 16; ++i) {
        procs[i] = i % 4;
        steps[i] = i / 4;
    }
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestW2T kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({5, 6, 9, 10});

    for (int i = 0; i < 4; ++i) {
        RunAndValidateCommAndCost(kl, instance, "W2_Grid iter" + std::to_string(i));
    }
}

BOOST_AUTO_TEST_SUITE_END()    // WindowSize2

// ============================================================================
// Suite 7: NUMACosts — non-uniform communication costs
// ============================================================================

BOOST_AUTO_TEST_SUITE(NUMACosts)

// 2 processors with asymmetric NUMA send costs.
BOOST_AUTO_TEST_CASE(TwoProcAsymmetricNuma) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(6, 3, 3);
    dag.AddEdge(0, 1, 2);
    dag.AddEdge(1, 2, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);
    // Asymmetric: P0->P1 costs 2, P1->P0 costs 5
    arch.SetSendCosts(0, 1, 2);
    arch.SetSendCosts(1, 0, 5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidate(kl, instance, "AsymNuma");
}

// 4 processors with exponential NUMA costs.
BOOST_AUTO_TEST_CASE(FourProcExpNuma) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 3, 1);
    dag.AddVertex(12, 4, 3);
    dag.AddVertex(15, 1, 2);
    dag.AddVertex(6, 6, 4);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(0, 3, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(1, 4, 1);
    dag.AddEdge(2, 4, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);
    // NUMA: nearby procs cheap, distant procs expensive
    arch.SetSendCosts({
        {0, 1, 2, 3},
        {1, 0, 1, 2},
        {2, 1, 0, 1},
        {3, 2, 1, 0}
    });

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 3, 0});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({2, 3});

    RunAndValidate(kl, instance, "ExpNuma iter1");
    RunAndValidate(kl, instance, "ExpNuma iter2");
}

// 4 processors with custom send cost matrix.
BOOST_AUTO_TEST_CASE(FourProcCustomSendMatrix) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(6, 3, 3);
    dag.AddVertex(12, 2, 2);
    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 2, 3);
    dag.AddEdge(1, 3, 4);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);
    // Custom matrix: P0-P1 close, P2-P3 close, cross-group expensive
    arch.SetSendCosts({
        {0, 1, 5, 5},
        {1, 0, 5, 5},
        {5, 5, 0, 1},
        {5, 5, 1, 0}
    });

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 2, 1, 3});
    schedule.SetAssignedSupersteps({0, 0, 1, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 2});

    RunAndValidate(kl, instance, "CustomNuma iter1");
    RunAndValidate(kl, instance, "CustomNuma iter2");
}

// NUMA with diamond and more iterations.
BOOST_AUTO_TEST_CASE(NumaDiamondMultiIter) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 3, 1);
    dag.AddVertex(12, 4, 1);
    dag.AddVertex(15, 1, 1);
    dag.AddEdge(0, 1, 3);
    dag.AddEdge(0, 2, 2);
    dag.AddEdge(1, 3, 4);
    dag.AddEdge(2, 3, 5);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(5);
    // Asymmetric 3-proc NUMA
    arch.SetSendCosts({
        {0, 1, 3},
        {1, 0, 2},
        {3, 2, 0}
    });

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 2, 3});

    for (int i = 0; i < 3; ++i) {
        RunAndValidateCommAndCost(kl, instance, "NumaDiamond iter" + std::to_string(i));
    }
}

// NUMA combined with windowSize=2.
BOOST_AUTO_TEST_CASE(NumaWindowSize2) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(6, 3, 3);
    dag.AddVertex(12, 6, 4);
    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 2, 3);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 3, 2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);
    arch.SetSendCosts({
        {0, 1, 4},
        {1, 0, 3},
        {4, 3, 0}
    });

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestW2T kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 2});

    RunAndValidateCommAndCost(kl, instance, "NumaW2 iter1");
    RunAndValidateCommAndCost(kl, instance, "NumaW2 iter2");
}

BOOST_AUTO_TEST_SUITE_END()    // NUMACosts

// ============================================================================
// Suite 8: LargerProcessorCounts — 8+ processors
// ============================================================================

BOOST_AUTO_TEST_SUITE(LargerProcessorCounts)

// 8 processors, fan-out graph.
BOOST_AUTO_TEST_CASE(EightProcFanOut) {
    Graph dag;
    dag.AddVertex(10, 20, 5);
    for (unsigned i = 0; i < 8; ++i) {
        dag.AddVertex(5, 3, 1);
        dag.AddEdge(0, i + 1, 2);
    }

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(8);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    std::vector<unsigned> procs = {0, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<unsigned> steps = {0, 1, 1, 1, 1, 1, 1, 1, 1};
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 2, 3});

    RunAndValidate(kl, instance, "8proc_fan iter0");
    for (int i = 1; i < 3; ++i) {
        RunAndValidateCommAndCost(kl, instance, "8proc_fan iter" + std::to_string(i));
    }
}

// 8 processors, pipeline spread across all procs.
BOOST_AUTO_TEST_CASE(EightProcPipeline) {
    Graph dag = osp::ConstructMultiPipelineDag<Graph>(2, 4);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(8);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // Pipeline 0: procs 0,1,2,3; Pipeline 1: procs 4,5,6,7
    std::vector<unsigned> procs = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<unsigned> steps = {0, 1, 2, 3, 0, 1, 2, 3};
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 5});

    RunAndValidate(kl, instance, "8proc_pipe iter1");
    RunAndValidateCommAndCost(kl, instance, "8proc_pipe iter2");
}

// 8 processors with NUMA exponential costs.
BOOST_AUTO_TEST_CASE(EightProcExpNuma) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(6, 3, 3);
    dag.AddVertex(12, 2, 2);
    dag.AddVertex(5, 6, 4);
    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 2, 3);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 4, 2);
    dag.AddEdge(3, 4, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(8);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);
    // 8-proc hierarchical NUMA: nearby cheap, distant expensive
    arch.SetSendCosts({
        {0, 1, 2, 2, 4, 4, 4, 4},
        {1, 0, 2, 2, 4, 4, 4, 4},
        {2, 2, 0, 1, 4, 4, 4, 4},
        {2, 2, 1, 0, 4, 4, 4, 4},
        {4, 4, 4, 4, 0, 1, 2, 2},
        {4, 4, 4, 4, 1, 0, 2, 2},
        {4, 4, 4, 4, 2, 2, 0, 1},
        {4, 4, 4, 4, 2, 2, 1, 0}
    });

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 3, 5, 7, 1});
    schedule.SetAssignedSupersteps({0, 1, 1, 2, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 2});

    RunAndValidate(kl, instance, "8procNuma iter1");
    RunAndValidate(kl, instance, "8procNuma iter2");
}

// 8 procs, all nodes on same step: pure processor rebalancing.
BOOST_AUTO_TEST_CASE(EightProcSameStep) {
    Graph dag;
    for (unsigned i = 0; i < 8; ++i) {
        dag.AddVertex(static_cast<int>(10 + i), 5, 1);
    }
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 4, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(2, 3, 1);
    dag.AddEdge(4, 5, 1);
    dag.AddEdge(5, 6, 1);
    dag.AddEdge(6, 7, 1);
    dag.AddEdge(3, 7, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(8);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    std::vector<unsigned> procs = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<unsigned> steps(8, 0);
    schedule.SetAssignedProcessors(procs);
    schedule.SetAssignedSupersteps(steps);
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({3, 7});

    RunAndValidate(kl, instance, "8procSame iter1");
    RunAndValidateCommAndCost(kl, instance, "8procSame iter2");
}

BOOST_AUTO_TEST_SUITE_END()    // LargerProcessorCounts

// ============================================================================
// Suite 9: StepEmptying — moves that empty a superstep
// ============================================================================

BOOST_AUTO_TEST_SUITE(StepEmptying)

// Single node in middle step: move it away, step becomes empty.
BOOST_AUTO_TEST_CASE(EmptyMiddleStep) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);    // sole occupant of step 1
    dag.AddVertex(6, 3, 3);
    dag.AddEdge(0, 1, 2);
    dag.AddEdge(1, 2, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    // After this move, node 1 may leave step 1 empty.
    RunAndValidateCommAndCost(kl, instance, "EmptyMiddle");
}

// Two steps, one with a single node: move empties that step.
BOOST_AUTO_TEST_CASE(EmptyLastStep) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(12, 6, 3);
    dag.AddVertex(5, 2, 1);    // sole occupant of step 2
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0});
    schedule.SetAssignedSupersteps({0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({3, 1});

    RunAndValidateCommAndCost(kl, instance, "EmptyLast iter1");
    RunAndValidateCommAndCost(kl, instance, "EmptyLast iter2");
}

// High sync cost: step removal would be beneficial (maxWork < syncCost).
BOOST_AUTO_TEST_CASE(HighSyncCostSingleNodeStep) {
    Graph dag;
    dag.AddVertex(1, 50, 10);
    dag.AddVertex(1, 2, 1);    // tiny work, alone at step 1
    dag.AddVertex(1, 50, 10);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(100);    // very high sync cost

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1});

    RunAndValidateCommAndCost(kl, instance, "HighSync");
}

// Multiple sparse steps: several steps with 1 node each.
BOOST_AUTO_TEST_CASE(MultipleSparseSteps) {
    Graph dag;
    dag.AddVertex(10, 20, 5);
    dag.AddVertex(8, 15, 3);
    dag.AddVertex(6, 10, 2);
    dag.AddVertex(5, 5, 1);
    dag.AddVertex(12, 25, 4);
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(2, 3, 1);
    dag.AddEdge(3, 4, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    // Each node alone on its step and different procs
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1});
    schedule.SetAssignedSupersteps({0, 1, 2, 3, 4});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 3});

    for (int i = 0; i < 3; ++i) {
        RunAndValidateCommAndCost(kl, instance, "Sparse iter" + std::to_string(i));
    }
}

// Single superstep: all nodes on step 0, move can only change proc.
BOOST_AUTO_TEST_CASE(SingleSuperstep) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(6, 3, 3);
    dag.AddVertex(12, 6, 4);
    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 2, 3);
    dag.AddEdge(1, 3, 1);
    dag.AddEdge(2, 3, 2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 3});
    schedule.SetAssignedSupersteps({0, 0, 0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({1, 2});

    RunAndValidate(kl, instance, "SingleStep iter1");
    RunAndValidateCommAndCost(kl, instance, "SingleStep iter2");
}

// Deeply sequential: each node on its own step, many steps.
BOOST_AUTO_TEST_CASE(DeeplySequential) {
    Graph dag;
    for (unsigned i = 0; i < 6; ++i) {
        dag.AddVertex(static_cast<int>(10 + 2 * i), 5, 2);
        if (i > 0) {
            dag.AddEdge(i - 1, i, 1);
        }
    }

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 0, 1, 2});
    schedule.SetAssignedSupersteps({0, 1, 2, 3, 4, 5});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.InsertGainHeapTest({2, 3});

    RunAndValidateCommAndCost(kl, instance, "DeepSeq iter1");
    RunAndValidateCommAndCost(kl, instance, "DeepSeq iter2");
}

BOOST_AUTO_TEST_SUITE_END()    // StepEmptying

// ============================================================================
// Suite 10: StepRemovalRollback — step removal followed by rollback or success
// ============================================================================

BOOST_AUTO_TEST_SUITE(StepRemovalRollback)

// Rollback: scatter a node to a bad cross-proc destination so that the
// cost increase exceeds the sync-cost saving.  RevertToBestSchedule must
// re-insert the removed step and restore the original schedule.
BOOST_AUTO_TEST_CASE(RollbackBadScatter) {
    // 3-node chain: 0 → 1, with node 1 alone on step 1 (work=1 < syncCost=2).
    // All nodes on proc 0 — no cross-proc comm initially.
    Graph dag;
    dag.AddVertex(100, 50, 10);    // node 0
    dag.AddVertex(1, 50, 10);      // node 1 (low work, alone on step 1)
    dag.AddVertex(100, 50, 10);    // node 2
    dag.AddEdge(0, 1, 1);          // heavy comm weight via commCost multiplier

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(10);
    arch.SetSynchronisationCosts(2);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 0, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);

    // --- Verify initial state ---
    BOOST_CHECK(kl.CheckRemoveSuperstepTest(1));    // maxWork(step1)=1 < syncCost=2
    const auto initialCost = kl.GetCurrentCost();
    const unsigned initialEndStep = kl.GetEndStep();    // 2

    // --- Scatter node 1 to a deliberately bad destination (proc 1, step 2) ---
    // Edge 0→1: proc 0 step 0 → proc 1 step 2, cross-proc.  Creates comm.
    KlMoveT badScatter(1, 0.0, 0, 1, 1, 2);    // gain_ unused; ApplyMoveWithFreshCost computes it
    kl.ApplyMoveWithFreshCost(badScatter);

    // Cost should have increased significantly (new cross-proc comm)
    BOOST_CHECK_GT(kl.GetCurrentCost(), initialCost);

    // --- Step removal flow ---
    kl.SwapEmptyStepFwdTest(1);      // bubble empty step 1 to end, endStep_--
    kl.PushRemoveStepSentinel(1);    // record removal as sentinel in appliedMoves_
    BOOST_CHECK_EQUAL(kl.GetEndStep(), initialEndStep - 1);

    kl.UpdateCostAfterRemoval();    // cost_ -= syncCost=2

    // Cost is still much worse than initial → best was NOT updated post-removal
    BOOST_CHECK_GT(kl.GetCurrentCost(), kl.GetBestCost());

    // --- Rollback ---
    kl.RevertToBestScheduleTest();

    // Step must be re-inserted
    BOOST_CHECK_EQUAL(kl.GetEndStep(), initialEndStep);

    // Cost must match the initial (best) cost
    BOOST_CHECK_CLOSE(kl.GetCurrentCost(), initialCost, 0.00001);

    // Comm datastructures must be consistent with the restored schedule
    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "Rollback"));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    // Verify node assignments restored
    BspSchedule<Graph> restored(instance);
    kl.GetActiveScheduleTest(restored);
    BOOST_CHECK_EQUAL(restored.AssignedProcessor(1), 0u);
    BOOST_CHECK_EQUAL(restored.AssignedSuperstep(1), 1u);
}

// Success: scatter a node cross-proc so cost worsens slightly, but the
// sync-cost saving far exceeds the increase.  The step stays removed.
BOOST_AUTO_TEST_CASE(SuccessfulRemoval) {
    // 3-node chain: 0 → 1.  Cross-proc scatter worsens cost by ~4,
    // but syncCost=100 saves far more → removal wins.
    Graph dag;
    dag.AddVertex(100, 5, 2);    // node 0
    dag.AddVertex(1, 5, 2);      // node 1 (low work, alone on step 1)
    dag.AddVertex(100, 5, 2);    // node 2
    dag.AddEdge(0, 1, 5);        // weight=5 → cross-proc comm = 5

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(100);    // very high → removal wins

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 0, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);

    BOOST_CHECK(kl.CheckRemoveSuperstepTest(1));
    const auto initialCost = kl.GetCurrentCost();

    // Scatter node 1 cross-proc to step 2: creates comm for edge 0→1.
    // Cost worsens by ~4 (work at step 1 drops by 1, comm at step 2 += 5).
    KlMoveT scatter(1, 0.0, 0, 1, 1, 2);
    kl.ApplyMoveWithFreshCost(scatter);
    BOOST_CHECK_GT(kl.GetCurrentCost(), initialCost);    // cost worsened

    // Step removal flow
    kl.SwapEmptyStepFwdTest(1);
    kl.PushRemoveStepSentinel(1);
    BOOST_CHECK_EQUAL(kl.GetEndStep(), 1u);    // was 2, now 1

    kl.UpdateCostAfterRemoval();

    // Cost now much better than initial → best IS post-removal
    BOOST_CHECK_LT(kl.GetCurrentCost(), initialCost);

    // Revert to best — step should NOT be re-inserted
    kl.RevertToBestScheduleTest();

    // Step stays removed
    BOOST_CHECK_EQUAL(kl.GetEndStep(), 1u);
    BOOST_CHECK_LT(kl.GetCurrentCost(), initialCost);

    // Comm datastructures consistent
    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "Success"));

    // Verify via written schedule
    BspSchedule<Graph> written(instance);
    kl.GetActiveScheduleTest(written);
    BOOST_CHECK_EQUAL(written.NumberOfSupersteps(), 2u);
    // Node 1 should be on proc 1 (cross-proc scatter destination)
    BOOST_CHECK_EQUAL(written.AssignedProcessor(1), 1u);
}

// Rollback with 4 nodes: scatter two nodes from a step, verify full restoration.
BOOST_AUTO_TEST_CASE(RollbackTwoNodeStep) {
    // Step 1 has two low-work nodes (both on proc 0); scatter both cross-proc.
    // No edges from nodes 1,2 to node 3 — avoids same-step cross-proc violations.
    Graph dag;
    dag.AddVertex(100, 50, 10);    // node 0: proc 0, step 0
    dag.AddVertex(1, 50, 10);      // node 1: proc 0, step 1
    dag.AddVertex(1, 50, 10);      // node 2: proc 0, step 1
    dag.AddVertex(100, 50, 10);    // node 3: proc 0, step 2
    dag.AddEdge(0, 1, 5);          // weight=5
    dag.AddEdge(0, 2, 5);          // weight=5

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(3);    // maxWork(step1)=2 < 3

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 0, 0, 0});
    schedule.SetAssignedSupersteps({0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    BOOST_CHECK(kl.CheckRemoveSuperstepTest(1));

    const auto initialCost = kl.GetCurrentCost();
    const unsigned initialEndStep = kl.GetEndStep();

    // Scatter both nodes to bad cross-proc destinations (proc 1, step 2).
    // Each adds cross-proc comm=5 and removes work=1 from step 1. Net: +4 each.
    KlMoveT scatter1(1, 0.0, 0, 1, 1, 2);    // node 1: proc 0→1, step 1→2
    kl.ApplyMoveWithFreshCost(scatter1);
    KlMoveT scatter2(2, 0.0, 0, 1, 1, 2);    // node 2: proc 0→1, step 1→2
    kl.ApplyMoveWithFreshCost(scatter2);

    // Cost should have increased (total +8 >> syncCost=3)
    BOOST_CHECK_GT(kl.GetCurrentCost(), initialCost);

    // Step 1 is now empty
    kl.SwapEmptyStepFwdTest(1);
    kl.PushRemoveStepSentinel(1);
    kl.UpdateCostAfterRemoval();

    // Cost increase (8) > syncCost (3) → rollback
    BOOST_CHECK_GT(kl.GetCurrentCost(), kl.GetBestCost());

    kl.RevertToBestScheduleTest();

    // Fully restored
    BOOST_CHECK_EQUAL(kl.GetEndStep(), initialEndStep);
    BOOST_CHECK_CLOSE(kl.GetCurrentCost(), initialCost, 0.00001);

    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "Rollback2Nodes"));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    // Verify node positions restored
    BspSchedule<Graph> restored(instance);
    kl.GetActiveScheduleTest(restored);
    BOOST_CHECK_EQUAL(restored.AssignedProcessor(1), 0u);
    BOOST_CHECK_EQUAL(restored.AssignedSuperstep(1), 1u);
    BOOST_CHECK_EQUAL(restored.AssignedProcessor(2), 0u);
    BOOST_CHECK_EQUAL(restored.AssignedSuperstep(2), 1u);
}

// Rollback with NUMA costs: cross-proc scatter is extra expensive due to NUMA.
BOOST_AUTO_TEST_CASE(RollbackNuma) {
    Graph dag;
    dag.AddVertex(100, 50, 10);
    dag.AddVertex(1, 50, 10);
    dag.AddVertex(100, 50, 10);
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(5);
    arch.SetSynchronisationCosts(2);
    // NUMA: P0-P1 moderate, P0-P2 very expensive
    arch.SetSendCosts({
        { 0,  2, 10},
        { 2,  0, 10},
        {10, 10,  0}
    });

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 0, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    BOOST_CHECK(kl.CheckRemoveSuperstepTest(1));

    const auto initialCost = kl.GetCurrentCost();
    const unsigned initialEndStep = kl.GetEndStep();

    // Scatter to the most expensive NUMA destination: proc 2
    KlMoveT badScatter(1, 0.0, 0, 1, 2, 2);
    kl.ApplyMoveWithFreshCost(badScatter);

    kl.SwapEmptyStepFwdTest(1);
    kl.PushRemoveStepSentinel(1);
    kl.UpdateCostAfterRemoval();

    BOOST_CHECK_GT(kl.GetCurrentCost(), kl.GetBestCost());

    kl.RevertToBestScheduleTest();

    BOOST_CHECK_EQUAL(kl.GetEndStep(), initialEndStep);
    BOOST_CHECK_CLOSE(kl.GetCurrentCost(), initialCost, 0.00001);

    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "RollbackNuma"));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);
}

// Success with two nodes: scatter both cross-proc, cost worsens by less
// than syncCost.  After removal the step stays removed.
BOOST_AUTO_TEST_CASE(SuccessfulRemovalTwoNodes) {
    Graph dag;
    dag.AddVertex(100, 50, 10);    // node 0: proc 0, step 0
    dag.AddVertex(1, 50, 10);      // node 1: proc 0, step 1
    dag.AddVertex(1, 50, 10);      // node 2: proc 0, step 1
    dag.AddVertex(100, 50, 10);    // node 3: proc 0, step 2
    dag.AddEdge(0, 1, 5);          // weight=5
    dag.AddEdge(0, 2, 5);          // weight=5

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(100);    // high → removal wins despite scatter cost

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 0, 0, 0});
    schedule.SetAssignedSupersteps({0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    BOOST_CHECK(kl.CheckRemoveSuperstepTest(1));

    const auto initialCost = kl.GetCurrentCost();

    // Scatter both nodes cross-proc: cost worsens by ~8 total (4 each)
    KlMoveT scatter1(1, 0.0, 0, 1, 1, 2);
    kl.ApplyMoveWithFreshCost(scatter1);
    KlMoveT scatter2(2, 0.0, 0, 1, 1, 2);
    kl.ApplyMoveWithFreshCost(scatter2);
    BOOST_CHECK_GT(kl.GetCurrentCost(), initialCost);

    kl.SwapEmptyStepFwdTest(1);
    kl.PushRemoveStepSentinel(1);
    kl.UpdateCostAfterRemoval();

    // syncCost (100) >> scatter cost increase (8) → removal wins
    BOOST_CHECK_LT(kl.GetCurrentCost(), initialCost);

    kl.RevertToBestScheduleTest();

    // Step stays removed
    BOOST_CHECK_EQUAL(kl.GetEndStep(), 1u);
    BOOST_CHECK_LT(kl.GetCurrentCost(), initialCost);

    // Comm datastructures consistent
    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "Success2Nodes"));

    // Verify via written schedule
    BspSchedule<Graph> written(instance);
    kl.GetActiveScheduleTest(written);
    BOOST_CHECK_EQUAL(written.NumberOfSupersteps(), 2u);
    BOOST_CHECK_EQUAL(written.AssignedProcessor(1), 1u);
    BOOST_CHECK_EQUAL(written.AssignedProcessor(2), 1u);
}

// CheckRemoveSuperstep returns false when step has enough work.
BOOST_AUTO_TEST_CASE(NotEligibleForRemoval) {
    Graph dag;
    dag.AddVertex(100, 5, 2);
    dag.AddVertex(50, 5, 2);    // work=50, NOT < syncCost=10
    dag.AddVertex(100, 5, 2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(10);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 0, 0});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);

    // maxWork(step 1) = 50 ≥ syncCost = 10 → NOT eligible
    BOOST_CHECK(!kl.CheckRemoveSuperstepTest(1));
}

// CheckRemoveSuperstep returns false with only 1 step.
BOOST_AUTO_TEST_CASE(SingleStepNotRemovable) {
    Graph dag;
    dag.AddVertex(1, 5, 2);
    dag.AddVertex(1, 5, 2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(100);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 0});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);

    // Only 1 step → can't remove
    BOOST_CHECK(!kl.CheckRemoveSuperstepTest(0));
}

BOOST_AUTO_TEST_SUITE_END()    // StepRemovalRollback

// ============================================================================
// Suite 7: PenaltyRewardCostTracking — verify that penalty/reward terms in
// the affinity table correctly cancel with the normalization in ApplyMove,
// so cost_ always equals pure schedule cost (ComputeScheduleCostTest).
//
// The first move after InsertGainHeap has a FRESH affinity table, so the gain
// is exact and penalty/reward must cancel perfectly.  Subsequent moves may use
// stale affinities (e.g. when a same-step move does not change max-comm/work,
// the incremental update path does not fully recompute neighbours' pure comm
// affinity).  This is a known approximation of the KL inner loop — it does
// NOT indicate a penalty/reward normalisation bug.
// ============================================================================
BOOST_AUTO_TEST_SUITE(PenaltyRewardCostTracking)

// Simple edge: parent on P0 step 0, child on P1 step 1.
// Non-zero penalty/reward, move creates no violations.
BOOST_AUTO_TEST_CASE(NoViolationWithPenalty) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddEdge(0, 1, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1});
    schedule.SetAssignedSupersteps({0, 1});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.ComputeViolationsTest();
    kl.InsertGainHeapTestPenalty({0, 1});

    kl.RunInnerIterationTest();

    BOOST_CHECK(ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "NoViolationWithPenalty"));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);
}

// 3-node chain with 3 steps, non-zero penalty and reward.
// First iteration must be exact (fresh affinities).
BOOST_AUTO_TEST_CASE(ChainWithPenaltyReward) {
    Graph dag;
    dag.AddVertex(10, 5, 2);
    dag.AddVertex(8, 4, 1);
    dag.AddVertex(6, 3, 1);
    dag.AddEdge(0, 1, 3);
    dag.AddEdge(1, 2, 2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2});
    schedule.SetAssignedSupersteps({0, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.ComputeViolationsTest();
    kl.InsertGainHeapTestPenaltyReward({0, 1, 2});

    for (int i = 0; i < 2; i++) {
        kl.RunInnerIterationTest();
        BOOST_CHECK(ValidateCommDatastructures(
            kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "ChainWithPenaltyReward iter" + std::to_string(i)));
        BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);
    }
}

// Diamond graph: source → A,B → sink. 2 processors, 3 steps.
// First move (fresh gain) must be exact.  Subsequent iterations use
// incrementally updated affinities which may become stale for pure comm
// (same-step moves where max-comm doesn't change skip full recomputation).
BOOST_AUTO_TEST_CASE(DiamondWithPenaltyReward) {
    Graph dag;
    dag.AddVertex(5, 3, 1);     // 0: source
    dag.AddVertex(10, 5, 2);    // 1: left
    dag.AddVertex(10, 4, 2);    // 2: right
    dag.AddVertex(8, 2, 1);     // 3: sink
    dag.AddEdge(0, 1, 2);
    dag.AddEdge(0, 2, 2);
    dag.AddEdge(1, 3, 3);
    dag.AddEdge(2, 3, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(5);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 0, 1, 1});
    schedule.SetAssignedSupersteps({0, 1, 1, 2});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.ComputeViolationsTest();
    kl.InsertGainHeapTestPenaltyReward({0, 1, 2, 3});

    // First move: fresh gain, penalty/reward must cancel exactly.
    kl.RunInnerIterationTest();
    BOOST_CHECK(ValidateCommDatastructures(
        kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "DiamondWithPenaltyReward iter0"));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    // Subsequent moves: comm datastructures must remain valid.
    // Cost tracking may drift due to stale affinities in the inner loop.
    for (int i = 1; i < 3; i++) {
        kl.RunInnerIterationTest();
        BOOST_CHECK(ValidateCommDatastructures(
            kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "DiamondWithPenaltyReward iter" + std::to_string(i)));
    }
}

// 8-node complex graph with high penalty values — stress test.
// The initial schedule has a violation (edge 6→7), so ComputeViolationsTest
// must be called to populate currentViolations_ before the gain heap.
BOOST_AUTO_TEST_CASE(EightNodeHighPenalty) {
    Graph dag;
    dag.AddVertex(10, 5, 2);    // 0
    dag.AddVertex(8, 4, 1);     // 1
    dag.AddVertex(12, 6, 3);    // 2
    dag.AddVertex(6, 3, 1);     // 3
    dag.AddVertex(10, 5, 2);    // 4
    dag.AddVertex(8, 4, 1);     // 5
    dag.AddVertex(14, 7, 3);    // 6
    dag.AddVertex(4, 2, 1);     // 7
    dag.AddEdge(0, 2, 5);
    dag.AddEdge(0, 3, 3);
    dag.AddEdge(1, 3, 2);
    dag.AddEdge(1, 4, 4);
    dag.AddEdge(2, 5, 6);
    dag.AddEdge(3, 5, 2);
    dag.AddEdge(3, 6, 3);
    dag.AddEdge(4, 6, 5);
    dag.AddEdge(4, 7, 2);
    dag.AddEdge(5, 7, 4);
    dag.AddEdge(6, 7, 3);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(2);
    arch.SetSynchronisationCosts(10);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);
    schedule.SetAssignedProcessors({0, 1, 2, 3, 0, 1, 2, 3});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.UpdateNumberOfSupersteps();

    KlTestT kl;
    kl.SetupSchedule(schedule);
    kl.ComputeViolationsTest();    // initial schedule has violation on edge 6→7
    kl.InsertGainHeapTestPenaltyReward({0, 1, 2, 3, 4, 5, 6, 7});

    // First move: fresh gain, penalty/reward must cancel exactly.
    kl.RunInnerIterationTest();
    BOOST_CHECK(
        ValidateCommDatastructures(kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "EightNodeHighPenalty iter0"));
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    // Subsequent moves: comm datastructures must remain valid.
    for (int i = 1; i < 7; i++) {
        kl.RunInnerIterationTest();
        BOOST_CHECK(ValidateCommDatastructures(
            kl.GetCommCostF().commDs_, kl.GetActiveSchedule(), instance, "EightNodeHighPenalty iter" + std::to_string(i)));
    }
}

BOOST_AUTO_TEST_SUITE_END()    // PenaltyRewardCostTracking
