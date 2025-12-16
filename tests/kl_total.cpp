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

#define BOOST_TEST_MODULE kl
#include <boost/test/unit_test.hpp>
#include <filesystem>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_test.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include_mt.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

template <typename GraphT>
void AddMemWeights(GraphT &dag) {
    int memWeight = 1;
    int commWeight = 7;

    for (const auto &v : dag.Vertices()) {
        dag.SetVertexWorkWeight(v, static_cast<VMemwT<GraphT>>(memWeight++ % 10 + 2));
        dag.SetVertexMemWeight(v, static_cast<VMemwT<GraphT>>(memWeight++ % 10 + 2));
        dag.SetVertexCommWeight(v, static_cast<VCommwT<GraphT>>(commWeight++ % 10 + 2));
    }
}

template <typename TableT>
void CheckEqualAffinityTable(TableT &table1, TableT &table2, const std::set<size_t> &nodes) {
    BOOST_CHECK_EQUAL(table1.size(), table2.size());

    for (auto i : nodes) {
        for (size_t j = 0; j < table1[i].size(); ++j) {
            for (size_t k = 0; k < table1[i][j].size(); ++k) {
                BOOST_CHECK(std::abs(table1[i][j][k] - table2[i][j][k]) < 0.000001);

                if (std::abs(table1[i][j][k] - table2[i][j][k]) > 0.000001) {
                    std::cout << "Mismatch at [" << i << "][" << j << "][" << k << "]: table_1=" << table1[i][j][k]
                              << ", table_2=" << table2[i][j][k] << std::endl;
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(KlImproverSmokeTest) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;

    Graph dag;

    const VertexType v1 = dag.AddVertex(2, 9, 2);
    const VertexType v2 = dag.AddVertex(3, 8, 4);
    const VertexType v3 = dag.AddVertex(4, 7, 3);
    const VertexType v4 = dag.AddVertex(5, 6, 2);
    const VertexType v5 = dag.AddVertex(6, 5, 6);
    const VertexType v6 = dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);
    const VertexType v8 = dag.AddVertex(9, 2, 1);

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v1, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);

    BspArchitecture<Graph> arch;

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

    schedule.UpdateNumberOfSupersteps();

    using KlImproverT = KlTotalCommImprover<Graph, NoLocalSearchMemoryConstraint, 1, true>;
    KlImproverT kl;

    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK_EQUAL(schedule.SatisfiesPrecedenceConstraints(), true);
}

BOOST_AUTO_TEST_CASE(KlImproverOnTestGraphs) {
    std::vector<std::string> filenamesGraph = TestGraphs();

    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    GreedyBspScheduler<ComputationalDagEdgeIdxVectorImplDefIntT> testScheduler;

    for (auto &filenameGraph : filenamesGraph) {
        BspInstance<Graph> instance;

        bool statusGraph
            = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / filenameGraph).string(), instance.GetComputationalDag());

        instance.GetArchitecture().SetSynchronisationCosts(5);
        instance.GetArchitecture().SetCommunicationCosts(5);
        instance.GetArchitecture().SetNumberOfProcessors(4);

        if (!statusGraph) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        AddMemWeights(instance.GetComputationalDag());

        BspSchedule<Graph> schedule(instance);
        const auto result = testScheduler.ComputeSchedule(schedule);

        BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
        BOOST_CHECK_EQUAL(&schedule.GetInstance(), &instance);
        BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

        KlTotalCommImprover<Graph> kl;

        auto status = kl.ImproveSchedule(schedule);

        BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
        BOOST_CHECK_EQUAL(schedule.SatisfiesPrecedenceConstraints(), true);
    }
}

BOOST_AUTO_TEST_CASE(KlImproverSuperstepRemovalTest) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;

    Graph dag;

    const VertexType v1 = dag.AddVertex(2, 9, 2);
    const VertexType v2 = dag.AddVertex(3, 8, 4);
    const VertexType v3 = dag.AddVertex(4, 7, 3);
    const VertexType v4 = dag.AddVertex(1, 6, 2);
    const VertexType v5 = dag.AddVertex(6, 5, 6);
    const VertexType v6 = dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);
    const VertexType v8 = dag.AddVertex(9, 2, 1);

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v2, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);

    BspArchitecture<Graph> arch;

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);
    instance.GetArchitecture().SetSynchronisationCosts(50);
    // Create a schedule with an almost empty superstep (step 1)
    schedule.SetAssignedProcessors({0, 0, 0, 0, 1, 1, 1, 1});
    schedule.SetAssignedSupersteps({0, 0, 0, 0, 1, 2, 2, 2});

    schedule.UpdateNumberOfSupersteps();
    unsigned originalSteps = schedule.NumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    KlImprover<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double> kl;

    auto status = kl.ImproveSchedule(schedule);

    BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
    BOOST_CHECK_LT(schedule.NumberOfSupersteps(), originalSteps);
}

BOOST_AUTO_TEST_CASE(KlImproverInnerLoopTest) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;

    Graph dag;

    const VertexType v1 = dag.AddVertex(2, 9, 2);
    const VertexType v2 = dag.AddVertex(3, 8, 4);
    const VertexType v3 = dag.AddVertex(4, 7, 3);
    const VertexType v4 = dag.AddVertex(5, 6, 2);
    const VertexType v5 = dag.AddVertex(6, 5, 6);
    const VertexType v6 = dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);
    const VertexType v8 = dag.AddVertex(9, 2, 1);

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v1, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);

    BspArchitecture<Graph> arch;

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

    schedule.UpdateNumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    using KlImproverTest = KlImproverTest<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double>;
    KlImproverTest kl;

    kl.SetupSchedule(schedule);

    auto &klActiveSchedule = kl.GetActiveSchedule();

    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(0), 5.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(0), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(1), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(1), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(2), 7.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(2), 6.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(3), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(3), 8.0);

    BOOST_CHECK_EQUAL(klActiveSchedule.NumSteps(), 4);
    BOOST_CHECK_EQUAL(klActiveSchedule.IsFeasible(), true);

    auto nodeSelection = kl.InsertGainHeapTestPenalty({2, 3});

    auto &affinity = kl.GetAffinityTable();

    BOOST_CHECK_CLOSE(affinity[v3][0][0], 5.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][0][1], 4.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][0][2], 9.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][1][0], -0.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][1][1], -4.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][1][2], 4.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v4][0][0], 5.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][0][1], 5.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][0][2], 5.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][1][0], -2.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][1][1], -6.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][1][2], -3.5, 0.00001);

    auto recomputeMaxGain = kl.RunInnerIterationTest();
    std::cout << "------------------------recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    recomputeMaxGain = kl.RunInnerIterationTest();
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    recomputeMaxGain = kl.RunInnerIterationTest();
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    recomputeMaxGain = kl.RunInnerIterationTest();
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);
}

BOOST_AUTO_TEST_CASE(KlImproverInnerLoopPenaltyTest) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;

    Graph dag;

    const VertexType v1 = dag.AddVertex(2, 9, 2);
    const VertexType v2 = dag.AddVertex(3, 8, 4);
    const VertexType v3 = dag.AddVertex(4, 7, 3);
    const VertexType v4 = dag.AddVertex(5, 6, 2);
    const VertexType v5 = dag.AddVertex(6, 5, 6);
    const VertexType v6 = dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);
    const VertexType v8 = dag.AddVertex(9, 2, 1);

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v1, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);

    BspArchitecture<Graph> arch;

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

    schedule.UpdateNumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    using KlImproverTest = KlImproverTest<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double>;
    KlImproverTest kl;

    kl.SetupSchedule(schedule);

    // auto &kl_active_schedule = kl.GetActiveSchedule();

    BOOST_CHECK_CLOSE(51.5, kl.GetCurrentCost(), 0.00001);

    auto nodeSelection = kl.InsertGainHeapTestPenalty({7});

    auto recomputeMaxGain = kl.RunInnerIterationTest();
    std::cout << "-----------recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    recomputeMaxGain = kl.RunInnerIterationTest();
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    recomputeMaxGain = kl.RunInnerIterationTest();
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);

    recomputeMaxGain = kl.RunInnerIterationTest();
    std::cout << "recompute max_gain: { ";
    for (const auto &[key, value] : recomputeMaxGain) {
        std::cout << key << " ";
    }
    std::cout << "}" << std::endl;

    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCostTest(), kl.GetCurrentCost(), 0.00001);
}

BOOST_AUTO_TEST_CASE(KlImproverViolationHandlingTest) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;

    Graph dag;

    const VertexType v1 = dag.AddVertex(2, 9, 2);
    const VertexType v2 = dag.AddVertex(3, 8, 4);
    const VertexType v3 = dag.AddVertex(4, 7, 3);
    const VertexType v4 = dag.AddVertex(5, 6, 2);
    const VertexType v5 = dag.AddVertex(6, 5, 6);
    const VertexType v6 = dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);
    const VertexType v8 = dag.AddVertex(9, 2, 1);

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v1, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);

    BspArchitecture<Graph> arch;

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.SetAssignedProcessors({0, 1, 0, 0, 1, 0, 0, 1});    // v1->v2 is on same step, different procs
    schedule.SetAssignedSupersteps({0, 0, 2, 1, 2, 2, 3, 3});

    schedule.UpdateNumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    KlImproverTest<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double> kl;

    kl.SetupSchedule(schedule);

    kl.ComputeViolationsTest();

    BOOST_CHECK_EQUAL(kl.IsFeasible(), false);

    KlImprover<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double> klImprover;
    klImprover.ImproveSchedule(schedule);

    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());
}

BOOST_AUTO_TEST_CASE(KlBase1) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;

    Graph dag;

    const VertexType v1 = dag.AddVertex(2, 9, 2);
    const VertexType v2 = dag.AddVertex(3, 8, 4);
    const VertexType v3 = dag.AddVertex(4, 7, 3);
    const VertexType v4 = dag.AddVertex(5, 6, 2);
    const VertexType v5 = dag.AddVertex(6, 5, 6);
    const VertexType v6 = dag.AddVertex(7, 4, 2);
    const VertexType v7 = dag.AddVertex(8, 3, 4);
    const VertexType v8 = dag.AddVertex(9, 2, 1);

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v1, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);

    BspArchitecture<Graph> arch;

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.SetAssignedProcessors({0, 0, 0, 0, 0, 0, 0, 0});
    schedule.SetAssignedSupersteps({0, 0, 0, 0, 0, 0, 0, 0});

    schedule.UpdateNumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    KlImproverTest<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double> kl;

    kl.SetupSchedule(schedule);

    auto &klActiveSchedule = kl.GetActiveSchedule();

    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(0), 44.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(0), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.NumSteps(), 1);
    BOOST_CHECK_CLOSE(kl.GetCurrentCost(), 44.0, 0.00001);
    BOOST_CHECK_EQUAL(kl.IsFeasible(), true);
    BOOST_CHECK_CLOSE(kl.GetCommCostF().ComputeScheduleCost(), 44.0, 0.00001);

    using KlMove = KlMoveStruct<double, VertexType>;

    KlMove move1(v1, 2.0 - 13.5, 0, 0, 1, 0);

    kl.ApplyMoveTest(move1);

    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(0), 42.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(0), 2.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.NumSteps(), 1);
    BOOST_CHECK_EQUAL(kl.IsFeasible(), false);
    BOOST_CHECK_CLOSE(kl.GetCurrentCost(), kl.GetCommCostF().ComputeScheduleCost(), 0.00001);

    KlMove move2(v2, 3.0 + 4.5 - 4.0, 0, 0, 1, 0);

    kl.ApplyMoveTest(move2);

    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(0), 39.0);         // 42-3
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(0), 5.0);    // 2+3
    BOOST_CHECK_EQUAL(klActiveSchedule.NumSteps(), 1);
    BOOST_CHECK_EQUAL(kl.IsFeasible(), false);
    BOOST_CHECK_CLOSE(kl.GetCurrentCost(), kl.GetCommCostF().ComputeScheduleCost(), 0.00001);

    kl.InsertGainHeapTest({0, 1, 2, 3, 4, 5, 6, 7});

    auto &affinity = kl.GetAffinityTable();

    BOOST_CHECK_CLOSE(affinity[v1][0][1], 2.0 - 4.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v1][1][1], 0.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v2][0][1], 3.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v3][0][1], 4.0, 0.00001);

    KlMove move3(v7, 7.0, 0, 0, 1, 0);
    kl.ApplyMoveTest(move3);
    BOOST_CHECK_EQUAL(kl.IsFeasible(), false);

    KlMove move4(v2, 7.0, 1, 0, 0, 0);
    kl.ApplyMoveTest(move4);
    BOOST_CHECK_EQUAL(kl.IsFeasible(), false);

    KlMove move5(v1, 7.0, 1, 0, 0, 0);
    kl.ApplyMoveTest(move5);
    BOOST_CHECK_EQUAL(kl.IsFeasible(), true);
}

BOOST_AUTO_TEST_CASE(KlBase2) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;

    Graph dag;

    const VertexType v1 = dag.AddVertex(2, 9, 2);
    const VertexType v2 = dag.AddVertex(3, 8, 4);
    const VertexType v3 = dag.AddVertex(4, 7, 3);
    const VertexType v4 = dag.AddVertex(5, 6, 2);
    const VertexType v5 = dag.AddVertex(6, 5, 6);
    const VertexType v6 = dag.AddVertex(7, 4, 2);
    const VertexType v7 = dag.AddVertex(8, 3, 4);
    const VertexType v8 = dag.AddVertex(9, 2, 1);

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v1, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);

    BspArchitecture<Graph> arch;

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.SetAssignedProcessors({0, 1, 0, 0, 1, 0, 0, 1});
    schedule.SetAssignedSupersteps({0, 1, 1, 1, 2, 2, 3, 3});

    schedule.UpdateNumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    KlImproverTest<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double> kl;

    kl.SetupSchedule(schedule);

    auto &klActiveSchedule = kl.GetActiveSchedule();

    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(0), 2.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(0), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(1), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(1), 3.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(2), 7.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(2), 6.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(3), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(3), 8.0);

    BOOST_CHECK_EQUAL(klActiveSchedule.NumSteps(), 4);
    BOOST_CHECK_CLOSE(kl.GetCurrentCost(), kl.GetCommCostF().ComputeScheduleCost(), 0.00001);
    BOOST_CHECK_EQUAL(kl.IsFeasible(), true);

    using KlMove = KlMoveStruct<double, VertexType>;

    KlMove move1(v1, 0.0 - 4.5, 0, 0, 1, 0);

    kl.ApplyMoveTest(move1);

    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(0), 2.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(0), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(1), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(1), 3.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(2), 7.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(2), 6.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(3), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(3), 8.0);
    BOOST_CHECK_EQUAL(kl.IsFeasible(), true);
    BOOST_CHECK_CLOSE(kl.GetCurrentCost(), kl.GetCommCostF().ComputeScheduleCost(), 0.00001);

    KlMove move2(v2, -1.0 - 8.5, 1, 1, 0, 0);

    kl.ApplyMoveTest(move2);

    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(0), 3.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(0), 2.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(1), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(1), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(2), 7.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(2), 6.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(3), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(3), 8.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.NumSteps(), 4);
    BOOST_CHECK_EQUAL(kl.IsFeasible(), false);
    BOOST_CHECK_CLOSE(kl.GetCurrentCost(), kl.GetCommCostF().ComputeScheduleCost(), 0.00001);

    KlMove moveX(v2, -2.0 + 8.5, 0, 0, 1, 0);

    kl.ApplyMoveTest(moveX);

    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(0), 5.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(0), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(1), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(1), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(2), 7.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(2), 6.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(3), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(3), 8.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.NumSteps(), 4);
    BOOST_CHECK_EQUAL(kl.IsFeasible(), true);
    BOOST_CHECK_CLOSE(kl.GetCurrentCost(), kl.GetCommCostF().ComputeScheduleCost(), 0.00001);

    kl.InsertGainHeapTest({0, 1, 2, 3, 4, 5, 6, 7});

    auto &affinity = kl.GetAffinityTable();

    BOOST_CHECK_CLOSE(affinity[v1][0][1], -4.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v1][0][2], -2.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v1][1][1], 2.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v1][1][2], 0.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v2][0][1], 9.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v2][0][2], 11.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v2][1][1], 3.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v2][1][2], 0.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v3][0][0], 0.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][0][1], 4.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][0][2], 4.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][1][0], -0.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][1][1], -4.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][1][2], -1.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v4][0][0], 0.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][0][1], 5.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][0][2], 5.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][1][0], -2.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][1][1], -6.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][1][2], -3.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v5][0][0], 9.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v5][0][1], 9.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v5][0][2], 8.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v5][1][0], 0.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v5][1][1], 0.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v5][1][2], 6.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v6][0][0], 7.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v6][0][1], 1.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v6][0][2], 6.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v6][1][0], 3.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v6][1][1], 10.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v6][1][2], 10.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v7][0][0], 8.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v7][0][1], 0.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v7][1][0], 7.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v7][1][1], 8.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v8][0][0], 8.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v8][0][1], 8.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v8][1][0], 8.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v8][1][1], 1.0, 0.00001);
}

BOOST_AUTO_TEST_CASE(KlBase3) {
    using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
    using VertexType = Graph::VertexIdx;

    Graph dag;

    const VertexType v1 = dag.AddVertex(2, 9, 2);
    const VertexType v2 = dag.AddVertex(3, 8, 4);
    const VertexType v3 = dag.AddVertex(4, 7, 3);
    const VertexType v4 = dag.AddVertex(5, 6, 2);
    const VertexType v5 = dag.AddVertex(6, 5, 6);
    const VertexType v6 = dag.AddVertex(7, 4, 2);
    const VertexType v7 = dag.AddVertex(8, 3, 4);
    const VertexType v8 = dag.AddVertex(9, 2, 1);

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v1, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);

    BspArchitecture<Graph> arch;

    BspInstance<Graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

    schedule.UpdateNumberOfSupersteps();

    using CostF = KlTotalCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint, 1, true>;
    KlImproverTest<Graph, CostF, NoLocalSearchMemoryConstraint, 1, double> kl;

    kl.SetupSchedule(schedule);

    auto &klActiveSchedule = kl.GetActiveSchedule();

    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(0), 5.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(0), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(1), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(1), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(2), 7.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(2), 6.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepMaxWork(3), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures_.StepSecondMaxWork(3), 8.0);

    BOOST_CHECK_EQUAL(klActiveSchedule.NumSteps(), 4);
    BOOST_CHECK_EQUAL(klActiveSchedule.IsFeasible(), true);

    kl.InsertGainHeapTestPenalty({0, 1, 2, 3, 4, 5, 6, 7});

    auto &affinity = kl.GetAffinityTable();

    BOOST_CHECK_CLOSE(affinity[v1][0][1], 1.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v1][0][2], 3.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v1][1][1], 2.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v1][1][2], 16.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v2][0][1], 15, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v2][0][2], 11.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v2][1][1], 3.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v2][1][2], 0.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v3][0][0], 5.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][0][1], 4.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][0][2], 9.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][1][0], -0.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][1][1], -4.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v3][1][2], 4, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v4][0][0], 5.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][0][1], 5.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][0][2], 5.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][1][0], -2.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][1][1], -6.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v4][1][2], -3.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v5][0][0], 9.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v5][0][1], 9.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v5][0][2], 13.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v5][1][0], 5.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v5][1][1], 0.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v5][1][2], 6.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v6][0][0], 7.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v6][0][1], 1.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v6][0][2], 6.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v6][1][0], 9.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v6][1][1], 10.5, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v6][1][2], 10.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v7][0][0], 8.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v7][0][1], 0.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v7][1][0], 7.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v7][1][1], 8.0, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v8][0][0], 14.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v8][0][1], 8.5, 0.00001);

    BOOST_CHECK_CLOSE(affinity[v8][1][0], 8.0, 0.00001);
    BOOST_CHECK_CLOSE(affinity[v8][1][1], 1.0, 0.00001);
}

// BOOST_AUTO_TEST_CASE(KlImprover_incremental_update_test) {

//     using graph = ComputationalDagEdgeIdxVectorImplDefIntT;
//     using VertexType = graph::VertexIdx;
//     using kl_move = KlMoveStruct<double, VertexType>;

//     graph dag;

//     const VertexType v1 = dag.AddVertex(2, 9, 2);
//     const VertexType v2 = dag.AddVertex(3, 8, 4);
//     const VertexType v3 = dag.AddVertex(4, 7, 3);
//     const VertexType v4 = dag.AddVertex(5, 6, 2);
//     const VertexType v5 = dag.AddVertex(6, 5, 6);
//     const VertexType v6 = dag.AddVertex(7, 4, 2);
//     const VertexType v7 = dag.AddVertex(8, 3, 4);
//     const VertexType v8 = dag.AddVertex(9, 2, 1);

//     dag.AddEdge(v1, v2, 2);
//     dag.AddEdge(v1, v3, 2);
//     dag.AddEdge(v1, v4, 2);
//     dag.AddEdge(v2, v5, 12);
//     dag.AddEdge(v3, v5, 6);
//     dag.AddEdge(v3, v6, 7);
//     dag.AddEdge(v5, v8, 9);
//     dag.AddEdge(v4, v8, 9);

//     BspArchitecture<graph> arch;

//     BspInstance<graph> instance(dag, arch);

//     BspSchedule schedule(instance);

//     schedule.SetAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
//     schedule.SetAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

//     schedule.UpdateNumberOfSupersteps();

//     using cost_f = KlTotalCommCostFunction<graph, double, NoLocalSearchMemoryConstraint, 1, true>;
//     using KlImproverTest = KlImproverTest<graph, cost_f, NoLocalSearchMemoryConstraint, 1, double>;
//     KlImproverTest kl;

//     kl.SetupSchedule(schedule);

//     auto node_selection = kl.InsertGainHeapTest({0, 1, 2, 3, 4, 5, 6, 7});

//     std::set<VertexType> nodes_to_check = {0, 1, 2, 3, 4, 5, 6, 7};
//     auto& affinity = kl.GetAffinityTable();

//     kl_move move_1(v7, 0.0, 0, 3, 0, 2);
//     kl.UpdateAffinityTableTest(move_1, node_selection);

//     BspSchedule<graph> test_sched_1(instance);
//     kl.GetActiveSchedule_test(test_sched_1);
//     KlImproverTest kl_1;
//     kl_1.SetupSchedule(test_sched_1);
//     kl_1.InsertGainHeapTest({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v7);

//     check_equal_affinity_table(affinity, kl_1.GetAffinityTable(), nodes_to_check);

//     kl_move move_2(v4, 0.0, 0, 1 , 0, 2);
//     kl.UpdateAffinityTableTest(move_2, node_selection);

//     BspSchedule<graph> test_sched_2(instance);
//     kl.GetActiveSchedule_test(test_sched_2);
//     KlImproverTest kl_2;
//     kl_2.SetupSchedule(test_sched_2);
//     kl_2.InsertGainHeapTest({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v4);

//     check_equal_affinity_table(affinity, kl_2.GetAffinityTable(), nodes_to_check);

//     kl_move move_3(v2, 0.0, 1, 0 , 0, 0);
//     kl.UpdateAffinityTableTest(move_3, node_selection);

//     BspSchedule<graph> test_sched_3(instance);
//     kl.GetActiveSchedule_test(test_sched_3);
//     KlImproverTest kl_3;
//     kl_3.SetupSchedule(test_sched_3);
//     kl_3.InsertGainHeapTest({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v2);

//     check_equal_affinity_table(affinity, kl_3.GetAffinityTable(), nodes_to_check);

//     kl_move move_4(v6, 0.0, 0, 2 , 1, 3);
//     kl.UpdateAffinityTableTest(move_4, node_selection);

//     BspSchedule<graph> test_sched_4(instance);
//     kl.GetActiveSchedule_test(test_sched_4);
//     KlImproverTest kl_4;
//     kl_4.SetupSchedule(test_sched_4);
//     kl_4.InsertGainHeapTest({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v6);

//     check_equal_affinity_table(affinity, kl_4.GetAffinityTable(), nodes_to_check);

//     kl_move move_5(v8, 0.0, 1, 3 , 0, 2);
//     kl.UpdateAffinityTableTest(move_5, node_selection);

//     BspSchedule<graph> test_sched_5(instance);
//     kl.GetActiveSchedule_test(test_sched_5);
//     KlImproverTest kl_5;
//     kl_5.SetupSchedule(test_sched_5);
//     kl_5.InsertGainHeapTest({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v8);

//     check_equal_affinity_table(affinity, kl_5.GetAffinityTable(), nodes_to_check);

//     kl_move move_6(v3, 0.0, 0, 1 , 1, 1);
//     kl.UpdateAffinityTableTest(move_6, node_selection);

//     BspSchedule<graph> test_sched_6(instance);
//     kl.GetActiveSchedule_test(test_sched_6);
//     KlImproverTest kl_6;
//     kl_6.SetupSchedule(test_sched_6);
//     kl_6.InsertGainHeapTest({0, 1, 2, 3, 4, 5, 6, 7});

//     nodes_to_check.erase(v3);

//     check_equal_affinity_table(affinity, kl_6.GetAffinityTable(), nodes_to_check);

// };

// BOOST_AUTO_TEST_CASE(kl_total_comm_large_test_graphs) {
//     std::vector<std::string> filenames_graph = large_spaa_graphs();
//     using graph = ComputationalDagEdgeIdxVectorImplDefIntT;

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {
//         GreedyBspScheduler<ComputationalDagEdgeIdxVectorImplDefIntT> test_scheduler;
//         BspInstance<graph> instance;
//         bool status_graph = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / filename_graph).string(),
//                                                                             instance.GetComputationalDag());

//         instance.GetArchitecture().SetSynchronisationCosts(500);
//         instance.GetArchitecture().SetCommunicationCosts(5);
//         instance.GetArchitecture().SetNumberOfProcessors(4);

//         std::vector<std::vector<int>> send_cost = {{0,1,4,4},
//                                                    {1,0,4,4},
//                                                    {4,4,0,1},
//                                                    {4,4,1,0}};

//         instance.GetArchitecture().SetSendCosts(send_cost);

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         }

//         add_mem_weights(instance.GetComputationalDag());

//         BspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.ComputeSchedule(schedule);

//         schedule.UpdateNumberOfSupersteps();

//         std::cout << "initial scedule with costs: " << schedule.computeTotalCosts() << " and " << schedule.NumberOfSupersteps()
//         << " number of supersteps"<< std::endl;

//         BspSchedule<graph> schedule_2(schedule);

//         BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
//         BOOST_CHECK_EQUAL(&schedule.GetInstance(), &instance);
//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

//         KlTotalCommImprover<graph,NoLocalSearchMemoryConstraint,1,true> kl;

//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl.ImproveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();

//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "kl new finished in " << duration << " seconds, costs: " << schedule.computeTotalCosts() << " with " <<
//         schedule.NumberOfSupersteps() << " number of supersteps"<< std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule.SatisfiesPrecedenceConstraints(), true);

//         // kl_total_comm_test<graph> kl_old;

//         // start_time = std::chrono::high_resolution_clock::now();
//         // status = kl_old.improve_schedule_test_2(schedule_2);
//         // finish_time = std::chrono::high_resolution_clock::now();

//         // duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         // std::cout << "kl old finished in " << duration << " seconds, costs: " << schedule_2.computeTotalCosts() << " with "
//         << schedule_2.NumberOfSupersteps() << " number of supersteps"<< std::endl;

//         // BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         // BOOST_CHECK_EQUAL(schedule_2.SatisfiesPrecedenceConstraints(), true);

//     }
// }

// BOOST_AUTO_TEST_CASE(kl_total_comm_large_test_graphs_mt) {
//     std::vector<std::string> filenames_graph = large_spaa_graphs();
//     using graph = ComputationalDagEdgeIdxVectorImplDefIntT;

//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {
//         GreedyBspScheduler<ComputationalDagEdgeIdxVectorImplDefIntT> test_scheduler;
//         BspInstance<graph> instance;
//         bool status_graph = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / filename_graph).string(),
//                                                                             instance.GetComputationalDag());

//         instance.GetArchitecture().SetSynchronisationCosts(500);
//         instance.GetArchitecture().SetCommunicationCosts(5);
//         instance.GetArchitecture().SetNumberOfProcessors(4);

//         std::vector<std::vector<int>> send_cost = {{0,1,4,4},
//                                                    {1,0,4,4},
//                                                    {4,4,0,1},
//                                                    {4,4,1,0}};

//         instance.GetArchitecture().SetSendCosts(send_cost);

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         }

//         add_mem_weights(instance.GetComputationalDag());

//         BspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.ComputeSchedule(schedule);

//         schedule.UpdateNumberOfSupersteps();

//         std::cout << "initial scedule with costs: " << schedule.computeTotalCosts() << " and " << schedule.NumberOfSupersteps()
//         << " number of supersteps"<< std::endl;

//         BspSchedule<graph> schedule_2(schedule);

//         BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
//         BOOST_CHECK_EQUAL(&schedule.GetInstance(), &instance);
//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

//         KlTotalCommImprover_mt<graph,NoLocalSearchMemoryConstraint,1,true> kl;

//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl.ImproveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();

//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "kl new finished in " << duration << " seconds, costs: " << schedule.computeTotalCosts() << " with " <<
//         schedule.NumberOfSupersteps() << " number of supersteps"<< std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule.SatisfiesPrecedenceConstraints(), true);

//         // kl_total_comm_test<graph> kl_old;

//         // start_time = std::chrono::high_resolution_clock::now();
//         // status = kl_old.improve_schedule_test_2(schedule_2);
//         // finish_time = std::chrono::high_resolution_clock::now();

//         // duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         // std::cout << "kl old finished in " << duration << " seconds, costs: " << schedule_2.computeTotalCosts() << " with "
//         << schedule_2.NumberOfSupersteps() << " number of supersteps"<< std::endl;

//         // BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         // BOOST_CHECK_EQUAL(schedule_2.SatisfiesPrecedenceConstraints(), true);

//     }
// }
