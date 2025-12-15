
#define BOOST_TEST_MODULE kl_bsp_affinity
#include <boost/test/unit_test.hpp>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_improver_test.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;
using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using KlActiveScheduleT = kl_active_schedule<Graph, double, no_local_search_memory_constraint>;

BOOST_AUTO_TEST_CASE(SimpleParentChildTest) {
    using VertexType = Graph::VertexIdx;

    Graph dag;
    const VertexType v0 = dag.AddVertex(10, 5, 2);    // work=10, mem=5, comm=2
    const VertexType v1 = dag.AddVertex(8, 4, 1);     // work=8, mem=4, comm=1
    dag.AddEdge(v0, v1, 3);                           // edge weight=3

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);

    BspInstance<Graph> instance(dag, arch);
    instance.SetCommunicationCosts(10);    // comm multiplier
    instance.SetSynchronisationCosts(5);

    BspSchedule schedule(instance);
    schedule.setAssignedProcessors({0, 1});    // v0 on p0, v1 on p1
    schedule.setAssignedSupersteps({0, 1});    // v0 in step 0, v1 in step 1
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    // Insert only v0 into gain heap to control which node moves
    auto nodeSelection = kl.insert_gain_heap_test({0});

    // Run one iteration - this will move v0 to its best position
    auto recomputeMaxGain = kl.run_inner_iteration_test();

    // Compare costs after move
    double afterRecomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterTracked = kl.get_current_cost();

    BOOST_CHECK_CLOSE(afterRecomputed, afterTracked, 0.00001);
}

/**
 * Helper to validate comm datastructures by comparing with freshly computed ones
 */
template <typename Graph>
bool ValidateCommDatastructures(const max_comm_datastructure<Graph, double, KlActiveScheduleT> &commDsIncremental,
                                KlActiveScheduleT &activeSched,
                                const BspInstance<Graph> &instance,
                                const std::string &context) {
    // 1. Clone Schedule
    BspSchedule<Graph> currentSchedule(instance);
    activeSched.write_schedule(currentSchedule);

    // 2. Fresh Computation
    KlActiveScheduleT klSchedFresh;
    klSchedFresh.initialize(currentSchedule);

    max_comm_datastructure<Graph, double, KlActiveScheduleT> commDsFresh;
    commDsFresh.initialize(klSchedFresh);

    // Compute for all steps
    unsigned maxStep = currentSchedule.NumberOfSupersteps();
    commDsFresh.compute_comm_datastructures(0, maxStep > 0 ? maxStep - 1 : 0);

    bool allMatch = true;
    // std::cout << "\nValidating comm datastructures " << context << ":" << std::endl;

    // 3. Validate Comm Costs
    for (unsigned step = 0; step < maxStep; ++step) {
        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            auto sendInc = commDsIncremental.step_proc_send(step, p);
            auto sendFresh = commDsFresh.step_proc_send(step, p);
            auto recvInc = commDsIncremental.step_proc_receive(step, p);
            auto recvFresh = commDsFresh.step_proc_receive(step, p);

            if (std::abs(sendInc - sendFresh) > 1e-6 || std::abs(recvInc - recvFresh) > 1e-6) {
                allMatch = false;
                std::cout << "  MISMATCH at step " << step << " proc " << p << ":" << std::endl;
                std::cout << "    Incremental: send=" << sendInc << ", recv=" << recvInc << std::endl;
                std::cout << "    Fresh:       send=" << sendFresh << ", recv=" << recvFresh << std::endl;
            }
        }
    }

    // 4. Validate Lambda Maps
    for (const auto v : instance.Vertices()) {
        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
            unsigned countInc = 0;
            if (commDsIncremental.node_lambda_map.has_proc_entry(v, p)) {
                countInc = commDsIncremental.node_lambda_map.get_proc_entry(v, p);
            }

            unsigned countFresh = 0;
            if (commDsFresh.node_lambda_map.has_proc_entry(v, p)) {
                countFresh = commDsFresh.node_lambda_map.get_proc_entry(v, p);
            }

            if (countInc != countFresh) {
                allMatch = false;
                std::cout << "  LAMBDA MISMATCH at node " << v << " proc " << p << ":" << std::endl;
                std::cout << "    Incremental: " << countInc << std::endl;
                std::cout << "    Fresh:       " << countFresh << std::endl;
            }
        }
    }

    return allMatch;
}

/**
 * Helper to validate affinity tables by comparing with freshly computed ones
 */
template <typename GraphT, typename CommCostFunctionT, typename MemoryConstraintT, unsigned windowSize, typename CostT>
bool ValidateAffinityTables(KlImproverTest<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT> &klIncremental,
                            const BspInstance<GraphT> &instance,
                            const std::string &context) {
    // 1. Get current schedule from incremental
    BspSchedule<GraphT> currentSchedule(instance);
    klIncremental.get_active_schedule_test(currentSchedule);

    // 2. Create fresh kl_improver and compute all affinities from scratch
    KlImproverTest<GraphT, CommCostFunctionT, MemoryConstraintT, windowSize, CostT> klFresh;
    klFresh.setup_schedule(currentSchedule);

    // Get selected nodes from incremental
    std::vector<VertexIdxT<GraphT>> selectedNodes;

    const size_t activeCount = klIncremental.get_affinity_table().size();
    for (size_t i = 0; i < activeCount; ++i) {
        selectedNodes.push_back(klIncremental.get_affinity_table().get_selected_nodes()[i]);
    }

    std::cout << "\n  [" << context << "] Validating " << selectedNodes.size() << " selected nodes: { ";
    for (const auto n : selectedNodes) {
        std::cout << n << " ";
    }
    std::cout << "}" << std::endl;

    // Compute affinities for all selected nodes
    klFresh.insert_gain_heap_test(selectedNodes);

    bool allMatch = true;
    const unsigned numProcs = instance.NumberOfProcessors();
    const unsigned numSteps = klIncremental.get_active_schedule().num_steps();

    // 3. Compare affinity tables for each selected node

    for (const auto &node : selectedNodes) {
        const auto &affinityInc = klIncremental.get_affinity_table().get_affinity_table(node);
        const auto &affinityFresh = klFresh.get_affinity_table().get_affinity_table(node);

        unsigned nodeStep = klIncremental.get_active_schedule().assigned_superstep(node);

        for (unsigned p = 0; p < numProcs; ++p) {
            if (p >= affinityInc.size() || p >= affinityFresh.size()) {
                continue;
            }

            for (unsigned idx = 0; idx < affinityInc[p].size() && idx < affinityFresh[p].size(); ++idx) {
                int stepOffset = static_cast<int>(idx) - static_cast<int>(windowSize);
                int targetStepSigned = static_cast<int>(nodeStep) + stepOffset;

                // Skip affinities for supersteps that don't exist
                if (targetStepSigned < 0 || targetStepSigned >= static_cast<int>(numSteps)) {
                    continue;
                }

                double valInc = affinityInc[p][idx];
                double valFresh = affinityFresh[p][idx];

                if (std::abs(valInc - valFresh) > 1e-4) {
                    allMatch = false;

                    std::cout << "  AFFINITY MISMATCH [" << context << "]: node=" << node << " to P" << p << " S"
                              << targetStepSigned << " (offset=" << stepOffset << ")" << std::endl;
                    std::cout << "    Incremental: " << valInc << std::endl;
                    std::cout << "    Fresh:       " << valFresh << std::endl;
                    std::cout << "    Difference:  " << (valInc - valFresh) << std::endl;
                }
            }
        }
    }

    return allMatch;
}

BOOST_AUTO_TEST_CASE(TestUpdateDatastructureAfterMove) {
    Graph dag;

    // Create 6 vertices with specific comm weights
    dag.AddVertex(1, 10, 1);    // 0
    dag.AddVertex(1, 1, 1);     // 1
    dag.AddVertex(1, 5, 1);     // 2
    dag.AddVertex(1, 1, 1);     // 3
    dag.AddVertex(1, 2, 1);     // 4
    dag.AddVertex(1, 1, 1);     // 5

    // Add edges
    dag.AddEdge(0, 1, 1);
    dag.AddEdge(2, 3, 1);
    dag.AddEdge(4, 5, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(3);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    // Schedule:
    // Proc 0: Node 0, 4, 5
    // Proc 1: Node 1, 2
    // Proc 2: Node 3
    schedule.setAssignedProcessors({0, 1, 1, 2, 0, 0});
    // Steps: 0, 1, 0, 1, 0, 0
    schedule.setAssignedSupersteps({0, 1, 0, 1, 0, 0});
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({0});
    kl.run_inner_iteration_test();

    double afterRecomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterTracked = kl.get_current_cost();

    BOOST_CHECK(ValidateCommDatastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_update_datastructure_after_move"));
    BOOST_CHECK_CLOSE(afterRecomputed, afterTracked, 0.00001);
}

BOOST_AUTO_TEST_CASE(TestMultipleSequentialMoves) {
    Graph dag;

    // Create a linear chain: 0 -> 1 -> 2 -> 3
    dag.AddVertex(1, 10, 1);    // 0
    dag.AddVertex(1, 8, 1);     // 1
    dag.AddVertex(1, 6, 1);     // 2
    dag.AddVertex(1, 4, 1);     // 3

    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);
    dag.AddEdge(2, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    schedule.setAssignedProcessors({0, 1, 2, 3});
    schedule.setAssignedSupersteps({0, 0, 0, 0});
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({1});
    kl.run_inner_iteration_test();

    double afterMove1Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove1Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_multiple_sequential_moves_1"));
    BOOST_CHECK_CLOSE(afterMove1Recomputed, afterMove1Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove2Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove2Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_multiple_sequential_moves_2"));
    BOOST_CHECK_CLOSE(afterMove2Recomputed, afterMove2Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove3Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove3Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_multiple_sequential_moves_3"));
    BOOST_CHECK_CLOSE(afterMove3Recomputed, afterMove3Tracked, 0.00001);

    // After: Node 0 has 3 local children
    // Send cost = 10 * 0 = 0 (all local)
    // Work cost 4
    BOOST_CHECK_CLOSE(afterMove3Tracked, 4.0, 0.00001);
}

BOOST_AUTO_TEST_CASE(TestNodeWithMultipleChildren) {
    Graph dag;

    // Tree structure: Node 0 has three children (1, 2, 3)
    dag.AddVertex(1, 1, 1);    // 0
    dag.AddVertex(1, 1, 1);    // 1
    dag.AddVertex(1, 1, 1);    // 2
    dag.AddVertex(1, 1, 1);    // 3

    dag.AddEdge(0, 1, 1);
    dag.AddEdge(0, 2, 1);
    dag.AddEdge(0, 3, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    schedule.setAssignedProcessors({0, 1, 2, 3});
    schedule.setAssignedSupersteps({0, 0, 0, 0});
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({1});
    kl.get_comm_cost_f().compute_schedule_cost();
    kl.run_inner_iteration_test();

    double afterMove1Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove1Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_node_with_multiple_children"));
    BOOST_CHECK_CLOSE(afterMove1Recomputed, afterMove1Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove2Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove2Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_node_with_multiple_children_2"));
    BOOST_CHECK_CLOSE(afterMove2Recomputed, afterMove2Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove3Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove3Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_node_with_multiple_children_3"));
    BOOST_CHECK_CLOSE(afterMove3Recomputed, afterMove3Tracked, 0.00001);

    // After: Node 0 has 3 local children
    // Send cost = 10 * 0 = 0 (all local)
    // Work cost 4
    BOOST_CHECK_CLOSE(afterMove3Tracked, 4.0, 0.00001);
}

BOOST_AUTO_TEST_CASE(TestCrossStepMoves) {
    Graph dag;

    // 0 -> 1 -> 2
    dag.AddVertex(1, 10, 1);    // 0
    dag.AddVertex(1, 8, 1);     // 1
    dag.AddVertex(1, 6, 1);     // 2

    dag.AddEdge(0, 1, 1);
    dag.AddEdge(1, 2, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    schedule.setAssignedProcessors({0, 1, 0});
    schedule.setAssignedSupersteps({0, 1, 2});
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({1});
    kl.run_inner_iteration_test();

    double afterMove1Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove1Tracked = kl.get_current_cost();
    BOOST_CHECK(
        ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_cross_step_moves_1"));
    BOOST_CHECK_CLOSE(afterMove1Recomputed, afterMove1Tracked, 0.00001);
}

BOOST_AUTO_TEST_CASE(TestComplexScenario) {
    std::cout << "Test case complex scenario" << std::endl;
    Graph dag;

    const auto v1 = dag.AddVertex(2, 9, 2);
    const auto v2 = dag.AddVertex(3, 8, 4);
    const auto v3 = dag.AddVertex(4, 7, 3);
    const auto v4 = dag.AddVertex(5, 6, 2);
    const auto v5 = dag.AddVertex(6, 5, 6);
    const auto v6 = dag.AddVertex(7, 4, 2);
    dag.AddVertex(8, 3, 4);                    // v7 (index 6)
    const auto v8 = dag.AddVertex(9, 2, 1);    // v8 (index 7)

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v1, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);    // P0, P1
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({v3, v1});
    kl.run_inner_iteration_test();

    double afterMove1Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove1Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move1"));
    BOOST_CHECK_CLOSE(afterMove1Recomputed, afterMove1Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove2Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove2Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move2"));
    BOOST_CHECK(ValidateAffinityTables(kl, instance, "complex_move2"));
    BOOST_CHECK_CLOSE(afterMove2Recomputed, afterMove2Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove3Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove3Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move3"));
    BOOST_CHECK_CLOSE(afterMove3Recomputed, afterMove3Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove4Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove4Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move4"));
    BOOST_CHECK_CLOSE(afterMove4Recomputed, afterMove4Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove5Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove5Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move5"));
    BOOST_CHECK_CLOSE(afterMove5Recomputed, afterMove5Tracked, 0.00001);
}

BOOST_AUTO_TEST_CASE(TestComplexScenarioOnlyCompute) {
    Graph dag;

    const auto v1 = dag.AddVertex(2, 9, 2);
    const auto v2 = dag.AddVertex(3, 8, 4);
    const auto v3 = dag.AddVertex(4, 7, 3);
    const auto v4 = dag.AddVertex(5, 6, 2);
    const auto v5 = dag.AddVertex(6, 5, 6);
    const auto v6 = dag.AddVertex(7, 4, 2);
    const auto v7 = dag.AddVertex(8, 3, 4);    // v7 (index 6)
    const auto v8 = dag.AddVertex(9, 2, 1);    // v8 (index 7)

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v1, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);    // P0, P1
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({v1});
    kl.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move1"));
    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    KlImproverTest kl2;
    kl2.setup_schedule(schedule);

    kl2.insert_gain_heap_test({v2});
    kl2.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl2.get_comm_cost_f().comm_ds, kl2.get_active_schedule(), instance, "complex_move2"));
    BOOST_CHECK_CLOSE(kl2.get_comm_cost_f().compute_schedule_cost_test(), kl2.get_current_cost(), 0.00001);

    KlImproverTest kl3;
    kl3.setup_schedule(schedule);

    kl3.insert_gain_heap_test({v3});
    kl3.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl3.get_comm_cost_f().comm_ds, kl3.get_active_schedule(), instance, "complex_move3"));
    BOOST_CHECK_CLOSE(kl3.get_comm_cost_f().compute_schedule_cost_test(), kl3.get_current_cost(), 0.00001);

    KlImproverTest kl4;
    kl4.setup_schedule(schedule);

    kl4.insert_gain_heap_test({v4});
    kl4.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl4.get_comm_cost_f().comm_ds, kl4.get_active_schedule(), instance, "complex_move4"));
    BOOST_CHECK_CLOSE(kl4.get_comm_cost_f().compute_schedule_cost_test(), kl4.get_current_cost(), 0.00001);

    KlImproverTest kl5;
    kl5.setup_schedule(schedule);

    kl5.insert_gain_heap_test({v5});
    kl5.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl5.get_comm_cost_f().comm_ds, kl5.get_active_schedule(), instance, "complex_move5"));
    BOOST_CHECK_CLOSE(kl5.get_comm_cost_f().compute_schedule_cost_test(), kl5.get_current_cost(), 0.00001);

    KlImproverTest kl6;
    kl6.setup_schedule(schedule);

    kl6.insert_gain_heap_test({v6});
    kl6.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl6.get_comm_cost_f().comm_ds, kl6.get_active_schedule(), instance, "complex_move6"));
    BOOST_CHECK_CLOSE(kl6.get_comm_cost_f().compute_schedule_cost_test(), kl6.get_current_cost(), 0.00001);

    KlImproverTest kl7;
    kl7.setup_schedule(schedule);

    kl7.insert_gain_heap_test({v7});
    kl7.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl7.get_comm_cost_f().comm_ds, kl7.get_active_schedule(), instance, "complex_move7"));
    BOOST_CHECK_CLOSE(kl7.get_comm_cost_f().compute_schedule_cost_test(), kl7.get_current_cost(), 0.00001);

    KlImproverTest kl8;
    kl8.setup_schedule(schedule);

    kl8.insert_gain_heap_test({v8});
    kl8.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl8.get_comm_cost_f().comm_ds, kl8.get_active_schedule(), instance, "complex_move8"));
    BOOST_CHECK_CLOSE(kl8.get_comm_cost_f().compute_schedule_cost_test(), kl8.get_current_cost(), 0.00001);
}

BOOST_AUTO_TEST_CASE(TestComplexScenarioOnlyCompute2) {
    Graph dag;

    const auto v1 = dag.AddVertex(2, 9, 2);
    const auto v2 = dag.AddVertex(3, 8, 4);
    const auto v3 = dag.AddVertex(4, 7, 3);
    const auto v4 = dag.AddVertex(5, 6, 2);
    const auto v5 = dag.AddVertex(6, 5, 6);
    const auto v6 = dag.AddVertex(7, 4, 2);
    const auto v7 = dag.AddVertex(8, 3, 4);    // v7 (index 6)
    const auto v8 = dag.AddVertex(9, 2, 1);    // v8 (index 7)

    dag.AddEdge(v1, v2, 2);
    dag.AddEdge(v1, v5, 2);
    dag.AddEdge(v1, v6, 2);
    dag.AddEdge(v1, v3, 2);
    dag.AddEdge(v1, v4, 2);
    dag.AddEdge(v2, v5, 12);
    dag.AddEdge(v2, v6, 2);
    dag.AddEdge(v2, v7, 2);
    dag.AddEdge(v2, v8, 2);
    dag.AddEdge(v3, v5, 6);
    dag.AddEdge(v3, v6, 7);
    dag.AddEdge(v3, v7, 2);
    dag.AddEdge(v3, v8, 2);
    dag.AddEdge(v5, v8, 9);
    dag.AddEdge(v4, v8, 9);
    dag.AddEdge(v5, v7, 2);
    dag.AddEdge(v6, v7, 2);
    dag.AddEdge(v7, v8, 2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);    // P0, P1
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({v1});
    kl.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move1"));
    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    KlImproverTest kl2;
    kl2.setup_schedule(schedule);

    kl2.insert_gain_heap_test({v2});
    kl2.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl2.get_comm_cost_f().comm_ds, kl2.get_active_schedule(), instance, "complex_move2"));
    BOOST_CHECK_CLOSE(kl2.get_comm_cost_f().compute_schedule_cost_test(), kl2.get_current_cost(), 0.00001);

    KlImproverTest kl3;
    kl3.setup_schedule(schedule);

    kl3.insert_gain_heap_test({v3});
    kl3.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl3.get_comm_cost_f().comm_ds, kl3.get_active_schedule(), instance, "complex_move3"));
    BOOST_CHECK_CLOSE(kl3.get_comm_cost_f().compute_schedule_cost_test(), kl3.get_current_cost(), 0.00001);

    KlImproverTest kl4;
    kl4.setup_schedule(schedule);

    kl4.insert_gain_heap_test({v4});
    kl4.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl4.get_comm_cost_f().comm_ds, kl4.get_active_schedule(), instance, "complex_move4"));
    BOOST_CHECK_CLOSE(kl4.get_comm_cost_f().compute_schedule_cost_test(), kl4.get_current_cost(), 0.00001);

    KlImproverTest kl5;
    kl5.setup_schedule(schedule);

    kl5.insert_gain_heap_test({v5});
    kl5.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl5.get_comm_cost_f().comm_ds, kl5.get_active_schedule(), instance, "complex_move5"));
    BOOST_CHECK_CLOSE(kl5.get_comm_cost_f().compute_schedule_cost_test(), kl5.get_current_cost(), 0.00001);

    KlImproverTest kl6;
    kl6.setup_schedule(schedule);

    kl6.insert_gain_heap_test({v6});
    kl6.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl6.get_comm_cost_f().comm_ds, kl6.get_active_schedule(), instance, "complex_move6"));
    BOOST_CHECK_CLOSE(kl6.get_comm_cost_f().compute_schedule_cost_test(), kl6.get_current_cost(), 0.00001);

    KlImproverTest kl7;
    kl7.setup_schedule(schedule);

    kl7.insert_gain_heap_test({v7});
    kl7.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl7.get_comm_cost_f().comm_ds, kl7.get_active_schedule(), instance, "complex_move7"));
    BOOST_CHECK_CLOSE(kl7.get_comm_cost_f().compute_schedule_cost_test(), kl7.get_current_cost(), 0.00001);

    KlImproverTest kl8;
    kl8.setup_schedule(schedule);

    kl8.insert_gain_heap_test({v8});
    kl8.run_inner_iteration_test();

    BOOST_CHECK(ValidateCommDatastructures(kl8.get_comm_cost_f().comm_ds, kl8.get_active_schedule(), instance, "complex_move8"));
    BOOST_CHECK_CLOSE(kl8.get_comm_cost_f().compute_schedule_cost_test(), kl8.get_current_cost(), 0.00001);
}

BOOST_AUTO_TEST_CASE(TestGridGraphComplexMoves) {
    // Construct 5x5 Grid Graph (25 nodes, indices 0-24)
    Graph dag = osp::construct_grid_dag<Graph>(5, 5);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(4);    // P0..P3
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    // Assign Processors and Supersteps
    std::vector<unsigned> procs(25);
    std::vector<unsigned> steps(25);

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

    // Override: Node 7 (1,2) to P3, S1.
    procs[7] = 3;
    steps[7] = 1;

    schedule.setAssignedProcessors(procs);
    schedule.setAssignedSupersteps(steps);
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({12, 8, 7});
    kl.run_inner_iteration_test();

    double afterMove1Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove1Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "grid_move1"));
    BOOST_CHECK_CLOSE(afterMove1Recomputed, afterMove1Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove2Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove2Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "grid_move2"));
    BOOST_CHECK_CLOSE(afterMove2Recomputed, afterMove2Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove3Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove3Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "grid_move3"));
    BOOST_CHECK_CLOSE(afterMove3Recomputed, afterMove3Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove4Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove4Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "grid_move4"));
    BOOST_CHECK_CLOSE(afterMove4Recomputed, afterMove4Tracked, 0.00001);
}

BOOST_AUTO_TEST_CASE(TestButterflyGraphMoves) {
    // Stages=2 -> 3 levels of 4 nodes each = 12 nodes.
    // Level 0: 0-3. Level 1: 4-7. Level 2: 8-11.
    Graph dag = osp::construct_butterfly_dag<Graph>(2);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    // Assign:
    // Level 0: P0, Step 0
    // Level 1: P1, Step 1
    // Level 2: P0, Step 2
    std::vector<unsigned> procs(12);
    std::vector<unsigned> steps(12);
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

    schedule.setAssignedProcessors(procs);
    schedule.setAssignedSupersteps(steps);
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({4, 6, 0});
    kl.run_inner_iteration_test();

    double afterMove1Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove1Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "butterfly_move1"));
    BOOST_CHECK_CLOSE(afterMove1Recomputed, afterMove1Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove2Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove2Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "butterfly_move2"));
    BOOST_CHECK_CLOSE(afterMove2Recomputed, afterMove2Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove3Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove3Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "butterfly_move3"));
    BOOST_CHECK_CLOSE(afterMove3Recomputed, afterMove3Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove4Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove4Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "butterfly_move4"));
    BOOST_CHECK_CLOSE(afterMove4Recomputed, afterMove4Tracked, 0.00001);
}

BOOST_AUTO_TEST_CASE(TestLadderGraphMoves) {
    // Ladder with 5 rungs -> 6 pairs of nodes = 12 nodes.
    // Pairs: (0,1), (2,3), ... (10,11).
    Graph dag = osp::construct_ladder_dag<Graph>(5);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    BspSchedule<Graph> schedule(instance);

    // Assign:
    // Even nodes (Left rail): P0
    // Odd nodes (Right rail): P1
    // Steps: Pair i at Step i.
    std::vector<unsigned> procs(12);
    std::vector<unsigned> steps(12);
    for (unsigned i = 0; i < 6; ++i) {
        procs[2 * i] = 0;
        steps[2 * i] = i;
        procs[2 * i + 1] = 1;
        steps[2 * i + 1] = i;
    }

    schedule.setAssignedProcessors(procs);
    schedule.setAssignedSupersteps(steps);
    schedule.updateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({1, 3, 0, 2});
    kl.run_inner_iteration_test();

    double afterMove1Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove1Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "ladder_move1"));
    BOOST_CHECK_CLOSE(afterMove1Recomputed, afterMove1Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove2Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove2Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "ladder_move2"));
    BOOST_CHECK_CLOSE(afterMove2Recomputed, afterMove2Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove3Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove3Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "ladder_move3"));
    BOOST_CHECK_CLOSE(afterMove3Recomputed, afterMove3Tracked, 0.00001);

    kl.run_inner_iteration_test();

    double afterMove4Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double afterMove4Tracked = kl.get_current_cost();
    BOOST_CHECK(ValidateCommDatastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "ladder_move4"));
    BOOST_CHECK_CLOSE(afterMove4Recomputed, afterMove4Tracked, 0.00001);
}
