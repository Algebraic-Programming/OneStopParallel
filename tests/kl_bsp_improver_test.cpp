
#define BOOST_TEST_MODULE kl_bsp_improver
#include <boost/test/unit_test.hpp>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/CoarsenRefineSchedulers/MultiLevelHillClimbing.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing_for_comm_schedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_improver_test.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include_mt.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

template <typename GraphT>
void AddMemWeights(GraphT &dag) {
    int memWeight = 1;
    int commWeight = 7;

    for (const auto &v : dag.vertices()) {
        dag.set_vertex_work_weight(v, static_cast<VMemwT<GraphT>>(memWeight++ % 10 + 2));
        dag.set_vertex_mem_weight(v, static_cast<VMemwT<GraphT>>(memWeight++ % 10 + 2));
        dag.set_vertex_comm_weight(v, static_cast<VCommwT<GraphT>>(commWeight++ % 10 + 2));
    }
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

    using CommCostT = KlBspCommCostFunction<Graph, double, NoLocalSearchMemoryConstraint>;
    using KlImproverTest = KlImproverTest<Graph, CommCostT>;

    KlImproverTest kl;

    kl.SetupSchedule(schedule);

    auto &klActiveSchedule = kl.GetActiveSchedule();

    // Verify work datastructures are set up correctly
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures.StepMaxWork(0), 5.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures.StepSecondMaxWork(0), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures.StepMaxWork(1), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures.StepSecondMaxWork(1), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures.StepMaxWork(2), 7.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures.StepSecondMaxWork(2), 6.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures.StepMaxWork(3), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.workDatastructures.StepSecondMaxWork(3), 8.0);

    BOOST_CHECK_EQUAL(klActiveSchedule.NumSteps(), 4);
    BOOST_CHECK_EQUAL(klActiveSchedule.IsFeasible(), true);

    // Check initial cost consistency
    double initialRecomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    double initialTracked = kl.GetCurrentCost();
    BOOST_CHECK_CLOSE(initialRecomputed, initialTracked, 0.00001);

    // Insert nodes into gain heap
    auto nodeSelection = kl.InsertGainHeapTestPenalty({2, 3});

    // Run first iteration and check cost consistency
    auto recomputeMaxGain = kl.RunInnerIterationTest();

    double iter1Recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    double iter1Tracked = kl.GetCurrentCost();
    BOOST_CHECK_CLOSE(iter1Recomputed, iter1Tracked, 0.00001);

    // Run second iteration
    auto &node3Affinity = kl.GetAffinityTable()[3];

    recomputeMaxGain = kl.RunInnerIterationTest();

    double iter2Recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    double iter2Tracked = kl.GetCurrentCost();

    BOOST_CHECK_CLOSE(iter2Recomputed, iter2Tracked, 0.00001);

    // Run third iteration
    recomputeMaxGain = kl.RunInnerIterationTest();

    double iter3Recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    double iter3Tracked = kl.GetCurrentCost();
    BOOST_CHECK_CLOSE(iter3Recomputed, iter3Tracked, 0.00001);

    // Run fourth iteration
    recomputeMaxGain = kl.RunInnerIterationTest();

    double iter4Recomputed = kl.GetCommCostF().ComputeScheduleCostTest();
    double iter4Tracked = kl.GetCurrentCost();
    BOOST_CHECK_CLOSE(iter4Recomputed, iter4Tracked, 0.00001);
}

// BOOST_AUTO_TEST_CASE(kl_lambda_total_comm_large_test_graphs) {
//     std::vector<std::string> filenames_graph = large_spaa_graphs();
//     using graph = computational_dag_edge_idx_vector_impl_def_int_t;
//     // Getting root git directory
//     std::filesystem::path cwd = std::filesystem::current_path();
//     std::cout << cwd << std::endl;
//     while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
//         cwd = cwd.parent_path();
//         std::cout << cwd << std::endl;
//     }

//     for (auto &filename_graph : filenames_graph) {
//         GreedyBspScheduler<computational_dag_edge_idx_vector_impl_def_int_t> test_scheduler;
//         BspInstance<graph> instance;
//         bool status_graph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filename_graph).string(),
//                                                                               instance.getComputationalDag());

//         instance.getArchitecture().setSynchronisationCosts(500);
//         instance.getArchitecture().setCommunicationCosts(5);
//         instance.getArchitecture().setNumberOfProcessors(4);

//         std::vector<std::vector<int>> send_cost = {{0, 1, 4, 4}, {1, 0, 4, 4}, {4, 4, 0, 1}, {4, 4, 1, 0}};

//         instance.getArchitecture().SetSendCosts(send_cost);

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         }

//         add_mem_weights(instance.getComputationalDag());

//         BspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.computeSchedule(schedule);

//         schedule.updateNumberOfSupersteps();

//         std::cout << "initial scedule with costs: " << schedule.computeCosts() << " and "
//                   << schedule.numberOfSupersteps() << " number of supersteps" << std::endl;

//         BspSchedule<graph> schedule_2(schedule);

//         BOOST_CHECK_EQUAL(RETURN_STATUS::OSP_SUCCESS, result);
//         BOOST_CHECK_EQUAL(&schedule.getInstance(), &instance);
//         BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

//         kl_total_lambda_comm_improver<graph, no_local_search_memory_constraint, 1> kl_total_lambda;
//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl_total_lambda.improveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "kl lambda new finished in " << duration << " seconds, costs: " << schedule.computeCosts()
//                   << " and lambda costs: " << schedule.computeTotalLambdaCosts() << " with "
//                   << schedule.numberOfSupersteps() << " number of supersteps" << std::endl;

//         BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);

//         kl_bsp_comm_improver_mt<graph, no_local_search_memory_constraint, 1> kl(42);
//         kl.setTimeQualityParameter(2.0);
//         start_time = std::chrono::high_resolution_clock::now();
//         status = kl.improveSchedule(schedule);
//         finish_time = std::chrono::high_resolution_clock::now();
//         duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "kl new finished in " << duration << " seconds, costs: " << schedule.computeCosts() << " with "
//                   << schedule.numberOfSupersteps() << " number of supersteps" << std::endl;

//         BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);

//         BspScheduleCS<graph> schedule_cs(schedule);

//         HillClimbingForCommSteps<graph> hc_comm_steps;
//         start_time = std::chrono::high_resolution_clock::now();
//         status = hc_comm_steps.improveSchedule(schedule_cs);
//         finish_time = std::chrono::high_resolution_clock::now();

//         duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "hc_comm_steps finished in " << duration << " seconds, costs: " << schedule_cs.computeCosts()
//                   << " with " << schedule_cs.numberOfSupersteps() << " number of supersteps" << std::endl;

//         BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule.satisfiesPrecedenceConstraints(), true);

//         kl_total_lambda.improveSchedule(schedule_2);

//         HillClimbingScheduler<graph> hc;

//         start_time = std::chrono::high_resolution_clock::now();
//         status = hc.improveSchedule(schedule_2);
//         finish_time = std::chrono::high_resolution_clock::now();

//         duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "hc finished in " << duration << " seconds, costs: " << schedule_2.computeCosts() << " with "
//                   << schedule_2.numberOfSupersteps() << " number of supersteps" << std::endl;

//         BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule_2.satisfiesPrecedenceConstraints(), true);

//         BspScheduleCS<graph> schedule_cs_2(schedule_2);

//         start_time = std::chrono::high_resolution_clock::now();
//         status = hc_comm_steps.improveSchedule(schedule_cs_2);
//         finish_time = std::chrono::high_resolution_clock::now();

//         duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "hc_comm_steps finished in " << duration << " seconds, costs: " << schedule_cs_2.computeCosts()
//                   << " with " << schedule_cs_2.numberOfSupersteps() << " number of supersteps" << std::endl;

//         BOOST_CHECK(status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule_cs_2.satisfiesPrecedenceConstraints(), true);
//     }
// }
