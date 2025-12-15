
#define BOOST_TEST_MODULE kl_bsp_improver
#include <boost/test/unit_test.hpp>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/scheduler/CoarsenRefineSchedulers/MultiLevelHillClimbing.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing_for_comm_schedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/comm_cost_modules/kl_bsp_comm_cost.hpp"
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

    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

    schedule.UpdateNumberOfSupersteps();

    using CommCostT = kl_bsp_comm_cost_function<Graph, double, no_local_search_memory_constraint>;
    using KlImproverTest = kl_improver_test<Graph, CommCostT>;

    KlImproverTest kl;

    kl.setup_schedule(schedule);

    auto &klActiveSchedule = kl.get_active_schedule();

    // Verify work datastructures are set up correctly
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_max_work(0), 5.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_second_max_work(0), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_max_work(1), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_second_max_work(1), 0.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_max_work(2), 7.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_second_max_work(2), 6.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_max_work(3), 9.0);
    BOOST_CHECK_EQUAL(klActiveSchedule.work_datastructures.step_second_max_work(3), 8.0);

    BOOST_CHECK_EQUAL(klActiveSchedule.num_steps(), 4);
    BOOST_CHECK_EQUAL(klActiveSchedule.is_feasible(), true);

    // Check initial cost consistency
    double initialRecomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double initialTracked = kl.get_current_cost();
    BOOST_CHECK_CLOSE(initialRecomputed, initialTracked, 0.00001);

    // Insert nodes into gain heap
    auto nodeSelection = kl.insert_gain_heap_test_penalty({2, 3});

    // Run first iteration and check cost consistency
    auto recomputeMaxGain = kl.run_inner_iteration_test();

    double iter1Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double iter1Tracked = kl.get_current_cost();
    BOOST_CHECK_CLOSE(iter1Recomputed, iter1Tracked, 0.00001);

    // Run second iteration
    auto &node3Affinity = kl.get_affinity_table()[3];

    recomputeMaxGain = kl.run_inner_iteration_test();

    double iter2Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double iter2Tracked = kl.get_current_cost();

    BOOST_CHECK_CLOSE(iter2Recomputed, iter2Tracked, 0.00001);

    // Run third iteration
    recomputeMaxGain = kl.run_inner_iteration_test();

    double iter3Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double iter3Tracked = kl.get_current_cost();
    BOOST_CHECK_CLOSE(iter3Recomputed, iter3Tracked, 0.00001);

    // Run fourth iteration
    recomputeMaxGain = kl.run_inner_iteration_test();

    double iter4Recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double iter4Tracked = kl.get_current_cost();
    BOOST_CHECK_CLOSE(iter4Recomputed, iter4Tracked, 0.00001);
}

// BOOST_AUTO_TEST_CASE(kl_lambda_total_comm_large_test_graphs) {
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
//         bool status_graph = file_reader::readComputationalDagHyperdagFormatDB((cwd / filename_graph).string(),
//                                                                               instance.GetComputationalDag());

//         instance.GetArchitecture().SetSynchronisationCosts(500);
//         instance.GetArchitecture().SetCommunicationCosts(5);
//         instance.GetArchitecture().SetNumberOfProcessors(4);

//         std::vector<std::vector<int>> send_cost = {{0, 1, 4, 4}, {1, 0, 4, 4}, {4, 4, 0, 1}, {4, 4, 1, 0}};

//         instance.GetArchitecture().SetSendCosts(send_cost);

//         if (!status_graph) {

//             std::cout << "Reading files failed." << std::endl;
//             BOOST_CHECK(false);
//         }

//         add_mem_weights(instance.GetComputationalDag());

//         BspSchedule<graph> schedule(instance);
//         const auto result = test_scheduler.ComputeSchedule(schedule);

//         schedule.UpdateNumberOfSupersteps();

//         std::cout << "initial scedule with costs: " << schedule.ComputeCosts() << " and "
//                   << schedule.NumberOfSupersteps() << " number of supersteps" << std::endl;

//         BspSchedule<graph> schedule_2(schedule);

//         BOOST_CHECK_EQUAL(ReturnStatus::OSP_SUCCESS, result);
//         BOOST_CHECK_EQUAL(&schedule.GetInstance(), &instance);
//         BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

//         kl_total_lambda_comm_improver<graph, no_local_search_memory_constraint, 1> kl_total_lambda;
//         auto start_time = std::chrono::high_resolution_clock::now();
//         auto status = kl_total_lambda.ImproveSchedule(schedule);
//         auto finish_time = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "kl lambda new finished in " << duration << " seconds, costs: " << schedule.ComputeCosts()
//                   << " and lambda costs: " << schedule.computeTotalLambdaCosts() << " with "
//                   << schedule.NumberOfSupersteps() << " number of supersteps" << std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule.SatisfiesPrecedenceConstraints(), true);

//         kl_bsp_comm_improver_mt<graph, no_local_search_memory_constraint, 1> kl(42);
//         kl.setTimeQualityParameter(2.0);
//         start_time = std::chrono::high_resolution_clock::now();
//         status = kl.ImproveSchedule(schedule);
//         finish_time = std::chrono::high_resolution_clock::now();
//         duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "kl new finished in " << duration << " seconds, costs: " << schedule.ComputeCosts() << " with "
//                   << schedule.NumberOfSupersteps() << " number of supersteps" << std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule.SatisfiesPrecedenceConstraints(), true);

//         BspScheduleCS<graph> schedule_cs(schedule);

//         HillClimbingForCommSteps<graph> hc_comm_steps;
//         start_time = std::chrono::high_resolution_clock::now();
//         status = hc_comm_steps.ImproveSchedule(schedule_cs);
//         finish_time = std::chrono::high_resolution_clock::now();

//         duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "hc_comm_steps finished in " << duration << " seconds, costs: " << schedule_cs.ComputeCosts()
//                   << " with " << schedule_cs.NumberOfSupersteps() << " number of supersteps" << std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule.SatisfiesPrecedenceConstraints(), true);

//         kl_total_lambda.ImproveSchedule(schedule_2);

//         HillClimbingScheduler<graph> hc;

//         start_time = std::chrono::high_resolution_clock::now();
//         status = hc.ImproveSchedule(schedule_2);
//         finish_time = std::chrono::high_resolution_clock::now();

//         duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "hc finished in " << duration << " seconds, costs: " << schedule_2.ComputeCosts() << " with "
//                   << schedule_2.NumberOfSupersteps() << " number of supersteps" << std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule_2.SatisfiesPrecedenceConstraints(), true);

//         BspScheduleCS<graph> schedule_cs_2(schedule_2);

//         start_time = std::chrono::high_resolution_clock::now();
//         status = hc_comm_steps.ImproveSchedule(schedule_cs_2);
//         finish_time = std::chrono::high_resolution_clock::now();

//         duration = std::chrono::duration_cast<std::chrono::seconds>(finish_time - start_time).count();

//         std::cout << "hc_comm_steps finished in " << duration << " seconds, costs: " << schedule_cs_2.ComputeCosts()
//                   << " with " << schedule_cs_2.NumberOfSupersteps() << " number of supersteps" << std::endl;

//         BOOST_CHECK(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND);
//         BOOST_CHECK_EQUAL(schedule_cs_2.SatisfiesPrecedenceConstraints(), true);
//     }
// }
