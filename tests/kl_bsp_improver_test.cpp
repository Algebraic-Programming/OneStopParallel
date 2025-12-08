
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

template<typename Graph_t>
void add_mem_weights(Graph_t &dag) {

    int mem_weight = 1;
    int comm_weight = 7;

    for (const auto &v : dag.vertices()) {

        dag.set_vertex_work_weight(v, static_cast<v_memw_t<Graph_t>>(mem_weight++ % 10 + 2));
        dag.set_vertex_mem_weight(v, static_cast<v_memw_t<Graph_t>>(mem_weight++ % 10 + 2));
        dag.set_vertex_comm_weight(v, static_cast<v_commw_t<Graph_t>>(comm_weight++ % 10 + 2));
    }
}

BOOST_AUTO_TEST_CASE(kl_improver_inner_loop_test) {

    using graph = computational_dag_edge_idx_vector_impl_def_int_t;
    using VertexType = graph::vertex_idx;

    graph dag;

    const VertexType v1 = dag.add_vertex(2, 9, 2);
    const VertexType v2 = dag.add_vertex(3, 8, 4);
    const VertexType v3 = dag.add_vertex(4, 7, 3);
    const VertexType v4 = dag.add_vertex(5, 6, 2);
    const VertexType v5 = dag.add_vertex(6, 5, 6);
    const VertexType v6 = dag.add_vertex(7, 4, 2);
    dag.add_vertex(8, 3, 4);
    const VertexType v8 = dag.add_vertex(9, 2, 1);

    dag.add_edge(v1, v2, 2);
    dag.add_edge(v1, v3, 2);
    dag.add_edge(v1, v4, 2);
    dag.add_edge(v2, v5, 12);
    dag.add_edge(v3, v5, 6);
    dag.add_edge(v3, v6, 7);
    dag.add_edge(v5, v8, 9);
    dag.add_edge(v4, v8, 9);

    BspArchitecture<graph> arch;

    BspInstance<graph> instance(dag, arch);

    BspSchedule schedule(instance);

    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});

    schedule.updateNumberOfSupersteps();

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;

    kl.setup_schedule(schedule);

    auto &kl_active_schedule = kl.get_active_schedule();

    // Verify work datastructures are set up correctly
    BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(0), 5.0);
    BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(0), 0.0);
    BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(1), 9.0);
    BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(1), 0.0);
    BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(2), 7.0);
    BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(2), 6.0);
    BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_max_work(3), 9.0);
    BOOST_CHECK_EQUAL(kl_active_schedule.work_datastructures.step_second_max_work(3), 8.0);

    BOOST_CHECK_EQUAL(kl_active_schedule.num_steps(), 4);
    BOOST_CHECK_EQUAL(kl_active_schedule.is_feasible(), true);

    // Check initial cost consistency
    double initial_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double initial_tracked = kl.get_current_cost();
    BOOST_CHECK_CLOSE(initial_recomputed, initial_tracked, 0.00001);

    // Insert nodes into gain heap
    auto node_selection = kl.insert_gain_heap_test_penalty({2, 3});

    // Run first iteration and check cost consistency
    auto recompute_max_gain = kl.run_inner_iteration_test();

    double iter1_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double iter1_tracked = kl.get_current_cost();
    BOOST_CHECK_CLOSE(iter1_recomputed, iter1_tracked, 0.00001);

    // Run second iteration
    auto &node3_affinity = kl.get_affinity_table()[3];

    recompute_max_gain = kl.run_inner_iteration_test();

    double iter2_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double iter2_tracked = kl.get_current_cost();

    BOOST_CHECK_CLOSE(iter2_recomputed, iter2_tracked, 0.00001);

    // Run third iteration
    recompute_max_gain = kl.run_inner_iteration_test();

    double iter3_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double iter3_tracked = kl.get_current_cost();
    BOOST_CHECK_CLOSE(iter3_recomputed, iter3_tracked, 0.00001);

    // Run fourth iteration
    recompute_max_gain = kl.run_inner_iteration_test();

    double iter4_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double iter4_tracked = kl.get_current_cost();
    BOOST_CHECK_CLOSE(iter4_recomputed, iter4_tracked, 0.00001);
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