
#define BOOST_TEST_MODULE kl_bsp_affinity
#include <boost/test/unit_test.hpp>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_improver_test.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;
using graph = computational_dag_edge_idx_vector_impl_def_int_t;
using kl_active_schedule_t = kl_active_schedule<graph, double, no_local_search_memory_constraint>;

BOOST_AUTO_TEST_CASE(simple_parent_child_test) {
    using VertexType = graph::vertex_idx;

    graph dag;
    const VertexType v0 = dag.add_vertex(10, 5, 2);    // work=10, mem=5, comm=2
    const VertexType v1 = dag.add_vertex(8, 4, 1);     // work=8, mem=4, comm=1
    dag.add_edge(v0, v1, 3);                           // edge weight=3

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(2);

    BspInstance<graph> instance(dag, arch);
    instance.setCommunicationCosts(10);    // comm multiplier
    instance.setSynchronisationCosts(5);

    BspSchedule schedule(instance);
    schedule.setAssignedProcessors({0, 1});    // v0 on p0, v1 on p1
    schedule.setAssignedSupersteps({0, 1});    // v0 in step 0, v1 in step 1
    schedule.updateNumberOfSupersteps();

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    // Insert only v0 into gain heap to control which node moves
    auto node_selection = kl.insert_gain_heap_test({0});

    // Run one iteration - this will move v0 to its best position
    auto recompute_max_gain = kl.run_inner_iteration_test();

    // Compare costs after move
    double after_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_tracked = kl.get_current_cost();

    BOOST_CHECK_CLOSE(after_recomputed, after_tracked, 0.00001);
}

/**
 * Helper to validate comm datastructures by comparing with freshly computed ones
 */
template <typename Graph>
bool validate_comm_datastructures(const max_comm_datastructure<Graph, double, kl_active_schedule_t> &comm_ds_incremental,
                                  kl_active_schedule_t &active_sched,
                                  const BspInstance<Graph> &instance,
                                  const std::string &context) {
    // 1. Clone Schedule
    BspSchedule<Graph> current_schedule(instance);
    active_sched.write_schedule(current_schedule);

    // 2. Fresh Computation
    kl_active_schedule_t kl_sched_fresh;
    kl_sched_fresh.initialize(current_schedule);

    max_comm_datastructure<Graph, double, kl_active_schedule_t> comm_ds_fresh;
    comm_ds_fresh.initialize(kl_sched_fresh);

    // Compute for all steps
    unsigned max_step = current_schedule.numberOfSupersteps();
    comm_ds_fresh.compute_comm_datastructures(0, max_step > 0 ? max_step - 1 : 0);

    bool all_match = true;
    // std::cout << "\nValidating comm datastructures " << context << ":" << std::endl;

    // 3. Validate Comm Costs
    for (unsigned step = 0; step < max_step; ++step) {
        for (unsigned p = 0; p < instance.numberOfProcessors(); ++p) {
            auto send_inc = comm_ds_incremental.step_proc_send(step, p);
            auto send_fresh = comm_ds_fresh.step_proc_send(step, p);
            auto recv_inc = comm_ds_incremental.step_proc_receive(step, p);
            auto recv_fresh = comm_ds_fresh.step_proc_receive(step, p);

            if (std::abs(send_inc - send_fresh) > 1e-6 || std::abs(recv_inc - recv_fresh) > 1e-6) {
                all_match = false;
                std::cout << "  MISMATCH at step " << step << " proc " << p << ":" << std::endl;
                std::cout << "    Incremental: send=" << send_inc << ", recv=" << recv_inc << std::endl;
                std::cout << "    Fresh:       send=" << send_fresh << ", recv=" << recv_fresh << std::endl;
            }
        }
    }

    // 4. Validate Lambda Maps
    for (const auto v : instance.vertices()) {
        for (unsigned p = 0; p < instance.numberOfProcessors(); ++p) {
            unsigned count_inc = 0;
            if (comm_ds_incremental.node_lambda_map.has_proc_entry(v, p)) {
                count_inc = comm_ds_incremental.node_lambda_map.get_proc_entry(v, p);
            }

            unsigned count_fresh = 0;
            if (comm_ds_fresh.node_lambda_map.has_proc_entry(v, p)) {
                count_fresh = comm_ds_fresh.node_lambda_map.get_proc_entry(v, p);
            }

            if (count_inc != count_fresh) {
                all_match = false;
                std::cout << "  LAMBDA MISMATCH at node " << v << " proc " << p << ":" << std::endl;
                std::cout << "    Incremental: " << count_inc << std::endl;
                std::cout << "    Fresh:       " << count_fresh << std::endl;
            }
        }
    }

    return all_match;
}

/**
 * Helper to validate affinity tables by comparing with freshly computed ones
 */
template <typename Graph_t, typename comm_cost_function_t, typename MemoryConstraint_t, unsigned window_size, typename cost_t>
bool validate_affinity_tables(kl_improver_test<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t> &kl_incremental,
                              const BspInstance<Graph_t> &instance,
                              const std::string &context) {
    // 1. Get current schedule from incremental
    BspSchedule<Graph_t> current_schedule(instance);
    kl_incremental.get_active_schedule_test(current_schedule);

    // 2. Create fresh kl_improver and compute all affinities from scratch
    kl_improver_test<Graph_t, comm_cost_function_t, MemoryConstraint_t, window_size, cost_t> kl_fresh;
    kl_fresh.setup_schedule(current_schedule);

    // Get selected nodes from incremental
    std::vector<vertex_idx_t<Graph_t>> selected_nodes;

    const size_t active_count = kl_incremental.get_affinity_table().size();
    for (size_t i = 0; i < active_count; ++i) {
        selected_nodes.push_back(kl_incremental.get_affinity_table().get_selected_nodes()[i]);
    }

    std::cout << "\n  [" << context << "] Validating " << selected_nodes.size() << " selected nodes: { ";
    for (const auto n : selected_nodes) {
        std::cout << n << " ";
    }
    std::cout << "}" << std::endl;

    // Compute affinities for all selected nodes
    kl_fresh.insert_gain_heap_test(selected_nodes);

    bool all_match = true;
    const unsigned num_procs = instance.numberOfProcessors();
    const unsigned num_steps = kl_incremental.get_active_schedule().num_steps();

    // 3. Compare affinity tables for each selected node

    for (const auto &node : selected_nodes) {
        const auto &affinity_inc = kl_incremental.get_affinity_table().get_affinity_table(node);
        const auto &affinity_fresh = kl_fresh.get_affinity_table().get_affinity_table(node);

        unsigned node_step = kl_incremental.get_active_schedule().assigned_superstep(node);

        for (unsigned p = 0; p < num_procs; ++p) {
            if (p >= affinity_inc.size() || p >= affinity_fresh.size()) {
                continue;
            }

            for (unsigned idx = 0; idx < affinity_inc[p].size() && idx < affinity_fresh[p].size(); ++idx) {
                int step_offset = static_cast<int>(idx) - static_cast<int>(window_size);
                int target_step_signed = static_cast<int>(node_step) + step_offset;

                // Skip affinities for supersteps that don't exist
                if (target_step_signed < 0 || target_step_signed >= static_cast<int>(num_steps)) {
                    continue;
                }

                double val_inc = affinity_inc[p][idx];
                double val_fresh = affinity_fresh[p][idx];

                if (std::abs(val_inc - val_fresh) > 1e-4) {
                    all_match = false;

                    std::cout << "  AFFINITY MISMATCH [" << context << "]: node=" << node << " to P" << p << " S"
                              << target_step_signed << " (offset=" << step_offset << ")" << std::endl;
                    std::cout << "    Incremental: " << val_inc << std::endl;
                    std::cout << "    Fresh:       " << val_fresh << std::endl;
                    std::cout << "    Difference:  " << (val_inc - val_fresh) << std::endl;
                }
            }
        }
    }

    return all_match;
}

BOOST_AUTO_TEST_CASE(test_update_datastructure_after_move) {
    graph dag;

    // Create 6 vertices with specific comm weights
    dag.add_vertex(1, 10, 1);    // 0
    dag.add_vertex(1, 1, 1);     // 1
    dag.add_vertex(1, 5, 1);     // 2
    dag.add_vertex(1, 1, 1);     // 3
    dag.add_vertex(1, 2, 1);     // 4
    dag.add_vertex(1, 1, 1);     // 5

    // Add edges
    dag.add_edge(0, 1, 1);
    dag.add_edge(2, 3, 1);
    dag.add_edge(4, 5, 1);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(3);
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

    // Schedule:
    // Proc 0: Node 0, 4, 5
    // Proc 1: Node 1, 2
    // Proc 2: Node 3
    schedule.setAssignedProcessors({0, 1, 1, 2, 0, 0});
    // Steps: 0, 1, 0, 1, 0, 0
    schedule.setAssignedSupersteps({0, 1, 0, 1, 0, 0});
    schedule.updateNumberOfSupersteps();

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({0});
    kl.run_inner_iteration_test();

    double after_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_tracked = kl.get_current_cost();

    BOOST_CHECK(validate_comm_datastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_update_datastructure_after_move"));
    BOOST_CHECK_CLOSE(after_recomputed, after_tracked, 0.00001);
}

BOOST_AUTO_TEST_CASE(test_multiple_sequential_moves) {
    graph dag;

    // Create a linear chain: 0 -> 1 -> 2 -> 3
    dag.add_vertex(1, 10, 1);    // 0
    dag.add_vertex(1, 8, 1);     // 1
    dag.add_vertex(1, 6, 1);     // 2
    dag.add_vertex(1, 4, 1);     // 3

    dag.add_edge(0, 1, 1);
    dag.add_edge(1, 2, 1);
    dag.add_edge(2, 3, 1);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(4);
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

    schedule.setAssignedProcessors({0, 1, 2, 3});
    schedule.setAssignedSupersteps({0, 0, 0, 0});
    schedule.updateNumberOfSupersteps();

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({1});
    kl.run_inner_iteration_test();

    double after_move1_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move1_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_multiple_sequential_moves_1"));
    BOOST_CHECK_CLOSE(after_move1_recomputed, after_move1_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move2_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move2_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_multiple_sequential_moves_2"));
    BOOST_CHECK_CLOSE(after_move2_recomputed, after_move2_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move3_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move3_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_multiple_sequential_moves_3"));
    BOOST_CHECK_CLOSE(after_move3_recomputed, after_move3_tracked, 0.00001);

    // After: Node 0 has 3 local children
    // Send cost = 10 * 0 = 0 (all local)
    // Work cost 4
    BOOST_CHECK_CLOSE(after_move3_tracked, 4.0, 0.00001);
}

BOOST_AUTO_TEST_CASE(test_node_with_multiple_children) {
    graph dag;

    // Tree structure: Node 0 has three children (1, 2, 3)
    dag.add_vertex(1, 1, 1);    // 0
    dag.add_vertex(1, 1, 1);    // 1
    dag.add_vertex(1, 1, 1);    // 2
    dag.add_vertex(1, 1, 1);    // 3

    dag.add_edge(0, 1, 1);
    dag.add_edge(0, 2, 1);
    dag.add_edge(0, 3, 1);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(4);
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

    schedule.setAssignedProcessors({0, 1, 2, 3});
    schedule.setAssignedSupersteps({0, 0, 0, 0});
    schedule.updateNumberOfSupersteps();

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({1});
    kl.get_comm_cost_f().compute_schedule_cost();
    kl.run_inner_iteration_test();

    double after_move1_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move1_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_node_with_multiple_children"));
    BOOST_CHECK_CLOSE(after_move1_recomputed, after_move1_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move2_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move2_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_node_with_multiple_children_2"));
    BOOST_CHECK_CLOSE(after_move2_recomputed, after_move2_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move3_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move3_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_node_with_multiple_children_3"));
    BOOST_CHECK_CLOSE(after_move3_recomputed, after_move3_tracked, 0.00001);

    // After: Node 0 has 3 local children
    // Send cost = 10 * 0 = 0 (all local)
    // Work cost 4
    BOOST_CHECK_CLOSE(after_move3_tracked, 4.0, 0.00001);
}

BOOST_AUTO_TEST_CASE(test_cross_step_moves) {
    graph dag;

    // 0 -> 1 -> 2
    dag.add_vertex(1, 10, 1);    // 0
    dag.add_vertex(1, 8, 1);     // 1
    dag.add_vertex(1, 6, 1);     // 2

    dag.add_edge(0, 1, 1);
    dag.add_edge(1, 2, 1);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(2);
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

    schedule.setAssignedProcessors({0, 1, 0});
    schedule.setAssignedSupersteps({0, 1, 2});
    schedule.updateNumberOfSupersteps();

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({1});
    kl.run_inner_iteration_test();

    double after_move1_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move1_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(
        kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "test_cross_step_moves_1"));
    BOOST_CHECK_CLOSE(after_move1_recomputed, after_move1_tracked, 0.00001);
}

BOOST_AUTO_TEST_CASE(test_complex_scenario) {
    std::cout << "Test case complex scenario" << std::endl;
    graph dag;

    const auto v1 = dag.add_vertex(2, 9, 2);
    const auto v2 = dag.add_vertex(3, 8, 4);
    const auto v3 = dag.add_vertex(4, 7, 3);
    const auto v4 = dag.add_vertex(5, 6, 2);
    const auto v5 = dag.add_vertex(6, 5, 6);
    const auto v6 = dag.add_vertex(7, 4, 2);
    dag.add_vertex(8, 3, 4);                    // v7 (index 6)
    const auto v8 = dag.add_vertex(9, 2, 1);    // v8 (index 7)

    dag.add_edge(v1, v2, 2);
    dag.add_edge(v1, v3, 2);
    dag.add_edge(v1, v4, 2);
    dag.add_edge(v2, v5, 12);
    dag.add_edge(v3, v5, 6);
    dag.add_edge(v3, v6, 7);
    dag.add_edge(v5, v8, 9);
    dag.add_edge(v4, v8, 9);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(2);    // P0, P1
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.updateNumberOfSupersteps();

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({v3, v1});
    kl.run_inner_iteration_test();

    double after_move1_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move1_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move1"));
    BOOST_CHECK_CLOSE(after_move1_recomputed, after_move1_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move2_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move2_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move2"));
    BOOST_CHECK(validate_affinity_tables(kl, instance, "complex_move2"));
    BOOST_CHECK_CLOSE(after_move2_recomputed, after_move2_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move3_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move3_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move3"));
    BOOST_CHECK_CLOSE(after_move3_recomputed, after_move3_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move4_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move4_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move4"));
    BOOST_CHECK_CLOSE(after_move4_recomputed, after_move4_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move5_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move5_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move5"));
    BOOST_CHECK_CLOSE(after_move5_recomputed, after_move5_tracked, 0.00001);
}

BOOST_AUTO_TEST_CASE(test_complex_scenario_only_compute) {
    graph dag;

    const auto v1 = dag.add_vertex(2, 9, 2);
    const auto v2 = dag.add_vertex(3, 8, 4);
    const auto v3 = dag.add_vertex(4, 7, 3);
    const auto v4 = dag.add_vertex(5, 6, 2);
    const auto v5 = dag.add_vertex(6, 5, 6);
    const auto v6 = dag.add_vertex(7, 4, 2);
    const auto v7 = dag.add_vertex(8, 3, 4);    // v7 (index 6)
    const auto v8 = dag.add_vertex(9, 2, 1);    // v8 (index 7)

    dag.add_edge(v1, v2, 2);
    dag.add_edge(v1, v3, 2);
    dag.add_edge(v1, v4, 2);
    dag.add_edge(v2, v5, 12);
    dag.add_edge(v3, v5, 6);
    dag.add_edge(v3, v6, 7);
    dag.add_edge(v5, v8, 9);
    dag.add_edge(v4, v8, 9);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(2);    // P0, P1
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.updateNumberOfSupersteps();

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({v1});
    kl.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move1"));
    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    kl_improver_test kl2;
    kl2.setup_schedule(schedule);

    kl2.insert_gain_heap_test({v2});
    kl2.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl2.get_comm_cost_f().comm_ds, kl2.get_active_schedule(), instance, "complex_move2"));
    BOOST_CHECK_CLOSE(kl2.get_comm_cost_f().compute_schedule_cost_test(), kl2.get_current_cost(), 0.00001);

    kl_improver_test kl3;
    kl3.setup_schedule(schedule);

    kl3.insert_gain_heap_test({v3});
    kl3.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl3.get_comm_cost_f().comm_ds, kl3.get_active_schedule(), instance, "complex_move3"));
    BOOST_CHECK_CLOSE(kl3.get_comm_cost_f().compute_schedule_cost_test(), kl3.get_current_cost(), 0.00001);

    kl_improver_test kl4;
    kl4.setup_schedule(schedule);

    kl4.insert_gain_heap_test({v4});
    kl4.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl4.get_comm_cost_f().comm_ds, kl4.get_active_schedule(), instance, "complex_move4"));
    BOOST_CHECK_CLOSE(kl4.get_comm_cost_f().compute_schedule_cost_test(), kl4.get_current_cost(), 0.00001);

    kl_improver_test kl5;
    kl5.setup_schedule(schedule);

    kl5.insert_gain_heap_test({v5});
    kl5.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl5.get_comm_cost_f().comm_ds, kl5.get_active_schedule(), instance, "complex_move5"));
    BOOST_CHECK_CLOSE(kl5.get_comm_cost_f().compute_schedule_cost_test(), kl5.get_current_cost(), 0.00001);

    kl_improver_test kl6;
    kl6.setup_schedule(schedule);

    kl6.insert_gain_heap_test({v6});
    kl6.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl6.get_comm_cost_f().comm_ds, kl6.get_active_schedule(), instance, "complex_move6"));
    BOOST_CHECK_CLOSE(kl6.get_comm_cost_f().compute_schedule_cost_test(), kl6.get_current_cost(), 0.00001);

    kl_improver_test kl7;
    kl7.setup_schedule(schedule);

    kl7.insert_gain_heap_test({v7});
    kl7.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl7.get_comm_cost_f().comm_ds, kl7.get_active_schedule(), instance, "complex_move7"));
    BOOST_CHECK_CLOSE(kl7.get_comm_cost_f().compute_schedule_cost_test(), kl7.get_current_cost(), 0.00001);

    kl_improver_test kl8;
    kl8.setup_schedule(schedule);

    kl8.insert_gain_heap_test({v8});
    kl8.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl8.get_comm_cost_f().comm_ds, kl8.get_active_schedule(), instance, "complex_move8"));
    BOOST_CHECK_CLOSE(kl8.get_comm_cost_f().compute_schedule_cost_test(), kl8.get_current_cost(), 0.00001);
}

BOOST_AUTO_TEST_CASE(test_complex_scenario_only_compute_2) {
    graph dag;

    const auto v1 = dag.add_vertex(2, 9, 2);
    const auto v2 = dag.add_vertex(3, 8, 4);
    const auto v3 = dag.add_vertex(4, 7, 3);
    const auto v4 = dag.add_vertex(5, 6, 2);
    const auto v5 = dag.add_vertex(6, 5, 6);
    const auto v6 = dag.add_vertex(7, 4, 2);
    const auto v7 = dag.add_vertex(8, 3, 4);    // v7 (index 6)
    const auto v8 = dag.add_vertex(9, 2, 1);    // v8 (index 7)

    dag.add_edge(v1, v2, 2);
    dag.add_edge(v1, v5, 2);
    dag.add_edge(v1, v6, 2);
    dag.add_edge(v1, v3, 2);
    dag.add_edge(v1, v4, 2);
    dag.add_edge(v2, v5, 12);
    dag.add_edge(v2, v6, 2);
    dag.add_edge(v2, v7, 2);
    dag.add_edge(v2, v8, 2);
    dag.add_edge(v3, v5, 6);
    dag.add_edge(v3, v6, 7);
    dag.add_edge(v3, v7, 2);
    dag.add_edge(v3, v8, 2);
    dag.add_edge(v5, v8, 9);
    dag.add_edge(v4, v8, 9);
    dag.add_edge(v5, v7, 2);
    dag.add_edge(v6, v7, 2);
    dag.add_edge(v7, v8, 2);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(2);    // P0, P1
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.updateNumberOfSupersteps();

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({v1});
    kl.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "complex_move1"));
    BOOST_CHECK_CLOSE(kl.get_comm_cost_f().compute_schedule_cost_test(), kl.get_current_cost(), 0.00001);

    kl_improver_test kl2;
    kl2.setup_schedule(schedule);

    kl2.insert_gain_heap_test({v2});
    kl2.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl2.get_comm_cost_f().comm_ds, kl2.get_active_schedule(), instance, "complex_move2"));
    BOOST_CHECK_CLOSE(kl2.get_comm_cost_f().compute_schedule_cost_test(), kl2.get_current_cost(), 0.00001);

    kl_improver_test kl3;
    kl3.setup_schedule(schedule);

    kl3.insert_gain_heap_test({v3});
    kl3.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl3.get_comm_cost_f().comm_ds, kl3.get_active_schedule(), instance, "complex_move3"));
    BOOST_CHECK_CLOSE(kl3.get_comm_cost_f().compute_schedule_cost_test(), kl3.get_current_cost(), 0.00001);

    kl_improver_test kl4;
    kl4.setup_schedule(schedule);

    kl4.insert_gain_heap_test({v4});
    kl4.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl4.get_comm_cost_f().comm_ds, kl4.get_active_schedule(), instance, "complex_move4"));
    BOOST_CHECK_CLOSE(kl4.get_comm_cost_f().compute_schedule_cost_test(), kl4.get_current_cost(), 0.00001);

    kl_improver_test kl5;
    kl5.setup_schedule(schedule);

    kl5.insert_gain_heap_test({v5});
    kl5.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl5.get_comm_cost_f().comm_ds, kl5.get_active_schedule(), instance, "complex_move5"));
    BOOST_CHECK_CLOSE(kl5.get_comm_cost_f().compute_schedule_cost_test(), kl5.get_current_cost(), 0.00001);

    kl_improver_test kl6;
    kl6.setup_schedule(schedule);

    kl6.insert_gain_heap_test({v6});
    kl6.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl6.get_comm_cost_f().comm_ds, kl6.get_active_schedule(), instance, "complex_move6"));
    BOOST_CHECK_CLOSE(kl6.get_comm_cost_f().compute_schedule_cost_test(), kl6.get_current_cost(), 0.00001);

    kl_improver_test kl7;
    kl7.setup_schedule(schedule);

    kl7.insert_gain_heap_test({v7});
    kl7.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl7.get_comm_cost_f().comm_ds, kl7.get_active_schedule(), instance, "complex_move7"));
    BOOST_CHECK_CLOSE(kl7.get_comm_cost_f().compute_schedule_cost_test(), kl7.get_current_cost(), 0.00001);

    kl_improver_test kl8;
    kl8.setup_schedule(schedule);

    kl8.insert_gain_heap_test({v8});
    kl8.run_inner_iteration_test();

    BOOST_CHECK(validate_comm_datastructures(kl8.get_comm_cost_f().comm_ds, kl8.get_active_schedule(), instance, "complex_move8"));
    BOOST_CHECK_CLOSE(kl8.get_comm_cost_f().compute_schedule_cost_test(), kl8.get_current_cost(), 0.00001);
}

BOOST_AUTO_TEST_CASE(test_grid_graph_complex_moves) {
    // Construct 5x5 Grid Graph (25 nodes, indices 0-24)
    graph dag = osp::construct_grid_dag<graph>(5, 5);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(4);    // P0..P3
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

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

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({12, 8, 7});
    kl.run_inner_iteration_test();

    double after_move1_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move1_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "grid_move1"));
    BOOST_CHECK_CLOSE(after_move1_recomputed, after_move1_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move2_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move2_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "grid_move2"));
    BOOST_CHECK_CLOSE(after_move2_recomputed, after_move2_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move3_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move3_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "grid_move3"));
    BOOST_CHECK_CLOSE(after_move3_recomputed, after_move3_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move4_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move4_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "grid_move4"));
    BOOST_CHECK_CLOSE(after_move4_recomputed, after_move4_tracked, 0.00001);
}

BOOST_AUTO_TEST_CASE(test_butterfly_graph_moves) {
    // Stages=2 -> 3 levels of 4 nodes each = 12 nodes.
    // Level 0: 0-3. Level 1: 4-7. Level 2: 8-11.
    graph dag = osp::construct_butterfly_dag<graph>(2);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(2);
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

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

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({4, 6, 0});
    kl.run_inner_iteration_test();

    double after_move1_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move1_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "butterfly_move1"));
    BOOST_CHECK_CLOSE(after_move1_recomputed, after_move1_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move2_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move2_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "butterfly_move2"));
    BOOST_CHECK_CLOSE(after_move2_recomputed, after_move2_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move3_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move3_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "butterfly_move3"));
    BOOST_CHECK_CLOSE(after_move3_recomputed, after_move3_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move4_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move4_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "butterfly_move4"));
    BOOST_CHECK_CLOSE(after_move4_recomputed, after_move4_tracked, 0.00001);
}

BOOST_AUTO_TEST_CASE(test_ladder_graph_moves) {
    // Ladder with 5 rungs -> 6 pairs of nodes = 12 nodes.
    // Pairs: (0,1), (2,3), ... (10,11).
    graph dag = osp::construct_ladder_dag<graph>(5);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(2);
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

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

    using comm_cost_t = kl_bsp_comm_cost_function<graph, double, no_local_search_memory_constraint>;
    using kl_improver_test = kl_improver_test<graph, comm_cost_t>;

    kl_improver_test kl;
    kl.setup_schedule(schedule);

    kl.insert_gain_heap_test({1, 3, 0, 2});
    kl.run_inner_iteration_test();

    double after_move1_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move1_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "ladder_move1"));
    BOOST_CHECK_CLOSE(after_move1_recomputed, after_move1_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move2_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move2_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "ladder_move2"));
    BOOST_CHECK_CLOSE(after_move2_recomputed, after_move2_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move3_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move3_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "ladder_move3"));
    BOOST_CHECK_CLOSE(after_move3_recomputed, after_move3_tracked, 0.00001);

    kl.run_inner_iteration_test();

    double after_move4_recomputed = kl.get_comm_cost_f().compute_schedule_cost_test();
    double after_move4_tracked = kl.get_current_cost();
    BOOST_CHECK(validate_comm_datastructures(kl.get_comm_cost_f().comm_ds, kl.get_active_schedule(), instance, "ladder_move4"));
    BOOST_CHECK_CLOSE(after_move4_recomputed, after_move4_tracked, 0.00001);
}
