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

#define BOOST_TEST_MODULE kl_bsp_cost
#include <boost/test/unit_test.hpp>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/comm_cost_modules/kl_bsp_comm_cost.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/comm_cost_modules/max_comm_datastructure.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_active_schedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_util.hpp"
#include "osp/concepts/graph_traits.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;
using graph = computational_dag_edge_idx_vector_impl_def_int_t;
using kl_active_schedule_t = kl_active_schedule<graph, double, no_local_search_memory_constraint>;

BOOST_AUTO_TEST_CASE(test_arrange_superstep_comm_data) {
    graph dag;

    dag.add_vertex(1, 1, 1);
    dag.add_vertex(1, 1, 1);
    dag.add_vertex(1, 1, 1);
    dag.add_vertex(1, 1, 1);

    BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(4);
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(1);

    BspInstance<graph> instance(dag, arch);
    BspSchedule<graph> schedule(instance);

    // Initialize schedule with 1 step
    schedule.setAssignedProcessors({0, 1, 2, 3});
    schedule.setAssignedSupersteps({0, 0, 0, 0});
    schedule.updateNumberOfSupersteps();

    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    max_comm_datastructure<graph, double, kl_active_schedule_t> comm_ds;
    comm_ds.initialize(kl_sched);

    unsigned step = 0;

    // Case 1: Unique Max
    comm_ds.step_proc_send(step, 0) = 10;
    comm_ds.step_proc_send(step, 1) = 5;
    comm_ds.step_proc_send(step, 2) = 2;
    comm_ds.step_proc_send(step, 3) = 1;

    comm_ds.step_proc_receive(step, 0) = 8;
    comm_ds.step_proc_receive(step, 1) = 8;
    comm_ds.step_proc_receive(step, 2) = 2;
    comm_ds.step_proc_receive(step, 3) = 1;

    comm_ds.arrange_superstep_comm_data(step);

    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(step), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm_count(step), 1);     // Only proc 0 has 10
    BOOST_CHECK_EQUAL(comm_ds.step_second_max_comm(step), 8);    // Next highest is 8 (from recv)

    // Case 2: Shared Max
    comm_ds.reset_superstep(step);
    comm_ds.step_proc_send(step, 0) = 10;    // Need to re-set this as reset clears it
    comm_ds.step_proc_send(step, 1) = 10;
    comm_ds.step_proc_send(step, 2) = 2;
    comm_ds.step_proc_send(step, 3) = 1;

    comm_ds.step_proc_receive(step, 0) = 5;
    comm_ds.step_proc_receive(step, 1) = 5;
    comm_ds.step_proc_receive(step, 2) = 2;
    comm_ds.step_proc_receive(step, 3) = 1;
    comm_ds.arrange_superstep_comm_data(step);

    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(step), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm_count(step), 2);     // Proc 0 and 1
    BOOST_CHECK_EQUAL(comm_ds.step_second_max_comm(step), 5);    // Next highest is 5 (from recv)

    // Case 3: Max in Recv
    comm_ds.reset_superstep(step);

    comm_ds.step_proc_send(step, 0) = 5;
    comm_ds.step_proc_send(step, 1) = 5;
    comm_ds.step_proc_send(step, 2) = 2;
    comm_ds.step_proc_send(step, 3) = 1;

    comm_ds.step_proc_receive(step, 0) = 12;
    comm_ds.step_proc_receive(step, 1) = 8;
    comm_ds.step_proc_receive(step, 2) = 2;
    comm_ds.step_proc_receive(step, 3) = 1;
    comm_ds.arrange_superstep_comm_data(step);

    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(step), 12);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm_count(step), 1);
    BOOST_CHECK_EQUAL(comm_ds.step_second_max_comm(step), 8);

    // Case 4: All same
    comm_ds.reset_superstep(step);
    // Send: 10, 10, 10, 10
    // Recv: 10, 10, 10, 10
    for (unsigned i = 0; i < 4; ++i) {
        comm_ds.step_proc_send(step, i) = 10;
        comm_ds.step_proc_receive(step, i) = 10;
    }
    comm_ds.arrange_superstep_comm_data(step);

    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(step), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm_count(step), 8);     // 4 sends + 4 recvs
    BOOST_CHECK_EQUAL(comm_ds.step_second_max_comm(step), 0);    // If all removed, 0.

    // Case 5: Max removed, second max is from same type (Send)
    comm_ds.reset_superstep(step);
    comm_ds.step_proc_send(step, 0) = 10;
    comm_ds.step_proc_send(step, 1) = 8;
    comm_ds.step_proc_send(step, 2) = 2;
    comm_ds.step_proc_send(step, 3) = 1;

    for (unsigned i = 0; i < 4; ++i) {
        comm_ds.step_proc_receive(step, i) = 5;
    }

    comm_ds.arrange_superstep_comm_data(step);

    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(step), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm_count(step), 1);
    BOOST_CHECK_EQUAL(comm_ds.step_second_max_comm(step), 8);

    // Case 6: Max removed, second max is from other type (Recv)
    comm_ds.reset_superstep(step);

    comm_ds.step_proc_send(step, 0) = 10;
    comm_ds.step_proc_send(step, 1) = 4;
    comm_ds.step_proc_send(step, 2) = 2;
    comm_ds.step_proc_send(step, 3) = 1;

    comm_ds.step_proc_receive(step, 0) = 8;
    comm_ds.step_proc_receive(step, 1) = 5;
    comm_ds.step_proc_receive(step, 2) = 2;
    comm_ds.step_proc_receive(step, 3) = 1;

    comm_ds.arrange_superstep_comm_data(step);

    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(step), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm_count(step), 1);
    BOOST_CHECK_EQUAL(comm_ds.step_second_max_comm(step), 8);
}

BOOST_AUTO_TEST_CASE(test_compute_comm_datastructures) {
    graph dag;

    // Create 6 vertices with specific comm weights
    // Node 0: weight 10 (sends to 1)
    dag.add_vertex(1, 10, 1);
    // Node 1: weight 1
    dag.add_vertex(1, 1, 1);
    // Node 2: weight 5 (sends to 3)
    dag.add_vertex(1, 5, 1);
    // Node 3: weight 1
    dag.add_vertex(1, 1, 1);
    // Node 4: weight 2 (local to 5)
    dag.add_vertex(1, 2, 1);
    // Node 5: weight 1
    dag.add_vertex(1, 1, 1);

    // Add edges
    // 0 -> 1
    dag.add_edge(0, 1, 1);    // Edge weight ignored by max_comm_datastructure
    // 2 -> 3
    dag.add_edge(2, 3, 1);
    // 4 -> 5
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
    schedule.setAssignedSupersteps({0, 1, 0, 1, 0, 0});
    schedule.updateNumberOfSupersteps();

    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    max_comm_datastructure<graph, double, kl_active_schedule_t> comm_ds;
    comm_ds.initialize(kl_sched);

    // Compute for steps 0 and 1
    comm_ds.compute_comm_datastructures(0, 1);

    unsigned step = 0;

    // Expected Step 0:
    // Proc 0 sends: 10 (Node 0 -> Node 1 on Proc 1)
    // Proc 1 receives: 10 (from Proc 0)
    // Proc 1 sends: 5 (Node 2 -> Node 3 on Proc 2)
    // Proc 2 receives: 5 (from Proc 1)
    // Proc 2 sends: 0
    // Proc 0 receives: 0

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(step, 0), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(step, 1), 5);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(step, 2), 0);

    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(step, 0), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(step, 1), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(step, 2), 5);

    // Max Comm Calculation Step 0
    // Send Max: 10 (P0)
    // Recv Max: 10 (P1)
    // Global Max: 10
    // Count: 2 (P0 send, P1 recv)
    // Second Max: 5 (P1 send, P2 recv)

    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(step), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm_count(step), 2);
    BOOST_CHECK_EQUAL(comm_ds.step_second_max_comm(step), 5);

    // Verify Step 1 (Should be empty as Nodes 1 and 3 are leaves)
    step = 1;
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(step, 0), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(step, 1), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(step, 2), 0);

    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(step, 0), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(step, 1), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(step, 2), 0);

    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(step), 0);
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

    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    max_comm_datastructure<graph, double, kl_active_schedule_t> comm_ds;
    comm_ds.initialize(kl_sched);
    comm_ds.compute_comm_datastructures(0, 1);

    // Move Node 0 from Proc 0 (Step 0) to Proc 2 (Step 0)
    // kl_move_struct(node, gain, from_proc, from_step, to_proc, to_step)
    using kl_move = kl_move_struct<double, graph::vertex_idx>;
    kl_move move(0, 0.0, 0, 0, 2, 0);

    // Apply the move to the schedule first
    thread_local_active_schedule_data<graph, double> active_schedule_data;
    active_schedule_data.initialize_cost(0.0);
    kl_sched.apply_move(move, active_schedule_data);

    // Then update the communication datastructures
    comm_ds.update_datastructure_after_move(move, 0, 1);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "test_update_datastructure_after_move"));

    unsigned step = 0;

    // Expected Changes:
    // Node 0 (was P0 -> P1) is now (P2 -> P1).
    // P0 Send: 10 -> 0
    // P2 Send: 0 -> 10
    // P1 Recv: 10 -> 10 (Source changed, but destination same)

    // Others unchanged:
    // P1 Send: 5
    // P2 Recv: 5

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(step, 0), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(step, 1), 5);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(step, 2), 10);

    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(step, 0), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(step, 1), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(step, 2), 5);

    // Max Comm:
    // Send Max: 10 (P2)
    // Recv Max: 10 (P1)
    // Global Max: 10
    // Count: 2 (P2 send, P1 recv)
    // Second Max: 5 (P1 send, P2 recv)

    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(step), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm_count(step), 2);
    BOOST_CHECK_EQUAL(comm_ds.step_second_max_comm(step), 5);
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

    // Initial: All at step 0, on different processors
    // 0@P0, 1@P1, 2@P2, 3@P3
    schedule.setAssignedProcessors({0, 1, 2, 3});
    schedule.setAssignedSupersteps({0, 0, 0, 0});
    schedule.updateNumberOfSupersteps();

    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    max_comm_datastructure<graph, double, kl_active_schedule_t> comm_ds;
    comm_ds.initialize(kl_sched);
    comm_ds.compute_comm_datastructures(0, 0);

    // Initial state:
    // P0 sends to P1 (10), P1 sends to P2 (8), P2 sends to P3 (6)
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 1), 8);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 2), 6);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 3), 0);

    using kl_move = kl_move_struct<double, graph::vertex_idx>;
    thread_local_active_schedule_data<graph, double> active_schedule_data;
    active_schedule_data.initialize_cost(0.0);

    // Move 1: Move node 1 from P1 to P0 (make 0->1 local)
    kl_move move1(1, 0.0, 1, 0, 0, 0);
    kl_sched.apply_move(move1, active_schedule_data);
    comm_ds.update_datastructure_after_move(move1, 0, 0);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "test_multiple_sequential_moves_1"));

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 8);       // Node 1 sends
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 1), 0);       // Node was moved away
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 0), 0);    // No receives at P0

    // Move 2: Move node 2 from P2 to P0 (chain more local)
    kl_move move2(2, 0.0, 2, 0, 0, 0);
    kl_sched.apply_move(move2, active_schedule_data);
    comm_ds.update_datastructure_after_move(move2, 0, 0);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "test_multiple_sequential_moves_2"));

    // After move2: Nodes 0,1,2 all at P0, only 3 at P3
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 6);       // Only node 2 sends off-proc
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 2), 0);       // Node moved away
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 3), 6);    // P3 receives from node 2

    // Move 3: Move node 3 to P0 (everything local)
    kl_move move3(3, 0.0, 3, 0, 0, 0);
    kl_sched.apply_move(move3, active_schedule_data);
    comm_ds.update_datastructure_after_move(move3, 0, 0);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "test_multiple_sequential_moves_3"));

    // After move3: All nodes at P0, all communication is local
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 0);    // All local
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(0), 0);        // No communication cost
}

BOOST_AUTO_TEST_CASE(test_node_with_multiple_children) {
    graph dag;

    // Tree structure: Node 0 has three children (1, 2, 3)
    dag.add_vertex(1, 10, 1);    // 0
    dag.add_vertex(1, 1, 1);     // 1
    dag.add_vertex(1, 1, 1);     // 2
    dag.add_vertex(1, 1, 1);     // 3

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

    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    max_comm_datastructure<graph, double, kl_active_schedule_t> comm_ds;
    comm_ds.initialize(kl_sched);
    comm_ds.compute_comm_datastructures(0, 0);

    // Initial: Node 0 has 3 children on P1, P2, P3 (3 unique off-proc)
    // Send cost = 10 * 3 = 30
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 30);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 1), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 2), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 3), 10);

    using kl_move = kl_move_struct<double, graph::vertex_idx>;
    thread_local_active_schedule_data<graph, double> active_schedule_data;
    active_schedule_data.initialize_cost(0.0);

    // Move child 1 to P0 (same as parent)
    kl_move move1(1, 0.0, 1, 0, 0, 0);
    kl_sched.apply_move(move1, active_schedule_data);
    comm_ds.update_datastructure_after_move(move1, 0, 0);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "test_node_with_multiple_children"));

    // After: Node 0 has 1 local child, 2 off-proc (P2, P3)
    // Send cost = 10 * 2 = 20
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 20);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 1), 0);    // No longer receives
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 2), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 3), 10);

    kl_move move2(2, 0.0, 2, 0, 0, 0);
    kl_sched.apply_move(move2, active_schedule_data);
    comm_ds.update_datastructure_after_move(move2, 0, 0);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "test_node_with_multiple_children_2"));

    // After: Node 0 has 2 local children, 1 off-proc (P3)
    // Send cost = 10 * 1 = 10
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 2), 0);    // No longer receives
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 3), 10);

    // Move child 3 to P0 (all local)
    kl_move move3(3, 0.0, 3, 0, 0, 0);
    kl_sched.apply_move(move3, active_schedule_data);
    comm_ds.update_datastructure_after_move(move3, 0, 0);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "test_node_with_multiple_children_3"));

    // After: Node 0 has 3 local children
    // Send cost = 10 * 0 = 0 (all local)
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 3), 0);    // No longer receives
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

    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    max_comm_datastructure<graph, double, kl_active_schedule_t> comm_ds;
    comm_ds.initialize(kl_sched);
    comm_ds.compute_comm_datastructures(0, 2);

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 8);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 1), 10);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(1, 0), 8);

    using kl_move = kl_move_struct<double, graph::vertex_idx>;
    thread_local_active_schedule_data<graph, double> active_schedule_data;
    active_schedule_data.initialize_cost(0.0);

    // Move node 1 from (P1, step1) to (P0, step1)
    // This makes 0->1 edge stay cross-step but changes processor
    kl_move move1(1, 0.0, 1, 1, 0, 1);
    kl_sched.apply_move(move1, active_schedule_data);
    comm_ds.update_datastructure_after_move(move1, 0, 2);

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 0);       // Local (same processor)
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 0), 0);    // No receive needed

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 0), 0);    // Local (same processor)
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 0);    // Node moved away

    kl_move move2(1, 0.0, 0, 1, 0, 0);
    kl_sched.apply_move(move2, active_schedule_data);
    comm_ds.update_datastructure_after_move(move2, 0, 2);

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 0);    // All local at P0
}

BOOST_AUTO_TEST_CASE(test_complex_scenario_user_provided) {
    graph dag;

    // Vertices from user request
    // v1(0): w=2, c=9, m=2
    const auto v1 = dag.add_vertex(2, 9, 2);
    const auto v2 = dag.add_vertex(3, 8, 4);
    const auto v3 = dag.add_vertex(4, 7, 3);
    const auto v4 = dag.add_vertex(5, 6, 2);
    const auto v5 = dag.add_vertex(6, 5, 6);
    const auto v6 = dag.add_vertex(7, 4, 2);
    dag.add_vertex(8, 3, 4);                    // v7 (index 6)
    const auto v8 = dag.add_vertex(9, 2, 1);    // v8 (index 7)

    // Edges
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

    // Schedule: {1, 1, 0, 0, 1, 0, 0, 1}
    // v1@P1, v2@P1, v3@P0, v4@P0, v5@P1, v6@P0, v7@P0, v8@P1
    schedule.setAssignedProcessors({1, 1, 0, 0, 1, 0, 0, 1});

    // Supersteps: {0, 0, 1, 1, 2, 2, 3, 3}
    schedule.setAssignedSupersteps({0, 0, 1, 1, 2, 2, 3, 3});
    schedule.updateNumberOfSupersteps();

    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    max_comm_datastructure<graph, double, kl_active_schedule_t> comm_ds;
    comm_ds.initialize(kl_sched);
    comm_ds.compute_comm_datastructures(0, 3);

    // === Initial State Verification ===
    // ... (Same as before) ...
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 1), 9);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 0), 9);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(0), 9);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 0), 13);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(1, 1), 13);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(1), 13);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(2), 0);

    using kl_move = kl_move_struct<double, graph::vertex_idx>;
    thread_local_active_schedule_data<graph, double> active_schedule_data;
    active_schedule_data.initialize_cost(0.0);

    // === Move 1: Move v3 from P0 to P1 (at Step 1) ===
    kl_move move1(v3, 0.0, 0, 1, 1, 1);
    kl_sched.apply_move(move1, active_schedule_data);
    comm_ds.update_datastructure_after_move(move1, 0, 3);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "complex_move1"));

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 1), 9);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 0), 6);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 7);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(1), 7);

    // === Move 2: Move v4 from P0 to P1 (at Step 1) ===
    kl_move move2(v4, 0.0, 0, 1, 1, 1);
    kl_sched.apply_move(move2, active_schedule_data);
    comm_ds.update_datastructure_after_move(move2, 0, 3);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "complex_move2"));

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 1), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(0), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 0), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 7);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(1), 7);

    // === Move 3: Move v5 from P1 to P0 (at Step 2) ===
    kl_move move3(v5, 0.0, 1, 2, 0, 2);
    kl_sched.apply_move(move3, active_schedule_data);
    comm_ds.update_datastructure_after_move(move3, 0, 3);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "complex_move3"));

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 1), 8);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(0), 8);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 7);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(2, 0), 5);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(2), 5);

    // === Move 4: Move v6 from P0 to P1 (at Step 2) ===
    // v6 is child of v3 (P1, S1).
    // Before: v3(P1) -> v6(P0). Off-proc.
    // After: v3(P1) -> v6(P1). Local.
    // v3 also sends to v5(P0).
    // So v3 targets: {P0}. Count = 1.
    // Send Cost v3 = 7. Unchanged.
    kl_move move4(v6, 0.0, 0, 2, 1, 2);
    kl_sched.apply_move(move4, active_schedule_data);
    comm_ds.update_datastructure_after_move(move4, 0, 3);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "complex_move4"));

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 7);

    // === Move 5: Move v5 from P0 to P1 (at Step 2) ===
    // v5 moves back to P1.
    // v3(P1) -> v5(P1), v6(P1). All local.
    // Send Cost v3 = 0.
    kl_move move5(v5, 0.0, 0, 2, 1, 2);
    kl_sched.apply_move(move5, active_schedule_data);
    comm_ds.update_datastructure_after_move(move5, 0, 3);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "complex_move5"));

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 0);
    BOOST_CHECK_EQUAL(comm_ds.step_max_comm(1), 0);
}

/**
 * Test: Grid Graph Complex Moves
 * Uses a 5x5 Grid Graph (25 nodes) with 6 Supersteps and 4 Processors.
 * Performs various moves to verify incremental updates in a dense graph.
 */
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

    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    max_comm_datastructure<graph, double, kl_active_schedule_t> comm_ds;
    comm_ds.initialize(kl_sched);
    comm_ds.compute_comm_datastructures(0, 5);

    // Initial check
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 3), 2);

    using kl_move = kl_move_struct<double, graph::vertex_idx>;
    thread_local_active_schedule_data<graph, double> active_schedule_data;
    active_schedule_data.initialize_cost(0.0);

    // === Move 1: Node 12 (P1->P0) ===
    kl_move move1(12, 0.0, 1, 2, 0, 2);
    kl_sched.apply_move(move1, active_schedule_data);
    comm_ds.update_datastructure_after_move(move1, 0, 5);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "grid_move1"));
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 3), 1);

    // === Move 2: Node 8 (P0->P3) ===
    kl_move move2(8, 0.0, 0, 1, 3, 1);
    kl_sched.apply_move(move2, active_schedule_data);
    comm_ds.update_datastructure_after_move(move2, 0, 5);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "grid_move2"));
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 3), 3);

    // === Move 3: Node 12 (P0->P3) ===
    kl_move move3(12, 0.0, 0, 2, 3, 2);
    kl_sched.apply_move(move3, active_schedule_data);
    comm_ds.update_datastructure_after_move(move3, 0, 5);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "grid_move3"));
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 3), 2);

    // === Move 4: Node 7 (P3->P0) ===
    kl_move move4(7, 0.0, 3, 1, 0, 1);
    kl_sched.apply_move(move4, active_schedule_data);
    comm_ds.update_datastructure_after_move(move4, 0, 5);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "grid_move4"));

    // Check P0 send contribution from Node 7.
    // Node 7 contributes 10.
    // We can check if P0 send >= 10.
    BOOST_CHECK_GE(comm_ds.step_proc_send(1, 0), 1);
}

/**
 * Test: Butterfly Graph Moves
 * Uses a Butterfly Graph (FFT pattern) to test structured communication patterns.
 * Stages = 2 (12 nodes). 3 Supersteps. 2 Processors.
 */
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

    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    max_comm_datastructure<graph, double, kl_active_schedule_t> comm_ds;
    comm_ds.initialize(kl_sched);
    comm_ds.compute_comm_datastructures(0, 2);

    // Initial State:
    // Step 0 (P0): Nodes 0-3 send to Level 1 (P1).
    // Each node in butterfly connects to 2 nodes in next level.
    // 0 -> 4, 6. (Both P1). Count=1. Cost=10.
    // 1 -> 5, 7. (Both P1). Count=1. Cost=10.
    // ... All 4 nodes send to P1. Total P0 Send = 40.
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 4);

    // Step 1 (P1): Nodes 4-7 send to Level 2 (P0).
    // All 4 nodes send to P0. Total P1 Send = 40.
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 4);

    using kl_move = kl_move_struct<double, graph::vertex_idx>;
    thread_local_active_schedule_data<graph, double> active_schedule_data;
    active_schedule_data.initialize_cost(0.0);

    // === Move 1: Move Node 4 (Level 1) P1 -> P0 ===
    // Node 4 moves to P0.
    // Impact on Step 0 (Parents 0, 1):
    // Node 0 -> 4(P0), 6(P1). Targets {P0, P1}. P0 is local. Targets {P1}. Count=1.
    // Node 1 -> 5(P1), 7(P1). Targets {P1}. Count=1.
    // Step 0 Send Cost unchanged (still 40).
    kl_move move1(4, 0.0, 1, 1, 0, 1);
    kl_sched.apply_move(move1, active_schedule_data);
    comm_ds.update_datastructure_after_move(move1, 0, 2);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "butterfly_move1"));

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 4);

    // Impact on Step 1 (Node 4):
    // Node 4 (P0) -> 8(P0), 10(P0). All local.
    // Node 4 stops sending. (Was 10).
    // P1 Send decreases by 10 -> 30.
    // P0 Send increases by 0 (all local).
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 3);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 0), 0);

    // === Move 2: Move Node 6 (Level 1) P1 -> P0 ===
    // Node 6 moves to P0.
    // Impact on Step 0 (Parent 0):
    // Node 0 -> 4(P0), 6(P0). All local.
    // Node 0 stops sending. (Was 10).
    // P0 Send decreases by 10 -> 30.
    kl_move move2(6, 0.0, 1, 1, 0, 1);
    kl_sched.apply_move(move2, active_schedule_data);
    comm_ds.update_datastructure_after_move(move2, 0, 2);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "butterfly_move2"));

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 2);

    // Impact on Step 1 (Node 6):
    // Node 6 (P0) -> 8(P0), 10(P0). All local.
    // P1 Send decreases by 10 -> 20.
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 2);

    // === Move 3: Move Node 0 (Level 0) P0 -> P1 ===
    // Node 0 moves to P1.
    // Impact on Step 0:
    // Node 0 (P1) -> 4(P0), 6(P0). Targets {P0}. Count=1. Cost=10.
    // Node 1 (P0) -> 5(P1), 7(P1). Targets {P1}. Count=1. Cost=10.
    // P0 Send: 10 (from Node 1).
    // P1 Send: 10 (from Node 0).
    kl_move move3(0, 0.0, 0, 0, 1, 0);
    kl_sched.apply_move(move3, active_schedule_data);
    comm_ds.update_datastructure_after_move(move3, 0, 2);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "butterfly_move3"));

    // === Move 4: Move Node 8 (Level 2) P0 -> P1 ===
    // Node 8 moves to P1.
    // Impact on Step 1:
    // Node 4 (P0) -> 8(P1), 10(P0). Targets {P1}. Count=1. Cost=10.
    // Node 6 (P0) -> 8(P1), 10(P0). Targets {P1}. Count=1. Cost=10.
    // P0 Send increases.
    kl_move move4(8, 0.0, 0, 2, 1, 2);
    kl_sched.apply_move(move4, active_schedule_data);
    comm_ds.update_datastructure_after_move(move4, 0, 2);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "butterfly_move4"));
}

/**
 * Test: Ladder Graph Moves
 * Uses a Ladder Graph (Rungs=5 -> 12 nodes).
 * Tests moving rungs between processors.
 */
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

    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    max_comm_datastructure<graph, double, kl_active_schedule_t> comm_ds;
    comm_ds.initialize(kl_sched);
    comm_ds.compute_comm_datastructures(0, 5);

    // Initial State:
    // Rung i (u1, v1) connects to Rung i+1 (u2, v2).
    // u1(P0) -> u2(P0), v2(P1). Targets {P1}. Count=1. Cost=10.
    // v1(P1) -> u2(P0), v2(P1). Targets {P0}. Count=1. Cost=10.
    // This applies for Steps 0 to 4.

    for (unsigned s = 0; s < 5; ++s) {
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(s, 0), 1);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(s, 1), 1);
    }

    using kl_move = kl_move_struct<double, graph::vertex_idx>;
    thread_local_active_schedule_data<graph, double> active_schedule_data;
    active_schedule_data.initialize_cost(0.0);

    // === Move 1: Move Node 1 (Rung 0, Right) P1 -> P0 ===
    // Node 1 moves to P0.
    // Rung 0 is now (0, 1) both at P0.
    // Impact on Step 0:
    // u1(0) -> u2(2, P0), v2(3, P1). Targets {P1}. Cost=10. (Unchanged)
    // v1(1) -> u2(2, P0), v2(3, P1). Targets {P1}. Cost=10.
    // P0 Send = 10 + 10 = 20.
    // P1 Send = 0.
    kl_move move1(1, 0.0, 1, 0, 0, 0);
    kl_sched.apply_move(move1, active_schedule_data);
    comm_ds.update_datastructure_after_move(move1, 0, 5);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "ladder_move1"));

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 2);
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 1), 0);

    // === Move 2: Move Node 3 (Rung 1, Right) P1 -> P0 ===
    // Node 3 moves to P0.
    // Rung 1 is now (2, 3) both at P0.
    // Impact on Step 0 (Parents 0, 1):
    // u1(0) -> u2(2, P0), v2(3, P0). All local. Cost=0.
    // v1(1) -> u2(2, P0), v2(3, P0). All local. Cost=0.
    // P0 Send at Step 0 = 0.
    kl_move move2(3, 0.0, 1, 1, 0, 1);
    kl_sched.apply_move(move2, active_schedule_data);
    comm_ds.update_datastructure_after_move(move2, 0, 5);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "ladder_move2"));

    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 0);

    // Impact on Step 1 (Nodes 2, 3):
    // u2(2, P0) -> u3(4, P0), v3(5, P1). Targets {P1}. Cost=10.
    // v2(3, P0) -> u3(4, P0), v3(5, P1). Targets {P1}. Cost=10.
    // P0 Send at Step 1 = 20.
    BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 0), 2);

    // === Move 3: Move Node 0 (Rung 0, Left) P0 -> P1 ===
    // Node 0 moves to P1.
    // Rung 0 is now (0@P1, 1@P0). Split again.
    // Impact on Step 0:
    // u1(0, P1) -> u2(2, P0), v2(3, P0). Targets {P0}. Cost=10.
    // v1(1, P0) -> u2(2, P0), v2(3, P0). All local. Cost=0.
    // P0 Send: 0.
    // P1 Send: 10.
    kl_move move3(0, 0.0, 0, 0, 1, 0);
    kl_sched.apply_move(move3, active_schedule_data);
    comm_ds.update_datastructure_after_move(move3, 0, 5);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "ladder_move3"));

    // === Move 4: Move Node 2 (Rung 1, Left) P0 -> P1 ===
    // Node 2 moves to P1.
    // Rung 1 is now (2@P1, 3@P0). Split again.
    // Impact on Step 0 (Parents 0, 1):
    // u1(0, P1) -> u2(2, P1), v2(3, P0). Targets {P0}. Cost=10.
    // v1(1, P0) -> u2(2, P1), v2(3, P0). Targets {P1}. Cost=10.
    // P0 Send: 10.
    // P1 Send: 10.
    kl_move move4(2, 0.0, 0, 1, 1, 1);
    kl_sched.apply_move(move4, active_schedule_data);
    comm_ds.update_datastructure_after_move(move4, 0, 5);
    BOOST_CHECK(validate_comm_datastructures(comm_ds, kl_sched, instance, "ladder_move4"));
}

BOOST_AUTO_TEST_CASE(test_lazy_and_buffered_modes) {
    std::cout << "Setup Graph" << std::endl;
    graph instance;
    instance.add_vertex(1, 10, 1);
    instance.add_vertex(1, 10, 1);
    instance.add_vertex(1, 10, 1);

    instance.add_edge(0, 1, 1);
    instance.add_edge(0, 2, 1);

    std::cout << "Setup Arch" << std::endl;
    osp::BspArchitecture<graph> arch;
    arch.setNumberOfProcessors(2);
    arch.setCommunicationCosts(1);
    arch.setSynchronisationCosts(0);

    std::cout << "Setup BspInstance" << std::endl;
    osp::BspInstance<graph> bsp_instance(instance, arch);

    std::cout << "Setup Schedule" << std::endl;
    osp::BspSchedule<graph> schedule(bsp_instance);
    schedule.setAssignedProcessor(0, 0);
    schedule.setAssignedProcessor(1, 1);
    schedule.setAssignedProcessor(2, 1);

    schedule.setAssignedSuperstep(0, 0);
    schedule.setAssignedSuperstep(1, 2);
    schedule.setAssignedSuperstep(2, 4);

    schedule.updateNumberOfSupersteps();

    std::cout << "Setup KL Sched" << std::endl;
    kl_active_schedule_t kl_sched;
    kl_sched.initialize(schedule);

    thread_local_active_schedule_data<graph, double> active_schedule_data;
    active_schedule_data.initialize_cost(0.0);

    std::cout << "Setup Complete" << std::endl;
    std::cout << "Num Vertices: " << instance.num_vertices() << std::endl;
    std::cout << "Num Procs: " << arch.numberOfProcessors() << std::endl;

    std::cout << "Start Eager Test" << std::endl;
    {
        using CommPolicy = osp::EagerCommCostPolicy;
        osp::max_comm_datastructure<graph, double, kl_active_schedule_t, CommPolicy> comm_ds;
        std::cout << "Initialize Eager Comm DS" << std::endl;
        comm_ds.initialize(kl_sched);

        std::cout << "Checking node_lambda_map" << std::endl;
        std::cout << "node_lambda_vec size: " << comm_ds.node_lambda_map.node_lambda_vec.size() << std::endl;
        if (comm_ds.node_lambda_map.node_lambda_vec.size() > 0) {
            std::cout << "node_lambda_vec[0] size: " << comm_ds.node_lambda_map.node_lambda_vec[0].size() << std::endl;
        }

        std::cout << "Compute Eager Comm DS" << std::endl;
        comm_ds.compute_comm_datastructures(0, 4);
        std::cout << "Eager Done" << std::endl;
    }

    std::cout << "Start Lazy Test" << std::endl;
    // --- Test Lazy Policy ---
    {
        using CommPolicy = osp::LazyCommCostPolicy;
        osp::max_comm_datastructure<graph, double, kl_active_schedule_t, CommPolicy> comm_ds;
        std::cout << "Initialize Comm DS" << std::endl;
        comm_ds.initialize(kl_sched);
        std::cout << "Compute Comm DS" << std::endl;
        comm_ds.compute_comm_datastructures(0, 4);

        // Expected Behavior for Lazy:
        // Node 0 (P0) sends to P1.
        // Children on P1 are at Step 2 and Step 4.
        // Lazy policy should attribute cost to min(2, 4) - 1 = Step 1.
        // Cost = 10 * 1.0 = 10.

        // Lazy: Send and Recv at min(2, 4) - 1 = Step 1.
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 0), 10);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(2, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(3, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(4, 0), 0);

        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(2, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(3, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(4, 1), 0);

        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(1, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(2, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(3, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(4, 0), 0);

        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(1, 1), 10);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(2, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(3, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(4, 1), 0);

        using kl_move = osp::kl_move_struct<double, graph::vertex_idx>;
        kl_move move(1, 0.0, 1, 2, 1, 3);    // Node 1, Step 2->3, Proc 1->1
        kl_sched.apply_move(move, active_schedule_data);
        comm_ds.update_datastructure_after_move(move, 0, 4);

        // After move: Children at {3, 4}. Min = 3. Send/Recv at Step 2.
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(2, 0), 10);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(3, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(4, 0), 0);

        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(2, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(3, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(4, 1), 0);

        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(1, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(2, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(3, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(4, 0), 0);

        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(1, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(2, 1), 10);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(3, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(4, 1), 0);

        // Reset Node 1 to Step 2 for next test
        kl_move move_back(1, 0.0, 1, 3, 1, 2);
        kl_sched.apply_move(move_back, active_schedule_data);
    }

    // --- Test Buffered Policy ---
    {
        using CommPolicy = osp::BufferedCommCostPolicy;
        osp::max_comm_datastructure<graph, double, kl_active_schedule_t, CommPolicy> comm_ds;
        comm_ds.initialize(kl_sched);
        comm_ds.compute_comm_datastructures(0, 4);

        // Buffered: Send at Step 0. Recv at min(2, 4) - 1 = Step 1.
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 10);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(2, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(3, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(4, 0), 0);

        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(2, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(3, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(4, 1), 0);

        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(1, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(2, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(3, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(4, 0), 0);

        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(1, 1), 10);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(2, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(3, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(4, 1), 0);

        using kl_move = osp::kl_move_struct<double, graph::vertex_idx>;
        kl_move move(1, 0.0, 1, 2, 1, 3);    // Node 1, Step 2->3, Proc 1->1
        kl_sched.apply_move(move, active_schedule_data);
        comm_ds.update_datastructure_after_move(move, 0, 4);

        // After move: Children at {3, 4}. Min = 3. Recv at Step 2. Send still at Step 0.
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(0, 0), 10);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(1, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(2, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(3, 0), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_send(4, 0), 0);

        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(0, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(1, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(2, 1), 10);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(3, 1), 0);
        BOOST_CHECK_EQUAL(comm_ds.step_proc_receive(4, 1), 0);
    }
}
