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

#define BOOST_TEST_MODULE kl_util
#include <boost/test/unit_test.hpp>
#include <numeric>
#include <set>


#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_util.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_active_schedule.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/bsp/model/BspSchedule.hpp"

using namespace osp;
using graph = computational_dag_edge_idx_vector_impl_def_int_t;
using kl_active_schedule_t = kl_active_schedule<graph, double, no_local_search_memory_constraint>;

// Test fixture for setting up a schedule
struct ScheduleFixture {
    BspInstance<graph> instance;
    BspSchedule<graph> schedule;
    kl_active_schedule_t active_schedule;

    ScheduleFixture() : schedule(instance) {
        // Setup a simple graph and schedule
        auto& dag = instance.getComputationalDag();
        for (int i = 0; i < 20; ++i) {
            dag.add_vertex(i + 1, i + 1, i + 1);
        }
        for (unsigned i = 0; i < 19; ++i) {
            dag.add_edge(i, i + 1, 1);
        }

        instance.getArchitecture().setNumberOfProcessors(4);
        instance.getArchitecture().setCommunicationCosts(1);
        instance.getArchitecture().setSynchronisationCosts(10);

        std::vector<unsigned> procs(20);
        std::vector<unsigned> steps(20);
        for (unsigned i = 0; i < 20; ++i) {
            procs[i] = i % 4;
            // Each node in its own superstep creates a valid initial schedule.
            steps[i] = i;
        }

        schedule.setAssignedProcessors(std::move(procs));
        schedule.setAssignedSupersteps(std::move(steps));
        schedule.updateNumberOfSupersteps();

        active_schedule.initialize(schedule);
    }
};

BOOST_FIXTURE_TEST_SUITE(kl_util_tests, ScheduleFixture)

// Tests for reward_penalty_strategy
BOOST_AUTO_TEST_CASE(reward_penalty_strategy_test) {
    reward_penalty_strategy<double, int, kl_active_schedule_t> rps;
    rps.initialize(active_schedule, 10.0, 20.0);

    BOOST_CHECK_EQUAL(rps.max_weight, 20.0);
    BOOST_CHECK_CLOSE(rps.initial_penalty, std::sqrt(20.0), 1e-9);

    rps.init_reward_penalty(2.0);
    BOOST_CHECK_CLOSE(rps.penalty, std::sqrt(20.0) * 2.0, 1e-9);
    BOOST_CHECK_CLOSE(rps.reward, 20.0 * 2.0, 1e-9);
}

// Tests for lock managers
template<typename LockManager>
void test_lock_manager() {
    LockManager lm;
    lm.initialize(10);

    BOOST_CHECK(!lm.is_locked(5));
    lm.lock(5);
    BOOST_CHECK(lm.is_locked(5));
    BOOST_CHECK(!lm.is_locked(6));
    lm.unlock(5);
    BOOST_CHECK(!lm.is_locked(5));

    lm.lock(1);
    lm.lock(3);
    lm.lock(5);
    BOOST_CHECK(lm.is_locked(3));
    lm.clear();
    BOOST_CHECK(!lm.is_locked(1));
    BOOST_CHECK(!lm.is_locked(3));
    BOOST_CHECK(!lm.is_locked(5));
}

BOOST_AUTO_TEST_CASE(lock_managers_test) {
    test_lock_manager<set_vertex_lock_manger<unsigned>>();
    test_lock_manager<vector_vertex_lock_manger<unsigned>>();
}

// Tests for adaptive_affinity_table
BOOST_AUTO_TEST_CASE(adaptive_affinity_table_test) {
    using affinity_table_t = adaptive_affinity_table<graph, double, int, kl_active_schedule_t, 1>;
    affinity_table_t table;
    table.initialize(active_schedule, 5);

    BOOST_CHECK_EQUAL(table.size(), 0);

    // Insert
    BOOST_CHECK(table.insert(0));
    BOOST_CHECK_EQUAL(table.size(), 1);
    BOOST_CHECK(table.is_selected(0));
    BOOST_CHECK(!table.is_selected(1));
    BOOST_CHECK(!table.insert(0)); // already present

    // Remove
    table.remove(0);
    BOOST_CHECK_EQUAL(table.size(), 0);
    BOOST_CHECK(!table.is_selected(0));

    // Insert more to test resizing
    for (unsigned i = 0; i < 10; ++i) {
        BOOST_CHECK(table.insert(i));
    }
    BOOST_CHECK_EQUAL(table.size(), 10);
    for (unsigned i = 0; i < 10; ++i) {
        BOOST_CHECK(table.is_selected(i));
    }

    // Test trim
    table.remove(3);
    table.remove(5);
    table.remove(7);
    BOOST_CHECK_EQUAL(table.size(), 7);

    table.trim();
    BOOST_CHECK_EQUAL(table.size(), 7);

    // After trim, the gaps should be filled.
    std::set<unsigned> expected_selected = {0, 1, 2, 4, 6, 8, 9};
    std::set<unsigned> actual_selected;
    const auto& selected_nodes_vec = table.get_selected_nodes();
    for(size_t i = 0; i < table.size(); ++i) {
        actual_selected.insert(static_cast<unsigned>(selected_nodes_vec[i]));
    }
    BOOST_CHECK(expected_selected == actual_selected);

    for(unsigned i = 0; i < 20; ++i) {
        if (expected_selected.count(i)) {
            BOOST_CHECK(table.is_selected(i));
        } else {
            BOOST_CHECK(!table.is_selected(i));
        }
    }

    // Check that indices are correct
    for(size_t i = 0; i < table.size(); ++i) {
        BOOST_CHECK_EQUAL(table.get_selected_nodes_idx(selected_nodes_vec[i]), i);
    }

    // Test reset
    table.reset_node_selection();
    BOOST_CHECK_EQUAL(table.size(), 0);
    BOOST_CHECK(!table.is_selected(0));
    BOOST_CHECK(!table.is_selected(1));
}

// Tests for static_affinity_table
BOOST_AUTO_TEST_CASE(static_affinity_table_test) {
    using affinity_table_t = static_affinity_table<graph, double, int, kl_active_schedule_t, 1>;
    affinity_table_t table;
    table.initialize(active_schedule, 0); // size is ignored

    BOOST_CHECK_EQUAL(table.size(), 0);

    // Insert
    BOOST_CHECK(table.insert(0));
    BOOST_CHECK_EQUAL(table.size(), 1);
    BOOST_CHECK(table.is_selected(0));
    BOOST_CHECK(!table.is_selected(1));
    table.insert(0); // should be a no-op on size
    BOOST_CHECK_EQUAL(table.size(), 1);

    // Remove
    table.remove(0);
    BOOST_CHECK_EQUAL(table.size(), 0);
    BOOST_CHECK(!table.is_selected(0));

    // Insert multiple
    for (unsigned i = 0; i < 10; ++i) {
        table.insert(i);
    }
    BOOST_CHECK_EQUAL(table.size(), 10);

    // Test reset
    table.reset_node_selection();
    BOOST_CHECK_EQUAL(table.size(), 0);
    BOOST_CHECK(!table.is_selected(0));
}


// Tests for vertex_selection_strategy
BOOST_AUTO_TEST_CASE(vertex_selection_strategy_test) {
    using affinity_table_t = adaptive_affinity_table<graph, double, int, kl_active_schedule_t, 1>;
    using selection_strategy_t = vertex_selection_strategy<graph, affinity_table_t, int, kl_active_schedule_t>;
    
    selection_strategy_t strategy;
    std::mt19937 gen(0);
    const unsigned end_step = active_schedule.num_steps() - 1;
    strategy.initialize(active_schedule, gen, 0, end_step);
    strategy.selection_threshold = 5;

    // Test permutation selection
    strategy.setup(0, end_step);
    BOOST_CHECK_EQUAL(strategy.permutation.size(), 20);

    affinity_table_t table;
    table.initialize(active_schedule, 20);

    strategy.select_nodes_permutation_threshold(5, table);
    BOOST_CHECK_EQUAL(table.size(), 5);
    BOOST_CHECK_EQUAL(strategy.permutation_idx, 5);

    strategy.select_nodes_permutation_threshold(5, table);
    BOOST_CHECK_EQUAL(table.size(), 10);
    BOOST_CHECK_EQUAL(strategy.permutation_idx, 10);

    strategy.select_nodes_permutation_threshold(15, table);
    BOOST_CHECK_EQUAL(table.size(), 20);
    BOOST_CHECK_EQUAL(strategy.permutation_idx, 0); // should wrap around and reshuffle

    table.reset_node_selection();
    strategy.max_work_counter = 0;
    strategy.select_nodes_max_work_proc(5, table, 0, 4);
    // In the new fixture, steps 0-4 contain nodes 0-4 respectively.
    // select_nodes_max_work_proc will select one node from each step.
    BOOST_CHECK_EQUAL(table.size(), 5);
    BOOST_CHECK(table.is_selected(0));
    BOOST_CHECK(table.is_selected(1));
    BOOST_CHECK(table.is_selected(2));
    BOOST_CHECK(table.is_selected(3));
    BOOST_CHECK(table.is_selected(4));
    BOOST_CHECK_EQUAL(strategy.max_work_counter, 5);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(kl_active_schedule_tests, ScheduleFixture)

using VertexType = graph::vertex_idx;

BOOST_AUTO_TEST_CASE(kl_move_struct_test) {
    using kl_move = kl_move_struct<double, VertexType>;
    kl_move move(5, 10.0, 1, 2, 3, 4);

    kl_move reversed = move.reverse_move();

    BOOST_CHECK_EQUAL(reversed.node, 5);
    BOOST_CHECK_EQUAL(reversed.gain, -10.0);
    BOOST_CHECK_EQUAL(reversed.from_proc, 3);
    BOOST_CHECK_EQUAL(reversed.from_step, 4);
    BOOST_CHECK_EQUAL(reversed.to_proc, 1);
    BOOST_CHECK_EQUAL(reversed.to_step, 2);
}

BOOST_AUTO_TEST_CASE(work_datastructures_initialization_test) {
    auto& wd = active_schedule.work_datastructures;

    // Step 0: node 0 on proc 0, work 1. Other procs have 0 work.
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 0), 1);
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 1), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 2), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 3), 0);
    BOOST_CHECK_EQUAL(wd.step_max_work(0), 1);
    BOOST_CHECK_EQUAL(wd.step_second_max_work(0), 0);
    BOOST_CHECK_EQUAL(wd.step_max_work_processor_count[0], 1);

    // Step 4: node 4 on proc 0, work 5.
    BOOST_CHECK_EQUAL(wd.step_proc_work(4, 0), 5);
    BOOST_CHECK_EQUAL(wd.step_proc_work(4, 1), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(4, 2), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(4, 3), 0);
    BOOST_CHECK_EQUAL(wd.step_max_work(4), 5);
    BOOST_CHECK_EQUAL(wd.step_second_max_work(4), 0);
    BOOST_CHECK_EQUAL(wd.step_max_work_processor_count[4], 1);
}

BOOST_AUTO_TEST_CASE(work_datastructures_apply_move_test) {
    auto& wd = active_schedule.work_datastructures;
    using kl_move = kl_move_struct<double, VertexType>;

    // Move within same superstep
    // Move node 0 (work 1) from proc 0 to proc 3 in step 0
    kl_move move1(0, 0.0, 0, 0, 3, 0);
    wd.apply_move(move1, 1); // work_weight of node 0 is 1

    // Before: {1,0,0,0}, After: {0,0,0,1}
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 0), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 1), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 2), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 3), 1);
    BOOST_CHECK_EQUAL(wd.step_max_work(0), 1);
    BOOST_CHECK_EQUAL(wd.step_second_max_work(0), 0);
    BOOST_CHECK_EQUAL(wd.step_max_work_processor_count[0], 1);

    // Move to different superstep
    // Move node 4 (work 5) from proc 0, step 4 to proc 1, step 0
    kl_move move2(4, 0.0, 0, 4, 1, 0);
    wd.apply_move(move2, 5); // work_weight of node 4 is 5

    // Step 0 state after move1: {0,0,0,1}. max=1
    // After move2: {0,5,0,1}. max=5
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 0), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 1), 5);
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 2), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(0, 3), 1);
    BOOST_CHECK_EQUAL(wd.step_max_work(0), 5);
    BOOST_CHECK_EQUAL(wd.step_second_max_work(0), 1);
    BOOST_CHECK_EQUAL(wd.step_max_work_processor_count[0], 1);

    // Step 4 state before move2: {5,0,0,0}. max=5
    // After move2: {0,0,0,0}. max=0
    BOOST_CHECK_EQUAL(wd.step_proc_work(4, 0), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(4, 1), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(4, 2), 0);
    BOOST_CHECK_EQUAL(wd.step_proc_work(4, 3), 0);
    BOOST_CHECK_EQUAL(wd.step_max_work(4), 0);
    BOOST_CHECK_EQUAL(wd.step_second_max_work(4), 0);
    BOOST_CHECK_EQUAL(wd.step_max_work_processor_count[4], 3); // All 4 procs have work 0, so count is 3.
}

BOOST_AUTO_TEST_CASE(active_schedule_initialization_test) {
    BOOST_CHECK_EQUAL(active_schedule.num_steps(), 20);
    BOOST_CHECK_EQUAL(active_schedule.assigned_processor(0), 0);
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(0), 0);
    BOOST_CHECK_EQUAL(active_schedule.assigned_processor(19), 3);
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(19), 19);
    BOOST_CHECK(active_schedule.is_feasible());
}

BOOST_AUTO_TEST_CASE(active_schedule_apply_move_test) {
    using kl_move = kl_move_struct<double, VertexType>;
    using thread_data_t = thread_local_active_schedule_data<graph, double>;
    thread_data_t thread_data;
    thread_data.initialize_cost(0);

    // Move node 1 (step 1) to step 0. This should create a violation with node 0 (step 0).
    // Edge 0 -> 1.
    kl_move move(1, 0.0, 1, 1, 1, 0);
    active_schedule.apply_move(move, thread_data);

    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(1), 0);
    BOOST_CHECK_EQUAL(active_schedule.getSetSchedule().step_processor_vertices[1][1].count(1), 0);
    BOOST_CHECK_EQUAL(active_schedule.getSetSchedule().step_processor_vertices[0][1].count(1), 1);

    BOOST_CHECK(!thread_data.feasible);
    BOOST_CHECK_EQUAL(thread_data.current_violations.size(), 1);
    BOOST_CHECK_EQUAL(thread_data.new_violations.size(), 1);
    BOOST_CHECK(thread_data.new_violations.count(0));
}

BOOST_AUTO_TEST_CASE(active_schedule_compute_violations_test) {
    using thread_data_t = thread_local_active_schedule_data<graph, double>;
    thread_data_t thread_data;

    // Manually create a violation
    schedule.setAssignedSuperstep(1, 0); // node 1 is now in step 0 (was 1)
    schedule.setAssignedSuperstep(0, 1); // node 0 is now in step 1 (was 0)
    // Now we have a violation for edge 0 -> 1, since step(0) > step(1)
    active_schedule.initialize(schedule);
    
    active_schedule.compute_violations(thread_data);

    BOOST_CHECK(!thread_data.feasible);
    BOOST_CHECK_EQUAL(thread_data.current_violations.size(), 1);
}

BOOST_AUTO_TEST_CASE(active_schedule_revert_moves_test) {
    using kl_move = kl_move_struct<double, VertexType>;
    using thread_data_t = thread_local_active_schedule_data<graph, double>;
    
    kl_active_schedule_t original_schedule;
    original_schedule.initialize(schedule);

    thread_data_t thread_data;
    thread_data.initialize_cost(0);

    kl_move move1(0, 0.0, 0, 0, 1, 0);
    kl_move move2(1, 0.0, 1, 1, 2, 1);
    active_schedule.apply_move(move1, thread_data);
    active_schedule.apply_move(move2, thread_data);

    BOOST_CHECK_EQUAL(active_schedule.assigned_processor(0), 1);
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(1), 1);

    struct dummy_comm_ds {
        void update_datastructure_after_move(const kl_move&, unsigned, unsigned) {}
    } comm_ds;

    // Revert both moves
    active_schedule.revert_schedule_to_bound(0, 0.0, true, comm_ds, thread_data, 0, 4);

    BOOST_CHECK_EQUAL(active_schedule.assigned_processor(0), original_schedule.assigned_processor(0));
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(0), original_schedule.assigned_superstep(0));
    BOOST_CHECK_EQUAL(active_schedule.assigned_processor(1), original_schedule.assigned_processor(1));
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(1), original_schedule.assigned_superstep(1));
}

BOOST_AUTO_TEST_CASE(active_schedule_revert_to_best_schedule_test) {
    using kl_move = kl_move_struct<double, VertexType>;
    using thread_data_t = thread_local_active_schedule_data<graph, double>;

    thread_data_t thread_data;
    thread_data.initialize_cost(100);

    // Apply 3 moves
    kl_move move1(0, 0.0, 0, 0, 1, 0); // node 0 from (p0,s0) to (p1,s0)
    active_schedule.apply_move(move1, thread_data);
    thread_data.update_cost(-10); // cost 90

    kl_move move2(1, 0.0, 1, 1, 2, 1); // node 1 from (p1,s1) to (p2,s1)
    active_schedule.apply_move(move2, thread_data);
    thread_data.update_cost(-10); // cost 80, best is here

    kl_move move3(2, 0.0, 2, 2, 3, 2); // node 2 from (p2,s2) to (p3,s2)
    active_schedule.apply_move(move3, thread_data);
    thread_data.update_cost(+5); // cost 85

    BOOST_CHECK_EQUAL(thread_data.best_schedule_idx, 2);
    BOOST_CHECK_EQUAL(thread_data.applied_moves.size(), 3);

    struct dummy_comm_ds {
        void update_datastructure_after_move(const kl_move&, unsigned, unsigned) {}
    } comm_ds;
    
    unsigned end_step = active_schedule.num_steps() - 1;
    // Revert to best. start_move=0 means no step removal logic is triggered.
    active_schedule.revert_to_best_schedule(0, 0, comm_ds, thread_data, 0, end_step);

    BOOST_CHECK_EQUAL(thread_data.cost, 80.0); // Check cost is reverted to best
    BOOST_CHECK_EQUAL(thread_data.applied_moves.size(), 0);
    BOOST_CHECK_EQUAL(thread_data.best_schedule_idx, 0); // Reset for next iteration

    // Check schedule state is after move2
    BOOST_CHECK_EQUAL(active_schedule.assigned_processor(0), 1); // from move1
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(0), 0);
    BOOST_CHECK_EQUAL(active_schedule.assigned_processor(1), 2); // from move2
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(1), 1);
    BOOST_CHECK_EQUAL(active_schedule.assigned_processor(2), 2); // Reverted, so original
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(2), 2); // Reverted, so original
}

BOOST_AUTO_TEST_CASE(active_schedule_swap_empty_step_fwd_test) {
    // Make step 1 empty by moving node 1 to step 0
    active_schedule.getVectorSchedule().setAssignedSuperstep(1, 0);
    active_schedule.initialize(active_schedule.getVectorSchedule()); // re-init to update set_schedule and work_ds

    BOOST_CHECK_EQUAL(active_schedule.get_step_total_work(1), 0);

    // Swap empty step 1 forward to position 3
    active_schedule.swap_empty_step_fwd(1, 3);

    // Node from original step 2 should be in step 1
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(2), 1);
    // Node from original step 3 should be in step 2
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(3), 2);
    // Step 3 should now be empty
    BOOST_CHECK_EQUAL(active_schedule.get_step_total_work(3), 0);
}

BOOST_AUTO_TEST_CASE(active_schedule_remove_empty_step_test) {
    // Make step 1 empty by moving node 1 to step 0
    active_schedule.getVectorSchedule().setAssignedSuperstep(1, 0);
    active_schedule.initialize(active_schedule.getVectorSchedule());

    unsigned original_num_steps = active_schedule.num_steps();
    unsigned original_step_of_node_8 = active_schedule.assigned_superstep(8); // should be 2

    active_schedule.remove_empty_step(1);

    BOOST_CHECK_EQUAL(active_schedule.num_steps(), original_num_steps - 1);
    // Node 8 should be shifted back by one step
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(8), original_step_of_node_8 - 1); // 8 -> 7
    // Node 3 (in step 3) should be shifted back by one step
    BOOST_CHECK_EQUAL(active_schedule.assigned_superstep(3), 2);
}

BOOST_AUTO_TEST_SUITE_END()