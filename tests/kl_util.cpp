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
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_util.hpp"

#include <boost/test/unit_test.hpp>
#include <numeric>
#include <set>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_active_schedule.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;
using Graph = computational_dag_edge_idx_vector_impl_def_int_t;
using KlActiveScheduleT = kl_active_schedule<Graph, double, no_local_search_memory_constraint>;

// Test fixture for setting up a schedule
struct ScheduleFixture {
    BspInstance<Graph> instance_;
    BspSchedule<Graph> schedule_;
    KlActiveScheduleT activeSchedule_;

    ScheduleFixture() : schedule_(instance_) {
        // Setup a simple graph and schedule
        auto &dag = instance_.GetComputationalDag();
        for (int i = 0; i < 20; ++i) {
            dag.add_vertex(i + 1, i + 1, i + 1);
        }
        for (unsigned i = 0; i < 19; ++i) {
            dag.add_edge(i, i + 1, 1);
        }

        instance_.GetArchitecture().setNumberOfProcessors(4);
        instance_.GetArchitecture().setCommunicationCosts(1);
        instance_.GetArchitecture().setSynchronisationCosts(10);

        std::vector<unsigned> procs(20);
        std::vector<unsigned> steps(20);
        for (unsigned i = 0; i < 20; ++i) {
            procs[i] = i % 4;
            // Each node in its own superstep creates a valid initial schedule.
            steps[i] = i;
        }

        schedule_.setAssignedProcessors(std::move(procs));
        schedule_.setAssignedSupersteps(std::move(steps));
        schedule_.updateNumberOfSupersteps();

        activeSchedule_.initialize(schedule_);
    }
};

BOOST_FIXTURE_TEST_SUITE(kl_util_tests, ScheduleFixture)

// Tests for reward_penalty_strategy
BOOST_AUTO_TEST_CASE(RewardPenaltyStrategyTest) {
    reward_penalty_strategy<double, int, KlActiveScheduleT> rps;
    rps.initialize(activeSchedule_, 10.0, 20.0);

    BOOST_CHECK_EQUAL(rps.max_weight, 20.0);
    BOOST_CHECK_CLOSE(rps.initial_penalty, std::sqrt(20.0), 1e-9);

    rps.init_reward_penalty(2.0);
    BOOST_CHECK_CLOSE(rps.penalty, std::sqrt(20.0) * 2.0, 1e-9);
    BOOST_CHECK_CLOSE(rps.reward, 20.0 * 2.0, 1e-9);
}

// Tests for lock managers
template <typename LockManager>
void TestLockManager() {
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

BOOST_AUTO_TEST_CASE(LockManagersTest) {
    TestLockManager<set_vertex_lock_manger<unsigned>>();
    TestLockManager<vector_vertex_lock_manger<unsigned>>();
}

// Tests for adaptive_affinity_table
BOOST_AUTO_TEST_CASE(AdaptiveAffinityTableTest) {
    using AffinityTableT = adaptive_affinity_table<Graph, double, KlActiveScheduleT, 1>;
    AffinityTableT table;
    table.initialize(activeSchedule_, 5);

    BOOST_CHECK_EQUAL(table.size(), 0);

    // Insert
    BOOST_CHECK(table.insert(0));
    BOOST_CHECK_EQUAL(table.size(), 1);
    BOOST_CHECK(table.is_selected(0));
    BOOST_CHECK(!table.is_selected(1));
    BOOST_CHECK(!table.insert(0));    // already present

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
    std::set<unsigned> expectedSelected = {0, 1, 2, 4, 6, 8, 9};
    std::set<unsigned> actualSelected;
    const auto &selectedNodesVec = table.get_selected_nodes();
    for (size_t i = 0; i < table.size(); ++i) {
        actualSelected.insert(static_cast<unsigned>(selectedNodesVec[i]));
    }
    BOOST_CHECK(expectedSelected == actualSelected);

    for (unsigned i = 0; i < 20; ++i) {
        if (expectedSelected.count(i)) {
            BOOST_CHECK(table.is_selected(i));
        } else {
            BOOST_CHECK(!table.is_selected(i));
        }
    }

    // Check that indices are correct
    for (size_t i = 0; i < table.size(); ++i) {
        BOOST_CHECK_EQUAL(table.get_selected_nodes_idx(selectedNodesVec[i]), i);
    }

    // Test reset
    table.reset_node_selection();
    BOOST_CHECK_EQUAL(table.size(), 0);
    BOOST_CHECK(!table.is_selected(0));
    BOOST_CHECK(!table.is_selected(1));
}

// Tests for static_affinity_table
BOOST_AUTO_TEST_CASE(StaticAffinityTableTest) {
    using AffinityTableT = static_affinity_table<Graph, double, KlActiveScheduleT, 1>;
    AffinityTableT table;
    table.initialize(activeSchedule_, 0);    // size is ignored

    BOOST_CHECK_EQUAL(table.size(), 0);

    // Insert
    BOOST_CHECK(table.insert(0));
    BOOST_CHECK_EQUAL(table.size(), 1);
    BOOST_CHECK(table.is_selected(0));
    BOOST_CHECK(!table.is_selected(1));
    table.insert(0);    // should be a no-op on size
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
BOOST_AUTO_TEST_CASE(VertexSelectionStrategyTest) {
    using AffinityTableT = adaptive_affinity_table<Graph, double, KlActiveScheduleT, 1>;
    using SelectionStrategyT = vertex_selection_strategy<Graph, AffinityTableT, KlActiveScheduleT>;

    SelectionStrategyT strategy;
    std::mt19937 gen(0);
    const unsigned endStep = activeSchedule_.num_steps() - 1;
    strategy.initialize(activeSchedule_, gen, 0, endStep);
    strategy.selection_threshold = 5;

    // Test permutation selection
    strategy.setup(0, endStep);
    BOOST_CHECK_EQUAL(strategy.permutation.size(), 20);

    AffinityTableT table;
    table.initialize(activeSchedule_, 20);

    strategy.select_nodes_permutation_threshold(5, table);
    BOOST_CHECK_EQUAL(table.size(), 5);
    BOOST_CHECK_EQUAL(strategy.permutation_idx, 5);

    strategy.select_nodes_permutation_threshold(5, table);
    BOOST_CHECK_EQUAL(table.size(), 10);
    BOOST_CHECK_EQUAL(strategy.permutation_idx, 10);

    strategy.select_nodes_permutation_threshold(15, table);
    BOOST_CHECK_EQUAL(table.size(), 20);
    BOOST_CHECK_EQUAL(strategy.permutation_idx, 0);    // should wrap around and reshuffle

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

using VertexType = Graph::vertex_idx;

BOOST_AUTO_TEST_CASE(KlMoveStructTest) {
    using KlMove = kl_move_struct<double, VertexType>;
    KlMove move(5, 10.0, 1, 2, 3, 4);

    KlMove reversed = move.reverse_move();

    BOOST_CHECK_EQUAL(reversed.node, 5);
    BOOST_CHECK_EQUAL(reversed.gain, -10.0);
    BOOST_CHECK_EQUAL(reversed.from_proc, 3);
    BOOST_CHECK_EQUAL(reversed.from_step, 4);
    BOOST_CHECK_EQUAL(reversed.to_proc, 1);
    BOOST_CHECK_EQUAL(reversed.to_step, 2);
}

BOOST_AUTO_TEST_CASE(WorkDatastructuresInitializationTest) {
    auto &wd = activeSchedule_.work_datastructures;

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

BOOST_AUTO_TEST_CASE(WorkDatastructuresApplyMoveTest) {
    auto &wd = activeSchedule_.work_datastructures;
    using KlMove = kl_move_struct<double, VertexType>;

    // Move within same superstep
    // Move node 0 (work 1) from proc 0 to proc 3 in step 0
    KlMove move1(0, 0.0, 0, 0, 3, 0);
    wd.apply_move(move1, 1);    // work_weight of node 0 is 1

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
    KlMove move2(4, 0.0, 0, 4, 1, 0);
    wd.apply_move(move2, 5);    // work_weight of node 4 is 5

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
    BOOST_CHECK_EQUAL(wd.step_max_work_processor_count[4], 3);    // All 4 procs have work 0, so count is 3.
}

BOOST_AUTO_TEST_CASE(ActiveScheduleInitializationTest) {
    BOOST_CHECK_EQUAL(activeSchedule_.num_steps(), 20);
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_processor(0), 0);
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(0), 0);
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_processor(19), 3);
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(19), 19);
    BOOST_CHECK(activeSchedule_.is_feasible());
}

BOOST_AUTO_TEST_CASE(ActiveScheduleApplyMoveTest) {
    using KlMove = kl_move_struct<double, VertexType>;
    using ThreadDataT = thread_local_active_schedule_data<Graph, double>;
    ThreadDataT threadData;
    threadData.initialize_cost(0);

    // Move node 1 (step 1) to step 0. This should create a violation with node 0 (step 0).
    // Edge 0 -> 1.
    KlMove move(1, 0.0, 1, 1, 1, 0);
    activeSchedule_.apply_move(move, threadData);

    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(1), 0);
    BOOST_CHECK_EQUAL(activeSchedule_.getSetSchedule().step_processor_vertices[1][1].count(1), 0);
    BOOST_CHECK_EQUAL(activeSchedule_.getSetSchedule().step_processor_vertices[0][1].count(1), 1);

    BOOST_CHECK(!threadData.feasible);
    BOOST_CHECK_EQUAL(threadData.current_violations.size(), 1);
    BOOST_CHECK_EQUAL(threadData.new_violations.size(), 1);
    BOOST_CHECK(threadData.new_violations.count(0));
}

BOOST_AUTO_TEST_CASE(ActiveScheduleComputeViolationsTest) {
    using ThreadDataT = thread_local_active_schedule_data<Graph, double>;
    ThreadDataT threadData;

    // Manually create a violation
    schedule_.setAssignedSuperstep(1, 0);    // node 1 is now in step 0 (was 1)
    schedule_.setAssignedSuperstep(0, 1);    // node 0 is now in step 1 (was 0)
    // Now we have a violation for edge 0 -> 1, since step(0) > step(1)
    activeSchedule_.initialize(schedule_);

    activeSchedule_.compute_violations(threadData);

    BOOST_CHECK(!threadData.feasible);
    BOOST_CHECK_EQUAL(threadData.current_violations.size(), 1);
}

BOOST_AUTO_TEST_CASE(ActiveScheduleRevertMovesTest) {
    using KlMove = kl_move_struct<double, VertexType>;
    using ThreadDataT = thread_local_active_schedule_data<Graph, double>;

    KlActiveScheduleT originalSchedule;
    originalSchedule.initialize(schedule_);

    ThreadDataT threadData;
    threadData.initialize_cost(0);

    KlMove move1(0, 0.0, 0, 0, 1, 0);
    KlMove move2(1, 0.0, 1, 1, 2, 1);
    activeSchedule_.apply_move(move1, threadData);
    activeSchedule_.apply_move(move2, threadData);

    BOOST_CHECK_EQUAL(activeSchedule_.assigned_processor(0), 1);
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(1), 1);

    struct DummyCommDs {
        void UpdateDatastructureAfterMove(const KlMove &, unsigned, unsigned) {}
    } commDs;

    // Revert both moves
    activeSchedule_.revert_schedule_to_bound(0, 0.0, true, commDs, threadData, 0, 4);

    BOOST_CHECK_EQUAL(activeSchedule_.assigned_processor(0), originalSchedule.assigned_processor(0));
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(0), originalSchedule.assigned_superstep(0));
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_processor(1), originalSchedule.assigned_processor(1));
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(1), originalSchedule.assigned_superstep(1));
}

BOOST_AUTO_TEST_CASE(ActiveScheduleRevertToBestScheduleTest) {
    using KlMove = kl_move_struct<double, VertexType>;
    using ThreadDataT = thread_local_active_schedule_data<Graph, double>;

    ThreadDataT threadData;
    threadData.initialize_cost(100);

    // Apply 3 moves
    KlMove move1(0, 0.0, 0, 0, 1, 0);    // node 0 from (p0,s0) to (p1,s0)
    activeSchedule_.apply_move(move1, threadData);
    threadData.update_cost(-10);    // cost 90

    KlMove move2(1, 0.0, 1, 1, 2, 1);    // node 1 from (p1,s1) to (p2,s1)
    activeSchedule_.apply_move(move2, threadData);
    threadData.update_cost(-10);    // cost 80, best is here

    KlMove move3(2, 0.0, 2, 2, 3, 2);    // node 2 from (p2,s2) to (p3,s2)
    activeSchedule_.apply_move(move3, threadData);
    threadData.update_cost(+5);    // cost 85

    BOOST_CHECK_EQUAL(threadData.best_schedule_idx, 2);
    BOOST_CHECK_EQUAL(threadData.applied_moves.size(), 3);

    struct DummyCommDs {
        void UpdateDatastructureAfterMove(const KlMove &, unsigned, unsigned) {}
    } commDs;

    unsigned endStep = activeSchedule_.num_steps() - 1;
    // Revert to best. start_move=0 means no step removal logic is triggered.
    activeSchedule_.revert_to_best_schedule(0, 0, commDs, threadData, 0, endStep);

    BOOST_CHECK_EQUAL(threadData.cost, 80.0);    // Check cost is reverted to best
    BOOST_CHECK_EQUAL(threadData.applied_moves.size(), 0);
    BOOST_CHECK_EQUAL(threadData.best_schedule_idx, 0);    // Reset for next iteration

    // Check schedule state is after move2
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_processor(0), 1);    // from move1
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(0), 0);
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_processor(1), 2);    // from move2
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(1), 1);
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_processor(2), 2);    // Reverted, so original
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(2), 2);    // Reverted, so original
}

BOOST_AUTO_TEST_CASE(ActiveScheduleSwapEmptyStepFwdTest) {
    // Make step 1 empty by moving node 1 to step 0
    activeSchedule_.getVectorSchedule().setAssignedSuperstep(1, 0);
    activeSchedule_.initialize(activeSchedule_.getVectorSchedule());    // re-init to update set_schedule and work_ds

    BOOST_CHECK_EQUAL(activeSchedule_.get_step_total_work(1), 0);

    // Swap empty step 1 forward to position 3
    activeSchedule_.swap_empty_step_fwd(1, 3);

    // Node from original step 2 should be in step 1
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(2), 1);
    // Node from original step 3 should be in step 2
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(3), 2);
    // Step 3 should now be empty
    BOOST_CHECK_EQUAL(activeSchedule_.get_step_total_work(3), 0);
}

BOOST_AUTO_TEST_CASE(ActiveScheduleRemoveEmptyStepTest) {
    // Make step 1 empty by moving node 1 to step 0
    activeSchedule_.getVectorSchedule().setAssignedSuperstep(1, 0);
    activeSchedule_.initialize(activeSchedule_.getVectorSchedule());

    unsigned originalNumSteps = activeSchedule_.num_steps();
    unsigned originalStepOfNode8 = activeSchedule_.assigned_superstep(8);    // should be 2

    activeSchedule_.remove_empty_step(1);

    BOOST_CHECK_EQUAL(activeSchedule_.num_steps(), originalNumSteps - 1);
    // Node 8 should be shifted back by one step
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(8), originalStepOfNode8 - 1);    // 8 -> 7
    // Node 3 (in step 3) should be shifted back by one step
    BOOST_CHECK_EQUAL(activeSchedule_.assigned_superstep(3), 2);
}

BOOST_AUTO_TEST_SUITE_END()
