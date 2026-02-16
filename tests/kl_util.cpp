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
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_util.hpp"

#include <boost/test/unit_test.hpp>
#include <numeric>
#include <set>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_active_schedule.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;
using Graph = ComputationalDagEdgeIdxVectorImplDefIntT;
using KlActiveScheduleT = KlActiveSchedule<Graph, double, NoLocalSearchMemoryConstraint>;

// Test fixture for setting up a schedule
struct ScheduleFixture {
    BspInstance<Graph> instance_;
    BspSchedule<Graph> schedule_;
    KlActiveScheduleT activeSchedule_;

    ScheduleFixture() : schedule_(instance_) {
        // Setup a simple graph and schedule
        auto &dag = instance_.GetComputationalDag();
        for (int i = 0; i < 20; ++i) {
            dag.AddVertex(i + 1, i + 1, i + 1);
        }
        for (unsigned i = 0; i < 19; ++i) {
            dag.AddEdge(i, i + 1, 1);
        }

        instance_.GetArchitecture().SetNumberOfProcessors(4);
        instance_.GetArchitecture().SetCommunicationCosts(1);
        instance_.GetArchitecture().SetSynchronisationCosts(10);

        std::vector<unsigned> procs(20);
        std::vector<unsigned> steps(20);
        for (unsigned i = 0; i < 20; ++i) {
            procs[i] = i % 4;
            // Each node in its own superstep creates a valid initial schedule.
            steps[i] = i;
        }

        schedule_.SetAssignedProcessors(std::move(procs));
        schedule_.SetAssignedSupersteps(std::move(steps));
        schedule_.UpdateNumberOfSupersteps();

        activeSchedule_.Initialize(schedule_);
    }
};

BOOST_FIXTURE_TEST_SUITE(kl_util_tests, ScheduleFixture)

// Tests for RewardPenaltyStrategy
BOOST_AUTO_TEST_CASE(RewardPenaltyStrategyTest) {
    RewardPenaltyStrategy<double, int, KlActiveScheduleT> rps;
    rps.Initialize(activeSchedule_, 10.0, 20.0);

    BOOST_CHECK_EQUAL(rps.maxWeight_, 20.0);
    BOOST_CHECK_CLOSE(rps.initialPenalty_, std::sqrt(20.0), 1e-9);

    rps.InitRewardPenalty(2.0);
    BOOST_CHECK_CLOSE(rps.penalty_, std::sqrt(20.0) * 2.0, 1e-9);
    BOOST_CHECK_CLOSE(rps.reward_, 20.0 * 2.0, 1e-9);
}

// Tests for lock managers
template <typename LockManager>
void TestLockManager() {
    LockManager lm;
    lm.Initialize(10);

    BOOST_CHECK(!lm.IsLocked(5));
    lm.Lock(5);
    BOOST_CHECK(lm.IsLocked(5));
    BOOST_CHECK(!lm.IsLocked(6));
    lm.Unlock(5);
    BOOST_CHECK(!lm.IsLocked(5));

    lm.Lock(1);
    lm.Lock(3);
    lm.Lock(5);
    BOOST_CHECK(lm.IsLocked(3));
    lm.Clear();
    BOOST_CHECK(!lm.IsLocked(1));
    BOOST_CHECK(!lm.IsLocked(3));
    BOOST_CHECK(!lm.IsLocked(5));
}

BOOST_AUTO_TEST_CASE(LockManagersTest) {
    TestLockManager<SetVertexLockManager<unsigned>>();
    TestLockManager<VectorVertexLockManager<unsigned>>();
}

// Tests for AdaptiveAffinityTable
BOOST_AUTO_TEST_CASE(AdaptiveAffinityTableTest) {
    using AffinityTableT = AdaptiveAffinityTable<Graph, double, KlActiveScheduleT, 1>;
    AffinityTableT table;
    table.Initialize(activeSchedule_, 5);

    BOOST_CHECK_EQUAL(table.size(), 0);

    // Insert
    BOOST_CHECK(table.Insert(0));
    BOOST_CHECK_EQUAL(table.size(), 1);
    BOOST_CHECK(table.IsSelected(0));
    BOOST_CHECK(!table.IsSelected(1));
    BOOST_CHECK(!table.Insert(0));    // already present

    // Remove
    table.Remove(0);
    BOOST_CHECK_EQUAL(table.size(), 0);
    BOOST_CHECK(!table.IsSelected(0));

    // Insert more to test resizing
    for (unsigned i = 0; i < 10; ++i) {
        BOOST_CHECK(table.Insert(i));
    }
    BOOST_CHECK_EQUAL(table.size(), 10);
    for (unsigned i = 0; i < 10; ++i) {
        BOOST_CHECK(table.IsSelected(i));
    }

    // Test trim
    table.Remove(3);
    table.Remove(5);
    table.Remove(7);
    BOOST_CHECK_EQUAL(table.size(), 7);

    table.Trim();
    BOOST_CHECK_EQUAL(table.size(), 7);

    // After trim, the gaps should be filled.
    std::set<unsigned> expectedSelected = {0, 1, 2, 4, 6, 8, 9};
    std::set<unsigned> actualSelected;
    const auto &selectedNodesVec = table.GetSelectedNodes();
    for (size_t i = 0; i < table.size(); ++i) {
        actualSelected.insert(static_cast<unsigned>(selectedNodesVec[i]));
    }
    BOOST_CHECK(expectedSelected == actualSelected);

    for (unsigned i = 0; i < 20; ++i) {
        if (expectedSelected.count(i)) {
            BOOST_CHECK(table.IsSelected(i));
        } else {
            BOOST_CHECK(!table.IsSelected(i));
        }
    }

    // Check that indices are correct
    for (size_t i = 0; i < table.size(); ++i) {
        BOOST_CHECK_EQUAL(table.GetSelectedNodesIdx(selectedNodesVec[i]), i);
    }

    // Test reset
    table.ResetNodeSelection();
    BOOST_CHECK_EQUAL(table.size(), 0);
    BOOST_CHECK(!table.IsSelected(0));
    BOOST_CHECK(!table.IsSelected(1));
}

// Tests for StaticAffinityTable
BOOST_AUTO_TEST_CASE(StaticAffinityTableTest) {
    using AffinityTableT = StaticAffinityTable<Graph, double, KlActiveScheduleT, 1>;
    AffinityTableT table;
    table.Initialize(activeSchedule_, 0);    // size is ignored

    BOOST_CHECK_EQUAL(table.size(), 0);

    // Insert
    BOOST_CHECK(table.Insert(0));
    BOOST_CHECK_EQUAL(table.size(), 1);
    BOOST_CHECK(table.IsSelected(0));
    BOOST_CHECK(!table.IsSelected(1));
    table.Insert(0);    // should be a no-op on size
    BOOST_CHECK_EQUAL(table.size(), 1);

    // Remove
    table.Remove(0);
    BOOST_CHECK_EQUAL(table.size(), 0);
    BOOST_CHECK(!table.IsSelected(0));

    // Insert multiple
    for (unsigned i = 0; i < 10; ++i) {
        table.Insert(i);
    }
    BOOST_CHECK_EQUAL(table.size(), 10);

    // Test reset
    table.ResetNodeSelection();
    BOOST_CHECK_EQUAL(table.size(), 0);
    BOOST_CHECK(!table.IsSelected(0));
}

// Tests for VertexSelectionStrategy
BOOST_AUTO_TEST_CASE(VertexSelectionStrategyTest) {
    using AffinityTableT = AdaptiveAffinityTable<Graph, double, KlActiveScheduleT, 1>;
    using SelectionStrategyT = VertexSelectionStrategy<Graph, AffinityTableT, KlActiveScheduleT>;

    SelectionStrategyT strategy;
    std::mt19937 gen(0);
    const unsigned endStep = activeSchedule_.NumSteps() - 1;
    strategy.Initialize(activeSchedule_, gen, 0, endStep);
    strategy.selectionThreshold_ = 5;

    // Test permutation selection
    strategy.Setup(0, endStep);
    BOOST_CHECK_EQUAL(strategy.permutation_.size(), 20);

    AffinityTableT table;
    table.Initialize(activeSchedule_, 20);

    strategy.SelectNodesPermutationThreshold(5, table);
    BOOST_CHECK_EQUAL(table.size(), 5);
    BOOST_CHECK_EQUAL(strategy.permutationIdx_, 5);

    strategy.SelectNodesPermutationThreshold(5, table);
    BOOST_CHECK_EQUAL(table.size(), 10);
    BOOST_CHECK_EQUAL(strategy.permutationIdx_, 10);

    strategy.SelectNodesPermutationThreshold(15, table);
    BOOST_CHECK_EQUAL(table.size(), 20);
    BOOST_CHECK_EQUAL(strategy.permutationIdx_, 0);    // should wrap around and reshuffle

    table.ResetNodeSelection();
    strategy.maxWorkCounter_ = 0;
    strategy.SelectNodesMaxWorkProc(5, table, 0, 4);
    // In the new fixture, steps 0-4 contain nodes 0-4 respectively.
    // select_nodes_max_work_proc will select one node from each step.
    BOOST_CHECK_EQUAL(table.size(), 5);
    BOOST_CHECK(table.IsSelected(0));
    BOOST_CHECK(table.IsSelected(1));
    BOOST_CHECK(table.IsSelected(2));
    BOOST_CHECK(table.IsSelected(3));
    BOOST_CHECK(table.IsSelected(4));
    BOOST_CHECK_EQUAL(strategy.maxWorkCounter_, 5);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(KlActiveSchedule_tests, ScheduleFixture)

using VertexType = Graph::VertexIdx;

BOOST_AUTO_TEST_CASE(KlMoveStructTest) {
    using KlMove = KlMoveStruct<double, VertexType>;
    KlMove move(5, 10.0, 1, 2, 3, 4);

    KlMove reversed = move.ReverseMove();

    BOOST_CHECK_EQUAL(reversed.node_, 5);
    BOOST_CHECK_EQUAL(reversed.gain_, -10.0);
    BOOST_CHECK_EQUAL(reversed.fromProc_, 3);
    BOOST_CHECK_EQUAL(reversed.fromStep_, 4);
    BOOST_CHECK_EQUAL(reversed.toProc_, 1);
    BOOST_CHECK_EQUAL(reversed.toStep_, 2);
}

BOOST_AUTO_TEST_CASE(WorkDatastructuresInitializationTest) {
    auto &wd = activeSchedule_.workDatastructures_;

    // Step 0: node 0 on proc 0, work 1. Other procs have 0 work.
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 0), 1);
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 1), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 2), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 3), 0);
    BOOST_CHECK_EQUAL(wd.StepMaxWork(0), 1);
    BOOST_CHECK_EQUAL(wd.StepSecondMaxWork(0), 0);
    BOOST_CHECK_EQUAL(wd.stepMaxWorkProcessorCount_[0], 1);

    // Step 4: node 4 on proc 0, work 5.
    BOOST_CHECK_EQUAL(wd.StepProcWork(4, 0), 5);
    BOOST_CHECK_EQUAL(wd.StepProcWork(4, 1), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(4, 2), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(4, 3), 0);
    BOOST_CHECK_EQUAL(wd.StepMaxWork(4), 5);
    BOOST_CHECK_EQUAL(wd.StepSecondMaxWork(4), 0);
    BOOST_CHECK_EQUAL(wd.stepMaxWorkProcessorCount_[4], 1);
}

BOOST_AUTO_TEST_CASE(WorkDatastructuresApplyMoveTest) {
    auto &wd = activeSchedule_.workDatastructures_;
    using KlMove = KlMoveStruct<double, VertexType>;

    // Move within same superstep
    // Move node 0 (work 1) from proc 0 to proc 3 in step 0
    KlMove move1(0, 0.0, 0, 0, 3, 0);
    wd.ApplyMove(move1, 1);    // work_weight of node 0 is 1

    // Before: {1,0,0,0}, After: {0,0,0,1}
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 0), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 1), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 2), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 3), 1);
    BOOST_CHECK_EQUAL(wd.StepMaxWork(0), 1);
    BOOST_CHECK_EQUAL(wd.StepSecondMaxWork(0), 0);
    BOOST_CHECK_EQUAL(wd.stepMaxWorkProcessorCount_[0], 1);

    // Move to different superstep
    // Move node 4 (work 5) from proc 0, step 4 to proc 1, step 0
    KlMove move2(4, 0.0, 0, 4, 1, 0);
    wd.ApplyMove(move2, 5);    // work_weight of node 4 is 5

    // Step 0 state after move1: {0,0,0,1}. max=1
    // After move2: {0,5,0,1}. max=5
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 0), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 1), 5);
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 2), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(0, 3), 1);
    BOOST_CHECK_EQUAL(wd.StepMaxWork(0), 5);
    BOOST_CHECK_EQUAL(wd.StepSecondMaxWork(0), 1);
    BOOST_CHECK_EQUAL(wd.stepMaxWorkProcessorCount_[0], 1);

    // Step 4 state before move2: {5,0,0,0}. max=5
    // After move2: {0,0,0,0}. max=0
    BOOST_CHECK_EQUAL(wd.StepProcWork(4, 0), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(4, 1), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(4, 2), 0);
    BOOST_CHECK_EQUAL(wd.StepProcWork(4, 3), 0);
    BOOST_CHECK_EQUAL(wd.StepMaxWork(4), 0);
    BOOST_CHECK_EQUAL(wd.StepSecondMaxWork(4), 0);
    BOOST_CHECK_EQUAL(wd.stepMaxWorkProcessorCount_[4], 3);    // All 4 procs have work 0, so count is 3.
}

BOOST_AUTO_TEST_CASE(ActiveScheduleInitializationTest) {
    BOOST_CHECK_EQUAL(activeSchedule_.NumSteps(), 20);
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedProcessor(0), 0);
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(0), 0);
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedProcessor(19), 3);
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(19), 19);
    BOOST_CHECK(activeSchedule_.IsFeasible());
}

BOOST_AUTO_TEST_CASE(ActiveScheduleApplyMoveTest) {
    using KlMove = KlMoveStruct<double, VertexType>;
    using ThreadDataT = ThreadLocalActiveScheduleData<Graph, double>;
    ThreadDataT threadData;
    threadData.InitializeCost(0);

    // Move node 1 (step 1) to step 0. This should create a violation with node 0 (step 0).
    // Edge 0 -> 1.
    KlMove move(1, 0.0, 1, 1, 1, 0);
    activeSchedule_.ApplyMove(move, threadData);

    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(1), 0);
    BOOST_CHECK_EQUAL(activeSchedule_.GetSetSchedule().GetProcessorStepVertices()[1][1].count(1), 0);
    BOOST_CHECK_EQUAL(activeSchedule_.GetSetSchedule().GetProcessorStepVertices()[0][1].count(1), 1);

    BOOST_CHECK(!threadData.feasible_);
    BOOST_CHECK_EQUAL(threadData.currentViolations_.size(), 1);
    BOOST_CHECK_EQUAL(threadData.newViolations_.size(), 1);
    BOOST_CHECK(threadData.newViolations_.count(0));
}

BOOST_AUTO_TEST_CASE(ActiveScheduleComputeViolationsTest) {
    using ThreadDataT = ThreadLocalActiveScheduleData<Graph, double>;
    ThreadDataT threadData;

    // Manually create a violation
    schedule_.SetAssignedSuperstep(1, 0);    // node 1 is now in step 0 (was 1)
    schedule_.SetAssignedSuperstep(0, 1);    // node 0 is now in step 1 (was 0)
    // Now we have a violation for edge 0 -> 1, since step(0) > step(1)
    activeSchedule_.Initialize(schedule_);

    activeSchedule_.ComputeViolations(threadData);

    BOOST_CHECK(!threadData.feasible_);
    BOOST_CHECK_EQUAL(threadData.currentViolations_.size(), 1);
}

BOOST_AUTO_TEST_CASE(ActiveScheduleRevertMovesTest) {
    using KlMove = KlMoveStruct<double, VertexType>;
    using ThreadDataT = ThreadLocalActiveScheduleData<Graph, double>;

    KlActiveScheduleT originalSchedule;
    originalSchedule.Initialize(schedule_);

    ThreadDataT threadData;
    threadData.InitializeCost(0);

    KlMove move1(0, 0.0, 0, 0, 1, 0);
    KlMove move2(1, 0.0, 1, 1, 2, 1);
    activeSchedule_.ApplyMove(move1, threadData);
    activeSchedule_.ApplyMove(move2, threadData);

    BOOST_CHECK_EQUAL(activeSchedule_.AssignedProcessor(0), 1);
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(1), 1);

    struct DummyCommDs {
        void UpdateDatastructureAfterMove(const KlMove &, unsigned, unsigned) {}
    } commDs;

    // Revert both moves
    unsigned endStep = 4;
    activeSchedule_.RevertScheduleToBound(0, 0.0, true, commDs, threadData, 0, endStep);

    BOOST_CHECK_EQUAL(activeSchedule_.AssignedProcessor(0), originalSchedule.AssignedProcessor(0));
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(0), originalSchedule.AssignedSuperstep(0));
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedProcessor(1), originalSchedule.AssignedProcessor(1));
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(1), originalSchedule.AssignedSuperstep(1));
}

BOOST_AUTO_TEST_CASE(ActiveScheduleRevertToBestScheduleTest) {
    using KlMove = KlMoveStruct<double, VertexType>;
    using ThreadDataT = ThreadLocalActiveScheduleData<Graph, double>;

    ThreadDataT threadData;
    threadData.InitializeCost(100);

    // Apply 3 moves
    KlMove move1(0, 0.0, 0, 0, 1, 0);    // node 0 from (p0,s0) to (p1,s0)
    activeSchedule_.ApplyMove(move1, threadData);
    threadData.UpdateCost(-10);    // cost 90

    KlMove move2(1, 0.0, 1, 1, 2, 1);    // node 1 from (p1,s1) to (p2,s1)
    activeSchedule_.ApplyMove(move2, threadData);
    threadData.UpdateCost(-10);    // cost 80, best is here

    KlMove move3(2, 0.0, 2, 2, 3, 2);    // node 2 from (p2,s2) to (p3,s2)
    activeSchedule_.ApplyMove(move3, threadData);
    threadData.UpdateCost(+5);    // cost 85

    BOOST_CHECK_EQUAL(threadData.bestScheduleIdx_, 2);
    BOOST_CHECK_EQUAL(threadData.appliedMoves_.size(), 3);

    struct DummyCommDs {
        void UpdateDatastructureAfterMove(const KlMove &, unsigned, unsigned) {}
    } commDs;

    unsigned endStep = activeSchedule_.NumSteps() - 1;
    // Revert to best.
    activeSchedule_.RevertToBestSchedule(commDs, threadData, 0, endStep);

    BOOST_CHECK_EQUAL(threadData.cost_, 80.0);    // Check cost is reverted to best
    BOOST_CHECK_EQUAL(threadData.appliedMoves_.size(), 0);
    BOOST_CHECK_EQUAL(threadData.bestScheduleIdx_, 0);    // Reset for next iteration

    // Check schedule state is after move2
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedProcessor(0), 1);    // from move1
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(0), 0);
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedProcessor(1), 2);    // from move2
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(1), 1);
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedProcessor(2), 2);    // Reverted, so original
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(2), 2);    // Reverted, so original
}

BOOST_AUTO_TEST_CASE(ActiveScheduleSwapEmptyStepFwdTest) {
    // Make step 1 empty by moving node 1 to step 0
    activeSchedule_.GetVectorSchedule().SetAssignedSuperstep(1, 0);
    activeSchedule_.Initialize(activeSchedule_.GetVectorSchedule());    // re-init to update set_schedule and work_ds

    BOOST_CHECK_EQUAL(activeSchedule_.GetStepTotalWork(1), 0);

    // Swap empty step 1 forward to position 3
    activeSchedule_.SwapEmptyStepFwd(1, 3);

    // Node from original step 2 should be in step 1
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(2), 1);
    // Node from original step 3 should be in step 2
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(3), 2);
    // Step 3 should now be empty
    BOOST_CHECK_EQUAL(activeSchedule_.GetStepTotalWork(3), 0);
}

BOOST_AUTO_TEST_CASE(ActiveScheduleRemoveEmptyStepTest) {
    // Make step 1 empty by moving node 1 to step 0
    activeSchedule_.GetVectorSchedule().SetAssignedSuperstep(1, 0);
    activeSchedule_.Initialize(activeSchedule_.GetVectorSchedule());

    unsigned originalNumSteps = activeSchedule_.NumSteps();
    unsigned originalStepOfNode8 = activeSchedule_.AssignedSuperstep(8);    // should be 2

    activeSchedule_.RemoveEmptyStep(1);

    BOOST_CHECK_EQUAL(activeSchedule_.NumSteps(), originalNumSteps - 1);
    // Node 8 should be shifted back by one step
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(8), originalStepOfNode8 - 1);    // 8 -> 7
    // Node 3 (in step 3) should be shifted back by one step
    BOOST_CHECK_EQUAL(activeSchedule_.AssignedSuperstep(3), 2);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_CASE(StalenessTest) {
    // Define a schedule with staleness 2
    class StaleSchedule : public BspSchedule<Graph> {
      public:
        using BspSchedule<Graph>::BspSchedule;

        unsigned GetStaleness() const override { return 2; }
    };

    // Create a simple 2-node graph locally
    Graph dag;
    dag.AddVertex(1, 1, 1);    // 0
    dag.AddVertex(1, 1, 1);    // 1
    dag.AddEdge(0, 1, 1);

    BspArchitecture<Graph> arch;
    arch.SetNumberOfProcessors(2);
    arch.SetCommunicationCosts(1);
    arch.SetSynchronisationCosts(1);

    BspInstance<Graph> instance(dag, arch);
    StaleSchedule staleSchedule(instance);

    // Setup a simple scenario: 0 -> 1
    // 0 on P0, 1 on P1

    staleSchedule.SetAssignedProcessors({0, 1});

    // Case 1: Infeasible (step difference < staleness)
    // S[0]=0, S[1]=1. Diff=1. Staleness=2. 0+2 > 1.
    staleSchedule.SetAssignedSupersteps({0, 1});
    staleSchedule.UpdateNumberOfSupersteps();

    KlActiveScheduleT activeSched;
    activeSched.Initialize(staleSchedule);

    using ThreadDataT = ThreadLocalActiveScheduleData<Graph, double>;
    ThreadDataT threadData;
    activeSched.ComputeViolations(threadData);

    BOOST_CHECK(!threadData.feasible_);

    // Case 2: Feasible (step difference == staleness)
    // S[0]=0, S[1]=2. Diff=2. Staleness=2. 0+2 <= 2.
    staleSchedule.SetAssignedSuperstep(1, 2);
    staleSchedule.UpdateNumberOfSupersteps();    // Update num steps
    activeSched.Initialize(staleSchedule);
    activeSched.ComputeViolations(threadData);

    BOOST_CHECK(threadData.feasible_);

    // Case 3: Same processor, ignore staleness
    // P[0]=0, P[1]=0. S[0]=0, S[1]=1.
    // Same proc, so just S[0] <= S[1] is required.
    staleSchedule.SetAssignedProcessor(1, 0);
    staleSchedule.SetAssignedSuperstep(1, 1);
    staleSchedule.UpdateNumberOfSupersteps();
    activeSched.Initialize(staleSchedule);
    activeSched.ComputeViolations(threadData);

    BOOST_CHECK(threadData.feasible_);
}
