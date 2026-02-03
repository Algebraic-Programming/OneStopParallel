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

#define BOOST_TEST_MODULE IsomorphicSubgraphSchedulerCompactSparse
#include <boost/test/unit_test.hpp>
#include <numeric>
#include <set>

#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/dag_divider/isomorphism_divider/IsomorphicSubgraphScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

using GraphT = ComputationalDagVectorImplDefUnsignedT;
// Using the specialization parameters for CompactSparseGraph
// <keepVertexOrder, useWorkWeights, useCommWeights, useMemWeights, useVertTypes, VertT, EdgeT, WorkWeightType, CommWeightType,
// MemWeightType, VertexTypeTemplateType>
using ConstrGraphT
    = CompactSparseGraph<true, true, true, true, true, std::size_t, unsigned, unsigned, unsigned, unsigned, unsigned>;

using GroupT = typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group;

// A test class to expose private methods of IsomorphicSubgraphScheduler
template <typename GraphT, typename ConstrGraphT>
class IsomorphicSubgraphSchedulerTester : public IsomorphicSubgraphScheduler<GraphT, ConstrGraphT> {
  public:
    using IsomorphicSubgraphScheduler<GraphT, ConstrGraphT>::IsomorphicSubgraphScheduler;

    void TestTrimSubgraphGroups(std::vector<GroupT> &isomorphicGroups,
                                const BspInstance<GraphT> &instance,
                                std::vector<bool> &wasTrimmed) {
        this->TrimSubgraphGroups(isomorphicGroups, instance, wasTrimmed);
    }

    void TestScheduleIsomorphicGroup(const BspInstance<GraphT> &instance,
                                     const std::vector<GroupT> &isomorphicGroups,
                                     const SubgraphSchedule &subSched,
                                     std::vector<VertexIdxT<GraphT>> &partition) {
        this->ScheduleIsomorphicGroup(instance, isomorphicGroups, subSched, partition);
    }
};

BOOST_AUTO_TEST_SUITE(isomorphic_subgraph_scheduler_compact_sparse_test_suite)

BOOST_AUTO_TEST_CASE(EmptyGraphTest) {
    BspInstance<GraphT> instance;
    instance.GetArchitecture().SetNumberOfProcessors(4);

    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphScheduler<GraphT, ConstrGraphT> isoScheduler(greedyScheduler);

    auto partition = isoScheduler.ComputePartition(instance);
    BOOST_CHECK(partition.empty());
}

BOOST_AUTO_TEST_CASE(TrimSubgraphGroupsTestNoTrim) {
    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphSchedulerTester<GraphT, ConstrGraphT> tester(greedyScheduler);

    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();
    dag.AddVertex(1, 1, 1, 0);                                                      // 0
    dag.AddVertex(1, 1, 1, 0);                                                      // 1
    dag.AddVertex(1, 1, 1, 0);                                                      // 2
    dag.AddVertex(1, 1, 1, 0);                                                      // 3
    instance.GetArchitecture().SetProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0});    // 8 processors of type 0
    instance.SetDiagonalCompatibilityMatrix(1);

    // A single group with 4 subgraphs, each with 1 node.
    std::vector<GroupT> isoGroups = {GroupT{{{0}, {1}, {2}, {3}}}};

    std::vector<bool> wasTrimmed(isoGroups.size());
    // Group size (4) is a divisor of processor count for type 0 (8), so no trim.
    tester.TestTrimSubgraphGroups(isoGroups, instance, wasTrimmed);

    BOOST_REQUIRE_EQUAL(wasTrimmed.size(), 1);
    BOOST_CHECK(!wasTrimmed[0]);
    BOOST_CHECK_EQUAL(isoGroups.size(), 1);
    BOOST_CHECK_EQUAL(isoGroups[0].subgraphs_.size(), 4);    // Still 4 subgraphs in the group
}

BOOST_AUTO_TEST_CASE(TrimSubgraphGroupsTestWithTrim) {
    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphSchedulerTester<GraphT, ConstrGraphT> tester(greedyScheduler);
    tester.SetAllowTrimmedScheduler(false);

    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();
    dag.AddVertex(10, 1, 1, 0);                                                     // 0
    dag.AddVertex(10, 1, 1, 0);                                                     // 1
    dag.AddVertex(10, 1, 1, 0);                                                     // 2
    dag.AddVertex(10, 1, 1, 0);                                                     // 3
    dag.AddVertex(10, 1, 1, 0);                                                     // 4
    dag.AddVertex(10, 1, 1, 0);                                                     // 5
    instance.GetArchitecture().SetProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0});    // 8 processors of type 0
    instance.SetDiagonalCompatibilityMatrix(1);

    // 6 subgraphs, each with 1 node and work weight 10.
    std::vector<GroupT> isoGroups = {GroupT{{{0}, {1}, {2}, {3}, {4}, {5}}}};

    std::vector<bool> wasTrimmed(isoGroups.size());
    // Group size (6) is not a divisor of processor count for type 0 (8).
    // gcd(6, 8) = 2.
    // merge_size = 6 / 2 = 3.
    // The 6 subgraphs should be merged into 2 new subgraphs, each containing 3 old ones.
    tester.TestTrimSubgraphGroups(isoGroups, instance, wasTrimmed);

    BOOST_REQUIRE_EQUAL(wasTrimmed.size(), 1);
    BOOST_CHECK(wasTrimmed[0]);
    BOOST_CHECK_EQUAL(isoGroups.size(), 1);
    BOOST_REQUIRE_EQUAL(isoGroups[0].subgraphs_.size(), 2);    // Group now contains 2 merged subgraphs

    // Check that the new subgraphs are correctly merged.
    BOOST_CHECK_EQUAL(isoGroups[0].subgraphs_[0].size(), 3);
    BOOST_CHECK_EQUAL(isoGroups[0].subgraphs_[1].size(), 3);

    const auto &finalSgs = isoGroups[0].subgraphs_;
    std::set<unsigned> verticesSg0(finalSgs[0].begin(), finalSgs[0].end());
    std::set<unsigned> verticesSg1(finalSgs[1].begin(), finalSgs[1].end());
    std::set<unsigned> expectedSg0 = {0, 1, 2};
    std::set<unsigned> expectedSg1 = {3, 4, 5};
    BOOST_CHECK(verticesSg0 == expectedSg0);
    BOOST_CHECK(verticesSg1 == expectedSg1);
}

BOOST_AUTO_TEST_CASE(ScheduleIsomorphicGroupHeterogeneousArch) {
    // --- Setup ---
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();
    // Two isomorphic groups:
    // Group 0: {0,1}, {2,3} (type 0)
    // Group 1: {4}, {5} (type 1)
    dag.AddVertex(10, 1, 1, 0);
    dag.AddVertex(10, 1, 1, 0);    // 0, 1
    dag.AddVertex(10, 1, 1, 0);
    dag.AddVertex(10, 1, 1, 0);    // 2, 3
    dag.AddVertex(20, 1, 1, 1);    // 4
    dag.AddVertex(20, 1, 1, 1);    // 5
    dag.AddEdge(0, 1);
    dag.AddEdge(2, 3);
    dag.AddEdge(1, 4);
    dag.AddEdge(3, 5);

    // 2 procs of type 0, 2 procs of type 1
    instance.GetArchitecture().SetProcessorsWithTypes({0, 0, 1, 1});
    instance.SetDiagonalCompatibilityMatrix(2);

    std::vector<GroupT> isoGroups = {GroupT{{{0, 1}, {2, 3}}}, GroupT{{{4}, {5}}}};

    // Mock SubgraphSchedule from EFT scheduler
    // Group 0 (2 subgraphs) gets 2 workers of type 0
    // Group 1 (2 subgraphs) gets 2 workers of type 1
    SubgraphSchedule subSched;
    subSched.nodeAssignedWorkerPerType_.resize(2);
    subSched.nodeAssignedWorkerPerType_[0] = {2, 0};    // 2xT0 for group 0
    subSched.nodeAssignedWorkerPerType_[1] = {0, 2};    // 2xT1 for group 1
    subSched.wasTrimmed_ = {false, false};              // No trimming occurred

    std::vector<VertexIdxT<GraphT>> partition(dag.NumVertices());

    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphSchedulerTester<GraphT, ConstrGraphT> tester(greedyScheduler);

    // --- Execute ---
    tester.TestScheduleIsomorphicGroup(instance, isoGroups, subSched, partition);

    // --- Assert ---
    // Group 0 has 2 subgraphs, scheduled on 2 processors.
    // The internal scheduler for the representative {0,1} will likely put both on one processor.
    // So, {0,1} will be in one partition, and {2,3} will be in another.
    BOOST_CHECK_EQUAL(partition[0], partition[1]);
    BOOST_CHECK_EQUAL(partition[2], partition[3]);
    BOOST_CHECK_NE(partition[0], partition[2]);

    // Group 1 has 2 subgraphs, scheduled on 2 processors.
    // Each subgraph {4} and {5} gets its own partition.
    BOOST_CHECK_NE(partition[4], partition[5]);

    // Check that partitions for different groups are distinct
    BOOST_CHECK_NE(partition[0], partition[4]);
    BOOST_CHECK_NE(partition[0], partition[5]);
    BOOST_CHECK_NE(partition[2], partition[4]);
    BOOST_CHECK_NE(partition[2], partition[5]);
}

BOOST_AUTO_TEST_CASE(ScheduleIsomorphicGroupShuffledIDs) {
    // --- Setup ---
    // This test ensures that the isomorphism mapping works correctly even if
    // the vertex IDs of isomorphic subgraphs are not in the same relative order.
    // CompactSparseGraph's CreateInducedSubgraphMap reorders vertices topologically,
    // so this is a critical test for the recent fix.
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    // Group 0, Subgraph 1: 0 -> 1
    dag.AddVertex(10, 1, 1, 0);    // 0
    dag.AddVertex(20, 1, 1, 0);    // 1
    dag.AddEdge(0, 1);

    // Group 0, Subgraph 2 (isomorphic to 1, but with shuffled IDs): 3 -> 2
    dag.AddVertex(20, 1, 1, 0);    // 2 (work 20, corresponds to node 1)
    dag.AddVertex(10, 1, 1, 0);    // 3 (work 10, corresponds to node 0)
    dag.AddEdge(3, 2);

    // Architecture: 2 processors, so each subgraph gets its own partition space.
    instance.GetArchitecture().SetProcessorsWithTypes({0, 0});
    instance.SetDiagonalCompatibilityMatrix(1);

    // Manually define the isomorphic groups.
    // Subgraph 1 vertices: {0, 1}
    // Subgraph 2 vertices: {2, 3}
    std::vector<GroupT> isoGroups = {GroupT{{{0, 1}, {2, 3}}}};

    // Mock SubgraphSchedule: The single group gets all 2 processors.
    SubgraphSchedule subSched;
    subSched.nodeAssignedWorkerPerType_.resize(1);
    subSched.nodeAssignedWorkerPerType_[0] = {2};
    subSched.wasTrimmed_ = {false};    // No trimming occurred

    std::vector<VertexIdxT<GraphT>> partition(dag.NumVertices());

    // Use a simple greedy scheduler for the sub-problems.
    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphSchedulerTester<GraphT, ConstrGraphT> tester(greedyScheduler);

    // --- Execute ---
    tester.TestScheduleIsomorphicGroup(instance, isoGroups, subSched, partition);

    // --- Assert ---
    // The representative subgraph is {0, 1}. The greedy scheduler on 4 procs will likely
    // put both nodes on the same processor, creating a single partition for them.
    // The second subgraph {2, 3} is isomorphic. Node 3 corresponds to node 0,
    // and node 2 corresponds to node 1. They should also be placed in a single
    // partition together.

    // Check Subgraph 1 partitioning
    BOOST_CHECK_EQUAL(partition[0], partition[1]);

    // Check Subgraph 2 partitioning
    BOOST_CHECK_EQUAL(partition[2], partition[3]);

    // Check that the two subgraphs are in different partitions
    BOOST_CHECK_NE(partition[0], partition[2]);
}

BOOST_AUTO_TEST_CASE(ScheduleIsomorphicGroupDiamondShuffledIDs) {
    // --- Setup ---
    // This test ensures that the isomorphism mapping works correctly for diamond-shaped
    // subgraphs where vertex IDs are not in the same relative order.
    // Diamond pattern: source -> {mid1, mid2} -> sink
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    // Subgraph 1 (vertices 0-3): Standard diamond pattern
    //     0
    //    /
    //   1   2
    //    \ /
    //     3
    dag.AddVertex(10, 1, 1, 0);    // 0: source (work=10)
    dag.AddVertex(20, 1, 1, 0);    // 1: mid1 (work=20)
    dag.AddVertex(20, 1, 1, 0);    // 2: mid2 (work=20)
    dag.AddVertex(30, 1, 1, 0);    // 3: sink (work=30)
    dag.AddEdge(0, 1);
    dag.AddEdge(0, 2);
    dag.AddEdge(1, 3);
    dag.AddEdge(2, 3);

    // Subgraph 2 (vertices 4-7): Shuffled diamond pattern
    // Vertex order: sink=4, mid2=5, source=6, mid1=7
    //     6 (source)
    //    /
    //   7   5 (mid1, mid2)
    //    \ /
    //     4 (sink)
    dag.AddVertex(30, 1, 1, 0);    // 4: sink (work=30, corresponds to node 3)
    dag.AddVertex(20, 1, 1, 0);    // 5: mid2 (work=20, corresponds to node 2)
    dag.AddVertex(10, 1, 1, 0);    // 6: source (work=10, corresponds to node 0)
    dag.AddVertex(20, 1, 1, 0);    // 7: mid1 (work=20, corresponds to node 1)
    dag.AddEdge(6, 7);
    dag.AddEdge(6, 5);
    dag.AddEdge(7, 4);
    dag.AddEdge(5, 4);

    instance.GetArchitecture().SetProcessorsWithTypes({0, 0});
    instance.SetDiagonalCompatibilityMatrix(1);

    std::vector<GroupT> isoGroups = {GroupT{{{0, 1, 2, 3}, {4, 5, 6, 7}}}};

    SubgraphSchedule subSched;
    subSched.nodeAssignedWorkerPerType_.resize(1);
    subSched.nodeAssignedWorkerPerType_[0] = {2};
    subSched.wasTrimmed_ = {false};

    std::vector<VertexIdxT<GraphT>> partition(dag.NumVertices());

    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphSchedulerTester<GraphT, ConstrGraphT> tester(greedyScheduler);

    // --- Execute ---
    tester.TestScheduleIsomorphicGroup(instance, isoGroups, subSched, partition);

    // --- Assert ---
    // Subgraph 1: All nodes should be in the same partition (single-processor assignment likely)
    BOOST_CHECK_EQUAL(partition[0], partition[1]);
    BOOST_CHECK_EQUAL(partition[0], partition[2]);
    BOOST_CHECK_EQUAL(partition[0], partition[3]);

    // Subgraph 2: All nodes should be in the same partition
    BOOST_CHECK_EQUAL(partition[4], partition[5]);
    BOOST_CHECK_EQUAL(partition[4], partition[6]);
    BOOST_CHECK_EQUAL(partition[4], partition[7]);

    // The two subgraphs should be in different partitions
    BOOST_CHECK_NE(partition[0], partition[4]);
}

BOOST_AUTO_TEST_CASE(ScheduleIsomorphicGroupMultiLevelShuffledIDs) {
    // --- Setup ---
    // Three-level graph: src -> {mid1, mid2} -> sink
    // Tests that the mapping handles multiple nodes at the middle level correctly
    // when vertex IDs are interleaved between subgraphs.
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    // Subgraph 1: vertices in order 0 (src) -> {1, 2} (mid) -> 3 (sink)
    dag.AddVertex(5, 1, 1, 0);     // 0: source
    dag.AddVertex(10, 1, 1, 0);    // 1: mid1
    dag.AddVertex(10, 1, 1, 0);    // 2: mid2
    dag.AddVertex(15, 1, 1, 0);    // 3: sink

    // Subgraph 2: vertices interleaved with different relative positions
    // 4 (sink), 5 (mid2), 6 (mid1), 7 (src)
    dag.AddVertex(15, 1, 1, 0);    // 4: sink
    dag.AddVertex(10, 1, 1, 0);    // 5: mid2
    dag.AddVertex(10, 1, 1, 0);    // 6: mid1
    dag.AddVertex(5, 1, 1, 0);     // 7: source

    // Subgraph 3: Another permutation
    // 8 (mid1), 9 (src), 10 (sink), 11 (mid2)
    dag.AddVertex(10, 1, 1, 0);    // 8: mid1
    dag.AddVertex(5, 1, 1, 0);     // 9: source
    dag.AddVertex(15, 1, 1, 0);    // 10: sink
    dag.AddVertex(10, 1, 1, 0);    // 11: mid2

    // Edges for Subgraph 1
    dag.AddEdge(0, 1);
    dag.AddEdge(0, 2);
    dag.AddEdge(1, 3);
    dag.AddEdge(2, 3);

    // Edges for Subgraph 2 (7 is source, {5,6} are mid, 4 is sink)
    dag.AddEdge(7, 5);
    dag.AddEdge(7, 6);
    dag.AddEdge(5, 4);
    dag.AddEdge(6, 4);

    // Edges for Subgraph 3 (9 is source, {8,11} are mid, 10 is sink)
    dag.AddEdge(9, 8);
    dag.AddEdge(9, 11);
    dag.AddEdge(8, 10);
    dag.AddEdge(11, 10);

    instance.GetArchitecture().SetProcessorsWithTypes({0, 0, 0});
    instance.SetDiagonalCompatibilityMatrix(1);

    std::vector<GroupT> isoGroups = {GroupT{{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}}}};

    SubgraphSchedule subSched;
    subSched.nodeAssignedWorkerPerType_.resize(1);
    subSched.nodeAssignedWorkerPerType_[0] = {3};    // 3 processors for 3 subgraphs
    subSched.wasTrimmed_ = {false};

    std::vector<VertexIdxT<GraphT>> partition(dag.NumVertices());

    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphSchedulerTester<GraphT, ConstrGraphT> tester(greedyScheduler);

    // --- Execute ---
    tester.TestScheduleIsomorphicGroup(instance, isoGroups, subSched, partition);

    // --- Assert ---
    // Each subgraph should have all its nodes in the same partition
    // Subgraph 1
    BOOST_CHECK_EQUAL(partition[0], partition[1]);
    BOOST_CHECK_EQUAL(partition[0], partition[2]);
    BOOST_CHECK_EQUAL(partition[0], partition[3]);

    // Subgraph 2
    BOOST_CHECK_EQUAL(partition[4], partition[5]);
    BOOST_CHECK_EQUAL(partition[4], partition[6]);
    BOOST_CHECK_EQUAL(partition[4], partition[7]);

    // Subgraph 3
    BOOST_CHECK_EQUAL(partition[8], partition[9]);
    BOOST_CHECK_EQUAL(partition[8], partition[10]);
    BOOST_CHECK_EQUAL(partition[8], partition[11]);

    // All three subgraphs should be in different partitions
    BOOST_CHECK_NE(partition[0], partition[4]);
    BOOST_CHECK_NE(partition[0], partition[8]);
    BOOST_CHECK_NE(partition[4], partition[8]);
}

BOOST_AUTO_TEST_CASE(ScheduleIsomorphicGroupShuffledForkJoinPattern) {
    // --- Setup ---
    // This test ensures the mapping handles a larger fork-join pattern
    // where the second subgraph has completely reversed vertex ordering.
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    // Subgraph 1: Fork-join with 3 parallel branches
    // 0 -> {1, 2, 3} -> 4
    dag.AddVertex(10, 1, 1, 0);    // 0: source
    dag.AddVertex(25, 1, 1, 0);    // 1: branch1
    dag.AddVertex(25, 1, 1, 0);    // 2: branch2
    dag.AddVertex(25, 1, 1, 0);    // 3: branch3
    dag.AddVertex(40, 1, 1, 0);    // 4: sink
    dag.AddEdge(0, 1);
    dag.AddEdge(0, 2);
    dag.AddEdge(0, 3);
    dag.AddEdge(1, 4);
    dag.AddEdge(2, 4);
    dag.AddEdge(3, 4);

    // Subgraph 2: Same pattern but vertex IDs are completely reversed
    // 9 (source) -> {8, 7, 6} (branches) -> 5 (sink)
    dag.AddVertex(40, 1, 1, 0);    // 5: sink (corresponds to 4)
    dag.AddVertex(25, 1, 1, 0);    // 6: branch3 (corresponds to 3)
    dag.AddVertex(25, 1, 1, 0);    // 7: branch2 (corresponds to 2)
    dag.AddVertex(25, 1, 1, 0);    // 8: branch1 (corresponds to 1)
    dag.AddVertex(10, 1, 1, 0);    // 9: source (corresponds to 0)
    dag.AddEdge(9, 8);
    dag.AddEdge(9, 7);
    dag.AddEdge(9, 6);
    dag.AddEdge(8, 5);
    dag.AddEdge(7, 5);
    dag.AddEdge(6, 5);

    instance.GetArchitecture().SetProcessorsWithTypes({0, 0});
    instance.SetDiagonalCompatibilityMatrix(1);

    std::vector<GroupT> isoGroups = {GroupT{{{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}}};

    SubgraphSchedule subSched;
    subSched.nodeAssignedWorkerPerType_.resize(1);
    subSched.nodeAssignedWorkerPerType_[0] = {2};
    subSched.wasTrimmed_ = {false};

    std::vector<VertexIdxT<GraphT>> partition(dag.NumVertices());

    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphSchedulerTester<GraphT, ConstrGraphT> tester(greedyScheduler);

    // --- Execute ---
    tester.TestScheduleIsomorphicGroup(instance, isoGroups, subSched, partition);

    // --- Assert ---
    // Subgraph 1: All in same partition
    BOOST_CHECK_EQUAL(partition[0], partition[1]);
    BOOST_CHECK_EQUAL(partition[0], partition[2]);
    BOOST_CHECK_EQUAL(partition[0], partition[3]);
    BOOST_CHECK_EQUAL(partition[0], partition[4]);

    // Subgraph 2: All in same partition
    BOOST_CHECK_EQUAL(partition[5], partition[6]);
    BOOST_CHECK_EQUAL(partition[5], partition[7]);
    BOOST_CHECK_EQUAL(partition[5], partition[8]);
    BOOST_CHECK_EQUAL(partition[5], partition[9]);

    // The two subgraphs should be in different partitions
    BOOST_CHECK_NE(partition[0], partition[5]);
}

BOOST_AUTO_TEST_SUITE_END()
