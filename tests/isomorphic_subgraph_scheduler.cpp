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

#define BOOST_TEST_MODULE IsomorphicSubgraphScheduler
#include <boost/test/unit_test.hpp>
#include <numeric>
#include <set>

#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/dag_divider/isomorphism_divider/IsomorphicSubgraphScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "test_graphs.hpp"

using namespace osp;

using GraphT = computational_dag_vector_impl_def_t;
using ConstrGraphT = computational_dag_vector_impl_def_t;

using GroupT = typename OrbitGraphProcessor<GraphT, ConstrGraphT>::Group;

// A test class to expose private methods of IsomorphicSubgraphScheduler
template <typename GraphT, typename ConstrGraphT>
class IsomorphicSubgraphSchedulerTester : public IsomorphicSubgraphScheduler<GraphT, ConstrGraphT> {
  public:
    using IsomorphicSubgraphScheduler<GraphT, ConstrGraphT>::ConstrGraphT;

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

BOOST_AUTO_TEST_SUITE(isomorphic_subgraph_scheduler_test_suite)

BOOST_AUTO_TEST_CASE(EmptyGraphTest) {
    BspInstance<GraphT> instance;
    instance.GetArchitecture().setNumberOfProcessors(4);

    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphScheduler<GraphT, ConstrGraphT> isoScheduler(greedyScheduler);

    auto partition = isoScheduler.compute_partition(instance);
    BOOST_CHECK(partition.empty());
}

BOOST_AUTO_TEST_CASE(TrimSubgraphGroupsTestNoTrim) {
    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphSchedulerTester<GraphT, ConstrGraphT> tester(greedyScheduler);

    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();
    dag.AddVertex(1, 1, 1, 0);                                                     // 0
    dag.AddVertex(1, 1, 1, 0);                                                     // 1
    dag.AddVertex(1, 1, 1, 0);                                                     // 2
    dag.AddVertex(1, 1, 1, 0);                                                     // 3
    instance.GetArchitecture().setProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0});    // 8 processors of type 0
    instance.setDiagonalCompatibilityMatrix(1);

    // A single group with 4 subgraphs, each with 1 node.
    std::vector<GroupT> isoGroups = {GroupT{{{0}, {1}, {2}, {3}}}};

    std::vector<bool> wasTrimmed(isoGroups.size());
    // Group size (4) is a divisor of processor count for type 0 (8), so no trim.
    tester.TestTrimSubgraphGroups(isoGroups, instance, wasTrimmed);

    BOOST_REQUIRE_EQUAL(wasTrimmed.size(), 1);
    BOOST_CHECK(!wasTrimmed[0]);
    BOOST_CHECK_EQUAL(isoGroups.size(), 1);
    BOOST_CHECK_EQUAL(isoGroups[0].subgraphs.size(), 4);    // Still 4 subgraphs in the group
}

BOOST_AUTO_TEST_CASE(TrimSubgraphGroupsTestWithTrim) {
    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphSchedulerTester<GraphT, ConstrGraphT> tester(greedyScheduler);
    tester.setAllowTrimmedScheduler(false);

    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();
    dag.AddVertex(10, 1, 1, 0);                                                    // 0
    dag.AddVertex(10, 1, 1, 0);                                                    // 1
    dag.AddVertex(10, 1, 1, 0);                                                    // 2
    dag.AddVertex(10, 1, 1, 0);                                                    // 3
    dag.AddVertex(10, 1, 1, 0);                                                    // 4
    dag.AddVertex(10, 1, 1, 0);                                                    // 5
    instance.GetArchitecture().setProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0});    // 8 processors of type 0
    instance.setDiagonalCompatibilityMatrix(1);

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
    BOOST_REQUIRE_EQUAL(isoGroups[0].subgraphs.size(), 2);    // Group now contains 2 merged subgraphs

    // Check that the new subgraphs are correctly merged.
    BOOST_CHECK_EQUAL(isoGroups[0].subgraphs[0].size(), 3);
    BOOST_CHECK_EQUAL(isoGroups[0].subgraphs[1].size(), 3);

    const auto &finalSgs = isoGroups[0].subgraphs;
    std::set<unsigned> verticesSg0(finalSgs[0].begin(), finalSgs[0].end());
    std::set<unsigned> verticesSg1(finalSgs[1].begin(), finalSgs[1].end());
    std::set<unsigned> expectedSg0 = {0, 1, 2};
    std::set<unsigned> expectedSg1 = {3, 4, 5};
    BOOST_CHECK(verticesSg0 == expectedSg0);
    BOOST_CHECK(verticesSg1 == expectedSg1);
}

BOOST_AUTO_TEST_CASE(TrimSubgraphGroupsTestMultipleGroups) {
    GreedyBspScheduler<ConstrGraphT> greedyScheduler;
    IsomorphicSubgraphSchedulerTester<GraphT, ConstrGraphT> tester(greedyScheduler);
    tester.setAllowTrimmedScheduler(false);

    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();
    for (int i = 0; i < 6; ++i) {
        dag.AddVertex(1, 1, 1, 0);    // 0-5
    }
    for (int i = 0; i < 3; ++i) {
        dag.AddVertex(1, 1, 1, 0);    // 6-8, but we will use 10-12 in test
    }
    for (int i = 0; i < 2; ++i) {
        dag.AddVertex(1, 1, 1, 0);    // 9-10
    }
    for (int i = 0; i < 2; ++i) {
        dag.AddVertex(1, 1, 1, 0);    // 11-12
    }
    for (int i = 0; i < 8; ++i) {
        dag.AddVertex(1, 1, 1, 0);    // 13-20
    }
    for (int i = 0; i < 5; ++i) {
        dag.AddVertex(1, 1, 1, 0);    // 21-25
    }
    // Make sure all vertices used in iso_groups exist.
    // All are type 0.

    instance.GetArchitecture().setProcessorsWithTypes({0, 0, 0, 0, 0, 0, 0, 0, 0});    // 9 processors of type 0
    instance.setDiagonalCompatibilityMatrix(1);

    // Group 1: size 6. gcd(6, 9) = 3. merge_size = 6/3 = 2. -> 3 subgraphs of size 2.
    // Group 2: size 3. gcd(3, 9) = 3. merge_size = 3/3 = 1. -> no trim.
    // Group 3: size 5. gcd(5, 9) = 1. merge_size = 5/1 = 5. -> 1 subgraph of size 5.
    std::vector<GroupT> isoGroups = {
        GroupT{{{0}, {1}, {2}, {3}, {4}, {5}}},    // Group 1
        GroupT{{{10}, {11}, {12}}},                // Group 2
        GroupT{{{20}, {21}, {22}, {23}, {24}}}     // Group 3
    };

    std::vector<bool> wasTrimmed(isoGroups.size());
    tester.TestTrimSubgraphGroups(isoGroups, instance, wasTrimmed);

    BOOST_REQUIRE_EQUAL(isoGroups.size(), 3);
    BOOST_REQUIRE_EQUAL(wasTrimmed.size(), 3);

    BOOST_CHECK(wasTrimmed[0]);     // Group 1 should be trimmed
    BOOST_CHECK(!wasTrimmed[1]);    // Group 2 should not be trimmed
    BOOST_CHECK(wasTrimmed[2]);     // Group 3 should be trimmed
    // Check Group 1
    BOOST_REQUIRE_EQUAL(isoGroups[0].subgraphs.size(), 3);
    BOOST_CHECK_EQUAL(isoGroups[0].subgraphs[0].size(), 2);
    BOOST_CHECK_EQUAL(isoGroups[0].subgraphs[1].size(), 2);
    BOOST_CHECK_EQUAL(isoGroups[0].subgraphs[2].size(), 2);

    // Check Group 2
    BOOST_REQUIRE_EQUAL(isoGroups[1].subgraphs.size(), 3);
    BOOST_CHECK_EQUAL(isoGroups[1].subgraphs[0].size(), 1);

    // Check Group 3
    BOOST_REQUIRE_EQUAL(isoGroups[2].subgraphs.size(), 1);
    BOOST_CHECK_EQUAL(isoGroups[2].subgraphs[0].size(), 5);
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
    dag.add_edge(0, 1);
    dag.add_edge(2, 3);
    dag.add_edge(1, 4);
    dag.add_edge(3, 5);

    // 2 procs of type 0, 2 procs of type 1
    instance.GetArchitecture().setProcessorsWithTypes({0, 0, 1, 1});
    instance.setDiagonalCompatibilityMatrix(2);

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

    // Verify all partitions are unique as expected
    std::set<VertexIdxT<GraphT>> partitionIds;
    for (const auto &pId : partition) {
        partitionIds.insert(pId);
    }
    BOOST_CHECK_EQUAL(partitionIds.size(), 4);
}

BOOST_AUTO_TEST_CASE(ScheduleIsomorphicGroupShuffledIDs) {
    // --- Setup ---
    // This test ensures that the isomorphism mapping works correctly even if
    // the vertex IDs of isomorphic subgraphs are not in the same relative order.
    BspInstance<GraphT> instance;
    auto &dag = instance.GetComputationalDag();

    // Group 0, Subgraph 1: 0 -> 1
    dag.AddVertex(10, 1, 1, 0);    // 0
    dag.AddVertex(20, 1, 1, 0);    // 1
    dag.add_edge(0, 1);

    // Group 0, Subgraph 2 (isomorphic to 1, but with shuffled IDs): 3 -> 2
    dag.AddVertex(20, 1, 1, 0);    // 2 (work 20, corresponds to node 1)
    dag.AddVertex(10, 1, 1, 0);    // 3 (work 10, corresponds to node 0)
    dag.add_edge(3, 2);

    // Architecture: 2 processors, so each subgraph gets its own partition space.
    instance.GetArchitecture().setProcessorsWithTypes({0, 0});
    instance.setDiagonalCompatibilityMatrix(1);

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
    // The representative subgraph is {0, 1}. The greedy scheduler will likely put
    // both nodes on the same processor, creating a single partition for them.
    // Let's call this partition P0.
    // The second subgraph {2, 3} is isomorphic. Node 3 corresponds to node 0,
    // and node 2 corresponds to node 1. They should also be placed in a single
    // partition together, let's call it P1.
    // We expect P0 != P1.

    // Check Subgraph 1 partitioning
    BOOST_CHECK_EQUAL(partition[0], partition[1]);

    // Check Subgraph 2 partitioning
    BOOST_CHECK_EQUAL(partition[2], partition[3]);

    // Check that the two subgraphs are in different partitions
    BOOST_CHECK_NE(partition[0], partition[2]);
}

// BOOST_AUTO_TEST_CASE(ScheduleIsomorphicGroup_ComplexShuffled) {
//     // --- Setup ---
//     // This test uses a more complex structure (fork-join) with shuffled IDs
//     // to ensure the pattern replication is robust.
//     BspInstance<graph_t> instance;
//     auto& dag = instance.GetComputationalDag();

//     // Group 0, Subgraph 1: 0 -> {1,2} -> 3
//     dag.AddVertex(10, 1, 1, 0); // 0 (source)
//     dag.AddVertex(20, 1, 1, 0); // 1 (middle)
//     dag.AddVertex(20, 1, 1, 0); // 2 (middle)
//     dag.AddVertex(30, 1, 1, 0); // 3 (sink)
//     dag.add_edge(0, 1);
//     dag.add_edge(0, 2);
//     dag.add_edge(1, 3);
//     dag.add_edge(2, 3);

//     // Group 0, Subgraph 2 (isomorphic, but with shuffled IDs and different topology)
//     // Structure: 7 -> {5,4} -> 6
//     dag.AddVertex(20, 1, 1, 0); // 4 (middle, corresponds to node 2)
//     dag.AddVertex(20, 1, 1, 0); // 5 (middle, corresponds to node 1)
//     dag.AddVertex(30, 1, 1, 0); // 6 (sink, corresponds to node 3)
//     dag.AddVertex(10, 1, 1, 0); // 7 (source, corresponds to node 0)
//     dag.add_edge(7, 4);
//     dag.add_edge(7, 5);
//     dag.add_edge(4, 6);
//     dag.add_edge(5, 6);

//     // Architecture: 4 processors, so each subgraph gets its own partition space.
//     instance.GetArchitecture().setProcessorsWithTypes({0, 0, 0, 0});
//     instance.setDiagonalCompatibilityMatrix(1);

//     // Manually define the isomorphic groups.
//     std::vector<group_t> iso_groups = {
//         group_t{ { {0, 1, 2, 3}, {4, 5, 6, 7} } }
//     };

//     // Mock SubgraphSchedule: The single group gets all 4 processors.
//     SubgraphSchedule sub_sched;
//     sub_sched.node_assigned_worker_per_type.resize(1);
//     sub_sched.node_assigned_worker_per_type[0] = {4};

//     std::vector<VertexIdxT<graph_t>> partition(dag.NumVertices());

//     GreedyBspScheduler<constr_graph_t> greedy_scheduler;
//     IsomorphicSubgraphSchedulerTester<graph_t, constr_graph_t> tester(greedy_scheduler);

//     // --- Execute ---
//     tester.test_schedule_isomorphic_group(instance, iso_groups, sub_sched, partition);

//     // --- Assert ---
//     // The representative is {0,1,2,3}. The greedy scheduler on 4 procs will likely
//     // create 3 partitions: P_A for {0}, P_B for {1,2}, P_C for {3}.
//     // The second subgraph {4,5,6,7} is isomorphic.
//     // Node 7 corresponds to 0 (source).
//     // Nodes {4,5} correspond to {1,2} (middle).
//     // Node 6 corresponds to 3 (sink).
//     // We expect the same partitioning pattern, but with an offset.
//     // P_D for {7}, P_E for {4,5}, P_F for {6}.
//     // All partitions P_A..F should be distinct.

//     // Check partitioning within subgraphs based on structural roles
//     BOOST_CHECK_EQUAL(partition[1], partition[2]); // Middle nodes together
//     BOOST_CHECK_NE(partition[0], partition[1]);    // Source separate from middle
//     BOOST_CHECK_NE(partition[1], partition[3]);    // Middle separate from sink

//     BOOST_CHECK_EQUAL(partition[4], partition[5]); // Corresponding middle nodes together
//     BOOST_CHECK_NE(partition[7], partition[4]);    // Corresponding source separate
//     BOOST_CHECK_NE(partition[4], partition[6]);    // Corresponding middle separate

//     // Check that the partitions for corresponding nodes are different across subgraphs
//     BOOST_CHECK_NE(partition[0], partition[7]); // Sources
//     BOOST_CHECK_NE(partition[1], partition[5]); // Middle nodes
//     BOOST_CHECK_NE(partition[2], partition[4]); // Middle nodes
//     BOOST_CHECK_NE(partition[3], partition[6]); // Sinks
// }

BOOST_AUTO_TEST_SUITE_END()
