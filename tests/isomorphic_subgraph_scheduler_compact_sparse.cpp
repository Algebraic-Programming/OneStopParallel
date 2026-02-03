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

BOOST_AUTO_TEST_SUITE_END()
