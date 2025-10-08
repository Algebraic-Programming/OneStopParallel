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

#include "test_graphs.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/dag_divider/isomorphism_divider/IsomorphicSubgraphScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

#include <numeric>
#include <set>

using namespace osp;

using graph_t = computational_dag_vector_impl_def_t;
using constr_graph_t = computational_dag_vector_impl_def_t;

using group_t = typename OrbitGraphProcessor<graph_t, constr_graph_t>::Group;

// A test class to expose private methods of IsomorphicSubgraphScheduler
template <typename Graph_t, typename Constr_Graph_t>
class IsomorphicSubgraphSchedulerTester : public IsomorphicSubgraphScheduler<Graph_t, Constr_Graph_t> {
  public:
    using IsomorphicSubgraphScheduler<Graph_t, Constr_Graph_t>::IsomorphicSubgraphScheduler;

    void test_trim_subgraph_groups(std::vector<group_t>& isomorphic_groups,
                                   const unsigned min_proc_type_count) {
        this->trim_subgraph_groups(isomorphic_groups, min_proc_type_count);
    }

    void test_schedule_isomorphic_group(const BspInstance<Graph_t> &instance,
                                        const std::vector<group_t>& isomorphic_groups,
                                        const SubgraphSchedule &sub_sched,
                                        std::vector<vertex_idx_t<Graph_t>> &partition) {
        this->schedule_isomorphic_group(instance, isomorphic_groups, sub_sched, partition);
    }
};

BOOST_AUTO_TEST_SUITE(IsomorphicSubgraphSchedulerTestSuite)

BOOST_AUTO_TEST_CASE(EmptyGraphTest) {
    
    BspInstance<graph_t> instance;
    instance.getArchitecture().setNumberOfProcessors(4);

    GreedyBspScheduler<constr_graph_t> greedy_scheduler;
    IsomorphicSubgraphScheduler<graph_t, constr_graph_t> iso_scheduler(greedy_scheduler);

    auto partition = iso_scheduler.compute_partition(instance);
    BOOST_CHECK(partition.empty());
}

BOOST_AUTO_TEST_CASE(TrimSubgraphGroupsTest_NoTrim) {
    GreedyBspScheduler<constr_graph_t> greedy_scheduler;
    IsomorphicSubgraphSchedulerTester<graph_t, constr_graph_t> tester(greedy_scheduler);

    // A single group with 4 subgraphs, each with 1 node.
    std::vector<group_t> iso_groups = { group_t{ { {0}, {1}, {2}, {3} } } };

    // Group size (4) is a divisor of min_proc_type_count (8), so no trim.
    tester.test_trim_subgraph_groups(iso_groups, 8);

    BOOST_CHECK_EQUAL(iso_groups.size(), 1);
    BOOST_CHECK_EQUAL(iso_groups[0].subgraphs.size(), 4); // Still 4 subgraphs in the group
}

BOOST_AUTO_TEST_CASE(TrimSubgraphGroupsTest_WithTrim) {
    GreedyBspScheduler<constr_graph_t> greedy_scheduler;
    IsomorphicSubgraphSchedulerTester<graph_t, constr_graph_t> tester(greedy_scheduler);

    // 6 subgraphs, each with 1 node and work weight 10.
    std::vector<group_t> iso_groups = { group_t{ { {0}, {1}, {2}, {3}, {4}, {5} } } };

    // Group size (6) is not a divisor of min_proc_type_count (8).
    // gcd(6, 8) = 2.
    // merge_size = 6 / 2 = 3.
    // The 6 subgraphs should be merged into 2 new subgraphs, each containing 3 old ones.
    tester.test_trim_subgraph_groups(iso_groups, 8);

    BOOST_CHECK_EQUAL(iso_groups.size(), 1);
    BOOST_REQUIRE_EQUAL(iso_groups[0].subgraphs.size(), 2); // Group now contains 2 merged subgraphs

    // Check that the new subgraphs are correctly merged.
    BOOST_CHECK_EQUAL(iso_groups[0].subgraphs[0].size(), 3);
    BOOST_CHECK_EQUAL(iso_groups[0].subgraphs[1].size(), 3);

    const auto& final_sgs = iso_groups[0].subgraphs;
    std::set<unsigned> vertices_sg0(final_sgs[0].begin(), final_sgs[0].end());
    std::set<unsigned> vertices_sg1(final_sgs[1].begin(), final_sgs[1].end());
    std::set<unsigned> expected_sg0 = {0, 1, 2};
    std::set<unsigned> expected_sg1 = {3, 4, 5};
    BOOST_CHECK(vertices_sg0 == expected_sg0);
    BOOST_CHECK(vertices_sg1 == expected_sg1);
}

BOOST_AUTO_TEST_CASE(TrimSubgraphGroupsTest_MultipleGroups) {
    GreedyBspScheduler<constr_graph_t> greedy_scheduler;
    IsomorphicSubgraphSchedulerTester<graph_t, constr_graph_t> tester(greedy_scheduler);

    // Group 1: size 6. gcd(6, 9) = 3. merge_size = 6/3 = 2. -> 3 subgraphs of size 2.
    // Group 2: size 3. gcd(3, 9) = 3. merge_size = 3/3 = 1. -> no trim.
    // Group 3: size 5. gcd(5, 9) = 1. merge_size = 5/1 = 5. -> 1 subgraph of size 5.
    std::vector<group_t> iso_groups = {
        group_t{ { {0}, {1}, {2}, {3}, {4}, {5} } }, // Group 1
        group_t{ { {10}, {11}, {12} } },             // Group 2
        group_t{ { {20}, {21}, {22}, {23}, {24} } }  // Group 3
    };

    tester.test_trim_subgraph_groups(iso_groups, 9);

    BOOST_REQUIRE_EQUAL(iso_groups.size(), 3);

    // Check Group 1
    BOOST_REQUIRE_EQUAL(iso_groups[0].subgraphs.size(), 3);
    BOOST_CHECK_EQUAL(iso_groups[0].subgraphs[0].size(), 2);
    BOOST_CHECK_EQUAL(iso_groups[0].subgraphs[1].size(), 2);
    BOOST_CHECK_EQUAL(iso_groups[0].subgraphs[2].size(), 2);

    // Check Group 2
    BOOST_REQUIRE_EQUAL(iso_groups[1].subgraphs.size(), 3);
    BOOST_CHECK_EQUAL(iso_groups[1].subgraphs[0].size(), 1);

    // Check Group 3
    BOOST_REQUIRE_EQUAL(iso_groups[2].subgraphs.size(), 1);
    BOOST_CHECK_EQUAL(iso_groups[2].subgraphs[0].size(), 5);
}

BOOST_AUTO_TEST_CASE(ScheduleIsomorphicGroup_HeterogeneousArch) {
    // --- Setup ---
    BspInstance<graph_t> instance;
    auto& dag = instance.getComputationalDag();
    // Two isomorphic groups:
    // Group 0: {0,1}, {2,3} (type 0)
    // Group 1: {4}, {5} (type 1)
    dag.add_vertex(10, 1, 1, 0); dag.add_vertex(10, 1, 1, 0); // 0, 1
    dag.add_vertex(10, 1, 1, 0); dag.add_vertex(10, 1, 1, 0); // 2, 3
    dag.add_vertex(20, 1, 1, 1); // 4
    dag.add_vertex(20, 1, 1, 1); // 5
    dag.add_edge(0, 1); dag.add_edge(2, 3);
    dag.add_edge(1, 4); dag.add_edge(3, 5);

    // 2 procs of type 0, 2 procs of type 1
    instance.getArchitecture().setProcessorsWithTypes({0, 0, 1, 1});
    instance.setDiagonalCompatibilityMatrix(2);

    std::vector<group_t> iso_groups = {
        group_t{ { {0, 1}, {2, 3} } },
        group_t{ { {4}, {5} } }
    };

    // Mock SubgraphSchedule from EFT scheduler
    // Group 0 (2 subgraphs) gets 2 workers of type 0
    // Group 1 (2 subgraphs) gets 2 workers of type 1
    SubgraphSchedule sub_sched;
    sub_sched.node_assigned_worker_per_type.resize(2);
    sub_sched.node_assigned_worker_per_type[0] = {2, 0}; // 2xT0 for group 0
    sub_sched.node_assigned_worker_per_type[1] = {0, 2}; // 2xT1 for group 1

    std::vector<vertex_idx_t<graph_t>> partition(dag.num_vertices());

    GreedyBspScheduler<constr_graph_t> greedy_scheduler;
    IsomorphicSubgraphSchedulerTester<graph_t, constr_graph_t> tester(greedy_scheduler);

    // --- Execute ---
    tester.test_schedule_isomorphic_group(instance, iso_groups, sub_sched, partition);

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
    std::set<vertex_idx_t<graph_t>> partition_ids;
    for(const auto& p_id : partition) partition_ids.insert(p_id);
    BOOST_CHECK_EQUAL(partition_ids.size(), 4);
}

BOOST_AUTO_TEST_SUITE_END()