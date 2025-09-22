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

#define BOOST_TEST_MODULE WavefrontOrbitProcessor
#include <boost/test/unit_test.hpp>

#include "osp/dag_divider/isomorphism_divider/WavefrontOrbitProcessor.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

#include <set>
#include <numeric>

using namespace osp;

// Helper function to verify that the finalized subgraphs form a valid partition of the DAG's vertices.
template<typename Graph_t>
void check_partition(const Graph_t& dag, const std::vector<subgraph<Graph_t>>& subgraphs) {
    if (dag.num_vertices() == 0) {
        BOOST_CHECK(subgraphs.empty());
        return;
    }

    std::vector<int> vertex_counts(dag.num_vertices(), 0);
    size_t total_vertices_in_subgraphs = 0;
    for (const auto& sg : subgraphs) {
        total_vertices_in_subgraphs += sg.vertices.size();
        for (const auto& v : sg.vertices) {
            BOOST_REQUIRE_LT(v, dag.num_vertices());
            vertex_counts[v]++;
        }
    }

    BOOST_CHECK_EQUAL(total_vertices_in_subgraphs, dag.num_vertices());

    for (size_t i = 0; i < dag.num_vertices(); ++i) {
        BOOST_CHECK_EQUAL(vertex_counts[i], 1);
    }
}

BOOST_AUTO_TEST_CASE(WavefrontOrbitProcessor_SimplePipeline)
{
    using graph_t = computational_dag_vector_impl_def_t;
    graph_t dag;
    dag.add_vertex(10, 1, 1); // 0
    dag.add_vertex(10, 1, 1); // 1
    dag.add_vertex(10, 1, 1); // 2
    dag.add_vertex(10, 1, 1); // 3
    dag.add_edge(0, 1);
    dag.add_edge(1, 2);
    dag.add_edge(2, 3);

    WavefrontOrbitProcessor<graph_t> processor(2);
    processor.discover_isomorphic_groups(dag);

    auto finalized_subgraphs = processor.get_finalized_subgraphs();
    check_partition(dag, finalized_subgraphs);
    auto iso_groups = processor.get_isomorphic_groups();

    // Expect one large subgraph containing all nodes.
    BOOST_REQUIRE_EQUAL(finalized_subgraphs.size(), 1);
    BOOST_REQUIRE_EQUAL(iso_groups.size(), 1);
    BOOST_REQUIRE_EQUAL(iso_groups[0].size(), 1);

    const auto& sg = finalized_subgraphs[iso_groups[0][0]];
    BOOST_CHECK_EQUAL(sg.vertices.size(), 4);
}

BOOST_AUTO_TEST_CASE(WavefrontOrbitProcessor_ForkJoin)
{
    using graph_t = computational_dag_vector_impl_def_t;
    graph_t dag;
    dag.add_vertex(10, 1, 1); // 0
    dag.add_vertex(20, 1, 1); // 1
    dag.add_vertex(20, 1, 1); // 2
    dag.add_vertex(30, 1, 1); // 3
    dag.add_edge(0, 1);
    dag.add_edge(0, 2);
    dag.add_edge(1, 3);
    dag.add_edge(2, 3);

    WavefrontOrbitProcessor<graph_t> processor(2);
    processor.discover_isomorphic_groups(dag);

    auto finalized_subgraphs = processor.get_finalized_subgraphs();
    check_partition(dag, finalized_subgraphs);
    auto iso_groups = processor.get_isomorphic_groups();

    // The processor breaks at the fork to create a larger iso-group {{1},{2}}.
    // It then breaks at the join because the group size shrinks from 2 to 1.
    // This results in 4 finalized subgraphs, in 3 groups: {0}, {{1},{2}}, {3}.
    BOOST_REQUIRE_EQUAL(finalized_subgraphs.size(), 4);
    BOOST_REQUIRE_EQUAL(iso_groups.size(), 3);

    size_t group_of_1_count = 0;
    size_t group_of_2_count = 0;
    for (const auto& group : iso_groups) {
        if (group.size() == 1) group_of_1_count++;
        else if (group.size() == 2) group_of_2_count++;
    }
    BOOST_CHECK_EQUAL(group_of_1_count, 2); // Groups for {0} and {3}
    BOOST_CHECK_EQUAL(group_of_2_count, 1); // Group for {{1}, {2}}
}

BOOST_AUTO_TEST_CASE(WavefrontOrbitProcessor_Break)
{
    using graph_t = computational_dag_vector_impl_def_t;
    graph_t dag;
    // 4 parallel nodes, but only 3 of them have children.
    dag.add_vertex(10, 1, 1); // 0
    dag.add_vertex(10, 1, 1); // 1
    dag.add_vertex(10, 1, 1); // 2
    dag.add_vertex(10, 1, 1); // 3 (sink)
    dag.add_vertex(20, 1, 1); // 4
    dag.add_vertex(20, 1, 1); // 5
    dag.add_vertex(20, 1, 1); // 6
    dag.add_edge(0, 4);
    dag.add_edge(1, 5);
    dag.add_edge(2, 6);

    // With a threshold of 4, the group of 4 subgraphs {0},{1},{2},{3}
    // cannot be continued by the orbit of 3 children {4,5,6}, as the
    // new group size (3) would be less than the threshold. This forces a break.
    WavefrontOrbitProcessor<graph_t> processor(4);
    processor.discover_isomorphic_groups(dag);

    auto finalized_subgraphs = processor.get_finalized_subgraphs();
    check_partition(dag, finalized_subgraphs);
    auto iso_groups = processor.get_isomorphic_groups();

    // Expect two groups: one for the 4 parents, one for the 3 children.
    BOOST_REQUIRE_EQUAL(iso_groups.size(), 2);
    BOOST_REQUIRE_EQUAL(finalized_subgraphs.size(), 4);
}

BOOST_AUTO_TEST_CASE(WavefrontOrbitProcessor_ComplexMergeBreak)
{
    using graph_t = computational_dag_vector_impl_def_t;
    graph_t dag;
    dag.add_vertex(10, 1, 1); // 0
    dag.add_vertex(20, 1, 1); // 1
    dag.add_vertex(30, 1, 1); // 2
    dag.add_vertex(40, 1, 1); // 3
    dag.add_edge(0, 2);
    dag.add_edge(1, 2);
    dag.add_edge(2, 3);

    // The merge of two groups of size 1 ({0} and {1}) into a new group of size 1 ({2}) is not viable.
    // The processor should break the merge, finalizing the parents {0} and {1},
    // and creating a new subgraph for the child {2}, which then continues to {3}.
    WavefrontOrbitProcessor<graph_t> processor(2);
    processor.discover_isomorphic_groups(dag);

    auto finalized_subgraphs = processor.get_finalized_subgraphs();
    check_partition(dag, finalized_subgraphs);
    BOOST_REQUIRE_EQUAL(finalized_subgraphs.size(), 3);

    bool found_sg0 = false, found_sg1 = false, found_sg23 = false;
    for (const auto& sg : finalized_subgraphs) {
        std::set<vertex_idx_t<graph_t>> v_set(sg.vertices.begin(), sg.vertices.end());
        if (v_set == std::set<vertex_idx_t<graph_t>>{0}) found_sg0 = true;
        else if (v_set == std::set<vertex_idx_t<graph_t>>{1}) found_sg1 = true;
        else if (v_set == std::set<vertex_idx_t<graph_t>>{2, 3}) found_sg23 = true;
    }
    BOOST_CHECK(found_sg0 && found_sg1 && found_sg23);
}

BOOST_AUTO_TEST_CASE(WavefrontOrbitProcessor_MultiOrbitContinuation_NoSplit)
{
    using graph_t = computational_dag_vector_impl_def_t;
    graph_t dag;
    // Two parents, two child orbits. Each parent connects to one child from each orbit.
    dag.add_vertex(10, 1, 1); // 0 (p1)
    dag.add_vertex(10, 1, 1); // 1 (p2)
    dag.add_vertex(20, 1, 1); // 2 (c1)
    dag.add_vertex(20, 1, 1); // 3 (c2)
    dag.add_vertex(30, 1, 1); // 4 (d1)
    dag.add_vertex(30, 1, 1); // 5 (d2)
    dag.add_edge(0, 2); // p1 -> c1
    dag.add_edge(0, 4); // p1 -> d1
    dag.add_edge(1, 3); // p2 -> c2
    dag.add_edge(1, 5); // p2 -> d2

    // The parent group {p1, p2} should be continued by both child orbits {c1,c2} and {d1,d2}
    // without splitting, resulting in a single, larger isomorphic group of two subgraphs.
    WavefrontOrbitProcessor<graph_t> processor(2);
    processor.discover_isomorphic_groups(dag);

    auto finalized_subgraphs = processor.get_finalized_subgraphs();
    check_partition(dag, finalized_subgraphs);
    auto iso_groups = processor.get_isomorphic_groups();

    // Expect one final group of two large, isomorphic subgraphs.
    BOOST_REQUIRE_EQUAL(finalized_subgraphs.size(), 2);
    BOOST_REQUIRE_EQUAL(iso_groups.size(), 1);
    BOOST_REQUIRE_EQUAL(iso_groups[0].size(), 2);

    std::set<vertex_idx_t<graph_t>> sg1_v(finalized_subgraphs[0].vertices.begin(), finalized_subgraphs[0].vertices.end());
    std::set<vertex_idx_t<graph_t>> sg2_v(finalized_subgraphs[1].vertices.begin(), finalized_subgraphs[1].vertices.end());
    std::set<vertex_idx_t<graph_t>> expected_sgA_v = {0, 2, 4};
    std::set<vertex_idx_t<graph_t>> expected_sgB_v = {1, 3, 5};

    BOOST_CHECK((sg1_v == expected_sgA_v && sg2_v == expected_sgB_v) || (sg1_v == expected_sgB_v && sg2_v == expected_sgA_v));
}

BOOST_AUTO_TEST_CASE(WavefrontOrbitProcessor_ComplexMergeRejoin)
{
    using graph_t = computational_dag_vector_impl_def_t;
    graph_t dag;
    // A split followed by a join.
    // Parents {0,1}
    dag.add_vertex(10, 1, 1); // 0
    dag.add_vertex(10, 1, 1); // 1
    // Children that cause a split
    dag.add_vertex(20, 1, 1); // 2 (child of 0)
    dag.add_vertex(20, 1, 1); // 3 (child of 1)
    dag.add_vertex(30, 1, 1); // 4 (child of 1)
    // Join node
    dag.add_vertex(40, 1, 1); // 5

    dag.add_edge(0, 2);
    dag.add_edge(1, 3);
    dag.add_edge(1, 4);
    // Join node connects children from the two split branches
    dag.add_edge(2, 5);
    dag.add_edge(4, 5);

    // The split creates two families with a common ancestor. The join node {5}
    // has parents from these two different families, triggering a complex merge.
    // Because they share an ancestor, the "rejoin" logic is used: parents are
    // finalized, and the join node starts a new group under the common ancestor.
    WavefrontOrbitProcessor<graph_t> processor(2);
    processor.discover_isomorphic_groups(dag);

    auto finalized_subgraphs = processor.get_finalized_subgraphs();
    check_partition(dag, finalized_subgraphs);

    // Expect 3 finalized subgraphs: {0,2}, {1,3,4}, and {5}
    BOOST_REQUIRE_EQUAL(finalized_subgraphs.size(), 3);

    bool found_sgA = false, found_sgB = false, found_sgC = false;
    for (const auto& sg : finalized_subgraphs) {
        std::set<vertex_idx_t<graph_t>> v_set(sg.vertices.begin(), sg.vertices.end());
        if (v_set == std::set<vertex_idx_t<graph_t>>{0, 2}) found_sgA = true;
        else if (v_set == std::set<vertex_idx_t<graph_t>>{1, 3, 4}) found_sgB = true;
        else if (v_set == std::set<vertex_idx_t<graph_t>>{5}) found_sgC = true;
    }
    BOOST_CHECK(found_sgA && found_sgB && found_sgC);
}
