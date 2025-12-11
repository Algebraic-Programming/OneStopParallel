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

#define BOOST_TEST_MODULE OrbitGraphProcessor
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <iostream>
#include <set>

#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/auxiliary/io/dot_graph_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/dag_divider/isomorphism_divider/MerkleHashComputer.hpp"
#include "osp/dag_divider/isomorphism_divider/OrbitGraphProcessor.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "test_graphs.hpp"
#include "test_utils.hpp"

using namespace osp;
using graph_t = computational_dag_vector_impl_def_t;

template <typename Graph_t>
void check_partitioning(const Graph_t &dag, const OrbitGraphProcessor<Graph_t, Graph_t> &processor) {
    const auto &final_coarse_graph = processor.get_final_coarse_graph();
    const auto &final_groups = processor.get_final_groups();

    // Check that the final coarse graph is acyclic
    BOOST_CHECK(is_acyclic(final_coarse_graph));

    // Check that the final groups form a valid partition of the original DAG's vertices
    std::vector<int> vertex_counts(dag.num_vertices(), 0);
    size_t total_vertices_in_groups = 0;
    for (const auto &group : final_groups) {
        for (const auto &subgraph : group.subgraphs) {
            total_vertices_in_groups += subgraph.size();
            for (const auto &vertex : subgraph) {
                BOOST_REQUIRE_LT(vertex, dag.num_vertices());
                vertex_counts[vertex]++;
            }
        }
    }
    BOOST_CHECK_EQUAL(total_vertices_in_groups, dag.num_vertices());
    for (size_t i = 0; i < dag.num_vertices(); ++i) { BOOST_CHECK_EQUAL(vertex_counts[i], 1); }
}

// BOOST_AUTO_TEST_CASE(OrbitGraphProcessor_SmokeTest) {
//     // The test reads a file, but the path is absolute, so we don't need the project root here.
//     graph_t dag;
//     file_reader::readComputationalDagDotFormat("", dag);

//     OrbitGraphProcessor<graph_t, graph_t> processor(2); // Using a symmetry threshold of 2
//     MerkleHashComputer<graph_t, bwd_merkle_node_hash_func<graph_t>, true> hasher(dag, dag);
//     processor.discover_isomorphic_groups(dag, hasher);

//     const auto& coarse_graph = processor.get_coarse_graph();
//     const auto& final_coarse_graph = processor.get_final_coarse_graph();
//     const auto& final_groups = processor.get_final_groups();
//     const auto& final_contraction_map = processor.get_final_contraction_map();

//     DotFileWriter writer;
//     // Color by initial orbits
//     writer.write_colored_graph("orbit_graph_orbits_colored.dot", dag, processor.get_contraction_map());
//     writer.write_graph("orbit_graph_coarse_graph.dot", coarse_graph);

//     // Color by final merged groups
//     writer.write_colored_graph("orbit_graph_groups_colored.dot", dag, final_contraction_map);

//     // Color by final subgraphs (each subgraph gets a unique color)
//     std::vector<unsigned> subgraph_colors(dag.num_vertices());
//     unsigned current_subgraph_color = 0;
//     for (const auto& group : final_groups) {
//         for (const auto& subgraph : group.subgraphs) {
//             for (const auto& vertex : subgraph) {
//                 subgraph_colors[vertex] = current_subgraph_color;
//             }
//             current_subgraph_color++;
//         }
//     }
//     writer.write_colored_graph("orbit_graph_subgraphs_colored.dot", dag, subgraph_colors);
//     writer.write_graph("orbit_graph_final_coarse_graph.dot", final_coarse_graph);

//     BOOST_CHECK_GT(coarse_graph.num_vertices(), 0);
//     BOOST_CHECK_LT(coarse_graph.num_vertices(), dag.num_vertices());
//     BOOST_CHECK_GT(final_coarse_graph.num_vertices(), 0);
//     BOOST_CHECK_LE(final_coarse_graph.num_vertices(), coarse_graph.num_vertices());

//     check_partitioning(dag, processor);
// }

BOOST_AUTO_TEST_CASE(OrbitGraphProcessor_SimpleMerge) {
    graph_t dag;
    // Two parallel pipelines that are structurally identical
    // 0 -> 1
    // 2 -> 3
    dag.add_vertex(10, 1, 1);    // 0
    dag.add_vertex(10, 1, 1);    // 1
    dag.add_vertex(10, 1, 1);    // 2
    dag.add_vertex(10, 1, 1);    // 3
    dag.add_edge(0, 1);
    dag.add_edge(2, 3);

    // Initial orbits: {0, 2} and {1, 3}. Coarse graph: 0 -> 1
    // With threshold 2, these should be merged.
    OrbitGraphProcessor<graph_t, graph_t> processor;
    MerkleHashComputer<graph_t, bwd_merkle_node_hash_func<graph_t>, true> hasher(dag, dag);
    processor.discover_isomorphic_groups(dag, hasher);

    const auto &final_coarse_graph = processor.get_final_coarse_graph();
    const auto &final_groups = processor.get_final_groups();

    // Expect a single node in the final coarse graph
    BOOST_CHECK_EQUAL(final_coarse_graph.num_vertices(), 1);
    BOOST_CHECK_EQUAL(final_groups.size(), 1);

    // The single group should contain two subgraphs: {0,1} and {2,3}
    BOOST_REQUIRE_EQUAL(final_groups[0].subgraphs.size(), 2);
    std::set<int> sg1(final_groups[0].subgraphs[0].begin(), final_groups[0].subgraphs[0].end());
    std::set<int> sg2(final_groups[0].subgraphs[1].begin(), final_groups[0].subgraphs[1].end());
    std::set<int> expected_sgA = {0, 1};
    std::set<int> expected_sgB = {2, 3};

    BOOST_CHECK((sg1 == expected_sgA && sg2 == expected_sgB) || (sg1 == expected_sgB && sg2 == expected_sgA));

    check_partitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessor_ForkJoinNoMerge) {
    graph_t dag;
    // 0 -> {1, 2} -> 3. Nodes 1 and 2 are in the same orbit.
    dag.add_vertex(10, 1, 1);    // 0
    dag.add_vertex(20, 1, 1);    // 1
    dag.add_vertex(20, 1, 1);    // 2
    dag.add_vertex(30, 1, 1);    // 3
    dag.add_edge(0, 1);
    dag.add_edge(0, 2);
    dag.add_edge(1, 3);
    dag.add_edge(2, 3);

    // Initial orbits: {0}, {1,2}, {3}. Coarse graph: 0 -> 1 -> 2
    // Merging 0 and 1 would result in a group of size 1 ({0,1,2}), which is not viable (threshold 2).
    // Merging 1 and 2 would also result in a group of size 1 ({1,2,3}), not viable.
    OrbitGraphProcessor<graph_t, graph_t> processor;
    MerkleHashComputer<graph_t, bwd_merkle_node_hash_func<graph_t>, true> hasher(dag, dag);
    processor.discover_isomorphic_groups(dag, hasher);

    const auto &final_coarse_graph = processor.get_final_coarse_graph();
    const auto &final_groups = processor.get_final_groups();

    // Expect no merges, so final graph is same as initial coarse graph.
    BOOST_CHECK_EQUAL(final_coarse_graph.num_vertices(), 3);
    BOOST_CHECK_EQUAL(final_groups.size(), 3);

    // Check group structures
    // Group 0: {{0}}
    // Group 1: {{1}, {2}}
    // Group 2: {{3}}
    size_t group_of_1_count = 0;
    size_t group_of_2_count = 0;
    for (const auto &group : final_groups) {
        if (group.subgraphs.size() == 1) { group_of_1_count++; }
        if (group.subgraphs.size() == 2) { group_of_2_count++; }
    }
    BOOST_CHECK_EQUAL(group_of_1_count, 2);
    BOOST_CHECK_EQUAL(group_of_2_count, 1);

    check_partitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessor_PartitionCheck_MediumGraph) {
    const auto project_root = get_project_root();
    graph_t dag;
    file_reader::readComputationalDagHyperdagFormatDB((project_root / "data/spaa/tiny/instance_bicgstab.hdag").string(), dag);

    BOOST_REQUIRE_GT(dag.num_vertices(), 0);

    // Use a higher threshold to encourage more merging on this larger graph
    OrbitGraphProcessor<graph_t, graph_t> processor;
    MerkleHashComputer<graph_t, bwd_merkle_node_hash_func<graph_t>, true> hasher(dag, dag);
    processor.discover_isomorphic_groups(dag, hasher);

    // The main purpose of this test is to ensure the output is a valid partition.
    check_partitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessor_MultiPipelineMerge) {
    // 5 parallel pipelines of 4 nodes each.
    // Initial orbits: 4 groups of 5 identical nodes. Coarse graph: 0->1->2->3
    // With a threshold of 5, the entire graph should merge into a single group.
    const auto dag = construct_multi_pipeline_dag<graph_t>(5, 4);
    BOOST_REQUIRE_EQUAL(dag.num_vertices(), 20);

    OrbitGraphProcessor<graph_t, graph_t> processor;    // Set threshold to match pipeline count
    MerkleHashComputer<graph_t, bwd_merkle_node_hash_func<graph_t>, true> hasher(dag, dag);
    processor.discover_isomorphic_groups(dag, hasher);

    const auto &final_coarse_graph = processor.get_final_coarse_graph();
    const auto &final_groups = processor.get_final_groups();

    // Expect a single node in the final coarse graph
    BOOST_CHECK_EQUAL(final_coarse_graph.num_vertices(), 1);
    BOOST_CHECK_EQUAL(final_groups.size(), 1);

    // The single group should contain 5 subgraphs, each with 4 nodes.
    BOOST_REQUIRE_EQUAL(final_groups[0].subgraphs.size(), 5);
    BOOST_CHECK_EQUAL(final_groups[0].subgraphs[0].size(), 4);

    check_partitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessor_LadderNoMerge) {
    // A ladder graph with 10 rungs (22 nodes).
    // The bwd_merkle_hash is more discerning and creates more than 2 initial orbits
    // due to the different structures at the start and end of the ladder.
    // The coarsening logic will merge some of these, but the core cyclic structure
    // prevents a full merge. The exact number of final nodes is non-trivial,
    // but it should be greater than 1.
    const auto dag = construct_ladder_dag<graph_t>(10);
    BOOST_REQUIRE_EQUAL(dag.num_vertices(), 22);

    OrbitGraphProcessor<graph_t, graph_t> processor;
    MerkleHashComputer<graph_t, bwd_merkle_node_hash_func<graph_t>, true> hasher(dag, dag);
    processor.discover_isomorphic_groups(dag, hasher);

    const auto &initial_coarse_graph = processor.get_coarse_graph();
    const auto &final_coarse_graph = processor.get_final_coarse_graph();

    // Expect no merges, so final graph is the same as the initial coarse graph.
    BOOST_CHECK_EQUAL(final_coarse_graph.num_vertices(), initial_coarse_graph.num_vertices());
    BOOST_CHECK_GT(final_coarse_graph.num_vertices(), 1);

    check_partitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessor_AsymmetricNoMerge) {
    // A simple chain where every node is unique.
    // Since all groups are below the threshold, they will all be merged into one.
    const auto dag = construct_asymmetric_dag<graph_t>(30);
    BOOST_REQUIRE_EQUAL(dag.num_vertices(), 30);

    OrbitGraphProcessor<graph_t, graph_t> processor;
    MerkleHashComputer<graph_t, bwd_merkle_node_hash_func<graph_t>, true> hasher(dag, dag);
    processor.discover_isomorphic_groups(dag, hasher);

    const auto &final_coarse_graph = processor.get_final_coarse_graph();

    // Expect all nodes to be merged into a single coarse node.
    BOOST_CHECK_EQUAL(final_coarse_graph.num_vertices(), 1);

    check_partitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessor_BinaryTreeNoMerge) {
    // A binary out-tree of height 4.
    // Initial orbits are one per level. Coarse graph is a simple chain: 0->1->2->3->4 (5 nodes).
    // The logic allows merging groups that are below the symmetry threshold.
    // However, the `critical_path_weight` check prevents merges that would increase the
    // longest path in the coarse graph. This results in the chain being partially, but not
    // fully, collapsed. The expected outcome is 2 final coarse nodes.
    const auto dag = construct_binary_out_tree<graph_t>(4);
    BOOST_REQUIRE_EQUAL(dag.num_vertices(), (1 << 5) - 1);

    OrbitGraphProcessor<graph_t, graph_t> processor;
    MerkleHashComputer<graph_t, bwd_merkle_node_hash_func<graph_t>, true> hasher(dag, dag);
    processor.discover_isomorphic_groups(dag, hasher);

    const auto &final_coarse_graph = processor.get_final_coarse_graph();

    BOOST_CHECK_EQUAL(final_coarse_graph.num_vertices(), 3);

    check_partitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessor_ButterflyMerge) {
    const auto dag = construct_butterfly_dag<graph_t>(3);
    BOOST_REQUIRE_EQUAL(dag.num_vertices(), (3 + 1) * 8);

    OrbitGraphProcessor<graph_t, graph_t> processor;
    MerkleHashComputer<graph_t, bwd_merkle_node_hash_func<graph_t>, true> hasher(dag, dag);
    processor.discover_isomorphic_groups(dag, hasher);

    const auto &final_coarse_graph = processor.get_final_coarse_graph();
    BOOST_CHECK_EQUAL(final_coarse_graph.num_vertices(), 4);

    check_partitioning(dag, processor);
}
