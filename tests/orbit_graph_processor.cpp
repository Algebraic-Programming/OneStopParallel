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
using GraphT = ComputationalDagVectorImplDefUnsignedT;

template <typename GraphT>
void CheckPartitioning(const GraphT &dag, const OrbitGraphProcessor<GraphT, GraphT> &processor) {
    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();
    const auto &finalGroups = processor.GetFinalGroups();

    // Check that the final coarse graph is acyclic
    BOOST_CHECK(IsAcyclic(finalCoarseGraph));

    // Check that the final groups form a valid partition of the original DAG's vertices
    std::vector<int> vertexCounts(dag.NumVertices(), 0);
    size_t totalVerticesInGroups = 0;
    for (const auto &group : finalGroups) {
        for (const auto &subgraph : group.subgraphs_) {
            totalVerticesInGroups += subgraph.size();
            for (const auto &vertex : subgraph) {
                BOOST_REQUIRE_LT(vertex, dag.NumVertices());
                vertexCounts[vertex]++;
            }
        }
    }
    BOOST_CHECK_EQUAL(totalVerticesInGroups, dag.NumVertices());
    for (size_t i = 0; i < dag.NumVertices(); ++i) {
        BOOST_CHECK_EQUAL(vertexCounts[i], 1);
    }
}

// BOOST_AUTO_TEST_CASE(OrbitGraphProcessor_SmokeTest) {
//     // The test reads a file, but the path is absolute, so we don't need the project root here.
//     graph_t dag;
//     file_reader::ReadComputationalDagDotFormat("", dag);

//     OrbitGraphProcessor<graph_t, graph_t> processor(2); // Using a symmetry threshold of 2
//     MerkleHashComputer<graph_t, BwdMerkleNodeHashFunc<graph_t>, true> hasher(dag, dag);
//     processor.DiscoverIsomorphicGroups(dag, hasher);

//     const auto& coarse_graph = processor.GetCoarseGraph();
//     const auto& final_coarse_graph = processor.GetFinalCoarseGraph();
//     const auto& final_groups = processor.GetFinalGroups();
//     const auto& final_contraction_map = processor.get_final_contraction_map();

//     DotFileWriter writer;
//     // Color by initial orbits
//     writer.write_colored_graph("orbit_graph_orbits_colored.dot", dag, processor.get_contraction_map());
//     writer.WriteGraph("orbit_graph_coarse_graph.dot", coarse_graph);

//     // Color by final merged groups
//     writer.write_colored_graph("orbit_graph_groups_colored.dot", dag, final_contraction_map);

//     // Color by final subgraphs (each subgraph gets a unique color)
//     std::vector<unsigned> subgraph_colors(dag.NumVertices());
//     unsigned current_subgraph_color = 0;
//     for (const auto& group : final_groups) {
//         for (const auto& subgraph : group.subgraphs_) {
//             for (const auto& vertex : subgraph) {
//                 subgraph_colors[vertex] = current_subgraph_color;
//             }
//             current_subgraph_color++;
//         }
//     }
//     writer.write_colored_graph("orbit_graph_subgraphs_colored.dot", dag, subgraph_colors);
//     writer.WriteGraph("orbit_graph_final_coarse_graph.dot", final_coarse_graph);

//     BOOST_CHECK_GT(coarse_graph.NumVertices(), 0);
//     BOOST_CHECK_LT(coarse_graph.NumVertices(), dag.NumVertices());
//     BOOST_CHECK_GT(final_coarse_graph.NumVertices(), 0);
//     BOOST_CHECK_LE(final_coarse_graph.NumVertices(), coarse_graph.NumVertices());

//     check_partitioning(dag, processor);
// }

BOOST_AUTO_TEST_CASE(OrbitGraphProcessorSimpleMerge) {
    GraphT dag;
    // Two parallel pipelines that are structurally identical
    // 0 -> 1
    // 2 -> 3
    dag.AddVertex(10, 1, 1);    // 0
    dag.AddVertex(10, 1, 1);    // 1
    dag.AddVertex(10, 1, 1);    // 2
    dag.AddVertex(10, 1, 1);    // 3
    dag.AddEdge(0, 1);
    dag.AddEdge(2, 3);

    // Initial orbits: {0, 2} and {1, 3}. Coarse graph: 0 -> 1
    // With threshold 2, these should be merged.
    OrbitGraphProcessor<GraphT, GraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();
    const auto &finalGroups = processor.GetFinalGroups();

    // Expect a single node in the final coarse graph
    BOOST_CHECK_EQUAL(finalCoarseGraph.NumVertices(), 1);
    BOOST_CHECK_EQUAL(finalGroups.size(), 1);

    // The single group should contain two subgraphs: {0,1} and {2,3}
    BOOST_REQUIRE_EQUAL(finalGroups[0].subgraphs_.size(), 2);
    std::set<int> sg1(finalGroups[0].subgraphs_[0].begin(), finalGroups[0].subgraphs_[0].end());
    std::set<int> sg2(finalGroups[0].subgraphs_[1].begin(), finalGroups[0].subgraphs_[1].end());
    std::set<int> expectedSgA = {0, 1};
    std::set<int> expectedSgB = {2, 3};

    BOOST_CHECK((sg1 == expectedSgA && sg2 == expectedSgB) || (sg1 == expectedSgB && sg2 == expectedSgA));

    CheckPartitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessorForkJoinNoMerge) {
    GraphT dag;
    // 0 -> {1, 2} -> 3. Nodes 1 and 2 are in the same orbit.
    dag.AddVertex(10, 1, 1);    // 0
    dag.AddVertex(20, 1, 1);    // 1
    dag.AddVertex(20, 1, 1);    // 2
    dag.AddVertex(30, 1, 1);    // 3
    dag.AddEdge(0, 1);
    dag.AddEdge(0, 2);
    dag.AddEdge(1, 3);
    dag.AddEdge(2, 3);

    // Initial orbits: {0}, {1,2}, {3}. Coarse graph: 0 -> 1 -> 2
    // Merging 0 and 1 would result in a group of size 1 ({0,1,2}), which is not viable (threshold 2).
    // Merging 1 and 2 would also result in a group of size 1 ({1,2,3}), not viable.
    OrbitGraphProcessor<GraphT, GraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();
    const auto &finalGroups = processor.GetFinalGroups();

    // Expect no merges, so final graph is same as initial coarse graph.
    BOOST_CHECK_EQUAL(finalCoarseGraph.NumVertices(), 3);
    BOOST_CHECK_EQUAL(finalGroups.size(), 3);

    // Check group structures
    // Group 0: {{0}}
    // Group 1: {{1}, {2}}
    // Group 2: {{3}}
    size_t groupOf1Count = 0;
    size_t groupOf2Count = 0;
    for (const auto &group : finalGroups) {
        if (group.subgraphs_.size() == 1) {
            groupOf1Count++;
        }
        if (group.subgraphs_.size() == 2) {
            groupOf2Count++;
        }
    }
    BOOST_CHECK_EQUAL(groupOf1Count, 2);
    BOOST_CHECK_EQUAL(groupOf2Count, 1);

    CheckPartitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessorPartitionCheckMediumGraph) {
    const auto projectRoot = GetProjectRoot();
    GraphT dag;
    file_reader::ReadComputationalDagHyperdagFormatDB((projectRoot / "data/spaa/tiny/instance_bicgstab.hdag").string(), dag);

    BOOST_REQUIRE_GT(dag.NumVertices(), 0);

    // Use a higher threshold to encourage more merging on this larger graph
    OrbitGraphProcessor<GraphT, GraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    // The main purpose of this test is to ensure the output is a valid partition.
    CheckPartitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessorMultiPipelineMerge) {
    // 5 parallel pipelines of 4 nodes each.
    // Initial orbits: 4 groups of 5 identical nodes. Coarse graph: 0->1->2->3
    // With a threshold of 5, the entire graph should merge into a single group.
    const auto dag = ConstructMultiPipelineDag<GraphT>(5, 4);
    BOOST_REQUIRE_EQUAL(dag.NumVertices(), 20);

    OrbitGraphProcessor<GraphT, GraphT> processor;    // Set threshold to match pipeline count
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();
    const auto &finalGroups = processor.GetFinalGroups();

    // Expect a single node in the final coarse graph
    BOOST_CHECK_EQUAL(finalCoarseGraph.NumVertices(), 1);
    BOOST_CHECK_EQUAL(finalGroups.size(), 1);

    // The single group should contain 5 subgraphs, each with 4 nodes.
    BOOST_REQUIRE_EQUAL(finalGroups[0].subgraphs_.size(), 5);
    BOOST_CHECK_EQUAL(finalGroups[0].subgraphs_[0].size(), 4);

    CheckPartitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessorLadderNoMerge) {
    // A ladder graph with 10 rungs (22 nodes).
    // The bwd_merkle_hash is more discerning and creates more than 2 initial orbits
    // due to the different structures at the start and end of the ladder.
    // The coarsening logic will merge some of these, but the core cyclic structure
    // prevents a full merge. The exact number of final nodes is non-trivial,
    // but it should be greater than 1.
    const auto dag = ConstructLadderDag<GraphT>(10);
    BOOST_REQUIRE_EQUAL(dag.NumVertices(), 22);

    OrbitGraphProcessor<GraphT, GraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &initialCoarseGraph = processor.GetCoarseGraph();
    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();

    // Expect no merges, so final graph is the same as the initial coarse graph.
    BOOST_CHECK_EQUAL(finalCoarseGraph.NumVertices(), initialCoarseGraph.NumVertices());
    BOOST_CHECK_GT(finalCoarseGraph.NumVertices(), 1);

    CheckPartitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessorAsymmetricNoMerge) {
    // A simple chain where every node is unique.
    // Since all groups are below the threshold, they will all be merged into one.
    const auto dag = ConstructAsymmetricDag<GraphT>(30);
    BOOST_REQUIRE_EQUAL(dag.NumVertices(), 30);

    OrbitGraphProcessor<GraphT, GraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();

    // Expect all nodes to be merged into a single coarse node.
    BOOST_CHECK_EQUAL(finalCoarseGraph.NumVertices(), 1);

    CheckPartitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessorBinaryTreeNoMerge) {
    // A binary out-tree of height 4.
    // Initial orbits are one per level. Coarse graph is a simple chain: 0->1->2->3->4 (5 nodes).
    // The logic allows merging groups that are below the symmetry threshold.
    // However, the `critical_path_weight` check prevents merges that would increase the
    // longest path in the coarse graph. This results in the chain being partially, but not
    // fully, collapsed. The expected outcome is 2 final coarse nodes.
    const auto dag = ConstructBinaryOutTree<GraphT>(4);
    BOOST_REQUIRE_EQUAL(dag.NumVertices(), (1 << 5) - 1);

    OrbitGraphProcessor<GraphT, GraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();

    BOOST_CHECK_EQUAL(finalCoarseGraph.NumVertices(), 3);

    CheckPartitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessorButterflyMerge) {
    const auto dag = ConstructButterflyDag<GraphT>(3);
    BOOST_REQUIRE_EQUAL(dag.NumVertices(), (3 + 1) * 8);

    OrbitGraphProcessor<GraphT, GraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();
    BOOST_CHECK_EQUAL(finalCoarseGraph.NumVertices(), 4);

    CheckPartitioning(dag, processor);
}
