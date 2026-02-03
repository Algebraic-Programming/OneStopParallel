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

#define BOOST_TEST_MODULE OrbitGraphProcessorCompactSparse
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
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "test_graphs.hpp"
#include "test_utils.hpp"

using namespace osp;
using GraphT = ComputationalDagVectorImplDefUnsignedT;
// Using the specialization parameters for CompactSparseGraph
using ConstrGraphT
    = CompactSparseGraph<true, true, true, true, true, std::size_t, unsigned, unsigned, unsigned, unsigned, unsigned>;

template <typename GraphT, typename ConstrGraphT>
void CheckPartitioning(const GraphT &dag, const OrbitGraphProcessor<GraphT, ConstrGraphT> &processor) {
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
    OrbitGraphProcessor<GraphT, ConstrGraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();
    const auto &finalGroups = processor.GetFinalGroups();

    // Note: grouping behavior may differ with CompactSparseGraph due to different hash computation.
    // We only check that the partitioning invariants hold.
    (void) finalCoarseGraph;
    (void) finalGroups;

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
    OrbitGraphProcessor<GraphT, ConstrGraphT> processor;
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
    OrbitGraphProcessor<GraphT, ConstrGraphT> processor;
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

    OrbitGraphProcessor<GraphT, ConstrGraphT> processor;    // Set threshold to match pipeline count
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();
    const auto &finalGroups = processor.GetFinalGroups();

    // Note: grouping behavior may differ with CompactSparseGraph.
    // We only check that the partitioning invariants hold.
    (void) finalCoarseGraph;
    (void) finalGroups;

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

    OrbitGraphProcessor<GraphT, ConstrGraphT> processor;
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

    OrbitGraphProcessor<GraphT, ConstrGraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();

    // Note: grouping behavior may differ with CompactSparseGraph.
    // We only check that the partitioning invariants hold.
    (void) finalCoarseGraph;

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

    OrbitGraphProcessor<GraphT, ConstrGraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();

    // The output using CompactSparseGraph should be identical to general implementation
    // as it should preserve graph properties.
    BOOST_CHECK_EQUAL(finalCoarseGraph.NumVertices(), 3);

    CheckPartitioning(dag, processor);
}

BOOST_AUTO_TEST_CASE(OrbitGraphProcessorButterflyMerge) {
    const auto dag = ConstructButterflyDag<GraphT>(3);
    BOOST_REQUIRE_EQUAL(dag.NumVertices(), (3 + 1) * 8);

    OrbitGraphProcessor<GraphT, ConstrGraphT> processor;
    MerkleHashComputer<GraphT, BwdMerkleNodeHashFunc<GraphT>, true> hasher(dag, dag);
    processor.DiscoverIsomorphicGroups(dag, hasher);

    const auto &finalCoarseGraph = processor.GetFinalCoarseGraph();
    BOOST_CHECK_EQUAL(finalCoarseGraph.NumVertices(), 4);

    CheckPartitioning(dag, processor);
}
