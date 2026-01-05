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

#define BOOST_TEST_MODULE SequenceSplitterTest
#include <boost/test/unit_test.hpp>

#include "osp/dag_divider/wavefront_divider/RecursiveWavefrontDivider.hpp"
#include "osp/dag_divider/wavefront_divider/ScanWavefrontDivider.hpp"
#include "osp/dag_divider/wavefront_divider/SequenceGenerator.hpp"
#include "osp/dag_divider/wavefront_divider/SequenceSplitter.hpp"
#include "osp/dag_divider/wavefront_divider/WavefrontStatisticsCollector.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

BOOST_AUTO_TEST_CASE(VarianceSplitterTest) {
    osp::VarianceSplitter splitter(0.8, 0.1);

    // Test case 1: Clear split point
    std::vector<double> seq1 = {1, 1, 1, 1, 10, 10, 10, 10};
    std::vector<size_t> splits1 = splitter.Split(seq1);
    std::vector<size_t> expected1 = {4};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits1.begin(), splits1.end(), expected1.begin(), expected1.end());

    // Test case 2: No split needed (low variance)
    std::vector<double> seq2 = {1, 1.1, 1.2, 1.1, 1.3};
    std::vector<size_t> splits2 = splitter.Split(seq2);
    BOOST_CHECK(splits2.empty());

    // Test case 3: Empty sequence
    std::vector<double> seq3 = {};
    std::vector<size_t> splits3 = splitter.Split(seq3);
    BOOST_CHECK(splits3.empty());

    // Test case 4: Single element sequence
    std::vector<double> seq4 = {100.0};
    std::vector<size_t> splits4 = splitter.Split(seq4);
    BOOST_CHECK(splits4.empty());

    // Test case 5: Multiple splits
    std::vector<double> seq5 = {1, 1, 1, 20, 20, 20, 1, 1, 1};
    std::vector<size_t> splits5 = splitter.Split(seq5);
    std::vector<size_t> expected5 = {3, 6};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits5.begin(), splits5.end(), expected5.begin(), expected5.end());
}

BOOST_AUTO_TEST_CASE(LargestStepSplitterTest) {
    osp::LargestStepSplitter splitter(5.0, 2);

    // Test case 1: Clear step
    std::vector<double> seq1 = {1, 2, 3, 10, 11, 12};
    std::vector<size_t> splits1 = splitter.Split(seq1);
    std::vector<size_t> expected1 = {3};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits1.begin(), splits1.end(), expected1.begin(), expected1.end());

    // Test case 2: No significant step
    std::vector<double> seq2 = {1, 2, 3, 4, 5, 6};
    std::vector<size_t> splits2 = splitter.Split(seq2);
    BOOST_CHECK(splits2.empty());

    // Test case 3: Decreasing sequence
    std::vector<double> seq3 = {12, 11, 10, 3, 2, 1};
    std::vector<size_t> splits3 = splitter.Split(seq3);
    std::vector<size_t> expected3 = {3};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits3.begin(), splits3.end(), expected3.begin(), expected3.end());

    // Test case 4: Sequence too short
    std::vector<double> seq4 = {1, 10};
    std::vector<size_t> splits4 = splitter.Split(seq4);
    BOOST_CHECK(splits4.empty());

    // Test case 5: Multiple large steps
    std::vector<double> seq5 = {0, 1, 10, 11, 20, 21};
    std::vector<size_t> splits5 = splitter.Split(seq5);
    std::vector<size_t> expected5 = {2, 4};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits5.begin(), splits5.end(), expected5.begin(), expected5.end());
}

BOOST_AUTO_TEST_CASE(ThresholdScanSplitterTest) {
    osp::ThresholdScanSplitter splitter(5.0, 10.0);

    // Test case 1: Significant drop
    std::vector<double> seq1 = {20, 18, 16, 9, 8, 7};
    std::vector<size_t> splits1 = splitter.Split(seq1);
    std::vector<size_t> expected1 = {3};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits1.begin(), splits1.end(), expected1.begin(), expected1.end());

    // Test case 2: Crossing absolute threshold (rising)
    std::vector<double> seq2 = {5, 7, 9, 11, 13};
    std::vector<size_t> splits2 = splitter.Split(seq2);
    std::vector<size_t> expected2 = {3};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits2.begin(), splits2.end(), expected2.begin(), expected2.end());

    // Test case 3: Crossing absolute threshold (dropping)
    std::vector<double> seq3 = {15, 12, 11, 9, 8};
    std::vector<size_t> splits3 = splitter.Split(seq3);
    std::vector<size_t> expected3 = {3};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits3.begin(), splits3.end(), expected3.begin(), expected3.end());

    // Test case 4: No splits
    std::vector<double> seq4 = {1, 2, 3, 4, 5};
    std::vector<size_t> splits4 = splitter.Split(seq4);
    BOOST_CHECK(splits4.empty());

    // Test case 5: Empty sequence
    std::vector<double> seq5 = {};
    std::vector<size_t> splits5 = splitter.Split(seq5);
    BOOST_CHECK(splits5.empty());
}

using Graph = osp::ComputationalDagEdgeIdxVectorImplDefIntT;
using VertexType = Graph::VertexIdx;

BOOST_AUTO_TEST_CASE(ForwardAndBackwardPassTest) {
    Graph dag;
    const auto v1 = dag.AddVertex(2, 1, 9);
    const auto v2 = dag.AddVertex(3, 1, 8);
    const auto v3 = dag.AddVertex(4, 1, 7);
    const auto v4 = dag.AddVertex(5, 1, 6);
    const auto v5 = dag.AddVertex(6, 1, 5);
    const auto v6 = dag.AddVertex(7, 1, 4);
    const auto v7 = dag.AddVertex(8, 1, 3);    // Note: v7 is not connected in the example
    const auto v8 = dag.AddVertex(9, 1, 2);

    dag.AddEdge(v1, v2);
    dag.AddEdge(v1, v3);
    dag.AddEdge(v1, v4);
    dag.AddEdge(v2, v5);
    dag.AddEdge(v2, v6);
    dag.AddEdge(v3, v5);
    dag.AddEdge(v3, v6);
    dag.AddEdge(v5, v8);
    dag.AddEdge(v4, v8);

    // Manually defined level sets for this DAG
    const std::vector<std::vector<VertexType>> levelSets = {
        {v1}, // Level 0
        {v2, v3, v4}, // Level 1
        {v5, v6}, // Level 2
        {v8}, // Level 3
        {v7}  // Level 4 (isolated vertex)
    };

    osp::WavefrontStatisticsCollector<Graph> collector(dag, levelSets);

    // --- Test Forward Pass ---
    auto forwardStats = collector.ComputeForward();
    BOOST_REQUIRE_EQUAL(forwardStats.size(), 5);

    // Level 0
    BOOST_CHECK_EQUAL(forwardStats[0].connectedComponentsVertices_.size(), 1);
    BOOST_CHECK_EQUAL(forwardStats[0].connectedComponentsWeights_[0], 2);
    BOOST_CHECK_EQUAL(forwardStats[0].connectedComponentsMemories_[0], 9);

    // Level 1
    BOOST_CHECK_EQUAL(forwardStats[1].connectedComponentsVertices_.size(), 1);
    BOOST_CHECK_EQUAL(forwardStats[1].connectedComponentsWeights_[0], 2 + 3 + 4 + 5);    // v1,v2,v3,v4
    BOOST_CHECK_EQUAL(forwardStats[1].connectedComponentsMemories_[0], 9 + 8 + 7 + 6);

    // Level 2
    BOOST_CHECK_EQUAL(forwardStats[2].connectedComponentsVertices_.size(), 1);
    BOOST_CHECK_EQUAL(forwardStats[2].connectedComponentsWeights_[0], 14 + 6 + 7);    // v1-v6
    BOOST_CHECK_EQUAL(forwardStats[2].connectedComponentsMemories_[0], 30 + 5 + 4);

    // Level 3
    BOOST_CHECK_EQUAL(forwardStats[3].connectedComponentsVertices_.size(), 1);
    BOOST_CHECK_EQUAL(forwardStats[3].connectedComponentsWeights_[0], 27 + 9);    // v1-v6, v8
    BOOST_CHECK_EQUAL(forwardStats[3].connectedComponentsMemories_[0], 39 + 2);

    // Level 4 (isolated vertex shows up as a new component)
    BOOST_CHECK_EQUAL(forwardStats[4].connectedComponentsVertices_.size(), 2);

    // --- Test Backward Pass ---
    auto backwardStats = collector.ComputeBackward();
    BOOST_REQUIRE_EQUAL(backwardStats.size(), 5);

    // Level 4
    BOOST_CHECK_EQUAL(backwardStats[4].connectedComponentsVertices_.size(), 1);
    BOOST_CHECK_EQUAL(backwardStats[4].connectedComponentsWeights_[0], 8);    // v7
    BOOST_CHECK_EQUAL(backwardStats[4].connectedComponentsMemories_[0], 3);

    // Level 3
    BOOST_CHECK_EQUAL(backwardStats[3].connectedComponentsVertices_.size(), 2);    // {v8}, {v7}

    // Level 2
    BOOST_CHECK_EQUAL(backwardStats[2].connectedComponentsVertices_.size(), 3);    // {v5,v8}, {v6}, {v7}

    // Level 1
    BOOST_CHECK_EQUAL(backwardStats[1].connectedComponentsVertices_.size(), 2);    // {v2,v3,v4,v5,v6,v8}, {v7}

    // Level 0
    BOOST_CHECK_EQUAL(backwardStats[0].connectedComponentsVertices_.size(), 2);    // {v1-v6,v8}, {v7}
}

BOOST_AUTO_TEST_CASE(SequenceGenerationTest) {
    // --- Test Setup ---
    Graph dag;
    const auto v1 = dag.AddVertex(2, 1, 9);
    const auto v2 = dag.AddVertex(3, 1, 8);
    const auto v3 = dag.AddVertex(4, 1, 7);
    const auto v4 = dag.AddVertex(5, 1, 6);
    const auto v5 = dag.AddVertex(6, 1, 5);
    const auto v6 = dag.AddVertex(7, 1, 4);
    const auto v7 = dag.AddVertex(8, 1, 3);    // Isolated vertex
    const auto v8 = dag.AddVertex(9, 1, 2);

    dag.AddEdge(v1, v2);
    dag.AddEdge(v1, v3);
    dag.AddEdge(v1, v4);
    dag.AddEdge(v2, v5);
    dag.AddEdge(v2, v6);
    dag.AddEdge(v3, v5);
    dag.AddEdge(v3, v6);
    dag.AddEdge(v5, v8);
    dag.AddEdge(v4, v8);

    const std::vector<std::vector<VertexType>> levelSets = {
        {v1},
        {v2, v3, v4},
        {v5, v6},
        {v8},
        {v7}
    };

    osp::SequenceGenerator<Graph> generator(dag, levelSets);

    // --- Test Component Count ---
    auto componentSeq = generator.Generate(osp::SequenceMetric::COMPONENT_COUNT);
    std::vector<double> expectedComponents = {1.0, 1.0, 1.0, 1.0, 2.0};
    BOOST_CHECK_EQUAL_COLLECTIONS(componentSeq.begin(), componentSeq.end(), expectedComponents.begin(), expectedComponents.end());

    // --- Test Available Parallelism ---
    auto parallelismSeq = generator.Generate(osp::SequenceMetric::AVAILABLE_PARALLELISM);

    // Manual calculation for expected values:
    // L0: 2 / 1 = 2
    // L1: (2 + 3+4+5) / 2 = 14 / 2 = 7
    // L2: (14 + 6+7) / 3 = 27 / 3 = 9
    // L3: (27 + 9) / 4 = 36 / 4 = 9
    // L4: (36 + 8) / 5 = 44 / 5 = 8.8
    std::vector<double> expectedParallelism = {2.0, 7.0, 9.0, 9.0, 8.8};

    BOOST_REQUIRE_EQUAL(parallelismSeq.size(), expectedParallelism.size());
    for (size_t i = 0; i < parallelismSeq.size(); ++i) {
        BOOST_CHECK_CLOSE(parallelismSeq[i], expectedParallelism[i], 1e-9);
    }
}

struct TestFixture {
    Graph dag_;
    std::vector<VertexType> vertices_;

    TestFixture() {
        // --- Test Setup ---
        // Note: The compute_wavefronts method will determine the levels.
        // The actual level sets for this graph are:
        // Level 0: {v1, v7}
        // Level 1: {v2, v3, v4}
        // Level 2: {v5, v6}
        // Level 3: {v8}
        const auto v1 = dag_.AddVertex(2, 1, 9);
        const auto v2 = dag_.AddVertex(3, 1, 8);
        const auto v3 = dag_.AddVertex(4, 1, 7);
        const auto v4 = dag_.AddVertex(5, 1, 6);
        const auto v5 = dag_.AddVertex(6, 1, 5);
        const auto v6 = dag_.AddVertex(7, 1, 4);
        const auto v7 = dag_.AddVertex(8, 1, 3);    // Isolated vertex
        const auto v8 = dag_.AddVertex(9, 1, 2);

        vertices_ = {v1, v2, v3, v4, v5, v6, v7, v8};

        dag_.AddEdge(v1, v2);
        dag_.AddEdge(v1, v3);
        dag_.AddEdge(v1, v4);
        dag_.AddEdge(v2, v5);
        dag_.AddEdge(v2, v6);
        dag_.AddEdge(v3, v5);
        dag_.AddEdge(v3, v6);
        dag_.AddEdge(v5, v8);
        dag_.AddEdge(v4, v8);
    }
};

BOOST_FIXTURE_TEST_SUITE(scan_wavefront_divider_test_suite, TestFixture)

BOOST_AUTO_TEST_CASE(LargestStepDivisionTest) {
    osp::ScanWavefrontDivider<Graph> divider;
    divider.SetMetric(osp::SequenceMetric::AVAILABLE_PARALLELISM);
    divider.UseLargestStepSplitter(0.9, 1);

    auto sections = divider.Divide(dag_);

    // Expecting a cut after level 0. This results in 2 sections.
    BOOST_REQUIRE_EQUAL(sections.size(), 2);

    // Section 1: level 0. Components: {v1}, {v7}
    BOOST_REQUIRE_EQUAL(sections[0].size(), 2);

    // Section 2: levels 1, 2, 3. The rest of the main component.
    BOOST_REQUIRE_EQUAL(sections[1].size(), 1);
    BOOST_CHECK_EQUAL(sections[1][0].size(), 6);    // v2,v3,v4,v5,v6,v8
}

BOOST_AUTO_TEST_CASE(ThresholdScanDivisionTest) {
    osp::ScanWavefrontDivider<Graph> divider;
    divider.SetMetric(osp::SequenceMetric::AVAILABLE_PARALLELISM);
    divider.UseThresholdScanSplitter(2.0, 11.5);

    auto sections = divider.Divide(dag_);

    // A cut is expected when the sequence crosses 11.5 (at level 2) and crosses back (at level 3)
    // The splitter should return cuts at levels 2 and 3.
    // This results in 3 sections: {levels 0,1}, {level 2}, {level 3}
    BOOST_REQUIRE_EQUAL(sections.size(), 3);

    // Section 1: levels 0, 1. Components: {v1,v2,v3,v4}, {v7}
    BOOST_REQUIRE_EQUAL(sections[0].size(), 2);
    // Section 2: level 2. Vertices {v5, v6}. They are not connected within this level, so they are 2 components.
    BOOST_REQUIRE_EQUAL(sections[1].size(), 2);
    // Section 3: level 3. Components: {v8}
    BOOST_REQUIRE_EQUAL(sections[2].size(), 1);
}

BOOST_AUTO_TEST_CASE(NoCutDivisionTest) {
    osp::ScanWavefrontDivider<Graph> divider;
    divider.SetMetric(osp::SequenceMetric::COMPONENT_COUNT);
    divider.UseLargestStepSplitter(2.0, 2);

    auto sections = divider.Divide(dag_);

    // Expecting a single section containing all components
    BOOST_REQUIRE_EQUAL(sections.size(), 1);
    BOOST_REQUIRE_EQUAL(sections[0].size(), 2);    // Two final components
}

BOOST_AUTO_TEST_CASE(EmptyGraphTest) {
    osp::ScanWavefrontDivider<Graph> divider;
    Graph emptyDag;
    auto sections = divider.Divide(emptyDag);
    BOOST_CHECK(sections.empty());
}

BOOST_AUTO_TEST_SUITE_END()

struct TestFixture2 {
    Graph dag_;
    std::vector<VertexType> vertices_;

    TestFixture2() {
        // --- Test Setup ---
        // This graph is designed to have a component count sequence of {2, 2, 2, 1}
        // to properly test the recursive divider's splitting logic.
        // Levels: {v1,v2}, {v3,v4}, {v5,v6}, {v7}
        const auto v1 = dag_.AddVertex(1, 1, 1);
        const auto v2 = dag_.AddVertex(1, 1, 1);
        const auto v3 = dag_.AddVertex(1, 1, 1);
        const auto v4 = dag_.AddVertex(1, 1, 1);
        const auto v5 = dag_.AddVertex(1, 1, 1);
        const auto v6 = dag_.AddVertex(1, 1, 1);
        const auto v7 = dag_.AddVertex(1, 1, 1);

        vertices_ = {v1, v2, v3, v4, v5, v6, v7};

        dag_.AddEdge(v1, v3);
        dag_.AddEdge(v2, v4);
        dag_.AddEdge(v3, v5);
        dag_.AddEdge(v4, v6);
        dag_.AddEdge(v5, v7);
        dag_.AddEdge(v6, v7);
    }
};

BOOST_AUTO_TEST_SUITE(recursive_wavefront_divider_test_suite)

// --- Test Fixture 1: A simple DAG that merges from 2 components to 1 ---
struct TestFixtureSimpleMerge {
    Graph dag_;

    TestFixtureSimpleMerge() {
        // This graph is designed to have a component count sequence of {2, 2, 2, 1}
        // Levels: {v0,v1}, {v2,v3}, {v4,v5}, {v6}
        const auto v0 = dag_.AddVertex(1, 1, 1);
        const auto v1 = dag_.AddVertex(1, 1, 1);
        const auto v2 = dag_.AddVertex(1, 1, 1);
        const auto v3 = dag_.AddVertex(1, 1, 1);
        const auto v4 = dag_.AddVertex(1, 1, 1);
        const auto v5 = dag_.AddVertex(1, 1, 1);
        const auto v6 = dag_.AddVertex(1, 1, 1);

        dag_.AddEdge(v0, v2);
        dag_.AddEdge(v1, v3);
        dag_.AddEdge(v2, v4);
        dag_.AddEdge(v3, v5);
        dag_.AddEdge(v4, v6);
        dag_.AddEdge(v5, v6);
    }
};

BOOST_FIXTURE_TEST_SUITE(simple_merge_tests, TestFixtureSimpleMerge)

BOOST_AUTO_TEST_CASE(BasicRecursionTest) {
    osp::RecursiveWavefrontDivider<Graph> divider;
    divider.UseLargestStepSplitter(0.5, 1);
    auto sections = divider.Divide(dag_);

    // Expecting a cut after level 2, where component count drops from 2 to 1.
    // This results in 2 sections: {levels 0,1,2} and {level 3}.
    BOOST_REQUIRE_EQUAL(sections.size(), 2);

    // Section 1: levels 0-2. Components: {v0,v2,v4}, {v1,v3,v5}
    BOOST_REQUIRE_EQUAL(sections[0].size(), 2);

    // Section 2: level 3. Component: {v6}
    BOOST_REQUIRE_EQUAL(sections[1].size(), 1);
    BOOST_CHECK_EQUAL(sections[1][0].size(), 1);
}

BOOST_AUTO_TEST_CASE(NoCutHighThresholdTest) {
    // A high threshold should prevent any cuts.
    osp::RecursiveWavefrontDivider<Graph> divider;
    divider.UseLargestStepSplitter(2.0, 2);
    auto sections = divider.Divide(dag_);

    // Expecting a single section containing all components, which merge into one.
    BOOST_REQUIRE_EQUAL(sections.size(), 1);
    BOOST_REQUIRE_EQUAL(sections[0].size(), 1);
}

BOOST_AUTO_TEST_CASE(MinSubsequenceLengthTest) {
    // The graph has 4 wavefronts. A min_subseq_len of 5 should prevent division.
    osp::RecursiveWavefrontDivider<Graph> divider;
    divider.UseLargestStepSplitter(0.5, 5);
    auto sections = divider.Divide(dag_);

    BOOST_REQUIRE_EQUAL(sections.size(), 1);
    BOOST_REQUIRE_EQUAL(sections[0].size(), 1);
}

BOOST_AUTO_TEST_CASE(MaxDepthTest) {
    // Setting max_depth to 0 should prevent any recursion.
    osp::RecursiveWavefrontDivider<Graph> divider;
    divider.UseLargestStepSplitter(0.5, 2).SetMaxDepth(0);
    auto sections = divider.Divide(dag_);

    BOOST_REQUIRE_EQUAL(sections.size(), 1);
    BOOST_REQUIRE_EQUAL(sections[0].size(), 1);
}

BOOST_AUTO_TEST_CASE(EmptyGraphTest) {
    osp::RecursiveWavefrontDivider<Graph> divider;
    Graph emptyDag;
    auto sections = divider.Divide(emptyDag);
    BOOST_CHECK(sections.empty());
}

BOOST_AUTO_TEST_SUITE_END()

// --- Test Fixture 2: A DAG with multiple merge points for deeper recursion ---
struct TestFixtureMultiMerge {
    Graph dag_;

    TestFixtureMultiMerge() {
        // Sequence: {4, 4, 2, 2, 1, 1}. Two significant drops.
        // L0: 4 comp -> L2: 2 comp (drop of 2)
        // L2: 2 comp -> L4: 1 comp (drop of 1)
        const auto vL01 = dag_.AddVertex(1, 1, 1), vL02 = dag_.AddVertex(1, 1, 1), vL03 = dag_.AddVertex(1, 1, 1),
                   vL04 = dag_.AddVertex(1, 1, 1);
        const auto vL11 = dag_.AddVertex(1, 1, 1), vL12 = dag_.AddVertex(1, 1, 1), vL13 = dag_.AddVertex(1, 1, 1),
                   vL14 = dag_.AddVertex(1, 1, 1);
        const auto vL21 = dag_.AddVertex(1, 1, 1), vL22 = dag_.AddVertex(1, 1, 1);
        const auto vL31 = dag_.AddVertex(1, 1, 1), vL32 = dag_.AddVertex(1, 1, 1);
        const auto vL41 = dag_.AddVertex(1, 1, 1);
        const auto vL51 = dag_.AddVertex(1, 1, 1);

        dag_.AddEdge(vL01, vL11);
        dag_.AddEdge(vL02, vL12);
        dag_.AddEdge(vL03, vL13);
        dag_.AddEdge(vL04, vL14);
        dag_.AddEdge(vL11, vL21);
        dag_.AddEdge(vL12, vL21);
        dag_.AddEdge(vL13, vL22);
        dag_.AddEdge(vL14, vL22);
        dag_.AddEdge(vL21, vL31);
        dag_.AddEdge(vL22, vL32);
        dag_.AddEdge(vL31, vL41);
        dag_.AddEdge(vL32, vL41);
        dag_.AddEdge(vL41, vL51);
    }
};

BOOST_FIXTURE_TEST_SUITE(multi_merge_tests, TestFixtureMultiMerge)

BOOST_AUTO_TEST_CASE(MultipleRecursionTest) {
    osp::RecursiveWavefrontDivider<Graph> divider;
    // Threshold is 0.5. First cut is for drop of 2.0 (4->2). Second is for drop of 1.0 (2->1).
    divider.UseLargestStepSplitter(0.5, 2);
    auto sections = divider.Divide(dag_);

    // Expect 3 sections:
    // 1. Levels 0-1 (before first major cut)
    // 2. Levels 2-3 (before second major cut)
    // 3. Levels 4-5 (the remainder)
    BOOST_REQUIRE_EQUAL(sections.size(), 3);

    // Section 1: levels 0-1. 4 components.
    BOOST_CHECK_EQUAL(sections[0].size(), 4);
    // Section 2: levels 2-3. 2 components.
    BOOST_CHECK_EQUAL(sections[1].size(), 2);
    // Section 3: levels 4-5. 1 component.
    BOOST_CHECK_EQUAL(sections[2].size(), 1);
}

BOOST_AUTO_TEST_CASE(VarianceSplitterTest) {
    // This test uses the same multi-merge graph but with the variance splitter.
    // The sequence {4,4,2,2,1,1} has high variance and should be split.
    osp::RecursiveWavefrontDivider<Graph> divider;
    // var_mult of 0.99 ensures any reduction is accepted.
    // var_threshold of 0.1 ensures we start splitting.
    divider.UseVarianceSplitter(0.99, 0.1, 2);
    auto sections = divider.Divide(dag_);

    // The variance splitter should also identify the two main merge points.
    BOOST_REQUIRE_EQUAL(sections.size(), 3);
}

BOOST_AUTO_TEST_SUITE_END()    // End of MultiMergeTests

BOOST_AUTO_TEST_SUITE_END()    // End of DagDividerTestSuite
