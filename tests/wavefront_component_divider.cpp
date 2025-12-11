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
    std::vector<size_t> splits1 = splitter.split(seq1);
    std::vector<size_t> expected1 = {4};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits1.begin(), splits1.end(), expected1.begin(), expected1.end());

    // Test case 2: No split needed (low variance)
    std::vector<double> seq2 = {1, 1.1, 1.2, 1.1, 1.3};
    std::vector<size_t> splits2 = splitter.split(seq2);
    BOOST_CHECK(splits2.empty());

    // Test case 3: Empty sequence
    std::vector<double> seq3 = {};
    std::vector<size_t> splits3 = splitter.split(seq3);
    BOOST_CHECK(splits3.empty());

    // Test case 4: Single element sequence
    std::vector<double> seq4 = {100.0};
    std::vector<size_t> splits4 = splitter.split(seq4);
    BOOST_CHECK(splits4.empty());

    // Test case 5: Multiple splits
    std::vector<double> seq5 = {1, 1, 1, 20, 20, 20, 1, 1, 1};
    std::vector<size_t> splits5 = splitter.split(seq5);
    std::vector<size_t> expected5 = {3, 6};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits5.begin(), splits5.end(), expected5.begin(), expected5.end());
}

BOOST_AUTO_TEST_CASE(LargestStepSplitterTest) {
    osp::LargestStepSplitter splitter(5.0, 2);

    // Test case 1: Clear step
    std::vector<double> seq1 = {1, 2, 3, 10, 11, 12};
    std::vector<size_t> splits1 = splitter.split(seq1);
    std::vector<size_t> expected1 = {3};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits1.begin(), splits1.end(), expected1.begin(), expected1.end());

    // Test case 2: No significant step
    std::vector<double> seq2 = {1, 2, 3, 4, 5, 6};
    std::vector<size_t> splits2 = splitter.split(seq2);
    BOOST_CHECK(splits2.empty());

    // Test case 3: Decreasing sequence
    std::vector<double> seq3 = {12, 11, 10, 3, 2, 1};
    std::vector<size_t> splits3 = splitter.split(seq3);
    std::vector<size_t> expected3 = {3};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits3.begin(), splits3.end(), expected3.begin(), expected3.end());

    // Test case 4: Sequence too short
    std::vector<double> seq4 = {1, 10};
    std::vector<size_t> splits4 = splitter.split(seq4);
    BOOST_CHECK(splits4.empty());

    // Test case 5: Multiple large steps
    std::vector<double> seq5 = {0, 1, 10, 11, 20, 21};
    std::vector<size_t> splits5 = splitter.split(seq5);
    std::vector<size_t> expected5 = {2, 4};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits5.begin(), splits5.end(), expected5.begin(), expected5.end());
}

BOOST_AUTO_TEST_CASE(ThresholdScanSplitterTest) {
    osp::ThresholdScanSplitter splitter(5.0, 10.0);

    // Test case 1: Significant drop
    std::vector<double> seq1 = {20, 18, 16, 9, 8, 7};
    std::vector<size_t> splits1 = splitter.split(seq1);
    std::vector<size_t> expected1 = {3};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits1.begin(), splits1.end(), expected1.begin(), expected1.end());

    // Test case 2: Crossing absolute threshold (rising)
    std::vector<double> seq2 = {5, 7, 9, 11, 13};
    std::vector<size_t> splits2 = splitter.split(seq2);
    std::vector<size_t> expected2 = {3};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits2.begin(), splits2.end(), expected2.begin(), expected2.end());

    // Test case 3: Crossing absolute threshold (dropping)
    std::vector<double> seq3 = {15, 12, 11, 9, 8};
    std::vector<size_t> splits3 = splitter.split(seq3);
    std::vector<size_t> expected3 = {3};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits3.begin(), splits3.end(), expected3.begin(), expected3.end());

    // Test case 4: No splits
    std::vector<double> seq4 = {1, 2, 3, 4, 5};
    std::vector<size_t> splits4 = splitter.split(seq4);
    BOOST_CHECK(splits4.empty());

    // Test case 5: Empty sequence
    std::vector<double> seq5 = {};
    std::vector<size_t> splits5 = splitter.split(seq5);
    BOOST_CHECK(splits5.empty());
}

using graph = osp::computational_dag_edge_idx_vector_impl_def_int_t;
using VertexType = graph::vertex_idx;

BOOST_AUTO_TEST_CASE(ForwardAndBackwardPassTest) {
    graph dag;
    const auto v1 = dag.add_vertex(2, 1, 9);
    const auto v2 = dag.add_vertex(3, 1, 8);
    const auto v3 = dag.add_vertex(4, 1, 7);
    const auto v4 = dag.add_vertex(5, 1, 6);
    const auto v5 = dag.add_vertex(6, 1, 5);
    const auto v6 = dag.add_vertex(7, 1, 4);
    const auto v7 = dag.add_vertex(8, 1, 3);    // Note: v7 is not connected in the example
    const auto v8 = dag.add_vertex(9, 1, 2);

    dag.add_edge(v1, v2);
    dag.add_edge(v1, v3);
    dag.add_edge(v1, v4);
    dag.add_edge(v2, v5);
    dag.add_edge(v2, v6);
    dag.add_edge(v3, v5);
    dag.add_edge(v3, v6);
    dag.add_edge(v5, v8);
    dag.add_edge(v4, v8);

    // Manually defined level sets for this DAG
    const std::vector<std::vector<VertexType>> level_sets = {
        {v1}, // Level 0
        {v2, v3, v4}, // Level 1
        {v5, v6}, // Level 2
        {v8}, // Level 3
        {v7}  // Level 4 (isolated vertex)
    };

    osp::WavefrontStatisticsCollector<graph> collector(dag, level_sets);

    // --- Test Forward Pass ---
    auto forward_stats = collector.compute_forward();
    BOOST_REQUIRE_EQUAL(forward_stats.size(), 5);

    // Level 0
    BOOST_CHECK_EQUAL(forward_stats[0].connected_components_vertices.size(), 1);
    BOOST_CHECK_EQUAL(forward_stats[0].connected_components_weights[0], 2);
    BOOST_CHECK_EQUAL(forward_stats[0].connected_components_memories[0], 9);

    // Level 1
    BOOST_CHECK_EQUAL(forward_stats[1].connected_components_vertices.size(), 1);
    BOOST_CHECK_EQUAL(forward_stats[1].connected_components_weights[0], 2 + 3 + 4 + 5);    // v1,v2,v3,v4
    BOOST_CHECK_EQUAL(forward_stats[1].connected_components_memories[0], 9 + 8 + 7 + 6);

    // Level 2
    BOOST_CHECK_EQUAL(forward_stats[2].connected_components_vertices.size(), 1);
    BOOST_CHECK_EQUAL(forward_stats[2].connected_components_weights[0], 14 + 6 + 7);    // v1-v6
    BOOST_CHECK_EQUAL(forward_stats[2].connected_components_memories[0], 30 + 5 + 4);

    // Level 3
    BOOST_CHECK_EQUAL(forward_stats[3].connected_components_vertices.size(), 1);
    BOOST_CHECK_EQUAL(forward_stats[3].connected_components_weights[0], 27 + 9);    // v1-v6, v8
    BOOST_CHECK_EQUAL(forward_stats[3].connected_components_memories[0], 39 + 2);

    // Level 4 (isolated vertex shows up as a new component)
    BOOST_CHECK_EQUAL(forward_stats[4].connected_components_vertices.size(), 2);

    // --- Test Backward Pass ---
    auto backward_stats = collector.compute_backward();
    BOOST_REQUIRE_EQUAL(backward_stats.size(), 5);

    // Level 4
    BOOST_CHECK_EQUAL(backward_stats[4].connected_components_vertices.size(), 1);
    BOOST_CHECK_EQUAL(backward_stats[4].connected_components_weights[0], 8);    // v7
    BOOST_CHECK_EQUAL(backward_stats[4].connected_components_memories[0], 3);

    // Level 3
    BOOST_CHECK_EQUAL(backward_stats[3].connected_components_vertices.size(), 2);    // {v8}, {v7}

    // Level 2
    BOOST_CHECK_EQUAL(backward_stats[2].connected_components_vertices.size(), 3);    // {v5,v8}, {v6}, {v7}

    // Level 1
    BOOST_CHECK_EQUAL(backward_stats[1].connected_components_vertices.size(), 2);    // {v2,v3,v4,v5,v6,v8}, {v7}

    // Level 0
    BOOST_CHECK_EQUAL(backward_stats[0].connected_components_vertices.size(), 2);    // {v1-v6,v8}, {v7}
}

BOOST_AUTO_TEST_CASE(SequenceGenerationTest) {
    // --- Test Setup ---
    graph dag;
    const auto v1 = dag.add_vertex(2, 1, 9);
    const auto v2 = dag.add_vertex(3, 1, 8);
    const auto v3 = dag.add_vertex(4, 1, 7);
    const auto v4 = dag.add_vertex(5, 1, 6);
    const auto v5 = dag.add_vertex(6, 1, 5);
    const auto v6 = dag.add_vertex(7, 1, 4);
    const auto v7 = dag.add_vertex(8, 1, 3);    // Isolated vertex
    const auto v8 = dag.add_vertex(9, 1, 2);

    dag.add_edge(v1, v2);
    dag.add_edge(v1, v3);
    dag.add_edge(v1, v4);
    dag.add_edge(v2, v5);
    dag.add_edge(v2, v6);
    dag.add_edge(v3, v5);
    dag.add_edge(v3, v6);
    dag.add_edge(v5, v8);
    dag.add_edge(v4, v8);

    const std::vector<std::vector<VertexType>> level_sets = {
        {v1},
        {v2, v3, v4},
        {v5, v6},
        {v8},
        {v7}
    };

    osp::SequenceGenerator<graph> generator(dag, level_sets);

    // --- Test Component Count ---
    auto component_seq = generator.generate(osp::SequenceMetric::COMPONENT_COUNT);
    std::vector<double> expected_components = {1.0, 1.0, 1.0, 1.0, 2.0};
    BOOST_CHECK_EQUAL_COLLECTIONS(
        component_seq.begin(), component_seq.end(), expected_components.begin(), expected_components.end());

    // --- Test Available Parallelism ---
    auto parallelism_seq = generator.generate(osp::SequenceMetric::AVAILABLE_PARALLELISM);

    // Manual calculation for expected values:
    // L0: 2 / 1 = 2
    // L1: (2 + 3+4+5) / 2 = 14 / 2 = 7
    // L2: (14 + 6+7) / 3 = 27 / 3 = 9
    // L3: (27 + 9) / 4 = 36 / 4 = 9
    // L4: (36 + 8) / 5 = 44 / 5 = 8.8
    std::vector<double> expected_parallelism = {2.0, 7.0, 9.0, 9.0, 8.8};

    BOOST_REQUIRE_EQUAL(parallelism_seq.size(), expected_parallelism.size());
    for (size_t i = 0; i < parallelism_seq.size(); ++i) {
        BOOST_CHECK_CLOSE(parallelism_seq[i], expected_parallelism[i], 1e-9);
    }
}

struct TestFixture {
    graph dag;
    std::vector<VertexType> vertices;

    TestFixture() {
        // --- Test Setup ---
        // Note: The compute_wavefronts method will determine the levels.
        // The actual level sets for this graph are:
        // Level 0: {v1, v7}
        // Level 1: {v2, v3, v4}
        // Level 2: {v5, v6}
        // Level 3: {v8}
        const auto v1 = dag.add_vertex(2, 1, 9);
        const auto v2 = dag.add_vertex(3, 1, 8);
        const auto v3 = dag.add_vertex(4, 1, 7);
        const auto v4 = dag.add_vertex(5, 1, 6);
        const auto v5 = dag.add_vertex(6, 1, 5);
        const auto v6 = dag.add_vertex(7, 1, 4);
        const auto v7 = dag.add_vertex(8, 1, 3);    // Isolated vertex
        const auto v8 = dag.add_vertex(9, 1, 2);

        vertices = {v1, v2, v3, v4, v5, v6, v7, v8};

        dag.add_edge(v1, v2);
        dag.add_edge(v1, v3);
        dag.add_edge(v1, v4);
        dag.add_edge(v2, v5);
        dag.add_edge(v2, v6);
        dag.add_edge(v3, v5);
        dag.add_edge(v3, v6);
        dag.add_edge(v5, v8);
        dag.add_edge(v4, v8);
    }
};

BOOST_FIXTURE_TEST_SUITE(ScanWavefrontDividerTestSuite, TestFixture)

BOOST_AUTO_TEST_CASE(LargestStepDivisionTest) {
    osp::ScanWavefrontDivider<graph> divider;
    divider.set_metric(osp::SequenceMetric::AVAILABLE_PARALLELISM);
    divider.use_largest_step_splitter(0.9, 1);

    auto sections = divider.divide(dag);

    // Expecting a cut after level 0. This results in 2 sections.
    BOOST_REQUIRE_EQUAL(sections.size(), 2);

    // Section 1: level 0. Components: {v1}, {v7}
    BOOST_REQUIRE_EQUAL(sections[0].size(), 2);

    // Section 2: levels 1, 2, 3. The rest of the main component.
    BOOST_REQUIRE_EQUAL(sections[1].size(), 1);
    BOOST_CHECK_EQUAL(sections[1][0].size(), 6);    // v2,v3,v4,v5,v6,v8
}

BOOST_AUTO_TEST_CASE(ThresholdScanDivisionTest) {
    osp::ScanWavefrontDivider<graph> divider;
    divider.set_metric(osp::SequenceMetric::AVAILABLE_PARALLELISM);
    divider.use_threshold_scan_splitter(2.0, 11.5);

    auto sections = divider.divide(dag);

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
    osp::ScanWavefrontDivider<graph> divider;
    divider.set_metric(osp::SequenceMetric::COMPONENT_COUNT);
    divider.use_largest_step_splitter(2.0, 2);

    auto sections = divider.divide(dag);

    // Expecting a single section containing all components
    BOOST_REQUIRE_EQUAL(sections.size(), 1);
    BOOST_REQUIRE_EQUAL(sections[0].size(), 2);    // Two final components
}

BOOST_AUTO_TEST_CASE(EmptyGraphTest) {
    osp::ScanWavefrontDivider<graph> divider;
    graph empty_dag;
    auto sections = divider.divide(empty_dag);
    BOOST_CHECK(sections.empty());
}

BOOST_AUTO_TEST_SUITE_END()

struct TestFixture_2 {
    graph dag;
    std::vector<VertexType> vertices;

    TestFixture_2() {
        // --- Test Setup ---
        // This graph is designed to have a component count sequence of {2, 2, 2, 1}
        // to properly test the recursive divider's splitting logic.
        // Levels: {v1,v2}, {v3,v4}, {v5,v6}, {v7}
        const auto v1 = dag.add_vertex(1, 1, 1);
        const auto v2 = dag.add_vertex(1, 1, 1);
        const auto v3 = dag.add_vertex(1, 1, 1);
        const auto v4 = dag.add_vertex(1, 1, 1);
        const auto v5 = dag.add_vertex(1, 1, 1);
        const auto v6 = dag.add_vertex(1, 1, 1);
        const auto v7 = dag.add_vertex(1, 1, 1);

        vertices = {v1, v2, v3, v4, v5, v6, v7};

        dag.add_edge(v1, v3);
        dag.add_edge(v2, v4);
        dag.add_edge(v3, v5);
        dag.add_edge(v4, v6);
        dag.add_edge(v5, v7);
        dag.add_edge(v6, v7);
    }
};

BOOST_AUTO_TEST_SUITE(RecursiveWavefrontDividerTestSuite)

// --- Test Fixture 1: A simple DAG that merges from 2 components to 1 ---
struct TestFixture_SimpleMerge {
    graph dag;

    TestFixture_SimpleMerge() {
        // This graph is designed to have a component count sequence of {2, 2, 2, 1}
        // Levels: {v0,v1}, {v2,v3}, {v4,v5}, {v6}
        const auto v0 = dag.add_vertex(1, 1, 1);
        const auto v1 = dag.add_vertex(1, 1, 1);
        const auto v2 = dag.add_vertex(1, 1, 1);
        const auto v3 = dag.add_vertex(1, 1, 1);
        const auto v4 = dag.add_vertex(1, 1, 1);
        const auto v5 = dag.add_vertex(1, 1, 1);
        const auto v6 = dag.add_vertex(1, 1, 1);

        dag.add_edge(v0, v2);
        dag.add_edge(v1, v3);
        dag.add_edge(v2, v4);
        dag.add_edge(v3, v5);
        dag.add_edge(v4, v6);
        dag.add_edge(v5, v6);
    }
};

BOOST_FIXTURE_TEST_SUITE(SimpleMergeTests, TestFixture_SimpleMerge)

BOOST_AUTO_TEST_CASE(BasicRecursionTest) {
    osp::RecursiveWavefrontDivider<graph> divider;
    divider.use_largest_step_splitter(0.5, 1);
    auto sections = divider.divide(dag);

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
    osp::RecursiveWavefrontDivider<graph> divider;
    divider.use_largest_step_splitter(2.0, 2);
    auto sections = divider.divide(dag);

    // Expecting a single section containing all components, which merge into one.
    BOOST_REQUIRE_EQUAL(sections.size(), 1);
    BOOST_REQUIRE_EQUAL(sections[0].size(), 1);
}

BOOST_AUTO_TEST_CASE(MinSubsequenceLengthTest) {
    // The graph has 4 wavefronts. A min_subseq_len of 5 should prevent division.
    osp::RecursiveWavefrontDivider<graph> divider;
    divider.use_largest_step_splitter(0.5, 5);
    auto sections = divider.divide(dag);

    BOOST_REQUIRE_EQUAL(sections.size(), 1);
    BOOST_REQUIRE_EQUAL(sections[0].size(), 1);
}

BOOST_AUTO_TEST_CASE(MaxDepthTest) {
    // Setting max_depth to 0 should prevent any recursion.
    osp::RecursiveWavefrontDivider<graph> divider;
    divider.use_largest_step_splitter(0.5, 2).set_max_depth(0);
    auto sections = divider.divide(dag);

    BOOST_REQUIRE_EQUAL(sections.size(), 1);
    BOOST_REQUIRE_EQUAL(sections[0].size(), 1);
}

BOOST_AUTO_TEST_CASE(EmptyGraphTest) {
    osp::RecursiveWavefrontDivider<graph> divider;
    graph empty_dag;
    auto sections = divider.divide(empty_dag);
    BOOST_CHECK(sections.empty());
}

BOOST_AUTO_TEST_SUITE_END()

// --- Test Fixture 2: A DAG with multiple merge points for deeper recursion ---
struct TestFixture_MultiMerge {
    graph dag;

    TestFixture_MultiMerge() {
        // Sequence: {4, 4, 2, 2, 1, 1}. Two significant drops.
        // L0: 4 comp -> L2: 2 comp (drop of 2)
        // L2: 2 comp -> L4: 1 comp (drop of 1)
        const auto v_l0_1 = dag.add_vertex(1, 1, 1), v_l0_2 = dag.add_vertex(1, 1, 1), v_l0_3 = dag.add_vertex(1, 1, 1),
                   v_l0_4 = dag.add_vertex(1, 1, 1);
        const auto v_l1_1 = dag.add_vertex(1, 1, 1), v_l1_2 = dag.add_vertex(1, 1, 1), v_l1_3 = dag.add_vertex(1, 1, 1),
                   v_l1_4 = dag.add_vertex(1, 1, 1);
        const auto v_l2_1 = dag.add_vertex(1, 1, 1), v_l2_2 = dag.add_vertex(1, 1, 1);
        const auto v_l3_1 = dag.add_vertex(1, 1, 1), v_l3_2 = dag.add_vertex(1, 1, 1);
        const auto v_l4_1 = dag.add_vertex(1, 1, 1);
        const auto v_l5_1 = dag.add_vertex(1, 1, 1);

        dag.add_edge(v_l0_1, v_l1_1);
        dag.add_edge(v_l0_2, v_l1_2);
        dag.add_edge(v_l0_3, v_l1_3);
        dag.add_edge(v_l0_4, v_l1_4);
        dag.add_edge(v_l1_1, v_l2_1);
        dag.add_edge(v_l1_2, v_l2_1);
        dag.add_edge(v_l1_3, v_l2_2);
        dag.add_edge(v_l1_4, v_l2_2);
        dag.add_edge(v_l2_1, v_l3_1);
        dag.add_edge(v_l2_2, v_l3_2);
        dag.add_edge(v_l3_1, v_l4_1);
        dag.add_edge(v_l3_2, v_l4_1);
        dag.add_edge(v_l4_1, v_l5_1);
    }
};

BOOST_FIXTURE_TEST_SUITE(MultiMergeTests, TestFixture_MultiMerge)

BOOST_AUTO_TEST_CASE(MultipleRecursionTest) {
    osp::RecursiveWavefrontDivider<graph> divider;
    // Threshold is 0.5. First cut is for drop of 2.0 (4->2). Second is for drop of 1.0 (2->1).
    divider.use_largest_step_splitter(0.5, 2);
    auto sections = divider.divide(dag);

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
    osp::RecursiveWavefrontDivider<graph> divider;
    // var_mult of 0.99 ensures any reduction is accepted.
    // var_threshold of 0.1 ensures we start splitting.
    divider.use_variance_splitter(0.99, 0.1, 2);
    auto sections = divider.divide(dag);

    // The variance splitter should also identify the two main merge points.
    BOOST_REQUIRE_EQUAL(sections.size(), 3);
}

BOOST_AUTO_TEST_SUITE_END()    // End of MultiMergeTests

BOOST_AUTO_TEST_SUITE_END()    // End of DagDividerTestSuite
