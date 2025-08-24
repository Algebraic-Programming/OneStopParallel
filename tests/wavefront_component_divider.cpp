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
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/dag_divider/wavefront_divider/SequenceSplitter.hpp" 
#include "osp/dag_divider/wavefront_divider/WavefrontStatisticsCollector.hpp"


BOOST_AUTO_TEST_CASE(VarianceSplitterTest) {
    osp::VarianceSplitter splitter(0.5, 0.1);

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
    std::vector<double> seq5 = {0, 0, 0, 10, 10, 10, 0, 0, 0};
    std::vector<size_t> splits5 = splitter.split(seq5);
    std::vector<size_t> expected5 = {3, 6};
    BOOST_CHECK_EQUAL_COLLECTIONS(splits5.begin(), splits5.end(), expected5.begin(), expected5.end());
}

BOOST_AUTO_TEST_CASE(LargestStepSplitterTest) {
    osp::LargestStepSplitter splitter(5.0, 3);

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
    const auto v7 = dag.add_vertex(8, 1, 3); // Note: v7 is not connected in the example
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
        {v1},       // Level 0
        {v2, v3, v4}, // Level 1
        {v5, v6},   // Level 2
        {v8},       // Level 3
        {v7}        // Level 4 (isolated vertex)
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
    BOOST_CHECK_EQUAL(forward_stats[1].connected_components_weights[0], 2 + 3 + 4 + 5); // v1,v2,v3,v4
    BOOST_CHECK_EQUAL(forward_stats[1].connected_components_memories[0], 9 + 8 + 7 + 6);

    // Level 2
    BOOST_CHECK_EQUAL(forward_stats[2].connected_components_vertices.size(), 1);
    BOOST_CHECK_EQUAL(forward_stats[2].connected_components_weights[0], 14 + 6 + 7); // v1-v6
    BOOST_CHECK_EQUAL(forward_stats[2].connected_components_memories[0], 30 + 5 + 4);

    // Level 3
    BOOST_CHECK_EQUAL(forward_stats[3].connected_components_vertices.size(), 1);
    BOOST_CHECK_EQUAL(forward_stats[3].connected_components_weights[0], 27 + 9); // v1-v6, v8
    BOOST_CHECK_EQUAL(forward_stats[3].connected_components_memories[0], 39 + 2);

    // Level 4 (isolated vertex shows up as a new component)
    BOOST_CHECK_EQUAL(forward_stats[4].connected_components_vertices.size(), 2);


    // --- Test Backward Pass ---
    auto backward_stats = collector.compute_backward();
    BOOST_REQUIRE_EQUAL(backward_stats.size(), 5);

    // Level 4
    BOOST_CHECK_EQUAL(backward_stats[4].connected_components_vertices.size(), 1);
    BOOST_CHECK_EQUAL(backward_stats[4].connected_components_weights[0], 8); // v7
    BOOST_CHECK_EQUAL(backward_stats[4].connected_components_memories[0], 3);

    // Level 3
    BOOST_CHECK_EQUAL(backward_stats[3].connected_components_vertices.size(), 2); // {v8}, {v7}

    // Level 2
    BOOST_CHECK_EQUAL(backward_stats[2].connected_components_vertices.size(), 3); // {v5,v8}, {v6}, {v7}

    // Level 1
    BOOST_CHECK_EQUAL(backward_stats[1].connected_components_vertices.size(), 2); // {v2,v3,v4,v5,v6,v8}, {v7}

    // Level 0
    BOOST_CHECK_EQUAL(backward_stats[0].connected_components_vertices.size(), 2); // {v1-v6,v8}, {v7}
}