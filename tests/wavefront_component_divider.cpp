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
#include "osp/dag_divider/wavefront_divider/SequenceSplitter.hpp" 

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