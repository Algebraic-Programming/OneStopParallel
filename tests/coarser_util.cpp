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

#define BOOST_TEST_MODULE COARSER_UTIL_TEST
#include "osp/coarser/coarser_util.hpp"

#include <boost/test/unit_test.hpp>
#include <set>

#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"

using namespace osp;
using namespace osp::coarser_util;

using GraphType = CompactSparseGraph<true, true, true, true, true>;

BOOST_AUTO_TEST_CASE(ContractionMapValidity) {
    const std::vector<vertex_idx_t<GraphType>> contractionmap1 = {0, 1, 2, 3};
    BOOST_CHECK(check_valid_contraction_map<GraphType>(contractionmap1));

    const std::vector<vertex_idx_t<GraphType>> contractionmap2 = {1, 2, 3};
    BOOST_CHECK(not check_valid_contraction_map<GraphType>(contractionmap2));

    const std::vector<vertex_idx_t<GraphType>> contractionmap3 = {0, 1, 3, 4};
    BOOST_CHECK(not check_valid_contraction_map<GraphType>(contractionmap3));

    const std::vector<vertex_idx_t<GraphType>> contractionmap4 = {0, 1, 0, 1};
    BOOST_CHECK(check_valid_contraction_map<GraphType>(contractionmap4));

    const std::vector<vertex_idx_t<GraphType>> contractionmap5 = {2, 1, 2, 0, 1, 1};
    BOOST_CHECK(check_valid_contraction_map<GraphType>(contractionmap5));
}

BOOST_AUTO_TEST_CASE(ExpansionMapValidity) {
    const std::vector<std::vector<vertex_idx_t<GraphType>>> expansionmap1 = {{0}, {1}, {2}, {3}};
    BOOST_CHECK(check_valid_expansion_map<GraphType>(expansionmap1));

    const std::vector<std::vector<vertex_idx_t<GraphType>>> expansionmap2 = {{0}, {2}, {3}};
    BOOST_CHECK(not check_valid_expansion_map<GraphType>(expansionmap2));

    const std::vector<std::vector<vertex_idx_t<GraphType>>> expansionmap3 = {
        {0, 3}
    };
    BOOST_CHECK(not check_valid_expansion_map<GraphType>(expansionmap3));

    const std::vector<std::vector<vertex_idx_t<GraphType>>> expansionmap4 = {
        {0, 3},
        {2, 1, 4},
        {5}
    };
    BOOST_CHECK(check_valid_expansion_map<GraphType>(expansionmap4));

    const std::vector<std::vector<vertex_idx_t<GraphType>>> expansionmap5 = {{0}, {}, {2}, {3}, {1}};
    BOOST_CHECK(not check_valid_expansion_map<GraphType>(expansionmap5));
}

BOOST_AUTO_TEST_CASE(ContractionMapCoarsening) {
    std::set<std::pair<vertex_idx_t<GraphType>, vertex_idx_t<GraphType>>> edges({
        {0, 1},
        {1, 2}
    });
    GraphType graph(6, edges);

    GraphType coarseGraph1;

    std::vector<vertex_idx_t<GraphType>> contractionMap({0, 0, 1, 1, 2, 3});
    BOOST_CHECK(construct_coarse_dag(graph, coarseGraph1, contractionMap));
    BOOST_CHECK(contractionMap == std::vector<vertex_idx_t<GraphType>>({0, 0, 1, 1, 2, 3}));

    BOOST_CHECK_EQUAL(coarseGraph1.NumVertices(), 4);
    BOOST_CHECK_EQUAL(coarseGraph1.NumEdges(), 1);

    BOOST_CHECK_EQUAL(coarseGraph1.OutDegree(0), 1);
    BOOST_CHECK_EQUAL(coarseGraph1.OutDegree(1), 0);
    BOOST_CHECK_EQUAL(coarseGraph1.OutDegree(2), 0);

    BOOST_CHECK_EQUAL(coarseGraph1.InDegree(0), 0);
    BOOST_CHECK_EQUAL(coarseGraph1.InDegree(1), 1);
    BOOST_CHECK_EQUAL(coarseGraph1.InDegree(2), 0);

    for (const auto &vert : coarseGraph1.Children(0)) {
        BOOST_CHECK_EQUAL(vert, 1);
    }

    for (const auto &vert : coarseGraph1.Parents(1)) {
        BOOST_CHECK_EQUAL(vert, 0);
    }
}
