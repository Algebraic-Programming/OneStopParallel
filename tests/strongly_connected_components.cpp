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

#define BOOST_TEST_MODULE StronglyConnectedComponentsTest
#include <boost/test/unit_test.hpp>

#include "osp/graph_algorithms/strongly_connected_components.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

#include <algorithm>
#include <set>
#include <vector>

// Helper function to compare SCC results.
template <typename VertexType>
void check_sccs_equal(const std::vector<std::vector<VertexType>> &result,
                      const std::vector<std::vector<VertexType>> &expected) {
    auto to_set_of_sets = [](const std::vector<std::vector<VertexType>> &vec_of_vecs) {
        std::set<std::set<VertexType>> set_of_sets;
        for (const auto &inner_vec : vec_of_vecs) {
            set_of_sets.insert(std::set<VertexType>(inner_vec.begin(), inner_vec.end()));
        }
        return set_of_sets;
    };

    auto result_set = to_set_of_sets(result);
    auto expected_set = to_set_of_sets(expected);

    BOOST_CHECK(result_set == expected_set);
}

using graph = osp::computational_dag_edge_idx_vector_impl_def_int_t;
using VertexType = graph::vertex_idx;

BOOST_AUTO_TEST_SUITE(StronglyConnectedComponentsTestSuite)

BOOST_AUTO_TEST_CASE(EmptyGraphTest) {
    graph g;
    auto sccs = osp::strongly_connected_components(g);
    BOOST_CHECK(sccs.empty());
}

BOOST_AUTO_TEST_CASE(NoEdgesTest) {
    graph g;
    g.add_vertex(1,1,1);
    g.add_vertex(1,1,1);
    g.add_vertex(1,1,1);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {{0}, {1}, {2}};
    check_sccs_equal(sccs, expected);
}

BOOST_AUTO_TEST_CASE(LineGraphTest) {
    graph g;
    g.add_vertex(1,1,1); 
    g.add_vertex(1,1,1); 
    g.add_vertex(1,1,1); 
    g.add_vertex(1,1,1); 
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {{0}, {1}, {2}, {3}};
    check_sccs_equal(sccs, expected);
}

BOOST_AUTO_TEST_CASE(SimpleCycleTest) {
    graph g;
    g.add_vertex(1,1,1);
    g.add_vertex(1,1,1);
    g.add_vertex(1,1,1);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 0);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {{0, 1, 2}};
    check_sccs_equal(sccs, expected);
}

BOOST_AUTO_TEST_CASE(FullGraphIsSCCTest) {
    graph g;
    g.add_vertex(1,1,1);
    g.add_vertex(1,1,1);
    g.add_vertex(1,1,1);
    g.add_edge(0, 1);
    g.add_edge(1, 0);
    g.add_edge(1, 2);
    g.add_edge(2, 1);
    g.add_edge(0, 2);
    g.add_edge(2, 0);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {{0, 1, 2}};
    check_sccs_equal(sccs, expected);
}

BOOST_AUTO_TEST_CASE(MultipleSCCsTest) {
 
    graph g;
    for (int i = 0; i < 8; ++i)
        g.add_vertex(1,1,1); 

    g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(2, 0); // SCC {0,1,2}
    g.add_edge(3, 4); g.add_edge(4, 3); // SCC {3,4}
    g.add_edge(5, 6); g.add_edge(6, 5); // SCC {5,6}
    // SCC {7}

    g.add_edge(2, 3); g.add_edge(3, 5); g.add_edge(4, 6); g.add_edge(5, 7);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {{0, 1, 2}, {3, 4}, {5, 6}, {7}};
    check_sccs_equal(sccs, expected);
}

BOOST_AUTO_TEST_CASE(ComplexGraphFromPaperTest) {

    graph g;
    for (int i = 0; i < 8; ++i) g.add_vertex(1,1,1); 
    g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(1, 4); g.add_edge(1, 5);
    g.add_edge(2, 3); g.add_edge(2, 6); g.add_edge(3, 2); g.add_edge(3, 7);
    g.add_edge(4, 0); g.add_edge(4, 5); g.add_edge(5, 6); g.add_edge(6, 5);
    g.add_edge(7, 3); g.add_edge(7, 6);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {{0, 1, 4}, {2, 3, 7}, {5, 6}};
    check_sccs_equal(sccs, expected);
}

BOOST_AUTO_TEST_SUITE_END()