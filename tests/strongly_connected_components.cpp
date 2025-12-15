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
#include "osp/graph_algorithms/strongly_connected_components.hpp"

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <set>
#include <vector>

#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

// Helper function to compare SCC results.
template <typename VertexType>
void CheckSccsEqual(const std::vector<std::vector<VertexType>> &result, const std::vector<std::vector<VertexType>> &expected) {
    auto toSetOfSets = [](const std::vector<std::vector<VertexType>> &vecOfVecs) {
        std::set<std::set<VertexType>> setOfSets;
        for (const auto &innerVec : vecOfVecs) {
            setOfSets.insert(std::set<VertexType>(innerVec.begin(), innerVec.end()));
        }
        return setOfSets;
    };

    auto resultSet = toSetOfSets(result);
    auto expectedSet = toSetOfSets(expected);

    BOOST_CHECK(resultSet == expectedSet);
}

using Graph = osp::computational_dag_edge_idx_vector_impl_def_int_t;
using VertexType = Graph::VertexIdx;

BOOST_AUTO_TEST_SUITE(strongly_connected_components_test_suite)

BOOST_AUTO_TEST_CASE(EmptyGraphTest) {
    Graph g;
    auto sccs = osp::strongly_connected_components(g);
    BOOST_CHECK(sccs.empty());
}

BOOST_AUTO_TEST_CASE(NoEdgesTest) {
    Graph g;
    g.AddVertex(1, 1, 1);
    g.AddVertex(1, 1, 1);
    g.AddVertex(1, 1, 1);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {{0}, {1}, {2}};
    CheckSccsEqual(sccs, expected);
}

BOOST_AUTO_TEST_CASE(LineGraphTest) {
    Graph g;
    g.AddVertex(1, 1, 1);
    g.AddVertex(1, 1, 1);
    g.AddVertex(1, 1, 1);
    g.AddVertex(1, 1, 1);
    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    g.AddEdge(2, 3);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {{0}, {1}, {2}, {3}};
    CheckSccsEqual(sccs, expected);
}

BOOST_AUTO_TEST_CASE(SimpleCycleTest) {
    Graph g;
    g.AddVertex(1, 1, 1);
    g.AddVertex(1, 1, 1);
    g.AddVertex(1, 1, 1);
    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    g.AddEdge(2, 0);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {
        {0, 1, 2}
    };
    CheckSccsEqual(sccs, expected);
}

BOOST_AUTO_TEST_CASE(FullGraphIsSCCTest) {
    Graph g;
    g.AddVertex(1, 1, 1);
    g.AddVertex(1, 1, 1);
    g.AddVertex(1, 1, 1);
    g.AddEdge(0, 1);
    g.AddEdge(1, 0);
    g.AddEdge(1, 2);
    g.AddEdge(2, 1);
    g.AddEdge(0, 2);
    g.AddEdge(2, 0);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {
        {0, 1, 2}
    };
    CheckSccsEqual(sccs, expected);
}

BOOST_AUTO_TEST_CASE(MultipleSCCsTest) {
    Graph g;
    for (int i = 0; i < 8; ++i) {
        g.AddVertex(1, 1, 1);
    }

    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    g.AddEdge(2, 0);    // SCC {0,1,2}
    g.AddEdge(3, 4);
    g.AddEdge(4, 3);    // SCC {3,4}
    g.AddEdge(5, 6);
    g.AddEdge(6, 5);    // SCC {5,6}
    // SCC {7}

    g.AddEdge(2, 3);
    g.AddEdge(3, 5);
    g.AddEdge(4, 6);
    g.AddEdge(5, 7);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {
        {0, 1, 2},
        {3, 4},
        {5, 6},
        {7}
    };
    CheckSccsEqual(sccs, expected);
}

BOOST_AUTO_TEST_CASE(ComplexGraphFromPaperTest) {
    Graph g;
    for (int i = 0; i < 8; ++i) {
        g.AddVertex(1, 1, 1);
    }
    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    g.AddEdge(1, 4);
    g.AddEdge(1, 5);
    g.AddEdge(2, 3);
    g.AddEdge(2, 6);
    g.AddEdge(3, 2);
    g.AddEdge(3, 7);
    g.AddEdge(4, 0);
    g.AddEdge(4, 5);
    g.AddEdge(5, 6);
    g.AddEdge(6, 5);
    g.AddEdge(7, 3);
    g.AddEdge(7, 6);

    auto sccs = osp::strongly_connected_components(g);
    std::vector<std::vector<VertexType>> expected = {
        {0, 1, 4},
        {2, 3, 7},
        {5, 6}
    };
    CheckSccsEqual(sccs, expected);
}

BOOST_AUTO_TEST_SUITE_END()
