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

#define BOOST_TEST_MODULE ApproxEdgeReduction

#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

computational_dag_vector_impl_def_t ConstrGraph1() {
    computational_dag_vector_impl_def_t graph;

    using VertexIdx = computational_dag_vector_impl_def_t::vertex_idx;

    VertexIdx v1 = graph.add_vertex(1, 2, 3, 4);
    VertexIdx v2 = graph.add_vertex(5, 6, 7, 8);
    VertexIdx v3 = graph.add_vertex(9, 10, 11, 12);
    VertexIdx v4 = graph.add_vertex(13, 14, 15, 16);
    VertexIdx v5 = graph.add_vertex(17, 18, 19, 20);
    VertexIdx v6 = graph.add_vertex(21, 22, 23, 24);
    VertexIdx v7 = graph.add_vertex(25, 26, 27, 28);
    VertexIdx v8 = graph.add_vertex(29, 30, 31, 32);

    graph.add_edge(v1, v2);
    graph.add_edge(v1, v3);
    graph.add_edge(v1, v4);
    graph.add_edge(v2, v5);

    graph.add_edge(v3, v5);
    graph.add_edge(v3, v6);
    graph.add_edge(v2, v7);
    graph.add_edge(v5, v8);
    graph.add_edge(v4, v8);

    return graph;
}

BOOST_AUTO_TEST_CASE(TestUtil1) {
    const computational_dag_vector_impl_def_t graph = ConstrGraph1();

    // using vertex_idx = computational_dag_vector_impl_def_t::vertex_idx;
}

BOOST_AUTO_TEST_CASE(ComputationalDagConstructor) {
    using VertexType = vertex_idx_t<boost_graph_int_t>;

    const std::vector<std::vector<VertexType>> out({
        {7},
        {},
        {0},
        {2},
        {},
        {2, 0},
        {1, 2, 0},
        {},
        {4},
        {6, 1, 5}
    });
    const std::vector<int> workW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});
    const std::vector<int> commW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    const boost_graph_int_t graph(out, workW, commW);
    const boost_graph_int_t graphEmpty;

    std::vector<VertexType> topOrder;
    std::vector<size_t> indexInTopOrder;

    topOrder = GetTopOrderReverse(graph);

    BOOST_CHECK(topOrder.size() == graph.NumVertices());

    indexInTopOrder = sorting_arrangement(topOrder);

    for (const auto &i : topOrder) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_GT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    topOrder = GetTopOrderMaxChildren(graph);

    BOOST_CHECK(topOrder.size() == graph.NumVertices());

    indexInTopOrder = sorting_arrangement(topOrder);

    for (const auto &i : topOrder) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    topOrder = GetTopOrderRandom(graph);

    BOOST_CHECK(topOrder.size() == graph.NumVertices());

    indexInTopOrder = sorting_arrangement(topOrder);

    for (const auto &i : topOrder) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    topOrder = GetTopOrderMinIndex(graph);

    BOOST_CHECK(topOrder.size() == graph.NumVertices());

    indexInTopOrder = sorting_arrangement(topOrder);

    for (const auto &i : topOrder) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    topOrder = GetTopOrderGorder(graph);

    BOOST_CHECK(topOrder.size() == graph.NumVertices());

    indexInTopOrder = sorting_arrangement(topOrder);

    for (const auto &i : topOrder) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    topOrder = GetTopOrder(graph);

    BOOST_CHECK(topOrder.size() == graph.NumVertices());
    BOOST_CHECK(GetTopOrder(graphEmpty).size() == graphEmpty.NumVertices());

    indexInTopOrder = sorting_arrangement(topOrder);

    for (const auto &i : topOrder) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    size_t idx = 0;
    std::vector<VertexType> bfsViewTopSort;
    for (const auto &v : bfs_top_sort_view(graph)) {
        bfsViewTopSort.push_back(v);
        BOOST_CHECK_EQUAL(topOrder[idx], v);
        ++idx;
    }

    BOOST_CHECK_EQUAL(bfsViewTopSort.size(), graph.NumVertices());

    indexInTopOrder = sorting_arrangement(bfsViewTopSort);
    for (const auto &i : bfsViewTopSort) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    std::vector<VertexType> dfsViewTopSort;
    for (const auto &v : top_sort_view(graph)) {
        dfsViewTopSort.push_back(v);
    }

    BOOST_CHECK_EQUAL(dfsViewTopSort.size(), graph.NumVertices());

    indexInTopOrder = sorting_arrangement(dfsViewTopSort);
    for (const auto &i : dfsViewTopSort) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    BOOST_CHECK_EQUAL(dfsViewTopSort[0], 9);
    BOOST_CHECK_EQUAL(dfsViewTopSort[1], 5);
    BOOST_CHECK_EQUAL(dfsViewTopSort[2], 6);
    BOOST_CHECK_EQUAL(dfsViewTopSort[3], 1);
    BOOST_CHECK_EQUAL(dfsViewTopSort[4], 8);
    BOOST_CHECK_EQUAL(dfsViewTopSort[5], 4);
    BOOST_CHECK_EQUAL(dfsViewTopSort[6], 3);
    BOOST_CHECK_EQUAL(dfsViewTopSort[7], 2);
    BOOST_CHECK_EQUAL(dfsViewTopSort[8], 0);
    BOOST_CHECK_EQUAL(dfsViewTopSort[9], 7);

    std::vector<VertexType> locViewTopSort;

    for (const auto &v : locality_top_sort_view(graph)) {
        locViewTopSort.push_back(v);
    }

    BOOST_CHECK_EQUAL(locViewTopSort.size(), graph.NumVertices());

    indexInTopOrder = sorting_arrangement(locViewTopSort);
    for (const auto &i : locViewTopSort) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    BOOST_CHECK_EQUAL(locViewTopSort[0], 3);
    BOOST_CHECK_EQUAL(locViewTopSort[1], 8);
    BOOST_CHECK_EQUAL(locViewTopSort[2], 4);
    BOOST_CHECK_EQUAL(locViewTopSort[3], 9);
    BOOST_CHECK_EQUAL(locViewTopSort[4], 5);
    BOOST_CHECK_EQUAL(locViewTopSort[5], 6);
    BOOST_CHECK_EQUAL(locViewTopSort[6], 1);
    BOOST_CHECK_EQUAL(locViewTopSort[7], 2);
    BOOST_CHECK_EQUAL(locViewTopSort[8], 0);
    BOOST_CHECK_EQUAL(locViewTopSort[9], 7);

    std::vector<VertexType> maxChildrenViewTopSort;
    for (const auto &v : max_children_top_sort_view(graph)) {
        maxChildrenViewTopSort.push_back(v);
    }

    BOOST_CHECK_EQUAL(maxChildrenViewTopSort.size(), graph.NumVertices());

    indexInTopOrder = sorting_arrangement(maxChildrenViewTopSort);
    for (const auto &i : maxChildrenViewTopSort) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    BOOST_CHECK_EQUAL(maxChildrenViewTopSort[0], 9);
    BOOST_CHECK_EQUAL(maxChildrenViewTopSort[1], 6);
    BOOST_CHECK_EQUAL(maxChildrenViewTopSort[2], 5);
    BOOST_CHECK_EQUAL(maxChildrenViewTopSort[3], 3);
    BOOST_CHECK_EQUAL(maxChildrenViewTopSort[4], 2);
    BOOST_CHECK_EQUAL(maxChildrenViewTopSort[5], 0);
    BOOST_CHECK_EQUAL(maxChildrenViewTopSort[6], 8);
    BOOST_CHECK_EQUAL(maxChildrenViewTopSort[7], 1);
    BOOST_CHECK_EQUAL(maxChildrenViewTopSort[8], 4);
    BOOST_CHECK_EQUAL(maxChildrenViewTopSort[9], 7);

    std::vector<VertexType> randomViewTopSort;
    for (const auto &v : random_top_sort_view(graph)) {
        randomViewTopSort.push_back(v);
    }
    BOOST_CHECK_EQUAL(randomViewTopSort.size(), graph.NumVertices());

    indexInTopOrder = sorting_arrangement(randomViewTopSort);

    for (const auto &i : randomViewTopSort) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(TopSortTemplateOverloadCsr) {
    using VertexType = vertex_idx_t<boost_graph_int_t>;

    const std::vector<std::vector<VertexType>> out({
        {7},
        {},
        {0},
        {2},
        {},
        {2, 0},
        {1, 2, 0},
        {},
        {4},
        {6, 1, 5}
    });
    const std::vector<int> workW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});
    const std::vector<int> commW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    const boost_graph_int_t graph(out, workW, commW);

    CompactSparseGraph<false> graphCsr(graph);

    BOOST_CHECK_EQUAL(graphCsr.NumVertices(), 10);
    BOOST_CHECK_EQUAL(graphCsr.NumEdges(), 12);

    auto topOrder = GetTopOrder(graphCsr);
    BOOST_CHECK_EQUAL(topOrder.size(), graphCsr.NumVertices());

    std::vector<size_t> expectedTopOrder{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    size_t idx = 0;
    for (const auto &v : top_sort_view(graphCsr)) {
        BOOST_CHECK_EQUAL(topOrder[idx], v);
        BOOST_CHECK_EQUAL(expectedTopOrder[idx], v);
        ++idx;
    }
}
