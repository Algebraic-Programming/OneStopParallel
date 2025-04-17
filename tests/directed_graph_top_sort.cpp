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

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

#include "graph_algorithms/computational_dag_util.hpp"
#include "graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "graph_algorithms/directed_graph_path_util.hpp"
#include "graph_algorithms/directed_graph_top_sort.hpp"
#include "graph_algorithms/directed_graph_util.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

computational_dag_vector_impl_def_t constr_graph_1() {

    computational_dag_vector_impl_def_t graph;

    using vertex_idx = computational_dag_vector_impl_def_t::vertex_idx;

    vertex_idx v1 = graph.add_vertex(1, 2, 3, 4);
    vertex_idx v2 = graph.add_vertex(5, 6, 7, 8);
    vertex_idx v3 = graph.add_vertex(9, 10, 11, 12);
    vertex_idx v4 = graph.add_vertex(13, 14, 15, 16);
    vertex_idx v5 = graph.add_vertex(17, 18, 19, 20);
    vertex_idx v6 = graph.add_vertex(21, 22, 23, 24);
    vertex_idx v7 = graph.add_vertex(25, 26, 27, 28);
    vertex_idx v8 = graph.add_vertex(29, 30, 31, 32);

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
};

BOOST_AUTO_TEST_CASE(test_util_1) {

    const computational_dag_vector_impl_def_t graph = constr_graph_1();

    // using vertex_idx = computational_dag_vector_impl_def_t::vertex_idx;
};

BOOST_AUTO_TEST_CASE(ComputationalDagConstructor) {

    using VertexType = vertex_idx_t<boost_graph>;

    const std::vector<std::vector<VertexType>> out(

        {{7}, {}, {0}, {2}, {}, {2, 0}, {1, 2, 0}, {}, {4}, {6, 1, 5}}

    );
    const std::vector<int> workW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});
    const std::vector<int> commW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    const boost_graph graph(out, workW, commW);
    const boost_graph graph_empty;

    std::vector<VertexType> top_order = GetTopOrder(AS_IT_COMES, graph);
    BOOST_CHECK(top_order.size() == graph.num_vertices());
    BOOST_CHECK(GetTopOrder(AS_IT_COMES, graph_empty).size() == graph_empty.num_vertices());

    std::vector<size_t> index_in_top_order = sorting_arrangement(top_order);

    for (const auto &i : top_order) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    size_t idx = 0;
    std::vector<VertexType> bfs_view_top_sort;
    for (const auto &v : bfs_top_sort_view(graph)) {
        bfs_view_top_sort.push_back(v);
        BOOST_CHECK_EQUAL(top_order[idx], v);
        ++idx;
    }

    BOOST_CHECK_EQUAL(bfs_view_top_sort.size(), graph.num_vertices());

    index_in_top_order = sorting_arrangement(bfs_view_top_sort);
    for (const auto &i : bfs_view_top_sort) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    std::vector<VertexType> dfs_view_top_sort;
    for (const auto &v : dfs_top_sort_view(graph)) {
        dfs_view_top_sort.push_back(v);
    }

    BOOST_CHECK_EQUAL(dfs_view_top_sort.size(), graph.num_vertices());

    index_in_top_order = sorting_arrangement(dfs_view_top_sort);
    for (const auto &i : dfs_view_top_sort) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    BOOST_CHECK_EQUAL(dfs_view_top_sort[0], 9);
    BOOST_CHECK_EQUAL(dfs_view_top_sort[1], 5);
    BOOST_CHECK_EQUAL(dfs_view_top_sort[2], 6);
    BOOST_CHECK_EQUAL(dfs_view_top_sort[3], 1);
    BOOST_CHECK_EQUAL(dfs_view_top_sort[4], 8);
    BOOST_CHECK_EQUAL(dfs_view_top_sort[5], 4);
    BOOST_CHECK_EQUAL(dfs_view_top_sort[6], 3);
    BOOST_CHECK_EQUAL(dfs_view_top_sort[7], 2);
    BOOST_CHECK_EQUAL(dfs_view_top_sort[8], 0);
    BOOST_CHECK_EQUAL(dfs_view_top_sort[9], 7);

    std::vector<VertexType> loc_view_top_sort;

    for (const auto &v : locality_top_sort_view(graph)) {

        loc_view_top_sort.push_back(v);
    }

    BOOST_CHECK_EQUAL(loc_view_top_sort.size(), graph.num_vertices());

    index_in_top_order = sorting_arrangement(loc_view_top_sort);
    for (const auto &i : loc_view_top_sort) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    BOOST_CHECK_EQUAL(loc_view_top_sort[0], 3);
    BOOST_CHECK_EQUAL(loc_view_top_sort[1], 8);
    BOOST_CHECK_EQUAL(loc_view_top_sort[2], 4);
    BOOST_CHECK_EQUAL(loc_view_top_sort[3], 9);
    BOOST_CHECK_EQUAL(loc_view_top_sort[4], 5);
    BOOST_CHECK_EQUAL(loc_view_top_sort[5], 6);
    BOOST_CHECK_EQUAL(loc_view_top_sort[6], 1);
    BOOST_CHECK_EQUAL(loc_view_top_sort[7], 2);
    BOOST_CHECK_EQUAL(loc_view_top_sort[8], 0);
    BOOST_CHECK_EQUAL(loc_view_top_sort[9], 7);

    std::vector<VertexType> max_children_view_top_sort;
    for (const auto &v : max_children_top_sort_view(graph)) {
        max_children_view_top_sort.push_back(v);
    }

    BOOST_CHECK_EQUAL(max_children_view_top_sort.size(), graph.num_vertices());

    index_in_top_order = sorting_arrangement(max_children_view_top_sort);
    for (const auto &i : max_children_view_top_sort) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    std::cout << "max_children_view_top_sort: ";
    for (const auto &i : max_children_view_top_sort) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    BOOST_CHECK_EQUAL(max_children_view_top_sort[0], 9);
    BOOST_CHECK_EQUAL(max_children_view_top_sort[1], 6);
    BOOST_CHECK_EQUAL(max_children_view_top_sort[2], 5);
    BOOST_CHECK_EQUAL(max_children_view_top_sort[3], 3);
    BOOST_CHECK_EQUAL(max_children_view_top_sort[4], 2);
    BOOST_CHECK_EQUAL(max_children_view_top_sort[5], 0);
    BOOST_CHECK_EQUAL(max_children_view_top_sort[6], 8);
    BOOST_CHECK_EQUAL(max_children_view_top_sort[7], 1);
    BOOST_CHECK_EQUAL(max_children_view_top_sort[8], 4);
    BOOST_CHECK_EQUAL(max_children_view_top_sort[9], 7);

    std::vector<VertexType> random_view_top_sort;
    for (const auto &v : random_top_sort_view(graph)) {
        random_view_top_sort.push_back(v);
    }
    BOOST_CHECK_EQUAL(random_view_top_sort.size(), graph.num_vertices());

    index_in_top_order = sorting_arrangement(random_view_top_sort);

    for (const auto &i : random_view_top_sort) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }
}