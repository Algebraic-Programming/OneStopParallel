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

BOOST_AUTO_TEST_CASE(test_empty_dag) {

    computational_dag_vector_impl_def_t graph;
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 0);

    size_t idx = 0;
    for (const auto &v : graph.vertices()) {
        graph.in_degree(v);
        idx++;
    }
    BOOST_CHECK_EQUAL(idx, 0);
};

BOOST_AUTO_TEST_CASE(test_dag) {

    const computational_dag_vector_impl_def_t graph = constr_graph_1();

    using vertex_idx = computational_dag_vector_impl_def_t::vertex_idx;

    BOOST_CHECK_EQUAL(graph.num_edges(), 9);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 8);

    std::vector<vertex_idx> vertices{0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<std::vector<vertex_idx>> out_neighbors{{1, 2, 3}, {4, 6}, {4, 5}, {7}, {7}, {}, {}, {}};

    std::vector<std::vector<vertex_idx>> in_neighbors{{}, {0}, {0}, {0}, {1, 2}, {2}, {1}, {4, 3}};

    size_t idx = 0;

    for (const auto &v : graph.vertices()) {

        BOOST_CHECK_EQUAL(v, vertices[idx++]);

        size_t i = 0;
        for (const auto &e : graph.children(v)) {
            BOOST_CHECK_EQUAL(e, out_neighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.parents(v)) {
            BOOST_CHECK_EQUAL(e, in_neighbors[v][i++]);
        }

        BOOST_CHECK_EQUAL(graph.in_degree(v), in_neighbors[v].size());
        BOOST_CHECK_EQUAL(graph.out_degree(v), out_neighbors[v].size());
    }
};

BOOST_AUTO_TEST_CASE(test_constr_dag) {

    computational_dag_vector_impl_def_int_t graph;

    graph.add_vertex(1, 2, 3);
    graph.add_vertex(5, 6, 7);
    graph.add_vertex(9, 10, 11);
    graph.add_vertex(13, 14, 15);

    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(0, 3);

    computational_dag_vector_impl_def_int_t graph_2(graph);

    BOOST_CHECK_EQUAL(graph_2.num_edges(), 3);
    BOOST_CHECK_EQUAL(graph_2.num_vertices(), 4);
    BOOST_CHECK_EQUAL(graph_2.vertex_work_weight(0), 1);
    BOOST_CHECK_EQUAL(graph_2.vertex_comm_weight(0), 2);
    BOOST_CHECK_EQUAL(graph_2.vertex_mem_weight(0), 3);
    BOOST_CHECK_EQUAL(graph_2.vertex_work_weight(1), 5);
    BOOST_CHECK_EQUAL(graph_2.vertex_comm_weight(1), 6);
    BOOST_CHECK_EQUAL(graph_2.vertex_mem_weight(1), 7);
    BOOST_CHECK_EQUAL(graph_2.vertex_work_weight(2), 9);
    BOOST_CHECK_EQUAL(graph_2.vertex_comm_weight(2), 10);
    BOOST_CHECK_EQUAL(graph_2.vertex_mem_weight(2), 11);
    BOOST_CHECK_EQUAL(graph_2.vertex_work_weight(3), 13);
    BOOST_CHECK_EQUAL(graph_2.vertex_comm_weight(3), 14);

    boost_graph_int_t b_g1;

    b_g1.add_vertex(1, 2, 3, 4);
    b_g1.add_vertex(5, 6, 7, 8);

    b_g1.add_edge(0, 1, 9);

    computational_dag_vector_impl_def_int_t graph_3(b_g1);

    BOOST_CHECK_EQUAL(graph_3.num_edges(), 1);
    BOOST_CHECK_EQUAL(graph_3.num_vertices(), 2);
    BOOST_CHECK_EQUAL(graph_3.vertex_work_weight(0), 1);
    BOOST_CHECK_EQUAL(graph_3.vertex_comm_weight(0), 2);
    BOOST_CHECK_EQUAL(graph_3.vertex_mem_weight(0), 3);
    BOOST_CHECK_EQUAL(graph_3.vertex_work_weight(1), 5);
    BOOST_CHECK_EQUAL(graph_3.vertex_comm_weight(1), 6);
    BOOST_CHECK_EQUAL(graph_3.vertex_mem_weight(1), 7);
}