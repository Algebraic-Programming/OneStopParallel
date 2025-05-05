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
#include "graph_algorithms/directed_graph_path_util.hpp"
#include "graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

computational_dag_edge_idx_vector_impl_def_t constr_graph_1() {

    computational_dag_edge_idx_vector_impl_def_t graph;

    using vertex_idx = computational_dag_edge_idx_vector_impl_def_t::vertex_idx;

    vertex_idx v1 = graph.add_vertex(1, 2, 3, 4);
    vertex_idx v2 = graph.add_vertex(5, 6, 7, 8);
    vertex_idx v3 = graph.add_vertex(9, 10, 11, 12);
    vertex_idx v4 = graph.add_vertex(13, 14, 15, 16);
    vertex_idx v5 = graph.add_vertex(17, 18, 19, 20);
    vertex_idx v6 = graph.add_vertex(21, 22, 23, 24);
    vertex_idx v7 = graph.add_vertex(25, 26, 27, 28);
    vertex_idx v8 = graph.add_vertex(29, 30, 31, 32);

    auto pair = graph.add_edge(v1, v2);
    BOOST_CHECK_EQUAL(pair.second, true);

    pair = graph.add_edge(v1, v3);
    BOOST_CHECK_EQUAL(pair.second, true);

    pair = graph.add_edge(v1, v4);
    BOOST_CHECK_EQUAL(pair.second, true);

    pair = graph.add_edge(v2, v5);
    BOOST_CHECK_EQUAL(pair.second, true);

    pair = graph.add_edge(v2, v7);
    BOOST_CHECK_EQUAL(pair.second, true);

    pair = graph.add_edge(v3, v5);
    BOOST_CHECK_EQUAL(pair.second, true);

    pair = graph.add_edge(v3, v6);
    BOOST_CHECK_EQUAL(pair.second, true);

    pair = graph.add_edge(v4, v8);
    BOOST_CHECK_EQUAL(pair.second, true);

    pair = graph.add_edge(v5, v8);
    BOOST_CHECK_EQUAL(pair.second, true);

    BOOST_CHECK_EQUAL(graph.num_edges(), 9);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 8);

    return graph;
}

BOOST_AUTO_TEST_CASE(test_empty_dag_edge_idx) {

    computational_dag_edge_idx_vector_impl_def_t graph;
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 0);
}

BOOST_AUTO_TEST_CASE(test_dag_edge_idx) {

    computational_dag_edge_idx_vector_impl_def_t graph = constr_graph_1();

    using vertex_idx = computational_dag_edge_idx_vector_impl_def_t::vertex_idx;

    std::vector<vertex_idx> edge_sources{0, 0, 0, 1, 1, 2, 2, 3, 4};
    std::vector<vertex_idx> edge_targets{1, 2, 3, 4, 6, 4, 5, 7, 7};

    size_t edge_idx = 0;
    for (const auto &edge : graph.edges()) {

        BOOST_CHECK_EQUAL(edge.source, edge_sources[edge_idx]);
        BOOST_CHECK_EQUAL(edge.target, edge_targets[edge_idx]);
        edge_idx++;
    }

    std::vector<vertex_idx> vertices{0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<std::vector<vertex_idx>> out_neighbors{{1, 2, 3}, {4, 6}, {4, 5}, {7}, {7}, {}, {}, {}};

    std::vector<std::vector<vertex_idx>> in_neighbors{{}, {0}, {0}, {0}, {1, 2}, {2}, {1}, {3, 4}};

    size_t idx = 0;

    for (const auto &v : graph.vertices()) {

        BOOST_CHECK_EQUAL(v, vertices[idx++]);

        size_t i = 0;
        for (const auto &e : graph.children(v)) {
            BOOST_CHECK_EQUAL(e, out_neighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.out_edges(v)) {
            BOOST_CHECK_EQUAL(e.target, out_neighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.out_edges(v)) {
            BOOST_CHECK_EQUAL(target(e, graph), out_neighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.out_edges(v)) {
            BOOST_CHECK_EQUAL(graph.target(e), out_neighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.parents(v)) {
            BOOST_CHECK_EQUAL(e, in_neighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.in_edges(v)) {
            BOOST_CHECK_EQUAL(e.source, in_neighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.in_edges(v)) {
            BOOST_CHECK_EQUAL(source(e, graph), in_neighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.in_edges(v)) {
            BOOST_CHECK_EQUAL(graph.source(e), in_neighbors[v][i++]);
        }

        BOOST_CHECK_EQUAL(graph.in_degree(v), in_neighbors[v].size());
        BOOST_CHECK_EQUAL(graph.out_degree(v), out_neighbors[v].size());
    }
}

BOOST_AUTO_TEST_CASE(test_util_1) {

    const computational_dag_edge_idx_vector_impl_def_t graph = constr_graph_1();

    BOOST_CHECK_EQUAL(graph.num_edges(), 9);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 8);

    auto sources = source_vertices(graph);
    BOOST_CHECK_EQUAL(sources.size(), 1);
    BOOST_CHECK_EQUAL(sources[0], 0);

    auto sinks = sink_vertices(graph);
    BOOST_CHECK_EQUAL(sinks.size(), 3);
    BOOST_CHECK_EQUAL(sinks[0], 5);
    BOOST_CHECK_EQUAL(sinks[1], 6);
    BOOST_CHECK_EQUAL(sinks[2], 7);

    const auto pair = edge_desc(0, 1, graph);
    BOOST_CHECK_EQUAL(pair.second, true);
    BOOST_CHECK_EQUAL(source(pair.first,graph), 0);
    BOOST_CHECK_EQUAL(target(pair.first,graph), 1);
    BOOST_CHECK_EQUAL(edge(0, 1, graph), true);
    
    const auto pair2 = edge_desc(0, 4, graph);
    BOOST_CHECK_EQUAL(pair2.second, false);
    BOOST_CHECK_EQUAL(edge(0, 4, graph), false);

    const auto pair3 = edge_desc(1, 4, graph);
    BOOST_CHECK_EQUAL(pair3.second, true);
    BOOST_CHECK_EQUAL(source(pair3.first,graph), 1);
    BOOST_CHECK_EQUAL(target(pair3.first,graph), 4);
    BOOST_CHECK_EQUAL(edge(1, 4, graph), true);

    BOOST_CHECK_EQUAL(has_path(0, 1, graph), true);
    BOOST_CHECK_EQUAL(has_path(0, 2, graph), true);
    BOOST_CHECK_EQUAL(has_path(0, 3, graph), true);
    BOOST_CHECK_EQUAL(has_path(0, 4, graph), true);
    BOOST_CHECK_EQUAL(has_path(0, 5, graph), true);
    BOOST_CHECK_EQUAL(has_path(0, 6, graph), true);
    BOOST_CHECK_EQUAL(has_path(0, 7, graph), true);
    BOOST_CHECK_EQUAL(has_path(1, 0, graph), false);
    BOOST_CHECK_EQUAL(has_path(2, 0, graph), false);
    BOOST_CHECK_EQUAL(has_path(3, 0, graph), false);
    BOOST_CHECK_EQUAL(has_path(4, 0, graph), false);
    BOOST_CHECK_EQUAL(has_path(5, 0, graph), false);
    BOOST_CHECK_EQUAL(has_path(6, 0, graph), false);
    BOOST_CHECK_EQUAL(has_path(7, 0, graph), false);
    BOOST_CHECK_EQUAL(has_path(1, 4, graph), true);
    BOOST_CHECK_EQUAL(has_path(1, 7, graph), true);
    BOOST_CHECK_EQUAL(has_path(1, 6, graph), true);
    BOOST_CHECK_EQUAL(has_path(2, 4, graph), true);
    BOOST_CHECK_EQUAL(has_path(2, 5, graph), true);
    BOOST_CHECK_EQUAL(has_path(2, 7, graph), true);
    BOOST_CHECK_EQUAL(has_path(3, 7, graph), true);
    BOOST_CHECK_EQUAL(has_path(4, 7, graph), true);
    BOOST_CHECK_EQUAL(has_path(1, 2, graph), false);
    BOOST_CHECK_EQUAL(has_path(1, 3, graph), false);
    BOOST_CHECK_EQUAL(has_path(2, 1, graph), false);
    BOOST_CHECK_EQUAL(has_path(2, 3, graph), false);
    BOOST_CHECK_EQUAL(has_path(2, 6, graph), false);
    BOOST_CHECK_EQUAL(has_path(3, 1, graph), false);
    BOOST_CHECK_EQUAL(has_path(3, 2, graph), false);
    BOOST_CHECK_EQUAL(has_path(3, 4, graph), false);
    BOOST_CHECK_EQUAL(has_path(3, 5, graph), false);
    BOOST_CHECK_EQUAL(has_path(3, 6, graph), false);
    BOOST_CHECK_EQUAL(has_path(4, 1, graph), false);
    BOOST_CHECK_EQUAL(has_path(4, 2, graph), false);
    BOOST_CHECK_EQUAL(has_path(4, 3, graph), false);
    BOOST_CHECK_EQUAL(has_path(4, 5, graph), false);
    BOOST_CHECK_EQUAL(has_path(4, 6, graph), false);
    BOOST_CHECK_EQUAL(has_path(5, 1, graph), false);
    BOOST_CHECK_EQUAL(has_path(5, 2, graph), false);
    BOOST_CHECK_EQUAL(has_path(5, 3, graph), false);
    BOOST_CHECK_EQUAL(has_path(5, 4, graph), false);
    BOOST_CHECK_EQUAL(has_path(5, 6, graph), false);
    BOOST_CHECK_EQUAL(has_path(5, 7, graph), false);
    BOOST_CHECK_EQUAL(has_path(6, 1, graph), false);
    BOOST_CHECK_EQUAL(has_path(6, 2, graph), false);
    BOOST_CHECK_EQUAL(has_path(6, 3, graph), false);
    BOOST_CHECK_EQUAL(has_path(6, 4, graph), false);
    BOOST_CHECK_EQUAL(has_path(6, 5, graph), false);
    BOOST_CHECK_EQUAL(has_path(6, 7, graph), false);
    BOOST_CHECK_EQUAL(has_path(7, 1, graph), false);
    BOOST_CHECK_EQUAL(has_path(7, 2, graph), false);
    BOOST_CHECK_EQUAL(has_path(7, 3, graph), false);
    BOOST_CHECK_EQUAL(has_path(7, 4, graph), false);
    BOOST_CHECK_EQUAL(has_path(7, 5, graph), false);
    BOOST_CHECK_EQUAL(has_path(7, 6, graph), false);

    const auto long_edges = long_edges_in_triangles(graph);

    BOOST_CHECK_EQUAL(long_edges.size(), 0);

};
