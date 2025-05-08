/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#define BOOST_TEST_MODULE Sparse_Compact_Graph
#include <boost/test/unit_test.hpp>

#include <graph_implementations/adj_list_impl/compact_sparse_graph.hpp>
#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(Empty_Graph_keep_order) {
    Compact_Sparse_Graph<true> graph;

    BOOST_CHECK_EQUAL(graph.num_vertices(), 0);
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);
};

BOOST_AUTO_TEST_CASE(Empty_Graph_reorder) {
    Compact_Sparse_Graph<false> graph;

    BOOST_CHECK_EQUAL(graph.num_vertices(), 0);
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);
};

BOOST_AUTO_TEST_CASE(No_Edges_Graph_keep_order) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({});

    Compact_Sparse_Graph<true> graph(10, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 10);
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);
};

BOOST_AUTO_TEST_CASE(No_Edges_Graph_reorder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({});

    Compact_Sparse_Graph<false> graph(10, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 10);
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);

    std::vector<std::size_t> perm(10, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graph_perm = graph.get_pullback_permutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graph_perm.cbegin(), graph_perm.cend()));
};

BOOST_AUTO_TEST_CASE(LineGraph_keep_order) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}});

    Compact_Sparse_Graph<true> graph(8, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 8);
    BOOST_CHECK_EQUAL(graph.num_edges(), 7);

    std::size_t cntr = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr);
        ++cntr;
    }
    BOOST_CHECK_EQUAL(graph.num_vertices(), cntr);

    for (const auto &vert : graph.vertices()) {
        if (vert != 7) {
            BOOST_CHECK_EQUAL(graph.out_degree(vert), 1);
            for (const std::size_t &chld : graph.children(vert)) {
                BOOST_CHECK_EQUAL(chld, vert + 1);
            }
        } else {
            BOOST_CHECK_EQUAL(graph.out_degree(vert), 0);
            for (const std::size_t &chld : graph.children(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(chld, 100);
            }
        }
    }
    for (const auto &vert : graph.vertices()) {
        if (vert != 0) {
            BOOST_CHECK_EQUAL(graph.in_degree(vert), 1);
            for (const std::size_t &par : graph.parents(vert)) {
                BOOST_CHECK_EQUAL(par, vert - 1);
            }
        } else {
            BOOST_CHECK_EQUAL(graph.in_degree(vert), 0);
            for (const std::size_t &par : graph.parents(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(par, 100);
            }
        }
    }

    for (const auto &vert : graph.vertices()) {
        if (vert != 0) {
            BOOST_CHECK_EQUAL(graph.vertex_work_weight(vert), 2);
        } else {
            BOOST_CHECK_EQUAL(graph.vertex_work_weight(vert), 1);
        }
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.vertex_type(vert), 0);
    }
};


BOOST_AUTO_TEST_CASE(LineGraph_reorder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}});

    Compact_Sparse_Graph<false> graph(8, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 8);
    BOOST_CHECK_EQUAL(graph.num_edges(), 7);

    std::size_t cntr = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr);
        ++cntr;
    }
    BOOST_CHECK_EQUAL(graph.num_vertices(), cntr);

    for (const auto &vert : graph.vertices()) {
        if (vert != 7) {
            BOOST_CHECK_EQUAL(graph.out_degree(vert), 1);
            for (const std::size_t &chld : graph.children(vert)) {
                BOOST_CHECK_EQUAL(chld, vert + 1);
            }
        } else {
            BOOST_CHECK_EQUAL(graph.out_degree(vert), 0);
            for (const std::size_t &chld : graph.children(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(chld, 100);
            }
        }
    }
    for (const auto &vert : graph.vertices()) {
        if (vert != 0) {
            BOOST_CHECK_EQUAL(graph.in_degree(vert), 1);
            for (const std::size_t &par : graph.parents(vert)) {
                BOOST_CHECK_EQUAL(par, vert - 1);
            }
        } else {
            BOOST_CHECK_EQUAL(graph.in_degree(vert), 0);
            for (const std::size_t &par : graph.parents(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(par, 100);
            }
        }
    }

    for (const auto &vert : graph.vertices()) {
        if (vert != 0) {
            BOOST_CHECK_EQUAL(graph.vertex_work_weight(vert), 2);
        } else {
            BOOST_CHECK_EQUAL(graph.vertex_work_weight(vert), 1);
        }
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.vertex_type(vert), 0);
    }

    std::vector<std::size_t> perm(8, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graph_perm = graph.get_pullback_permutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graph_perm.cbegin(), graph_perm.cend()));

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(perm[vert], graph_perm[vert]);
    }
};


BOOST_AUTO_TEST_CASE(Graph1_keep_order) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({{0, 1}, {2, 3}, {6, 10}, {7, 9}, {0, 2}, {4, 6}, {1, 6}, {6, 7}, {5, 6}, {3, 7}, {1, 2}});

    Compact_Sparse_Graph<true> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 11);
    BOOST_CHECK_EQUAL(graph.num_edges(), 11);

    std::size_t cntr = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr);
        ++cntr;
    }
    BOOST_CHECK_EQUAL(graph.num_vertices(), cntr);

    std::vector<std::vector<std::size_t>> out_edges({
        {1, 2},
        {2, 6},
        {3},
        {7},
        {6},
        {6},
        {7, 10},
        {9},
        {},
        {},
        {}
    });

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.out_degree(vert), out_edges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : graph.children(vert)) {
            BOOST_CHECK_EQUAL(chld, out_edges[vert][cntr]);
            ++cntr;
        }
    }

    std::vector<std::vector<std::size_t>> in_edges({
        {},
        {0},
        {0, 1},
        {2},
        {},
        {},
        {1, 4, 5},
        {3, 6},
        {},
        {7},
        {6}
    });

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.in_degree(vert), in_edges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : graph.parents(vert)) {
            BOOST_CHECK_EQUAL(par, in_edges[vert][cntr]);
            ++cntr;
        }
    }
    
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.vertex_work_weight(vert), 1 + in_edges[vert].size());
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.vertex_type(vert), 0);
    }
};

BOOST_AUTO_TEST_CASE(Graph1_reorder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({{0, 1}, {2, 3}, {6, 10}, {7, 9}, {0, 2}, {4, 6}, {1, 6}, {6, 7}, {5, 6}, {3, 7}, {1, 2}});

    Compact_Sparse_Graph<false> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 11);
    BOOST_CHECK_EQUAL(graph.num_edges(), 11);

    std::size_t cntr = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr);
        ++cntr;
    }
    BOOST_CHECK_EQUAL(graph.num_vertices(), cntr);

    std::vector<std::size_t> perm(11, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graph_perm = graph.get_pullback_permutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graph_perm.cbegin(), graph_perm.cend()));

    std::vector<std::vector<std::size_t>> out_edges({
        {1, 2},
        {2, 6},
        {3},
        {7},
        {6},
        {6},
        {7, 10},
        {9},
        {},
        {},
        {}
    });

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.out_degree(vert), out_edges[ graph_perm[vert] ].size());
        std::size_t ori_vert = graph_perm[vert];
        
        std::size_t previous_chld = 0;
        std::size_t cntr = 0;
        for (const auto &chld : graph.children(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previous_chld, chld);
            }

            BOOST_CHECK(std::find(out_edges[ori_vert].cbegin(), out_edges[ori_vert].cend(), graph_perm[chld]) != out_edges[ori_vert].cend());

            previous_chld = chld;
            ++cntr;
        }
    }

    std::vector<std::vector<std::size_t>> in_edges({
        {},
        {0},
        {0, 1},
        {2},
        {},
        {},
        {1, 4, 5},
        {3, 6},
        {},
        {7},
        {6}
    });

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.in_degree(vert), in_edges[ graph_perm[vert] ].size());
        std::size_t ori_vert = graph_perm[vert];
        
        std::size_t previous_par = 0;
        std::size_t cntr = 0;
        for (const auto &par : graph.parents(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previous_par, par);
            }

            BOOST_CHECK(std::find(in_edges[ori_vert].cbegin(), in_edges[ori_vert].cend(), graph_perm[par]) != in_edges[ori_vert].cend());

            previous_par = par;
            ++cntr;
        }
    }
    
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.vertex_work_weight(vert), 1 + in_edges[graph_perm[vert]].size());
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.vertex_type(vert), 0);
    }
};

BOOST_AUTO_TEST_CASE(Graph_contruction) {

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


    Compact_Sparse_Graph<true, false, false, false, false, vertex_idx> copy_graph(graph.num_vertices(), graph.edges());
    BOOST_CHECK_EQUAL(copy_graph.num_vertices(), 8);
    BOOST_CHECK_EQUAL(copy_graph.num_edges(), 9);


    std::vector<std::vector<std::size_t>> out_edges({
        {1, 2, 3},
        {4, 6},
        {4, 5},
        {7},
        {7},
        {},
        {},
        {}
    });

    for (const auto &vert : copy_graph.vertices()) {
        BOOST_CHECK_EQUAL(copy_graph.out_degree(vert), out_edges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : copy_graph.children(vert)) {
            BOOST_CHECK_EQUAL(chld, out_edges[vert][cntr]);
            ++cntr;
        }
    }

    std::vector<std::vector<std::size_t>> in_edges({
        {},
        {0},
        {0},
        {0},
        {1, 2},
        {2},
        {1},
        {3, 4}
    });

    for (const auto &vert : copy_graph.vertices()) {
        BOOST_CHECK_EQUAL(copy_graph.in_degree(vert), in_edges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : copy_graph.parents(vert)) {
            BOOST_CHECK_EQUAL(par, in_edges[vert][cntr]);
            ++cntr;
        }
    }

    Compact_Sparse_Graph<false, false, false, false, false, vertex_idx> reorder_graph(graph.num_vertices(), graph.edges());
    BOOST_CHECK_EQUAL(reorder_graph.num_vertices(), 8);
    BOOST_CHECK_EQUAL(reorder_graph.num_edges(), 9);

    std::vector<std::size_t> perm(8, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graph_perm = reorder_graph.get_pullback_permutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graph_perm.cbegin(), graph_perm.cend()));

    for (const auto &vert : reorder_graph.vertices()) {
        BOOST_CHECK_EQUAL(reorder_graph.out_degree(vert), out_edges[ graph_perm[vert] ].size());
        std::size_t ori_vert = graph_perm[vert];
        
        std::size_t previous_chld = 0;
        std::size_t cntr = 0;
        for (const auto &chld : reorder_graph.children(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previous_chld, chld);
            }

            BOOST_CHECK(std::find(out_edges[ori_vert].cbegin(), out_edges[ori_vert].cend(), graph_perm[chld]) != out_edges[ori_vert].cend());

            previous_chld = chld;
            ++cntr;
        }
    }

    for (const auto &vert : reorder_graph.vertices()) {
        BOOST_CHECK_EQUAL(reorder_graph.in_degree(vert), in_edges[ graph_perm[vert] ].size());
        std::size_t ori_vert = graph_perm[vert];
        
        std::size_t previous_par = 0;
        std::size_t cntr = 0;
        for (const auto &par : reorder_graph.parents(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previous_par, par);
            }

            BOOST_CHECK(std::find(in_edges[ori_vert].cbegin(), in_edges[ori_vert].cend(), graph_perm[par]) != in_edges[ori_vert].cend());

            previous_par = par;
            ++cntr;
        }
    }
}