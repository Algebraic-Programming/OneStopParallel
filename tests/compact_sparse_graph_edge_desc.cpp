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

#define BOOST_TEST_MODULE Sparse_Compact_Graph_Edge_Desc
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph_edge_desc.hpp"

#include <boost/test/unit_test.hpp>

using namespace osp;

BOOST_AUTO_TEST_CASE(Empty_Graph_keep_order) {
    Compact_Sparse_Graph_EdgeDesc<true> graph;

    BOOST_CHECK_EQUAL(graph.num_vertices(), 0);
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);

    for (const auto &edge : graph.edges()) {
        BOOST_CHECK(false);
        BOOST_CHECK_EQUAL(edge, 100);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.source(edge), graph.target(edge)));
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK(false);
        BOOST_CHECK_EQUAL(vert, 100);

        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
    }
}

BOOST_AUTO_TEST_CASE(Empty_Graph_reorder) {
    Compact_Sparse_Graph_EdgeDesc<false> graph;

    BOOST_CHECK_EQUAL(graph.num_vertices(), 0);
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);

    for (const auto &edge : graph.edges()) {
        BOOST_CHECK(false);
        BOOST_CHECK_EQUAL(edge, 100);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.source(edge), graph.target(edge)));
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK(false);
        BOOST_CHECK_EQUAL(vert, 100);

        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
    }
}

BOOST_AUTO_TEST_CASE(No_Edges_Graph_keep_order) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({});

    Compact_Sparse_Graph_EdgeDesc<true> graph(10, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 10);
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);

    for (const auto &edge : graph.edges()) {
        BOOST_CHECK(false);
        BOOST_CHECK_EQUAL(edge, 100);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.source(edge), graph.target(edge)));
    }

    std::size_t vert_counter = 0;
    for (const auto &vert : graph.vertices()) {
        vert_counter++;

        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
    }

    BOOST_CHECK_EQUAL(vert_counter, graph.num_vertices());
}

BOOST_AUTO_TEST_CASE(No_Edges_Graph_reorder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({});

    Compact_Sparse_Graph_EdgeDesc<false> graph(10, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 10);
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);

    std::size_t vert_counter = 0;
    for (const auto &vert : graph.vertices()) {
        vert_counter++;

        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
    }

    BOOST_CHECK_EQUAL(vert_counter, graph.num_vertices());

    std::vector<std::size_t> perm(10, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graph_perm = graph.get_pullback_permutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graph_perm.cbegin(), graph_perm.cend()));
}

BOOST_AUTO_TEST_CASE(LineGraph_keep_order) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {5, 6},
        {6, 7}
    });

    Compact_Sparse_Graph_EdgeDesc<true> graph(8, edges);

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
            for (const std::size_t &chld : graph.children(vert)) { BOOST_CHECK_EQUAL(chld, vert + 1); }
            auto chldren = graph.children(vert);
            BOOST_CHECK_EQUAL(chldren.crend() - chldren.crbegin(), graph.out_degree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) { BOOST_CHECK_EQUAL(*it, vert + 1); }

        } else {
            BOOST_CHECK_EQUAL(graph.out_degree(vert), 0);
            for (const std::size_t &chld : graph.children(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(chld, 100);
            }
            auto chldren = graph.children(vert);
            BOOST_CHECK_EQUAL(chldren.crend() - chldren.crbegin(), graph.out_degree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(*it, 100);
            }
        }
    }
    for (const auto &vert : graph.vertices()) {
        if (vert != 0) {
            BOOST_CHECK_EQUAL(graph.in_degree(vert), 1);
            for (const std::size_t &par : graph.parents(vert)) { BOOST_CHECK_EQUAL(par, vert - 1); }
            auto prnts = graph.parents(vert);
            BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.in_degree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) { BOOST_CHECK_EQUAL(*it, vert - 1); }
        } else {
            BOOST_CHECK_EQUAL(graph.in_degree(vert), 0);
            for (const std::size_t &par : graph.parents(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(par, 100);
            }
            auto prnts = graph.parents(vert);
            BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.in_degree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(*it, 100);
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

    for (const auto &vert : graph.vertices()) { BOOST_CHECK_EQUAL(graph.vertex_type(vert), 0); }

    std::size_t edge_counter = 0;
    for (const auto &edge : graph.edges()) {
        BOOST_CHECK_EQUAL(graph.source(edge), edge_counter);
        BOOST_CHECK_EQUAL(graph.target(edge), edge_counter + 1);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.source(edge), graph.target(edge)));

        ++edge_counter;
    }
    BOOST_CHECK_EQUAL(edge_counter, graph.num_edges());

    edge_counter = 0;
    for (const auto &edge : osp::edges(graph)) {
        BOOST_CHECK_EQUAL(source(edge, graph), edge_counter);
        BOOST_CHECK_EQUAL(target(edge, graph), edge_counter + 1);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.source(edge), graph.target(edge)));

        ++edge_counter;
    }
    BOOST_CHECK_EQUAL(edge_counter, graph.num_edges());

    std::size_t vert_counter = 0;
    for (const auto &vert : graph.vertices()) {
        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.source(edge), vert - 1);
            BOOST_CHECK_EQUAL(graph.target(edge), vert);
        }

        for (const auto &edge : in_edges(vert, graph)) {
            BOOST_CHECK_EQUAL(source(edge, graph), vert - 1);
            BOOST_CHECK_EQUAL(target(edge, graph), vert);
        }

        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.source(edge), vert);
            BOOST_CHECK_EQUAL(graph.target(edge), vert + 1);
        }

        for (const auto &edge : out_edges(vert, graph)) {
            BOOST_CHECK_EQUAL(source(edge, graph), vert);
            BOOST_CHECK_EQUAL(target(edge, graph), vert + 1);
        }

        ++vert_counter;
    }
    BOOST_CHECK_EQUAL(vert_counter, graph.num_vertices());
}

BOOST_AUTO_TEST_CASE(LineGraph_reorder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {5, 6},
        {6, 7}
    });

    Compact_Sparse_Graph_EdgeDesc<false> graph(8, edges);

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
            for (const std::size_t &chld : graph.children(vert)) { BOOST_CHECK_EQUAL(chld, vert + 1); }
            auto chldren = graph.children(vert);
            BOOST_CHECK_EQUAL(chldren.crend() - chldren.crbegin(), graph.out_degree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) { BOOST_CHECK_EQUAL(*it, vert + 1); }
        } else {
            BOOST_CHECK_EQUAL(graph.out_degree(vert), 0);
            for (const std::size_t &chld : graph.children(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(chld, 100);
            }
            auto chldren = graph.children(vert);
            BOOST_CHECK_EQUAL(chldren.crend() - chldren.crbegin(), graph.out_degree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(*it, 100);
            }
        }
    }
    for (const auto &vert : graph.vertices()) {
        if (vert != 0) {
            BOOST_CHECK_EQUAL(graph.in_degree(vert), 1);
            for (const std::size_t &par : graph.parents(vert)) { BOOST_CHECK_EQUAL(par, vert - 1); }
            auto prnts = graph.parents(vert);
            BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.in_degree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) { BOOST_CHECK_EQUAL(*it, vert - 1); }
        } else {
            BOOST_CHECK_EQUAL(graph.in_degree(vert), 0);
            for (const std::size_t &par : graph.parents(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(par, 100);
            }
            auto prnts = graph.parents(vert);
            BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.in_degree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(*it, 100);
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

    for (const auto &vert : graph.vertices()) { BOOST_CHECK_EQUAL(graph.vertex_type(vert), 0); }

    std::vector<std::size_t> perm(8, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graph_perm = graph.get_pullback_permutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graph_perm.cbegin(), graph_perm.cend()));

    for (const auto &vert : graph.vertices()) { BOOST_CHECK_EQUAL(perm[vert], graph_perm[vert]); }

    std::size_t edge_counter = 0;
    for (const auto &edge : graph.edges()) {
        BOOST_CHECK_EQUAL(graph.source(edge), edge_counter);
        BOOST_CHECK_EQUAL(graph.target(edge), edge_counter + 1);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.source(edge), graph.target(edge)));

        ++edge_counter;
    }
    BOOST_CHECK_EQUAL(edge_counter, graph.num_edges());

    std::size_t vert_counter = 0;
    for (const auto &vert : graph.vertices()) {
        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.source(edge), vert - 1);
            BOOST_CHECK_EQUAL(graph.target(edge), vert);
        }
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.source(edge), vert);
            BOOST_CHECK_EQUAL(graph.target(edge), vert + 1);
        }

        ++vert_counter;
    }
    BOOST_CHECK_EQUAL(vert_counter, graph.num_vertices());
}

BOOST_AUTO_TEST_CASE(Graph1_keep_order) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });

    Compact_Sparse_Graph_EdgeDesc<true> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 11);
    BOOST_CHECK_EQUAL(graph.num_edges(), 11);

    std::size_t cntr_0 = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr_0);
        ++cntr_0;
    }
    BOOST_CHECK_EQUAL(graph.num_vertices(), cntr_0);

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
        auto chldrn = graph.children(vert);
        BOOST_CHECK_EQUAL(chldrn.crend() - chldrn.crbegin(), graph.out_degree(vert));
        for (auto it = chldrn.crbegin(); it != chldrn.crend(); ++it) {
            --cntr;
            BOOST_CHECK_EQUAL(*it, out_edges[vert][cntr]);
        }
    }

    for (const auto &vert : graph.vertices()) {
        std::size_t cntr = 0;
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.source(edge), vert);
            BOOST_CHECK_EQUAL(graph.target(edge), out_edges[vert][cntr]);
            ++cntr;
        }
        BOOST_CHECK_EQUAL(cntr, graph.out_degree(vert));
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
        auto prnts = graph.parents(vert);
        BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.in_degree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            --cntr;
            BOOST_CHECK_EQUAL(*it, in_edges[vert][cntr]);
        }
    }

    for (const auto &vert : graph.vertices()) {
        std::size_t cntr = 0;
        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.source(edge), in_edges[vert][cntr]);
            BOOST_CHECK_EQUAL(graph.target(edge), vert);
            ++cntr;
        }
        BOOST_CHECK_EQUAL(cntr, graph.in_degree(vert));
    }

    std::size_t edge_cntr = 0;
    for (const auto &edge : graph.edges()) {
        BOOST_CHECK_EQUAL(edge, edge_cntr);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.source(edge), graph.target(edge)));

        ++edge_cntr;
    }
    BOOST_CHECK_EQUAL(edge_cntr, graph.num_edges());

    edge_cntr = 0;
    for (const auto &vert : graph.vertices()) {
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(edge, edge_cntr);
            ++edge_cntr;
        }
    }
    BOOST_CHECK_EQUAL(edge_cntr, graph.num_edges());

    for (const auto &vert : graph.vertices()) { BOOST_CHECK_EQUAL(graph.vertex_work_weight(vert), 1 + in_edges[vert].size()); }

    for (const auto &vert : graph.vertices()) { BOOST_CHECK_EQUAL(graph.vertex_type(vert), 0); }
}

BOOST_AUTO_TEST_CASE(Graph1_reorder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });

    Compact_Sparse_Graph_EdgeDesc<false> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 11);
    BOOST_CHECK_EQUAL(graph.num_edges(), 11);

    std::size_t cntr_0 = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr_0);
        ++cntr_0;
    }
    BOOST_CHECK_EQUAL(graph.num_vertices(), cntr_0);

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
        BOOST_CHECK_EQUAL(graph.out_degree(vert), out_edges[graph_perm[vert]].size());
        std::size_t ori_vert = graph_perm[vert];

        std::size_t previous_chld = 0;
        std::size_t cntr = 0;
        for (const auto &chld : graph.children(vert)) {
            if (cntr > 0) { BOOST_CHECK_LE(previous_chld, chld); }

            BOOST_CHECK(std::find(out_edges[ori_vert].cbegin(), out_edges[ori_vert].cend(), graph_perm[chld])
                        != out_edges[ori_vert].cend());

            previous_chld = chld;
            ++cntr;
        }
        auto chldrn = graph.children(vert);
        BOOST_CHECK_EQUAL(chldrn.crend() - chldrn.crbegin(), graph.out_degree(vert));
        for (auto it = chldrn.crbegin(); it != chldrn.crend(); ++it) {
            if (cntr < graph.out_degree(vert)) { BOOST_CHECK_GE(previous_chld, *it); }

            --cntr;
            BOOST_CHECK(std::find(out_edges[ori_vert].cbegin(), out_edges[ori_vert].cend(), graph_perm[*it])
                        != out_edges[ori_vert].cend());

            previous_chld = *it;
        }
    }

    for (const auto &vert : graph.vertices()) {
        std::size_t cntr = 0;
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.source(edge), vert);
            ++cntr;
        }
        BOOST_CHECK_EQUAL(cntr, graph.out_degree(vert));
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
        BOOST_CHECK_EQUAL(graph.in_degree(vert), in_edges[graph_perm[vert]].size());
        std::size_t ori_vert = graph_perm[vert];

        std::size_t previous_par = 0;
        std::size_t cntr = 0;
        for (const auto &par : graph.parents(vert)) {
            if (cntr > 0) { BOOST_CHECK_LE(previous_par, par); }

            BOOST_CHECK(std::find(in_edges[ori_vert].cbegin(), in_edges[ori_vert].cend(), graph_perm[par])
                        != in_edges[ori_vert].cend());

            previous_par = par;
            ++cntr;
        }
        auto prnts = graph.parents(vert);
        BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.in_degree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            if (cntr < graph.out_degree(vert)) { BOOST_CHECK_GE(previous_par, *it); }

            --cntr;
            BOOST_CHECK(std::find(in_edges[ori_vert].cbegin(), in_edges[ori_vert].cend(), graph_perm[*it])
                        != in_edges[ori_vert].cend());

            previous_par = *it;
        }
    }

    for (const auto &vert : graph.vertices()) {
        std::size_t cntr = 0;
        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.target(edge), vert);
            ++cntr;
        }
        BOOST_CHECK_EQUAL(cntr, graph.in_degree(vert));
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.vertex_work_weight(vert), 1 + in_edges[graph_perm[vert]].size());
    }

    for (const auto &vert : graph.vertices()) { BOOST_CHECK_EQUAL(graph.vertex_type(vert), 0); }

    std::size_t edge_cntr = 0;
    for (const auto &edge : graph.edges()) {
        BOOST_CHECK_EQUAL(edge, edge_cntr);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.source(edge), graph.target(edge)));

        ++edge_cntr;
    }
    BOOST_CHECK_EQUAL(edge_cntr, graph.num_edges());

    edge_cntr = 0;
    for (const auto &vert : graph.vertices()) {
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(edge, edge_cntr);
            ++edge_cntr;
        }
    }
    BOOST_CHECK_EQUAL(edge_cntr, graph.num_edges());
}

BOOST_AUTO_TEST_CASE(Graph1_e_comm_keep_order) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });
    const std::vector<unsigned> edge_weights({3, 6, 12, 874, 134, 67, 234, 980, 123, 152, 34});

    Compact_Sparse_Graph_EdgeDesc<true, true, true, true, true, true> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 11);
    BOOST_CHECK_EQUAL(graph.num_edges(), 11);

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const auto &[src, tgt] = edges[i];
        graph.set_edge_comm_weight(src, tgt, edge_weights[i]);
    }

    for (const auto &edge : graph.edges()) {
        const auto src = graph.source(edge);
        const auto tgt = graph.target(edge);

        auto it = std::find(edges.cbegin(), edges.cend(), std::make_pair(src, tgt));
        BOOST_CHECK(it != edges.cend());

        auto ind = std::distance(edges.cbegin(), it);
        BOOST_CHECK_EQUAL(edge_weights[static_cast<std::size_t>(ind)], graph.edge_comm_weight(edge));
    }
}

BOOST_AUTO_TEST_CASE(Graph1_e_comm_reorder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });
    const std::vector<unsigned> edge_weights({3, 6, 12, 874, 134, 67, 234, 980, 123, 152, 34});

    Compact_Sparse_Graph_EdgeDesc<false, true, true, true, true, true> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.num_vertices(), 11);
    BOOST_CHECK_EQUAL(graph.num_edges(), 11);

    std::vector<std::size_t> perm(11, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graph_perm = graph.get_pullback_permutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graph_perm.cbegin(), graph_perm.cend()));

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const auto &[src, tgt] = edges[i];
        graph.set_edge_comm_weight(src, tgt, edge_weights[i]);
    }

    for (const auto &edge : graph.edges()) {
        const auto src = graph_perm[graph.source(edge)];
        const auto tgt = graph_perm[graph.target(edge)];

        auto it = std::find(edges.cbegin(), edges.cend(), std::make_pair(src, tgt));
        BOOST_CHECK(it != edges.cend());

        auto ind = std::distance(edges.cbegin(), it);
        BOOST_CHECK_EQUAL(edge_weights[static_cast<std::size_t>(ind)], graph.edge_comm_weight(edge));
    }
}
