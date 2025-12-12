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
#include "osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"

#include <boost/test/unit_test.hpp>

#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(EmptyGraphKeepOrder) {
    CompactSparseGraph<true> graph;

    BOOST_CHECK_EQUAL(graph.NumVertices(), 0);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);
}

BOOST_AUTO_TEST_CASE(EmptyGraphReorder) {
    CompactSparseGraph<false> graph;

    BOOST_CHECK_EQUAL(graph.NumVertices(), 0);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);
}

BOOST_AUTO_TEST_CASE(NoEdgesGraphKeepOrder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({});

    CompactSparseGraph<true> graph(10, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 10);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);
}

BOOST_AUTO_TEST_CASE(NoEdgesGraphReorder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({});

    CompactSparseGraph<false> graph(10, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 10);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);

    std::vector<std::size_t> perm(10, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graphPerm.cbegin(), graphPerm.cend()));
}

BOOST_AUTO_TEST_CASE(LineGraphKeepOrder) {
    const std::set<std::pair<std::size_t, std::size_t>> edges({
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {5, 6},
        {6, 7}
    });

    CompactSparseGraph<true> graph(8, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 8);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 7);

    std::size_t cntr = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr);
        ++cntr;
    }
    BOOST_CHECK_EQUAL(graph.NumVertices(), cntr);

    for (const auto &vert : graph.vertices()) {
        if (vert != 7) {
            BOOST_CHECK_EQUAL(graph.out_degree(vert), 1);
            for (const std::size_t &chld : graph.children(vert)) {
                BOOST_CHECK_EQUAL(chld, vert + 1);
            }
            auto chldren = graph.children(vert);
            BOOST_CHECK_EQUAL(chldren.crend() - chldren.crbegin(), graph.out_degree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
                BOOST_CHECK_EQUAL(*it, vert + 1);
            }

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
            for (const std::size_t &par : graph.parents(vert)) {
                BOOST_CHECK_EQUAL(par, vert - 1);
            }
            auto prnts = graph.parents(vert);
            BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.in_degree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
                BOOST_CHECK_EQUAL(*it, vert - 1);
            }
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
            BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 2);
        } else {
            BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 1);
        }
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), 0);
    }
}

BOOST_AUTO_TEST_CASE(LineGraphReorder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {5, 6},
        {6, 7}
    });

    CompactSparseGraph<false> graph(8, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 8);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 7);

    std::size_t cntr = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr);
        ++cntr;
    }
    BOOST_CHECK_EQUAL(graph.NumVertices(), cntr);

    for (const auto &vert : graph.vertices()) {
        if (vert != 7) {
            BOOST_CHECK_EQUAL(graph.out_degree(vert), 1);
            for (const std::size_t &chld : graph.children(vert)) {
                BOOST_CHECK_EQUAL(chld, vert + 1);
            }
            auto chldren = graph.children(vert);
            BOOST_CHECK_EQUAL(chldren.crend() - chldren.crbegin(), graph.out_degree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
                BOOST_CHECK_EQUAL(*it, vert + 1);
            }
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
            for (const std::size_t &par : graph.parents(vert)) {
                BOOST_CHECK_EQUAL(par, vert - 1);
            }
            auto prnts = graph.parents(vert);
            BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.in_degree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
                BOOST_CHECK_EQUAL(*it, vert - 1);
            }
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
            BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 2);
        } else {
            BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 1);
        }
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), 0);
    }

    std::vector<std::size_t> perm(8, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graphPerm.cbegin(), graphPerm.cend()));

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(perm[vert], graphPerm[vert]);
    }
}

BOOST_AUTO_TEST_CASE(Graph1KeepOrder) {
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

    CompactSparseGraph<true> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 11);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 11);

    std::size_t cntr0 = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr0);
        ++cntr0;
    }
    BOOST_CHECK_EQUAL(graph.NumVertices(), cntr0);

    std::vector<std::vector<std::size_t>> outEdges({
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
        BOOST_CHECK_EQUAL(graph.out_degree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : graph.children(vert)) {
            BOOST_CHECK_EQUAL(chld, outEdges[vert][cntr]);
            ++cntr;
        }
        auto chldrn = graph.children(vert);
        BOOST_CHECK_EQUAL(chldrn.crend() - chldrn.crbegin(), graph.out_degree(vert));
        for (auto it = chldrn.crbegin(); it != chldrn.crend(); ++it) {
            --cntr;
            BOOST_CHECK_EQUAL(*it, outEdges[vert][cntr]);
        }

        cntr = 0;
        for (const auto &e : osp::out_edges(vert, graph)) {
            BOOST_CHECK_EQUAL(Traget(e, graph), outEdges[vert][cntr++]);
        }
    }

    std::vector<std::vector<std::size_t>> inEdges({
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
        BOOST_CHECK_EQUAL(graph.in_degree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : graph.parents(vert)) {
            BOOST_CHECK_EQUAL(par, inEdges[vert][cntr]);
            ++cntr;
        }
        auto prnts = graph.parents(vert);
        BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.in_degree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            --cntr;
            BOOST_CHECK_EQUAL(*it, inEdges[vert][cntr]);
        }

        cntr = 0;
        for (const auto &e : osp::in_edges(vert, graph)) {
            BOOST_CHECK_EQUAL(Source(e, graph), inEdges[vert][cntr++]);
        }
    }

    unsigned count = 0;
    for (const auto &e : osp::edges(graph)) {
        std::cout << e.source << " -> " << e.target << std::endl;
        count++;
    }

    BOOST_CHECK_EQUAL(count, 11);

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 1 + inEdges[vert].size());
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), 0);
    }
}

BOOST_AUTO_TEST_CASE(Graph1Reorder) {
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

    CompactSparseGraph<false> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 11);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 11);

    std::size_t cntr0 = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr0);
        ++cntr0;
    }
    BOOST_CHECK_EQUAL(graph.NumVertices(), cntr0);

    std::vector<std::size_t> perm(11, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graphPerm.cbegin(), graphPerm.cend()));

    std::vector<std::vector<std::size_t>> outEdges({
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
        BOOST_CHECK_EQUAL(graph.out_degree(vert), outEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousChld = 0;
        std::size_t cntr = 0;
        for (const auto &chld : graph.children(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousChld, chld);
            }

            BOOST_CHECK(std::find(outEdges[oriVert].cbegin(), outEdges[oriVert].cend(), graphPerm[chld])
                        != outEdges[oriVert].cend());

            previousChld = chld;
            ++cntr;
        }
        auto chldrn = graph.children(vert);
        BOOST_CHECK_EQUAL(chldrn.crend() - chldrn.crbegin(), graph.out_degree(vert));
        for (auto it = chldrn.crbegin(); it != chldrn.crend(); ++it) {
            if (cntr < graph.out_degree(vert)) {
                BOOST_CHECK_GE(previousChld, *it);
            }

            --cntr;
            BOOST_CHECK(std::find(outEdges[oriVert].cbegin(), outEdges[oriVert].cend(), graphPerm[*it])
                        != outEdges[oriVert].cend());

            previousChld = *it;
        }
    }

    std::vector<std::vector<std::size_t>> inEdges({
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
        BOOST_CHECK_EQUAL(graph.in_degree(vert), inEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousPar = 0;
        std::size_t cntr = 0;
        for (const auto &par : graph.parents(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousPar, par);
            }

            BOOST_CHECK(std::find(inEdges[oriVert].cbegin(), inEdges[oriVert].cend(), graphPerm[par]) != inEdges[oriVert].cend());

            previousPar = par;
            ++cntr;
        }
        auto prnts = graph.parents(vert);
        BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.in_degree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            if (cntr < graph.out_degree(vert)) {
                BOOST_CHECK_GE(previousPar, *it);
            }

            --cntr;
            BOOST_CHECK(std::find(inEdges[oriVert].cbegin(), inEdges[oriVert].cend(), graphPerm[*it]) != inEdges[oriVert].cend());

            previousPar = *it;
        }
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 1 + inEdges[graphPerm[vert]].size());
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), 0);
    }
}

BOOST_AUTO_TEST_CASE(GraphEdgeContruction) {
    computational_dag_edge_idx_vector_impl_def_t graph;

    using VertexIdx = computational_dag_edge_idx_vector_impl_def_t::vertex_idx;

    VertexIdx v1 = graph.add_vertex(1, 2, 3, 4);
    VertexIdx v2 = graph.add_vertex(5, 6, 7, 8);
    VertexIdx v3 = graph.add_vertex(9, 10, 11, 12);
    VertexIdx v4 = graph.add_vertex(13, 14, 15, 16);
    VertexIdx v5 = graph.add_vertex(17, 18, 19, 20);
    VertexIdx v6 = graph.add_vertex(21, 22, 23, 24);
    VertexIdx v7 = graph.add_vertex(25, 26, 27, 28);
    VertexIdx v8 = graph.add_vertex(29, 30, 31, 32);

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

    BOOST_CHECK_EQUAL(graph.NumEdges(), 9);
    BOOST_CHECK_EQUAL(graph.NumVertices(), 8);

    Compact_Sparse_Graph<true, false, false, false, false, VertexIdx> copyGraph(graph.NumVertices(), edge_view(graph));
    BOOST_CHECK_EQUAL(copyGraph.NumVertices(), 8);
    BOOST_CHECK_EQUAL(copyGraph.NumEdges(), 9);

    std::vector<std::vector<std::size_t>> outEdges({
        {1, 2, 3},
        {4, 6},
        {4, 5},
        {7},
        {7},
        {},
        {},
        {}
    });

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.out_degree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : copyGraph.children(vert)) {
            BOOST_CHECK_EQUAL(chld, outEdges[vert][cntr]);
            ++cntr;
        }
    }

    std::vector<std::vector<std::size_t>> inEdges({
        {},
        {0},
        {0},
        {0},
        {1, 2},
        {2},
        {1},
        {3, 4}
    });

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.in_degree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : copyGraph.parents(vert)) {
            BOOST_CHECK_EQUAL(par, inEdges[vert][cntr]);
            ++cntr;
        }
    }

    Compact_Sparse_Graph<false, false, false, false, false, VertexIdx> reorderGraph(graph.NumVertices(), edge_view(graph));
    BOOST_CHECK_EQUAL(reorderGraph.NumVertices(), 8);
    BOOST_CHECK_EQUAL(reorderGraph.NumEdges(), 9);

    std::vector<std::size_t> perm(8, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graphPerm = reorderGraph.get_pullback_permutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graphPerm.cbegin(), graphPerm.cend()));

    for (const auto &vert : reorderGraph.vertices()) {
        BOOST_CHECK_EQUAL(reorderGraph.out_degree(vert), outEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousChld = 0;
        std::size_t cntr = 0;
        for (const auto &chld : reorderGraph.children(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousChld, chld);
            }

            BOOST_CHECK(std::find(outEdges[oriVert].cbegin(), outEdges[oriVert].cend(), graphPerm[chld])
                        != outEdges[oriVert].cend());

            previousChld = chld;
            ++cntr;
        }
    }

    for (const auto &vert : reorderGraph.vertices()) {
        BOOST_CHECK_EQUAL(reorderGraph.in_degree(vert), inEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousPar = 0;
        std::size_t cntr = 0;
        for (const auto &par : reorderGraph.parents(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousPar, par);
            }

            BOOST_CHECK(std::find(inEdges[oriVert].cbegin(), inEdges[oriVert].cend(), graphPerm[par]) != inEdges[oriVert].cend());

            previousPar = par;
            ++cntr;
        }
    }
}

BOOST_AUTO_TEST_CASE(GraphWorkWeightsKeepOrder) {
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

    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    CompactSparseGraph<true, true> graph(11, edges, ww);

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), ww[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexWorkWeight(vert, wt);
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), wt);
    }
}

BOOST_AUTO_TEST_CASE(GraphWorkWeightsReorder) {
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

    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    CompactSparseGraph<false, true> graph(11, edges, ww);

    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), ww[graphPerm[vert]]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexWorkWeight(graphPerm[vert], wt);
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), wt);
    }
}

BOOST_AUTO_TEST_CASE(GraphCommWeightsKeepOrder) {
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

    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    std::vector<unsigned> cw(11);
    std::iota(cw.begin(), cw.end(), 11);

    CompactSparseGraph<true, true, true> graph(11, edges, ww, cw);

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), ww[vert]);
    }

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), cw[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexCommWeight(vert, wt);
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), wt);
    }
}

BOOST_AUTO_TEST_CASE(GraphCommWeightsReorder) {
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

    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    std::vector<unsigned> cw(11);
    std::iota(cw.begin(), cw.end(), 11);

    CompactSparseGraph<false, true, true> graph(11, edges, ww, cw);

    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), ww[graphPerm[vert]]);
    }

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), cw[graphPerm[vert]]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexCommWeight(graphPerm[vert], wt);
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), wt);
    }
}

BOOST_AUTO_TEST_CASE(GraphMemWeightsKeepOrder) {
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

    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    std::vector<unsigned> cw(11);
    std::iota(cw.begin(), cw.end(), 11);

    std::vector<unsigned> mw(11);
    std::iota(mw.begin(), mw.end(), 22);

    CompactSparseGraph<true, true, true, true> graph(11, edges, ww, cw, mw);

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), ww[vert]);
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), cw[vert]);
    }

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexMemWeight(vert), mw[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexMemWeight(vert, wt);
        BOOST_CHECK_EQUAL(graph.VertexMemWeight(vert), wt);
    }
}

BOOST_AUTO_TEST_CASE(GraphMemWeightsReorder) {
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

    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    std::vector<unsigned> cw(11);
    std::iota(cw.begin(), cw.end(), 11);

    std::vector<unsigned> mw(11);
    std::iota(mw.begin(), mw.end(), 22);

    CompactSparseGraph<false, true, true, true> graph(11, edges, ww, cw, mw);

    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), ww[graphPerm[vert]]);
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), cw[graphPerm[vert]]);
    }

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexMemWeight(vert), mw[graphPerm[vert]]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexMemWeight(graphPerm[vert], wt);
        BOOST_CHECK_EQUAL(graph.VertexMemWeight(vert), wt);
    }
}

BOOST_AUTO_TEST_CASE(GraphVtypeKeepOrder) {
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

    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    std::vector<unsigned> cw(11);
    std::iota(cw.begin(), cw.end(), 11);

    std::vector<unsigned> mw(11);
    std::iota(mw.begin(), mw.end(), 22);

    std::vector<unsigned> vt(11);
    std::iota(vt.begin(), vt.end(), 33);

    CompactSparseGraph<true, true, true, true, true> graph(11, edges, ww, cw, mw, vt);

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), ww[vert]);
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), cw[vert]);
        BOOST_CHECK_EQUAL(graph.VertexMemWeight(vert), mw[vert]);
    }

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), vt[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexType(vert, wt);
        BOOST_CHECK_EQUAL(graph.VertexType(vert), wt);
    }
}

BOOST_AUTO_TEST_CASE(GraphVtypeReorder) {
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

    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    std::vector<unsigned> cw(11);
    std::iota(cw.begin(), cw.end(), 11);

    std::vector<unsigned> mw(11);
    std::iota(mw.begin(), mw.end(), 22);

    std::vector<unsigned> vt(11);
    std::iota(vt.begin(), vt.end(), 33);

    CompactSparseGraph<false, true, true, true, true> graph(11, edges, ww, cw, mw, vt);

    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), ww[graphPerm[vert]]);
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), cw[graphPerm[vert]]);
        BOOST_CHECK_EQUAL(graph.VertexMemWeight(vert), mw[graphPerm[vert]]);
    }

    for (auto vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), vt[graphPerm[vert]]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexType(graphPerm[vert], wt);
        BOOST_CHECK_EQUAL(graph.VertexType(vert), wt);
    }
}

BOOST_AUTO_TEST_CASE(GraphTypeCopyContruction) {
    computational_dag_edge_idx_vector_impl_def_t graph;

    using VertexIdx = computational_dag_edge_idx_vector_impl_def_t::vertex_idx;

    VertexIdx v1 = graph.add_vertex(1, 2, 3, 4);
    VertexIdx v2 = graph.add_vertex(5, 6, 7, 8);
    VertexIdx v3 = graph.add_vertex(9, 10, 11, 12);
    VertexIdx v4 = graph.add_vertex(13, 14, 15, 16);
    VertexIdx v5 = graph.add_vertex(17, 18, 19, 20);
    VertexIdx v6 = graph.add_vertex(21, 22, 23, 24);
    VertexIdx v7 = graph.add_vertex(25, 26, 27, 28);
    VertexIdx v8 = graph.add_vertex(29, 30, 31, 32);

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

    BOOST_CHECK_EQUAL(graph.NumEdges(), 9);
    BOOST_CHECK_EQUAL(graph.NumVertices(), 8);

    Compact_Sparse_Graph<true,
                         true,
                         true,
                         true,
                         true,
                         VertexIdx,
                         std::size_t,
                         computational_dag_edge_idx_vector_impl_def_t::vertex_work_weight_type,
                         computational_dag_edge_idx_vector_impl_def_t::vertex_comm_weight_type,
                         computational_dag_edge_idx_vector_impl_def_t::vertex_mem_weight_type,
                         computational_dag_edge_idx_vector_impl_def_t::vertex_type_type>
        copyGraph(graph);
    BOOST_CHECK_EQUAL(copyGraph.NumVertices(), 8);
    BOOST_CHECK_EQUAL(copyGraph.NumEdges(), 9);

    std::vector<std::vector<std::size_t>> outEdges({
        {1, 2, 3},
        {4, 6},
        {4, 5},
        {7},
        {7},
        {},
        {},
        {}
    });

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), copyGraph.VertexWorkWeight(vert));
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), copyGraph.VertexCommWeight(vert));
        BOOST_CHECK_EQUAL(graph.VertexMemWeight(vert), copyGraph.VertexMemWeight(vert));
        BOOST_CHECK_EQUAL(graph.VertexType(vert), copyGraph.VertexType(vert));
    }

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.out_degree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : copyGraph.children(vert)) {
            BOOST_CHECK_EQUAL(chld, outEdges[vert][cntr]);
            ++cntr;
        }
    }

    std::vector<std::vector<std::size_t>> inEdges({
        {},
        {0},
        {0},
        {0},
        {1, 2},
        {2},
        {1},
        {3, 4}
    });

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.in_degree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : copyGraph.parents(vert)) {
            BOOST_CHECK_EQUAL(par, inEdges[vert][cntr]);
            ++cntr;
        }
    }

    Compact_Sparse_Graph<false,
                         true,
                         true,
                         true,
                         true,
                         VertexIdx,
                         std::size_t,
                         computational_dag_edge_idx_vector_impl_def_t::vertex_work_weight_type,
                         computational_dag_edge_idx_vector_impl_def_t::vertex_comm_weight_type,
                         computational_dag_edge_idx_vector_impl_def_t::vertex_mem_weight_type,
                         computational_dag_edge_idx_vector_impl_def_t::vertex_type_type>
        reorderGraph(graph);
    BOOST_CHECK_EQUAL(reorderGraph.NumVertices(), 8);
    BOOST_CHECK_EQUAL(reorderGraph.NumEdges(), 9);

    std::vector<std::size_t> perm(8, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graphPerm = reorderGraph.get_pullback_permutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graphPerm.cbegin(), graphPerm.cend()));

    for (const auto &vert : reorderGraph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(graphPerm[vert]), reorderGraph.VertexWorkWeight(vert));
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(graphPerm[vert]), reorderGraph.VertexCommWeight(vert));
        BOOST_CHECK_EQUAL(graph.VertexMemWeight(graphPerm[vert]), reorderGraph.VertexMemWeight(vert));
        BOOST_CHECK_EQUAL(graph.VertexType(graphPerm[vert]), reorderGraph.VertexType(vert));
    }

    for (const auto &vert : reorderGraph.vertices()) {
        BOOST_CHECK_EQUAL(reorderGraph.out_degree(vert), outEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousChld = 0;
        std::size_t cntr = 0;
        for (const auto &chld : reorderGraph.children(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousChld, chld);
            }

            BOOST_CHECK(std::find(outEdges[oriVert].cbegin(), outEdges[oriVert].cend(), graphPerm[chld])
                        != outEdges[oriVert].cend());

            previousChld = chld;
            ++cntr;
        }
    }

    for (const auto &vert : reorderGraph.vertices()) {
        BOOST_CHECK_EQUAL(reorderGraph.in_degree(vert), inEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousPar = 0;
        std::size_t cntr = 0;
        for (const auto &par : reorderGraph.parents(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousPar, par);
            }

            BOOST_CHECK(std::find(inEdges[oriVert].cbegin(), inEdges[oriVert].cend(), graphPerm[par]) != inEdges[oriVert].cend());

            previousPar = par;
            ++cntr;
        }
    }
}

BOOST_AUTO_TEST_CASE(Graph1CopyKeepOrder) {
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

    CompactSparseGraph<true> graph(11, edges);
    CompactSparseGraph<true> copyGraph(graph);

    BOOST_CHECK_EQUAL(copyGraph.NumVertices(), 11);
    BOOST_CHECK_EQUAL(copyGraph.NumEdges(), 11);

    std::size_t cntr0 = 0;
    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr0);
        ++cntr0;
    }
    BOOST_CHECK_EQUAL(copyGraph.NumVertices(), cntr0);

    std::vector<std::vector<std::size_t>> outEdges({
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

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.out_degree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : copyGraph.children(vert)) {
            BOOST_CHECK_EQUAL(chld, outEdges[vert][cntr]);
            ++cntr;
        }
    }

    std::vector<std::vector<std::size_t>> inEdges({
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

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.in_degree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : copyGraph.parents(vert)) {
            BOOST_CHECK_EQUAL(par, inEdges[vert][cntr]);
            ++cntr;
        }
    }

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.VertexWorkWeight(vert), 1 + inEdges[vert].size());
    }

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.VertexType(vert), 0);
    }
}

BOOST_AUTO_TEST_CASE(Graph1MoveKeepOrder) {
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

    CompactSparseGraph<true> graph(11, edges);
    CompactSparseGraph<true> copyGraph(std::move(graph));

    BOOST_CHECK_EQUAL(copyGraph.NumVertices(), 11);
    BOOST_CHECK_EQUAL(copyGraph.NumEdges(), 11);

    std::size_t cntr0 = 0;
    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr0);
        ++cntr0;
    }
    BOOST_CHECK_EQUAL(copyGraph.NumVertices(), cntr0);

    std::vector<std::vector<std::size_t>> outEdges({
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

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.out_degree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : copyGraph.children(vert)) {
            BOOST_CHECK_EQUAL(chld, outEdges[vert][cntr]);
            ++cntr;
        }
    }

    std::vector<std::vector<std::size_t>> inEdges({
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

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.in_degree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : copyGraph.parents(vert)) {
            BOOST_CHECK_EQUAL(par, inEdges[vert][cntr]);
            ++cntr;
        }
    }

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.VertexWorkWeight(vert), 1 + inEdges[vert].size());
    }

    for (const auto &vert : copyGraph.vertices()) {
        BOOST_CHECK_EQUAL(copyGraph.VertexType(vert), 0);
    }
}

BOOST_AUTO_TEST_CASE(Graph1CopyReorder) {
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

    CompactSparseGraph<false> oriGraph(11, edges);
    CompactSparseGraph<false> graph(oriGraph);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 11);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 11);

    std::size_t cntr0 = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr0);
        ++cntr0;
    }
    BOOST_CHECK_EQUAL(graph.NumVertices(), cntr0);

    std::vector<std::size_t> perm(11, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graphPerm.cbegin(), graphPerm.cend()));

    std::vector<std::vector<std::size_t>> outEdges({
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
        BOOST_CHECK_EQUAL(graph.out_degree(vert), outEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousChld = 0;
        std::size_t cntr = 0;
        for (const auto &chld : graph.children(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousChld, chld);
            }

            BOOST_CHECK(std::find(outEdges[oriVert].cbegin(), outEdges[oriVert].cend(), graphPerm[chld])
                        != outEdges[oriVert].cend());

            previousChld = chld;
            ++cntr;
        }
    }

    std::vector<std::vector<std::size_t>> inEdges({
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
        BOOST_CHECK_EQUAL(graph.in_degree(vert), inEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousPar = 0;
        std::size_t cntr = 0;
        for (const auto &par : graph.parents(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousPar, par);
            }

            BOOST_CHECK(std::find(inEdges[oriVert].cbegin(), inEdges[oriVert].cend(), graphPerm[par]) != inEdges[oriVert].cend());

            previousPar = par;
            ++cntr;
        }
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 1 + inEdges[graphPerm[vert]].size());
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), 0);
    }
}

BOOST_AUTO_TEST_CASE(Graph1MoveReorder) {
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

    CompactSparseGraph<false> oriGraph(11, edges);
    CompactSparseGraph<false> graph(std::move(oriGraph));

    BOOST_CHECK_EQUAL(graph.NumVertices(), 11);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 11);

    std::size_t cntr0 = 0;
    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr0);
        ++cntr0;
    }
    BOOST_CHECK_EQUAL(graph.NumVertices(), cntr0);

    std::vector<std::size_t> perm(11, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graphPerm.cbegin(), graphPerm.cend()));

    std::vector<std::vector<std::size_t>> outEdges({
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
        BOOST_CHECK_EQUAL(graph.out_degree(vert), outEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousChld = 0;
        std::size_t cntr = 0;
        for (const auto &chld : graph.children(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousChld, chld);
            }

            BOOST_CHECK(std::find(outEdges[oriVert].cbegin(), outEdges[oriVert].cend(), graphPerm[chld])
                        != outEdges[oriVert].cend());

            previousChld = chld;
            ++cntr;
        }
    }

    std::vector<std::vector<std::size_t>> inEdges({
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
        BOOST_CHECK_EQUAL(graph.in_degree(vert), inEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousPar = 0;
        std::size_t cntr = 0;
        for (const auto &par : graph.parents(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousPar, par);
            }

            BOOST_CHECK(std::find(inEdges[oriVert].cbegin(), inEdges[oriVert].cend(), graphPerm[par]) != inEdges[oriVert].cend());

            previousPar = par;
            ++cntr;
        }
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 1 + inEdges[graphPerm[vert]].size());
    }

    for (const auto &vert : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), 0);
    }
}
