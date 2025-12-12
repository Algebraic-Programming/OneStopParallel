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

BOOST_AUTO_TEST_CASE(EmptyGraphKeepOrder) {
    CompactSparseGraphEdgeDesc<true> graph;

    BOOST_CHECK_EQUAL(graph.NumVertices(), 0);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);

    for (const auto &edge : graph.edges()) {
        BOOST_CHECK(false);
        BOOST_CHECK_EQUAL(edge, 100);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.Source(edge), graph.Traget(edge)));
    }

    for (const auto &vert : graph.Vertices()) {
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

BOOST_AUTO_TEST_CASE(EmptyGraphReorder) {
    CompactSparseGraphEdgeDesc<false> graph;

    BOOST_CHECK_EQUAL(graph.NumVertices(), 0);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);

    for (const auto &edge : graph.edges()) {
        BOOST_CHECK(false);
        BOOST_CHECK_EQUAL(edge, 100);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.Source(edge), graph.Traget(edge)));
    }

    for (const auto &vert : graph.Vertices()) {
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

BOOST_AUTO_TEST_CASE(NoEdgesGraphKeepOrder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({});

    CompactSparseGraphEdgeDesc<true> graph(10, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 10);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);

    for (const auto &edge : graph.edges()) {
        BOOST_CHECK(false);
        BOOST_CHECK_EQUAL(edge, 100);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.Source(edge), graph.Traget(edge)));
    }

    std::size_t vertCounter = 0;
    for (const auto &vert : graph.Vertices()) {
        vertCounter++;

        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
    }

    BOOST_CHECK_EQUAL(vertCounter, graph.NumVertices());
}

BOOST_AUTO_TEST_CASE(NoEdgesGraphReorder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({});

    CompactSparseGraphEdgeDesc<false> graph(10, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 10);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);

    std::size_t vertCounter = 0;
    for (const auto &vert : graph.Vertices()) {
        vertCounter++;

        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK(false);
            BOOST_CHECK_EQUAL(edge, 100);
        }
    }

    BOOST_CHECK_EQUAL(vertCounter, graph.NumVertices());

    std::vector<std::size_t> perm(10, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graphPerm.cbegin(), graphPerm.cend()));
}

BOOST_AUTO_TEST_CASE(LineGraphKeepOrder) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {5, 6},
        {6, 7}
    });

    CompactSparseGraphEdgeDesc<true> graph(8, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 8);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 7);

    std::size_t cntr = 0;
    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr);
        ++cntr;
    }
    BOOST_CHECK_EQUAL(graph.NumVertices(), cntr);

    for (const auto &vert : graph.Vertices()) {
        if (vert != 7) {
            BOOST_CHECK_EQUAL(graph.OutDegree(vert), 1);
            for (const std::size_t &chld : graph.Children(vert)) {
                BOOST_CHECK_EQUAL(chld, vert + 1);
            }
            auto chldren = graph.Children(vert);
            BOOST_CHECK_EQUAL(chldren.crend() - chldren.crbegin(), graph.OutDegree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
                BOOST_CHECK_EQUAL(*it, vert + 1);
            }

        } else {
            BOOST_CHECK_EQUAL(graph.OutDegree(vert), 0);
            for (const std::size_t &chld : graph.Children(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(chld, 100);
            }
            auto chldren = graph.Children(vert);
            BOOST_CHECK_EQUAL(chldren.crend() - chldren.crbegin(), graph.OutDegree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(*it, 100);
            }
        }
    }
    for (const auto &vert : graph.Vertices()) {
        if (vert != 0) {
            BOOST_CHECK_EQUAL(graph.InDegree(vert), 1);
            for (const std::size_t &par : graph.Parents(vert)) {
                BOOST_CHECK_EQUAL(par, vert - 1);
            }
            auto prnts = graph.Parents(vert);
            BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
                BOOST_CHECK_EQUAL(*it, vert - 1);
            }
        } else {
            BOOST_CHECK_EQUAL(graph.InDegree(vert), 0);
            for (const std::size_t &par : graph.Parents(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(par, 100);
            }
            auto prnts = graph.Parents(vert);
            BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(*it, 100);
            }
        }
    }

    for (const auto &vert : graph.Vertices()) {
        if (vert != 0) {
            BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 2);
        } else {
            BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 1);
        }
    }

    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), 0);
    }

    std::size_t edgeCounter = 0;
    for (const auto &edge : graph.edges()) {
        BOOST_CHECK_EQUAL(graph.Source(edge), edgeCounter);
        BOOST_CHECK_EQUAL(graph.Traget(edge), edgeCounter + 1);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.Source(edge), graph.Traget(edge)));

        ++edgeCounter;
    }
    BOOST_CHECK_EQUAL(edgeCounter, graph.NumEdges());

    edgeCounter = 0;
    for (const auto &edge : osp::edges(graph)) {
        BOOST_CHECK_EQUAL(Source(edge, graph), edgeCounter);
        BOOST_CHECK_EQUAL(Traget(edge, graph), edgeCounter + 1);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.Source(edge), graph.Traget(edge)));

        ++edgeCounter;
    }
    BOOST_CHECK_EQUAL(edgeCounter, graph.NumEdges());

    std::size_t vertCounter = 0;
    for (const auto &vert : graph.Vertices()) {
        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.Source(edge), vert - 1);
            BOOST_CHECK_EQUAL(graph.Traget(edge), vert);
        }

        for (const auto &edge : InEdges(vert, graph)) {
            BOOST_CHECK_EQUAL(Source(edge, graph), vert - 1);
            BOOST_CHECK_EQUAL(Traget(edge, graph), vert);
        }

        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.Source(edge), vert);
            BOOST_CHECK_EQUAL(graph.Traget(edge), vert + 1);
        }

        for (const auto &edge : OutEdges(vert, graph)) {
            BOOST_CHECK_EQUAL(Source(edge, graph), vert);
            BOOST_CHECK_EQUAL(Traget(edge, graph), vert + 1);
        }

        ++vertCounter;
    }
    BOOST_CHECK_EQUAL(vertCounter, graph.NumVertices());
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

    CompactSparseGraphEdgeDesc<false> graph(8, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 8);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 7);

    std::size_t cntr = 0;
    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(vert, cntr);
        ++cntr;
    }
    BOOST_CHECK_EQUAL(graph.NumVertices(), cntr);

    for (const auto &vert : graph.Vertices()) {
        if (vert != 7) {
            BOOST_CHECK_EQUAL(graph.OutDegree(vert), 1);
            for (const std::size_t &chld : graph.Children(vert)) {
                BOOST_CHECK_EQUAL(chld, vert + 1);
            }
            auto chldren = graph.Children(vert);
            BOOST_CHECK_EQUAL(chldren.crend() - chldren.crbegin(), graph.OutDegree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
                BOOST_CHECK_EQUAL(*it, vert + 1);
            }
        } else {
            BOOST_CHECK_EQUAL(graph.OutDegree(vert), 0);
            for (const std::size_t &chld : graph.Children(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(chld, 100);
            }
            auto chldren = graph.Children(vert);
            BOOST_CHECK_EQUAL(chldren.crend() - chldren.crbegin(), graph.OutDegree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(*it, 100);
            }
        }
    }
    for (const auto &vert : graph.Vertices()) {
        if (vert != 0) {
            BOOST_CHECK_EQUAL(graph.InDegree(vert), 1);
            for (const std::size_t &par : graph.Parents(vert)) {
                BOOST_CHECK_EQUAL(par, vert - 1);
            }
            auto prnts = graph.Parents(vert);
            BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
                BOOST_CHECK_EQUAL(*it, vert - 1);
            }
        } else {
            BOOST_CHECK_EQUAL(graph.InDegree(vert), 0);
            for (const std::size_t &par : graph.Parents(vert)) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(par, 100);
            }
            auto prnts = graph.Parents(vert);
            BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
                BOOST_CHECK(false);
                BOOST_CHECK_EQUAL(*it, 100);
            }
        }
    }

    for (const auto &vert : graph.Vertices()) {
        if (vert != 0) {
            BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 2);
        } else {
            BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 1);
        }
    }

    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), 0);
    }

    std::vector<std::size_t> perm(8, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graphPerm.cbegin(), graphPerm.cend()));

    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(perm[vert], graphPerm[vert]);
    }

    std::size_t edgeCounter = 0;
    for (const auto &edge : graph.edges()) {
        BOOST_CHECK_EQUAL(graph.Source(edge), edgeCounter);
        BOOST_CHECK_EQUAL(graph.Traget(edge), edgeCounter + 1);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.Source(edge), graph.Traget(edge)));

        ++edgeCounter;
    }
    BOOST_CHECK_EQUAL(edgeCounter, graph.NumEdges());

    std::size_t vertCounter = 0;
    for (const auto &vert : graph.Vertices()) {
        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.Source(edge), vert - 1);
            BOOST_CHECK_EQUAL(graph.Traget(edge), vert);
        }
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.Source(edge), vert);
            BOOST_CHECK_EQUAL(graph.Traget(edge), vert + 1);
        }

        ++vertCounter;
    }
    BOOST_CHECK_EQUAL(vertCounter, graph.NumVertices());
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

    CompactSparseGraphEdgeDesc<true> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 11);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 11);

    std::size_t cntr0 = 0;
    for (const auto &vert : graph.Vertices()) {
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

    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(graph.OutDegree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : graph.Children(vert)) {
            BOOST_CHECK_EQUAL(chld, outEdges[vert][cntr]);
            ++cntr;
        }
        auto chldrn = graph.Children(vert);
        BOOST_CHECK_EQUAL(chldrn.crend() - chldrn.crbegin(), graph.OutDegree(vert));
        for (auto it = chldrn.crbegin(); it != chldrn.crend(); ++it) {
            --cntr;
            BOOST_CHECK_EQUAL(*it, outEdges[vert][cntr]);
        }
    }

    for (const auto &vert : graph.Vertices()) {
        std::size_t cntr = 0;
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.Source(edge), vert);
            BOOST_CHECK_EQUAL(graph.Traget(edge), outEdges[vert][cntr]);
            ++cntr;
        }
        BOOST_CHECK_EQUAL(cntr, graph.OutDegree(vert));
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

    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(graph.InDegree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : graph.Parents(vert)) {
            BOOST_CHECK_EQUAL(par, inEdges[vert][cntr]);
            ++cntr;
        }
        auto prnts = graph.Parents(vert);
        BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            --cntr;
            BOOST_CHECK_EQUAL(*it, inEdges[vert][cntr]);
        }
    }

    for (const auto &vert : graph.Vertices()) {
        std::size_t cntr = 0;
        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.Source(edge), inEdges[vert][cntr]);
            BOOST_CHECK_EQUAL(graph.Traget(edge), vert);
            ++cntr;
        }
        BOOST_CHECK_EQUAL(cntr, graph.InDegree(vert));
    }

    std::size_t edgeCntr = 0;
    for (const auto &edge : graph.edges()) {
        BOOST_CHECK_EQUAL(edge, edgeCntr);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.Source(edge), graph.Traget(edge)));

        ++edgeCntr;
    }
    BOOST_CHECK_EQUAL(edgeCntr, graph.NumEdges());

    edgeCntr = 0;
    for (const auto &vert : graph.Vertices()) {
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(edge, edgeCntr);
            ++edgeCntr;
        }
    }
    BOOST_CHECK_EQUAL(edgeCntr, graph.NumEdges());

    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 1 + inEdges[vert].size());
    }

    for (const auto &vert : graph.Vertices()) {
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

    CompactSparseGraphEdgeDesc<false> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 11);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 11);

    std::size_t cntr0 = 0;
    for (const auto &vert : graph.Vertices()) {
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

    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(graph.OutDegree(vert), outEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousChld = 0;
        std::size_t cntr = 0;
        for (const auto &chld : graph.Children(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousChld, chld);
            }

            BOOST_CHECK(std::find(outEdges[oriVert].cbegin(), outEdges[oriVert].cend(), graphPerm[chld])
                        != outEdges[oriVert].cend());

            previousChld = chld;
            ++cntr;
        }
        auto chldrn = graph.Children(vert);
        BOOST_CHECK_EQUAL(chldrn.crend() - chldrn.crbegin(), graph.OutDegree(vert));
        for (auto it = chldrn.crbegin(); it != chldrn.crend(); ++it) {
            if (cntr < graph.OutDegree(vert)) {
                BOOST_CHECK_GE(previousChld, *it);
            }

            --cntr;
            BOOST_CHECK(std::find(outEdges[oriVert].cbegin(), outEdges[oriVert].cend(), graphPerm[*it])
                        != outEdges[oriVert].cend());

            previousChld = *it;
        }
    }

    for (const auto &vert : graph.Vertices()) {
        std::size_t cntr = 0;
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.Source(edge), vert);
            ++cntr;
        }
        BOOST_CHECK_EQUAL(cntr, graph.OutDegree(vert));
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

    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(graph.InDegree(vert), inEdges[graphPerm[vert]].size());
        std::size_t oriVert = graphPerm[vert];

        std::size_t previousPar = 0;
        std::size_t cntr = 0;
        for (const auto &par : graph.Parents(vert)) {
            if (cntr > 0) {
                BOOST_CHECK_LE(previousPar, par);
            }

            BOOST_CHECK(std::find(inEdges[oriVert].cbegin(), inEdges[oriVert].cend(), graphPerm[par]) != inEdges[oriVert].cend());

            previousPar = par;
            ++cntr;
        }
        auto prnts = graph.Parents(vert);
        BOOST_CHECK_EQUAL(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            if (cntr < graph.OutDegree(vert)) {
                BOOST_CHECK_GE(previousPar, *it);
            }

            --cntr;
            BOOST_CHECK(std::find(inEdges[oriVert].cbegin(), inEdges[oriVert].cend(), graphPerm[*it]) != inEdges[oriVert].cend());

            previousPar = *it;
        }
    }

    for (const auto &vert : graph.Vertices()) {
        std::size_t cntr = 0;
        for (const auto &edge : graph.in_edges(vert)) {
            BOOST_CHECK_EQUAL(graph.Traget(edge), vert);
            ++cntr;
        }
        BOOST_CHECK_EQUAL(cntr, graph.InDegree(vert));
    }

    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), 1 + inEdges[graphPerm[vert]].size());
    }

    for (const auto &vert : graph.Vertices()) {
        BOOST_CHECK_EQUAL(graph.VertexType(vert), 0);
    }

    std::size_t edgeCntr = 0;
    for (const auto &edge : graph.edges()) {
        BOOST_CHECK_EQUAL(edge, edgeCntr);

        BOOST_CHECK_EQUAL(edge, graph.edge(graph.Source(edge), graph.Traget(edge)));

        ++edgeCntr;
    }
    BOOST_CHECK_EQUAL(edgeCntr, graph.NumEdges());

    edgeCntr = 0;
    for (const auto &vert : graph.Vertices()) {
        for (const auto &edge : graph.out_edges(vert)) {
            BOOST_CHECK_EQUAL(edge, edgeCntr);
            ++edgeCntr;
        }
    }
    BOOST_CHECK_EQUAL(edgeCntr, graph.NumEdges());
}

BOOST_AUTO_TEST_CASE(Graph1ECommKeepOrder) {
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
    const std::vector<unsigned> edgeWeights({3, 6, 12, 874, 134, 67, 234, 980, 123, 152, 34});

    CompactSparseGraphEdgeDesc<true, true, true, true, true, true> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 11);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 11);

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const auto &[src, tgt] = edges[i];
        graph.SetEdgeCommWeight(src, tgt, edgeWeights[i]);
    }

    for (const auto &edge : graph.edges()) {
        const auto src = graph.Source(edge);
        const auto tgt = graph.Traget(edge);

        auto it = std::find(edges.cbegin(), edges.cend(), std::make_pair(src, tgt));
        BOOST_CHECK(it != edges.cend());

        auto ind = std::distance(edges.cbegin(), it);
        BOOST_CHECK_EQUAL(edgeWeights[static_cast<std::size_t>(ind)], graph.EdgeCommWeight(edge));
    }
}

BOOST_AUTO_TEST_CASE(Graph1ECommReorder) {
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
    const std::vector<unsigned> edgeWeights({3, 6, 12, 874, 134, 67, 234, 980, 123, 152, 34});

    CompactSparseGraphEdgeDesc<false, true, true, true, true, true> graph(11, edges);

    BOOST_CHECK_EQUAL(graph.NumVertices(), 11);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 11);

    std::vector<std::size_t> perm(11, 0);
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<std::size_t> &graphPerm = graph.GetPullbackPermutation();
    BOOST_CHECK(std::is_permutation(perm.cbegin(), perm.cend(), graphPerm.cbegin(), graphPerm.cend()));

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const auto &[src, tgt] = edges[i];
        graph.SetEdgeCommWeight(src, tgt, edgeWeights[i]);
    }

    for (const auto &edge : graph.edges()) {
        const auto src = graphPerm[graph.Source(edge)];
        const auto tgt = graphPerm[graph.Traget(edge)];

        auto it = std::find(edges.cbegin(), edges.cend(), std::make_pair(src, tgt));
        BOOST_CHECK(it != edges.cend());

        auto ind = std::distance(edges.cbegin(), it);
        BOOST_CHECK_EQUAL(edgeWeights[static_cast<std::size_t>(ind)], graph.EdgeCommWeight(edge));
    }
}
