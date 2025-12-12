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

#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

boost_graph_int_t ConstrGraph1() {
    boost_graph_int_t graph;

    using VertexIdx = boost_graph_int_t::vertex_idx;

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

    return graph;
}

BOOST_AUTO_TEST_CASE(TestEmptyDagBoostGraphAdapter) {
    boost_graph_int_t graph;
    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);
    BOOST_CHECK_EQUAL(graph.NumVertices(), 0);
}

BOOST_AUTO_TEST_CASE(TestBoostGraphAdapter1) {
    boost_graph_int_t graph = ConstrGraph1();

    using VertexIdx = boost_graph_int_t::vertex_idx;

    std::vector<VertexIdx> edgeSources{0, 0, 0, 1, 1, 2, 2, 3, 4};
    std::vector<VertexIdx> edgeTargets{1, 2, 3, 4, 6, 4, 5, 7, 7};

    size_t edgeIdx = 0;
    for (const auto &edge : graph.edges()) {
        BOOST_CHECK_EQUAL(source(edge, graph), edgeSources[edgeIdx]);
        BOOST_CHECK_EQUAL(target(edge, graph), edgeTargets[edgeIdx]);
        edgeIdx++;
    }

    edgeIdx = 0;
    for (const auto &edge : edges(graph)) {
        BOOST_CHECK_EQUAL(source(edge, graph), edgeSources[edgeIdx]);
        BOOST_CHECK_EQUAL(target(edge, graph), edgeTargets[edgeIdx]);
        edgeIdx++;
    }

    std::vector<VertexIdx> vertices{0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<std::vector<VertexIdx>> outNeighbors{
        {1, 2, 3},
        {4, 6},
        {4, 5},
        {7},
        {7},
        {},
        {},
        {}
    };

    std::vector<std::vector<VertexIdx>> inNeighbors{
        {},
        {0},
        {0},
        {0},
        {1, 2},
        {2},
        {1},
        {3, 4}
    };

    size_t idx = 0;

    for (const auto &v : graph.vertices()) {
        BOOST_CHECK_EQUAL(v, vertices[idx++]);

        size_t i = 0;
        for (const auto &e : graph.children(v)) {
            BOOST_CHECK_EQUAL(e, outNeighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.out_edges(v)) {
            BOOST_CHECK_EQUAL(target(e, graph), outNeighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.out_edges(v)) {
            BOOST_CHECK_EQUAL(graph.target(e), outNeighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.parents(v)) {
            BOOST_CHECK_EQUAL(e, inNeighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.in_edges(v)) {
            BOOST_CHECK_EQUAL(source(e, graph), inNeighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.in_edges(v)) {
            BOOST_CHECK_EQUAL(graph.source(e), inNeighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : in_edges(v, graph)) {
            BOOST_CHECK_EQUAL(source(e, graph), inNeighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : out_edges(v, graph)) {
            BOOST_CHECK_EQUAL(target(e, graph), outNeighbors[v][i++]);
        }

        BOOST_CHECK_EQUAL(graph.in_degree(v), inNeighbors[v].size());
        BOOST_CHECK_EQUAL(graph.out_degree(v), outNeighbors[v].size());
    }
}

BOOST_AUTO_TEST_CASE(TestUtil1) {
    const boost_graph_int_t graph = ConstrGraph1();

    BOOST_CHECK_EQUAL(graph.NumEdges(), 9);
    BOOST_CHECK_EQUAL(graph.NumVertices(), 8);

    auto sources = source_vertices(graph);
    BOOST_CHECK_EQUAL(sources.size(), 1);
    BOOST_CHECK_EQUAL(sources[0], 0);

    auto sinks = sink_vertices(graph);
    BOOST_CHECK_EQUAL(sinks.size(), 3);
    BOOST_CHECK_EQUAL(sinks[0], 5);
    BOOST_CHECK_EQUAL(sinks[1], 6);
    BOOST_CHECK_EQUAL(sinks[2], 7);

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
}

BOOST_AUTO_TEST_CASE(TestConstrDag) {
    boost_graph_int_t graph;

    graph.add_vertex(1, 2, 3);
    graph.add_vertex(5, 6, 7);
    graph.add_vertex(9, 10, 11);
    graph.add_vertex(13, 14, 15);

    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(0, 3);

    boost_graph_int_t graph2(graph);

    BOOST_CHECK_EQUAL(graph2.NumEdges(), 3);
    BOOST_CHECK_EQUAL(graph2.NumVertices(), 4);
    BOOST_CHECK_EQUAL(graph2.VertexWorkWeight(0), 1);
    BOOST_CHECK_EQUAL(graph2.VertexCommWeight(0), 2);
    BOOST_CHECK_EQUAL(graph2.VertexMemWeight(0), 3);
    BOOST_CHECK_EQUAL(graph2.VertexWorkWeight(1), 5);
    BOOST_CHECK_EQUAL(graph2.VertexCommWeight(1), 6);
    BOOST_CHECK_EQUAL(graph2.VertexMemWeight(1), 7);
    BOOST_CHECK_EQUAL(graph2.VertexWorkWeight(2), 9);
    BOOST_CHECK_EQUAL(graph2.VertexCommWeight(2), 10);
    BOOST_CHECK_EQUAL(graph2.VertexMemWeight(2), 11);
    BOOST_CHECK_EQUAL(graph2.VertexWorkWeight(3), 13);
    BOOST_CHECK_EQUAL(graph2.VertexCommWeight(3), 14);

    computational_dag_edge_idx_vector_impl_def_int_t graphOther;

    graphOther.add_vertex(1, 2, 3, 4);
    graphOther.add_vertex(5, 6, 7, 8);
    graphOther.add_edge(0, 1, 9);

    boost_graph_int_t graph3(graphOther);

    BOOST_CHECK_EQUAL(graph3.NumEdges(), 1);
    BOOST_CHECK_EQUAL(graph3.NumVertices(), 2);
    BOOST_CHECK_EQUAL(graph3.VertexWorkWeight(0), 1);
    BOOST_CHECK_EQUAL(graph3.VertexCommWeight(0), 2);
    BOOST_CHECK_EQUAL(graph3.VertexMemWeight(0), 3);
    BOOST_CHECK_EQUAL(graph3.VertexWorkWeight(1), 5);
    BOOST_CHECK_EQUAL(graph3.VertexCommWeight(1), 6);
    BOOST_CHECK_EQUAL(graph3.VertexMemWeight(1), 7);
}

BOOST_AUTO_TEST_CASE(TestBoostGraphConst1) {
    boost_graph_int_t graph(10u);
    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);
    BOOST_CHECK_EQUAL(graph.NumVertices(), 10);
}

BOOST_AUTO_TEST_CASE(TestBoostGraphConst2) {
    boost_graph_int_t graph1 = ConstrGraph1();

    boost_graph_int_t graphCopy(graph1);
    BOOST_CHECK_EQUAL(graphCopy.NumEdges(), 9);
    BOOST_CHECK_EQUAL(graphCopy.NumVertices(), 8);

    BOOST_CHECK_EQUAL(has_path(2, 7, graphCopy), true);
    BOOST_CHECK_EQUAL(has_path(3, 7, graphCopy), true);
    BOOST_CHECK_EQUAL(has_path(4, 7, graphCopy), true);
    BOOST_CHECK_EQUAL(has_path(1, 2, graphCopy), false);
    BOOST_CHECK_EQUAL(has_path(1, 3, graphCopy), false);
    BOOST_CHECK_EQUAL(has_path(2, 1, graphCopy), false);

    boost_graph_int_t graphCopy2 = graph1;

    BOOST_CHECK_EQUAL(graph1.NumEdges(), 9);
    BOOST_CHECK_EQUAL(graph1.NumVertices(), 8);

    BOOST_CHECK_EQUAL(has_path(2, 7, graph1), true);
    BOOST_CHECK_EQUAL(has_path(3, 7, graph1), true);
    BOOST_CHECK_EQUAL(has_path(4, 7, graph1), true);
    BOOST_CHECK_EQUAL(has_path(1, 2, graph1), false);
    BOOST_CHECK_EQUAL(has_path(1, 3, graph1), false);
    BOOST_CHECK_EQUAL(has_path(2, 1, graph1), false);

    BOOST_CHECK_EQUAL(graphCopy2.NumEdges(), 9);
    BOOST_CHECK_EQUAL(graphCopy2.NumVertices(), 8);

    BOOST_CHECK_EQUAL(has_path(2, 7, graphCopy2), true);
    BOOST_CHECK_EQUAL(has_path(3, 7, graphCopy2), true);
    BOOST_CHECK_EQUAL(has_path(4, 7, graphCopy2), true);
    BOOST_CHECK_EQUAL(has_path(1, 2, graphCopy2), false);
    BOOST_CHECK_EQUAL(has_path(1, 3, graphCopy2), false);
    BOOST_CHECK_EQUAL(has_path(2, 1, graphCopy2), false);

    boost_graph_int_t graphMove1(std::move(graphCopy));

    BOOST_CHECK_EQUAL(graphCopy.NumEdges(), 0);
    BOOST_CHECK_EQUAL(graphCopy.NumVertices(), 0);

    BOOST_CHECK_EQUAL(graphMove1.NumEdges(), 9);
    BOOST_CHECK_EQUAL(graphMove1.NumVertices(), 8);

    BOOST_CHECK_EQUAL(has_path(2, 7, graphMove1), true);
    BOOST_CHECK_EQUAL(has_path(3, 7, graphMove1), true);
    BOOST_CHECK_EQUAL(has_path(4, 7, graphMove1), true);
    BOOST_CHECK_EQUAL(has_path(1, 2, graphMove1), false);
    BOOST_CHECK_EQUAL(has_path(1, 3, graphMove1), false);
    BOOST_CHECK_EQUAL(has_path(2, 1, graphMove1), false);

    boost_graph_int_t graphMove2 = std::move(graphCopy2);
    BOOST_CHECK_EQUAL(graphCopy2.NumEdges(), 0);
    BOOST_CHECK_EQUAL(graphCopy2.NumVertices(), 0);

    BOOST_CHECK_EQUAL(graphMove2.NumEdges(), 9);
    BOOST_CHECK_EQUAL(graphMove2.NumVertices(), 8);

    BOOST_CHECK_EQUAL(has_path(2, 7, graphMove2), true);
    BOOST_CHECK_EQUAL(has_path(3, 7, graphMove2), true);
    BOOST_CHECK_EQUAL(has_path(4, 7, graphMove2), true);
    BOOST_CHECK_EQUAL(has_path(1, 2, graphMove2), false);
    BOOST_CHECK_EQUAL(has_path(1, 3, graphMove2), false);
    BOOST_CHECK_EQUAL(has_path(2, 1, graphMove2), false);
}
