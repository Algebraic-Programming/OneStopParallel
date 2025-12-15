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

#include "osp/graph_algorithms/directed_graph_util.hpp"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_edge_view.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

ComputationalDagVectorImplDefUnsignedT ConstrGraph1() {
    ComputationalDagVectorImplDefUnsignedT graph;

    using VertexIdx = ComputationalDagVectorImplDefUnsignedT::VertexIdx;

    VertexIdx v1 = graph.AddVertex(1, 2, 3, 4);
    VertexIdx v2 = graph.AddVertex(5, 6, 7, 8);
    VertexIdx v3 = graph.AddVertex(9, 10, 11, 12);
    VertexIdx v4 = graph.AddVertex(13, 14, 15, 16);
    VertexIdx v5 = graph.AddVertex(17, 18, 19, 20);
    VertexIdx v6 = graph.AddVertex(21, 22, 23, 24);
    VertexIdx v7 = graph.AddVertex(25, 26, 27, 28);
    VertexIdx v8 = graph.AddVertex(29, 30, 31, 32);

    graph.AddEdge(v1, v2);
    graph.AddEdge(v1, v3);
    graph.AddEdge(v1, v4);
    graph.AddEdge(v2, v5);

    graph.AddEdge(v3, v5);
    graph.AddEdge(v3, v6);
    graph.AddEdge(v2, v7);
    graph.AddEdge(v5, v8);
    graph.AddEdge(v4, v8);

    return graph;
}

BOOST_AUTO_TEST_CASE(TestEmptyGraph) {
    ComputationalDagVectorImplDefUnsignedT graph;

    using VertexIdx = ComputationalDagVectorImplDefUnsignedT::VertexIdx;

    BOOST_CHECK_EQUAL(graph.NumEdges(), 0);
    BOOST_CHECK_EQUAL(graph.NumVertices(), 0);

    std::vector<VertexIdx> sources = SourceVertices(graph);
    BOOST_CHECK_EQUAL(sources.size(), 0);

    std::vector<VertexIdx> sinks = SinkVertices(graph);
    BOOST_CHECK_EQUAL(sinks.size(), 0);

    BOOST_CHECK_EQUAL(IsAcyclic(graph), true);
    BOOST_CHECK_EQUAL(IsConnected(graph), true);
}

BOOST_AUTO_TEST_CASE(TestUtil1) {
    ComputationalDagVectorImplDefUnsignedT graph = ConstrGraph1();

    using VertexIdx = ComputationalDagVectorImplDefUnsignedT::VertexIdx;

    BOOST_CHECK_EQUAL(graph.NumEdges(), 9);
    BOOST_CHECK_EQUAL(graph.NumVertices(), 8);

    std::vector<VertexIdx> sources = SourceVertices(graph);
    BOOST_CHECK_EQUAL(sources.size(), 1);
    BOOST_CHECK_EQUAL(sources[0], 0);

    std::vector<VertexIdx> sourcesS;
    for (const auto &v : SourceVerticesView(graph)) {
        sourcesS.push_back(v);
    }
    BOOST_CHECK_EQUAL(sourcesS.size(), 1);
    BOOST_CHECK_EQUAL(sourcesS[0], 0);

    std::vector<VertexIdx> sinks = SinkVertices(graph);
    BOOST_CHECK_EQUAL(sinks.size(), 3);
    BOOST_CHECK_EQUAL(sinks[0], 5);
    BOOST_CHECK_EQUAL(sinks[1], 6);
    BOOST_CHECK_EQUAL(sinks[2], 7);

    std::vector<VertexIdx> sinksS;
    for (const auto &v : SinkVerticesView(graph)) {
        sinksS.push_back(v);
    }

    BOOST_CHECK_EQUAL(sinksS.size(), 3);
    BOOST_CHECK_EQUAL(sinksS[0], 5);
    BOOST_CHECK_EQUAL(sinksS[1], 6);
    BOOST_CHECK_EQUAL(sinksS[2], 7);

    std::vector<VertexIdx> bfs;

    for (const auto &v : BfsView(graph, 1)) {
        bfs.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfs.size(), 4);
    BOOST_CHECK_EQUAL(bfs[0], 1);
    BOOST_CHECK_EQUAL(bfs[1], 4);
    BOOST_CHECK_EQUAL(bfs[2], 6);
    BOOST_CHECK_EQUAL(bfs[3], 7);

    auto t = Successors(1, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfs.begin(), bfs.end(), t.begin(), t.end());

    bfs.clear();

    for (const auto &v : BfsView(graph, 5)) {
        bfs.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfs.size(), 1);
    BOOST_CHECK_EQUAL(bfs[0], 5);

    t = Successors(5, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfs.begin(), bfs.end(), t.begin(), t.end());

    bfs.clear();

    for (const auto &v : BfsView(graph, 0)) {
        bfs.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfs.size(), 8);
    BOOST_CHECK_EQUAL(bfs[0], 0);
    BOOST_CHECK_EQUAL(bfs[1], 1);
    BOOST_CHECK_EQUAL(bfs[2], 2);
    BOOST_CHECK_EQUAL(bfs[3], 3);
    BOOST_CHECK_EQUAL(bfs[4], 4);
    BOOST_CHECK_EQUAL(bfs[5], 6);
    BOOST_CHECK_EQUAL(bfs[6], 5);
    BOOST_CHECK_EQUAL(bfs[7], 7);

    t = Successors(0, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfs.begin(), bfs.end(), t.begin(), t.end());

    std::vector<VertexIdx> dfs;

    for (const auto &v : DfsView(graph, 1)) {
        dfs.push_back(v);
    }

    BOOST_CHECK_EQUAL(dfs.size(), 4);
    BOOST_CHECK_EQUAL(dfs[0], 1);
    BOOST_CHECK_EQUAL(dfs[1], 6);
    BOOST_CHECK_EQUAL(dfs[2], 4);
    BOOST_CHECK_EQUAL(dfs[3], 7);

    dfs.clear();
    for (const auto &v : DfsView(graph, 5)) {
        dfs.push_back(v);
    }

    BOOST_CHECK_EQUAL(dfs.size(), 1);
    BOOST_CHECK_EQUAL(dfs[0], 5);

    dfs.clear();

    for (const auto &v : DfsView(graph, 0)) {
        dfs.push_back(v);
    }

    BOOST_CHECK_EQUAL(dfs.size(), 8);
    BOOST_CHECK_EQUAL(dfs[0], 0);
    BOOST_CHECK_EQUAL(dfs[1], 3);
    BOOST_CHECK_EQUAL(dfs[2], 7);
    BOOST_CHECK_EQUAL(dfs[3], 2);
    BOOST_CHECK_EQUAL(dfs[4], 5);
    BOOST_CHECK_EQUAL(dfs[5], 4);
    BOOST_CHECK_EQUAL(dfs[6], 1);
    BOOST_CHECK_EQUAL(dfs[7], 6);

    std::vector<VertexIdx> bfsReverse;

    for (const auto &v : BfsReverseView(graph, 1)) {
        bfsReverse.push_back(v);
    }
    BOOST_CHECK_EQUAL(bfsReverse.size(), 2);
    BOOST_CHECK_EQUAL(bfsReverse[0], 1);
    BOOST_CHECK_EQUAL(bfsReverse[1], 0);

    t = Ancestors(1, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfsReverse.begin(), bfsReverse.end(), t.begin(), t.end());

    bfsReverse.clear();

    for (const auto &v : BfsReverseView(graph, 5)) {
        bfsReverse.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfsReverse.size(), 3);
    BOOST_CHECK_EQUAL(bfsReverse[0], 5);
    BOOST_CHECK_EQUAL(bfsReverse[1], 2);
    BOOST_CHECK_EQUAL(bfsReverse[2], 0);

    t = Ancestors(5, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfsReverse.begin(), bfsReverse.end(), t.begin(), t.end());

    bfsReverse.clear();

    for (const auto &v : BfsReverseView(graph, 0)) {
        bfsReverse.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfsReverse.size(), 1);
    BOOST_CHECK_EQUAL(bfsReverse[0], 0);

    t = Ancestors(0, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfsReverse.begin(), bfsReverse.end(), t.begin(), t.end());

    bfsReverse.clear();

    for (const auto &v : BfsReverseView(graph, 7)) {
        bfsReverse.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfsReverse.size(), 6);
    BOOST_CHECK_EQUAL(bfsReverse[0], 7);
    BOOST_CHECK_EQUAL(bfsReverse[1], 4);
    BOOST_CHECK_EQUAL(bfsReverse[2], 3);
    BOOST_CHECK_EQUAL(bfsReverse[3], 1);
    BOOST_CHECK_EQUAL(bfsReverse[4], 2);
    BOOST_CHECK_EQUAL(bfsReverse[5], 0);

    t = Ancestors(7, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfsReverse.begin(), bfsReverse.end(), t.begin(), t.end());

    BOOST_CHECK_EQUAL(Edge(0, 1, graph), true);
    BOOST_CHECK_EQUAL(Edge(0, 2, graph), true);
    BOOST_CHECK_EQUAL(Edge(0, 3, graph), true);
    BOOST_CHECK_EQUAL(Edge(0, 4, graph), false);
    BOOST_CHECK_EQUAL(Edge(0, 5, graph), false);
    BOOST_CHECK_EQUAL(Edge(0, 6, graph), false);
    BOOST_CHECK_EQUAL(Edge(0, 7, graph), false);

    BOOST_CHECK_EQUAL(Edge(1, 0, graph), false);
    BOOST_CHECK_EQUAL(Edge(1, 1, graph), false);
    BOOST_CHECK_EQUAL(Edge(1, 2, graph), false);
    BOOST_CHECK_EQUAL(Edge(1, 3, graph), false);
    BOOST_CHECK_EQUAL(Edge(1, 4, graph), true);
    BOOST_CHECK_EQUAL(Edge(1, 5, graph), false);
    BOOST_CHECK_EQUAL(Edge(1, 6, graph), true);
    BOOST_CHECK_EQUAL(Edge(1, 7, graph), false);

    BOOST_CHECK_EQUAL(Edge(2, 0, graph), false);
    BOOST_CHECK_EQUAL(Edge(2, 1, graph), false);
    BOOST_CHECK_EQUAL(Edge(2, 2, graph), false);
    BOOST_CHECK_EQUAL(Edge(2, 3, graph), false);
    BOOST_CHECK_EQUAL(Edge(2, 4, graph), true);
    BOOST_CHECK_EQUAL(Edge(2, 5, graph), true);
    BOOST_CHECK_EQUAL(Edge(2, 6, graph), false);
    BOOST_CHECK_EQUAL(Edge(2, 7, graph), false);

    BOOST_CHECK_EQUAL(Edge(3, 0, graph), false);
    BOOST_CHECK_EQUAL(Edge(3, 1, graph), false);
    BOOST_CHECK_EQUAL(Edge(3, 2, graph), false);
    BOOST_CHECK_EQUAL(Edge(3, 3, graph), false);
    BOOST_CHECK_EQUAL(Edge(3, 4, graph), false);
    BOOST_CHECK_EQUAL(Edge(3, 5, graph), false);
    BOOST_CHECK_EQUAL(Edge(3, 6, graph), false);
    BOOST_CHECK_EQUAL(Edge(3, 7, graph), true);

    BOOST_CHECK_EQUAL(Edge(4, 0, graph), false);
    BOOST_CHECK_EQUAL(Edge(4, 1, graph), false);
    BOOST_CHECK_EQUAL(Edge(4, 2, graph), false);
    BOOST_CHECK_EQUAL(Edge(4, 3, graph), false);
    BOOST_CHECK_EQUAL(Edge(4, 4, graph), false);
    BOOST_CHECK_EQUAL(Edge(4, 5, graph), false);
    BOOST_CHECK_EQUAL(Edge(4, 6, graph), false);
    BOOST_CHECK_EQUAL(Edge(4, 7, graph), true);

    BOOST_CHECK_EQUAL(Edge(5, 0, graph), false);
    BOOST_CHECK_EQUAL(Edge(5, 1, graph), false);
    BOOST_CHECK_EQUAL(Edge(5, 2, graph), false);
    BOOST_CHECK_EQUAL(Edge(5, 3, graph), false);
    BOOST_CHECK_EQUAL(Edge(5, 4, graph), false);
    BOOST_CHECK_EQUAL(Edge(5, 5, graph), false);
    BOOST_CHECK_EQUAL(Edge(5, 6, graph), false);
    BOOST_CHECK_EQUAL(Edge(5, 7, graph), false);

    BOOST_CHECK_EQUAL(Edge(6, 0, graph), false);
    BOOST_CHECK_EQUAL(Edge(6, 1, graph), false);
    BOOST_CHECK_EQUAL(Edge(6, 2, graph), false);
    BOOST_CHECK_EQUAL(Edge(6, 3, graph), false);
    BOOST_CHECK_EQUAL(Edge(6, 4, graph), false);
    BOOST_CHECK_EQUAL(Edge(6, 5, graph), false);
    BOOST_CHECK_EQUAL(Edge(6, 6, graph), false);
    BOOST_CHECK_EQUAL(Edge(6, 7, graph), false);

    BOOST_CHECK_EQUAL(Edge(7, 0, graph), false);
    BOOST_CHECK_EQUAL(Edge(7, 1, graph), false);
    BOOST_CHECK_EQUAL(Edge(7, 2, graph), false);
    BOOST_CHECK_EQUAL(Edge(7, 3, graph), false);
    BOOST_CHECK_EQUAL(Edge(7, 4, graph), false);
    BOOST_CHECK_EQUAL(Edge(7, 5, graph), false);
    BOOST_CHECK_EQUAL(Edge(7, 6, graph), false);

    BOOST_CHECK_EQUAL(IsSource(0, graph), true);
    BOOST_CHECK_EQUAL(IsSource(1, graph), false);
    BOOST_CHECK_EQUAL(IsSource(2, graph), false);
    BOOST_CHECK_EQUAL(IsSource(3, graph), false);
    BOOST_CHECK_EQUAL(IsSource(4, graph), false);
    BOOST_CHECK_EQUAL(IsSource(5, graph), false);
    BOOST_CHECK_EQUAL(IsSource(6, graph), false);
    BOOST_CHECK_EQUAL(IsSource(7, graph), false);

    BOOST_CHECK_EQUAL(IsSink(0, graph), false);
    BOOST_CHECK_EQUAL(IsSink(1, graph), false);
    BOOST_CHECK_EQUAL(IsSink(2, graph), false);
    BOOST_CHECK_EQUAL(IsSink(3, graph), false);
    BOOST_CHECK_EQUAL(IsSink(4, graph), false);
    BOOST_CHECK_EQUAL(IsSink(5, graph), true);
    BOOST_CHECK_EQUAL(IsSink(6, graph), true);
    BOOST_CHECK_EQUAL(IsSink(7, graph), true);

    BOOST_CHECK_EQUAL(HasPath(0, 1, graph), true);
    BOOST_CHECK_EQUAL(HasPath(0, 2, graph), true);
    BOOST_CHECK_EQUAL(HasPath(0, 3, graph), true);
    BOOST_CHECK_EQUAL(HasPath(0, 4, graph), true);
    BOOST_CHECK_EQUAL(HasPath(0, 5, graph), true);
    BOOST_CHECK_EQUAL(HasPath(0, 6, graph), true);
    BOOST_CHECK_EQUAL(HasPath(0, 7, graph), true);
    BOOST_CHECK_EQUAL(HasPath(1, 0, graph), false);
    BOOST_CHECK_EQUAL(HasPath(2, 0, graph), false);
    BOOST_CHECK_EQUAL(HasPath(3, 0, graph), false);
    BOOST_CHECK_EQUAL(HasPath(4, 0, graph), false);
    BOOST_CHECK_EQUAL(HasPath(5, 0, graph), false);
    BOOST_CHECK_EQUAL(HasPath(6, 0, graph), false);
    BOOST_CHECK_EQUAL(HasPath(7, 0, graph), false);
    BOOST_CHECK_EQUAL(HasPath(1, 4, graph), true);
    BOOST_CHECK_EQUAL(HasPath(1, 7, graph), true);
    BOOST_CHECK_EQUAL(HasPath(1, 6, graph), true);
    BOOST_CHECK_EQUAL(HasPath(2, 4, graph), true);
    BOOST_CHECK_EQUAL(HasPath(2, 5, graph), true);
    BOOST_CHECK_EQUAL(HasPath(2, 7, graph), true);
    BOOST_CHECK_EQUAL(HasPath(3, 7, graph), true);
    BOOST_CHECK_EQUAL(HasPath(4, 7, graph), true);
    BOOST_CHECK_EQUAL(HasPath(1, 2, graph), false);
    BOOST_CHECK_EQUAL(HasPath(1, 3, graph), false);
    BOOST_CHECK_EQUAL(HasPath(2, 1, graph), false);
    BOOST_CHECK_EQUAL(HasPath(2, 3, graph), false);
    BOOST_CHECK_EQUAL(HasPath(2, 6, graph), false);
    BOOST_CHECK_EQUAL(HasPath(3, 1, graph), false);
    BOOST_CHECK_EQUAL(HasPath(3, 2, graph), false);
    BOOST_CHECK_EQUAL(HasPath(3, 4, graph), false);
    BOOST_CHECK_EQUAL(HasPath(3, 5, graph), false);
    BOOST_CHECK_EQUAL(HasPath(3, 6, graph), false);
    BOOST_CHECK_EQUAL(HasPath(4, 1, graph), false);
    BOOST_CHECK_EQUAL(HasPath(4, 2, graph), false);
    BOOST_CHECK_EQUAL(HasPath(4, 3, graph), false);
    BOOST_CHECK_EQUAL(HasPath(4, 5, graph), false);
    BOOST_CHECK_EQUAL(HasPath(4, 6, graph), false);
    BOOST_CHECK_EQUAL(HasPath(5, 1, graph), false);
    BOOST_CHECK_EQUAL(HasPath(5, 2, graph), false);
    BOOST_CHECK_EQUAL(HasPath(5, 3, graph), false);
    BOOST_CHECK_EQUAL(HasPath(5, 4, graph), false);
    BOOST_CHECK_EQUAL(HasPath(5, 6, graph), false);
    BOOST_CHECK_EQUAL(HasPath(5, 7, graph), false);
    BOOST_CHECK_EQUAL(HasPath(6, 1, graph), false);
    BOOST_CHECK_EQUAL(HasPath(6, 2, graph), false);
    BOOST_CHECK_EQUAL(HasPath(6, 3, graph), false);
    BOOST_CHECK_EQUAL(HasPath(6, 4, graph), false);
    BOOST_CHECK_EQUAL(HasPath(6, 5, graph), false);
    BOOST_CHECK_EQUAL(HasPath(6, 7, graph), false);
    BOOST_CHECK_EQUAL(HasPath(7, 1, graph), false);
    BOOST_CHECK_EQUAL(HasPath(7, 2, graph), false);
    BOOST_CHECK_EQUAL(HasPath(7, 3, graph), false);
    BOOST_CHECK_EQUAL(HasPath(7, 4, graph), false);
    BOOST_CHECK_EQUAL(HasPath(7, 5, graph), false);
    BOOST_CHECK_EQUAL(HasPath(7, 6, graph), false);

    std::vector<VertexIdx> edgeSource = {0, 0, 0, 1, 1, 2, 2, 3, 4};
    std::vector<VertexIdx> edgeTarget = {1, 2, 3, 4, 6, 4, 5, 7, 7};

    size_t i = 0;
    for (const auto &e : EdgeView(graph)) {
        BOOST_CHECK_EQUAL(e.source, edgeSource[i]);
        BOOST_CHECK_EQUAL(e.target, edgeTarget[i]);

        ++i;
    }

    BOOST_CHECK_EQUAL(IsAcyclic(graph), true);
    BOOST_CHECK_EQUAL(IsConnected(graph), true);

    graph.AddEdge(7, 5);
    BOOST_CHECK_EQUAL(IsAcyclic(graph), true);
    graph.AddEdge(7, 0);
    BOOST_CHECK_EQUAL(IsAcyclic(graph), false);

    graph.AddVertex(1, 2, 3, 4);
    BOOST_CHECK_EQUAL(IsConnected(graph), false);
}

BOOST_AUTO_TEST_CASE(ComputationalDagConstructor) {
    using VertexType = VertexIdxT<BoostGraphIntT>;

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

    const BoostGraphIntT graph(out, workW, commW);
    const BoostGraphIntT graphEmpty;

    BOOST_CHECK_EQUAL(graph.NumEdges(), 12);
    BOOST_CHECK_EQUAL(graph.NumVertices(), 10);
    BOOST_CHECK_EQUAL(graphEmpty.NumEdges(), 0);
    BOOST_CHECK_EQUAL(graphEmpty.NumVertices(), 0);
    BOOST_CHECK_EQUAL(graph.NumVertexTypes(), 1);

    BOOST_CHECK_EQUAL(IsAcyclic(graph), true);
    BOOST_CHECK_EQUAL(IsAcyclic(graphEmpty), true);
    BOOST_CHECK_EQUAL(IsConnected(graph), false);
    BOOST_CHECK_EQUAL(IsConnected(graphEmpty), true);

    const auto longEdges = LongEdgesInTriangles(graph);

    BOOST_CHECK_EQUAL(graph.NumVertices(), std::distance(graph.Vertices().begin(), graph.Vertices().end()));
    BOOST_CHECK_EQUAL(graph.NumEdges(), std::distance(Edges(graph).begin(), Edges(graph).end()));
    for (const auto &v : graph.Vertices()) {
        BOOST_CHECK_EQUAL(graph.InDegree(v), std::distance(graph.Parents(v).begin(), graph.Parents(v).end()));
        BOOST_CHECK_EQUAL(graph.OutDegree(v), std::distance(graph.Children(v).begin(), graph.Children(v).end()));
    }

    for (const auto i : graph.Vertices()) {
        const auto v = graph.GetBoostGraph()[i];
        BOOST_CHECK_EQUAL(v.workWeight, workW[i]);
        BOOST_CHECK_EQUAL(v.workWeight, graph.VertexWorkWeight(i));
        BOOST_CHECK_EQUAL(v.communicationWeight, commW[i]);
        BOOST_CHECK_EQUAL(v.communicationWeight, graph.VertexCommWeight(i));
    }

    BOOST_CHECK_EQUAL(sumOfVerticesWorkWeights({0, 1}, graph), 2);
    {
        int sumOfWorkWeights = graph.VertexWorkWeight(0) + graph.VertexWorkWeight(1);
        BOOST_CHECK_EQUAL(2, sumOfWorkWeights);
    }
    BOOST_CHECK_EQUAL(SumOfVerticesWorkWeights({5, 3}, graph), 4);
    BOOST_CHECK_EQUAL(SumOfVerticesWorkWeights({}, graph), 0);
    BOOST_CHECK_EQUAL(SumOfVerticesWorkWeights({0, 1, 2, 3, 4, 5}, graph), 9);

    BOOST_CHECK_EQUAL(SumOfVerticesWorkWeights({}, graphEmpty), 0);

    std::size_t numEdges = 0;
    for (const auto &vertex : graph.Vertices()) {
        numEdges += graph.OutDegree(vertex);
        for (const auto &parent : graph.Parents(vertex)) {
            BOOST_CHECK(std::any_of(
                graph.Children(parent).cbegin(), graph.Children(parent).cend(), [vertex](VertexType k) { return k == vertex; }));
        }
    }

    for (const auto &vertex : graph.Vertices()) {
        for (const auto &child : graph.Children(vertex)) {
            BOOST_CHECK(std::any_of(
                graph.Parents(child).cbegin(), graph.Parents(child).cend(), [vertex](VertexType k) { return k == vertex; }));
        }
    }

    std::vector<VertexType> topOrder = GetTopOrder(graph);
    BOOST_CHECK(topOrder.size() == graph.NumVertices());
    BOOST_CHECK(GetTopOrder(graphEmpty).size() == graphEmpty.NumVertices());

    std::vector<size_t> indexInTopOrder = SortingArrangement(topOrder);

    for (const auto &i : topOrder) {
        for (const auto &j : graph.Children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    topOrder = GetTopOrderMaxChildren(graph);
    BOOST_CHECK(topOrder.size() == graph.NumVertices());
    BOOST_CHECK(GetTopOrder(graphEmpty).size() == graphEmpty.NumVertices());

    indexInTopOrder = SortingArrangement(topOrder);

    for (const auto &i : topOrder) {
        for (const auto &j : graph.Children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    topOrder = GetTopOrderRandom(graph);
    BOOST_CHECK(topOrder.size() == graph.NumVertices());
    BOOST_CHECK(GetTopOrderRandom(graphEmpty).size() == graphEmpty.NumVertices());

    indexInTopOrder = SortingArrangement(topOrder);

    for (const auto &i : topOrder) {
        for (const auto &j : graph.Children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    topOrder = GetTopOrderMinIndex(graph);
    BOOST_CHECK(topOrder.size() == graph.NumVertices());
    BOOST_CHECK(GetTopOrderMinIndex(graphEmpty).size() == graphEmpty.NumVertices());

    indexInTopOrder = SortingArrangement(topOrder);

    for (const auto &i : topOrder) {
        for (const auto &j : graph.Children(i)) {
            BOOST_CHECK_LT(indexInTopOrder[i], indexInTopOrder[j]);
        }
    }

    std::set<VertexType> allNodes;
    for (const auto &vertex : graph.Vertices()) {
        allNodes.emplace(vertex);
    }
    std::set<VertexType> nodesA({8, 0});
    std::set<VertexType> nodesB({6, 2, 5, 3});
    std::set<VertexType> nodesC({6, 9, 1});

    std::vector<bool> boolA(graph.NumVertices(), false);
    std::vector<bool> boolB(graph.NumVertices(), false);
    std::vector<bool> boolC(graph.NumVertices(), false);

    for (auto &i : nodesA) {
        boolA[i] = true;
    }
    for (auto &i : nodesB) {
        boolB[i] = true;
    }
    for (auto &i : nodesC) {
        boolC[i] = true;
    }

    BOOST_CHECK(GetFilteredTopOrder(boolA, graph) == std::vector<VertexType>({0, 8})
                || GetFilteredTopOrder(boolA, graph) == std::vector<VertexType>({8, 0}));
    BOOST_CHECK(GetFilteredTopOrder(boolB, graph)[3] == 2);
    BOOST_CHECK(GetFilteredTopOrder(boolC, graph) == std::vector<VertexType>({9, 6, 1}));

    BOOST_CHECK_EQUAL(LongestPath(allNodes, graph), 4);
    BOOST_CHECK_EQUAL(LongestPath(nodesA, graph), 0);
    BOOST_CHECK_EQUAL(LongestPath(nodesB, graph), 1);
    BOOST_CHECK_EQUAL(LongestPath(nodesC, graph), 2);

    BOOST_CHECK_EQUAL(LongestPath({}, graphEmpty), 0);

    std::vector<VertexType> longestPath = LongestChain(graph);

    std::vector<VertexType> longChain1({9, 6, 2, 0, 7});
    std::vector<VertexType> longChain2({9, 5, 2, 0, 7});

    BOOST_CHECK_EQUAL(LongestPath(allNodes, graph) + 1, LongestChain(graph).size());
    BOOST_CHECK(longestPath == longChain1 || longestPath == longChain2);

    BOOST_CHECK(LongestChain(graphEmpty) == std::vector<VertexType>({}));

    BOOST_CHECK(Ancestors(9, graph) == std::vector<VertexType>({9}));
    BOOST_CHECK(Ancestors(2, graph) == std::vector<VertexType>({2, 3, 5, 6, 9}));
    BOOST_CHECK(Ancestors(4, graph) == std::vector<VertexType>({4, 8}));
    BOOST_CHECK(Ancestors(5, graph) == std::vector<VertexType>({5, 9}));
    BOOST_CHECK(Successors(9, graph) == std::vector<VertexType>({9, 6, 1, 5, 2, 0, 7}));
    BOOST_CHECK(Successors(3, graph) == std::vector<VertexType>({3, 2, 0, 7}));
    BOOST_CHECK(Successors(0, graph) == std::vector<VertexType>({0, 7}));
    BOOST_CHECK(Successors(8, graph) == std::vector<VertexType>({8, 4}));
    BOOST_CHECK(Successors(4, graph) == std::vector<VertexType>({4}));

    std::vector<unsigned> topDist({4, 3, 3, 1, 2, 2, 2, 5, 1, 1});
    std::vector<unsigned> bottomDist({2, 1, 3, 4, 1, 4, 4, 1, 2, 5});

    BOOST_CHECK(GetTopNodeDistance(graph) == topDist);
    BOOST_CHECK(GetBottomNodeDistance(graph) == bottomDist);

    const std::vector<std::vector<VertexType>> graphSecondOut = {
        {1, 2},
        {3, 4},
        {4, 5},
        {6},
        {},
        {6},
        {},
    };
    const std::vector<int> graphSecondWorkW = {1, 1, 1, 1, 1, 1, 3};
    const std::vector<int> graphSecondCommW = graphSecondWorkW;

    BoostGraphIntT graphSecond(graphSecondOut, graphSecondWorkW, graphSecondCommW);

    std::vector<unsigned> topDistSecond({1, 2, 2, 3, 3, 3, 4});
    std::vector<unsigned> bottomDistSecond({4, 3, 3, 2, 1, 2, 1});

    BOOST_CHECK(GetTopNodeDistance(graphSecond) == topDistSecond);
    BOOST_CHECK(GetBottomNodeDistance(graphSecond) == bottomDistSecond);

    std::vector<double> poissonParams({0.0000001, 0.08, 0.1, 0.2, 0.5, 1, 4});

    for (unsigned loops = 0; loops < 10; loops++) {
        for (unsigned noise = 0; noise < 6; noise++) {
            for (auto &poisPara : poissonParams) {
                std::vector<int> posetIntMap = get_strict_poset_integer_map(noise, poisPara, graph);

                for (const auto &vertex : graph.Vertices()) {
                    for (const auto &child : graph.Children(vertex)) {
                        BOOST_CHECK_LE(posetIntMap[vertex] + 1, posetIntMap[child]);
                    }
                }
            }
        }
    }

    BOOST_CHECK(CriticalPathWeight(graph) == 7);

    auto wavefronts = ComputeWavefronts(graph);

    std::vector<std::vector<VertexType>> expectedWavefronts = {
        {3, 8, 9},
        {4, 6, 5},
        {1, 2},
        {0},
        {7}
    };

    size_t size = 0;
    size_t counter = 0;
    for (const auto &wavefront : wavefronts) {
        size += wavefront.size();
        BOOST_CHECK(!wavefront.empty());

        BOOST_CHECK_EQUAL_COLLECTIONS(
            wavefront.begin(), wavefront.end(), expectedWavefronts[counter].begin(), expectedWavefronts[counter].end());

        counter++;
    }

    BOOST_CHECK_EQUAL(size, graph.NumVertices());

    // const std::pair<std::vector<VertexType>, ComputationalDag> rev_graph_pair = graph.reverse_graph();
    // const std::vector<VertexType> &vertex_mapping_rev_graph = rev_graph_pair.first;
    // const ComputationalDag &rev_graph = rev_graph_pair.second;

    // BOOST_CHECK_EQUAL(graph.NumberOfVertices(), rev_graph.NumberOfVertices());
    // BOOST_CHECK_EQUAL(graph.numberOfEdges(), rev_graph.numberOfEdges());

    // for (VertexType vert = 0; vert < graph.NumberOfVertices(); vert++) {
    //     BOOST_CHECK_EQUAL(graph.nodeWorkWeight(vert), rev_graph.nodeWorkWeight(vertex_mapping_rev_graph[vert]));
    //     BOOST_CHECK_EQUAL(graph.nodeCommunicationWeight(vert),
    //     rev_graph.nodeCommunicationWeight(vertex_mapping_rev_graph[vert]));
    //     BOOST_CHECK_EQUAL(graph.nodeMemoryWeight(vert), rev_graph.nodeMemoryWeight(vertex_mapping_rev_graph[vert]));
    //     BOOST_CHECK_EQUAL(graph.nodeType(vert), rev_graph.nodeType(vertex_mapping_rev_graph[vert]));
    // }

    // for (VertexType vert_1 = 0; vert_1 < graph.NumberOfVertices(); vert_1++) {
    //     for (VertexType vert_2 = 0; vert_2 < graph.NumberOfVertices(); vert_2++) {
    //         bool edge_in_graph = boost::edge(vert_1, vert_2, graph.getGraph()).second;
    //         bool rev_edge_in_rev_graph = boost::edge(vertex_mapping_rev_graph[vert_2],
    //         vertex_mapping_rev_graph[vert_1], rev_graph.getGraph()).second; BOOST_CHECK_EQUAL(edge_in_graph,
    //         rev_edge_in_rev_graph);
    //     }
    // }
}

BOOST_AUTO_TEST_CASE(TestEdgeViewIndexedAccess) {
    ComputationalDagVectorImplDefUnsignedT graph = ConstrGraph1();
    auto allEdges = EdgeView(graph);

    // Check initial iterator
    auto it = allEdges.begin();

    // Check each edge by index
    for (size_t i = 0; i < graph.NumEdges(); ++i) {
        // Construct iterator directly to index i
        auto indexedIt = decltype(allEdges)::iterator(i, graph);
        BOOST_CHECK(indexedIt == it);
        BOOST_CHECK(*indexedIt == *it);

        ++it;
    }

    // Check end condition
    auto endIt = decltype(allEdges)::iterator(graph.NumEdges(), graph);
    BOOST_CHECK(endIt == allEdges.end());

    // Check out of bounds
    auto oobIt = decltype(allEdges)::iterator(graph.NumEdges() + 5, graph);
    BOOST_CHECK(oobIt == allEdges.end());
}
