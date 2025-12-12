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

@author Toni Boehnlein, Pal Andras Papp, Raphael S. Steiner, Christos Konstantinos Matzoros
*/

#define BOOST_TEST_MODULE SubGraphs
#include <boost/test/unit_test.hpp>

#include "osp/graph_algorithms/specialised_graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_implementations/adj_list_impl/cdag_vertex_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(SubGraphCompactSparseGraph) {
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
    CompactSparseGraph<true, true, true, true, true> graph(11, edges);
    CompactSparseGraph<true, true, true, true, true> subGraph;

    unsigned cntr = 0;
    for (const auto &vert : graph.vertices()) {
        graph.SetVertexWorkWeight(vert, cntr++);
        graph.SetVertexCommWeight(vert, cntr++);
        graph.SetVertexMemWeight(vert, cntr++);
        graph.SetVertexType(vert, cntr++);
    }

    const std::vector<VertexIdxT<Compact_Sparse_Graph<true, true, true, true, true>>> selectVert({2, 3, 10, 6, 7});
    const auto vertCorrespondence = create_induced_subgraph_map(graph, subGraph, selectVert);
    BOOST_CHECK_EQUAL(subGraph.NumVertices(), selectVert.size());
    BOOST_CHECK_EQUAL(subGraph.NumEdges(), 4);

    for (const auto &vert : selectVert) {
        BOOST_CHECK_LT(vertCorrespondence.at(vert), selectVert.size());

        for (const auto &otherVert : selectVert) {
            if (vertCorrespondence.at(vert) == vertCorrespondence.at(otherVert)) {
                BOOST_CHECK_EQUAL(vert, otherVert);
            }
        }
    }

    for (const auto &vert : selectVert) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), subGraph.VertexWorkWeight(vertCorrespondence.at(vert)));
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), subGraph.VertexCommWeight(vertCorrespondence.at(vert)));
        BOOST_CHECK_EQUAL(graph.VertexMemWeight(vert), subGraph.VertexMemWeight(vertCorrespondence.at(vert)));
        BOOST_CHECK_EQUAL(graph.VertexType(vert), subGraph.VertexType(vertCorrespondence.at(vert)));
    }
}

BOOST_AUTO_TEST_CASE(SubGraphDagVectorImpl) {
    using VImpl = cdag_vertex_impl<std::size_t, unsigned, unsigned, unsigned, unsigned>;

    computational_dag_vector_impl<VImpl> graph;
    computational_dag_vector_impl<VImpl> subGraph;

    const std::size_t numVert = 11;
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

    unsigned cntr = 0;
    for (std::size_t i = 0U; i < numVert; ++i) {
        graph.add_vertex(cntr, cntr + 1U, cntr + 2U, cntr + 3U);
        cntr += 4U;
    }
    for (const auto &[src, tgt] : edges) {
        graph.add_edge(src, tgt);
    }

    const std::vector<VertexIdxT<computational_dag_vector_impl<VImpl>>> selectVert({2, 3, 10, 6, 7});
    const auto vertCorrespondence = create_induced_subgraph_map(graph, subGraph, selectVert);
    BOOST_CHECK_EQUAL(subGraph.NumVertices(), selectVert.size());
    BOOST_CHECK_EQUAL(subGraph.NumEdges(), 4);

    for (const auto &vert : selectVert) {
        BOOST_CHECK_LT(vertCorrespondence.at(vert), selectVert.size());

        for (const auto &otherVert : selectVert) {
            if (vertCorrespondence.at(vert) == vertCorrespondence.at(otherVert)) {
                BOOST_CHECK_EQUAL(vert, otherVert);
            }
        }
    }

    for (const auto &vert : selectVert) {
        BOOST_CHECK_EQUAL(graph.VertexWorkWeight(vert), subGraph.VertexWorkWeight(vertCorrespondence.at(vert)));
        BOOST_CHECK_EQUAL(graph.VertexCommWeight(vert), subGraph.VertexCommWeight(vertCorrespondence.at(vert)));
        BOOST_CHECK_EQUAL(graph.VertexMemWeight(vert), subGraph.VertexMemWeight(vertCorrespondence.at(vert)));
        BOOST_CHECK_EQUAL(graph.VertexType(vert), subGraph.VertexType(vertCorrespondence.at(vert)));
    }
}
