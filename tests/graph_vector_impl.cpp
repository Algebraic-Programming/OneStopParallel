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

#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/dag_vector_adapter.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

computational_dag_vector_impl_def_t ConstrGraph1() {
    computational_dag_vector_impl_def_t graph;

    using VertexIdx = computational_dag_vector_impl_def_t::vertex_idx;

    VertexIdx v1 = graph.add_vertex(1, 2, 3, 4);
    VertexIdx v2 = graph.add_vertex(5, 6, 7, 8);
    VertexIdx v3 = graph.add_vertex(9, 10, 11, 12);
    VertexIdx v4 = graph.add_vertex(13, 14, 15, 16);
    VertexIdx v5 = graph.add_vertex(17, 18, 19, 20);
    VertexIdx v6 = graph.add_vertex(21, 22, 23, 24);
    VertexIdx v7 = graph.add_vertex(25, 26, 27, 28);
    VertexIdx v8 = graph.add_vertex(29, 30, 31, 32);

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
}

BOOST_AUTO_TEST_CASE(TestEmptyDag) {
    computational_dag_vector_impl_def_t graph;
    BOOST_CHECK_EQUAL(graph.num_edges(), 0);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 0);

    size_t idx = 0;
    for (const auto &v : graph.vertices()) {
        graph.in_degree(v);
        idx++;
    }
    BOOST_CHECK_EQUAL(idx, 0);
}

BOOST_AUTO_TEST_CASE(TestDag) {
    const computational_dag_vector_impl_def_t graph = ConstrGraph1();

    using VertexIdx = computational_dag_vector_impl_def_t::vertex_idx;

    BOOST_CHECK_EQUAL(graph.num_edges(), 9);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 8);

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
        {4, 3}
    };

    size_t idx = 0;

    for (const auto &v : graph.vertices()) {
        BOOST_CHECK_EQUAL(v, vertices[idx++]);

        size_t i = 0;
        for (const auto &e : graph.children(v)) {
            BOOST_CHECK_EQUAL(e, outNeighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : graph.parents(v)) {
            BOOST_CHECK_EQUAL(e, inNeighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : out_edges(v, graph)) {
            BOOST_CHECK_EQUAL(target(e, graph), outNeighbors[v][i++]);
        }

        i = 0;
        for (const auto &e : in_edges(v, graph)) {
            BOOST_CHECK_EQUAL(source(e, graph), inNeighbors[v][i++]);
        }

        BOOST_CHECK_EQUAL(graph.in_degree(v), inNeighbors[v].size());
        BOOST_CHECK_EQUAL(graph.out_degree(v), outNeighbors[v].size());
    }

    unsigned count = 0;
    for (const auto &e : edges(graph)) {
        std::cout << e.source << " -> " << e.target << std::endl;
        count++;
    }
    BOOST_CHECK_EQUAL(count, 9);
}

BOOST_AUTO_TEST_CASE(TestConstrDag) {
    computational_dag_vector_impl_def_int_t graph;

    graph.add_vertex(1, 2, 3);
    graph.add_vertex(5, 6, 7);
    graph.add_vertex(9, 10, 11);
    graph.add_vertex(13, 14, 15);

    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(0, 3);

    computational_dag_vector_impl_def_int_t graph2(graph);

    BOOST_CHECK_EQUAL(graph2.num_edges(), 3);
    BOOST_CHECK_EQUAL(graph2.num_vertices(), 4);
    BOOST_CHECK_EQUAL(graph2.vertex_work_weight(0), 1);
    BOOST_CHECK_EQUAL(graph2.vertex_comm_weight(0), 2);
    BOOST_CHECK_EQUAL(graph2.vertex_mem_weight(0), 3);
    BOOST_CHECK_EQUAL(graph2.vertex_work_weight(1), 5);
    BOOST_CHECK_EQUAL(graph2.vertex_comm_weight(1), 6);
    BOOST_CHECK_EQUAL(graph2.vertex_mem_weight(1), 7);
    BOOST_CHECK_EQUAL(graph2.vertex_work_weight(2), 9);
    BOOST_CHECK_EQUAL(graph2.vertex_comm_weight(2), 10);
    BOOST_CHECK_EQUAL(graph2.vertex_mem_weight(2), 11);
    BOOST_CHECK_EQUAL(graph2.vertex_work_weight(3), 13);
    BOOST_CHECK_EQUAL(graph2.vertex_comm_weight(3), 14);

    boost_graph_int_t bG1;

    bG1.add_vertex(1, 2, 3, 4);
    bG1.add_vertex(5, 6, 7, 8);

    bG1.add_edge(0, 1, 9);

    computational_dag_vector_impl_def_int_t graph3(bG1);

    BOOST_CHECK_EQUAL(graph3.num_edges(), 1);
    BOOST_CHECK_EQUAL(graph3.num_vertices(), 2);
    BOOST_CHECK_EQUAL(graph3.vertex_work_weight(0), 1);
    BOOST_CHECK_EQUAL(graph3.vertex_comm_weight(0), 2);
    BOOST_CHECK_EQUAL(graph3.vertex_mem_weight(0), 3);
    BOOST_CHECK_EQUAL(graph3.vertex_work_weight(1), 5);
    BOOST_CHECK_EQUAL(graph3.vertex_comm_weight(1), 6);
    BOOST_CHECK_EQUAL(graph3.vertex_mem_weight(1), 7);

    computational_dag_vector_impl_def_int_t graph4(graph3);

    BOOST_CHECK_EQUAL(graph4.num_edges(), 1);
    BOOST_CHECK_EQUAL(graph4.num_vertices(), 2);
    BOOST_CHECK_EQUAL(graph4.vertex_work_weight(0), 1);
    BOOST_CHECK_EQUAL(graph4.vertex_comm_weight(0), 2);
    BOOST_CHECK_EQUAL(graph4.vertex_mem_weight(0), 3);
    BOOST_CHECK_EQUAL(graph4.vertex_work_weight(1), 5);
    BOOST_CHECK_EQUAL(graph4.vertex_comm_weight(1), 6);
    BOOST_CHECK_EQUAL(graph4.vertex_mem_weight(1), 7);

    computational_dag_vector_impl_def_int_t graphMove1(std::move(graph4));

    BOOST_CHECK_EQUAL(graph4.num_edges(), 0);
    BOOST_CHECK_EQUAL(graph4.num_vertices(), 0);

    BOOST_CHECK_EQUAL(graphMove1.num_edges(), 1);
    BOOST_CHECK_EQUAL(graphMove1.num_vertices(), 2);
    BOOST_CHECK_EQUAL(graphMove1.vertex_work_weight(0), 1);
    BOOST_CHECK_EQUAL(graphMove1.vertex_comm_weight(0), 2);
    BOOST_CHECK_EQUAL(graphMove1.vertex_mem_weight(0), 3);
    BOOST_CHECK_EQUAL(graphMove1.vertex_work_weight(1), 5);
    BOOST_CHECK_EQUAL(graphMove1.vertex_comm_weight(1), 6);
    BOOST_CHECK_EQUAL(graphMove1.vertex_mem_weight(1), 7);
}

BOOST_AUTO_TEST_CASE(TestDagVectorAdapter) {
    std::vector<int> vertices{0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<std::vector<int>> outNeighbors{
        {1, 2, 3},
        {4, 6},
        {4, 5},
        {7},
        {7},
        {},
        {},
        {}
    };

    std::vector<std::vector<int>> inNeighbors{
        {},
        {0},
        {0},
        {0},
        {1, 2},
        {2},
        {1},
        {4, 3}
    };

    using VImpl = cdag_vertex_impl<unsigned, int, int, int, unsigned>;
    using GraphT = dag_vector_adapter<VImpl, int>;

    GraphT graph(outNeighbors, inNeighbors);

    size_t idx = 0;

    for (const auto &v : graph.vertices()) {
        BOOST_CHECK_EQUAL(v, vertices[idx++]);

        unsigned vv = static_cast<unsigned>(v);

        size_t i = 0;
        for (const auto &e : graph.children(v)) {
            BOOST_CHECK_EQUAL(e, outNeighbors[vv][i++]);
        }

        i = 0;
        for (const auto &e : graph.parents(v)) {
            BOOST_CHECK_EQUAL(e, inNeighbors[vv][i++]);
        }

        i = 0;
        for (const auto &e : out_edges(v, graph)) {
            BOOST_CHECK_EQUAL(target(e, graph), outNeighbors[vv][i++]);
        }

        i = 0;
        for (const auto &e : in_edges(v, graph)) {
            BOOST_CHECK_EQUAL(source(e, graph), inNeighbors[vv][i++]);
        }

        BOOST_CHECK_EQUAL(graph.in_degree(v), inNeighbors[vv].size());
        BOOST_CHECK_EQUAL(graph.out_degree(v), outNeighbors[vv].size());
    }

    unsigned count = 0;
    for (const auto &e : edges(graph)) {
        std::cout << e.source << " -> " << e.target << std::endl;
        count++;
    }
    BOOST_CHECK_EQUAL(count, 9);
}
