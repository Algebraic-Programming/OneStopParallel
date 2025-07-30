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

#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"
#include "osp/graph_algorithms/directed_graph_edge_view.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

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

BOOST_AUTO_TEST_CASE(test_empty_graph) {

    computational_dag_vector_impl_def_t graph;

    using vertex_idx = computational_dag_vector_impl_def_t::vertex_idx;

    BOOST_CHECK_EQUAL(graph.num_edges(), 0);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 0);

    std::vector<vertex_idx> sources = source_vertices(graph);
    BOOST_CHECK_EQUAL(sources.size(), 0);

    std::vector<vertex_idx> sinks = sink_vertices(graph);
    BOOST_CHECK_EQUAL(sinks.size(), 0);

    BOOST_CHECK_EQUAL(is_acyclic(graph), true);
    BOOST_CHECK_EQUAL(is_connected(graph), true);
};

BOOST_AUTO_TEST_CASE(test_util_1) {

    computational_dag_vector_impl_def_t graph = constr_graph_1();

    using vertex_idx = computational_dag_vector_impl_def_t::vertex_idx;

    BOOST_CHECK_EQUAL(graph.num_edges(), 9);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 8);

    std::vector<vertex_idx> sources = source_vertices(graph);
    BOOST_CHECK_EQUAL(sources.size(), 1);
    BOOST_CHECK_EQUAL(sources[0], 0);

    std::vector<vertex_idx> sources_s;
    for (const auto &v : source_vertices_view(graph)) {
        sources_s.push_back(v);
    }
    BOOST_CHECK_EQUAL(sources_s.size(), 1);
    BOOST_CHECK_EQUAL(sources_s[0], 0);

    std::vector<vertex_idx> sinks = sink_vertices(graph);
    BOOST_CHECK_EQUAL(sinks.size(), 3);
    BOOST_CHECK_EQUAL(sinks[0], 5);
    BOOST_CHECK_EQUAL(sinks[1], 6);
    BOOST_CHECK_EQUAL(sinks[2], 7);

    std::vector<vertex_idx> sinks_s;
    for (const auto &v : sink_vertices_view(graph)) {
        sinks_s.push_back(v);
    }

    BOOST_CHECK_EQUAL(sinks_s.size(), 3);
    BOOST_CHECK_EQUAL(sinks_s[0], 5);
    BOOST_CHECK_EQUAL(sinks_s[1], 6);
    BOOST_CHECK_EQUAL(sinks_s[2], 7);

    std::vector<vertex_idx> bfs;

    for (const auto &v : bfs_view(graph, 1)) {
        bfs.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfs.size(), 4);
    BOOST_CHECK_EQUAL(bfs[0], 1);
    BOOST_CHECK_EQUAL(bfs[1], 4);
    BOOST_CHECK_EQUAL(bfs[2], 6);
    BOOST_CHECK_EQUAL(bfs[3], 7);

    auto t = successors(1, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfs.begin(), bfs.end(), t.begin(), t.end());

    bfs.clear();

    for (const auto &v : bfs_view(graph, 5)) {
        bfs.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfs.size(), 1);
    BOOST_CHECK_EQUAL(bfs[0], 5);

    t = successors(5, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfs.begin(), bfs.end(), t.begin(), t.end());

    bfs.clear();

    for (const auto &v : bfs_view(graph, 0)) {
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

    t = successors(0, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfs.begin(), bfs.end(), t.begin(), t.end());

    std::vector<vertex_idx> dfs;

    for (const auto &v : dfs_view(graph, 1)) {
        dfs.push_back(v);
    }

    BOOST_CHECK_EQUAL(dfs.size(), 4);
    BOOST_CHECK_EQUAL(dfs[0], 1);
    BOOST_CHECK_EQUAL(dfs[1], 6);
    BOOST_CHECK_EQUAL(dfs[2], 4);
    BOOST_CHECK_EQUAL(dfs[3], 7);

    dfs.clear();
    for (const auto &v : dfs_view(graph, 5)) {
        dfs.push_back(v);
    }

    BOOST_CHECK_EQUAL(dfs.size(), 1);
    BOOST_CHECK_EQUAL(dfs[0], 5);

    dfs.clear();

    for (const auto &v : dfs_view(graph, 0)) {
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

    std::vector<vertex_idx> bfs_reverse;

    for (const auto &v : bfs_reverse_view(graph, 1)) {
        bfs_reverse.push_back(v);
    }
    BOOST_CHECK_EQUAL(bfs_reverse.size(), 2);
    BOOST_CHECK_EQUAL(bfs_reverse[0], 1);
    BOOST_CHECK_EQUAL(bfs_reverse[1], 0);

    t = ancestors(1, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfs_reverse.begin(), bfs_reverse.end(), t.begin(), t.end());

    bfs_reverse.clear();

    for (const auto &v : bfs_reverse_view(graph, 5)) {
        bfs_reverse.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfs_reverse.size(), 3);
    BOOST_CHECK_EQUAL(bfs_reverse[0], 5);
    BOOST_CHECK_EQUAL(bfs_reverse[1], 2);
    BOOST_CHECK_EQUAL(bfs_reverse[2], 0);

    t = ancestors(5, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfs_reverse.begin(), bfs_reverse.end(), t.begin(), t.end());

    bfs_reverse.clear();

    for (const auto &v : bfs_reverse_view(graph, 0)) {
        bfs_reverse.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfs_reverse.size(), 1);
    BOOST_CHECK_EQUAL(bfs_reverse[0], 0);

    t = ancestors(0, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfs_reverse.begin(), bfs_reverse.end(), t.begin(), t.end());

    bfs_reverse.clear();

    for (const auto &v : bfs_reverse_view(graph, 7)) {
        bfs_reverse.push_back(v);
    }

    BOOST_CHECK_EQUAL(bfs_reverse.size(), 6);
    BOOST_CHECK_EQUAL(bfs_reverse[0], 7);
    BOOST_CHECK_EQUAL(bfs_reverse[1], 4);
    BOOST_CHECK_EQUAL(bfs_reverse[2], 3);
    BOOST_CHECK_EQUAL(bfs_reverse[3], 1);
    BOOST_CHECK_EQUAL(bfs_reverse[4], 2);
    BOOST_CHECK_EQUAL(bfs_reverse[5], 0);

    t = ancestors(7, graph);
    BOOST_CHECK_EQUAL_COLLECTIONS(bfs_reverse.begin(), bfs_reverse.end(), t.begin(), t.end());

    BOOST_CHECK_EQUAL(edge(0, 1, graph), true);
    BOOST_CHECK_EQUAL(edge(0, 2, graph), true);
    BOOST_CHECK_EQUAL(edge(0, 3, graph), true);
    BOOST_CHECK_EQUAL(edge(0, 4, graph), false);
    BOOST_CHECK_EQUAL(edge(0, 5, graph), false);
    BOOST_CHECK_EQUAL(edge(0, 6, graph), false);
    BOOST_CHECK_EQUAL(edge(0, 7, graph), false);

    BOOST_CHECK_EQUAL(edge(1, 0, graph), false);
    BOOST_CHECK_EQUAL(edge(1, 1, graph), false);
    BOOST_CHECK_EQUAL(edge(1, 2, graph), false);
    BOOST_CHECK_EQUAL(edge(1, 3, graph), false);
    BOOST_CHECK_EQUAL(edge(1, 4, graph), true);
    BOOST_CHECK_EQUAL(edge(1, 5, graph), false);
    BOOST_CHECK_EQUAL(edge(1, 6, graph), true);
    BOOST_CHECK_EQUAL(edge(1, 7, graph), false);

    BOOST_CHECK_EQUAL(edge(2, 0, graph), false);
    BOOST_CHECK_EQUAL(edge(2, 1, graph), false);
    BOOST_CHECK_EQUAL(edge(2, 2, graph), false);
    BOOST_CHECK_EQUAL(edge(2, 3, graph), false);
    BOOST_CHECK_EQUAL(edge(2, 4, graph), true);
    BOOST_CHECK_EQUAL(edge(2, 5, graph), true);
    BOOST_CHECK_EQUAL(edge(2, 6, graph), false);
    BOOST_CHECK_EQUAL(edge(2, 7, graph), false);

    BOOST_CHECK_EQUAL(edge(3, 0, graph), false);
    BOOST_CHECK_EQUAL(edge(3, 1, graph), false);
    BOOST_CHECK_EQUAL(edge(3, 2, graph), false);
    BOOST_CHECK_EQUAL(edge(3, 3, graph), false);
    BOOST_CHECK_EQUAL(edge(3, 4, graph), false);
    BOOST_CHECK_EQUAL(edge(3, 5, graph), false);
    BOOST_CHECK_EQUAL(edge(3, 6, graph), false);
    BOOST_CHECK_EQUAL(edge(3, 7, graph), true);

    BOOST_CHECK_EQUAL(edge(4, 0, graph), false);
    BOOST_CHECK_EQUAL(edge(4, 1, graph), false);
    BOOST_CHECK_EQUAL(edge(4, 2, graph), false);
    BOOST_CHECK_EQUAL(edge(4, 3, graph), false);
    BOOST_CHECK_EQUAL(edge(4, 4, graph), false);
    BOOST_CHECK_EQUAL(edge(4, 5, graph), false);
    BOOST_CHECK_EQUAL(edge(4, 6, graph), false);
    BOOST_CHECK_EQUAL(edge(4, 7, graph), true);

    BOOST_CHECK_EQUAL(edge(5, 0, graph), false);
    BOOST_CHECK_EQUAL(edge(5, 1, graph), false);
    BOOST_CHECK_EQUAL(edge(5, 2, graph), false);
    BOOST_CHECK_EQUAL(edge(5, 3, graph), false);
    BOOST_CHECK_EQUAL(edge(5, 4, graph), false);
    BOOST_CHECK_EQUAL(edge(5, 5, graph), false);
    BOOST_CHECK_EQUAL(edge(5, 6, graph), false);
    BOOST_CHECK_EQUAL(edge(5, 7, graph), false);

    BOOST_CHECK_EQUAL(edge(6, 0, graph), false);
    BOOST_CHECK_EQUAL(edge(6, 1, graph), false);
    BOOST_CHECK_EQUAL(edge(6, 2, graph), false);
    BOOST_CHECK_EQUAL(edge(6, 3, graph), false);
    BOOST_CHECK_EQUAL(edge(6, 4, graph), false);
    BOOST_CHECK_EQUAL(edge(6, 5, graph), false);
    BOOST_CHECK_EQUAL(edge(6, 6, graph), false);
    BOOST_CHECK_EQUAL(edge(6, 7, graph), false);

    BOOST_CHECK_EQUAL(edge(7, 0, graph), false);
    BOOST_CHECK_EQUAL(edge(7, 1, graph), false);
    BOOST_CHECK_EQUAL(edge(7, 2, graph), false);
    BOOST_CHECK_EQUAL(edge(7, 3, graph), false);
    BOOST_CHECK_EQUAL(edge(7, 4, graph), false);
    BOOST_CHECK_EQUAL(edge(7, 5, graph), false);
    BOOST_CHECK_EQUAL(edge(7, 6, graph), false);

    BOOST_CHECK_EQUAL(is_source(0, graph), true);
    BOOST_CHECK_EQUAL(is_source(1, graph), false);
    BOOST_CHECK_EQUAL(is_source(2, graph), false);
    BOOST_CHECK_EQUAL(is_source(3, graph), false);
    BOOST_CHECK_EQUAL(is_source(4, graph), false);
    BOOST_CHECK_EQUAL(is_source(5, graph), false);
    BOOST_CHECK_EQUAL(is_source(6, graph), false);
    BOOST_CHECK_EQUAL(is_source(7, graph), false);

    BOOST_CHECK_EQUAL(is_sink(0, graph), false);
    BOOST_CHECK_EQUAL(is_sink(1, graph), false);
    BOOST_CHECK_EQUAL(is_sink(2, graph), false);
    BOOST_CHECK_EQUAL(is_sink(3, graph), false);
    BOOST_CHECK_EQUAL(is_sink(4, graph), false);
    BOOST_CHECK_EQUAL(is_sink(5, graph), true);
    BOOST_CHECK_EQUAL(is_sink(6, graph), true);
    BOOST_CHECK_EQUAL(is_sink(7, graph), true);

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

    std::vector<vertex_idx> edge_source = {0, 0, 0, 1, 1, 2, 2, 3, 4};
    std::vector<vertex_idx> edge_target = {1, 2, 3, 4, 6, 4, 5, 7, 7};

    size_t i = 0;
    for (const auto &e : edge_view(graph)) {

        BOOST_CHECK_EQUAL(e.source, edge_source[i]);
        BOOST_CHECK_EQUAL(e.target, edge_target[i]);

        ++i;
    }

    BOOST_CHECK_EQUAL(is_acyclic(graph), true);
    BOOST_CHECK_EQUAL(is_connected(graph), true);

    graph.add_edge(7, 5);
    BOOST_CHECK_EQUAL(is_acyclic(graph), true);
    graph.add_edge(7, 0);
    BOOST_CHECK_EQUAL(is_acyclic(graph), false);

    graph.add_vertex(1, 2, 3, 4);
    BOOST_CHECK_EQUAL(is_connected(graph), false);
};

BOOST_AUTO_TEST_CASE(ComputationalDagConstructor) {

    using VertexType = vertex_idx_t<boost_graph_int_t>;

    const std::vector<std::vector<VertexType>> out(

        {{7}, {}, {0}, {2}, {}, {2, 0}, {1, 2, 0}, {}, {4}, {6, 1, 5}}

    );
    const std::vector<int> workW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});
    const std::vector<int> commW({1, 1, 1, 1, 2, 3, 2, 1, 1, 1});

    const boost_graph_int_t graph(out, workW, commW);
    const boost_graph_int_t graph_empty;

    BOOST_CHECK_EQUAL(graph.num_edges(), 12);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 10);
    BOOST_CHECK_EQUAL(graph_empty.num_edges(), 0);
    BOOST_CHECK_EQUAL(graph_empty.num_vertices(), 0);
    BOOST_CHECK_EQUAL(graph.num_vertex_types(), 1);

    BOOST_CHECK_EQUAL(is_acyclic(graph), true);
    BOOST_CHECK_EQUAL(is_acyclic(graph_empty), true);
    BOOST_CHECK_EQUAL(is_connected(graph), false);
    BOOST_CHECK_EQUAL(is_connected(graph_empty), true);

    const auto long_edges = long_edges_in_triangles(graph);

    BOOST_CHECK_EQUAL(graph.num_vertices(), std::distance(graph.vertices().begin(), graph.vertices().end()));
    BOOST_CHECK_EQUAL(graph.num_edges(), std::distance(edges(graph).begin(), edges(graph).end()));
    for (const auto &v : graph.vertices()) {
        BOOST_CHECK_EQUAL(graph.in_degree(v), std::distance(graph.parents(v).begin(), graph.parents(v).end()));
        BOOST_CHECK_EQUAL(graph.out_degree(v), std::distance(graph.children(v).begin(), graph.children(v).end()));
    }

    for (const auto i : graph.vertices()) {
        const auto v = graph.get_boost_graph()[i];
        BOOST_CHECK_EQUAL(v.workWeight, workW[i]);
        BOOST_CHECK_EQUAL(v.workWeight, graph.vertex_work_weight(i));
        BOOST_CHECK_EQUAL(v.communicationWeight, commW[i]);
        BOOST_CHECK_EQUAL(v.communicationWeight, graph.vertex_comm_weight(i));
    }

    BOOST_CHECK_EQUAL(sumOfVerticesWorkWeights({0, 1}, graph), 2);
    {
        int sum_of_work_weights = graph.vertex_work_weight(0) + graph.vertex_work_weight(1);
        BOOST_CHECK_EQUAL(2, sum_of_work_weights);
    }
    BOOST_CHECK_EQUAL(sumOfVerticesWorkWeights({5, 3}, graph), 4);
    BOOST_CHECK_EQUAL(sumOfVerticesWorkWeights({}, graph), 0);
    BOOST_CHECK_EQUAL(sumOfVerticesWorkWeights({0, 1, 2, 3, 4, 5}, graph), 9);

    BOOST_CHECK_EQUAL(sumOfVerticesWorkWeights({}, graph_empty), 0);

    std::size_t num_edges = 0;
    for (const auto &vertex : graph.vertices()) {
        num_edges += graph.out_degree(vertex);
        for (const auto &parent : graph.parents(vertex)) {
            BOOST_CHECK(std::any_of(graph.children(parent).cbegin(), graph.children(parent).cend(),
                                    [vertex](VertexType k) { return k == vertex; }));
        }
    }

    for (const auto &vertex : graph.vertices()) {
        for (const auto &child : graph.children(vertex)) {

            BOOST_CHECK(std::any_of(graph.parents(child).cbegin(), graph.parents(child).cend(),
                                    [vertex](VertexType k) { return k == vertex; }));
        }
    }

    std::vector<VertexType> top_order = GetTopOrder(graph);
    BOOST_CHECK(top_order.size() == graph.num_vertices());
    BOOST_CHECK(GetTopOrder(graph_empty).size() == graph_empty.num_vertices());

    std::vector<size_t> index_in_top_order = sorting_arrangement(top_order);

    for (const auto &i : top_order) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    top_order = GetTopOrderMaxChildren(graph);
    BOOST_CHECK(top_order.size() == graph.num_vertices());
    BOOST_CHECK(GetTopOrder(graph_empty).size() == graph_empty.num_vertices());

    index_in_top_order = sorting_arrangement(top_order);

    for (const auto &i : top_order) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    top_order = GetTopOrderRandom(graph);
    BOOST_CHECK(top_order.size() == graph.num_vertices());
    BOOST_CHECK(GetTopOrderRandom(graph_empty).size() == graph_empty.num_vertices());

    index_in_top_order = sorting_arrangement(top_order);

    for (const auto &i : top_order) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    top_order = GetTopOrderMinIndex(graph);
    BOOST_CHECK(top_order.size() == graph.num_vertices());
    BOOST_CHECK(GetTopOrderMinIndex(graph_empty).size() == graph_empty.num_vertices());

    index_in_top_order = sorting_arrangement(top_order);

    for (const auto &i : top_order) {
        for (const auto &j : graph.children(i)) {
            BOOST_CHECK_LT(index_in_top_order[i], index_in_top_order[j]);
        }
    }

    std::set<VertexType> all_nodes;
    for (const auto &vertex : graph.vertices()) {
        all_nodes.emplace(vertex);
    }
    std::set<VertexType> nodes_a({8, 0});
    std::set<VertexType> nodes_b({6, 2, 5, 3});
    std::set<VertexType> nodes_c({6, 9, 1});

    std::vector<bool> bool_a(graph.num_vertices(), false);
    std::vector<bool> bool_b(graph.num_vertices(), false);
    std::vector<bool> bool_c(graph.num_vertices(), false);

    for (auto &i : nodes_a) {
        bool_a[i] = true;
    }
    for (auto &i : nodes_b) {
        bool_b[i] = true;
    }
    for (auto &i : nodes_c) {
        bool_c[i] = true;
    }

    BOOST_CHECK(GetFilteredTopOrder(bool_a, graph) == std::vector<VertexType>({0, 8}) ||
                GetFilteredTopOrder(bool_a, graph) == std::vector<VertexType>({8, 0}));
    BOOST_CHECK(GetFilteredTopOrder(bool_b, graph)[3] == 2);
    BOOST_CHECK(GetFilteredTopOrder(bool_c, graph) == std::vector<VertexType>({9, 6, 1}));

    BOOST_CHECK_EQUAL(longestPath(all_nodes, graph), 4);
    BOOST_CHECK_EQUAL(longestPath(nodes_a, graph), 0);
    BOOST_CHECK_EQUAL(longestPath(nodes_b, graph), 1);
    BOOST_CHECK_EQUAL(longestPath(nodes_c, graph), 2);

    BOOST_CHECK_EQUAL(longestPath({}, graph_empty), 0);

    std::vector<VertexType> longest_path = longestChain(graph);

    std::vector<VertexType> long_chain1({9, 6, 2, 0, 7});
    std::vector<VertexType> long_chain2({9, 5, 2, 0, 7});

    BOOST_CHECK_EQUAL(longestPath(all_nodes, graph) + 1, longestChain(graph).size());
    BOOST_CHECK(longest_path == long_chain1 || longest_path == long_chain2);

    BOOST_CHECK(longestChain(graph_empty) == std::vector<VertexType>({}));

    BOOST_CHECK(ancestors(9, graph) == std::vector<VertexType>({9}));
    BOOST_CHECK(ancestors(2, graph) == std::vector<VertexType>({2, 3, 5, 6, 9}));
    BOOST_CHECK(ancestors(4, graph) == std::vector<VertexType>({4, 8}));
    BOOST_CHECK(ancestors(5, graph) == std::vector<VertexType>({5, 9}));
    BOOST_CHECK(successors(9, graph) == std::vector<VertexType>({9, 6, 1, 5, 2, 0, 7}));
    BOOST_CHECK(successors(3, graph) == std::vector<VertexType>({3, 2, 0, 7}));
    BOOST_CHECK(successors(0, graph) == std::vector<VertexType>({0, 7}));
    BOOST_CHECK(successors(8, graph) == std::vector<VertexType>({8, 4}));
    BOOST_CHECK(successors(4, graph) == std::vector<VertexType>({4}));

    std::vector<unsigned> top_dist({4, 3, 3, 1, 2, 2, 2, 5, 1, 1});
    std::vector<unsigned> bottom_dist({2, 1, 3, 4, 1, 4, 4, 1, 2, 5});

    BOOST_CHECK(get_top_node_distance(graph) == top_dist);
    BOOST_CHECK(get_bottom_node_distance(graph) == bottom_dist);

    const std::vector<std::vector<VertexType>> graph_second_Out = {
        {1, 2}, {3, 4}, {4, 5}, {6}, {}, {6}, {},
    };
    const std::vector<int> graph_second_workW = {1, 1, 1, 1, 1, 1, 3};
    const std::vector<int> graph_second_commW = graph_second_workW;

    boost_graph_int_t graph_second(graph_second_Out, graph_second_workW, graph_second_commW);

    std::vector<unsigned> top_dist_second({1, 2, 2, 3, 3, 3, 4});
    std::vector<unsigned> bottom_dist_second({4, 3, 3, 2, 1, 2, 1});

    BOOST_CHECK(get_top_node_distance(graph_second) == top_dist_second);
    BOOST_CHECK(get_bottom_node_distance(graph_second) == bottom_dist_second);

    std::vector<double> poisson_params({0.0000001, 0.08, 0.1, 0.2, 0.5, 1, 4});

    for (unsigned loops = 0; loops < 10; loops++) {
        for (unsigned noise = 0; noise < 6; noise++) {
            for (auto &pois_para : poisson_params) {

                std::vector<int> poset_int_map = get_strict_poset_integer_map(noise, pois_para, graph);

                for (const auto &vertex : graph.vertices()) {
                    for (const auto &child : graph.children(vertex)) {
                        BOOST_CHECK_LE(poset_int_map[vertex] + 1, poset_int_map[child]);
                    }
                }
            }
        }
    }

    BOOST_CHECK(critical_path_weight(graph) == 7);

    auto wavefronts = compute_wavefronts(graph);

    std::vector<std::vector<VertexType>> expected_wavefronts = {{3, 8, 9}, {4, 6, 5}, {1, 2}, {0}, {7}};

    size_t size = 0;
    size_t counter = 0;
    for (const auto &wavefront : wavefronts) {
        size += wavefront.size();
        BOOST_CHECK(!wavefront.empty());

        BOOST_CHECK_EQUAL_COLLECTIONS(wavefront.begin(), wavefront.end(), expected_wavefronts[counter].begin(),
                                      expected_wavefronts[counter].end());

        counter++;
    }

    BOOST_CHECK_EQUAL(size, graph.num_vertices());

    // const std::pair<std::vector<VertexType>, ComputationalDag> rev_graph_pair = graph.reverse_graph();
    // const std::vector<VertexType> &vertex_mapping_rev_graph = rev_graph_pair.first;
    // const ComputationalDag &rev_graph = rev_graph_pair.second;

    // BOOST_CHECK_EQUAL(graph.numberOfVertices(), rev_graph.numberOfVertices());
    // BOOST_CHECK_EQUAL(graph.numberOfEdges(), rev_graph.numberOfEdges());

    // for (VertexType vert = 0; vert < graph.numberOfVertices(); vert++) {
    //     BOOST_CHECK_EQUAL(graph.nodeWorkWeight(vert), rev_graph.nodeWorkWeight(vertex_mapping_rev_graph[vert]));
    //     BOOST_CHECK_EQUAL(graph.nodeCommunicationWeight(vert),
    //     rev_graph.nodeCommunicationWeight(vertex_mapping_rev_graph[vert]));
    //     BOOST_CHECK_EQUAL(graph.nodeMemoryWeight(vert), rev_graph.nodeMemoryWeight(vertex_mapping_rev_graph[vert]));
    //     BOOST_CHECK_EQUAL(graph.nodeType(vert), rev_graph.nodeType(vertex_mapping_rev_graph[vert]));
    // }

    // for (VertexType vert_1 = 0; vert_1 < graph.numberOfVertices(); vert_1++) {
    //     for (VertexType vert_2 = 0; vert_2 < graph.numberOfVertices(); vert_2++) {
    //         bool edge_in_graph = boost::edge(vert_1, vert_2, graph.getGraph()).second;
    //         bool rev_edge_in_rev_graph = boost::edge(vertex_mapping_rev_graph[vert_2],
    //         vertex_mapping_rev_graph[vert_1], rev_graph.getGraph()).second; BOOST_CHECK_EQUAL(edge_in_graph,
    //         rev_edge_in_rev_graph);
    //     }
    // }
}