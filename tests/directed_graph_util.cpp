#define BOOST_TEST_MODULE ApproxEdgeReduction

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

#include "graph_algorithms/directed_graph_util.hpp"
#include "graph_implementations/computational_dag_vector_impl.hpp"

using namespace osp;

computational_dag_vector_impl<cdag_vertex_impl> constr_graph_1() {

    computational_dag_vector_impl graph;

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

BOOST_AUTO_TEST_CASE(test_util_1) {

    const computational_dag_vector_impl graph = constr_graph_1();

    BOOST_CHECK_EQUAL(graph.num_edges(), 9);
    BOOST_CHECK_EQUAL(graph.num_vertices(), 8);

    auto sources = source_vertices(graph);
    BOOST_CHECK_EQUAL(sources.size(), 1);
    BOOST_CHECK_EQUAL(sources[0], 0);

    std::vector<vertex_idx> sources_s;
    for(const auto &v : source_vertices_view(graph)) {
        sources_s.push_back(v);
    }
    BOOST_CHECK_EQUAL(sources_s.size(), 1);
    BOOST_CHECK_EQUAL(sources_s[0], 0);

    auto sinks = sink_vertices(graph);
    BOOST_CHECK_EQUAL(sinks.size(), 3);
    BOOST_CHECK_EQUAL(sinks[0], 5);
    BOOST_CHECK_EQUAL(sinks[1], 6);
    BOOST_CHECK_EQUAL(sinks[2], 7);

    std::vector<vertex_idx> sinks_s;
    for(const auto &v : sink_vertices_view(graph)) {
        sinks_s.push_back(v);
    }
    
    BOOST_CHECK_EQUAL(sinks_s.size(), 3);
    BOOST_CHECK_EQUAL(sinks_s[0], 5);
    BOOST_CHECK_EQUAL(sinks_s[1], 6);
    BOOST_CHECK_EQUAL(sinks_s[2], 7);

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
};