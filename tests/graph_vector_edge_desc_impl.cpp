#define BOOST_TEST_MODULE ApproxEdgeReduction

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

#include "graph_implementations/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_algorithms/directed_graph_util.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(test_1) {

    
    computational_dag_edge_idx_vector_impl graph;

    vertex_idx v1 = graph.add_vertex(1, 2, 3, 4);
    vertex_idx v2 = graph.add_vertex(5, 6, 7, 8);
    vertex_idx v3 = graph.add_vertex(9, 10, 11, 12);

    graph.add_edge(v1, v2);
    graph.add_edge(v2, v3);
    graph.add_edge(v1, v3);

  
    
    std::vector<vertex_idx> sources = source_vertices(graph);
    std::vector<vertex_idx> sinks = sink_vertices(graph);
    
    BOOST_CHECK_EQUAL(sources.size(), 1);
    BOOST_CHECK_EQUAL(sinks.size(), 1);
    
    BOOST_CHECK_EQUAL(sources[0], v1);
    BOOST_CHECK_EQUAL(sinks[0], v3);

    std::cout << "Sources: ";
    for (const vertex_idx v_idx : source_vertices_iterator(graph)) {
        std::cout << v_idx << " ";
    }
    std::cout << std::endl;

    std::cout << std::endl << "Sinks: ";
    for (const vertex_idx v_idx : sinks) {
        std::cout << v_idx << " ";
    }
    std::cout << std::endl;
    

};
