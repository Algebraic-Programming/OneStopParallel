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

#define BOOST_TEST_MODULE TransitiveReduction
#include <boost/test/unit_test.hpp>

#include "test_graphs.hpp"
#include "osp/graph_algorithms/transitive_reduction.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

using namespace osp;
using graph_t = computational_dag_vector_impl_def_t;

BOOST_AUTO_TEST_SUITE(TransitiveReduction)

// Test with a simple chain graph that has a transitive edge
BOOST_AUTO_TEST_CASE(SimpleTransitiveEdge) {
    graph_t dag;
    dag.add_vertex(1, 1, 1); // 0
    dag.add_vertex(1, 1, 1); // 1
    dag.add_vertex(1, 1, 1); // 2
    dag.add_edge(0, 1);
    dag.add_edge(1, 2);
    dag.add_edge(0, 2); // Transitive edge

    BOOST_REQUIRE_EQUAL(dag.num_vertices(), 3);
    BOOST_REQUIRE_EQUAL(dag.num_edges(), 3);

    graph_t reduced_sparse, reduced_dense;
    transitive_reduction_sparse(dag, reduced_sparse);
    transitive_reduction_dense(dag, reduced_dense);

    BOOST_CHECK_EQUAL(reduced_sparse.num_vertices(), 3);
    BOOST_CHECK_EQUAL(reduced_sparse.num_edges(), 2);
    BOOST_CHECK_EQUAL(reduced_dense.num_vertices(), 3);
    BOOST_CHECK_EQUAL(reduced_dense.num_edges(), 2);

    BOOST_CHECK(checkOrderedIsomorphism(reduced_sparse, reduced_dense));
}

// Test with a graph that has no transitive edges
BOOST_AUTO_TEST_CASE(NoTransitiveEdges) {
    const auto dag = construct_ladder_dag<graph_t>(3); // A ladder graph has no transitive edges
    BOOST_REQUIRE_EQUAL(dag.num_vertices(), 8);
    BOOST_REQUIRE_EQUAL(dag.num_edges(), 11);

    graph_t reduced_sparse, reduced_dense;
    transitive_reduction_sparse(dag, reduced_sparse);
    transitive_reduction_dense(dag, reduced_dense);

    BOOST_CHECK_EQUAL(reduced_sparse.num_edges(), dag.num_edges());
    BOOST_CHECK_EQUAL(reduced_dense.num_edges(), dag.num_edges());

    BOOST_CHECK(checkOrderedIsomorphism(reduced_sparse, reduced_dense));
}

// Test with a more complex graph containing multiple transitive edges
BOOST_AUTO_TEST_CASE(ComplexGraph) {
    graph_t dag;
    // 0 -> 1, 0 -> 2, 0 -> 3 (transitive)
    // 1 -> 3
    // 2 -> 3
    // 3 -> 4
    // 0 -> 4 (transitive)
    dag.add_vertex(1, 1, 1); // 0
    dag.add_vertex(1, 1, 1); // 1
    dag.add_vertex(1, 1, 1); // 2
    dag.add_vertex(1, 1, 1); // 3
    dag.add_vertex(1, 1, 1); // 4

    dag.add_edge(0, 1);
    dag.add_edge(0, 2);
    dag.add_edge(1, 3);
    dag.add_edge(2, 3);
    dag.add_edge(3, 4);
    // Add transitive edges
    dag.add_edge(0, 3); // transitive via 0->1->3 or 0->2->3
    dag.add_edge(0, 4); // transitive via 0->...->3->4

    BOOST_REQUIRE_EQUAL(dag.num_vertices(), 5);
    BOOST_REQUIRE_EQUAL(dag.num_edges(), 7);

    graph_t reduced_sparse, reduced_dense;
    transitive_reduction_sparse(dag, reduced_sparse);
    transitive_reduction_dense(dag, reduced_dense);

    BOOST_CHECK_EQUAL(reduced_sparse.num_vertices(), 5);
    BOOST_CHECK_EQUAL(reduced_sparse.num_edges(), 5);
    BOOST_CHECK_EQUAL(reduced_dense.num_vertices(), 5);
    BOOST_CHECK_EQUAL(reduced_dense.num_edges(), 5);

    BOOST_CHECK(checkOrderedIsomorphism(reduced_sparse, reduced_dense));
}

BOOST_AUTO_TEST_SUITE_END()
