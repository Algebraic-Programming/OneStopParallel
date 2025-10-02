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

#define BOOST_TEST_MODULE BSP_SCHEDULE_RECOMP
#include <boost/test/unit_test.hpp>

#include "osp/dag_divider/isomorphism_divider/MerkleHashComputer.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "test_utils.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(BspScheduleRecomp_test)
{
    using graph_t = computational_dag_vector_impl_def_t;
    graph_t graph;

    const auto project_root = get_project_root();
    file_reader::readComputationalDagHyperdagFormat((project_root / "data/spaa/tiny/instance_bicgstab.hdag").string(), graph);

    MerkleHashComputer<graph_t, uniform_node_hash_func<vertex_idx_t<graph_t>>> m_hash(graph);

    BOOST_CHECK_EQUAL(m_hash.get_vertex_hashes().size(), graph.num_vertices());
    
    for (const auto& v : source_vertices_view(graph)) {
        BOOST_CHECK_EQUAL(m_hash.get_vertex_hash(v), 11);
    }

    size_t num = 0;
    for (const auto& pair : m_hash.get_orbits()) {

        num += pair.second.size();
        std::cout << "orbit " << pair.first << ": ";
        for (const auto& v : pair.second) {
            std::cout << v << ", ";
        } 
        std::cout << std::endl;
    }

    BOOST_CHECK_EQUAL(num, graph.num_vertices());

    BOOST_CHECK_EQUAL(m_hash.get_vertex_hash(41), m_hash.get_vertex_hash(47));
    BOOST_CHECK_EQUAL(m_hash.get_vertex_hash(28), m_hash.get_vertex_hash(18));
    BOOST_CHECK_EQUAL(m_hash.get_vertex_hash(43), m_hash.get_vertex_hash(48));
    BOOST_CHECK_EQUAL(m_hash.get_vertex_hash(29), m_hash.get_vertex_hash(22));
    BOOST_CHECK(m_hash.get_vertex_hash(3) != m_hash.get_vertex_hash(12));
    BOOST_CHECK(m_hash.get_vertex_hash(53) != m_hash.get_vertex_hash(29));

};


using graph = computational_dag_vector_impl_def_t;
using VertexType = vertex_idx_t<graph>;


BOOST_AUTO_TEST_CASE(MerkleIsomorphismTest_IdenticalGraphsAreIsomorphic) {
    graph dag1;
    const auto v1 = dag1.add_vertex(0, 10, 1);
    const auto v2 = dag1.add_vertex(1, 20, 1);
    const auto v3 = dag1.add_vertex(0, 30, 1);
    dag1.add_edge(v1, v2);
    dag1.add_edge(v2, v3);

    graph dag2;
    const auto vA = dag2.add_vertex(0, 10, 1);
    const auto vB = dag2.add_vertex(1, 20, 1);
    const auto vC = dag2.add_vertex(0, 30, 1);
    dag2.add_edge(vA, vB);
    dag2.add_edge(vB, vC);

    bool test = are_isomorphic_by_merkle_hash<graph, uniform_node_hash_func<VertexType>, true>(dag1, dag2);
    BOOST_CHECK(test);
    test = are_isomorphic_by_merkle_hash<graph, uniform_node_hash_func<VertexType>, false>(dag1, dag2);
    BOOST_CHECK(test);
}

// Test case 2: Graphs with different numbers of vertices should not be isomorphic.
BOOST_AUTO_TEST_CASE(MerkleIsomorphismTest_DifferentVertexCount) {
    graph dag1;
    dag1.add_vertex(0, 10, 1);
    dag1.add_vertex(1, 20, 1);

    graph dag2;
    dag2.add_vertex(0, 10, 1);

    BOOST_CHECK_EQUAL(are_isomorphic_by_merkle_hash(dag1, dag2), false);
}

// Test case 3: Graphs with the same size but different structures should not be isomorphic.
BOOST_AUTO_TEST_CASE(MerkleIsomorphismTest_SameSizeDifferentStructure) {
    graph dag1; // A -> B -> C
    const auto v1_1 = dag1.add_vertex(0, 1, 1);
    const auto v1_2 = dag1.add_vertex(0, 1, 1);
    const auto v1_3 = dag1.add_vertex(0, 1, 1);
    dag1.add_edge(v1_1, v1_2);
    dag1.add_edge(v1_2, v1_3);

    graph dag2; // A -> B, A -> C
    const auto v2_1 = dag2.add_vertex(0, 1, 1);
    const auto v2_2 = dag2.add_vertex(0, 1, 1);
    const auto v2_3 = dag2.add_vertex(0, 1, 1);
    dag2.add_edge(v2_1, v2_2);
    dag2.add_edge(v2_1, v2_3);

    BOOST_CHECK_EQUAL(are_isomorphic_by_merkle_hash(dag1, dag2), false);
}

// Test case 4: Structurally identical graphs with different vertex labeling should be isomorphic.
BOOST_AUTO_TEST_CASE(MerkleIsomorphismTest_IsomorphicWithDifferentLabels) {
    graph dag1;
    const auto v1_1 = dag1.add_vertex(0, 1, 1); // Source
    const auto v1_2 = dag1.add_vertex(0, 1, 1);
    const auto v1_3 = dag1.add_vertex(0, 1, 1); // Sink
    dag1.add_edge(v1_1, v1_2);
    dag1.add_edge(v1_2, v1_3);

    graph dag2;
    // Same structure as dag1, but vertices are added in a different order.
    const auto v2_3 = dag2.add_vertex(0, 1, 1); // Sink
    const auto v2_1 = dag2.add_vertex(0, 1, 1); // Source
    const auto v2_2 = dag2.add_vertex(0, 1, 1);
    dag2.add_edge(v2_1, v2_2);
    dag2.add_edge(v2_2, v2_3);

    BOOST_CHECK(are_isomorphic_by_merkle_hash(dag1, dag2));
}

// Test case 5: A more complex example based on your provided DAG.
BOOST_AUTO_TEST_CASE(MerkleIsomorphismTest_ComplexIsomorphicGraphs) {
    graph dag1;
    {
        const auto v1 = dag1.add_vertex(2, 9, 2); const auto v2 = dag1.add_vertex(3, 8, 4);
        const auto v3 = dag1.add_vertex(4, 7, 3); const auto v4 = dag1.add_vertex(5, 6, 2);
        const auto v5 = dag1.add_vertex(6, 5, 6); const auto v6 = dag1.add_vertex(7, 4, 2);
        dag1.add_vertex(8, 3, 4); const auto v8 = dag1.add_vertex(9, 2, 1);
        dag1.add_edge(v1, v2); dag1.add_edge(v1, v3); dag1.add_edge(v1, v4);
        dag1.add_edge(v1, v5); dag1.add_edge(v1, v8); dag1.add_edge(v2, v5);
        dag1.add_edge(v2, v6); dag1.add_edge(v2, v8); dag1.add_edge(v3, v5);
        dag1.add_edge(v3, v6); dag1.add_edge(v5, v8); dag1.add_edge(v4, v8);
    }

    graph dag2;
    {
        // Same structure, different vertex variable names and creation order.
        const auto n8 = dag2.add_vertex(9, 2, 1);
        dag2.add_vertex(8, 3, 4);
        const auto n6 = dag2.add_vertex(7, 4, 2); const auto n5 = dag2.add_vertex(6, 5, 6);
        const auto n4 = dag2.add_vertex(5, 6, 2); const auto n3 = dag2.add_vertex(4, 7, 3);
        const auto n2 = dag2.add_vertex(3, 8, 4); const auto n1 = dag2.add_vertex(2, 9, 2);
        dag2.add_edge(n1, n2); dag2.add_edge(n1, n3); dag2.add_edge(n1, n4);
        dag2.add_edge(n1, n5); dag2.add_edge(n1, n8); dag2.add_edge(n2, n5);
        dag2.add_edge(n2, n6); dag2.add_edge(n2, n8); dag2.add_edge(n3, n5);
        dag2.add_edge(n3, n6); dag2.add_edge(n5, n8); dag2.add_edge(n4, n8);
    }
    
    BOOST_CHECK(are_isomorphic_by_merkle_hash(dag1, dag2));
}