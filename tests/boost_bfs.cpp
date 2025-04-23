#define BOOST_TEST_MODULE Boost_BFS
#include <boost/test/unit_test.hpp>

#include "model/ComputationalDag.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/visitors.hpp>

BOOST_AUTO_TEST_CASE(SimplePrintingBFS) {
    ComputationalDag dag;

    const VertexType v1 = dag.addVertex(2, 9);
    const VertexType v2 = dag.addVertex(3, 8);
    const VertexType v3 = dag.addVertex(4, 7);
    const VertexType v4 = dag.addVertex(5, 6);
    const VertexType v5 = dag.addVertex(6, 5);
    const VertexType v6 = dag.addVertex(7, 4);
    const VertexType v7 = dag.addVertex(8, 3);
    const VertexType v8 = dag.addVertex(9, 2);

    dag.addEdge(v1, v2, 2);
    dag.addEdge(v1, v3, 3);
    dag.addEdge(v1, v4, 4);
    dag.addEdge(v2, v5, 5);
    dag.addEdge(v3, v5, 6);
    dag.addEdge(v3, v6, 7);
    dag.addEdge(v2, v7, 8);
    dag.addEdge(v5, v8, 9);
    dag.addEdge(v4, v8, 9);

    std::vector<VertexType> expectedSuccessors {v2, v5, v7, v8};
    std::vector<VertexType> actualSuccessors = dag.successors(v2);
    std::sort(actualSuccessors.begin(), actualSuccessors.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(actualSuccessors.begin(), actualSuccessors.end(), expectedSuccessors.begin(), expectedSuccessors.end());

    std::vector<VertexType> expectedAncestors {v1, v2};
    std::vector<VertexType> actualAncestors = dag.ancestors(v2);
    std::sort(actualAncestors.begin(), actualAncestors.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(actualAncestors.begin(), actualAncestors.end(), expectedAncestors.begin(), expectedAncestors.end());
}