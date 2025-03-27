#define BOOST_TEST_MODULE Bsp_Architecture
#include <boost/test/unit_test.hpp>

#include "model/bsp/BspInstance.hpp"
#include "graph_implementations/computational_dag_vector_impl.hpp"


using namespace osp;

BOOST_AUTO_TEST_CASE(test_1)
{
    BspArchitecture architecture(4, 2, 3);
    computational_dag_vector_impl graph;
    
    BspInstance instance(graph, architecture);

    BOOST_CHECK_EQUAL(instance.numberOfVertices(), 0);
    BOOST_CHECK_EQUAL(instance.numberOfProcessors(), 4);

}

