#define BOOST_TEST_MODULE ConnectedComponentPartitioner_test
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <chrono>

#include "dag_divider/ConnectedComponentDivider.hpp"
#include "dag_divider/ConnectedComponentScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"

BOOST_AUTO_TEST_CASE(ConnectedComponentPart_test) {

    ComputationalDag dag;

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 0);
    BOOST_CHECK_EQUAL(dag.numberOfEdges(), 0);

    VertexType v1 = dag.addVertex(2, 1);
    VertexType v2 = dag.addVertex(3, 1);
    VertexType v3 = dag.addVertex(4, 1);
    VertexType v4 = dag.addVertex(5, 1);

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 4);

    VertexType v5 = dag.addVertex(6, 1);
    VertexType v6 = dag.addVertex(7, 1);
    VertexType v7 = dag.addVertex(8, 1);
    VertexType v8 = dag.addVertex(9, 1);

    BOOST_CHECK_EQUAL(dag.numberOfVertices(), 8);
    BOOST_CHECK_EQUAL(dag.numberOfEdges(), 0);

   
    // EdgeType e12 =
    dag.addEdge(v1, v2, 1);
    // EdgeType e13 =
    dag.addEdge(v1, v3, 1);
    // EdgeType e14 =
    dag.addEdge(v1, v4, 1000);
    // EdgeType e25 =
    dag.addEdge(v2, v5, 1);

    // EdgeType e35 =
    dag.addEdge(v3, v6, 1);
    
    dag.addEdge(v3, v5, 1);
    // EdgeType e36 =
    
    // EdgeType e27 =
    dag.addEdge(v2, v7, 1);
    // EdgeType e58 =
    dag.addEdge(v5, v8, 1);
    // EdgeType e48 =
    dag.addEdge(v4, v8, 1);

    ConnectedComponentDivider partitioner;

    partitioner.compute_connected_components(dag);

    GreedyBspScheduler bsp_scheduler;
    ConnectedComponentScheduler scheduler(bsp_scheduler);

    BspArchitecture arch;
    arch.setNumberOfProcessors(6);
    BspInstance instance(dag, arch);

    auto [status, schedule] = scheduler.computeSchedule(instance);

    BOOST_CHECK_EQUAL(status, SUCCESS);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

    BOOST_CHECK(partitioner.get_sub_dags().size() == 1);
    BOOST_CHECK(partitioner.get_sub_dags()[0].numberOfVertices() == 8);
    BOOST_CHECK(partitioner.get_sub_dags()[0].numberOfEdges() == 9);

    for (unsigned i = 0; i < 8; i++) {
        BOOST_CHECK_EQUAL(partitioner.get_component()[i], 0);
        BOOST_CHECK(partitioner.get_vertex_map()[i] <= i + 1);
        BOOST_CHECK(partitioner.get_vertex_mapping()[0].at(i) <= 1 + i);
    }


    //     std::vector<std::unordered_map<unsigned, unsigned>> vertex_mapping;
    // std::vector<unsigned> component;
    // std::vector<unsigned> vertex_map;


    VertexType v9 = dag.addVertex(2, 1);
    VertexType v10 = dag.addVertex(3, 1);
    VertexType v11 = dag.addVertex(4, 1);
    VertexType v12 = dag.addVertex(5, 1);

    dag.addEdge(v9, v10, 1);
    dag.addEdge(v9, v11, 1);
    dag.addEdge(v9, v12, 1);
    dag.addEdge(v10, v11, 1);

    partitioner.compute_connected_components(dag);

    BOOST_CHECK_EQUAL(partitioner.get_sub_dags().size(), 2);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[0].numberOfVertices(), 8);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[0].numberOfEdges(), 9);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[1].numberOfVertices(), 4);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[1].numberOfEdges(), 4);

    for (unsigned i = 0; i < 8; i++) {
        BOOST_CHECK_EQUAL(partitioner.get_component()[i], 0);
        BOOST_CHECK(partitioner.get_vertex_map()[i] <= i+1);
        BOOST_CHECK(partitioner.get_vertex_mapping()[0].at(i) <= 1 + i);
    }

    for (unsigned i = 8; i < 12; i++) {
        BOOST_CHECK_EQUAL(partitioner.get_component()[i], 1);
        BOOST_CHECK(partitioner.get_vertex_map()[i] <= 1 + i - 8 );
        BOOST_CHECK(partitioner.get_vertex_mapping()[1].at(i-8) <= 1 + i);
    }

   
    BspInstance instance_new(dag, arch);

    auto [status_new, schedule_new] = scheduler.computeSchedule(instance_new);

    BOOST_CHECK_EQUAL(status_new, SUCCESS);
    BOOST_CHECK(schedule_new.satisfiesPrecedenceConstraints());

}


