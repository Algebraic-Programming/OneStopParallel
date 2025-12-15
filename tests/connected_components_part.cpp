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

#define BOOST_TEST_MODULE ConnectedComponentPartitioner_test
#include <boost/test/unit_test.hpp>

#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/dag_divider/ConnectedComponentDivider.hpp"
#include "osp/dag_divider/ConnectedComponentScheduler.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(ConnectedComponentPartTest) {
    BspInstance<ComputationalDagVectorImplDefIntT> instance;
    ComputationalDagVectorImplDefIntT &dag = instance.GetComputationalDag();
    using VertexType = VertexIdxT<ComputationalDagVectorImplDefIntT>;

    BOOST_CHECK_EQUAL(dag.NumVertices(), 0);
    BOOST_CHECK_EQUAL(dag.NumEdges(), 0);

    VertexType v1 = dag.AddVertex(2, 1, 2);
    VertexType v2 = dag.AddVertex(3, 1, 2);
    VertexType v3 = dag.AddVertex(4, 1, 2);
    VertexType v4 = dag.AddVertex(5, 1, 2);
    VertexType v5 = dag.AddVertex(6, 1, 2);
    VertexType v6 = dag.AddVertex(7, 1, 2);
    VertexType v7 = dag.AddVertex(8, 1, 2);
    VertexType v8 = dag.AddVertex(9, 1, 2);

    BOOST_CHECK_EQUAL(dag.NumVertices(), 8);
    BOOST_CHECK_EQUAL(dag.NumEdges(), 0);

    dag.AddEdge(v1, v2);
    dag.AddEdge(v1, v3);
    dag.AddEdge(v1, v4);
    dag.AddEdge(v2, v5);
    dag.AddEdge(v3, v6);
    dag.AddEdge(v3, v5);
    dag.AddEdge(v2, v7);
    dag.AddEdge(v5, v8);
    dag.AddEdge(v4, v8);

    ConnectedComponentDivider<ComputationalDagVectorImplDefIntT, ComputationalDagVectorImplDefIntT> partitioner;

    partitioner.divide(dag);

    GreedyBspScheduler<boost_graph_int_t> bspScheduler;
    ConnectedComponentScheduler<ComputationalDagVectorImplDefIntT, boost_graph_int_t> scheduler(bspScheduler);

    BspArchitecture<ComputationalDagVectorImplDefIntT> arch = instance.GetArchitecture();
    arch.SetNumberOfProcessors(6);

    BspSchedule<ComputationalDagVectorImplDefIntT> schedule(instance);
    auto status = scheduler.ComputeSchedule(schedule);

    BOOST_CHECK_EQUAL(status, ReturnStatus::OSP_SUCCESS);
    BOOST_CHECK(schedule.SatisfiesPrecedenceConstraints());

    BOOST_CHECK(partitioner.get_sub_dags().size() == 1);
    BOOST_CHECK(partitioner.get_sub_dags()[0].NumVertices() == 8);
    BOOST_CHECK(partitioner.get_sub_dags()[0].NumEdges() == 9);

    for (unsigned i = 0; i < 8; i++) {
        BOOST_CHECK_EQUAL(partitioner.get_component()[i], 0);
        BOOST_CHECK(partitioner.get_vertex_map()[i] <= i + 1);
        BOOST_CHECK(partitioner.get_vertex_mapping()[0].at(i) <= 1 + i);
    }

    VertexType v9 = dag.AddVertex(2, 1, 4);
    VertexType v10 = dag.AddVertex(3, 1, 6);
    VertexType v11 = dag.AddVertex(4, 1, 6);
    VertexType v12 = dag.AddVertex(5, 1, 6);

    dag.AddEdge(v9, v10);
    dag.AddEdge(v9, v11);
    dag.AddEdge(v9, v12);
    dag.AddEdge(v10, v11);

    partitioner.compute_connected_components(dag);

    BOOST_CHECK_EQUAL(partitioner.get_sub_dags().size(), 2);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[0].NumVertices(), 8);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[0].NumEdges(), 9);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[1].NumVertices(), 4);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[1].NumEdges(), 4);

    for (unsigned i = 0; i < 8; i++) {
        BOOST_CHECK_EQUAL(partitioner.get_component()[i], 0);
        BOOST_CHECK(partitioner.get_vertex_map()[i] <= i + 1);
        BOOST_CHECK(partitioner.get_vertex_mapping()[0].at(i) <= 1 + i);
    }

    for (unsigned i = 8; i < 12; i++) {
        BOOST_CHECK_EQUAL(partitioner.get_component()[i], 1);
        BOOST_CHECK(partitioner.get_vertex_map()[i] <= 1 + i - 8);
        BOOST_CHECK(partitioner.get_vertex_mapping()[1].at(i - 8) <= 1 + i);
    }

    BspInstance<ComputationalDagVectorImplDefIntT> instanceNew(dag, arch);
    BspSchedule<ComputationalDagVectorImplDefIntT> scheduleNew(instanceNew);

    auto statusNew = scheduler.ComputeSchedule(scheduleNew);

    BOOST_CHECK_EQUAL(statusNew, ReturnStatus::OSP_SUCCESS);
    BOOST_CHECK(scheduleNew.SatisfiesPrecedenceConstraints());
}
