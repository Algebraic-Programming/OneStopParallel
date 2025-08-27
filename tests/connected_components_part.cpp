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
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(ConnectedComponentPart_test) {

    BspInstance<computational_dag_vector_impl_def_int_t> instance;
    computational_dag_vector_impl_def_int_t &dag = instance.getComputationalDag();
    using VertexType = vertex_idx_t<computational_dag_vector_impl_def_int_t>;

    BOOST_CHECK_EQUAL(dag.num_vertices(), 0);
    BOOST_CHECK_EQUAL(dag.num_edges(), 0);

    VertexType v1 = dag.add_vertex(2, 1, 2);
    VertexType v2 = dag.add_vertex(3, 1, 2);
    VertexType v3 = dag.add_vertex(4, 1, 2);
    VertexType v4 = dag.add_vertex(5, 1, 2);
    VertexType v5 = dag.add_vertex(6, 1, 2);
    VertexType v6 = dag.add_vertex(7, 1, 2);
    VertexType v7 = dag.add_vertex(8, 1, 2);
    VertexType v8 = dag.add_vertex(9, 1, 2);

    BOOST_CHECK_EQUAL(dag.num_vertices(), 8);
    BOOST_CHECK_EQUAL(dag.num_edges(), 0);

    dag.add_edge(v1, v2);
    dag.add_edge(v1, v3);
    dag.add_edge(v1, v4);
    dag.add_edge(v2, v5);
    dag.add_edge(v3, v6);
    dag.add_edge(v3, v5);
    dag.add_edge(v2, v7);
    dag.add_edge(v5, v8);
    dag.add_edge(v4, v8);

    ConnectedComponentDivider<computational_dag_vector_impl_def_int_t, computational_dag_vector_impl_def_int_t> partitioner;

    partitioner.divide(dag);

    GreedyBspScheduler<boost_graph_int_t> bsp_scheduler;
    ConnectedComponentScheduler<computational_dag_vector_impl_def_int_t, boost_graph_int_t> scheduler(bsp_scheduler);

    BspArchitecture<computational_dag_vector_impl_def_int_t> arch = instance.getArchitecture();
    arch.setNumberOfProcessors(6);

    BspSchedule<computational_dag_vector_impl_def_int_t> schedule(instance);
    auto status = scheduler.computeSchedule(schedule);

    BOOST_CHECK_EQUAL(status, RETURN_STATUS::OSP_SUCCESS);
    BOOST_CHECK(schedule.satisfiesPrecedenceConstraints());

    BOOST_CHECK(partitioner.get_sub_dags().size() == 1);
    BOOST_CHECK(partitioner.get_sub_dags()[0].num_vertices() == 8);
    BOOST_CHECK(partitioner.get_sub_dags()[0].num_edges() == 9);

    for (unsigned i = 0; i < 8; i++) {
        BOOST_CHECK_EQUAL(partitioner.get_component()[i], 0);
        BOOST_CHECK(partitioner.get_vertex_map()[i] <= i + 1);
        BOOST_CHECK(partitioner.get_vertex_mapping()[0].at(i) <= 1 + i);
    }

    VertexType v9 = dag.add_vertex(2, 1, 4);
    VertexType v10 = dag.add_vertex(3, 1, 6);
    VertexType v11 = dag.add_vertex(4, 1, 6);
    VertexType v12 = dag.add_vertex(5, 1, 6);

    dag.add_edge(v9, v10);
    dag.add_edge(v9, v11);
    dag.add_edge(v9, v12);
    dag.add_edge(v10, v11);

    partitioner.compute_connected_components(dag);

    BOOST_CHECK_EQUAL(partitioner.get_sub_dags().size(), 2);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[0].num_vertices(), 8);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[0].num_edges(), 9);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[1].num_vertices(), 4);
    BOOST_CHECK_EQUAL(partitioner.get_sub_dags()[1].num_edges(), 4);

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

    BspInstance<computational_dag_vector_impl_def_int_t> instance_new(dag, arch);
    BspSchedule<computational_dag_vector_impl_def_int_t> schedule_new(instance_new);

    auto status_new = scheduler.computeSchedule(schedule_new);

    BOOST_CHECK_EQUAL(status_new, RETURN_STATUS::OSP_SUCCESS);
    BOOST_CHECK(schedule_new.satisfiesPrecedenceConstraints());
}
