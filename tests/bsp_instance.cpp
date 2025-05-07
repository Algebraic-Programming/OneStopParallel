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

#define BOOST_TEST_MODULE Bsp_Architecture
#include <boost/test/unit_test.hpp>

#include "bsp/model/BspInstance.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"
#include "io/arch_file_reader.hpp"
#include "io/graph_file_reader.hpp"
#include <filesystem>
#include <iostream>

using namespace osp;

BOOST_AUTO_TEST_CASE(test_1) {
    BspArchitecture<computational_dag_vector_impl_def_t> architecture(4, 2, 3);
    computational_dag_vector_impl_def_t graph;

    BspInstance instance(graph, architecture);

    BOOST_CHECK_EQUAL(instance.numberOfVertices(), 0);
    BOOST_CHECK_EQUAL(instance.numberOfProcessors(), 4);
    BOOST_CHECK_EQUAL(instance.synchronisationCosts(), 3);
    BOOST_CHECK_EQUAL(instance.communicationCosts(), 2);

    BspArchitecture<computational_dag_vector_impl_def_t> architecture_2(6, 3, 1);

    instance.setArchitecture(architecture_2);

    BOOST_CHECK_EQUAL(instance.numberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(instance.synchronisationCosts(), 1);
    BOOST_CHECK_EQUAL(instance.communicationCosts(), 3);
    BOOST_CHECK_EQUAL(instance.numberOfVertices(), 0);
}

BOOST_AUTO_TEST_CASE(test_instance_bicgstab) {

    BspInstance<computational_dag_edge_idx_vector_impl_def_t> instance;
    instance.setNumberOfProcessors(4);
    instance.setCommunicationCosts(2);
    instance.setSynchronisationCosts(3);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::readComputationalDagHyperdagFormat(
        (cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(), instance.getComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertices(), 54);

    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertex_types(), 1);

    instance.getComputationalDag().set_vertex_type(0, 1);

    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertex_types(), 2);

    instance.getArchitecture().setProcessorType(0, 1);
    instance.setDiagonalCompatibilityMatrix(2);

    BOOST_CHECK_EQUAL(instance.isCompatible(0, 0), true);
    BOOST_CHECK_EQUAL(instance.isCompatible(1, 0), false);

    BspInstance<computational_dag_vector_impl_def_t> instance_t2(instance);

    BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().num_vertices(), instance.getComputationalDag().num_vertices());
    BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().num_vertex_types(),
                      instance.getComputationalDag().num_vertex_types());
    BOOST_CHECK_EQUAL(instance_t2.getComputationalDag().num_edges(), instance.getComputationalDag().num_edges());
    BOOST_CHECK_EQUAL(instance_t2.getArchitecture().numberOfProcessors(),
                      instance.getArchitecture().numberOfProcessors());
    BOOST_CHECK_EQUAL(instance_t2.getArchitecture().getNumberOfProcessorTypes(),
                      instance.getArchitecture().getNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instance_t2.getArchitecture().communicationCosts(),
                      instance.getArchitecture().communicationCosts());
    BOOST_CHECK_EQUAL(instance_t2.getArchitecture().synchronisationCosts(),
                      instance.getArchitecture().synchronisationCosts());

    BspInstance<computational_dag_edge_idx_vector_impl_def_t> instance_t3;

    instance_t3 = instance;

    BOOST_CHECK_EQUAL(instance_t3.getComputationalDag().num_vertices(), instance.getComputationalDag().num_vertices());
    BOOST_CHECK_EQUAL(instance_t3.getComputationalDag().num_vertex_types(),
                      instance.getComputationalDag().num_vertex_types());
    BOOST_CHECK_EQUAL(instance_t3.getComputationalDag().num_edges(), instance.getComputationalDag().num_edges());
    BOOST_CHECK_EQUAL(instance_t3.getArchitecture().numberOfProcessors(),
                      instance.getArchitecture().numberOfProcessors());
    BOOST_CHECK_EQUAL(instance_t3.getArchitecture().getNumberOfProcessorTypes(),
                      instance.getArchitecture().getNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instance_t3.getArchitecture().communicationCosts(),
                      instance.getArchitecture().communicationCosts());
    BOOST_CHECK_EQUAL(instance_t3.getArchitecture().synchronisationCosts(),
                      instance.getArchitecture().synchronisationCosts());

    BspInstance<computational_dag_edge_idx_vector_impl_def_t> instance_t4(std::move(instance_t3));

    BOOST_CHECK_EQUAL(instance_t4.getComputationalDag().num_vertices(), instance.getComputationalDag().num_vertices());
    BOOST_CHECK_EQUAL(instance_t4.getComputationalDag().num_vertex_types(),
                      instance.getComputationalDag().num_vertex_types());
    BOOST_CHECK_EQUAL(instance_t4.getComputationalDag().num_edges(), instance.getComputationalDag().num_edges());
    BOOST_CHECK_EQUAL(instance_t4.getArchitecture().numberOfProcessors(),
                      instance.getArchitecture().numberOfProcessors());
    BOOST_CHECK_EQUAL(instance_t4.getArchitecture().getNumberOfProcessorTypes(),
                      instance.getArchitecture().getNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instance_t4.getArchitecture().communicationCosts(),
                      instance.getArchitecture().communicationCosts());
    BOOST_CHECK_EQUAL(instance_t4.getArchitecture().synchronisationCosts(),
                      instance.getArchitecture().synchronisationCosts());

    BspInstance<computational_dag_edge_idx_vector_impl_def_t> instance_t5;

    instance_t5 = std::move(instance_t4);
    BOOST_CHECK_EQUAL(instance_t5.getComputationalDag().num_vertices(), instance.getComputationalDag().num_vertices());
    BOOST_CHECK_EQUAL(instance_t5.getComputationalDag().num_vertex_types(),
                      instance.getComputationalDag().num_vertex_types());
    BOOST_CHECK_EQUAL(instance_t5.getComputationalDag().num_edges(), instance.getComputationalDag().num_edges());
    BOOST_CHECK_EQUAL(instance_t5.getArchitecture().numberOfProcessors(),
                      instance.getArchitecture().numberOfProcessors());
    BOOST_CHECK_EQUAL(instance_t5.getArchitecture().getNumberOfProcessorTypes(),
                      instance.getArchitecture().getNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instance_t5.getArchitecture().communicationCosts(),
                      instance.getArchitecture().communicationCosts());
    BOOST_CHECK_EQUAL(instance_t5.getArchitecture().synchronisationCosts(),
                      instance.getArchitecture().synchronisationCosts());
};