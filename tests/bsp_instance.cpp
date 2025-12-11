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
#include <filesystem>
#include <iostream>

#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/hdag_graph_file_reader.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/util/CompatibleProcessorRange.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(Test1) {
    BspArchitecture<computational_dag_vector_impl_def_t> architecture(4, 2, 3);
    computational_dag_vector_impl_def_t graph;

    BspInstance instance(graph, architecture);

    BOOST_CHECK_EQUAL(instance.numberOfVertices(), 0);
    BOOST_CHECK_EQUAL(instance.numberOfProcessors(), 4);
    BOOST_CHECK_EQUAL(instance.synchronisationCosts(), 3);
    BOOST_CHECK_EQUAL(instance.communicationCosts(), 2);

    BspArchitecture<computational_dag_vector_impl_def_t> architecture2(6, 3, 1);

    instance.getArchitecture() = architecture2;

    BOOST_CHECK_EQUAL(instance.numberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(instance.synchronisationCosts(), 1);
    BOOST_CHECK_EQUAL(instance.communicationCosts(), 3);
    BOOST_CHECK_EQUAL(instance.numberOfVertices(), 0);
}

BOOST_AUTO_TEST_CASE(TestInstanceBicgstab) {
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

    bool status = file_reader::readComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(),
                                                                    instance.getComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertices(), 54);

    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertex_types(), 1);

    instance.getComputationalDag().set_vertex_type(0, 1);

    BOOST_CHECK_EQUAL(instance.getComputationalDag().num_vertex_types(), 2);

    instance.getArchitecture().setProcessorType(0, 1);
    instance.setDiagonalCompatibilityMatrix(2);

    BOOST_CHECK_EQUAL(instance.isCompatible(0, 0), true);
    BOOST_CHECK_EQUAL(instance.isCompatible(1, 0), false);

    CompatibleProcessorRange range(instance);

    BOOST_CHECK_EQUAL(range.compatible_processors_type(0).size(), 3);
    BOOST_CHECK_EQUAL(range.compatible_processors_type(1).size(), 1);

    std::cout << "Compatible processors type 0: " << std::endl;

    for (const auto &p : range.compatible_processors_type(0)) {
        std::cout << p;
    }
    std::cout << std::endl;

    std::cout << "Compatible processors type 1: " << std::endl;

    for (const auto &p : range.compatible_processors_type(1)) {
        std::cout << p;
    }
    std::cout << std::endl;

    BOOST_CHECK_EQUAL(range.compatible_processors_vertex(0).size(), 1);
    BOOST_CHECK_EQUAL(range.compatible_processors_vertex(1).size(), 3);
    BOOST_CHECK_EQUAL(range.compatible_processors_vertex(2).size(), 3);
    BOOST_CHECK_EQUAL(range.compatible_processors_vertex(3).size(), 3);

    BOOST_CHECK_EQUAL(range.compatible_processors_type(1)[0], 0);
    BOOST_CHECK_EQUAL(range.compatible_processors_type(0)[0], 1);
    BOOST_CHECK_EQUAL(range.compatible_processors_type(0)[1], 2);
    BOOST_CHECK_EQUAL(range.compatible_processors_type(0)[2], 3);

    BspInstance<computational_dag_vector_impl_def_t> instanceT2(instance);

    BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().num_vertices(), instance.getComputationalDag().num_vertices());
    BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().num_vertex_types(), instance.getComputationalDag().num_vertex_types());
    BOOST_CHECK_EQUAL(instanceT2.getComputationalDag().num_edges(), instance.getComputationalDag().num_edges());
    BOOST_CHECK_EQUAL(instanceT2.getArchitecture().numberOfProcessors(), instance.getArchitecture().numberOfProcessors());
    BOOST_CHECK_EQUAL(instanceT2.getArchitecture().getNumberOfProcessorTypes(),
                      instance.getArchitecture().getNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instanceT2.getArchitecture().communicationCosts(), instance.getArchitecture().communicationCosts());
    BOOST_CHECK_EQUAL(instanceT2.getArchitecture().synchronisationCosts(), instance.getArchitecture().synchronisationCosts());

    BspInstance<computational_dag_edge_idx_vector_impl_def_t> instanceT3;

    instanceT3 = instance;

    BOOST_CHECK_EQUAL(instanceT3.getComputationalDag().num_vertices(), instance.getComputationalDag().num_vertices());
    BOOST_CHECK_EQUAL(instanceT3.getComputationalDag().num_vertex_types(), instance.getComputationalDag().num_vertex_types());
    BOOST_CHECK_EQUAL(instanceT3.getComputationalDag().num_edges(), instance.getComputationalDag().num_edges());
    BOOST_CHECK_EQUAL(instanceT3.getArchitecture().numberOfProcessors(), instance.getArchitecture().numberOfProcessors());
    BOOST_CHECK_EQUAL(instanceT3.getArchitecture().getNumberOfProcessorTypes(),
                      instance.getArchitecture().getNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instanceT3.getArchitecture().communicationCosts(), instance.getArchitecture().communicationCosts());
    BOOST_CHECK_EQUAL(instanceT3.getArchitecture().synchronisationCosts(), instance.getArchitecture().synchronisationCosts());

    BspInstance<computational_dag_edge_idx_vector_impl_def_t> instanceT4(std::move(instanceT3));

    BOOST_CHECK_EQUAL(instanceT4.getComputationalDag().num_vertices(), instance.getComputationalDag().num_vertices());
    BOOST_CHECK_EQUAL(instanceT4.getComputationalDag().num_vertex_types(), instance.getComputationalDag().num_vertex_types());
    BOOST_CHECK_EQUAL(instanceT4.getComputationalDag().num_edges(), instance.getComputationalDag().num_edges());
    BOOST_CHECK_EQUAL(instanceT4.getArchitecture().numberOfProcessors(), instance.getArchitecture().numberOfProcessors());
    BOOST_CHECK_EQUAL(instanceT4.getArchitecture().getNumberOfProcessorTypes(),
                      instance.getArchitecture().getNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instanceT4.getArchitecture().communicationCosts(), instance.getArchitecture().communicationCosts());
    BOOST_CHECK_EQUAL(instanceT4.getArchitecture().synchronisationCosts(), instance.getArchitecture().synchronisationCosts());

    BspInstance<computational_dag_edge_idx_vector_impl_def_t> instanceT5;

    instanceT5 = std::move(instanceT4);
    BOOST_CHECK_EQUAL(instanceT5.getComputationalDag().num_vertices(), instance.getComputationalDag().num_vertices());
    BOOST_CHECK_EQUAL(instanceT5.getComputationalDag().num_vertex_types(), instance.getComputationalDag().num_vertex_types());
    BOOST_CHECK_EQUAL(instanceT5.getComputationalDag().num_edges(), instance.getComputationalDag().num_edges());
    BOOST_CHECK_EQUAL(instanceT5.getArchitecture().numberOfProcessors(), instance.getArchitecture().numberOfProcessors());
    BOOST_CHECK_EQUAL(instanceT5.getArchitecture().getNumberOfProcessorTypes(),
                      instance.getArchitecture().getNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instanceT5.getArchitecture().communicationCosts(), instance.getArchitecture().communicationCosts());
    BOOST_CHECK_EQUAL(instanceT5.getArchitecture().synchronisationCosts(), instance.getArchitecture().synchronisationCosts());
}
