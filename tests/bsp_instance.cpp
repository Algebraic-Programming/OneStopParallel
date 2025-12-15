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
    BspArchitecture<ComputationalDagEdgeIdxVectorImplDefT> architecture(4, 2, 3);
    ComputationalDagEdgeIdxVectorImplDefT graph;

    BspInstance instance(graph, architecture);

    BOOST_CHECK_EQUAL(instance.NumberOfVertices(), 0);
    BOOST_CHECK_EQUAL(instance.NumberOfProcessors(), 4);
    BOOST_CHECK_EQUAL(instance.SynchronisationCosts(), 3);
    BOOST_CHECK_EQUAL(instance.CommunicationCosts(), 2);

    BspArchitecture<ComputationalDagEdgeIdxVectorImplDefT> architecture2(6, 3, 1);

    instance.GetArchitecture() = architecture2;

    BOOST_CHECK_EQUAL(instance.NumberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(instance.SynchronisationCosts(), 1);
    BOOST_CHECK_EQUAL(instance.CommunicationCosts(), 3);
    BOOST_CHECK_EQUAL(instance.NumberOfVertices(), 0);
}

BOOST_AUTO_TEST_CASE(TestInstanceBicgstab) {
    BspInstance<ComputationalDagEdgeIdxVectorImplDefT> instance;
    instance.SetNumberOfProcessors(4);
    instance.SetCommunicationCosts(2);
    instance.SetSynchronisationCosts(3);

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    bool status = file_reader::ReadComputationalDagHyperdagFormatDB((cwd / "data/spaa/tiny/instance_bicgstab.hdag").string(),
                                                                    instance.GetComputationalDag());

    BOOST_CHECK(status);
    BOOST_CHECK_EQUAL(instance.GetComputationalDag().NumVertices(), 54);

    BOOST_CHECK_EQUAL(instance.GetComputationalDag().NumVertexTypes(), 1);

    instance.GetComputationalDag().SetVertexType(0, 1);

    BOOST_CHECK_EQUAL(instance.GetComputationalDag().NumVertexTypes(), 2);

    instance.GetArchitecture().SetProcessorType(0, 1);
    instance.SetDiagonalCompatibilityMatrix(2);

    BOOST_CHECK_EQUAL(instance.IsCompatible(0, 0), true);
    BOOST_CHECK_EQUAL(instance.IsCompatible(1, 0), false);

    CompatibleProcessorRange range(instance);

    BOOST_CHECK_EQUAL(range.CompatibleProcessorsType(0).size(), 3);
    BOOST_CHECK_EQUAL(range.CompatibleProcessorsType(1).size(), 1);

    std::cout << "Compatible processors type 0: " << std::endl;

    for (const auto &p : range.CompatibleProcessorsType(0)) {
        std::cout << p;
    }
    std::cout << std::endl;

    std::cout << "Compatible processors type 1: " << std::endl;

    for (const auto &p : range.CompatibleProcessorsType(1)) {
        std::cout << p;
    }
    std::cout << std::endl;

    BOOST_CHECK_EQUAL(range.CompatibleProcessorsVertex(0).size(), 1);
    BOOST_CHECK_EQUAL(range.CompatibleProcessorsVertex(1).size(), 3);
    BOOST_CHECK_EQUAL(range.CompatibleProcessorsVertex(2).size(), 3);
    BOOST_CHECK_EQUAL(range.CompatibleProcessorsVertex(3).size(), 3);

    BOOST_CHECK_EQUAL(range.CompatibleProcessorsType(1)[0], 0);
    BOOST_CHECK_EQUAL(range.CompatibleProcessorsType(0)[0], 1);
    BOOST_CHECK_EQUAL(range.CompatibleProcessorsType(0)[1], 2);
    BOOST_CHECK_EQUAL(range.CompatibleProcessorsType(0)[2], 3);

    BspInstance<ComputationalDagEdgeIdxVectorImplDefT> instanceT2(instance);

    BOOST_CHECK_EQUAL(instanceT2.GetComputationalDag().NumVertices(), instance.GetComputationalDag().NumVertices());
    BOOST_CHECK_EQUAL(instanceT2.GetComputationalDag().NumVertexTypes(), instance.GetComputationalDag().NumVertexTypes());
    BOOST_CHECK_EQUAL(instanceT2.GetComputationalDag().NumEdges(), instance.GetComputationalDag().NumEdges());
    BOOST_CHECK_EQUAL(instanceT2.GetArchitecture().NumberOfProcessors(), instance.GetArchitecture().NumberOfProcessors());
    BOOST_CHECK_EQUAL(instanceT2.GetArchitecture().GetNumberOfProcessorTypes(),
                      instance.GetArchitecture().GetNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instanceT2.GetArchitecture().CommunicationCosts(), instance.GetArchitecture().CommunicationCosts());
    BOOST_CHECK_EQUAL(instanceT2.GetArchitecture().SynchronisationCosts(), instance.GetArchitecture().SynchronisationCosts());

    BspInstance<ComputationalDagEdgeIdxVectorImplDefT> instanceT3;

    instanceT3 = instance;

    BOOST_CHECK_EQUAL(instanceT3.GetComputationalDag().NumVertices(), instance.GetComputationalDag().NumVertices());
    BOOST_CHECK_EQUAL(instanceT3.GetComputationalDag().NumVertexTypes(), instance.GetComputationalDag().NumVertexTypes());
    BOOST_CHECK_EQUAL(instanceT3.GetComputationalDag().NumEdges(), instance.GetComputationalDag().NumEdges());
    BOOST_CHECK_EQUAL(instanceT3.GetArchitecture().NumberOfProcessors(), instance.GetArchitecture().NumberOfProcessors());
    BOOST_CHECK_EQUAL(instanceT3.GetArchitecture().GetNumberOfProcessorTypes(),
                      instance.GetArchitecture().GetNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instanceT3.GetArchitecture().CommunicationCosts(), instance.GetArchitecture().CommunicationCosts());
    BOOST_CHECK_EQUAL(instanceT3.GetArchitecture().SynchronisationCosts(), instance.GetArchitecture().SynchronisationCosts());

    BspInstance<ComputationalDagEdgeIdxVectorImplDefT> instanceT4(std::move(instanceT3));

    BOOST_CHECK_EQUAL(instanceT4.GetComputationalDag().NumVertices(), instance.GetComputationalDag().NumVertices());
    BOOST_CHECK_EQUAL(instanceT4.GetComputationalDag().NumVertexTypes(), instance.GetComputationalDag().NumVertexTypes());
    BOOST_CHECK_EQUAL(instanceT4.GetComputationalDag().NumEdges(), instance.GetComputationalDag().NumEdges());
    BOOST_CHECK_EQUAL(instanceT4.GetArchitecture().NumberOfProcessors(), instance.GetArchitecture().NumberOfProcessors());
    BOOST_CHECK_EQUAL(instanceT4.GetArchitecture().GetNumberOfProcessorTypes(),
                      instance.GetArchitecture().GetNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instanceT4.GetArchitecture().CommunicationCosts(), instance.GetArchitecture().CommunicationCosts());
    BOOST_CHECK_EQUAL(instanceT4.GetArchitecture().SynchronisationCosts(), instance.GetArchitecture().SynchronisationCosts());

    BspInstance<ComputationalDagEdgeIdxVectorImplDefT> instanceT5;

    instanceT5 = std::move(instanceT4);
    BOOST_CHECK_EQUAL(instanceT5.GetComputationalDag().NumVertices(), instance.GetComputationalDag().NumVertices());
    BOOST_CHECK_EQUAL(instanceT5.GetComputationalDag().NumVertexTypes(), instance.GetComputationalDag().NumVertexTypes());
    BOOST_CHECK_EQUAL(instanceT5.GetComputationalDag().NumEdges(), instance.GetComputationalDag().NumEdges());
    BOOST_CHECK_EQUAL(instanceT5.GetArchitecture().NumberOfProcessors(), instance.GetArchitecture().NumberOfProcessors());
    BOOST_CHECK_EQUAL(instanceT5.GetArchitecture().GetNumberOfProcessorTypes(),
                      instance.GetArchitecture().GetNumberOfProcessorTypes());
    BOOST_CHECK_EQUAL(instanceT5.GetArchitecture().CommunicationCosts(), instance.GetArchitecture().CommunicationCosts());
    BOOST_CHECK_EQUAL(instanceT5.GetArchitecture().SynchronisationCosts(), instance.GetArchitecture().SynchronisationCosts());
}
