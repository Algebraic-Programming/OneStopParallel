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

#include "osp/bsp/model/BspArchitecture.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(ParameterizedConstructorTest) {
    std::vector<std::vector<int>> uniformSentCosts = {
        {0, 1, 1, 1},
        {1, 0, 1, 1},
        {1, 1, 0, 1},
        {1, 1, 1, 0}
    };

    BspArchitecture<ComputationalDagVectorImplDefIntT> architecture(4, 2, 3);
    BOOST_TEST(architecture.NumberOfProcessors() == 4);
    BOOST_TEST(architecture.CommunicationCosts() == 2);
    BOOST_TEST(architecture.SynchronisationCosts() == 3);
    BOOST_CHECK_EQUAL(architecture.GetMemoryConstraintType(), MemoryConstraintType::NONE);
    BOOST_CHECK_EQUAL(architecture.GetNumberOfProcessorTypes(), 1);
    BOOST_CHECK_EQUAL(architecture.IsNumaArchitecture(), false);

    BOOST_CHECK_EQUAL(architecture.MemoryBound(0), 100);
    BOOST_CHECK_EQUAL(architecture.MemoryBound(1), 100);
    BOOST_CHECK_EQUAL(architecture.MemoryBound(2), 100);
    BOOST_CHECK_EQUAL(architecture.MemoryBound(3), 100);

    BOOST_CHECK_EQUAL(architecture.ProcessorTypes()[0], 0);
    BOOST_CHECK_EQUAL(architecture.ProcessorTypes()[1], 0);
    BOOST_CHECK_EQUAL(architecture.ProcessorTypes()[2], 0);
    BOOST_CHECK_EQUAL(architecture.ProcessorTypes()[3], 0);

    BOOST_CHECK_EQUAL(architecture.ProcessorType(0), 0);
    BOOST_CHECK_EQUAL(architecture.ProcessorType(1), 0);
    BOOST_CHECK_EQUAL(architecture.ProcessorType(2), 0);
    BOOST_CHECK_EQUAL(architecture.ProcessorType(3), 0);

    BOOST_CHECK_EQUAL(architecture.CommunicationCosts(0, 1), 2);
    BOOST_CHECK_EQUAL(architecture.CommunicationCosts(0, 0), 0);

    BOOST_CHECK_EQUAL(architecture.GetProcessorTypeCount().size(), 1);
    BOOST_CHECK_EQUAL(architecture.GetProcessorTypeCount()[0], 4);

    BOOST_CHECK_EQUAL(architecture.GetNumberOfProcessorTypes(), 1);

    BOOST_CHECK_EQUAL(architecture.MaxMemoryBoundProcType(0), 100);

    BOOST_TEST(architecture.SendCost() == uniformSentCosts);

    std::vector<std::vector<int>> expectedSendCosts = {
        {0, 2, 2, 2},
        {2, 0, 2, 2},
        {2, 2, 0, 2},
        {2, 2, 2, 0}
    };

    architecture.SetSendCosts(expectedSendCosts);
    BOOST_TEST(architecture.SendCost() == expectedSendCosts);

    BOOST_CHECK_EQUAL(architecture.CommunicationCosts(0, 1), 4);
    BOOST_CHECK_EQUAL(architecture.CommunicationCosts(0, 0), 0);

    architecture.SetUniformSendCost();
    BOOST_TEST(architecture.SendCost() == uniformSentCosts);

    BOOST_CHECK_EQUAL(architecture.CommunicationCosts(0, 1), 2);
    BOOST_CHECK_EQUAL(architecture.CommunicationCosts(0, 0), 0);
}

BOOST_AUTO_TEST_CASE(Architecture) {
    // default constructor
    BspArchitecture<ComputationalDagVectorImplDefUnsignedT> test;
    BOOST_CHECK_EQUAL(test.NumberOfProcessors(), 2);
    BOOST_CHECK_EQUAL(test.CommunicationCosts(), 1);
    BOOST_CHECK_EQUAL(test.SynchronisationCosts(), 2);
    BOOST_CHECK_EQUAL(test.IsNumaArchitecture(), false);
    BOOST_CHECK_EQUAL(test.SendCosts(0, 1), 1);
    BOOST_CHECK_EQUAL(test.SendCosts(0, 0), 0);
    BOOST_CHECK_EQUAL(test.SendCosts(1, 1), 0);
    BOOST_CHECK_EQUAL(test.SendCosts(1, 0), 1);

    // constructor
    BspArchitecture<ComputationalDagVectorImplDefUnsignedT> test2(5, 7, 14);
    BOOST_CHECK_EQUAL(test2.NumberOfProcessors(), 5);
    BOOST_CHECK_EQUAL(test2.CommunicationCosts(), 7);
    BOOST_CHECK_EQUAL(test2.SynchronisationCosts(), 14);
    BOOST_CHECK_EQUAL(test2.IsNumaArchitecture(), false);

    for (unsigned i = 0; i < 5; i++) {
        for (unsigned j = 0; j < 5; j++) {
            if (i == j) {
                BOOST_CHECK_EQUAL(test2.SendCosts(i, j), 0);
                BOOST_CHECK_EQUAL(test2.CommunicationCosts(i, j), 0);
            } else {
                BOOST_CHECK_EQUAL(test2.SendCosts(i, j), 1);
                BOOST_CHECK_EQUAL(test2.CommunicationCosts(i, j), 7);
            }
        }
    }

    test2.SetCommunicationCosts(14);
    BOOST_CHECK_EQUAL(test2.CommunicationCosts(), 14);

    for (unsigned i = 0; i < 5; i++) {
        for (unsigned j = 0; j < 5; j++) {
            if (i == j) {
                BOOST_CHECK_EQUAL(test2.SendCosts(i, j), 0);
                BOOST_CHECK_EQUAL(test2.CommunicationCosts(i, j), 0);
            } else {
                BOOST_CHECK_EQUAL(test2.SendCosts(i, j), 1);
                BOOST_CHECK_EQUAL(test2.CommunicationCosts(i, j), 14);
            }
        }
    }

    test2.SetCommunicationCosts(0);
    BOOST_CHECK_EQUAL(test2.CommunicationCosts(), 0);

    for (unsigned i = 0; i < 5; i++) {
        for (unsigned j = 0; j < 5; j++) {
            if (i == j) {
                BOOST_CHECK_EQUAL(test2.SendCosts(i, j), 0);
                BOOST_CHECK_EQUAL(test2.CommunicationCosts(i, j), 0);
            } else {
                BOOST_CHECK_EQUAL(test2.SendCosts(i, j), 1);
                BOOST_CHECK_EQUAL(test2.CommunicationCosts(i, j), 0);
            }
        }
    }

    // constructor
    std::vector<std::vector<int>> sendCosts = {
        {0, 1, 1, 1, 1, 1},
        {1, 0, 1, 1, 1, 1},
        {1, 1, 0, 1, 1, 1},
        {1, 1, 1, 0, 1, 1},
        {1, 1, 1, 1, 0, 1},
        {1, 1, 1, 1, 1, 0}
    };

    BOOST_CHECK_THROW(BspArchitecture<ComputationalDagVectorImplDefIntT> test31(7, 42942, 0, sendCosts), std::invalid_argument);
    BOOST_CHECK_THROW(BspArchitecture<ComputationalDagVectorImplDefIntT> test32(5, 42942, 0, sendCosts), std::invalid_argument);

    BspArchitecture<ComputationalDagVectorImplDefIntT> test3(6, 47295, 0, sendCosts);
    BOOST_CHECK_EQUAL(test3.NumberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test3.CommunicationCosts(), 47295);
    BOOST_CHECK_EQUAL(test3.SynchronisationCosts(), 0);
    BOOST_CHECK_EQUAL(test3.IsNumaArchitecture(), false);

    for (unsigned i = 0; i < 6; i++) {
        for (unsigned j = 0; j < 6; j++) {
            if (i == j) {
                BOOST_CHECK_EQUAL(test3.SendCosts(i, j), 0);
                BOOST_CHECK_EQUAL(test3.CommunicationCosts(i, j), 0);

            } else {
                BOOST_CHECK_EQUAL(test3.SendCosts(i, j), 1);
                BOOST_CHECK_EQUAL(test3.CommunicationCosts(i, j), 47295);
            }
        }
    }

    // constructor
    std::vector<std::vector<int>> sendCosts2 = {
        {0, 1, 2, 1, 1, 1},
        {1, 0, 1, 1, 1, 1},
        {1, 1, 0, 1, 1, 1},
        {1, 1, 1, 0, 1, 1},
        {1, 1, 1, 1, 0, 1},
        {1, 1, 1, 1, 1, 0}
    };
    std::vector<std::vector<int>> sendCosts3 = {
        {0, 1, 1, 1, 1, 1},
        {1, 0, 1, 1, 1, 1},
        {1, 1, 0, 1, 1, 1},
        {3, 1, 1, 0, 1, 1},
        {1, 1, 1, 1, 0, 1},
        {1, 1, 1, 1, 1, 0}
    };

    BspArchitecture<ComputationalDagVectorImplDefIntT> test4(6, 0, 4294965, sendCosts2);
    BOOST_CHECK_EQUAL(test4.NumberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test4.CommunicationCosts(), 0);
    BOOST_CHECK_EQUAL(test4.SynchronisationCosts(), 4294965);
    BOOST_CHECK_EQUAL(test4.IsNumaArchitecture(), true);
    BOOST_CHECK_EQUAL(test4.SendCosts(0, 2), 2);

    BspArchitecture<ComputationalDagVectorImplDefIntT> test5(6, 0, 4294965, sendCosts3);
    BOOST_CHECK_EQUAL(test5.NumberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test5.CommunicationCosts(), 0);
    BOOST_CHECK_EQUAL(test5.SynchronisationCosts(), 4294965);
    BOOST_CHECK_EQUAL(test5.IsNumaArchitecture(), true);
    BOOST_CHECK_EQUAL(test5.SendCosts(3, 0), 3);

    test5.SetNumberOfProcessors(8);
    BOOST_CHECK_EQUAL(test5.NumberOfProcessors(), 8);
    BOOST_CHECK_EQUAL(test5.CommunicationCosts(), 0);
    BOOST_CHECK_EQUAL(test5.SynchronisationCosts(), 4294965);
    BOOST_CHECK_EQUAL(test5.SendCosts(3, 0), 1);
    BOOST_CHECK_EQUAL(test5.SendCosts(7, 7), 0);
    BOOST_CHECK_EQUAL(test5.SendCosts(7, 6), 1);
    BOOST_CHECK_EQUAL(test5.SendCosts(3, 5), 1);
    BOOST_CHECK_EQUAL(test5.IsNumaArchitecture(), false);

    test.SetNumberOfProcessors(5);
    BOOST_CHECK_EQUAL(test.NumberOfProcessors(), 5);
    BOOST_CHECK_EQUAL(test.CommunicationCosts(), 1);
    BOOST_CHECK_EQUAL(test.SynchronisationCosts(), 2);
    BOOST_CHECK_EQUAL(test.SendCosts(4, 3), 1);
    BOOST_CHECK_EQUAL(test.IsNumaArchitecture(), false);
}
