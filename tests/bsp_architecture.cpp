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

    BspArchitecture<computational_dag_vector_impl_def_int_t> architecture(4, 2, 3);
    BOOST_TEST(architecture.NumberOfProcessors() == 4);
    BOOST_TEST(architecture.CommunicationCosts() == 2);
    BOOST_TEST(architecture.SynchronisationCosts() == 3);
    BOOST_CHECK_EQUAL(architecture.GetMemoryConstraintType(), MemoryConstraintType::NONE);
    BOOST_CHECK_EQUAL(architecture.GetNumberOfProcessorTypes(), 1);
    BOOST_CHECK_EQUAL(architecture.IsNumaArchitecture(), false);

    BOOST_CHECK_EQUAL(architecture.memoryBound(0), 100);
    BOOST_CHECK_EQUAL(architecture.memoryBound(1), 100);
    BOOST_CHECK_EQUAL(architecture.memoryBound(2), 100);
    BOOST_CHECK_EQUAL(architecture.memoryBound(3), 100);

    BOOST_CHECK_EQUAL(architecture.processorTypes()[0], 0);
    BOOST_CHECK_EQUAL(architecture.processorTypes()[1], 0);
    BOOST_CHECK_EQUAL(architecture.processorTypes()[2], 0);
    BOOST_CHECK_EQUAL(architecture.processorTypes()[3], 0);

    BOOST_CHECK_EQUAL(architecture.ProcessorType(0), 0);
    BOOST_CHECK_EQUAL(architecture.ProcessorType(1), 0);
    BOOST_CHECK_EQUAL(architecture.ProcessorType(2), 0);
    BOOST_CHECK_EQUAL(architecture.ProcessorType(3), 0);

    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 1), 2);
    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 0), 0);

    BOOST_CHECK_EQUAL(architecture.getProcessorTypeCount().size(), 1);
    BOOST_CHECK_EQUAL(architecture.getProcessorTypeCount()[0], 4);

    BOOST_CHECK_EQUAL(architecture.GetNumberOfProcessorTypes(), 1);

    BOOST_CHECK_EQUAL(architecture.maxMemoryBoundProcType(0), 100);

    BOOST_TEST(architecture.sendCost() == uniformSentCosts);

    std::vector<std::vector<int>> expectedSendCosts = {
        {0, 2, 2, 2},
        {2, 0, 2, 2},
        {2, 2, 0, 2},
        {2, 2, 2, 0}
    };

    architecture.SetSendCosts(expectedSendCosts);
    BOOST_TEST(architecture.sendCost() == expectedSendCosts);

    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 1), 4);
    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 0), 0);

    architecture.SetUniformSendCost();
    BOOST_TEST(architecture.sendCost() == uniformSentCosts);

    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 1), 2);
    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 0), 0);
}

BOOST_AUTO_TEST_CASE(Architecture) {
    // default constructor
    BspArchitecture<computational_dag_vector_impl_def_t> test;
    BOOST_CHECK_EQUAL(test.NumberOfProcessors(), 2);
    BOOST_CHECK_EQUAL(test.CommunicationCosts(), 1);
    BOOST_CHECK_EQUAL(test.SynchronisationCosts(), 2);
    BOOST_CHECK_EQUAL(test.IsNumaArchitecture(), false);
    BOOST_CHECK_EQUAL(test.sendCosts(0, 1), 1);
    BOOST_CHECK_EQUAL(test.sendCosts(0, 0), 0);
    BOOST_CHECK_EQUAL(test.sendCosts(1, 1), 0);
    BOOST_CHECK_EQUAL(test.sendCosts(1, 0), 1);

    // constructor
    BspArchitecture<computational_dag_vector_impl_def_t> test2(5, 7, 14);
    BOOST_CHECK_EQUAL(test2.NumberOfProcessors(), 5);
    BOOST_CHECK_EQUAL(test2.CommunicationCosts(), 7);
    BOOST_CHECK_EQUAL(test2.SynchronisationCosts(), 14);
    BOOST_CHECK_EQUAL(test2.IsNumaArchitecture(), false);

    for (unsigned i = 0; i < 5; i++) {
        for (unsigned j = 0; j < 5; j++) {
            if (i == j) {
                BOOST_CHECK_EQUAL(test2.sendCosts(i, j), 0);
                BOOST_CHECK_EQUAL(test2.communicationCosts(i, j), 0);
            } else {
                BOOST_CHECK_EQUAL(test2.sendCosts(i, j), 1);
                BOOST_CHECK_EQUAL(test2.communicationCosts(i, j), 7);
            }
        }
    }

    test2.setCommunicationCosts(14);
    BOOST_CHECK_EQUAL(test2.CommunicationCosts(), 14);

    for (unsigned i = 0; i < 5; i++) {
        for (unsigned j = 0; j < 5; j++) {
            if (i == j) {
                BOOST_CHECK_EQUAL(test2.sendCosts(i, j), 0);
                BOOST_CHECK_EQUAL(test2.communicationCosts(i, j), 0);
            } else {
                BOOST_CHECK_EQUAL(test2.sendCosts(i, j), 1);
                BOOST_CHECK_EQUAL(test2.communicationCosts(i, j), 14);
            }
        }
    }

    test2.setCommunicationCosts(0);
    BOOST_CHECK_EQUAL(test2.CommunicationCosts(), 0);

    for (unsigned i = 0; i < 5; i++) {
        for (unsigned j = 0; j < 5; j++) {
            if (i == j) {
                BOOST_CHECK_EQUAL(test2.sendCosts(i, j), 0);
                BOOST_CHECK_EQUAL(test2.communicationCosts(i, j), 0);
            } else {
                BOOST_CHECK_EQUAL(test2.sendCosts(i, j), 1);
                BOOST_CHECK_EQUAL(test2.communicationCosts(i, j), 0);
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

    BOOST_CHECK_THROW(BspArchitecture<computational_dag_vector_impl_def_int_t> test31(7, 42942, 0, sendCosts),
                      std::invalid_argument);
    BOOST_CHECK_THROW(BspArchitecture<computational_dag_vector_impl_def_int_t> test32(5, 42942, 0, sendCosts),
                      std::invalid_argument);

    BspArchitecture<computational_dag_vector_impl_def_int_t> test3(6, 47295, 0, sendCosts);
    BOOST_CHECK_EQUAL(test3.NumberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test3.CommunicationCosts(), 47295);
    BOOST_CHECK_EQUAL(test3.SynchronisationCosts(), 0);
    BOOST_CHECK_EQUAL(test3.IsNumaArchitecture(), false);

    for (unsigned i = 0; i < 6; i++) {
        for (unsigned j = 0; j < 6; j++) {
            if (i == j) {
                BOOST_CHECK_EQUAL(test3.sendCosts(i, j), 0);
                BOOST_CHECK_EQUAL(test3.communicationCosts(i, j), 0);

            } else {
                BOOST_CHECK_EQUAL(test3.sendCosts(i, j), 1);
                BOOST_CHECK_EQUAL(test3.communicationCosts(i, j), 47295);
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

    BspArchitecture<computational_dag_vector_impl_def_int_t> test4(6, 0, 4294965, sendCosts2);
    BOOST_CHECK_EQUAL(test4.NumberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test4.CommunicationCosts(), 0);
    BOOST_CHECK_EQUAL(test4.SynchronisationCosts(), 4294965);
    BOOST_CHECK_EQUAL(test4.IsNumaArchitecture(), true);
    BOOST_CHECK_EQUAL(test4.sendCosts(0, 2), 2);

    BspArchitecture<computational_dag_vector_impl_def_int_t> test5(6, 0, 4294965, sendCosts3);
    BOOST_CHECK_EQUAL(test5.NumberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test5.CommunicationCosts(), 0);
    BOOST_CHECK_EQUAL(test5.SynchronisationCosts(), 4294965);
    BOOST_CHECK_EQUAL(test5.IsNumaArchitecture(), true);
    BOOST_CHECK_EQUAL(test5.sendCosts(3, 0), 3);

    test5.setNumberOfProcessors(8);
    BOOST_CHECK_EQUAL(test5.NumberOfProcessors(), 8);
    BOOST_CHECK_EQUAL(test5.CommunicationCosts(), 0);
    BOOST_CHECK_EQUAL(test5.SynchronisationCosts(), 4294965);
    BOOST_CHECK_EQUAL(test5.sendCosts(3, 0), 1);
    BOOST_CHECK_EQUAL(test5.sendCosts(7, 7), 0);
    BOOST_CHECK_EQUAL(test5.sendCosts(7, 6), 1);
    BOOST_CHECK_EQUAL(test5.sendCosts(3, 5), 1);
    BOOST_CHECK_EQUAL(test5.IsNumaArchitecture(), false);

    test.setNumberOfProcessors(5);
    BOOST_CHECK_EQUAL(test.NumberOfProcessors(), 5);
    BOOST_CHECK_EQUAL(test.CommunicationCosts(), 1);
    BOOST_CHECK_EQUAL(test.SynchronisationCosts(), 2);
    BOOST_CHECK_EQUAL(test.sendCosts(4, 3), 1);
    BOOST_CHECK_EQUAL(test.IsNumaArchitecture(), false);
}
