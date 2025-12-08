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

    std::vector<std::vector<int>> uniform_sent_costs = {{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 1, 0}};

    BspArchitecture<computational_dag_vector_impl_def_int_t> architecture(4, 2, 3);
    BOOST_TEST(architecture.numberOfProcessors() == 4);
    BOOST_TEST(architecture.communicationCosts() == 2);
    BOOST_TEST(architecture.synchronisationCosts() == 3);
    BOOST_CHECK_EQUAL(architecture.getMemoryConstraintType(), MEMORY_CONSTRAINT_TYPE::NONE);
    BOOST_CHECK_EQUAL(architecture.getNumberOfProcessorTypes(), 1);
    BOOST_CHECK_EQUAL(architecture.isNumaArchitecture(), false);

    BOOST_CHECK_EQUAL(architecture.memoryBound(0), 100);
    BOOST_CHECK_EQUAL(architecture.memoryBound(1), 100);
    BOOST_CHECK_EQUAL(architecture.memoryBound(2), 100);
    BOOST_CHECK_EQUAL(architecture.memoryBound(3), 100);

    BOOST_CHECK_EQUAL(architecture.processorTypes()[0], 0);
    BOOST_CHECK_EQUAL(architecture.processorTypes()[1], 0);
    BOOST_CHECK_EQUAL(architecture.processorTypes()[2], 0);
    BOOST_CHECK_EQUAL(architecture.processorTypes()[3], 0);

    BOOST_CHECK_EQUAL(architecture.processorType(0), 0);
    BOOST_CHECK_EQUAL(architecture.processorType(1), 0);
    BOOST_CHECK_EQUAL(architecture.processorType(2), 0);
    BOOST_CHECK_EQUAL(architecture.processorType(3), 0);

    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 1), 2);
    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 0), 0);

    BOOST_CHECK_EQUAL(architecture.getProcessorTypeCount().size(), 1);
    BOOST_CHECK_EQUAL(architecture.getProcessorTypeCount()[0], 4);

    BOOST_CHECK_EQUAL(architecture.getNumberOfProcessorTypes(), 1);

    BOOST_CHECK_EQUAL(architecture.maxMemoryBoundProcType(0), 100);

    BOOST_TEST(architecture.sendCostMatrix() == uniform_sent_costs);

    std::vector<std::vector<int>> expectedSendCosts = {{0, 2, 2, 2}, {2, 0, 2, 2}, {2, 2, 0, 2}, {2, 2, 2, 0}};

    architecture.SetSendCosts(expectedSendCosts);
    BOOST_TEST(architecture.sendCostMatrix() == expectedSendCosts);

    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 1), 4);
    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 0), 0);

    architecture.SetUniformSendCost();
    BOOST_TEST(architecture.sendCostMatrix() == uniform_sent_costs);

    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 1), 2);
    BOOST_CHECK_EQUAL(architecture.communicationCosts(0, 0), 0);
}

BOOST_AUTO_TEST_CASE(Architecture) {

    // default constructor
    BspArchitecture<computational_dag_vector_impl_def_t> test;
    BOOST_CHECK_EQUAL(test.numberOfProcessors(), 2);
    BOOST_CHECK_EQUAL(test.communicationCosts(), 1);
    BOOST_CHECK_EQUAL(test.synchronisationCosts(), 2);
    BOOST_CHECK_EQUAL(test.isNumaArchitecture(), false);
    BOOST_CHECK_EQUAL(test.sendCosts(0, 1), 1);
    BOOST_CHECK_EQUAL(test.sendCosts(0, 0), 0);
    BOOST_CHECK_EQUAL(test.sendCosts(1, 1), 0);
    BOOST_CHECK_EQUAL(test.sendCosts(1, 0), 1);

    // constructor
    BspArchitecture<computational_dag_vector_impl_def_t> test2(5, 7, 14);
    BOOST_CHECK_EQUAL(test2.numberOfProcessors(), 5);
    BOOST_CHECK_EQUAL(test2.communicationCosts(), 7);
    BOOST_CHECK_EQUAL(test2.synchronisationCosts(), 14);
    BOOST_CHECK_EQUAL(test2.isNumaArchitecture(), false);

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
    BOOST_CHECK_EQUAL(test2.communicationCosts(), 14);

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
    BOOST_CHECK_EQUAL(test2.communicationCosts(), 0);

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
    std::vector<std::vector<int>> send_costs = {{0, 1, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1}, {1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}};

    BOOST_CHECK_THROW(BspArchitecture<computational_dag_vector_impl_def_int_t> test31(7, 42942, 0, send_costs),
                      std::invalid_argument);
    BOOST_CHECK_THROW(BspArchitecture<computational_dag_vector_impl_def_int_t> test32(5, 42942, 0, send_costs),
                      std::invalid_argument);

    BspArchitecture<computational_dag_vector_impl_def_int_t> test3(6, 47295, 0, send_costs);
    BOOST_CHECK_EQUAL(test3.numberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test3.communicationCosts(), 47295);
    BOOST_CHECK_EQUAL(test3.synchronisationCosts(), 0);
    BOOST_CHECK_EQUAL(test3.isNumaArchitecture(), false);

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
    std::vector<std::vector<int>> send_costs2 = {{0, 1, 2, 1, 1, 1}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1}, {1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}};
    std::vector<std::vector<int>> send_costs3 = {{0, 1, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1}, {3, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}};

    BspArchitecture<computational_dag_vector_impl_def_int_t> test4(6, 0, 4294965, send_costs2);
    BOOST_CHECK_EQUAL(test4.numberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test4.communicationCosts(), 0);
    BOOST_CHECK_EQUAL(test4.synchronisationCosts(), 4294965);
    BOOST_CHECK_EQUAL(test4.isNumaArchitecture(), true);
    BOOST_CHECK_EQUAL(test4.sendCosts(0, 2), 2);

    BspArchitecture<computational_dag_vector_impl_def_int_t> test5(6, 0, 4294965, send_costs3);
    BOOST_CHECK_EQUAL(test5.numberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test5.communicationCosts(), 0);
    BOOST_CHECK_EQUAL(test5.synchronisationCosts(), 4294965);
    BOOST_CHECK_EQUAL(test5.isNumaArchitecture(), true);
    BOOST_CHECK_EQUAL(test5.sendCosts(3, 0), 3);

    test5.setNumberOfProcessors(8);
    BOOST_CHECK_EQUAL(test5.numberOfProcessors(), 8);
    BOOST_CHECK_EQUAL(test5.communicationCosts(), 0);
    BOOST_CHECK_EQUAL(test5.synchronisationCosts(), 4294965);
    BOOST_CHECK_EQUAL(test5.sendCosts(3, 0), 1);
    BOOST_CHECK_EQUAL(test5.sendCosts(7, 7), 0);
    BOOST_CHECK_EQUAL(test5.sendCosts(7, 6), 1);
    BOOST_CHECK_EQUAL(test5.sendCosts(3, 5), 1);
    BOOST_CHECK_EQUAL(test5.isNumaArchitecture(), false);

    test.setNumberOfProcessors(5);
    BOOST_CHECK_EQUAL(test.numberOfProcessors(), 5);
    BOOST_CHECK_EQUAL(test.communicationCosts(), 1);
    BOOST_CHECK_EQUAL(test.synchronisationCosts(), 2);
    BOOST_CHECK_EQUAL(test.sendCosts(4, 3), 1);
    BOOST_CHECK_EQUAL(test.isNumaArchitecture(), false);
}
