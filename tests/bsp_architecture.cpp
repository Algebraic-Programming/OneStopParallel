#define BOOST_TEST_MODULE Bsp_Architecture
#include <boost/test/unit_test.hpp>

#include "model/BspArchitecture.hpp"

BOOST_AUTO_TEST_CASE(ParameterizedConstructorTest)
{
    BspArchitecture architecture(4, 2, 3);
    BOOST_TEST(architecture.numberOfProcessors() == 4);
    BOOST_TEST(architecture.communicationCosts() == 2);
    BOOST_TEST(architecture.synchronisationCosts() == 3);

    std::vector<std::vector<unsigned int>> expectedSendCosts = {
        {0, 2, 2, 2},
        {2, 0, 2, 2},
        {2, 2, 0, 2},
        {2, 2, 2, 0}
    };

    architecture.setSendCosts(expectedSendCosts);
    BOOST_TEST(architecture.sendCostMatrix() == expectedSendCosts);
}

BOOST_AUTO_TEST_CASE(Architecture) {

    // default constructor
    BspArchitecture test;
    BOOST_CHECK_EQUAL(test.numberOfProcessors(), 2);
    BOOST_CHECK_EQUAL(test.communicationCosts(), 1);
    BOOST_CHECK_EQUAL(test.synchronisationCosts(), 2);
    BOOST_CHECK_EQUAL(test.isNumaArchitecture(), false);
    BOOST_CHECK_EQUAL(test.sendCosts(0, 1), 1);
    BOOST_CHECK_EQUAL(test.sendCosts(0, 0), 0);
    BOOST_CHECK_EQUAL(test.sendCosts(1, 1), 0);
    BOOST_CHECK_EQUAL(test.sendCosts(1, 0), 1);

    // constructor
    BspArchitecture test2(5, 7, 14);
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
    std::vector<std::vector<unsigned int>> send_costs = {{0, 1, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1},
                                                         {1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}};

    BOOST_CHECK_THROW(BspArchitecture test31(7, 4294967295, 0, send_costs), std::invalid_argument);
    BOOST_CHECK_THROW(BspArchitecture test32(5, 4294967295, 0, send_costs), std::invalid_argument);

    BspArchitecture test3(6, 4294967295, 0, send_costs);
    BOOST_CHECK_EQUAL(test3.numberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test3.communicationCosts(), 4294967295);
    BOOST_CHECK_EQUAL(test3.synchronisationCosts(), 0);
    BOOST_CHECK_EQUAL(test3.isNumaArchitecture(), false);

    for (unsigned i = 0; i < 6; i++) {
        for (unsigned j = 0; j < 6; j++) {
            if (i == j) {
                BOOST_CHECK_EQUAL(test3.sendCosts(i, j), 0);
                BOOST_CHECK_EQUAL(test3.communicationCosts(i, j), 0);

            } else {
                BOOST_CHECK_EQUAL(test3.sendCosts(i, j), 1);
                BOOST_CHECK_EQUAL(test3.communicationCosts(i, j), 4294967295);
            }
        }
    }

    // constructor
    std::vector<std::vector<unsigned int>> send_costs2 = {{0, 1, 2, 1, 1, 1}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1},
                                                          {1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}};
    std::vector<std::vector<unsigned int>> send_costs3 = {{0, 1, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1},
                                                          {3, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0}};

    BspArchitecture test4(6, 0, 4294967295, send_costs2);
    BOOST_CHECK_EQUAL(test4.numberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test4.communicationCosts(), 0);
    BOOST_CHECK_EQUAL(test4.synchronisationCosts(), 4294967295);
    BOOST_CHECK_EQUAL(test4.isNumaArchitecture(), true);
    BOOST_CHECK_EQUAL(test4.sendCosts(0, 2), 2);

    BspArchitecture test5(6, 0, 4294967295, send_costs3);
    BOOST_CHECK_EQUAL(test5.numberOfProcessors(), 6);
    BOOST_CHECK_EQUAL(test5.communicationCosts(), 0);
    BOOST_CHECK_EQUAL(test5.synchronisationCosts(), 4294967295);
    BOOST_CHECK_EQUAL(test5.isNumaArchitecture(), true);
    BOOST_CHECK_EQUAL(test5.sendCosts(3, 0), 3);

    test5.setNumberOfProcessors(8);
    BOOST_CHECK_EQUAL(test5.numberOfProcessors(), 8);
    BOOST_CHECK_EQUAL(test5.communicationCosts(), 0);
    BOOST_CHECK_EQUAL(test5.synchronisationCosts(), 4294967295);
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
    
};
