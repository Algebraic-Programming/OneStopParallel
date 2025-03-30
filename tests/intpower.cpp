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

#define BOOST_TEST_MODULE IntPower
#include <boost/test/unit_test.hpp>

#include "auxiliary/misc.hpp"


using namespace osp;

BOOST_AUTO_TEST_CASE(IntegerPowers) {
    BOOST_CHECK_EQUAL(intpow(0, 0), 1);
    BOOST_CHECK_EQUAL(intpow(5, 0), 1);
    BOOST_CHECK_EQUAL(intpow(9, 1), 9);
    BOOST_CHECK_EQUAL(intpow(2, 10), 1024);
    BOOST_CHECK_EQUAL(intpow(7, 3), 343);
    BOOST_CHECK_EQUAL(intpow(1, 349), 1);
    BOOST_CHECK_EQUAL(intpow(1, 4), 1);
    BOOST_CHECK_EQUAL(intpow(3, 2), 9);
    BOOST_CHECK_EQUAL(intpow(4, 3), 64);
}

BOOST_AUTO_TEST_CASE(Median_set) {
    std::set<int> a({0, 10, 20});
    std::set<int> b({-5, 8, 10, 732});
    std::set<int> c({-5, 10, 9, 732});

    BOOST_CHECK_EQUAL(Get_Median(a), 10);
    BOOST_CHECK_EQUAL(Get_Median(b), 9);
    BOOST_CHECK_EQUAL(Get_Median(c), 9);
}

BOOST_AUTO_TEST_CASE(Median_multiset) {
    std::multiset<int> a({0, 10, 20, 10});
    std::multiset<int> b({0, 0, 1});
    std::multiset<int> c({2, 4, 7, 233});

    BOOST_CHECK_EQUAL(Get_Median(a), 10);
    BOOST_CHECK_EQUAL(Get_Median(b), 0);
    BOOST_CHECK_EQUAL(Get_Median(c), 5);
}
