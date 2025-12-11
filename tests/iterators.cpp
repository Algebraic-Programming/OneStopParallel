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

#include "osp/graph_implementations/integral_range.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(IntegralRangeTest) {
    IntegralRange<unsigned> range(0, 10);
    BOOST_CHECK_EQUAL(range.Size(), 10);

    int count = 0;
    for (auto it = range.Begin(); it != range.End(); ++it) {
        BOOST_CHECK_EQUAL(*it, count);
        ++count;
    }
    BOOST_CHECK_EQUAL(count, 10);
    count = 9;
    for (auto it = range.Rbegin(); it != range.Rend(); ++it) {
        BOOST_CHECK_EQUAL(*it, count);
        --count;
    }
    BOOST_CHECK_EQUAL(count, -1);
    count = 0;
    for (auto it = range.Cbegin(); it != range.Cend(); ++it) {
        BOOST_CHECK_EQUAL(*it, count);
        ++count;
    }
    BOOST_CHECK_EQUAL(count, 10);
    count = 9;
    for (auto it = range.Crbegin(); it != range.Crend(); ++it) {
        BOOST_CHECK_EQUAL(*it, count);
        --count;
    }
    BOOST_CHECK_EQUAL(count, -1);

    count = 0;
    IntegralRange<unsigned> range2(10);
    BOOST_CHECK_EQUAL(range2.Size(), 10);

    for (auto v : range2) {
        BOOST_CHECK_EQUAL(v, count);
        ++count;
    }

    BOOST_CHECK_EQUAL(count, 10);
    count = 9;
    for (auto it = range2.Rbegin(); it != range2.Rend(); ++it) {
        BOOST_CHECK_EQUAL(*it, count);
        --count;
    }
    BOOST_CHECK_EQUAL(count, -1);

    count = 5;
    IntegralRange<unsigned> range3(5, 15);
    BOOST_CHECK_EQUAL(range3.Size(), 10);

    for (auto v : range3) {
        BOOST_CHECK_EQUAL(v, count);
        ++count;
    }
    BOOST_CHECK_EQUAL(count, 15);
    count = 14;
    for (auto it = range3.Rbegin(); it != range3.Rend(); ++it) {
        BOOST_CHECK_EQUAL(*it, count);
        --count;
    }
    BOOST_CHECK_EQUAL(count, 4);
}
