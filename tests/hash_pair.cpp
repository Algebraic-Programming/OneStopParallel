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

#define BOOST_TEST_MODULE Hash_Pair
#include <boost/test/unit_test.hpp>

#include "osp/auxiliary/hash_util.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(HashPair) {
    std::pair<int, int> p1({0, 0});
    std::pair<int, int> p2({1, 1});
    std::pair<int, int> p3({1, 2});
    std::pair<int, int> p4({2, 1});
    std::pair<int, int> p5({1, 3});
    std::pair<int, int> p6({2, 6});
    std::pair<int, int> p7 = p6;

    PairHash hasher;

    BOOST_CHECK(hasher(p7) == hasher(p6));

    // Can technically fail, but should not
    BOOST_CHECK(hasher(p1) != hasher(p2));
    BOOST_CHECK(hasher(p3) != hasher(p4));
    BOOST_CHECK(hasher(p2) != hasher(p3));
    BOOST_CHECK(hasher(p2) != hasher(p5));
    BOOST_CHECK(hasher(p4) != hasher(p6));
}
