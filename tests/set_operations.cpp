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

#define BOOST_TEST_MODULE Sets
#include <boost/test/unit_test.hpp>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "osp/auxiliary/misc.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(SetIntersection) {
    std::unordered_set<int> a({3, 6, 2, 6, 7});
    std::unordered_set<int> b({6, 6});
    std::unordered_set<int> c({6});
    std::unordered_set<int> d({1, 5});
    std::unordered_set<int> e({});
    std::unordered_set<int> f({1, 2, 3});
    std::unordered_set<int> g({3, 6, 2, 7});
    std::unordered_set<int> h({2, 3});
    std::unordered_set<int> i({3, 2});
    std::unordered_set<int> j({1});

    BOOST_CHECK(GetIntersection(a, b) == c);
    BOOST_CHECK(GetIntersection(b, a) == c);
    BOOST_CHECK(GetIntersection(c, a) == c);
    BOOST_CHECK(GetIntersection(g, g) == g);
    BOOST_CHECK(GetIntersection(a, g) == g);
    BOOST_CHECK(GetIntersection(a, a) == g);
    BOOST_CHECK(GetIntersection(a, f) == i);
    BOOST_CHECK(GetIntersection(a, e) == e);
    BOOST_CHECK(GetIntersection(d, f) == j);
}

BOOST_AUTO_TEST_CASE(SetIntersectionLarge) {
    std::vector<int> iota0To10k(10'000);
    std::iota(iota0To10k.begin(), iota0To10k.end(), 0);

    std::vector<int> iota10kTo20k(10'000);
    std::iota(iota10kTo20k.begin(), iota10kTo20k.end(), 10'000);

    std::unordered_set<int> iota0To10kSet(iota0To10k.begin(), iota0To10k.end());

    {    // Intersection of [0,10k] and [10k,20k]  -->  []
        std::unordered_set<int> iota10kTo20kSet(iota10kTo20k.begin(), iota10kTo20k.end());
        BOOST_CHECK(GetIntersection(iota0To10kSet, iota10kTo20kSet).empty());
    }

    {    // Intersection of [0,10k] and [0k,10k]  -->  [0k,10k]
        BOOST_CHECK(GetIntersection(iota0To10kSet, iota0To10kSet) == iota0To10kSet);
    }

    {    // Intersection of [0,10k] and [5k,10k]  -->  [5k,10k]
        std::vector<int> iota5kTo10k(5'000);
        std::iota(iota5kTo10k.begin(), iota5kTo10k.end(), 5'000);
        std::unordered_set<int> iota5kTo10kSet(iota5kTo10k.begin(), iota5kTo10k.end());

        BOOST_CHECK(GetIntersection(iota0To10kSet, iota5kTo10kSet) == iota5kTo10kSet);
    }
}

BOOST_AUTO_TEST_CASE(SetUnions) {
    std::unordered_set<int> a({3, 6, 2, 6, 7});
    std::unordered_set<int> b({6, 6});
    std::unordered_set<int> c({6});
    std::unordered_set<int> d({1, 5});
    std::unordered_set<int> e({});
    std::unordered_set<int> f({1, 2, 3});
    std::unordered_set<int> g({3, 6, 2, 7});
    std::unordered_set<int> h({2, 3});
    std::unordered_set<int> i({3, 2});
    std::unordered_set<int> j({1});
    std::unordered_set<int> k({1, 2, 3, 6, 7});
    std::unordered_set<int> l({1, 2, 3, 5});

    BOOST_CHECK(GetUnion(a, b) == g);
    BOOST_CHECK(GetUnion(b, a) == a);
    BOOST_CHECK(GetUnion(c, a) == g);
    BOOST_CHECK(GetUnion(g, g) == g);
    BOOST_CHECK(GetUnion(a, g) == g);
    BOOST_CHECK(GetUnion(a, a) == g);
    BOOST_CHECK(GetUnion(a, f) == k);
    BOOST_CHECK(GetUnion(a, e) == a);
    BOOST_CHECK(GetUnion(d, f) == l);
}

BOOST_AUTO_TEST_CASE(SetUnionLarge) {
    std::vector<int> iota0To10k(10'000);
    std::iota(iota0To10k.begin(), iota0To10k.end(), 0);

    std::vector<int> iota10kTo20k(10'000);
    std::iota(iota10kTo20k.begin(), iota10kTo20k.end(), 10'000);

    std::unordered_set<int> iota0To10kSet(iota0To10k.begin(), iota0To10k.end());

    {    // Union of [0,10k] and [10k,20k]  -->  [0k,20k]
        std::unordered_set<int> iota10kTo20kSet(iota10kTo20k.begin(), iota10kTo20k.end());
        std::unordered_set<int> expectedUnion(iota0To10k.begin(), iota0To10k.end());
        expectedUnion.insert(iota10kTo20k.begin(), iota10kTo20k.end());
        BOOST_CHECK(GetUnion(iota0To10kSet, iota10kTo20kSet) == expectedUnion);
    }

    {    // Union of [0,10k] and [0k,10k]  -->  [0k,10k]
        BOOST_CHECK(GetUnion(iota0To10kSet, iota0To10kSet) == iota0To10kSet);
    }

    {    // Union of [0,10k] and [5k,15k]  -->  [0k,15k]
        std::vector<int> iota5kTo15k(10'000);
        std::iota(iota5kTo15k.begin(), iota5kTo15k.end(), 5'000);
        std::unordered_set<int> iota5kTo15kSet(iota5kTo15k.begin(), iota5kTo15k.end());
        std::unordered_set<int> expectedUnion(iota0To10k.begin(), iota0To10k.end());
        expectedUnion.insert(iota5kTo15k.begin(), iota5kTo15k.end());
        BOOST_CHECK(GetUnion(iota0To10kSet, iota5kTo15kSet) == expectedUnion);
    }
}
