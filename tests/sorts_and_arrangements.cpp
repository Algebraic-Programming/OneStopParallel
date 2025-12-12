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

#define BOOST_TEST_MODULE Sorts_and_Arrangements
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "osp/auxiliary/misc.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(SortsAndArrangements1) {
    std::vector<int> a({4, 7, 2, -2, 4});
    std::vector<size_t> negTest1({4, 7, 2, 8, 4});
    std::vector<size_t> negTest2({8, 2, 4, 4, 7});
    std::vector<int> b = a;
    std::vector<int> aSort({-2, 2, 4, 4, 7});
    std::vector<size_t> aRe1({3, 2, 0, 4, 1});
    std::vector<size_t> aRe2({3, 2, 4, 0, 1});

    std::vector<size_t> re = sort_and_sorting_arrangement(a);
    BOOST_CHECK(re == aRe1 || re == aRe2);
    BOOST_CHECK(a == aSort);

    BOOST_CHECK(check_vector_is_rearrangement_of_0_to_N(re));
    BOOST_CHECK(check_vector_is_rearrangement_of_0_to_N(aRe1));
    BOOST_CHECK(!check_vector_is_rearrangement_of_0_to_N(negTest1));
    BOOST_CHECK(!check_vector_is_rearrangement_of_0_to_N(negTest2));

    std::cout << "b: ";
    for (auto &i : b) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;

    sort_like_arrangement(b, re);

    std::cout << "re: ";
    for (auto &i : re) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;

    std::cout << "b: ";
    for (auto &i : b) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;

    BOOST_CHECK(a == b);

    std::cout << "a: ";
    for (auto &i : a) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(SortsAndArrangements2) {
    std::vector<std::string> a({"aa", "z", "b", "trace", "racket"});
    std::vector<size_t> c({16, 901, 2, 8, 29});
    std::vector<size_t> b = c;
    std::vector<std::string> aSort({"b", "trace", "aa", "racket", "z"});
    std::vector<size_t> cRe({2, 3, 0, 4, 1});

    BOOST_CHECK(check_vector_is_rearrangement_of_0_to_N(cRe));
    BOOST_CHECK(!check_vector_is_rearrangement_of_0_to_N(c));

    BOOST_CHECK(sorting_arrangement(c) == cRe);
    BOOST_CHECK(c == b);

    sort_like(a, c);

    BOOST_CHECK(a == aSort);
    BOOST_CHECK(c == b);
}

BOOST_AUTO_TEST_CASE(SortsAndArrangements3) {
    std::vector<int> id({0, 1, 2, 3, 4, 5, 6});
    std::vector<int> v = id;
    std::vector<int> permA({0, 2, 1, 3, 4, 5, 6});
    std::vector<int> permB({0, 2, 1, 4, 5, 6, 3});
    std::vector<int> permC({1, 2, 0, 3, 4, 5, 6});

    sort_like(v, id);
    BOOST_CHECK(v == id);
    sort_like(v, permA);
    BOOST_CHECK(v == permA);
    sort_like(v, permA);
    BOOST_CHECK(v == id);

    sort_like(v, permB);
    BOOST_CHECK(v != permB);
    sort_like(v, permB);
    BOOST_CHECK(v != id);
    sort_like(v, permB);
    BOOST_CHECK(v == permB);
    sort_like(v, permB);
    BOOST_CHECK(v == id);

    sort_like(v, permC);
    BOOST_CHECK(v != permC);
    sort_like(v, permC);
    BOOST_CHECK(v == permC);
    sort_like(v, permC);
    BOOST_CHECK(v == id);
}
