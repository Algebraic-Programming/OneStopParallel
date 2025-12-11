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

#define BOOST_TEST_MODULE Divisor
#include "osp/auxiliary/math/divisors.hpp"

#include <boost/test/unit_test.hpp>

using namespace osp;

BOOST_AUTO_TEST_CASE(IntegerSqrt) {
    for (std::size_t root = 1U; root < 200U; ++root) {
        for (std::size_t num = root * root; num < (root + 1U) * (root + 1U); ++num) {
            BOOST_CHECK_EQUAL(intSqrtFloor(num), root);
        }
    }

    for (int root = 1; root < 300; ++root) {
        for (int num = root * root; num < (root + 1) * (root + 1); ++num) { BOOST_CHECK_EQUAL(intSqrtFloor(num), root); }
    }
}

BOOST_AUTO_TEST_CASE(Divisors) {
    for (std::size_t num = 1U; num < 1000U; ++num) {
        const std::vector<std::size_t> divs = divisorsList(num);
        for (const std::size_t &div : divs) {
            std::cout << div << ", ";
            BOOST_CHECK_EQUAL(num % div, 0U);
        }
        std::cout << "\n";

        auto it = divs.begin();
        for (std::size_t i = 1U; i <= num; ++i) {
            if (num % i == 0) {
                BOOST_CHECK(it != divs.end());
                BOOST_CHECK_EQUAL(i, *it);
                ++it;
            }
        }
        BOOST_CHECK(it == divs.end());
    }
}
