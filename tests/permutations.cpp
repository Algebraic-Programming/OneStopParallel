/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos K. Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#define BOOST_TEST_MODULE permutations
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <numeric>
#include <random>

#include "osp/auxiliary/permute.hpp"

namespace osp {

BOOST_AUTO_TEST_CASE(InPlacePermutationRandom) {
    std::vector<unsigned> vec(20);
    std::iota(vec.begin(), vec.end(), 0);
    std::vector<unsigned> sol(vec);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (unsigned i = 0; i < 5U; ++i) {
        std::shuffle(vec.begin(), vec.end(), gen);
        std::vector<unsigned> perm(vec);

        PermuteInplace(vec, perm);
        for (std::size_t j = 0; j < sol.size(); ++j) {
            BOOST_CHECK_EQUAL(vec[j], sol[j]);
            BOOST_CHECK_EQUAL(perm[j], sol[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(InPlacePermutationChar) {
    std::vector<char> vec({'a', 'b', 'c', 'd', 'e', 'f', 'g'});
    std::vector<std::size_t> perm({4, 0, 1, 2, 3, 6, 5});
    std::vector<char> sol({'b', 'c', 'd', 'e', 'a', 'g', 'f'});
    std::vector<std::size_t> permSol(perm.size());
    std::iota(permSol.begin(), permSol.end(), 0);

    PermuteInplace(vec, perm);
    for (std::size_t j = 0; j < sol.size(); ++j) {
        BOOST_CHECK_EQUAL(vec[j], sol[j]);
        BOOST_CHECK_EQUAL(perm[j], permSol[j]);
    }
}

BOOST_AUTO_TEST_CASE(InPlaceInversePermutationRandom) {
    std::vector<unsigned> vec(20);
    std::iota(vec.begin(), vec.end(), 0);
    std::vector<unsigned> sol(vec);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (unsigned i = 0; i < 5U; ++i) {
        std::shuffle(vec.begin(), vec.end(), gen);

        std::vector<unsigned> invPerm(vec.size());
        for (unsigned j = 0; j < vec.size(); ++j) {
            invPerm[vec[j]] = j;
        }

        InversePermuteInplace(vec, invPerm);
        for (std::size_t j = 0; j < sol.size(); ++j) {
            BOOST_CHECK_EQUAL(vec[j], sol[j]);
            BOOST_CHECK_EQUAL(invPerm[j], sol[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(InPlaceInversePermutationChar) {
    std::vector<char> vec({'a', 'b', 'c', 'd', 'e', 'f', 'g'});
    std::vector<std::size_t> perm({4, 0, 1, 2, 3, 6, 5});
    std::vector<char> sol({'e', 'a', 'b', 'c', 'd', 'g', 'f'});
    std::vector<std::size_t> permSol(perm.size());
    std::iota(permSol.begin(), permSol.end(), 0);

    InversePermuteInplace(vec, perm);
    for (std::size_t j = 0; j < sol.size(); ++j) {
        BOOST_CHECK_EQUAL(vec[j], sol[j]);
        BOOST_CHECK_EQUAL(perm[j], permSol[j]);
    }
}

}    // namespace osp
