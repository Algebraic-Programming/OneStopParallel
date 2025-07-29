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
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>

#include "osp/auxiliary/permute.hpp"

namespace osp {

BOOST_AUTO_TEST_CASE(In_Place_Permutation_random) {
    std::vector<unsigned> vec(20);
    std::iota(vec.begin(), vec.end(), 0);
    std::vector<unsigned> sol(vec);

    for (unsigned i = 0; i < 5U; ++i) {
        std::random_shuffle(vec.begin(), vec.end());
        std::vector<unsigned> perm(vec);

        permute_inplace(vec, perm);
        for (std::size_t j = 0; j < sol.size(); ++j) {
            BOOST_CHECK_EQUAL(vec[j], sol[j]);
            BOOST_CHECK_EQUAL(perm[j], sol[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(In_Place_Permutation_char) {
    std::vector<char> vec({'a', 'b', 'c', 'd', 'e', 'f', 'g'});
    std::vector<std::size_t> perm({4,0,1,2,3,6,5});
    std::vector<char> sol({'b', 'c', 'd', 'e', 'a', 'g', 'f'});
    std::vector<std::size_t> perm_sol(perm.size());
    std::iota(perm_sol.begin(), perm_sol.end(), 0);

    permute_inplace(vec, perm);
    for (std::size_t j = 0; j < sol.size(); ++j) {
        BOOST_CHECK_EQUAL(vec[j], sol[j]);
        BOOST_CHECK_EQUAL(perm[j], perm_sol[j]);
    }
}


BOOST_AUTO_TEST_CASE(In_Place_Inverse_Permutation_random) {
    std::vector<unsigned> vec(20);
    std::iota(vec.begin(), vec.end(), 0);
    std::vector<unsigned> sol(vec);

    for (unsigned i = 0; i < 5U; ++i) {
        std::random_shuffle(vec.begin(), vec.end());

        std::vector<unsigned> inv_perm(vec.size());
        for (unsigned j = 0; j < vec.size(); ++j) {
            inv_perm[vec[j]] = j;
        }

        inverse_permute_inplace(vec, inv_perm);
        for (std::size_t j = 0; j < sol.size(); ++j) {
            BOOST_CHECK_EQUAL(vec[j], sol[j]);
            BOOST_CHECK_EQUAL(inv_perm[j], sol[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(In_Place_Inverse_Permutation_char) {
    std::vector<char> vec({'a', 'b', 'c', 'd', 'e', 'f', 'g'});
    std::vector<std::size_t> perm({4,0,1,2,3,6,5});
    std::vector<char> sol({'e', 'a', 'b', 'c', 'd', 'g', 'f'});
    std::vector<std::size_t> perm_sol(perm.size());
    std::iota(perm_sol.begin(), perm_sol.end(), 0);

    inverse_permute_inplace(vec, perm);
    for (std::size_t j = 0; j < sol.size(); ++j) {
        BOOST_CHECK_EQUAL(vec[j], sol[j]);
        BOOST_CHECK_EQUAL(perm[j], perm_sol[j]);
    }
}




} // namespace osp