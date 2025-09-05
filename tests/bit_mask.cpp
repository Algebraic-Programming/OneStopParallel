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

#define BOOST_TEST_MODULE BitMasks
#include <boost/test/unit_test.hpp>

#include "osp/auxiliary/datastructures/bit_mask.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(BitMaskTest_1) {
    const std::size_t num_flags = 4U;
    BitMask mask(num_flags);

    for (unsigned i = 0; i < 25U; ++i) {
        for (std::size_t j = 0; j < num_flags; ++j) {
            BOOST_CHECK_EQUAL( mask.mask[j], bool(i & (1U << j)) );
        }
        ++mask;
    }
}


BOOST_AUTO_TEST_CASE(BitMaskTest_2) {
    const std::size_t num_flags = 6U;
    BitMask mask(num_flags);

    for (unsigned i = 0; i < 256U; ++i) {
        BitMask tmp = mask;
        BitMask post = mask++;
        for (std::size_t j = 0; j < num_flags; ++j) {
            BOOST_CHECK_EQUAL( tmp.mask[j], post.mask[j] );
        }
    }
}

BOOST_AUTO_TEST_CASE(BitMaskTest_3) {
    const std::size_t num_flags = 5U;
    BitMask mask(num_flags);

    for (unsigned i = 0; i < 256U; ++i) {
        BitMask tmp = mask++;
        ++tmp;
        for (std::size_t j = 0; j < num_flags; ++j) {
            BOOST_CHECK_EQUAL( tmp.mask[j], mask.mask[j] );
        }
    }
}