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
#include "osp/auxiliary/datastructures/bit_mask.hpp"

#include <boost/test/unit_test.hpp>

using namespace osp;

BOOST_AUTO_TEST_CASE(BitMaskTest1) {
    const std::size_t numFlags = 4U;
    BitMask mask(numFlags);

    for (unsigned i = 0; i < 25U; ++i) {
        for (std::size_t j = 0; j < numFlags; ++j) {
            BOOST_CHECK_EQUAL(mask.mask_[j], bool(i & (1U << j)));
        }
        ++mask;
    }
}

BOOST_AUTO_TEST_CASE(BitMaskTest2) {
    const std::size_t numFlags = 6U;
    BitMask mask(numFlags);

    for (unsigned i = 0; i < 256U; ++i) {
        BitMask tmp = mask;
        BitMask post = mask++;
        for (std::size_t j = 0; j < numFlags; ++j) {
            BOOST_CHECK_EQUAL(tmp.mask_[j], post.mask_[j]);
        }
    }
}

BOOST_AUTO_TEST_CASE(BitMaskTest3) {
    const std::size_t numFlags = 5U;
    BitMask mask(numFlags);

    for (unsigned i = 0; i < 256U; ++i) {
        BitMask tmp = mask++;
        ++tmp;
        for (std::size_t j = 0; j < numFlags; ++j) {
            BOOST_CHECK_EQUAL(tmp.mask_[j], mask.mask_[j]);
        }
    }
}
