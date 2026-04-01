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

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#define BOOST_TEST_MODULE AlignedAllocatorTests

#include "osp/auxiliary/sptrsv_simulator/WeakBarriers/aligned_allocator.hpp"

#include <boost/test/unit_test.hpp>
#include <vector>

using namespace osp;

BOOST_AUTO_TEST_CASE(TestAlignedAllocation32) {
    constexpr std::size_t alignment = 32U;

    std::vector<unsigned, AlignedAllocator<unsigned, alignment>> vec(7, 7U);
    BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);

    for (unsigned i = 0U; i < 2048U; ++i) {
        vec.emplace_back(i);
        BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);
    }

    vec.resize(8000U);
    BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);
    vec.resize(5U);
    BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);
}

BOOST_AUTO_TEST_CASE(TestAlignedAllocation16) {
    constexpr std::size_t alignment = 16U;

    std::vector<unsigned, AlignedAllocator<unsigned, alignment>> vec(7, 7U);
    BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);

    for (unsigned i = 0U; i < 2048U; ++i) {
        vec.emplace_back(i);
        BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);
    }

    vec.resize(8000U);
    BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);
    vec.resize(5U);
    BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);
}

BOOST_AUTO_TEST_CASE(TestAlignedAllocation64) {
    constexpr std::size_t alignment = 64U;

    std::vector<char, AlignedAllocator<char, alignment>> vec(7, 7U);
    BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);

    for (unsigned i = 0U; i < 2048U; ++i) {
        vec.emplace_back('a');
        BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);
    }

    vec.resize(8000U);
    BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);
    vec.resize(5U);
    BOOST_CHECK_EQUAL(reinterpret_cast<std::size_t>(static_cast<void *>(vec.data())) % alignment, 0U);
}
