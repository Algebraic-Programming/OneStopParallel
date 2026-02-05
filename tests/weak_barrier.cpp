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

@author Toni Boehnlein, Christos Matzoros, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#define BOOST_TEST_MODULE WeakBarrierTests

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "osp/auxiliary/sptrsv_simulator/WeakBarriers/flat_barrier.hpp"

using namespace osp;

BOOST_AUTO_TEST_CASE(TestAlignedAtomicFlag) {
    BOOST_CHECK_EQUAL(sizeof(AlignedAtomicFlag), 64U);
    BOOST_CHECK_EQUAL(alignof(AlignedAtomicFlag), 64U);
}

BOOST_AUTO_TEST_CASE(TestFlatBarrier_2Threads) {
    constexpr std::size_t numThreads = 2U;
    constexpr std::size_t numBarriers = 1024U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    std::array<FlatBarrier, 2U> barrier{FlatBarrier{numThreads}, FlatBarrier{numThreads}};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(cntr);
            }
            barrier[0].Arrive(threadId);
            barrier[0].Wait(threadId);
            barrier[1].Arrive(threadId);
            barrier[1].Wait(threadId);
        }
    };

    for (std::size_t threadId = 0U; threadId < numThreads; ++threadId) {
        threads[threadId] = std::thread(threadWork, threadId);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    BOOST_CHECK_EQUAL(ans.size(), numThreads * numBarriers);
    for (std::size_t ind = 0U; ind < ans.size(); ++ind) {
        BOOST_CHECK_EQUAL(ans[ind], ind / numThreads);
    }
}

BOOST_AUTO_TEST_CASE(TestFlatBarrier_128Threads) {
    constexpr std::size_t numThreads = 128U;
    constexpr std::size_t numBarriers = 8U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    std::array<FlatBarrier, 3U> barrier{FlatBarrier{numThreads}, FlatBarrier{numThreads}, FlatBarrier{numThreads}};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(cntr);
            }
            barrier[0].Arrive(threadId);
            barrier[0].Wait(threadId);
            barrier[1].Arrive(threadId);
            barrier[1].Wait(threadId);
        }
    };

    for (std::size_t threadId = 0U; threadId < numThreads; ++threadId) {
        threads[threadId] = std::thread(threadWork, threadId);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    BOOST_CHECK_EQUAL(ans.size(), numThreads * numBarriers);
    for (std::size_t ind = 0U; ind < ans.size(); ++ind) {
        BOOST_CHECK_EQUAL(ans[ind], ind / numThreads);
    }
}
