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
#include "osp/auxiliary/sptrsv_simulator/WeakBarriers/flat_checkpoint_counter_barrier.hpp"
#include "osp/auxiliary/sptrsv_simulator/WeakBarriers/flat_checkpoint_counter_barrier_cached.hpp"

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

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
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
    constexpr std::size_t numBarriers = 16U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    std::array<FlatBarrier, 3U> barrier{FlatBarrier{numThreads}, FlatBarrier{numThreads}, FlatBarrier{numThreads}};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
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

BOOST_AUTO_TEST_CASE(TestFlatBarrier_SSP_2Threads) {
    constexpr std::size_t numThreads = 2U;
    constexpr std::size_t numBarriers = 1024U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    constexpr std::size_t numSync = 4U;
    std::array<FlatBarrier, numSync> barrier{
        FlatBarrier{numThreads}, FlatBarrier{numThreads}, FlatBarrier{numThreads}, FlatBarrier{numThreads}};

    for (std::size_t threadId = 0U; threadId < numThreads; ++threadId) {
        barrier[1U].Arrive(threadId);
        barrier[2U].Arrive(threadId);
    }

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            barrier[(cntr - 2U + numSync) % numSync].Wait(threadId);
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(threadId);
            }
            barrier[cntr % numSync].Arrive(threadId);
        }
    };

    for (std::size_t threadId = 0U; threadId < numThreads; ++threadId) {
        threads[threadId] = std::thread(threadWork, threadId);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    BOOST_CHECK_EQUAL(ans.size(), numThreads * numBarriers);

    std::vector<std::size_t> cntrs(numThreads, 0);
    for (const std::size_t work : ans) {
        const std::size_t current = ++cntrs[work];
        for (const std::size_t cntr : cntrs) {
            BOOST_CHECK_GE(cntr, std::max(current, static_cast<std::size_t>(2U)) - 2U);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestFlatBarrier_SSP_128Threads) {
    constexpr std::size_t numThreads = 128U;
    constexpr std::size_t numBarriers = 16U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    constexpr std::size_t numSync = 4U;
    std::array<FlatBarrier, numSync> barrier{
        FlatBarrier{numThreads}, FlatBarrier{numThreads}, FlatBarrier{numThreads}, FlatBarrier{numThreads}};

    for (std::size_t threadId = 0U; threadId < numThreads; ++threadId) {
        barrier[1U].Arrive(threadId);
        barrier[2U].Arrive(threadId);
    }

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            barrier[(cntr - 2U + numSync) % numSync].Wait(threadId);
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(threadId);
            }
            barrier[cntr % numSync].Arrive(threadId);
        }
    };

    for (std::size_t threadId = 0U; threadId < numThreads; ++threadId) {
        threads[threadId] = std::thread(threadWork, threadId);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    BOOST_CHECK_EQUAL(ans.size(), numThreads * numBarriers);

    std::vector<std::size_t> cntrs(numThreads, 0);
    for (const std::size_t work : ans) {
        const std::size_t current = ++cntrs[work];
        for (const std::size_t cntr : cntrs) {
            BOOST_CHECK_GE(cntr, std::max(current, static_cast<std::size_t>(2U)) - 2U);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestAlignedAtomicCounter) {
    BOOST_CHECK_EQUAL(sizeof(AlignedAtomicCounter), 64U);
    BOOST_CHECK_EQUAL(alignof(AlignedAtomicCounter), 64U);
}


BOOST_AUTO_TEST_CASE(TestFlatCheckpointCounterBarrier_2Threads) {
    constexpr std::size_t numThreads = 2U;
    constexpr std::size_t numBarriers = 1024U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    FlatCheckpointCounterBarrier barrier{numThreads};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(cntr);
            }
            barrier.Arrive(threadId);
            barrier.Wait(threadId, 0U);
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

BOOST_AUTO_TEST_CASE(TestFlatCheckpointCounterBarrier_128Threads) {
    constexpr std::size_t numThreads = 128U;
    constexpr std::size_t numBarriers = 16U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    FlatCheckpointCounterBarrier barrier{numThreads};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(cntr);
            }
            barrier.Arrive(threadId);
            barrier.Wait(threadId, 0U);
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

BOOST_AUTO_TEST_CASE(TestFlatCheckpointCounterBarrier_SSP_2Threads) {
    constexpr std::size_t numThreads = 2U;
    constexpr std::size_t numBarriers = 1024U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    FlatCheckpointCounterBarrier barrier{numThreads};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            barrier.Wait(threadId, 1U);
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(threadId);
            }
            barrier.Arrive(threadId);
        }
    };

    for (std::size_t threadId = 0U; threadId < numThreads; ++threadId) {
        threads[threadId] = std::thread(threadWork, threadId);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    BOOST_CHECK_EQUAL(ans.size(), numThreads * numBarriers);

    std::vector<std::size_t> cntrs(numThreads, 0);
    for (const std::size_t work : ans) {
        const std::size_t current = ++cntrs[work];
        for (const std::size_t cntr : cntrs) {
            BOOST_CHECK_GE(cntr, std::max(current, static_cast<std::size_t>(2U)) - 2U);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestFlatCheckpointCounterBarrier_SSP_128Threads) {
    constexpr std::size_t numThreads = 128U;
    constexpr std::size_t numBarriers = 16U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    FlatCheckpointCounterBarrier barrier{numThreads};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            barrier.Wait(threadId, 1U);
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(threadId);
            }
            barrier.Arrive(threadId);
        }
    };

    for (std::size_t threadId = 0U; threadId < numThreads; ++threadId) {
        threads[threadId] = std::thread(threadWork, threadId);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    BOOST_CHECK_EQUAL(ans.size(), numThreads * numBarriers);

    std::vector<std::size_t> cntrs(numThreads, 0);
    for (const std::size_t work : ans) {
        const std::size_t current = ++cntrs[work];
        for (const std::size_t cntr : cntrs) {
            BOOST_CHECK_GE(cntr, std::max(current, static_cast<std::size_t>(2U)) - 2U);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestFlatCheckpointCounterBarrierCached_2Threads) {
    constexpr std::size_t numThreads = 2U;
    constexpr std::size_t numBarriers = 1024U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    FlatCheckpointCounterBarrierCached barrier{numThreads};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(cntr);
            }
            barrier.Arrive(threadId);
            barrier.Wait(threadId, 0U);
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

BOOST_AUTO_TEST_CASE(TestFlatCheckpointCounterBarrierCached_128Threads) {
    constexpr std::size_t numThreads = 128U;
    constexpr std::size_t numBarriers = 16U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    FlatCheckpointCounterBarrierCached barrier{numThreads};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(cntr);
            }
            barrier.Arrive(threadId);
            barrier.Wait(threadId, 0U);
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

BOOST_AUTO_TEST_CASE(TestFlatCheckpointCounterBarrierCached_SSP_2Threads) {
    constexpr std::size_t numThreads = 2U;
    constexpr std::size_t numBarriers = 1024U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    FlatCheckpointCounterBarrierCached barrier{numThreads};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            barrier.Wait(threadId, 1U);
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(threadId);
            }
            barrier.Arrive(threadId);
        }
    };

    for (std::size_t threadId = 0U; threadId < numThreads; ++threadId) {
        threads[threadId] = std::thread(threadWork, threadId);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    BOOST_CHECK_EQUAL(ans.size(), numThreads * numBarriers);

    std::vector<std::size_t> cntrs(numThreads, 0);
    for (const std::size_t work : ans) {
        const std::size_t current = ++cntrs[work];
        for (const std::size_t cntr : cntrs) {
            BOOST_CHECK_GE(cntr, std::max(current, static_cast<std::size_t>(2U)) - 2U);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestFlatCheckpointCounterBarrierCached_SSP_128Threads) {
    constexpr std::size_t numThreads = 128U;
    constexpr std::size_t numBarriers = 16U;

    std::vector<std::size_t> ans;
    ans.reserve(numThreads * numBarriers);

    std::mutex ans_mutex;

    FlatCheckpointCounterBarrierCached barrier{numThreads};

    std::vector<std::thread> threads(numThreads);

    auto threadWork = [&ans, &ans_mutex, numBarriers, &barrier](const std::size_t threadId) {
        for (std::size_t cntr = 0U; cntr < numBarriers; ++cntr) {
            barrier.Wait(threadId, 1U);
            {
                std::lock_guard<std::mutex> lock(ans_mutex);
                ans.emplace_back(threadId);
            }
            barrier.Arrive(threadId);
        }
    };

    for (std::size_t threadId = 0U; threadId < numThreads; ++threadId) {
        threads[threadId] = std::thread(threadWork, threadId);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    BOOST_CHECK_EQUAL(ans.size(), numThreads * numBarriers);

    std::vector<std::size_t> cntrs(numThreads, 0);
    for (const std::size_t work : ans) {
        const std::size_t current = ++cntrs[work];
        for (const std::size_t cntr : cntrs) {
            BOOST_CHECK_GE(cntr, std::max(current, static_cast<std::size_t>(2U)) - 2U);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestVectorPadding) {
    for (std::size_t i = 0U; i < 257; ++i) {
        const std::size_t numCacheLines = (i * sizeof(std::size_t) + CACHE_LINE_SIZE - 1U) / CACHE_LINE_SIZE;
        const std::size_t ans = RoundUpToCacheLine(i);

        BOOST_CHECK_LE(numCacheLines * CACHE_LINE_SIZE, ans * sizeof(std::size_t));
        if (ans > 0U) {
            BOOST_CHECK_GT(numCacheLines * CACHE_LINE_SIZE, (ans - 1U) * sizeof(std::size_t));
        }
    }
}