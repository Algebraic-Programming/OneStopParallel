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

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "osp/auxiliary/sptrsv_simulator/WeakBarriers/cpu_relax.hpp"
#include "osp/config/config.hpp"

namespace osp {

struct alignas(CACHE_LINE_SIZE) AlignedAtomicCounter {
    std::atomic<std::size_t> cntr_{0U};
    int8_t pad[CACHE_LINE_SIZE - sizeof(std::atomic<std::size_t>)];

    static_assert(std::atomic<std::size_t>::is_always_lock_free);
    static_assert(sizeof(int8_t) == 1U);
};

class FlatCheckpointCounterBarrier {
  private:
    std::vector<AlignedAtomicCounter> cntrs_;
    mutable std::vector<std::vector<std::size_t>> cachedCntrs_;

  public:
    FlatCheckpointCounterBarrier(std::size_t numThreads)
        : cntrs_(std::vector<AlignedAtomicCounter>(numThreads)),
          cachedCntrs_(std::vector<std::vector<std::size_t>>(numThreads, std::vector<std::size_t>(numThreads, 0U))) {};

    inline void Arrive(const std::size_t threadId);
    inline void Wait(const std::size_t threadId, const std::size_t diff) const;

    FlatCheckpointCounterBarrier() = delete;
    FlatCheckpointCounterBarrier(const FlatCheckpointCounterBarrier &) = delete;
    FlatCheckpointCounterBarrier(FlatCheckpointCounterBarrier &&) = delete;
    FlatCheckpointCounterBarrier &operator=(const FlatCheckpointCounterBarrier &) = delete;
    FlatCheckpointCounterBarrier &operator=(FlatCheckpointCounterBarrier &&) = delete;
    ~FlatCheckpointCounterBarrier() = default;
};

inline void FlatCheckpointCounterBarrier::Arrive(const std::size_t threadId) {
    const std::size_t curr = cntrs_[threadId].cntr_.fetch_add(1U, std::memory_order_release) + 1U;
    cachedCntrs_[threadId][threadId] = curr;
}

inline void FlatCheckpointCounterBarrier::Wait(const std::size_t threadId, const std::size_t diff) const {
    std::vector<std::size_t> &localCachedCntrs = cachedCntrs_[threadId];

    const std::size_t minVal = std::max(localCachedCntrs[threadId], diff) - diff;

    for (std::size_t ind = 0U; ind < cntrs_.size(); ++ind) {
        std::size_t loopCntr = 0U;
        while ((localCachedCntrs[ind] < minVal)
               && ((localCachedCntrs[ind] = cntrs_[ind].cntr_.load(std::memory_order_acquire)) < minVal)) {
            ++loopCntr;
            if (loopCntr % 128U == 0U) {
                cpu_relax();
            }
        }
    }
}

}    // end namespace osp
