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
#include <cstddef>
#include <vector>

#include "osp/auxiliary/sptrsv_simulator/WeakBarriers/cpu_relax.hpp"
#include "osp/auxiliary/sptrsv_simulator/WeakBarriers/flat_checkpoint_counter_barrier.hpp"

namespace osp {

class FlatCheckpointCounterBarrierCached {
  private:
    std::vector<AlignedAtomicCounter> cntrs_;
        // Change vs flat_checkpoint_counter_barrier.hpp: flatten 2D cache into 1D array
        // to improve locality and avoid nested vector indirections.
        std::vector<std::size_t> cachedCntrs_;
        // Keep explicit thread count for fast index math instead of cntrs_.size().
        std::size_t numThreads_ = 0U;

    inline std::size_t &Cached(std::size_t row, std::size_t col) {
        // Helper to map (row, col) to flat index.
        return cachedCntrs_[row * numThreads_ + col];
    }

    inline const std::size_t &Cached(std::size_t row, std::size_t col) const {
        // Const helper for the same flat index mapping.
        return cachedCntrs_[row * numThreads_ + col];
    }

  public:
    FlatCheckpointCounterBarrierCached(std::size_t numThreads)
        : cntrs_(std::vector<AlignedAtomicCounter>(numThreads)),
                    // Allocate one contiguous block instead of vector-of-vectors.
                    cachedCntrs_(numThreads * numThreads, 0U),
          numThreads_(numThreads) {}

    inline void Arrive(const std::size_t threadId);
    inline void Wait(const std::size_t threadId, const std::size_t diff) const;

    FlatCheckpointCounterBarrierCached() = delete;
    FlatCheckpointCounterBarrierCached(const FlatCheckpointCounterBarrierCached &) = delete;
    FlatCheckpointCounterBarrierCached(FlatCheckpointCounterBarrierCached &&) = delete;
    FlatCheckpointCounterBarrierCached &operator=(const FlatCheckpointCounterBarrierCached &) = delete;
    FlatCheckpointCounterBarrierCached &operator=(FlatCheckpointCounterBarrierCached &&) = delete;
    ~FlatCheckpointCounterBarrierCached() = default;
};

inline void FlatCheckpointCounterBarrierCached::Arrive(const std::size_t threadId) {
    const std::size_t curr = cntrs_[threadId].cntr_.fetch_add(1U, std::memory_order_release) + 1U;
    // Update cached counter via flat indexing helper.
    Cached(threadId, threadId) = curr;
}

inline void FlatCheckpointCounterBarrierCached::Wait(const std::size_t threadId, const std::size_t diff) const {
    // Compute row base once for flat cache; avoids vector-of-vectors access.
    const std::size_t base = threadId * numThreads_;
    // Cast away const instead of marking cachedCntrs_ mutable in this class.
    std::size_t *localCached = const_cast<std::size_t *>(cachedCntrs_.data() + base);
    const std::size_t localThreadVal = localCached[threadId];
    const std::size_t minVal = std::max(localThreadVal, diff) - diff;
    // Hoist data pointer and use numThreads_ instead of cntrs_.size().
    const AlignedAtomicCounter *cntrs = cntrs_.data();

    for (std::size_t ind = 0U; ind < numThreads_; ++ind) {
        std::size_t loopCntr = 0U;
        while ((localCached[ind] < minVal)
               && ((localCached[ind] = cntrs[ind].cntr_.load(std::memory_order_acquire)) < minVal)) {
            ++loopCntr;
            if (loopCntr % 128U == 0U) {
                cpu_relax();
            }
        }
    }
}

}