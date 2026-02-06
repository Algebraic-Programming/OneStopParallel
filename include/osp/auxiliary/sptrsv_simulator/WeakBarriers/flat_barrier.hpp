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

#include <atomic>
#include <cstdint>

#include "osp/auxiliary/sptrsv_simulator/WeakBarriers/cpu_relax.hpp"
#include "osp/config/config.hpp"

namespace osp {

struct alignas(CACHE_LINE_SIZE) AlignedAtomicFlag {
    std::atomic<bool> flag_{false};
    int8_t pad[CACHE_LINE_SIZE - sizeof(std::atomic<bool>)];

    static_assert(std::atomic<bool>::is_always_lock_free);
    static_assert(sizeof(int8_t) == 1U);
};

/**
 * @brief A weak synchronisation barrier which can be reused.
 * Instatiate with number of threads. Each thread should call "Arrive" with its thread id to indicate that its work has been
 * completed. Each thread can then call "Wait" to wait till all other threads have completed their work.
 *
 * WARNING: The barrier can be reused IF AND ONLY IF another synchronisation, i.e. through a second FlatBarrier, takes place in between
 * the "Wait" and "Arrive".
 *
 * WARNING: A thread calling "Wait" before calling "Arrive" with its thread id is undefined behaviour and can result in a deadlock.
 */
class FlatBarrier {
  private:
    std::vector<AlignedAtomicFlag> flags_;

  public:
    FlatBarrier(std::size_t numThreads) : flags_(std::vector<AlignedAtomicFlag>(numThreads)) {};

    inline void Arrive(const std::size_t threadId);
    inline void Wait(const std::size_t threadId) const;

    FlatBarrier() = delete;
    FlatBarrier(const FlatBarrier &) = delete;
    FlatBarrier(FlatBarrier &&) = delete;
    FlatBarrier &operator=(const FlatBarrier &) = delete;
    FlatBarrier &operator=(FlatBarrier &&) = delete;
    ~FlatBarrier() = default;
};

inline void FlatBarrier::Arrive(const std::size_t threadId) {
    const bool oldVal = flags_[threadId].flag_.load(std::memory_order_relaxed);
    flags_[threadId].flag_.store(!oldVal, std::memory_order_release);
}

inline void FlatBarrier::Wait(const std::size_t threadId) const {
    const bool val = flags_[threadId].flag_.load(std::memory_order_relaxed);
    for (const AlignedAtomicFlag &flag : flags_) {
        std::size_t cntr = 0U;
        while (flag.flag_.load(std::memory_order_acquire) != val) {
            ++cntr;
            if (cntr % 128U == 0U) {
                cpu_relax();
            }
        }
    }
}

}    // end namespace osp
