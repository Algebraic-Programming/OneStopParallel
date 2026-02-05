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

#include "osp/config/config.hpp"

namespace osp {

// Portable cpu_relax definition
#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>

inline void cpu_relax() { _mm_pause(); }
#elif defined(__aarch64__)
inline void cpu_relax() { asm volatile("yield" ::: "memory"); }
#else
inline void cpu_relax() { std::this_thread::yield(); }
#endif

struct alignas(CACHE_LINE_SIZE) AlignedAtomicFlag {
    std::atomic<bool> flag_;
    int8_t pad[CACHE_LINE_SIZE - sizeof(std::atomic<bool>)];

    static_assert(std::atomic<bool>::is_always_lock_free);
    static_assert(sizeof(int8_t) == 1U);
};

/**
 * @brief A weak synchronisation barrier which can be reused.
 * Instatiate with number of threads. Each thread should call "Arrive" with its thread id to indicate that its work has been
 * completed. Each thread can then call "Wait" to wait till all other threads have completed their work.
 *
 * The barrier can be reset and reused after calling "Reset" for each thread.
 *
 * WARNING: The reset is NOT synchronised, thus a second FlatBarrier is required to synchronise the reset of the barrier. That is
 * do NOT call "Reset" immediately after "Wait" as this could cause other threads not to see that the work has been completed.
 *
 */
class FlatBarrier {
  private:
    std::vector<AlignedAtomicFlag> flags_;

  public:
    FlatBarrier(std::size_t numThreads) : flags_(std::vector<AlignedAtomicFlag>(numThreads)) {};

    inline void Arrive(std::size_t threadId);
    inline void Wait() const;
    inline void Reset(std::size_t threadId);

    FlatBarrier() = delete;
    FlatBarrier(const FlatBarrier &) = delete;
    FlatBarrier(FlatBarrier &&) = delete;
    FlatBarrier &operator=(const FlatBarrier &) = delete;
    FlatBarrier &operator=(FlatBarrier &&) = delete;
    ~FlatBarrier() = default;
};

inline void FlatBarrier::Arrive(std::size_t threadId) { flags_[threadId].flag_.store(true, std::memory_order_relaxed); }

inline void FlatBarrier::Wait() const {
    for (const AlignedAtomicFlag &flag : flags_) {
        std::size_t cntr = 0U;
        while (not flag.flag_.load(std::memory_order_relaxed)) {
            ++cntr;
            if (cntr % 256U == 0U) {
                cpu_relax();
            }
        }
    }
}

inline void FlatBarrier::Reset(std::size_t threadId) { flags_[threadId].flag_.store(false, std::memory_order_relaxed); }

}    // end namespace osp
