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

namespace osp {

// Portable cpu_relax definition
#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
inline void cpu_relax() { _mm_pause(); }
#elif defined(__aarch64__)
inline void cpu_relax() { asm volatile("isb" ::: "memory"); }
#elif defined(__arm__)
inline void cpu_relax() { asm volatile("yield" ::: "memory"); }
#else
#include <atomic>
inline void cpu_relax() { std::atomic_signal_fence(std::memory_order_acquire); }
#endif

}    // end namespace osp
