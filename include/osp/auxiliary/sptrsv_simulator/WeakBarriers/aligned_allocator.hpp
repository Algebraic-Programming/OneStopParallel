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

#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>

namespace osp {

template <class T, std::size_t alignment = alignof(T)>
struct AlignedAllocator {
    static_assert(alignment > 0U, "Alignment must be a positive integer.");
    static_assert((alignment & (alignment - 1U)) == 0U, "Alignment must be a power of two.");
    static_assert(alignment % alignof(T) == 0U, "Alignment must be a multiple of the alignment of the type.");

    using value_type = T;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, alignment> &) noexcept {}

    inline T *allocate(std::size_t size);
    inline void deallocate(T *p, [[maybe_unused]] std::size_t size);

    template <typename U, typename... Args>
    inline void construct(U *p, Args &&...args) {
        new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
    }

    template <typename U>
    inline void destroy(U *p) noexcept {
        p->~U();
    }
};

template <class T, std::size_t alignment>
inline T *AlignedAllocator<T, alignment>::allocate(std::size_t size) {
    std::size_t allocationSize = ((size * sizeof(T) + alignment - 1U) / alignment) * alignment;
    return reinterpret_cast<T *>(std::aligned_alloc(alignment, allocationSize));
}

template <class T, std::size_t alignment>
inline void AlignedAllocator<T, alignment>::deallocate(T *p, [[maybe_unused]] std::size_t size) {
    std::free(p);
}

template <class T, std::size_t T_alignment, class U, std::size_t U_alignment>
constexpr bool operator==(const AlignedAllocator<T, T_alignment> &, const AlignedAllocator<U, U_alignment> &) noexcept {
    return (T_alignment == U_alignment);
}

template <class T, std::size_t T_alignment, class U, std::size_t U_alignment>
constexpr bool operator!=(const AlignedAllocator<T, T_alignment> &, const AlignedAllocator<U, U_alignment> &) noexcept {
    return (T_alignment != U_alignment);
}

}    // end namespace osp
