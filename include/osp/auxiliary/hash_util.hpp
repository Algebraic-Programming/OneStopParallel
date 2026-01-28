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

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <vector>

namespace osp {

template <typename VertexType, std::size_t defautlVal = 11U>
struct UniformNodeHashFunc {
    using ResultType = std::size_t;

    constexpr ResultType operator()(const VertexType &) { return defautlVal; }
};

template <typename VertexType>
struct VectorNodeHashFunc {
    const std::vector<std::size_t> &nodeHashes_;

    VectorNodeHashFunc(const std::vector<std::size_t> &nodeHashes) : nodeHashes_(nodeHashes) {}

    using ResultType = std::size_t;

    ResultType operator()(const VertexType &v) const { return nodeHashes_[v]; }
};

template <class T>
void HashCombine(std::size_t &seed, const T &v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const {
        std::size_t h1 = std::hash<T1>{}(p.first);

        const std::size_t h2 = std::hash<T2>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        HashCombine(h1, h2);
        return h1;
    }
};

}    // namespace osp
