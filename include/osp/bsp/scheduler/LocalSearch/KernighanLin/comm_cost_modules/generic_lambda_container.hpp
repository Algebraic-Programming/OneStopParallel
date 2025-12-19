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

#include <algorithm>
#include <cassert>
#include <vector>

namespace osp {

template <typename T>
struct DefaultHasEntry {
    static inline bool HasEntry(const T &val) { return val != 0; }
};

template <typename T>
struct DefaultHasEntry<std::vector<T>> {
    static inline bool HasEntry(const std::vector<T> &val) { return !val.empty(); }
};

/**
 * @brief Generic container for tracking child processor assignments in a BSP schedule using vectors.
 *
 * This structure tracks information about children assigned to each processor.
 * It uses a 2D vector for dense data.
 */
template <typename VertexIdxT, typename ValueType = unsigned, typename HasEntry = DefaultHasEntry<ValueType>>
struct GenericLambdaVectorContainer {
    /**
     * @brief Range adapter for iterating over non-zero/non-empty processor entries.
     */
    class LambdaVectorRange {
      private:
        const std::vector<ValueType> &vec_;

      public:
        class LambdaVectorIterator {
            using iterator_category = std::input_iterator_tag;
            using value_type = std::pair<unsigned, ValueType>;
            using difference_type = std::ptrdiff_t;
            using pointer = value_type *;
            using reference = value_type &;

          private:
            const std::vector<ValueType> &vec_;
            unsigned index_;

          public:
            LambdaVectorIterator(const std::vector<ValueType> &vec) : vec_(vec), index_(0) {
                while (index_ < vec_.size() && !HasEntry::HasEntry(vec_[index_])) {
                    ++index_;
                }
            }

            LambdaVectorIterator(const std::vector<ValueType> &vec, unsigned index) : vec_(vec), index_(index) {}

            LambdaVectorIterator &operator++() {
                ++index_;
                while (index_ < vec_.size() && !HasEntry::HasEntry(vec_[index_])) {
                    ++index_;
                }
                return *this;
            }

            value_type operator*() const { return std::make_pair(index_, vec_[index_]); }

            bool operator==(const LambdaVectorIterator &other) const { return index_ == other.index_; }

            bool operator!=(const LambdaVectorIterator &other) const { return !(*this == other); }
        };

        LambdaVectorRange(const std::vector<ValueType> &vec) : vec_(vec) {}

        LambdaVectorIterator begin() { return LambdaVectorIterator(vec_); }

        LambdaVectorIterator end() { return LambdaVectorIterator(vec_, static_cast<unsigned>(vec_.size())); }
    };

    /// 2D vector: for each node, stores processor assignment info
    std::vector<std::vector<ValueType>> nodeLambdaVec_;

    /// Number of processors in the system
    unsigned numProcs_ = 0;

    inline void Initialize(const VertexIdxT numVertices, const unsigned numProcs) {
        nodeLambdaVec_.assign(numVertices, std::vector<ValueType>(numProcs));
        numProcs_ = numProcs;
    }

    inline void ResetNode(const VertexIdxT node) { nodeLambdaVec_[node].assign(numProcs_, ValueType()); }

    inline void Clear() { nodeLambdaVec_.clear(); }

    inline bool HasProcEntry(const VertexIdxT node, const unsigned proc) const {
        return HasEntry::HasEntry(nodeLambdaVec_[node][proc]);
    }

    inline ValueType &GetProcEntry(const VertexIdxT node, const unsigned proc) { return nodeLambdaVec_[node][proc]; }

    inline ValueType GetProcEntry(const VertexIdxT node, const unsigned proc) const { return nodeLambdaVec_[node][proc]; }

    inline auto IterateProcEntries(const VertexIdxT node) { return LambdaVectorRange(nodeLambdaVec_[node]); }
};

}    // namespace osp
