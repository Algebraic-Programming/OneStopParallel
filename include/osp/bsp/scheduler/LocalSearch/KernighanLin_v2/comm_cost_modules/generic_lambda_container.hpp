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

template<typename T>
struct DefaultHasEntry {
    static inline bool has_entry(const T &val) { return val != 0; }
};

template<typename T>
struct DefaultHasEntry<std::vector<T>> {
    static inline bool has_entry(const std::vector<T> &val) { return !val.empty(); }
};

/**
 * @brief Generic container for tracking child processor assignments in a BSP schedule using vectors.
 *
 * This structure tracks information about children assigned to each processor.
 * It uses a 2D vector for dense data.
 */
template<typename vertex_idx_t, typename ValueType = unsigned, typename HasEntry = DefaultHasEntry<ValueType>>
struct generic_lambda_vector_container {

    /**
     * @brief Range adapter for iterating over non-zero/non-empty processor entries.
     */
    class lambda_vector_range {
      private:
        const std::vector<ValueType> &vec_;

      public:
        class lambda_vector_iterator {
            using iterator_category = std::input_iterator_tag;
            using value_type = std::pair<unsigned, ValueType>;
            using difference_type = std::ptrdiff_t;
            using pointer = value_type *;
            using reference = value_type &;

          private:
            const std::vector<ValueType> &vec_;
            unsigned index_;

          public:
            lambda_vector_iterator(const std::vector<ValueType> &vec) : vec_(vec), index_(0) {
                while (index_ < vec_.size() && !HasEntry::has_entry(vec_[index_])) {
                    ++index_;
                }
            }

            lambda_vector_iterator(const std::vector<ValueType> &vec, unsigned index) : vec_(vec), index_(index) {}

            lambda_vector_iterator &operator++() {
                ++index_;
                while (index_ < vec_.size() && !HasEntry::has_entry(vec_[index_])) {
                    ++index_;
                }
                return *this;
            }

            value_type operator*() const { return std::make_pair(index_, vec_[index_]); }

            bool operator==(const lambda_vector_iterator &other) const { return index_ == other.index_; }
            bool operator!=(const lambda_vector_iterator &other) const { return !(*this == other); }
        };

        lambda_vector_range(const std::vector<ValueType> &vec) : vec_(vec) {}

        lambda_vector_iterator begin() { return lambda_vector_iterator(vec_); }
        lambda_vector_iterator end() { return lambda_vector_iterator(vec_, static_cast<unsigned>(vec_.size())); }
    };

    /// 2D vector: for each node, stores processor assignment info
    std::vector<std::vector<ValueType>> node_lambda_vec;

    /// Number of processors in the system
    unsigned num_procs_ = 0;

    inline void initialize(const vertex_idx_t num_vertices, const unsigned num_procs) {
        node_lambda_vec.assign(num_vertices, std::vector<ValueType>(num_procs));
        num_procs_ = num_procs;
    }

    inline void reset_node(const vertex_idx_t node) { node_lambda_vec[node].assign(num_procs_, ValueType()); }

    inline void clear() { node_lambda_vec.clear(); }

    inline bool has_proc_entry(const vertex_idx_t node, const unsigned proc) const {
        return HasEntry::has_entry(node_lambda_vec[node][proc]);
    }

    inline ValueType &get_proc_entry(const vertex_idx_t node, const unsigned proc) {
        return node_lambda_vec[node][proc];
    }

    inline ValueType get_proc_entry(const vertex_idx_t node, const unsigned proc) const {
        return node_lambda_vec[node][proc];
    }

    inline auto iterate_proc_entries(const vertex_idx_t node) { return lambda_vector_range(node_lambda_vec[node]); }
};

} // namespace osp
