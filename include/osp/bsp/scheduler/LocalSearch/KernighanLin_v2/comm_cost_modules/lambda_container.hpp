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
#include <unordered_map>
#include <cassert>

namespace osp {

struct lambda_map_container {

    std::vector<std::unordered_map<unsigned,unsigned>> node_lambda_map;

    inline void initialize(const size_t num_vertices, const unsigned) { node_lambda_map.resize(num_vertices); }
    inline void reset_node(const size_t node) { node_lambda_map[node].clear(); }
    inline void clear() { node_lambda_map.clear(); }
    inline bool has_proc_entry(const size_t node, const unsigned proc) const { return (node_lambda_map[node].find(proc) != node_lambda_map[node].end()); }
    inline bool has_no_proc_entry(const size_t node, const unsigned proc) const { return (node_lambda_map[node].find(proc) == node_lambda_map[node].end()); }
    inline unsigned & get_proc_entry(const size_t node, const unsigned proc) { return node_lambda_map[node][proc]; }

    inline bool increase_proc_count(const size_t node, const unsigned proc) {
        if (has_proc_entry(node, proc)) {
            node_lambda_map[node][proc]++;
            return false;
        } else {
            node_lambda_map[node][proc] = 1;
            return true;
        }
    }

    inline bool decrease_proc_count(const size_t node, const unsigned proc) {
        assert(has_proc_entry(node, proc));
        if (node_lambda_map[node][proc] == 1) {
            node_lambda_map[node].erase(proc);
            return true;
        } else {
            node_lambda_map[node][proc]--;
            return false;
        }
    }

    inline const auto & iterate_proc_entries(const size_t node) {
        return node_lambda_map[node];
    }
};

struct lambda_vector_container {
   
    class lambda_vector_range {
        private:
            const std::vector<unsigned> & vec_;

        public:
        class lambda_vector_iterator {
        
            using iterator_category = std::input_iterator_tag;
            using value_type = std::pair<unsigned, unsigned>;
            using difference_type = std::ptrdiff_t;
            using pointer = value_type*;
            using reference = value_type&;
        private:
            const std::vector<unsigned>& vec_;
            size_t index_;
        public:

        lambda_vector_iterator(const std::vector<unsigned>& vec) : vec_(vec), index_(0) {
            // Advance to the first valid entry
            while (index_ < vec_.size() && vec_[index_] == 0) {
                ++index_;
            }
        }

        lambda_vector_iterator(const std::vector<unsigned>& vec, size_t index) : vec_(vec), index_(index) {}

        lambda_vector_iterator& operator++() {
                ++index_;
                while (index_ < vec_.size() && vec_[index_] == 0) {
                    ++index_;
                }
                return *this;
            }

            value_type operator*() const {
                return std::make_pair(static_cast<unsigned>(index_), vec_[index_]);
            }

            bool operator==(const lambda_vector_iterator& other) const {
                return index_ == other.index_;
            }

            bool operator!=(const lambda_vector_iterator& other) const {
                return !(*this == other);
            }
        };

        lambda_vector_range(const std::vector<unsigned>& vec) : vec_(vec) {}

        lambda_vector_iterator begin() { return lambda_vector_iterator(vec_); }
        lambda_vector_iterator end() { return lambda_vector_iterator(vec_, vec_.size()); }
    };

    std::vector<std::vector<unsigned>> node_lambda_vec;
    unsigned num_procs_ = 0;

    inline void initialize(const size_t num_vertices, const unsigned num_procs) { 
        node_lambda_vec.assign(num_vertices, {num_procs});
        num_procs_ = num_procs; 
    }

    inline void reset_node(const size_t node) { node_lambda_vec[node].assign(num_procs_, 0); }
    inline void clear() { node_lambda_vec.clear(); }
    inline bool has_proc_entry(const size_t node, const unsigned proc) const { return node_lambda_vec[node][proc] > 0; }
    inline bool has_no_proc_entry(const size_t node, const unsigned proc) const { return node_lambda_vec[node][proc] == 0; }
    inline unsigned & get_proc_entry(const size_t node, const unsigned proc) { return node_lambda_vec[node][proc]; }

    inline unsigned get_proc_entry(const size_t node, const unsigned proc) const {
        assert(has_proc_entry(node, proc));
        return node_lambda_vec[node][proc];
    }

    inline bool increase_proc_count(const size_t node, const unsigned proc) {
        node_lambda_vec[node][proc]++;
        return node_lambda_vec[node][proc] == 1;
    }

    inline bool decrease_proc_count(const size_t node, const unsigned proc) {
        assert(has_proc_entry(node, proc));
        node_lambda_vec[node][proc]--;
        return node_lambda_vec[node][proc] == 0;
    }

    inline auto iterate_proc_entries(const size_t node) {
        return lambda_vector_range(node_lambda_vec[node]);
    }
};

} // namespace osp