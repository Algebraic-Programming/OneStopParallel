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

#include <cassert>
#include <unordered_map>
#include <vector>

namespace osp {

/**
 * @brief Container for tracking child processor assignments in a BSP schedule using hash maps.
 *
 * This structure tracks how many children a node has that are assigned to each processor.
 * It uses unordered_map for sparse data representation.
 *
 * For each node, the map stores the count of children assigned to each processor, which is
 * important for computing communication costs in BSP scheduling.
 */
template <typename vertex_idx_t>
struct lambda_map_container {
    /// Vector of maps: for each node, maps processor ID to assignment count
    std::vector<std::unordered_map<unsigned, unsigned>> node_lambda_map;

    /**
     * @brief Initialize the container for a given number of vertices.
     * @param num_vertices Number of nodes in the schedule
     * @param (unused) Number of processors (not needed for map-based implementation)
     */
    inline void initialize(const vertex_idx_t num_vertices, const unsigned) { node_lambda_map.resize(num_vertices); }

    /**
     * @brief Reset all processor assignments for a specific node.
     * @param node Node index to reset
     */
    inline void reset_node(const vertex_idx_t node) { node_lambda_map[node].clear(); }

    /**
     * @brief Clear all data from the container.
     */
    inline void clear() { node_lambda_map.clear(); }

    /**
     * @brief Check if a processor has an entry for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if the processor has at least one assignment to the node
     */
    inline bool has_proc_entry(const vertex_idx_t node, const unsigned proc) const {
        return (node_lambda_map[node].find(proc) != node_lambda_map[node].end());
    }

    /**
     * @brief Check if a processor has no entry for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if the processor has no assignments to the node
     */
    inline bool has_no_proc_entry(const vertex_idx_t node, const unsigned proc) const {
        return (node_lambda_map[node].find(proc) == node_lambda_map[node].end());
    }

    /**
     * @brief Get a reference to the processor count for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return Reference to the count (creates entry if it doesn't exist)
     */
    inline unsigned &get_proc_entry(const vertex_idx_t node, const unsigned proc) { return node_lambda_map[node][proc]; }

    /**
     * @brief Get the processor count for a given node (const version).
     * @param node Node index
     * @param proc Processor ID
     * @return The count value for the processor at the node
     * @pre has_proc_entry(node, proc) must be true
     */
    inline unsigned get_proc_entry(const vertex_idx_t node, const unsigned proc) const {
        assert(has_proc_entry(node, proc));
        return node_lambda_map[node].at(proc);
    }

    /**
     * @brief Get the number of different processors to which a node has children assigned.
     * @param node Node index
     * @return The count of different processors the node is sending to
     */
    inline unsigned get_proc_count(const vertex_idx_t node) const { return static_cast<unsigned>(node_lambda_map[node].size()); }

    /**
     * @brief Increase the processor count for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if this is the first assignment of this processor to the node
     */
    inline bool increase_proc_count(const vertex_idx_t node, const unsigned proc) {
        if (has_proc_entry(node, proc)) {
            node_lambda_map[node][proc]++;
            return false;
        } else {
            node_lambda_map[node][proc] = 1;
            return true;
        }
    }

    /**
     * @brief Decrease the processor count for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if this was the last assignment of this processor to the node
     * @pre has_proc_entry(node, proc) must be true
     */
    inline bool decrease_proc_count(const vertex_idx_t node, const unsigned proc) {
        assert(has_proc_entry(node, proc));
        if (node_lambda_map[node][proc] == 1) {
            node_lambda_map[node].erase(proc);
            return true;
        } else {
            node_lambda_map[node][proc]--;
            return false;
        }
    }

    /**
     * @brief Get an iterable view of all processor entries for a node.
     * @param node Node index
     * @return Reference to the unordered_map of processor assignments for the node
     */
    inline const auto &iterate_proc_entries(const vertex_idx_t node) { return node_lambda_map[node]; }
};

/**
 * @brief Container for tracking child processor assignments in a BSP schedule using vectors.
 *
 * This structure tracks how many children a node has that are assigned to each processor.
 * It uses a 2D vector for dense data, making it efficient when most processors may have
 * children of nodes assigned to them, or when the processor count is relatively small.
 *
 * For each node, the vector stores the count of children assigned to each processor, which is
 * important for computing communication costs in BSP scheduling.
 */
template <typename vertex_idx_t>
struct lambda_vector_container {
    /**
     * @brief Range adapter for iterating over non-zero processor entries.
     *
     * Provides a range-based for loop interface that automatically skips processors
     * with zero assignments.
     */
    class lambda_vector_range {
      private:
        const std::vector<unsigned> &vec_;

      public:
        /**
         * @brief Iterator that skips zero entries in the lambda vector.
         *
         * Implements an input iterator that yields pairs of (processor_id, count)
         * for all processors with non-zero assignment counts.
         */
        class lambda_vector_iterator {
            using iterator_category = std::input_iterator_tag;
            using value_type = std::pair<unsigned, unsigned>;
            using difference_type = std::ptrdiff_t;
            using pointer = value_type *;
            using reference = value_type &;

          private:
            const std::vector<unsigned> &vec_;
            unsigned index_;

          public:
            /**
             * @brief Construct iterator at the beginning, skipping initial zeros.
             * @param vec Reference to the vector to iterate over
             */
            lambda_vector_iterator(const std::vector<unsigned> &vec) : vec_(vec), index_(0) {
                // Advance to the first valid entry
                while (index_ < vec_.size() && vec_[index_] == 0) {
                    ++index_;
                }
            }

            /**
             * @brief Construct iterator at a specific position.
             * @param vec Reference to the vector to iterate over
             * @param index Starting index
             */
            lambda_vector_iterator(const std::vector<unsigned> &vec, unsigned index) : vec_(vec), index_(index) {}

            /**
             * @brief Advance to the next non-zero entry.
             * @return Reference to this iterator
             */
            lambda_vector_iterator &operator++() {
                ++index_;
                while (index_ < vec_.size() && vec_[index_] == 0) {
                    ++index_;
                }
                return *this;
            }

            /**
             * @brief Dereference to get (processor_id, count) pair.
             * @return Pair of processor ID and its count
             */
            value_type operator*() const { return std::make_pair(index_, vec_[index_]); }

            /**
             * @brief Check equality with another iterator.
             * @param other Iterator to compare with
             * @return true if both iterators point to the same position
             */
            bool operator==(const lambda_vector_iterator &other) const { return index_ == other.index_; }

            /**
             * @brief Check inequality with another iterator.
             * @param other Iterator to compare with
             * @return true if iterators point to different positions
             */
            bool operator!=(const lambda_vector_iterator &other) const { return !(*this == other); }
        };

        /**
         * @brief Construct a range from a vector.
         * @param vec Reference to the vector to create range over
         */
        lambda_vector_range(const std::vector<unsigned> &vec) : vec_(vec) {}

        /// Get iterator to the first non-zero entry
        lambda_vector_iterator begin() { return lambda_vector_iterator(vec_); }

        /// Get iterator to the end
        lambda_vector_iterator end() { return lambda_vector_iterator(vec_, static_cast<unsigned>(vec_.size())); }
    };

    /// 2D vector: for each node, stores processor assignment counts
    std::vector<std::vector<unsigned>> node_lambda_vec;

    /// Number of processors in the system
    unsigned num_procs_ = 0;

    /**
     * @brief Initialize the container for a given number of vertices and processors.
     * @param num_vertices Number of nodes in the schedule
     * @param num_procs Number of processors in the system
     */
    inline void initialize(const vertex_idx_t num_vertices, const unsigned num_procs) {
        node_lambda_vec.assign(num_vertices, std::vector<unsigned>(num_procs, 0));
        num_procs_ = num_procs;
    }

    /**
     * @brief Reset all processor assignments for a specific node.
     * @param node Node index to reset
     */
    inline void reset_node(const vertex_idx_t node) { node_lambda_vec[node].assign(num_procs_, 0); }

    /**
     * @brief Clear all data from the container.
     */
    inline void clear() { node_lambda_vec.clear(); }

    /**
     * @brief Check if a processor has an entry for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if the processor has at least one assignment to the node
     */
    inline bool has_proc_entry(const vertex_idx_t node, const unsigned proc) const { return node_lambda_vec[node][proc] > 0; }

    /**
     * @brief Check if a processor has no entry for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if the processor has no assignments to the node
     */
    inline bool has_no_proc_entry(const vertex_idx_t node, const unsigned proc) const { return node_lambda_vec[node][proc] == 0; }

    /**
     * @brief Get a reference to the processor count for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return Reference to the count (allows modification)
     */
    inline unsigned &get_proc_entry(const vertex_idx_t node, const unsigned proc) { return node_lambda_vec[node][proc]; }

    /**
     * @brief Get the processor count for a given node (const version).
     * @param node Node index
     * @param proc Processor ID
     * @return The count value for the processor at the node
     * @pre has_proc_entry(node, proc) must be true
     */
    inline unsigned get_proc_entry(const vertex_idx_t node, const unsigned proc) const {
        assert(has_proc_entry(node, proc));
        return node_lambda_vec[node][proc];
    }

    /**
     * @brief Get the processor count for a given node (alias for compatibility).
     * @param node Node index
     * @param proc Processor ID
     * @return The count value for the processor at the node
     * @pre has_proc_entry(node, proc) must be true
     */
    inline unsigned get_proc_count(const vertex_idx_t node) const {
        unsigned count = 0;
        for (unsigned proc = 0; proc < num_procs_; ++proc) {
            if (node_lambda_vec[node][proc] > 0) {
                ++count;
            }
        }
        return count;
    }

    /**
     * @brief Increase the processor count for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if this is the first assignment of this processor to the node
     */
    inline bool increase_proc_count(const vertex_idx_t node, const unsigned proc) {
        node_lambda_vec[node][proc]++;
        return node_lambda_vec[node][proc] == 1;
    }

    /**
     * @brief Decrease the processor count for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if this was the last assignment of this processor to the node
     * @pre has_proc_entry(node, proc) must be true
     */
    inline bool decrease_proc_count(const vertex_idx_t node, const unsigned proc) {
        assert(has_proc_entry(node, proc));
        node_lambda_vec[node][proc]--;
        return node_lambda_vec[node][proc] == 0;
    }

    /**
     * @brief Get an iterable range over all non-zero processor entries for a node.
     * @param node Node index
     * @return Range object that can be used in range-based for loops
     */
    inline auto iterate_proc_entries(const vertex_idx_t node) { return lambda_vector_range(node_lambda_vec[node]); }
};

}    // namespace osp
