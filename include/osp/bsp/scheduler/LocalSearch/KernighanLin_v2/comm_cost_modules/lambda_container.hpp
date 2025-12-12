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
template <typename VertexIdxT>
struct LambdaMapContainer {
    /// Vector of maps: for each node, maps processor ID to assignment count
    std::vector<std::unordered_map<unsigned, unsigned>> nodeLambdaMap_;

    /**
     * @brief Initialize the container for a given number of vertices.
     * @param num_vertices Number of nodes in the schedule
     * @param (unused) Number of processors (not needed for map-based implementation)
     */
    inline void Initialize(const VertexIdxT numVertices, const unsigned) { nodeLambdaMap_.resize(numVertices); }

    /**
     * @brief Reset all processor assignments for a specific node.
     * @param node Node index to reset
     */
    inline void ResetNode(const VertexIdxT node) { nodeLambdaMap_[node].clear(); }

    /**
     * @brief Clear all data from the container.
     */
    inline void Clear() { nodeLambdaMap_.clear(); }

    /**
     * @brief Check if a processor has an entry for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if the processor has at least one assignment to the node
     */
    inline bool HasProcEntry(const VertexIdxT node, const unsigned proc) const {
        return (nodeLambdaMap_[node].find(proc) != nodeLambdaMap_[node].end());
    }

    /**
     * @brief Check if a processor has no entry for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if the processor has no assignments to the node
     */
    inline bool HasNoProcEntry(const VertexIdxT node, const unsigned proc) const {
        return (nodeLambdaMap_[node].find(proc) == nodeLambdaMap_[node].end());
    }

    /**
     * @brief Get a reference to the processor count for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return Reference to the count (creates entry if it doesn't exist)
     */
    inline unsigned &GetProcEntry(const VertexIdxT node, const unsigned proc) { return nodeLambdaMap_[node][proc]; }

    /**
     * @brief Get the processor count for a given node (const version).
     * @param node Node index
     * @param proc Processor ID
     * @return The count value for the processor at the node
     * @pre has_proc_entry(node, proc) must be true
     */
    inline unsigned GetProcEntry(const VertexIdxT node, const unsigned proc) const {
        assert(HasProcEntry(node, proc));
        return nodeLambdaMap_[node].at(proc);
    }

    /**
     * @brief Get the number of different processors to which a node has children assigned.
     * @param node Node index
     * @return The count of different processors the node is sending to
     */
    inline unsigned GetProcCount(const VertexIdxT node) const { return static_cast<unsigned>(nodeLambdaMap_[node].size()); }

    /**
     * @brief Increase the processor count for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if this is the first assignment of this processor to the node
     */
    inline bool IncreaseProcCount(const VertexIdxT node, const unsigned proc) {
        if (HasProcEntry(node, proc)) {
            nodeLambdaMap_[node][proc]++;
            return false;
        } else {
            nodeLambdaMap_[node][proc] = 1;
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
    inline bool DecreaseProcCount(const VertexIdxT node, const unsigned proc) {
        assert(HasProcEntry(node, proc));
        if (nodeLambdaMap_[node][proc] == 1) {
            nodeLambdaMap_[node].erase(proc);
            return true;
        } else {
            nodeLambdaMap_[node][proc]--;
            return false;
        }
    }

    /**
     * @brief Get an iterable view of all processor entries for a node.
     * @param node Node index
     * @return Reference to the unordered_map of processor assignments for the node
     */
    inline const auto &IterateProcEntries(const VertexIdxT node) { return nodeLambdaMap_[node]; }
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
template <typename VertexIdxT>
struct LambdaVectorContainer {
    /**
     * @brief Range adapter for iterating over non-zero processor entries.
     *
     * Provides a range-based for loop interface that automatically skips processors
     * with zero assignments.
     */
    class LambdaVectorRange {
      private:
        const std::vector<unsigned> &vec_;

      public:
        /**
         * @brief Iterator that skips zero entries in the lambda vector.
         *
         * Implements an input iterator that yields pairs of (processor_id, count)
         * for all processors with non-zero assignment counts.
         */
        class LambdaVectorIterator {
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
            LambdaVectorIterator(const std::vector<unsigned> &vec) : vec_(vec), index_(0) {
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
            LambdaVectorIterator(const std::vector<unsigned> &vec, unsigned index) : vec_(vec), index_(index) {}

            /**
             * @brief Advance to the next non-zero entry.
             * @return Reference to this iterator
             */
            LambdaVectorIterator &operator++() {
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
            bool operator==(const LambdaVectorIterator &other) const { return index_ == other.index_; }

            /**
             * @brief Check inequality with another iterator.
             * @param other Iterator to compare with
             * @return true if iterators point to different positions
             */
            bool operator!=(const LambdaVectorIterator &other) const { return !(*this == other); }
        };

        /**
         * @brief Construct a range from a vector.
         * @param vec Reference to the vector to create range over
         */
        LambdaVectorRange(const std::vector<unsigned> &vec) : vec_(vec) {}

        /// Get iterator to the first non-zero entry
        LambdaVectorIterator begin() { return LambdaVectorIterator(vec_); }

        /// Get iterator to the end
        LambdaVectorIterator end() { return LambdaVectorIterator(vec_, static_cast<unsigned>(vec_.size())); }
    };

    /// 2D vector: for each node, stores processor assignment counts
    std::vector<std::vector<unsigned>> nodeLambdaVec_;

    /// Number of processors in the system
    unsigned numProcs_ = 0;

    /**
     * @brief Initialize the container for a given number of vertices and processors.
     * @param num_vertices Number of nodes in the schedule
     * @param num_procs Number of processors in the system
     */
    inline void Initialize(const VertexIdxT numVertices, const unsigned numProcs) {
        nodeLambdaVec_.assign(numVertices, std::vector<unsigned>(numProcs, 0));
        numProcs_ = numProcs;
    }

    /**
     * @brief Reset all processor assignments for a specific node.
     * @param node Node index to reset
     */
    inline void ResetNode(const VertexIdxT node) { nodeLambdaVec_[node].assign(numProcs_, 0); }

    /**
     * @brief Clear all data from the container.
     */
    inline void Clear() { nodeLambdaVec_.clear(); }

    /**
     * @brief Check if a processor has an entry for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if the processor has at least one assignment to the node
     */
    inline bool HasProcEntry(const VertexIdxT node, const unsigned proc) const { return nodeLambdaVec_[node][proc] > 0; }

    /**
     * @brief Check if a processor has no entry for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if the processor has no assignments to the node
     */
    inline bool HasNoProcEntry(const VertexIdxT node, const unsigned proc) const { return nodeLambdaVec_[node][proc] == 0; }

    /**
     * @brief Get a reference to the processor count for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return Reference to the count (allows modification)
     */
    inline unsigned &GetProcEntry(const VertexIdxT node, const unsigned proc) { return nodeLambdaVec_[node][proc]; }

    /**
     * @brief Get the processor count for a given node (const version).
     * @param node Node index
     * @param proc Processor ID
     * @return The count value for the processor at the node
     * @pre has_proc_entry(node, proc) must be true
     */
    inline unsigned GetProcEntry(const VertexIdxT node, const unsigned proc) const {
        assert(HasProcEntry(node, proc));
        return nodeLambdaVec_[node][proc];
    }

    /**
     * @brief Get the processor count for a given node (alias for compatibility).
     * @param node Node index
     * @param proc Processor ID
     * @return The count value for the processor at the node
     * @pre has_proc_entry(node, proc) must be true
     */
    inline unsigned GetProcCount(const VertexIdxT node) const {
        unsigned count = 0;
        for (unsigned proc = 0; proc < numProcs_; ++proc) {
            if (nodeLambdaVec_[node][proc] > 0) {
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
    inline bool IncreaseProcCount(const VertexIdxT node, const unsigned proc) {
        nodeLambdaVec_[node][proc]++;
        return nodeLambdaVec_[node][proc] == 1;
    }

    /**
     * @brief Decrease the processor count for a given node.
     * @param node Node index
     * @param proc Processor ID
     * @return true if this was the last assignment of this processor to the node
     * @pre has_proc_entry(node, proc) must be true
     */
    inline bool DecreaseProcCount(const VertexIdxT node, const unsigned proc) {
        assert(HasProcEntry(node, proc));
        nodeLambdaVec_[node][proc]--;
        return nodeLambdaVec_[node][proc] == 0;
    }

    /**
     * @brief Get an iterable range over all non-zero processor entries for a node.
     * @param node Node index
     * @return Range object that can be used in range-based for loops
     */
    inline auto IterateProcEntries(const VertexIdxT node) { return LambdaVectorRange(nodeLambdaVec_[node]); }
};

}    // namespace osp
