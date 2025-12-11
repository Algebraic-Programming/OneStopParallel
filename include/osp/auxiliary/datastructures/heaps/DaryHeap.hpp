/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Christos K. Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace osp {

/**
 * @brief An indexed d-ary heap implementation.
 *
 * A d-ary heap is a generalization of a binary heap where each node has D children instead of 2.
 * This implementation is indexed, meaning it supports efficient O(log_D N) updates and removals
 * of arbitrary elements.
 *
 * @tparam Key The type of the keys, which must be hashable.
 * @tparam Value The type of the values (priorities), used for ordering.
 * @tparam D The number of children for each node (the 'd' in d-ary). Must be >= 2.
 * @tparam Compare The comparison function object type.
 */
template <typename Key, typename Value, unsigned int D, typename Compare>
class DaryHeap {
    static_assert(D >= 2, "D-ary heap must have at least 2 children per node.");

  private:
    struct NodeInfo {
        Value value;
        size_t position;
    };

  public:
    bool is_empty() const noexcept { return heap.empty(); }

    size_t size() const noexcept { return heap.size(); }

    bool contains(const Key &key) const { return node_info.count(key); }

    void push(const Key &key, const Value &value) {
        // emplace and check for success to avoid a separate lookup with contains()
        auto [it, success] = node_info.emplace(key, NodeInfo{value, heap.size()});
        if (!success) { throw std::invalid_argument("Key already exists in the heap."); }

        heap.push_back(key);
        sift_up(it->second.position);
    }

    const Key &top() const {
        if (is_empty()) { throw std::out_of_range("Heap is empty."); }
        return heap.front();
    }

    Key pop() {
        if (is_empty()) { throw std::out_of_range("Heap is empty."); }

        Key top_key = std::move(heap.front());

        node_info.erase(top_key);

        if (heap.size() > 1) {
            heap[0] = std::move(heap.back());
            heap.pop_back();
            node_info.at(heap[0]).position = 0;
            sift_down(0);
        } else {
            heap.pop_back();
        }

        return top_key;
    }

    void update(const Key &key, const Value &new_value) {
        auto it = node_info.find(key);
        if (it == node_info.end()) { throw std::invalid_argument("Key does not exist in the heap."); }
        auto &info = it->second;
        const Value old_value = info.value;

        if (comp(new_value, old_value)) {
            info.value = new_value;
            sift_up(info.position);
        } else if (comp(old_value, new_value)) {
            info.value = new_value;
            sift_down(info.position);
        }
    }

    void erase(const Key &key) {
        auto it = node_info.find(key);
        if (it == node_info.end()) { throw std::invalid_argument("Key does not exist in the heap."); }

        size_t index = it->second.position;
        size_t last_index = heap.size() - 1;

        if (index != last_index) {
            swap_nodes(index, last_index);
            heap.pop_back();
            node_info.erase(it);

            const Key &moved_key = heap[index];
            if (index > 0 && comp(node_info.at(moved_key).value, node_info.at(heap[parent(index)]).value)) {
                sift_up(index);
            } else {
                sift_down(index);
            }
        } else {
            heap.pop_back();
            node_info.erase(it);
        }
    }

    const Value &get_value(const Key &key) const {
        auto it = node_info.find(key);
        if (it == node_info.end()) { throw std::out_of_range("Key does not exist in the heap."); }
        return it->second.value;
    }

    /**
     * @brief Removes all elements from the heap.
     */
    void clear() noexcept {
        heap.clear();
        node_info.clear();
    }

  private:
    std::vector<Key> heap;
    std::unordered_map<Key, NodeInfo> node_info;
    Compare comp;

    inline size_t parent(size_t i) const noexcept { return (i - 1) / D; }

    inline size_t first_child(size_t i) const noexcept { return D * i + 1; }

    inline void swap_nodes(size_t i, size_t j) {
        node_info.at(heap[i]).position = j;
        node_info.at(heap[j]).position = i;
        std::swap(heap[i], heap[j]);
    }

    void sift_up(size_t index) {
        if (index == 0) { return; }

        Key key_to_sift = std::move(heap[index]);
        const Value &value_to_sift = node_info.at(key_to_sift).value;

        while (index > 0) {
            size_t p_idx = parent(index);
            if (comp(value_to_sift, node_info.at(heap[p_idx]).value)) {
                heap[index] = std::move(heap[p_idx]);
                node_info.at(heap[index]).position = index;
                index = p_idx;
            } else {
                break;
            }
        }
        heap[index] = std::move(key_to_sift);
        node_info.at(heap[index]).position = index;
    }

    void sift_down(size_t index) {
        Key key_to_sift = std::move(heap[index]);
        const Value &value_to_sift = node_info.at(key_to_sift).value;
        size_t size = heap.size();

        while (first_child(index) < size) {
            size_t best_child_idx = first_child(index);
            const size_t last_child_idx = std::min(best_child_idx + D, size);

            // Find the best child among the D children
            const Value *best_child_value = &node_info.at(heap[best_child_idx]).value;
            for (size_t i = best_child_idx + 1; i < last_child_idx; ++i) {
                const Value &current_child_value = node_info.at(heap[i]).value;
                if (comp(current_child_value, *best_child_value)) {
                    best_child_idx = i;
                    best_child_value = &current_child_value;
                }
            }

            // After finding the best child, compare with the sifting element
            if (comp(value_to_sift, *best_child_value)) { break; }

            // Move hole down
            heap[index] = std::move(heap[best_child_idx]);
            node_info.at(heap[index]).position = index;
            index = best_child_idx;
        }
        heap[index] = std::move(key_to_sift);
        node_info.at(heap[index]).position = index;
    }
};

template <typename Key, typename Value, unsigned int D>
using MaxDaryHeap = DaryHeap<Key, Value, D, std::greater<Value>>;

template <typename Key, typename Value, unsigned int D>
using MinDaryHeap = DaryHeap<Key, Value, D, std::less<Value>>;

template <typename Key, typename Value, typename Compare>
using IndexedHeap = DaryHeap<Key, Value, 2, Compare>;

template <typename Key, typename Value>
using MaxIndexedHeap = IndexedHeap<Key, Value, std::greater<Value>>;

template <typename Key, typename Value>
using MinIndexedHeap = IndexedHeap<Key, Value, std::less<Value>>;

}    // namespace osp
