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
template <typename Key, typename Value, unsigned int d, typename Compare>
class DaryHeap {
    static_assert(d >= 2, "D-ary heap must have at least 2 children per node.");

  private:
    struct NodeInfo {
        Value value_;
        size_t position_;
    };

  public:
    bool is_empty() const noexcept { return heap_.empty(); }

    size_t size() const noexcept { return heap_.size(); }

    bool contains(const Key &key) const { return nodeInfo_.count(key); }

    void push(const Key &key, const Value &value) {
        // emplace and check for success to avoid a separate lookup with contains()
        auto [it, success] = nodeInfo_.emplace(key, NodeInfo{value, heap_.size()});
        if (!success) {
            throw std::invalid_argument("Key already exists in the heap.");
        }

        heap_.push_back(key);
        SiftUp(it->second.position_);
    }

    const Key &top() const {
        if (is_empty()) {
            throw std::out_of_range("Heap is empty.");
        }
        return heap_.front();
    }

    Key pop() {
        if (is_empty()) {
            throw std::out_of_range("Heap is empty.");
        }

        Key topKey = std::move(heap_.front());

        nodeInfo_.erase(topKey);

        if (heap_.size() > 1) {
            heap_[0] = std::move(heap_.back());
            heap_.pop_back();
            nodeInfo_.at(heap_[0]).position_ = 0;
            SiftDown(0);
        } else {
            heap_.pop_back();
        }

        return topKey;
    }

    void update(const Key &key, const Value &newValue) {
        auto it = nodeInfo_.find(key);
        if (it == nodeInfo_.end()) {
            throw std::invalid_argument("Key does not exist in the heap.");
        }
        auto &info = it->second;
        const Value oldValue = info.value_;

        if (comp_(newValue, oldValue)) {
            info.value_ = newValue;
            SiftUp(info.position_);
        } else if (comp_(oldValue, newValue)) {
            info.value_ = newValue;
            SiftDown(info.position_);
        }
    }

    void erase(const Key &key) {
        auto it = nodeInfo_.find(key);
        if (it == nodeInfo_.end()) {
            throw std::invalid_argument("Key does not exist in the heap.");
        }

        size_t index = it->second.position_;
        size_t lastIndex = heap_.size() - 1;

        if (index != lastIndex) {
            SwapNodes(index, lastIndex);
            heap_.pop_back();
            nodeInfo_.erase(it);

            const Key &movedKey = heap_[index];
            if (index > 0 && comp_(nodeInfo_.at(movedKey).value_, nodeInfo_.at(heap_[Parent(index)]).value_)) {
                SiftUp(index);
            } else {
                SiftDown(index);
            }
        } else {
            heap_.pop_back();
            nodeInfo_.erase(it);
        }
    }

    const Value &get_value(const Key &key) const {
        auto it = nodeInfo_.find(key);
        if (it == nodeInfo_.end()) {
            throw std::out_of_range("Key does not exist in the heap.");
        }
        return it->second.value_;
    }

    /**
     * @brief Removes all elements from the heap.
     */
    void clear() noexcept {
        heap_.clear();
        nodeInfo_.clear();
    }

  private:
    std::vector<Key> heap_;
    std::unordered_map<Key, NodeInfo> nodeInfo_;
    Compare comp_;

    inline size_t Parent(size_t i) const noexcept { return (i - 1) / d; }

    inline size_t FirstChild(size_t i) const noexcept { return d * i + 1; }

    inline void SwapNodes(size_t i, size_t j) {
        nodeInfo_.at(heap_[i]).position_ = j;
        nodeInfo_.at(heap_[j]).position_ = i;
        std::swap(heap_[i], heap_[j]);
    }

    void SiftUp(size_t index) {
        if (index == 0) {
            return;
        }

        Key keyToSift = std::move(heap_[index]);
        const Value &valueToSift = nodeInfo_.at(keyToSift).value_;

        while (index > 0) {
            size_t pIdx = Parent(index);
            if (comp_(valueToSift, nodeInfo_.at(heap_[pIdx]).value_)) {
                heap_[index] = std::move(heap_[pIdx]);
                nodeInfo_.at(heap_[index]).position_ = index;
                index = pIdx;
            } else {
                break;
            }
        }
        heap_[index] = std::move(keyToSift);
        nodeInfo_.at(heap_[index]).position_ = index;
    }

    void SiftDown(size_t index) {
        Key keyToSift = std::move(heap_[index]);
        const Value &valueToSift = nodeInfo_.at(keyToSift).value_;
        size_t size = heap_.size();

        while (FirstChild(index) < size) {
            size_t bestChildIdx = FirstChild(index);
            const size_t lastChildIdx = std::min(bestChildIdx + d, size);

            // Find the best child among the D children
            const Value *bestChildValue = &nodeInfo_.at(heap_[bestChildIdx]).value_;
            for (size_t i = bestChildIdx + 1; i < lastChildIdx; ++i) {
                const Value &currentChildValue = nodeInfo_.at(heap_[i]).value_;
                if (comp_(currentChildValue, *bestChildValue)) {
                    bestChildIdx = i;
                    bestChildValue = &currentChildValue;
                }
            }

            // After finding the best child, compare with the sifting element
            if (comp_(valueToSift, *bestChildValue)) {
                break;
            }

            // Move hole down
            heap_[index] = std::move(heap_[bestChildIdx]);
            nodeInfo_.at(heap_[index]).position_ = index;
            index = bestChildIdx;
        }
        heap_[index] = std::move(keyToSift);
        nodeInfo_.at(heap_[index]).position_ = index;
    }
};

template <typename Key, typename Value, unsigned int d>
using MaxDaryHeap = DaryHeap<Key, Value, d, std::greater<Value>>;

template <typename Key, typename Value, unsigned int d>
using MinDaryHeap = DaryHeap<Key, Value, d, std::less<Value>>;

template <typename Key, typename Value, typename Compare>
using IndexedHeap = DaryHeap<Key, Value, 2, Compare>;

template <typename Key, typename Value>
using MaxIndexedHeap = IndexedHeap<Key, Value, std::greater<Value>>;

template <typename Key, typename Value>
using MinIndexedHeap = IndexedHeap<Key, Value, std::less<Value>>;

}    // namespace osp
