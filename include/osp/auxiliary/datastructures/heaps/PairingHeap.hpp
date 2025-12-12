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
#include <vector>

namespace osp {

template <typename Key, typename Value, typename Compare>
class PairingHeap {
  private:
    struct Node {
        Key key_;
        Value value_;
        Node *child_ = nullptr;           // Leftmost child
        Node *nextSibling_ = nullptr;     // Sibling to the right
        Node *prevOrParent_ = nullptr;    // If leftmost child, parent; otherwise, left sibling.
    };

    Node *root_ = nullptr;
    std::unordered_map<Key, Node *> nodeMap_;
    size_t numElements_ = 0;
    Compare comp_;

    // Melds two heaps together.
    Node *Meld(Node *heap1, Node *heap2) {
        if (!heap1) {
            return heap2;
        }
        if (!heap2) {
            return heap1;
        }

        if (comp_(heap2->value_, heap1->value_)) {
            std::swap(heap1, heap2);
        }

        // heap2 becomes the new leftmost child of heap1
        heap2->nextSibling_ = heap1->child_;
        if (heap1->child_) {
            heap1->child_->prevOrParent_ = heap2;
        }
        heap1->child_ = heap2;
        heap2->prevOrParent_ = heap1;

        return heap1;
    }

    // Merges a list of sibling heaps using a two-pass strategy.
    Node *MultipassMerge(Node *firstSibling) {
        if (!firstSibling) {
            return nullptr;
        }

        std::vector<Node *> heapList;
        Node *current = firstSibling;
        while (current) {
            Node *next = current->nextSibling_;
            current->nextSibling_ = nullptr;
            current->prevOrParent_ = nullptr;
            heapList.push_back(current);
            current = next;
        }

        if (heapList.size() <= 1) {
            return heapList.empty() ? nullptr : heapList[0];
        }

        // Merge pairs from left to right
        std::vector<Node *> mergedHeaps;
        mergedHeaps.reserve((heapList.size() + 1) / 2);
        for (size_t i = 0; i + 1 < heapList.size(); i += 2) {
            mergedHeaps.push_back(Meld(heapList[i], heapList[i + 1]));
        }
        if (heapList.size() % 2 == 1) {
            mergedHeaps.push_back(heapList.back());
        }

        // Merge resulting heaps from right to left
        Node *finalHeap = mergedHeaps.back();
        for (auto it = mergedHeaps.rbegin() + 1; it != mergedHeaps.rend(); ++it) {
            finalHeap = Meld(finalHeap, *it);
        }

        return finalHeap;
    }

    // Cuts a node from its parent and siblings.
    void Cut(Node *node) {
        if (node == root_) {
            return;
        }

        if (node->prevOrParent_->child_ == node) {    // is leftmost child
            node->prevOrParent_->child_ = node->nextSibling_;
        } else {    // is not leftmost child
            node->prevOrParent_->nextSibling_ = node->nextSibling_;
        }
        if (node->nextSibling_) {
            node->nextSibling_->prevOrParent_ = node->prevOrParent_;
        }
        node->nextSibling_ = nullptr;
        node->prevOrParent_ = nullptr;
    }

  public:
    PairingHeap() = default;

    ~PairingHeap() { Clear(); }

    PairingHeap(const PairingHeap &other) : numElements_(other.numElements_), comp_(other.comp_) {
        root_ = nullptr;
        if (!other.root_) {
            return;
        }

        std::unordered_map<const Node *, Node *> oldToNew;
        std::vector<const Node *> q;
        q.reserve(other.numElements_);

        // Create root
        root_ = new Node{other.root_->key_, other.root_->value_};
        nodeMap_[root_->key_] = root_;
        oldToNew[other.root_] = root_;
        q.push_back(other.root_);

        size_t head = 0;
        while (head < q.size()) {
            const Node *oldParent = q[head++];
            Node *newParent = oldToNew[oldParent];

            if (oldParent->child_) {
                const Node *oldChild = oldParent->child_;

                // First child
                Node *newChild = new Node{oldChild->key_, oldChild->value_};
                newParent->child_ = newChild;
                newChild->prevOrParent_ = newParent;
                nodeMap_[newChild->key_] = newChild;
                oldToNew[oldChild] = newChild;
                q.push_back(oldChild);

                // Siblings
                Node *prevNewSibling = newChild;
                while (oldChild->nextSibling_) {
                    oldChild = oldChild->nextSibling_;
                    newChild = new Node{oldChild->key_, oldChild->value_};

                    prevNewSibling->nextSibling_ = newChild;
                    newChild->prevOrParent_ = prevNewSibling;

                    nodeMap_[newChild->key_] = newChild;
                    oldToNew[oldChild] = newChild;
                    q.push_back(oldChild);

                    prevNewSibling = newChild;
                }
            }
        }
    }

    PairingHeap &operator=(const PairingHeap &other) {
        if (this != &other) {
            PairingHeap temp(other);
            std::swap(root_, temp.root_);
            std::swap(nodeMap_, temp.nodeMap_);
            std::swap(numElements_, temp.numElements_);
            std::swap(comp_, temp.comp_);
        }
        return *this;
    }

    PairingHeap(PairingHeap &&) = default;
    PairingHeap &operator=(PairingHeap &&) = default;

    // Checks if the heap is empty.
    bool IsEmpty() const { return root_ == nullptr; }

    // Returns the number of elements in the heap.
    size_t size() const { return numElements_; }

    // Checks if a key exists in the heap.
    bool Contains(const Key &key) const { return nodeMap_.count(key); }

    // Inserts a new key-value pair into the heap.
    void Push(const Key &key, const Value &value) {
        Node *newNode = new Node{key, value};
        // emplace and check for success to avoid a separate lookup with contains()
        const auto pair = nodeMap_.emplace(key, newNode);
        const bool &success = pair.second;
        if (!success) {
            delete newNode;    // Avoid memory leak if key already exists
            throw std::invalid_argument("Key already exists in the heap.");
        }
        root_ = Meld(root_, newNode);
        numElements_++;
    }

    // Returns the key with the minimum value without removing it.
    const Key &Top() const {
        if (IsEmpty()) {
            throw std::out_of_range("Heap is empty.");
        }
        return root_->key_;
    }

    // Removes and returns the key with the minimum value.
    Key Pop() {
        if (IsEmpty()) {
            throw std::out_of_range("Heap is empty.");
        }

        Node *oldRoot = root_;
        Key topKey = oldRoot->key_;

        root_ = MultipassMerge(oldRoot->child_);

        nodeMap_.erase(topKey);
        delete oldRoot;
        numElements_--;

        return topKey;
    }

    // Updates the value of an existing key.
    void Update(const Key &key, const Value &newValue) {
        auto it = nodeMap_.find(key);
        if (it == nodeMap_.end()) {
            throw std::invalid_argument("Key does not exist in the heap.");
        }

        Node *node = it->second;
        const Value oldValue = node->value_;

        if (comp_(newValue, oldValue)) {    // Decrease key
            node->value_ = newValue;
            if (node != root_) {
                Cut(node);
                root_ = Meld(root_, node);
            }
        } else if (comp_(oldValue, newValue)) {    // Increase key
            node->value_ = newValue;
            if (node != root_) {
                Cut(node);
                if (node->child_) {
                    root_ = Meld(root_, MultipassMerge(node->child_));
                    node->child_ = nullptr;
                }
                root_ = Meld(root_, node);
            } else {
                // The root's value increased, it might not be the minimum anymore.
                // We can treat it as if we popped it and re-inserted it, without the delete/new.
                Node *oldRoot = root_;
                root_ = MultipassMerge(oldRoot->child_);
                oldRoot->child_ = nullptr;
                root_ = Meld(root_, oldRoot);
            }
        } else {
            node->value_ = newValue;
        }
        // If values are equal, do nothing.
    }

    // Removes an arbitrary key from the heap.
    void Erase(const Key &key) {
        auto it = nodeMap_.find(key);
        if (it == nodeMap_.end()) {
            throw std::invalid_argument("Key does not exist in the heap.");
        }
        Node *nodeToErase = it->second;

        if (nodeToErase == root_) {
            Pop();
            return;
        }

        Cut(nodeToErase);

        // Merge its children into the main heap
        if (nodeToErase->child_) {
            root_ = Meld(root_, MultipassMerge(nodeToErase->child_));
            nodeToErase->child_ = nullptr;
        }

        nodeMap_.erase(key);
        delete nodeToErase;
        numElements_--;
    }

    // Gets the value for a given key.
    const Value &GetValue(const Key &key) const {
        auto it = nodeMap_.find(key);
        if (it == nodeMap_.end()) {
            throw std::out_of_range("Key does not exist in the heap.");
        }
        return it->second->value_;
    }

    // Removes all elements from the heap.
    void Clear() {
        if (!root_) {
            return;
        }

        // Iterative post-order traversal to delete all nodes
        std::vector<Node *> toVisit;
        if (numElements_ > 0) {
            toVisit.reserve(numElements_);
        }
        toVisit.push_back(root_);

        while (!toVisit.empty()) {
            Node *current = toVisit.back();
            toVisit.pop_back();

            Node *child = current->child_;
            while (child) {
                toVisit.push_back(child);
                child = child->nextSibling_;
            }
            delete current;
        }

        root_ = nullptr;
        nodeMap_.clear();
        numElements_ = 0;
    }

    // Retrieves keys with the top value, up to a specified limit.
    // If limit is 0, all keys with the top value are returned.
    std::vector<Key> GetTopKeys(size_t limit = 0) const {
        std::vector<Key> topKeys;
        if (IsEmpty()) {
            return topKeys;
        }

        if (limit > 0) {
            topKeys.reserve(limit);
        }

        const Value &topValue = root_->value_;
        std::vector<const Node *> q;
        q.push_back(root_);
        size_t head = 0;

        while (head < q.size()) {
            const Node *current = q[head++];

            if (comp_(topValue, current->value_)) {
                continue;
            }

            topKeys.push_back(current->key_);
            if (limit > 0 && topKeys.size() >= limit) {
                return topKeys;
            }

            Node *child = current->child_;
            while (child) {
                q.push_back(child);
                child = child->nextSibling_;
            }
        }
        return topKeys;
    }
};

template <typename Key, typename Value>
using MaxPairingHeap = PairingHeap<Key, Value, std::greater<Value>>;

template <typename Key, typename Value>
using MinPairingHeap = PairingHeap<Key, Value, std::less<Value>>;

}    // namespace osp
