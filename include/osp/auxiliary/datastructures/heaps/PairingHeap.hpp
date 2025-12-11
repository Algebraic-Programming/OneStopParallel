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
        Key key;
        Value value;
        Node *child = nullptr;             // Leftmost child
        Node *next_sibling = nullptr;      // Sibling to the right
        Node *prev_or_parent = nullptr;    // If leftmost child, parent; otherwise, left sibling.
    };

    Node *root = nullptr;
    std::unordered_map<Key, Node *> node_map;
    size_t num_elements = 0;
    Compare comp;

    // Melds two heaps together.
    Node *meld(Node *heap1, Node *heap2) {
        if (!heap1) { return heap2; }
        if (!heap2) { return heap1; }

        if (comp(heap2->value, heap1->value)) { std::swap(heap1, heap2); }

        // heap2 becomes the new leftmost child of heap1
        heap2->next_sibling = heap1->child;
        if (heap1->child) { heap1->child->prev_or_parent = heap2; }
        heap1->child = heap2;
        heap2->prev_or_parent = heap1;

        return heap1;
    }

    // Merges a list of sibling heaps using a two-pass strategy.
    Node *multipass_merge(Node *first_sibling) {
        if (!first_sibling) { return nullptr; }

        std::vector<Node *> heap_list;
        Node *current = first_sibling;
        while (current) {
            Node *next = current->next_sibling;
            current->next_sibling = nullptr;
            current->prev_or_parent = nullptr;
            heap_list.push_back(current);
            current = next;
        }

        if (heap_list.size() <= 1) { return heap_list.empty() ? nullptr : heap_list[0]; }

        // Merge pairs from left to right
        std::vector<Node *> merged_heaps;
        merged_heaps.reserve((heap_list.size() + 1) / 2);
        for (size_t i = 0; i + 1 < heap_list.size(); i += 2) { merged_heaps.push_back(meld(heap_list[i], heap_list[i + 1])); }
        if (heap_list.size() % 2 == 1) { merged_heaps.push_back(heap_list.back()); }

        // Merge resulting heaps from right to left
        Node *final_heap = merged_heaps.back();
        for (auto it = merged_heaps.rbegin() + 1; it != merged_heaps.rend(); ++it) { final_heap = meld(final_heap, *it); }

        return final_heap;
    }

    // Cuts a node from its parent and siblings.
    void cut(Node *node) {
        if (node == root) { return; }

        if (node->prev_or_parent->child == node) {    // is leftmost child
            node->prev_or_parent->child = node->next_sibling;
        } else {    // is not leftmost child
            node->prev_or_parent->next_sibling = node->next_sibling;
        }
        if (node->next_sibling) { node->next_sibling->prev_or_parent = node->prev_or_parent; }
        node->next_sibling = nullptr;
        node->prev_or_parent = nullptr;
    }

  public:
    PairingHeap() = default;

    ~PairingHeap() { clear(); }

    PairingHeap(const PairingHeap &other) : num_elements(other.num_elements), comp(other.comp) {
        root = nullptr;
        if (!other.root) { return; }

        std::unordered_map<const Node *, Node *> old_to_new;
        std::vector<const Node *> q;
        q.reserve(other.num_elements);

        // Create root
        root = new Node{other.root->key, other.root->value};
        node_map[root->key] = root;
        old_to_new[other.root] = root;
        q.push_back(other.root);

        size_t head = 0;
        while (head < q.size()) {
            const Node *old_parent = q[head++];
            Node *new_parent = old_to_new[old_parent];

            if (old_parent->child) {
                const Node *old_child = old_parent->child;

                // First child
                Node *new_child = new Node{old_child->key, old_child->value};
                new_parent->child = new_child;
                new_child->prev_or_parent = new_parent;
                node_map[new_child->key] = new_child;
                old_to_new[old_child] = new_child;
                q.push_back(old_child);

                // Siblings
                Node *prev_new_sibling = new_child;
                while (old_child->next_sibling) {
                    old_child = old_child->next_sibling;
                    new_child = new Node{old_child->key, old_child->value};

                    prev_new_sibling->next_sibling = new_child;
                    new_child->prev_or_parent = prev_new_sibling;

                    node_map[new_child->key] = new_child;
                    old_to_new[old_child] = new_child;
                    q.push_back(old_child);

                    prev_new_sibling = new_child;
                }
            }
        }
    }

    PairingHeap &operator=(const PairingHeap &other) {
        if (this != &other) {
            PairingHeap temp(other);
            std::swap(root, temp.root);
            std::swap(node_map, temp.node_map);
            std::swap(num_elements, temp.num_elements);
            std::swap(comp, temp.comp);
        }
        return *this;
    }

    PairingHeap(PairingHeap &&) = default;
    PairingHeap &operator=(PairingHeap &&) = default;

    // Checks if the heap is empty.
    bool is_empty() const { return root == nullptr; }

    // Returns the number of elements in the heap.
    size_t size() const { return num_elements; }

    // Checks if a key exists in the heap.
    bool contains(const Key &key) const { return node_map.count(key); }

    // Inserts a new key-value pair into the heap.
    void push(const Key &key, const Value &value) {
        Node *new_node = new Node{key, value};
        // emplace and check for success to avoid a separate lookup with contains()
        const auto pair = node_map.emplace(key, new_node);
        const bool &success = pair.second;
        if (!success) {
            delete new_node;    // Avoid memory leak if key already exists
            throw std::invalid_argument("Key already exists in the heap.");
        }
        root = meld(root, new_node);
        num_elements++;
    }

    // Returns the key with the minimum value without removing it.
    const Key &top() const {
        if (is_empty()) { throw std::out_of_range("Heap is empty."); }
        return root->key;
    }

    // Removes and returns the key with the minimum value.
    Key pop() {
        if (is_empty()) { throw std::out_of_range("Heap is empty."); }

        Node *old_root = root;
        Key top_key = old_root->key;

        root = multipass_merge(old_root->child);

        node_map.erase(top_key);
        delete old_root;
        num_elements--;

        return top_key;
    }

    // Updates the value of an existing key.
    void update(const Key &key, const Value &new_value) {
        auto it = node_map.find(key);
        if (it == node_map.end()) { throw std::invalid_argument("Key does not exist in the heap."); }

        Node *node = it->second;
        const Value old_value = node->value;

        if (comp(new_value, old_value)) {    // Decrease key
            node->value = new_value;
            if (node != root) {
                cut(node);
                root = meld(root, node);
            }
        } else if (comp(old_value, new_value)) {    // Increase key
            node->value = new_value;
            if (node != root) {
                cut(node);
                if (node->child) {
                    root = meld(root, multipass_merge(node->child));
                    node->child = nullptr;
                }
                root = meld(root, node);
            } else {
                // The root's value increased, it might not be the minimum anymore.
                // We can treat it as if we popped it and re-inserted it, without the delete/new.
                Node *old_root = root;
                root = multipass_merge(old_root->child);
                old_root->child = nullptr;
                root = meld(root, old_root);
            }
        } else {
            node->value = new_value;
        }
        // If values are equal, do nothing.
    }

    // Removes an arbitrary key from the heap.
    void erase(const Key &key) {
        auto it = node_map.find(key);
        if (it == node_map.end()) { throw std::invalid_argument("Key does not exist in the heap."); }
        Node *node_to_erase = it->second;

        if (node_to_erase == root) {
            pop();
            return;
        }

        cut(node_to_erase);

        // Merge its children into the main heap
        if (node_to_erase->child) {
            root = meld(root, multipass_merge(node_to_erase->child));
            node_to_erase->child = nullptr;
        }

        node_map.erase(key);
        delete node_to_erase;
        num_elements--;
    }

    // Gets the value for a given key.
    const Value &get_value(const Key &key) const {
        auto it = node_map.find(key);
        if (it == node_map.end()) { throw std::out_of_range("Key does not exist in the heap."); }
        return it->second->value;
    }

    // Removes all elements from the heap.
    void clear() {
        if (!root) { return; }

        // Iterative post-order traversal to delete all nodes
        std::vector<Node *> to_visit;
        if (num_elements > 0) { to_visit.reserve(num_elements); }
        to_visit.push_back(root);

        while (!to_visit.empty()) {
            Node *current = to_visit.back();
            to_visit.pop_back();

            Node *child = current->child;
            while (child) {
                to_visit.push_back(child);
                child = child->next_sibling;
            }
            delete current;
        }

        root = nullptr;
        node_map.clear();
        num_elements = 0;
    }

    // Retrieves keys with the top value, up to a specified limit.
    // If limit is 0, all keys with the top value are returned.
    std::vector<Key> get_top_keys(size_t limit = 0) const {
        std::vector<Key> top_keys;
        if (is_empty()) { return top_keys; }

        if (limit > 0) { top_keys.reserve(limit); }

        const Value &top_value = root->value;
        std::vector<const Node *> q;
        q.push_back(root);
        size_t head = 0;

        while (head < q.size()) {
            const Node *current = q[head++];

            if (comp(top_value, current->value)) { continue; }

            top_keys.push_back(current->key);
            if (limit > 0 && top_keys.size() >= limit) { return top_keys; }

            Node *child = current->child;
            while (child) {
                q.push_back(child);
                child = child->next_sibling;
            }
        }
        return top_keys;
    }
};

template <typename Key, typename Value>
using MaxPairingHeap = PairingHeap<Key, Value, std::greater<Value>>;

template <typename Key, typename Value>
using MinPairingHeap = PairingHeap<Key, Value, std::less<Value>>;

}    // namespace osp
