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

#define BOOST_TEST_MODULE HeapTest
#include <algorithm>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "osp/auxiliary/datastructures/heaps/DaryHeap.hpp"
#include "osp/auxiliary/datastructures/heaps/PairingHeap.hpp"

namespace osp::test {

// Wrapper for boost::heap::fibonacci_heap to match the test interface
template <typename Key, typename Value, bool IsMinHeap = true>
class BoostFibonacciHeapWrapper {
  private:
    struct Node {
        Key key;
        Value value;
    };

    struct NodeCompare {
        bool operator()(const Node &a, const Node &b) const {
            if constexpr (IsMinHeap) {
                return a.value > b.value;    // For min-heap
            } else {
                return a.value < b.value;    // For max-heap
            }
        }
    };

    using BoostHeap = boost::heap::fibonacci_heap<Node, boost::heap::compare<NodeCompare>>;
    using handle_type = typename BoostHeap::handle_type;

    BoostHeap heap;
    std::unordered_map<Key, handle_type> handles;

  public:
    BoostFibonacciHeapWrapper() = default;

    bool is_empty() const { return heap.empty(); }

    size_t size() const { return heap.size(); }

    bool contains(const Key &key) const { return handles.count(key); }

    const Key &top() const {
        if (is_empty()) { throw std::out_of_range("Heap is empty"); }
        return heap.top().key;
    }

    Key pop() {
        if (is_empty()) { throw std::out_of_range("Heap is empty"); }
        Key top_key = heap.top().key;
        heap.pop();
        handles.erase(top_key);
        return top_key;
    }

    void push(const Key &key, const Value &value) {
        if (contains(key)) { throw std::invalid_argument("Key already exists"); }
        handle_type handle = heap.push({key, value});
        handles[key] = handle;
    }

    Value get_value(const Key &key) const {
        if (!contains(key)) { throw std::out_of_range("Key not found"); }
        return (*handles.at(key)).value;
    }

    void update(const Key &key, const Value &new_value) {
        if (!contains(key)) { throw std::invalid_argument("Key not found for update"); }
        handle_type handle = handles.at(key);
        (*handle).value = new_value;
        heap.update(handle);
    }

    void erase(const Key &key) {
        if (!contains(key)) { throw std::invalid_argument("Key not found for erase"); }
        heap.erase(handles.at(key));
        handles.erase(key);
    }

    void clear() {
        heap.clear();
        handles.clear();
    }
};

template <typename Key, typename Value>
using MinBoostFibonacciHeap = BoostFibonacciHeapWrapper<Key, Value, true>;

template <typename Key, typename Value>
using MaxBoostFibonacciHeap = BoostFibonacciHeapWrapper<Key, Value, false>;

// Wrapper for std::set to match the test interface
template <typename Key, typename Value, bool IsMinHeap = true>
class StdSetWrapper {
  private:
    struct NodeCompare {
        bool operator()(const std::pair<Value, Key> &a, const std::pair<Value, Key> &b) const {
            if (a.first != b.first) {
                if constexpr (IsMinHeap) {
                    return a.first < b.first;    // For min-heap
                } else {
                    return a.first > b.first;    // For max-heap
                }
            }
            return a.second < b.second;    // Tie-breaking
        }
    };

    using SetType = std::set<std::pair<Value, Key>, NodeCompare>;
    SetType data_set;
    std::unordered_map<Key, Value> value_map;

  public:
    StdSetWrapper() = default;

    bool is_empty() const { return data_set.empty(); }

    size_t size() const { return data_set.size(); }

    bool contains(const Key &key) const { return value_map.count(key); }

    const Key &top() const {
        if (is_empty()) { throw std::out_of_range("Heap is empty"); }
        return data_set.begin()->second;
    }

    Key pop() {
        if (is_empty()) { throw std::out_of_range("Heap is empty"); }
        auto top_node = *data_set.begin();
        data_set.erase(data_set.begin());
        value_map.erase(top_node.second);
        return top_node.second;
    }

    void push(const Key &key, const Value &value) {
        if (contains(key)) { throw std::invalid_argument("Key already exists"); }
        data_set.insert({value, key});
        value_map[key] = value;
    }

    Value get_value(const Key &key) const {
        if (!contains(key)) { throw std::out_of_range("Key not found"); }
        return value_map.at(key);
    }

    void update(const Key &key, const Value &new_value) {
        if (!contains(key)) { throw std::invalid_argument("Key not found for update"); }
        Value old_value = value_map.at(key);
        if (old_value == new_value) { return; }
        data_set.erase({old_value, key});
        data_set.insert({new_value, key});
        value_map[key] = new_value;
    }

    void erase(const Key &key) {
        if (!contains(key)) { throw std::invalid_argument("Key not found for erase"); }
        Value value = value_map.at(key);
        data_set.erase({value, key});
        value_map.erase(key);
    }

    void clear() {
        data_set.clear();
        value_map.clear();
    }
};

template <typename Key, typename Value>
using MinStdSetHeap = StdSetWrapper<Key, Value, true>;

template <typename Key, typename Value>
using MaxStdSetHeap = StdSetWrapper<Key, Value, false>;

// Generic test suite for any min-heap implementation that follows the API.
template <typename HeapType>
void test_min_heap_functionality() {
    HeapType heap;

    // Basic properties of an empty heap
    BOOST_CHECK(heap.is_empty());
    BOOST_CHECK_EQUAL(heap.size(), 0);
    BOOST_CHECK(!heap.contains("A"));
    BOOST_CHECK_THROW(heap.top(), std::out_of_range);
    BOOST_CHECK_THROW(heap.pop(), std::out_of_range);

    // Push elements
    heap.push("A", 10);
    heap.push("B", 5);
    heap.push("C", 15);

    BOOST_CHECK(!heap.is_empty());
    BOOST_CHECK_EQUAL(heap.size(), 3);
    BOOST_CHECK(heap.contains("A"));
    BOOST_CHECK(heap.contains("B"));
    BOOST_CHECK(heap.contains("C"));
    BOOST_CHECK(!heap.contains("D"));

    // Check for duplicate key insertion
    BOOST_CHECK_THROW(heap.push("A", 20), std::invalid_argument);

    // Test top() and pop() for min-heap
    BOOST_CHECK_EQUAL(heap.top(), "B");
    BOOST_CHECK_EQUAL(heap.pop(), "B");
    BOOST_CHECK_EQUAL(heap.size(), 2);
    BOOST_CHECK(!heap.contains("B"));

    BOOST_CHECK_EQUAL(heap.top(), "A");
    BOOST_CHECK_EQUAL(heap.pop(), "A");

    BOOST_CHECK_EQUAL(heap.top(), "C");
    BOOST_CHECK_EQUAL(heap.pop(), "C");
    BOOST_CHECK(heap.is_empty());

    // Repopulate for update/erase tests
    heap.push("A", 10);
    heap.push("B", 5);
    heap.push("C", 15);
    heap.push("D", 2);
    heap.push("E", 20);

    // Test get_value
    BOOST_CHECK_EQUAL(heap.get_value("A"), 10);
    BOOST_CHECK_EQUAL(heap.get_value("D"), 2);
    BOOST_CHECK_THROW(heap.get_value("Z"), std::out_of_range);

    // Test update (decrease-key)
    heap.update("B", 1);    // B: 5 -> 1. Should be new top.
    BOOST_CHECK_EQUAL(heap.top(), "B");
    BOOST_CHECK_EQUAL(heap.get_value("B"), 1);

    // Test update (increase-key)
    heap.update("B", 25);    // B: 1 -> 25. D (2) should be new top.
    BOOST_CHECK_EQUAL(heap.top(), "D");
    BOOST_CHECK_EQUAL(heap.get_value("B"), 25);

    // Test update with same value
    heap.update("A", 10);
    BOOST_CHECK_EQUAL(heap.get_value("A"), 10);

    // Test erase
    heap.erase("D");    // Erase top element
    BOOST_CHECK_EQUAL(heap.size(), 4);
    BOOST_CHECK(!heap.contains("D"));
    BOOST_CHECK_EQUAL(heap.top(), "A");    // A (10) is new top

    heap.erase("E");    // Erase non-top element
    BOOST_CHECK_EQUAL(heap.size(), 3);
    BOOST_CHECK(!heap.contains("E"));
    BOOST_CHECK_THROW(heap.erase("Z"), std::invalid_argument);

    // Test clear
    heap.clear();
    BOOST_CHECK(heap.is_empty());
    BOOST_CHECK_EQUAL(heap.size(), 0);
}

template <typename HeapType>
void test_max_heap_functionality() {
    HeapType heap;
    heap.push("A", 10);
    heap.push("B", 5);
    heap.push("C", 15);

    // Test pop order for max-heap
    BOOST_CHECK_EQUAL(heap.top(), "C");
    heap.pop();
    BOOST_CHECK_EQUAL(heap.top(), "A");
    heap.pop();
    BOOST_CHECK_EQUAL(heap.top(), "B");
}

// Stress test with a larger number of elements
template <typename HeapType>
void stress_test_heap() {
    HeapType heap;
    const int num_items = 1000;

    for (int i = 0; i < num_items; ++i) { heap.push(std::to_string(i), i); }
    for (int i = 0; i < num_items / 2; ++i) { heap.update(std::to_string(i), i - num_items); }

    std::vector<int> popped_values;
    while (!heap.is_empty()) {
        popped_values.push_back(heap.get_value(heap.top()));
        heap.pop();
    }

    BOOST_CHECK_EQUAL(popped_values.size(), num_items);
    BOOST_CHECK(std::is_sorted(popped_values.begin(), popped_values.end()));
}

// Performance test suite for different heap workloads.
template <typename HeapType>
void run_performance_test(const std::string &heap_name, size_t num_items, size_t num_updates, size_t num_random_ops) {
    std::cout << "\n--- Performance Test for " << heap_name << " ---" << std::endl;

    std::vector<std::string> keys(num_items);
    std::vector<int> priorities(num_items);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(0, static_cast<int>(num_items * 10));

    for (size_t i = 0; i < num_items; ++i) {
        keys[i] = std::to_string(i);
        priorities[i] = distrib(gen);
    }

    HeapType heap;

    // Scenario 1: Bulk Insert
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_items; ++i) { heap.push(keys[i], priorities[i]); }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Bulk Insert (" << num_items << " items): " << duration.count() << " ms" << std::endl;

    // Scenario 2: Decrease Key
    std::uniform_int_distribution<size_t> key_distrib(0, num_items - 1);
    std::uniform_int_distribution<> dec_dist(1, 100);
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_updates; ++i) {
        size_t key_idx = key_distrib(gen);
        int new_prio = heap.get_value(keys[key_idx]) - dec_dist(gen);
        heap.update(keys[key_idx], new_prio);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Decrease Key (" << num_updates << " updates): " << duration.count() << " ms" << std::endl;

    // Scenario 3: Bulk Pop
    start = std::chrono::high_resolution_clock::now();
    while (!heap.is_empty()) { heap.pop(); }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Bulk Pop (" << num_items << " items): " << duration.count() << " ms" << std::endl;

    BOOST_CHECK(heap.is_empty());

    // Scenario 4: Random Operations (Push, Erase, Update)
    heap.clear();
    std::vector<std::string> present_keys;
    present_keys.reserve(num_items);
    std::vector<bool> key_in_heap(num_items, false);
    std::uniform_int_distribution<int> op_dist(0, 2);    // 0: push, 1: erase, 2: update

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_random_ops; ++i) {
        int op = op_dist(gen);
        if (op == 0 || present_keys.empty()) {    // Push
            size_t key_idx = key_distrib(gen);
            if (!key_in_heap[key_idx]) {
                heap.push(keys[key_idx], priorities[key_idx]);
                present_keys.push_back(keys[key_idx]);
                key_in_heap[key_idx] = true;
            }
        } else {    // Erase or Update
            std::uniform_int_distribution<size_t> present_key_dist(0, present_keys.size() - 1);
            size_t present_key_vec_idx = present_key_dist(gen);
            std::string key_to_op = present_keys[present_key_vec_idx];

            if (op == 1) {    // Erase a random element
                heap.erase(key_to_op);
                key_in_heap[std::stoul(key_to_op)] = false;
                std::swap(present_keys[present_key_vec_idx], present_keys.back());
                present_keys.pop_back();
            } else {    // op == 2, Update a random element (decrease key)
                int new_prio = heap.get_value(key_to_op) - dec_dist(gen);
                heap.update(key_to_op, new_prio);
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Random Ops (" << num_random_ops << " ops of push/erase/update): " << duration.count() << " ms" << std::endl;

    // Scenario 5: Mixed Workload with Re-initialization
    const size_t num_outer_loops_s5 = 500;
    const size_t num_inner_loops_s5 = 10;
    const size_t num_initial_pushes_s5 = 100;
    const size_t num_pushes_per_iter_s5 = 25;
    const size_t num_updates_per_iter_s5 = 25;

    // A large pool of keys to draw from for pushes, to avoid collisions.
    const size_t key_pool_size_s5 = num_outer_loops_s5 * (num_initial_pushes_s5 + num_inner_loops_s5 * num_pushes_per_iter_s5);
    std::vector<std::string> keys_s5(key_pool_size_s5);
    std::vector<int> priorities_s5(key_pool_size_s5);
    for (size_t i = 0; i < key_pool_size_s5; ++i) {
        keys_s5[i] = "s5_" + std::to_string(i);
        priorities_s5[i] = distrib(gen);
    }

    size_t key_idx_counter_s5 = 0;

    start = std::chrono::high_resolution_clock::now();

    for (size_t outer_i = 0; outer_i < num_outer_loops_s5; ++outer_i) {
        heap.clear();
        std::vector<std::string> present_keys_s5;
        present_keys_s5.reserve(num_initial_pushes_s5 + num_inner_loops_s5 * (num_pushes_per_iter_s5 - 1));

        // Initial push
        for (size_t i = 0; i < num_initial_pushes_s5; ++i) {
            const auto &key = keys_s5[key_idx_counter_s5];
            heap.push(key, priorities_s5[key_idx_counter_s5]);
            present_keys_s5.push_back(key);
            key_idx_counter_s5++;
        }

        for (size_t inner_i = 0; inner_i < num_inner_loops_s5; ++inner_i) {
            // 1. Pop once
            if (!heap.is_empty()) {
                std::string popped_key = heap.pop();
                // Remove from present_keys_s5 efficiently
                auto it = std::find(present_keys_s5.begin(), present_keys_s5.end(), popped_key);
                if (it != present_keys_s5.end()) {
                    std::swap(*it, present_keys_s5.back());
                    present_keys_s5.pop_back();
                }
            }

            // 2. Push 25 keys
            for (size_t j = 0; j < num_pushes_per_iter_s5; ++j) {
                const auto &key = keys_s5[key_idx_counter_s5];
                heap.push(key, priorities_s5[key_idx_counter_s5]);
                present_keys_s5.push_back(key);
                key_idx_counter_s5++;
            }

            // 3. Update 25 keys
            if (!present_keys_s5.empty()) {
                std::uniform_int_distribution<size_t> present_key_dist(0, present_keys_s5.size() - 1);
                for (size_t j = 0; j < num_updates_per_iter_s5; ++j) {
                    const auto &key_to_update = present_keys_s5[present_key_dist(gen)];
                    heap.update(key_to_update, heap.get_value(key_to_update) - dec_dist(gen));
                }
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Mixed Re-Init (" << num_outer_loops_s5 << " runs of init + " << num_inner_loops_s5
              << "x(pop/push/update)): " << duration.count() << " ms" << std::endl;
}
BOOST_AUTO_TEST_SUITE(HeapTests)

BOOST_AUTO_TEST_CASE(PairingHeapTest) { test_min_heap_functionality<MinPairingHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(MaxPairingHeapTest) { test_max_heap_functionality<MaxPairingHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(PairingHeapStressTest) { stress_test_heap<MinPairingHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(BoostFibonacciHeapTest) { test_min_heap_functionality<MinBoostFibonacciHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(MaxBoostFibonacciHeapTest) { test_max_heap_functionality<MaxBoostFibonacciHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(BoostFibonacciHeapStressTest) { stress_test_heap<MinBoostFibonacciHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(StdSetHeapTest) { test_min_heap_functionality<MinStdSetHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(MaxStdSetHeapTest) { test_max_heap_functionality<MaxStdSetHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(StdSetHeapStressTest) { stress_test_heap<MinStdSetHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(DaryHeap_D2_Test) { test_min_heap_functionality<MinDaryHeap<std::string, int, 2>>(); }

BOOST_AUTO_TEST_CASE(MaxDaryHeap_D2_Test) { test_max_heap_functionality<MaxDaryHeap<std::string, int, 2>>(); }

BOOST_AUTO_TEST_CASE(DaryHeap_D2_StressTest) { stress_test_heap<MinDaryHeap<std::string, int, 2>>(); }

BOOST_AUTO_TEST_CASE(DaryHeap_D4_Test) { test_min_heap_functionality<MinDaryHeap<std::string, int, 4>>(); }

BOOST_AUTO_TEST_CASE(MaxDaryHeap_D4_Test) { test_max_heap_functionality<MaxDaryHeap<std::string, int, 4>>(); }

BOOST_AUTO_TEST_CASE(DaryHeap_D4_StressTest) { stress_test_heap<MinDaryHeap<std::string, int, 4>>(); }

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(HeapPerformanceTests)

BOOST_AUTO_TEST_CASE(HeapPerformanceComparison) {
    const size_t num_items = 10000;
    const size_t num_updates = 5000;
    const size_t num_random_ops = 40000;

    run_performance_test<MinPairingHeap<std::string, int>>("Pairing Heap", num_items, num_updates, num_random_ops);
    run_performance_test<MinBoostFibonacciHeap<std::string, int>>("Boost Fibonacci Heap", num_items, num_updates, num_random_ops);
    run_performance_test<MinStdSetHeap<std::string, int>>("std::set", num_items, num_updates, num_random_ops);
    run_performance_test<MinDaryHeap<std::string, int, 2>>("Binary Heap (d=2)", num_items, num_updates, num_random_ops);
    run_performance_test<MinDaryHeap<std::string, int, 4>>("4-ary Heap (d=4)", num_items, num_updates, num_random_ops);
    run_performance_test<MinDaryHeap<std::string, int, 8>>("8-ary Heap (d=8)", num_items, num_updates, num_random_ops);
}

BOOST_AUTO_TEST_SUITE_END()

}    // namespace osp::test
