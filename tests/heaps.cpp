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
template <typename Key, typename Value, bool isMinHeap = true>
class BoostFibonacciHeapWrapper {
  private:
    struct Node {
        Key key_;
        Value value_;
    };

    struct NodeCompare {
        bool operator()(const Node &a, const Node &b) const {
            if constexpr (isMinHeap) {
                return a.value_ > b.value_;    // For min-heap
            } else {
                return a.value_ < b.value_;    // For max-heap
            }
        }
    };

    using BoostHeap = boost::heap::fibonacci_heap<Node, boost::heap::compare<NodeCompare>>;
    using HandleType = typename BoostHeap::handle_type;

    BoostHeap heap_;
    std::unordered_map<Key, HandleType> handles_;

  public:
    BoostFibonacciHeapWrapper() = default;

    bool IsEmpty() const { return heap_.empty(); }

    size_t size() const { return heap_.size(); }

    bool Contains(const Key &key) const { return handles_.count(key); }

    const Key &Top() const {
        if (IsEmpty()) {
            throw std::out_of_range("Heap is empty");
        }
        return heap_.top().key_;
    }

    Key Pop() {
        if (IsEmpty()) {
            throw std::out_of_range("Heap is empty");
        }
        Key topKey = heap_.top().key_;
        heap_.pop();
        handles_.erase(topKey);
        return topKey;
    }

    void Push(const Key &key, const Value &value) {
        if (Contains(key)) {
            throw std::invalid_argument("Key already exists");
        }
        HandleType handle = heap_.push({key, value});
        handles_[key] = handle;
    }

    Value GetValue(const Key &key) const {
        if (!Contains(key)) {
            throw std::out_of_range("Key not found");
        }
        return (*handles_.at(key)).value_;
    }

    void Update(const Key &key, const Value &newValue) {
        if (!Contains(key)) {
            throw std::invalid_argument("Key not found for Update");
        }
        HandleType handle = handles_.at(key);
        (*handle).value_ = newValue;
        heap_.update(handle);
    }

    void Erase(const Key &key) {
        if (!Contains(key)) {
            throw std::invalid_argument("Key not found for Erase");
        }
        heap_.erase(handles_.at(key));
        handles_.erase(key);
    }

    void Clear() {
        heap_.clear();
        handles_.clear();
    }
};

template <typename Key, typename Value>
using MinBoostFibonacciHeap = BoostFibonacciHeapWrapper<Key, Value, true>;

template <typename Key, typename Value>
using MaxBoostFibonacciHeap = BoostFibonacciHeapWrapper<Key, Value, false>;

// Wrapper for std::set to match the test interface
template <typename Key, typename Value, bool isMinHeap = true>
class StdSetWrapper {
  private:
    struct NodeCompare {
        bool operator()(const std::pair<Value, Key> &a, const std::pair<Value, Key> &b) const {
            if (a.first != b.first) {
                if constexpr (isMinHeap) {
                    return a.first < b.first;    // For min-heap
                } else {
                    return a.first > b.first;    // For max-heap
                }
            }
            return a.second < b.second;    // Tie-breaking
        }
    };

    using SetType = std::set<std::pair<Value, Key>, NodeCompare>;
    SetType dataSet_;
    std::unordered_map<Key, Value> valueMap_;

  public:
    StdSetWrapper() = default;

    bool IsEmpty() const { return dataSet_.empty(); }

    size_t size() const { return dataSet_.size(); }

    bool Contains(const Key &key) const { return valueMap_.count(key); }

    const Key &Top() const {
        if (IsEmpty()) {
            throw std::out_of_range("Heap is empty");
        }
        return dataSet_.begin()->second;
    }

    Key Pop() {
        if (IsEmpty()) {
            throw std::out_of_range("Heap is empty");
        }
        auto topNode = *dataSet_.begin();
        dataSet_.erase(dataSet_.begin());
        valueMap_.erase(topNode.second);
        return topNode.second;
    }

    void Push(const Key &key, const Value &value) {
        if (Contains(key)) {
            throw std::invalid_argument("Key already exists");
        }
        dataSet_.insert({value, key});
        valueMap_[key] = value;
    }

    Value GetValue(const Key &key) const {
        if (!Contains(key)) {
            throw std::out_of_range("Key not found");
        }
        return valueMap_.at(key);
    }

    void Update(const Key &key, const Value &newValue) {
        if (!Contains(key)) {
            throw std::invalid_argument("Key not found for Update");
        }
        Value oldValue = valueMap_.at(key);
        if (oldValue == newValue) {
            return;
        }
        dataSet_.erase({oldValue, key});
        dataSet_.insert({newValue, key});
        valueMap_[key] = newValue;
    }

    void Erase(const Key &key) {
        if (!Contains(key)) {
            throw std::invalid_argument("Key not found for Erase");
        }
        Value value = valueMap_.at(key);
        dataSet_.erase({value, key});
        valueMap_.erase(key);
    }

    void Clear() {
        dataSet_.clear();
        valueMap_.clear();
    }
};

template <typename Key, typename Value>
using MinStdSetHeap = StdSetWrapper<Key, Value, true>;

template <typename Key, typename Value>
using MaxStdSetHeap = StdSetWrapper<Key, Value, false>;

// Generic test suite for any min-heap implementation that follows the API.
template <typename HeapType>
void TestMinHeapFunctionality() {
    HeapType heap;

    // Basic properties of an empty heap
    BOOST_CHECK(heap.IsEmpty());
    BOOST_CHECK_EQUAL(heap.size(), 0);
    BOOST_CHECK(!heap.Contains("A"));
    BOOST_CHECK_THROW(heap.Top(), std::out_of_range);
    BOOST_CHECK_THROW(heap.Pop(), std::out_of_range);

    // Push elements
    heap.Push("A", 10);
    heap.Push("B", 5);
    heap.Push("C", 15);

    BOOST_CHECK(!heap.IsEmpty());
    BOOST_CHECK_EQUAL(heap.size(), 3);
    BOOST_CHECK(heap.Contains("A"));
    BOOST_CHECK(heap.Contains("B"));
    BOOST_CHECK(heap.Contains("C"));
    BOOST_CHECK(!heap.Contains("D"));

    // Check for duplicate key insertion
    BOOST_CHECK_THROW(heap.Push("A", 20), std::invalid_argument);

    // Test Top() and Pop() for min-heap
    BOOST_CHECK_EQUAL(heap.Top(), "B");
    BOOST_CHECK_EQUAL(heap.Pop(), "B");
    BOOST_CHECK_EQUAL(heap.size(), 2);
    BOOST_CHECK(!heap.Contains("B"));

    BOOST_CHECK_EQUAL(heap.Top(), "A");
    BOOST_CHECK_EQUAL(heap.Pop(), "A");

    BOOST_CHECK_EQUAL(heap.Top(), "C");
    BOOST_CHECK_EQUAL(heap.Pop(), "C");
    BOOST_CHECK(heap.IsEmpty());

    // Repopulate for Update/Erase tests
    heap.Push("A", 10);
    heap.Push("B", 5);
    heap.Push("C", 15);
    heap.Push("D", 2);
    heap.Push("E", 20);

    // Test GetValue
    BOOST_CHECK_EQUAL(heap.GetValue("A"), 10);
    BOOST_CHECK_EQUAL(heap.GetValue("D"), 2);
    BOOST_CHECK_THROW(heap.GetValue("Z"), std::out_of_range);

    // Test Update (decrease-key)
    heap.Update("B", 1);    // B: 5 -> 1. Should be new Top.
    BOOST_CHECK_EQUAL(heap.Top(), "B");
    BOOST_CHECK_EQUAL(heap.GetValue("B"), 1);

    // Test Update (increase-key)
    heap.Update("B", 25);    // B: 1 -> 25. D (2) should be new Top.
    BOOST_CHECK_EQUAL(heap.Top(), "D");
    BOOST_CHECK_EQUAL(heap.GetValue("B"), 25);

    // Test Update with same value
    heap.Update("A", 10);
    BOOST_CHECK_EQUAL(heap.GetValue("A"), 10);

    // Test Erase
    heap.Erase("D");    // Erase Top element
    BOOST_CHECK_EQUAL(heap.size(), 4);
    BOOST_CHECK(!heap.Contains("D"));
    BOOST_CHECK_EQUAL(heap.Top(), "A");    // A (10) is new Top

    heap.Erase("E");    // Erase non-Top element
    BOOST_CHECK_EQUAL(heap.size(), 3);
    BOOST_CHECK(!heap.Contains("E"));
    BOOST_CHECK_THROW(heap.Erase("Z"), std::invalid_argument);

    // Test Clear
    heap.Clear();
    BOOST_CHECK(heap.IsEmpty());
    BOOST_CHECK_EQUAL(heap.size(), 0);
}

template <typename HeapType>
void TestMaxHeapFunctionality() {
    HeapType heap;
    heap.Push("A", 10);
    heap.Push("B", 5);
    heap.Push("C", 15);

    // Test Pop order for max-heap
    BOOST_CHECK_EQUAL(heap.Top(), "C");
    heap.Pop();
    BOOST_CHECK_EQUAL(heap.Top(), "A");
    heap.Pop();
    BOOST_CHECK_EQUAL(heap.Top(), "B");
}

// Stress test with a larger number of elements
template <typename HeapType>
void StressTestHeap() {
    HeapType heap;
    const int numItems = 1000;

    for (int i = 0; i < numItems; ++i) {
        heap.Push(std::to_string(i), i);
    }
    for (int i = 0; i < numItems / 2; ++i) {
        heap.Update(std::to_string(i), i - numItems);
    }

    std::vector<int> poppedValues;
    while (!heap.IsEmpty()) {
        poppedValues.push_back(heap.GetValue(heap.Top()));
        heap.Pop();
    }

    BOOST_CHECK_EQUAL(poppedValues.size(), numItems);
    BOOST_CHECK(std::is_sorted(poppedValues.begin(), poppedValues.end()));
}

// Performance test suite for different heap workloads.
template <typename HeapType>
void RunPerformanceTest(const std::string &heapName, size_t numItems, size_t numUpdates, size_t numRandomOps) {
    std::cout << "\n--- Performance Test for " << heapName << " ---" << std::endl;

    std::vector<std::string> keys(numItems);
    std::vector<int> priorities(numItems);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(0, static_cast<int>(numItems * 10));

    for (size_t i = 0; i < numItems; ++i) {
        keys[i] = std::to_string(i);
        priorities[i] = distrib(gen);
    }

    HeapType heap;

    // Scenario 1: Bulk Insert
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numItems; ++i) {
        heap.Push(keys[i], priorities[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Bulk Insert (" << numItems << " items): " << duration.count() << " ms" << std::endl;

    // Scenario 2: Decrease Key
    std::uniform_int_distribution<size_t> keyDistrib(0, numItems - 1);
    std::uniform_int_distribution<> decDist(1, 100);
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numUpdates; ++i) {
        size_t keyIdx = keyDistrib(gen);
        int newPrio = heap.GetValue(keys[keyIdx]) - decDist(gen);
        heap.Update(keys[keyIdx], newPrio);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Decrease Key (" << numUpdates << " updates): " << duration.count() << " ms" << std::endl;

    // Scenario 3: Bulk Pop
    start = std::chrono::high_resolution_clock::now();
    while (!heap.IsEmpty()) {
        heap.Pop();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Bulk Pop (" << numItems << " items): " << duration.count() << " ms" << std::endl;

    BOOST_CHECK(heap.IsEmpty());

    // Scenario 4: Random Operations (Push, Erase, Update)
    heap.Clear();
    std::vector<std::string> presentKeys;
    presentKeys.reserve(numItems);
    std::vector<bool> keyInHeap(numItems, false);
    std::uniform_int_distribution<int> opDist(0, 2);    // 0: Push, 1: Erase, 2: Update

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numRandomOps; ++i) {
        int op = opDist(gen);
        if (op == 0 || presentKeys.empty()) {    // Push
            size_t keyIdx = keyDistrib(gen);
            if (!keyInHeap[keyIdx]) {
                heap.Push(keys[keyIdx], priorities[keyIdx]);
                presentKeys.push_back(keys[keyIdx]);
                keyInHeap[keyIdx] = true;
            }
        } else {    // Erase or Update
            std::uniform_int_distribution<size_t> presentKeyDist(0, presentKeys.size() - 1);
            size_t presentKeyVecIdx = presentKeyDist(gen);
            std::string keyToOp = presentKeys[presentKeyVecIdx];

            if (op == 1) {    // Erase a random element
                heap.Erase(keyToOp);
                keyInHeap[std::stoul(keyToOp)] = false;
                std::swap(presentKeys[presentKeyVecIdx], presentKeys.back());
                presentKeys.pop_back();
            } else {    // op == 2, Update a random element (decrease key)
                int newPrio = heap.GetValue(keyToOp) - decDist(gen);
                heap.Update(keyToOp, newPrio);
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Random Ops (" << numRandomOps << " ops of Push/Erase/Update): " << duration.count() << " ms" << std::endl;

    // Scenario 5: Mixed Workload with Re-initialization
    const size_t numOuterLoopsS5 = 500;
    const size_t numInnerLoopsS5 = 10;
    const size_t numInitialPushesS5 = 100;
    const size_t numPushesPerIterS5 = 25;
    const size_t numUpdatesPerIterS5 = 25;

    // A large pool of keys to draw from for pushes, to avoid collisions.
    const size_t keyPoolSizeS5 = numOuterLoopsS5 * (numInitialPushesS5 + numInnerLoopsS5 * numPushesPerIterS5);
    std::vector<std::string> keysS5(keyPoolSizeS5);
    std::vector<int> prioritiesS5(keyPoolSizeS5);
    for (size_t i = 0; i < keyPoolSizeS5; ++i) {
        keysS5[i] = "s5_" + std::to_string(i);
        prioritiesS5[i] = distrib(gen);
    }

    size_t keyIdxCounterS5 = 0;

    start = std::chrono::high_resolution_clock::now();

    for (size_t outerI = 0; outerI < numOuterLoopsS5; ++outerI) {
        heap.Clear();
        std::vector<std::string> presentKeysS5;
        presentKeysS5.reserve(numInitialPushesS5 + numInnerLoopsS5 * (numPushesPerIterS5 - 1));

        // Initial Push
        for (size_t i = 0; i < numInitialPushesS5; ++i) {
            const auto &key = keysS5[keyIdxCounterS5];
            heap.Push(key, prioritiesS5[keyIdxCounterS5]);
            presentKeysS5.push_back(key);
            keyIdxCounterS5++;
        }

        for (size_t innerI = 0; innerI < numInnerLoopsS5; ++innerI) {
            // 1. Pop once
            if (!heap.IsEmpty()) {
                std::string poppedKey = heap.Pop();
                // Remove from present_keys_s5 efficiently
                auto it = std::find(presentKeysS5.begin(), presentKeysS5.end(), poppedKey);
                if (it != presentKeysS5.end()) {
                    std::swap(*it, presentKeysS5.back());
                    presentKeysS5.pop_back();
                }
            }

            // 2. Push 25 keys
            for (size_t j = 0; j < numPushesPerIterS5; ++j) {
                const auto &key = keysS5[keyIdxCounterS5];
                heap.Push(key, prioritiesS5[keyIdxCounterS5]);
                presentKeysS5.push_back(key);
                keyIdxCounterS5++;
            }

            // 3. Update 25 keys
            if (!presentKeysS5.empty()) {
                std::uniform_int_distribution<size_t> presentKeyDist(0, presentKeysS5.size() - 1);
                for (size_t j = 0; j < numUpdatesPerIterS5; ++j) {
                    const auto &keyToUpdate = presentKeysS5[presentKeyDist(gen)];
                    heap.Update(keyToUpdate, heap.GetValue(keyToUpdate) - decDist(gen));
                }
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Mixed Re-Init (" << numOuterLoopsS5 << " runs of init + " << numInnerLoopsS5
              << "x(Pop/Push/Update)): " << duration.count() << " ms" << std::endl;
}
BOOST_AUTO_TEST_SUITE(heap_tests)

BOOST_AUTO_TEST_CASE(PairingHeapTest) { TestMinHeapFunctionality<MinPairingHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(MaxPairingHeapTest) { TestMaxHeapFunctionality<MaxPairingHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(PairingHeapStressTest) { StressTestHeap<MinPairingHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(BoostFibonacciHeapTest) { TestMinHeapFunctionality<MinBoostFibonacciHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(MaxBoostFibonacciHeapTest) { TestMaxHeapFunctionality<MaxBoostFibonacciHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(BoostFibonacciHeapStressTest) { StressTestHeap<MinBoostFibonacciHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(StdSetHeapTest) { TestMinHeapFunctionality<MinStdSetHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(MaxStdSetHeapTest) { TestMaxHeapFunctionality<MaxStdSetHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(StdSetHeapStressTest) { StressTestHeap<MinStdSetHeap<std::string, int>>(); }

BOOST_AUTO_TEST_CASE(DaryHeapD2Test) { TestMinHeapFunctionality<MinDaryHeap<std::string, int, 2>>(); }

BOOST_AUTO_TEST_CASE(MaxDaryHeapD2Test) { TestMaxHeapFunctionality<MaxDaryHeap<std::string, int, 2>>(); }

BOOST_AUTO_TEST_CASE(DaryHeapD2StressTest) { StressTestHeap<MinDaryHeap<std::string, int, 2>>(); }

BOOST_AUTO_TEST_CASE(DaryHeapD4Test) { TestMinHeapFunctionality<MinDaryHeap<std::string, int, 4>>(); }

BOOST_AUTO_TEST_CASE(MaxDaryHeapD4Test) { TestMaxHeapFunctionality<MaxDaryHeap<std::string, int, 4>>(); }

BOOST_AUTO_TEST_CASE(DaryHeapD4StressTest) { StressTestHeap<MinDaryHeap<std::string, int, 4>>(); }

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(heap_performance_tests)

BOOST_AUTO_TEST_CASE(HeapPerformanceComparison) {
    const size_t numItems = 10000;
    const size_t numUpdates = 5000;
    const size_t numRandomOps = 40000;

    RunPerformanceTest<MinPairingHeap<std::string, int>>("Pairing Heap", numItems, numUpdates, numRandomOps);
    RunPerformanceTest<MinBoostFibonacciHeap<std::string, int>>("Boost Fibonacci Heap", numItems, numUpdates, numRandomOps);
    RunPerformanceTest<MinStdSetHeap<std::string, int>>("std::set", numItems, numUpdates, numRandomOps);
    RunPerformanceTest<MinDaryHeap<std::string, int, 2>>("Binary Heap (d=2)", numItems, numUpdates, numRandomOps);
    RunPerformanceTest<MinDaryHeap<std::string, int, 4>>("4-ary Heap (d=4)", numItems, numUpdates, numRandomOps);
    RunPerformanceTest<MinDaryHeap<std::string, int, 8>>("8-ary Heap (d=8)", numItems, numUpdates, numRandomOps);
}

BOOST_AUTO_TEST_SUITE_END()

}    // namespace osp::test
