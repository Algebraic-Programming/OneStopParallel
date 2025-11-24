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
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace osp {

// unbiased random int generator
inline int randInt(int lim) {
    int rnd = std::rand();
    while (rnd >= RAND_MAX - RAND_MAX % lim)
        rnd = std::rand();

    return rnd % lim;
}

// pair of integers
template<typename T1, typename T2>
struct Pair {
    int a, b;

    explicit Pair(const T1 a_ = T1(), const T2 b_ = T2()) : a(a_), b(b_) {}

    template<typename T1_, typename T2_>
    bool operator<(const Pair<T1_, T2_> &other) const {
        return (a < other.a || (a == other.a && b < other.b));
    }

    std::ostream &operator<<(std::ostream &os) const {
        return os << ("(" + std::to_string(a) + ", " + std::to_string(b) + ")");
    }
};
using intPair = Pair<int, int>;

// triple of integers
template<typename T1, typename T2, typename T3>
struct Triple {
    T1 a;
    T2 b;
    T3 c;

    explicit Triple(const T1 a_ = T1(), const int b_ = T2(), const int c_ = T3()) : a(a_), b(b_), c(c_) {}

    std::ostream &operator<<(std::ostream &os) const {
        return os << "(" << std::to_string(a) << ", " << std::to_string(b) << ", " << std::to_string(c) << ")";
    }
};
using intTriple = Triple<int, int, int>;

inline bool isDisjoint(std::vector<intPair> &intervals) {

    sort(intervals.begin(), intervals.end());
    for (size_t i = 0; i + 1 < intervals.size(); ++i)
        if (intervals[i].b > intervals[i + 1].a)
            return false;

    return true;
}

// computes power of an integer
template<typename T>
constexpr T intpow(T base, unsigned exp) {
    static_assert(std::is_integral<T>::value);

    if (exp == 0U) {
        return 1;
    }
    if (exp == 1U) {
        return base;
    }

    T tmp = intpow(base, exp / 2U);
    if (exp % 2U == 0U) {
        return tmp * tmp;
    }
    return base * tmp * tmp;
}

struct contractionEdge {
    intPair edge;
    int nodeW;
    int edgeW;

    contractionEdge(const int from, const int to, const int Wnode, const int Wedge)
        : edge(from, to), nodeW(Wnode), edgeW(Wedge) {}

    bool operator<(const contractionEdge &other) const {
        return (nodeW < other.nodeW || (nodeW == other.nodeW && edgeW < other.edgeW));
    }
};



// List of initializaton methods available
static const std::vector<std::string> possibleModes{"random", "SJF",      "cilk",        "BSPg",  "ETF",
                                                    "BL-EST", "ETF-NUMA", "BL-EST-NUMA", "Layers"};

// modify problem filename by adding substring at the right place
inline std::string editFilename(const std::string &filename, const std::string &toInsert) {
    auto pos = filename.find("_coarse");
    if (pos == std::string::npos)
        pos = filename.find("_instance");
    if (pos == std::string::npos)
        return toInsert + filename;

    return filename.substr(0, pos) + toInsert + filename.substr(pos, filename.length() - pos);
}


// unordered set intersection
template<typename T>
std::unordered_set<T> get_intersection(const std::unordered_set<T> &a, const std::unordered_set<T> &b) {
    std::vector<T> result;
    const auto &larger = a.size() > b.size() ? a : b;
    const auto &smaller = a.size() <= b.size() ? a : b;
    for (const auto &each : smaller) {
        if (larger.find(each) != larger.end()) {
            result.emplace_back(each);
        }
    }
    return {result.begin(), result.end()};
}

// unordered set union
template<typename T>
std::unordered_set<T> get_union(const std::unordered_set<T> &a, const std::unordered_set<T> &b) {
    std::unordered_set<T> larger = a.size() > b.size() ? a : b;
    std::unordered_set<T> smaller = a.size() <= b.size() ? a : b;
    for (auto &elem : smaller) {
        larger.emplace(elem);
    }
    return larger;
}

// zip two vectors of equal length
template<typename S, typename T>
std::vector<std::pair<S, T>> zip(const std::vector<S> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());

    std::vector<std::pair<S, T>> result;
    result.resize(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = std::make_pair(a[i], b[i]);
    }

    return result;
}

template<typename S, typename T>
void unzip(std::vector<std::pair<S, T>> &zipped, std::vector<S> &a, std::vector<T> &b) {
    a.resize(zipped.size());
    b.resize(zipped.size());

    for (size_t i = 0; i < zipped.size(); i++) {
        a[i] = zipped[i].first;
        b[i] = zipped[i].second;
    }
}

template<typename T>
std::vector<size_t> sort_and_sorting_arrangement(std::vector<T> &a) {
    std::vector<size_t> rearrangement;
    rearrangement.resize(a.size());
    std::iota(rearrangement.begin(), rearrangement.end(), 0);

    std::vector<std::pair<T, size_t>> zipped = zip(a, rearrangement);
    std::sort(zipped.begin(), zipped.end());

    unzip(zipped, a, rearrangement);

    return rearrangement;
}

template<typename T, typename retT = size_t>
std::vector<retT> sorting_arrangement(const std::vector<T> &a, bool increasing = true) {
    std::vector<retT> rearrangement;
    rearrangement.resize(a.size());
    std::iota(rearrangement.begin(), rearrangement.end(), 0);

    std::vector<std::pair<T, retT>> zipped = zip(a, rearrangement);
    std::sort(zipped.begin(), zipped.end());
    if (!increasing) {
        std::reverse(zipped.begin(), zipped.end());
    }

    for (size_t i = 0; i < rearrangement.size(); i++) {
        rearrangement[i] = zipped[i].second;
    }

    return rearrangement;
}

// checks if a vector is rearrangement of 0... N-1
inline bool check_vector_is_rearrangement_of_0_to_N(const std::vector<size_t> &a) {
    std::vector<bool> contained(a.size(), false);
    for (auto &val : a) {
        if (val >= a.size()) {
            return false;
        } else if (contained[val]) {
            return false;
        } else {
            contained[val] = true;
        }
    }
    return true;
}

// sorts a vector like the arrangement
template<typename T>
void sort_like_arrangement(std::vector<T> &a, const std::vector<size_t> &arrangement) {
    assert(a.size() == arrangement.size());
    assert(check_vector_is_rearrangement_of_0_to_N(arrangement));

    std::vector<bool> moved(a.size(), false);
    for (size_t i = 0; i < a.size(); i++) {
        if (moved[i]) {
            continue;
        }
        T i_val = a[i];
        size_t prev_j = i;
        size_t j = arrangement[i];
        while (i != j) {
            a[prev_j] = a[j];
            moved[prev_j] = true;
            prev_j = j;
            j = arrangement[j];
        }
        a[prev_j] = i_val; // j == i
        moved[prev_j] = true;
    }
}

// sorts vector according to values in second vector w/o changing second vector
template<typename S, typename T>
void sort_like(std::vector<S> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());

    std::vector<size_t> arrangement = sorting_arrangement(b);
    sort_like_arrangement(a, arrangement);
}

/**
 * @brief Get median of a sorted set or multiset
 *
 * @tparam SetType
 * @tparam SetType::key_type
 * @param ordered_set
 * @return T KeyType of SetType
 */
template<class SetType, typename T = typename SetType::key_type>
T Get_Median(SetType ordered_set) {
    assert(ordered_set.size() != 0);
    typename SetType::iterator it = ordered_set.begin();
    if (ordered_set.size() % 2 == 1) {
        std::advance(it, ordered_set.size() / 2);
        return *it;
    } else {
        std::advance(it, (ordered_set.size() - 1) / 2);
        T val1 = *it;
        T val2 = *(++it);
        return val1 + (val2 - val1) / 2;
    }
}

/**
 * @brief Get lower_median of a sorted set or multiset
 *
 * @tparam SetType
 * @tparam SetType::key_type
 * @param ordered_set
 * @return T KeyType of SetType
 */
template<class SetType, typename T = typename SetType::key_type>
T Get_Lower_Median(SetType ordered_set) {
    assert(ordered_set.size() != 0);
    typename SetType::iterator it = ordered_set.begin();

    std::advance(it, (ordered_set.size() - 1) / 2);
    return *it;
}

/**
 * @brief Get top third percentile of a sorted set or multiset
 *
 * @tparam SetType
 * @tparam SetType::key_type
 * @param ordered_set
 * @return T KeyType of SetType
 */
template<class SetType, typename T = typename SetType::key_type>
T Get_upper_third_percentile(SetType ordered_set) {
    assert(ordered_set.size() != 0);
    typename SetType::iterator it = ordered_set.begin();

    std::advance(it, (ordered_set.size() / 3) + ((ordered_set.size() + 1) / 3));
    return *it;
}

/**
 * @brief Get lower third percentile of a sorted set or multiset
 *
 * @tparam SetType
 * @tparam SetType::key_type
 * @param ordered_set
 * @return T KeyType of SetType
 */
template<class SetType, typename T = typename SetType::key_type>
T Get_lower_third_percentile(SetType ordered_set) {
    assert(ordered_set.size() != 0);
    typename SetType::iterator it = ordered_set.begin();

    std::advance(it, (ordered_set.size() / 3));
    return *it;
}

} // namespace osp