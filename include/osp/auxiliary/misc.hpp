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
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace osp {

// unbiased random int generator
inline int RandInt(int lim) {
    int rnd = std::rand();
    while (rnd >= RAND_MAX - RAND_MAX % lim) {
        rnd = std::rand();
    }

    return rnd % lim;
}

// pair of integers
template <typename T1, typename T2>
struct Pair {
    int a_, b_;

    explicit Pair(const T1 a = T1(), const T2 b = T2()) : a_(a), b_(b) {}

    bool operator<(const Pair<T1, T2> &other) const {
        return (a_ < other.a_ || (a_ == other.a_ && b_ < other.b_));
    }

    std::ostream &operator<<(std::ostream &os) const {
        return os << ("(" + std::to_string(a_) + ", " + std::to_string(b_) + ")");
    }
};

using IntPair = Pair<int, int>;

// triple of integers
template <typename T1, typename T2, typename T3>
struct Triple {
    T1 a_;
    T2 b_;
    T3 c_;

    explicit Triple(const T1 a = T1(), const int b = T2(), const int c = T3()) : a_(a), b_(b), c_(c) {}

    std::ostream &operator<<(std::ostream &os) const {
        return os << "(" << std::to_string(a_) << ", " << std::to_string(b_) << ", " << std::to_string(c_) << ")";
    }
};

using IntTriple = Triple<int, int, int>;

inline bool IsDisjoint(std::vector<IntPair> &intervals) {
    sort(intervals.begin(), intervals.end());
    for (size_t i = 0; i + 1 < intervals.size(); ++i) {
        if (intervals[i].b_ > intervals[i + 1].a_) {
            return false;
        }
    }

    return true;
}

// computes power of an integer
template <typename T>
constexpr T Intpow(T base, unsigned exp) {
    static_assert(std::is_integral<T>::value);

    if (exp == 0U) {
        return 1;
    }
    if (exp == 1U) {
        return base;
    }

    T tmp = Intpow(base, exp / 2U);
    if (exp % 2U == 0U) {
        return tmp * tmp;
    }
    return base * tmp * tmp;
}

struct ContractionEdge {
    IntPair edge_;
    int nodeW_;
    int edgeW_;

    ContractionEdge(const int from, const int to, const int wnode, const int wedge)
        : edge_(from, to), nodeW_(wnode), edgeW_(wedge) {}

    bool operator<(const ContractionEdge &other) const {
        return (nodeW_ < other.nodeW_ || (nodeW_ == other.nodeW_ && edgeW_ < other.edgeW_));
    }
};

// List of initializaton methods available
static const std::vector<std::string> possibleModes{
    "random", "SJF", "cilk", "BSPg", "ETF", "BL-EST", "ETF-NUMA", "BL-EST-NUMA", "Layers"};

// modify problem filename by adding substring at the right place
inline std::string EditFilename(const std::string &filename, const std::string &toInsert) {
    auto pos = filename.find("_coarse");
    if (pos == std::string::npos) {
        pos = filename.find("_instance");
    }
    if (pos == std::string::npos) {
        return toInsert + filename;
    }

    return filename.substr(0, pos) + toInsert + filename.substr(pos, filename.length() - pos);
}

// unordered set intersection
template <typename T>
std::unordered_set<T> GetIntersection(const std::unordered_set<T> &a, const std::unordered_set<T> &b) {
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
template <typename T>
std::unordered_set<T> GetUnion(const std::unordered_set<T> &a, const std::unordered_set<T> &b) {
    std::unordered_set<T> larger = a.size() > b.size() ? a : b;
    std::unordered_set<T> smaller = a.size() <= b.size() ? a : b;
    for (auto &elem : smaller) {
        larger.emplace(elem);
    }
    return larger;
}

// zip two vectors of equal length
template <typename S, typename T>
std::vector<std::pair<S, T>> Zip(const std::vector<S> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());

    std::vector<std::pair<S, T>> result;
    result.resize(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = std::make_pair(a[i], b[i]);
    }

    return result;
}

template <typename S, typename T>
void Unzip(std::vector<std::pair<S, T>> &zipped, std::vector<S> &a, std::vector<T> &b) {
    a.resize(zipped.size());
    b.resize(zipped.size());

    for (size_t i = 0; i < zipped.size(); i++) {
        a[i] = zipped[i].first;
        b[i] = zipped[i].second;
    }
}

template <typename T>
std::vector<size_t> SortAndSortingArrangement(std::vector<T> &a) {
    std::vector<size_t> rearrangement;
    rearrangement.resize(a.size());
    std::iota(rearrangement.begin(), rearrangement.end(), 0);

    std::vector<std::pair<T, size_t>> zipped = zip(a, rearrangement);
    std::sort(zipped.begin(), zipped.end());

    unzip(zipped, a, rearrangement);

    return rearrangement;
}

template <typename T, typename RetT = size_t>
std::vector<RetT> SortingArrangement(const std::vector<T> &a, bool increasing = true) {
    std::vector<RetT> rearrangement;
    rearrangement.resize(a.size());
    std::iota(rearrangement.begin(), rearrangement.end(), 0);

    std::vector<std::pair<T, RetT>> zipped = zip(a, rearrangement);
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
inline bool CheckVectorIsRearrangementOf0ToN(const std::vector<size_t> &a) {
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
template <typename T>
void SortLikeArrangement(std::vector<T> &a, const std::vector<size_t> &arrangement) {
    assert(a.size() == arrangement.size());
    assert(CheckVectorIsRearrangementOf0ToN(arrangement));

    std::vector<bool> moved(a.size(), false);
    for (size_t i = 0; i < a.size(); i++) {
        if (moved[i]) {
            continue;
        }
        T iVal = a[i];
        size_t prevJ = i;
        size_t j = arrangement[i];
        while (i != j) {
            a[prevJ] = a[j];
            moved[prevJ] = true;
            prevJ = j;
            j = arrangement[j];
        }
        a[prevJ] = iVal;    // j == i
        moved[prevJ] = true;
    }
}

// sorts vector according to values in second vector w/o changing second vector
template <typename S, typename T>
void SortLike(std::vector<S> &a, const std::vector<T> &b) {
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
template <class SetType, typename T = typename SetType::key_type>
T GetMedian(SetType orderedSet) {
    assert(orderedSet.size() != 0);
    typename SetType::iterator it = orderedSet.begin();
    if (orderedSet.size() % 2 == 1) {
        std::advance(it, orderedSet.size() / 2);
        return *it;
    } else {
        std::advance(it, (orderedSet.size() - 1) / 2);
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
template <class SetType, typename T = typename SetType::key_type>
T GetLowerMedian(SetType orderedSet) {
    assert(orderedSet.size() != 0);
    typename SetType::iterator it = orderedSet.begin();

    std::advance(it, (orderedSet.size() - 1) / 2);
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
template <class SetType, typename T = typename SetType::key_type>
T GetUpperThirdPercentile(SetType orderedSet) {
    assert(orderedSet.size() != 0);
    typename SetType::iterator it = orderedSet.begin();

    std::advance(it, (orderedSet.size() / 3) + ((orderedSet.size() + 1) / 3));
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
template <class SetType, typename T = typename SetType::key_type>
T GetLowerThirdPercentile(SetType orderedSet) {
    assert(orderedSet.size() != 0);
    typename SetType::iterator it = orderedSet.begin();

    std::advance(it, (orderedSet.size() / 3));
    return *it;
}

}    // namespace osp
