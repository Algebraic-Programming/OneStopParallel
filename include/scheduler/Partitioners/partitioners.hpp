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
#include <random>
#include <vector>
#include <set>
#include <numeric>
#include <limits.h>
#include <cassert>
#include <stdexcept>

#include "auxiliary/auxiliary.hpp"

bool is_power_of(unsigned num, const unsigned pow);


struct KK_Tracking {
    int weight;
    std::vector<unsigned> positive;
    std::vector<unsigned> negative;

    KK_Tracking( const int& weight_, const unsigned& index  ) : weight(weight_), positive({index}), negative({}) {};

    KK_Tracking( const int& weight_, const std::vector<unsigned>& positive_, const std::vector<unsigned>& negative_  )
        : weight(weight_), positive(positive_), negative(negative_) {};

    static KK_Tracking take_difference( const KK_Tracking& first, const KK_Tracking& second );

    constexpr bool operator>(const KK_Tracking &second) const { return (this->weight > second.weight); };

    struct Comparator {
        constexpr bool operator()(const KK_Tracking &a, const KK_Tracking &b) const { return (a.weight > b.weight); };
    };
};




enum PartitionAlgorithm { Greedy, KarmarkarKarp, BinPacking, ILP };

float calculate_imbalance(const unsigned num_parts, const std::multiset<int, std::greater<int>>& weights, const std::vector<unsigned>& allocation);

struct weighted_bin {
    const unsigned id;
    int weight;

    weighted_bin(const unsigned id_) : id(id_), weight(0) {};

    weighted_bin operator+=(int wt) { weight+=wt; return *this; } 

    bool operator<(const weighted_bin& other) const { return (weight < other.weight) || ((weight == other.weight) && (id < other.id)) ;}
    // friend bool operator<(const weighted_bin& lhs, const weighted_bin& rhs) { return (lhs.weight < rhs.weight) || ((lhs.weight == rhs.weight) && (lhs.id < rhs.id)) ;}
};

std::vector<unsigned> greedy_partitioner(const unsigned num_parts, const std::multiset<int, std::greater<int>>& weights);

std::vector<unsigned> kk_partitioner_2(const std::multiset<int, std::greater<int>>& weights);

std::vector<unsigned> kk_partitioner(const unsigned num_parts, const std::multiset<int, std::greater<int>>& weights);

// TODO implement
std::vector<unsigned> binpacking_partitioner(const unsigned num_parts, const std::multiset<int, std::greater<int>>& weights);


std::pair<float, std::vector<unsigned>> hill_climb_weight_balance_single_superstep(const int runs, const unsigned num_parts, const std::multiset<int, std::greater<int>>& weights, std::vector<unsigned> allocation);