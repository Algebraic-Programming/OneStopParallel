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

#include <vector>
#include <set>
#include <iostream>

#include "algorithms/Partitioners/partitioners.hpp"

std::vector<unsigned> greedy_partitioner(const unsigned num_parts, const std::multiset<int, std::greater<int>>& weights) {
    std::vector<unsigned> allocation;
    allocation.reserve(weights.size());

    // std::vector<int> weights_of_bins(num_parts,0);

    // std::vector<unsigned> bins(num_parts);
    // std::iota(bins.begin(), bins.end(), 0);

    // auto cmp = [weights_of_bins](unsigned x, unsigned y){ return (weights_of_bins[x] < weights_of_bins[y]) || ((weights_of_bins[x] == weights_of_bins[y]) && ( x < y ) ) ;};
    // std::multiset<unsigned, decltype(cmp)> bins_sorted_increasingly(bins.cbegin(), bins.cend(), cmp);

    std::multiset<weighted_bin> bins_sorted_increasingly;
    for (unsigned i = 0 ; i < num_parts; i++) {
        bins_sorted_increasingly.emplace(i);
    }

    // std::cout << std::endl;
    for (auto& wt : weights) {
        weighted_bin smallest_bin = *bins_sorted_increasingly.begin();
        // std::cout << "id: " << smallest_bin.id << ", weight: " << smallest_bin.weight << ", + " << wt << std::endl;
        bins_sorted_increasingly.erase( bins_sorted_increasingly.begin() );
        allocation.emplace_back(smallest_bin.id);
        smallest_bin += wt;
        bins_sorted_increasingly.insert( smallest_bin );
    }

    return allocation;
};