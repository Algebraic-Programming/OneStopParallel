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

#include "scheduler/Partitioners/partitioners.hpp"

bool is_power_of(unsigned num, const unsigned pow) {
    if (num==0) return false;
    while (num != 1) {
        if (num%pow != 0) return false;
        num /= pow;
    }
    return true;
}

float calculate_imbalance(const unsigned num_parts, const std::multiset<int, std::greater<int>>& weights, const std::vector<unsigned>& allocation) {
    assert( weights.size() ==  allocation.size() );
    std::vector<int> bins(num_parts, 0);

    int i = 0;
    for (auto& wt : weights) {
        bins[allocation[i]] += wt;
        i++;
    }

    int avg_weight = 0;
    int avg_weight_remainder = 0;
    int max_weight = 0;

    for (auto& bin_wt : bins) {
        avg_weight_remainder += bin_wt%num_parts;
        avg_weight += bin_wt/num_parts;
        max_weight = std::max(max_weight, bin_wt);
    }

    if (max_weight == 0) {
        return 1;
    }
    float imbalance = float(max_weight)/(avg_weight+ (float(avg_weight_remainder)/float(num_parts)));

    return imbalance;
}