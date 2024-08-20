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

KK_Tracking KK_Tracking::take_difference( const KK_Tracking& first, const KK_Tracking& second ) {
    const KK_Tracking& larger = ( first > second ) ? first : second;
    const KK_Tracking& smaller = ( first > second ) ? second : first;

    int new_weight = larger.weight-smaller.weight;
    std::vector<unsigned> new_positive = larger.positive;
    std::vector<unsigned> new_negative = larger.negative;

    new_positive.insert(new_positive.cend(), smaller.negative.cbegin(), smaller.negative.cend());
    new_negative.insert(new_negative.cend(), smaller.positive.cbegin(), smaller.positive.cend());

    return KK_Tracking(new_weight, new_positive, new_negative);
}

std::vector<unsigned> kk_partitioner_2(const std::multiset<int, std::greater<int>>& weights) {
    if (weights.size() == 0) return {};

    std::multiset< KK_Tracking, KK_Tracking::Comparator> difference_queue;

    unsigned wt_ind = 0;
    for (auto& wt: weights) {
        difference_queue.emplace(wt, wt_ind);
        wt_ind++;
    }

    while ( difference_queue.size() > 1 ) {
        auto KK_it = difference_queue.begin();
        KK_Tracking largest = *KK_it;
        KK_it++;
        KK_Tracking second_largest = *KK_it;

        KK_Tracking difference = KK_Tracking::take_difference(largest, second_largest);

        difference_queue.erase(difference_queue.begin());
        difference_queue.erase(difference_queue.begin());

        difference_queue.insert(difference);
    }

    std::vector<unsigned> output(weights.size(),0);
    for ( auto& ind : difference_queue.cbegin()->positive ) {
        output[ind] = 1;
    }

    return output;
};


std::vector<unsigned> kk_partitioner(const  unsigned num_parts, const std::multiset<int, std::greater<int>>& weights) {
    const unsigned pow =2;
    if( ! is_power_of(num_parts, pow) ) {
        throw std::logic_error( "Recursive Karmarkar-Karp only powers of two" );
    }

    std::vector<unsigned> output;

    if (num_parts == 1) return std::vector<unsigned>(weights.size(),0);
    else if (num_parts == 2) {
        return kk_partitioner_2(weights);
    }
    else {
        unsigned num_parts_half = num_parts/2;

        const std::vector<unsigned> initial_split = kk_partitioner_2( weights);
        output = initial_split;

        std::vector<std::multiset<int, std::greater<int>>> weights_temp(2);
        unsigned wt_ind = 0;
        for (auto& wt : weights) {
            weights_temp[ initial_split[wt_ind] ].emplace(wt);
            wt_ind ++;
        }

        const std::vector<unsigned> recursive_split_0 = kk_partitioner( num_parts_half, weights_temp[0] );
        const std::vector<unsigned> recursive_split_1 = kk_partitioner( num_parts_half, weights_temp[1] );

        unsigned wt_ind_0 = 0;
        unsigned wt_ind_1 = 0;
        for (auto iter = weights.begin(); iter != weights.cend(); iter++) {
            output[ wt_ind_0 + wt_ind_1 ] *= num_parts_half;

            if ( initial_split[ wt_ind_0+wt_ind_1 ] == 0 ) {
                output[ wt_ind_0 + wt_ind_1 ] += recursive_split_0[wt_ind_0] ;
                wt_ind_0++;
            }
            else if ( initial_split[ wt_ind_0+wt_ind_1 ] ) {
                output[ wt_ind_0 + wt_ind_1 ] += recursive_split_1[wt_ind_1] ;
                wt_ind_1++;
            }
            else {
                throw std::runtime_error("KK_algorithm_failed.");
            }
        }
        
    }

    return output;
};