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

#include "algorithms/Partitioners/partitioners.hpp"

std::pair<float, std::vector<unsigned>> hill_climb_weight_balance_single_superstep(const int runs, const unsigned num_parts, const std::multiset<int, std::greater<int>>& weights, std::vector<unsigned> allocation) {
    assert( weights.size() ==  allocation.size() );
    assert( num_parts != 0);
    if (num_parts == 1 ) {
        std::vector<unsigned> new_allocation(weights.size(), 0);
        return {1, new_allocation};
    }

    std::vector<int> weight_vec(weights.size());
    std::vector<int> bins(num_parts, 0);
    std::vector<std::set<int, std::less<int>>> wt_indices_inside_bins(num_parts); //smaller index means larger weight

    int i = 0;
    for (auto& wt : weights) {
        bins[allocation[i]] += wt;
        weight_vec[i] = wt;
        wt_indices_inside_bins[allocation[i]].emplace(i);
        i++;
    }

    // std::cout << "Bin weights: ";
    // for (auto& bin_wt : bins) {
    //     std::cout << bin_wt << " ";
    // }
    // std::cout << std::endl;

    bool change = true;
    int no_change_counter = 0;
    int no_change_max = std::max( int(sqrt(runs)), 20 );

    for (int j = 0; (j < runs); j++ ) {
        if (no_change_counter > no_change_max) break;
        change = false;
    
        unsigned max_bin = 0;
        int max_weight = INT_MIN;
        unsigned min_bin = 0;
        int min_weight = INT_MAX;

        int bin_ind = 0;
        for (auto& bin_wt : bins) {
            if (bin_wt > max_weight) {
                max_weight = bin_wt;
                max_bin = bin_ind;
            }
            if (bin_wt <= min_weight) {
                min_weight = bin_wt;
                min_bin = bin_ind;
            }
            bin_ind++;
        }
        assert( bins[max_bin] == max_weight );
        assert( bins[min_bin] == min_weight );
        if (wt_indices_inside_bins[max_bin].size() <= 1) break;
        
        int max_min_diff = bins[max_bin] - bins[min_bin];

        auto it = wt_indices_inside_bins[max_bin].end();
        std::advance(it,-1);
        while( it != wt_indices_inside_bins[max_bin].cbegin() && *it == 0 ) {
            std::advance(it,-1);
        }
        int ind_min_wt_max_bin = *it;

        // checking exchanges with min_bin
        std::vector<int> exchange_indices;
        int exchange_sum = 0;
        auto min_bin_it = wt_indices_inside_bins[min_bin].begin();
        while ((weight_vec[ind_min_wt_max_bin] - exchange_sum > 0) ) {
            if (max_min_diff > weight_vec[ind_min_wt_max_bin] - exchange_sum) {
                // reassigning min weight of max bin to min bin
                assert( allocation[ind_min_wt_max_bin] == max_bin );
                bins[max_bin] -= weight_vec[ind_min_wt_max_bin] ;
                wt_indices_inside_bins[max_bin].erase(ind_min_wt_max_bin);
                allocation[ind_min_wt_max_bin] = min_bin;
                bins[min_bin] += weight_vec[ind_min_wt_max_bin];
                wt_indices_inside_bins[min_bin].emplace(ind_min_wt_max_bin);
                // reassigning nodes of min bin to max bin
                for (auto& ind : exchange_indices) {
                    assert( allocation[ind] == min_bin );
                    bins[min_bin] -= weight_vec[ind];
                    wt_indices_inside_bins[min_bin].erase(ind);
                    allocation[ind] = max_bin;
                    bins[max_bin] += weight_vec[ind];
                    wt_indices_inside_bins[max_bin].emplace(ind);
                }
                change = true;
                no_change_counter = 0;
                break;
            }

            if (min_bin_it == wt_indices_inside_bins[min_bin].cend()) break;

            bool added = false;
            while ( (min_bin_it != wt_indices_inside_bins[min_bin].cend()) && (!added) ) {
                if (weight_vec[ind_min_wt_max_bin] - exchange_sum - weight_vec[*min_bin_it] > 0) {
                    exchange_indices.emplace_back(*min_bin_it);
                    exchange_sum += weight_vec[*min_bin_it];
                    added = true;
                }
                std::advance(min_bin_it,1);
            }
        }
        
        if (change) continue;

        // checking random exchanges
        exchange_indices.clear();
        exchange_sum = 0;
        unsigned exchange_bin = randInt(num_parts-1);
        if (exchange_bin >= max_bin) exchange_bin++;
        assert( exchange_bin != max_bin );
        assert( 0 <= exchange_bin );
        assert( exchange_bin < num_parts );
        std::set<int, std::less<int>> remaining_wt_indices_in_exchange_bin = wt_indices_inside_bins[exchange_bin];

        it = wt_indices_inside_bins[max_bin].begin();
        int shift = randInt(wt_indices_inside_bins[max_bin].size());
        std::advance(it, shift );
        while(*it == 0 && shift != 0) {
            it = wt_indices_inside_bins[max_bin].begin();
            shift = randInt(shift);
            std::advance(it, shift );
        }
        int ind_of_rand_weight_in_max_bin = *it;

        int bin_size_diff = bins[max_bin]-bins[exchange_bin];

        while ((weight_vec[ind_of_rand_weight_in_max_bin] - exchange_sum > 0) ) {
            if (bin_size_diff > weight_vec[ind_of_rand_weight_in_max_bin] - exchange_sum) {
                // reassigning weight of max bin to exchange bin
                assert( allocation[ind_of_rand_weight_in_max_bin] == max_bin );
                bins[max_bin] -= weight_vec[ind_of_rand_weight_in_max_bin] ;
                wt_indices_inside_bins[max_bin].erase(ind_of_rand_weight_in_max_bin);
                allocation[ind_of_rand_weight_in_max_bin] = exchange_bin;
                bins[exchange_bin] += weight_vec[ind_of_rand_weight_in_max_bin];
                wt_indices_inside_bins[exchange_bin].emplace(ind_of_rand_weight_in_max_bin);
                // reassigning nodes of exchange bin to max bin
                for (auto& ind : exchange_indices) {
                    assert( allocation[ind] == exchange_bin );
                    bins[exchange_bin] -= weight_vec[ind];
                    wt_indices_inside_bins[exchange_bin].erase(ind);
                    allocation[ind] = max_bin;
                    bins[max_bin] += weight_vec[ind];
                    wt_indices_inside_bins[max_bin].emplace(ind);
                }
                change = true;
                no_change_counter = 0;
                break;
            }

            if ( remaining_wt_indices_in_exchange_bin.empty() ) break;


            bool added = false;
            while ( (! remaining_wt_indices_in_exchange_bin.empty()) && (!added) ) {
                auto remain_it = remaining_wt_indices_in_exchange_bin.begin();
                std::advance(remain_it, randInt(remaining_wt_indices_in_exchange_bin.size()));

                if (weight_vec[ind_of_rand_weight_in_max_bin] - exchange_sum - weight_vec[*remain_it] > 0) {
                    exchange_indices.emplace_back(*remain_it);
                    exchange_sum += weight_vec[*remain_it];
                    added = true;
                }
                remaining_wt_indices_in_exchange_bin.erase(remain_it);
            }
        }
        if (change) continue;
        no_change_counter++;
    }

    int max_weight = 0;
    int avg_weight = 0;
    int avg_weight_remainder = 0;

    for (auto& bin_wt : bins) {
        avg_weight_remainder += bin_wt%num_parts;
        avg_weight += bin_wt/num_parts;
    }

    for (auto& bin_wt : bins) {
        max_weight = std::max(max_weight, bin_wt);
    }

    // std::cout << "Bin weights: ";
    // for (auto& bin_wt : bins) {
    //     std::cout << bin_wt << " ";
    // }
    // std::cout << std::endl;


    if (max_weight == 0 ) {
        return {1, allocation};
    }
    float imbalance = float(max_weight)/(avg_weight+ (float(avg_weight_remainder)/float(num_parts)));

    return {imbalance, allocation};
}