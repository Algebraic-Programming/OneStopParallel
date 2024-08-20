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

#include "scheduler/HDagg/HDagg_simple.hpp"

std::pair<RETURN_STATUS, BspSchedule> HDagg_simple::computeSchedule(const BspInstance &instance) {

    BspSchedule sched(instance);
    
    std::vector<unsigned> top_dist = instance.getComputationalDag().get_top_node_distance();
    unsigned min_top_dist = UINT_MAX;
    unsigned max_top_dist = 0;
    for (auto& dist : top_dist) {
        min_top_dist = std::min(min_top_dist, dist);
        max_top_dist = std::max(max_top_dist, dist);
    }

    std::vector<std::vector<size_t>> level_sets( max_top_dist-min_top_dist+1, std::vector<size_t>({}) );
    for (size_t i = 0; i < top_dist.size(); i++) {
        level_sets[ top_dist[i] - min_top_dist ].emplace_back(i);
    }

    unsigned curr_superstep = 0;
    for (size_t curr_level = 0; curr_level < level_sets.size(); curr_level++) {
        Union_Find_Universe<size_t> uf_universe;

        for (const auto& node : level_sets[curr_level]) {
            uf_universe.add_object(node, instance.getComputationalDag().nodeWorkWeight(node));
        }

        std::vector<std::pair<std::vector<size_t>, unsigned>> components_and_weights = uf_universe.get_connected_components_and_weights();
        std::sort(components_and_weights.begin(), components_and_weights.end(), []( const std::pair<std::vector<size_t>, unsigned>& a,
                                                                                    const std::pair<std::vector<size_t>, unsigned>& b)
                                                                                    { return a.second > b.second; });
        std::multiset<int, std::greater<int>> weights;
        for (auto& comp_wt_pair : components_and_weights) {
            weights.emplace(comp_wt_pair.second);
        }
        // sanity test
        size_t counter = 0;
        for (auto wt_it = weights.begin(); wt_it != weights.cend(); wt_it++) {
            assert( *wt_it == components_and_weights[counter].second );
            counter++;
        }
        std::vector<unsigned> allocation = greedy_partitioner(instance.numberOfProcessors(), weights);
        float imbalance  = calculate_imbalance(instance.numberOfProcessors(), weights, allocation);
        std::pair<float, std::vector<unsigned>> improved_imbalance_and_allocation = hill_climb_weight_balance_single_superstep(params.hillclimb_balancer_iterations, instance.numberOfProcessors(), weights, allocation);
        if (improved_imbalance_and_allocation.first <= imbalance) {
            allocation = improved_imbalance_and_allocation.second;
            imbalance = improved_imbalance_and_allocation.first;
        }

        // trying to include next level
        while(curr_level + 1 < level_sets.size()) {
            for (const auto& node : level_sets[curr_level+1]) {
                uf_universe.add_object(node, instance.getComputationalDag().nodeWorkWeight(node));
            }
            for (const auto& node : level_sets[curr_level+1]) {
                for (const auto& in_edge : instance.getComputationalDag().in_edges(node)) {
                    if ( uf_universe.is_in_universe(in_edge.m_source) ) {
                        uf_universe.join_by_name(node, in_edge.m_source);
                    }
                }
            }

            std::vector<std::pair<std::vector<size_t>, unsigned>> new_components_and_weights = uf_universe.get_connected_components_and_weights();
            std::sort(new_components_and_weights.begin(), new_components_and_weights.end(), []( const std::pair<std::vector<size_t>, unsigned>& a,
                                                                                                const std::pair<std::vector<size_t>, unsigned>& b)
                                                                                                { return a.second > b.second; });
            std::multiset<int, std::greater<int>> new_weights;
            for (auto& comp_wt_pair : new_components_and_weights) {
                new_weights.emplace(comp_wt_pair.second);
            }
            // sanity test
            size_t counter = 0;
            for (auto wt_it = new_weights.begin(); wt_it != new_weights.cend(); wt_it++) {
                assert( *wt_it == new_components_and_weights[counter].second );
                counter++;
            }
            std::vector<unsigned> new_allocation = greedy_partitioner(instance.numberOfProcessors(), new_weights);
            float new_imbalance  = calculate_imbalance(instance.numberOfProcessors(), new_weights, new_allocation);
            std::pair<float, std::vector<unsigned>> new_improved_imbalance_and_allocation = hill_climb_weight_balance_single_superstep(params.hillclimb_balancer_iterations, instance.numberOfProcessors(), new_weights, new_allocation);
            if (new_improved_imbalance_and_allocation.first <= new_imbalance) {
                new_allocation = new_improved_imbalance_and_allocation.second;
                new_imbalance = new_improved_imbalance_and_allocation.first;
            }

            if (params.balance_function == HDagg_parameters::XLOGX) {
                std::vector<size_t> partition_weights(instance.numberOfProcessors(),0);
                size_t total_weight = 0;
                size_t wt_counter = 0;
                for (auto& wt: new_weights) {
                    partition_weights[new_allocation[wt_counter]] += wt;
                    total_weight += wt;
                    wt_counter++;
                }
                // std::vector<float> normalised_partition_weights(instance.numberOfProcessors(),0.0);
                new_imbalance = 0.0;
                for (size_t i = 0; i < partition_weights.size(); i++) {
                    float normalised_partition_weight = partition_weights[i] / total_weight;
                    new_imbalance += normalised_partition_weight * std::log2( normalised_partition_weight * instance.numberOfProcessors() );
                }
            }

            if (new_imbalance < params.balance_threshhold) {
                components_and_weights = new_components_and_weights;
                // weights = new_weights; // not needed
                allocation = new_allocation;
                imbalance = new_imbalance;
                curr_level++;
            } else {
                break;
            }           
        }
        // allocate nodes to superstep and processor
        for (size_t i = 0; i < components_and_weights.size(); i++) {
            for (auto node :components_and_weights[i].first) {
                sched.setAssignedSuperstep(node, curr_superstep);
                sched.setAssignedProcessor(node, allocation[i]);
            }
        }

        curr_superstep++;
    }


    // run Hungarian algorithm
    if (params.hungarian_alg) {
        Hungarian_alg_process_permuter helper;
        helper.improveSchedule(sched);
    } else {
        sched.setAutoCommunicationSchedule();
    }

    return std::make_pair(SUCCESS, sched);
}