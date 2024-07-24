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

#include "algorithms/Wavefront/Wavefront.hpp"

std::pair<RETURN_STATUS, BspSchedule> Wavefront::computeSchedule(const BspInstance &instance) {
    DAG graph(instance.getComputationalDag());
    assert(graph.n > 0);

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

    for (size_t curr_level = 0; curr_level < level_sets.size(); curr_level++) {
        

        std::vector<std::pair<size_t, unsigned>> components_and_weights;
        for (size_t i = 0; i < level_sets[curr_level].size(); i++) {
            components_and_weights.emplace_back(level_sets[curr_level][i], instance.getComputationalDag().nodeWorkWeight(level_sets[curr_level][i]));
        }
        std::sort(components_and_weights.begin(), components_and_weights.end(), []( const std::pair<size_t, unsigned>& a,
                                                                                    const std::pair<size_t, unsigned>& b)
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

        // allocate nodes to superstep and processor
        for (size_t i = 0; i < components_and_weights.size(); i++) {
            size_t node = components_and_weights[i].first;
            sched.setAssignedSuperstep(node, curr_level);
            sched.setAssignedProcessor(node, allocation[i]);
        }
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