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

#include "refine/superstep_clumps.hpp"

Clump::Clump() : node_set(std::unordered_set<int>()), total_weight(0) { }

Clump::Clump(const DAG& graph) {
    node_set.reserve(graph.n);
    total_weight = 0;
    for (int i = 0 ; i < graph.n ; i++) {
        node_set.emplace(i);
        total_weight += graph.workW[i];
    }
}

Clump::Clump(const DAG& graph, const std::unordered_set<int>& node_set_ ) {
    node_set = node_set_;
    total_weight = graph.workW_of_node_set(node_set_);
}

Clump::Clump(const SubDAG& graph) {
    node_set.reserve(graph.n);
    total_weight = 0;
    for (int i = 0 ; i < graph.n ; i++) {
        node_set.emplace( graph.sub_to_super.at(i) );
        total_weight += graph.workW[i];
    }
}

Clump::Clump(const SubDAG& graph, const std::unordered_set<int>& node_set_ ) {
    node_set = node_set_;
    total_weight = graph.workW_of_node_set(node_set_);
}

void LooseSchedule::add_loose_superstep(unsigned position, std::vector<std::unordered_set<int>>& vec_node_sets ) {
    std::vector<Clump> clump_collection;
    clump_collection.reserve(vec_node_sets.size());

    for (auto& node_set : vec_node_sets) {
        if (! node_set.empty())
            clump_collection.emplace_back(graph, node_set);
    }

    if (clump_collection.empty())
        throw std::runtime_error("Trying to a add an empty loose superstep, i.e., vector is empty or is a vector of empty sets.");

    if ( position != superstep_ordered_ids.size() ) {
        auto it = superstep_ordered_ids.begin();
        std::advance(it, position);
        superstep_ordered_ids.emplace( it , step_id_counter);
    }
    else if (position == superstep_ordered_ids.size()) {
        superstep_ordered_ids.emplace_back( step_id_counter );
    }
    else {
        throw std::runtime_error("Adding loose superstep position is invalid.");
    }

    supersteps.emplace(step_id_counter, clump_collection, params);

    step_id_counter++;
}


void LooseSchedule::add_loose_superstep_with_allocation(unsigned position, LooseSuperStep<Clump> superstep ) {
    if ( position != superstep_ordered_ids.size() ) {
        auto it = superstep_ordered_ids.begin();
        std::advance(it, position);
        superstep_ordered_ids.emplace( it , step_id_counter);
    }
    else if (position == superstep_ordered_ids.size()) {
        superstep_ordered_ids.emplace_back( step_id_counter );
    }
    else {
        throw std::runtime_error("Adding loose superstep position is invalid.");
    }

    supersteps.emplace(step_id_counter, superstep);

    step_id_counter++;
}


bool LooseSchedule::split_into_two_supersteps(std::multiset< LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator supersteps_iterator, const CutType cut_type ) {
    // getting superstep id and location
    LooseSuperStep<Clump> current_superstep = (*supersteps_iterator);
    unsigned superstep_id = current_superstep.id;
    std::vector<unsigned>::iterator superstep_location = superstep_ordered_ids.begin();
    unsigned superstep_location_integer = 0;
    while ( *superstep_location != superstep_id && superstep_location != superstep_ordered_ids.cend() ) { // TODO later: improve with better structure
        superstep_location++;
        superstep_location_integer++;
    }
    if (superstep_location == superstep_ordered_ids.cend()) {
        throw std::runtime_error("Superstep id not found");
    }

    // Test
    std::vector<unsigned>::iterator test_it = superstep_ordered_ids.begin();
    std::advance(test_it, superstep_location_integer);
    assert( test_it == superstep_location );

    // top disconnected components
    std::vector<std::unordered_set<int>> vec_top_node_sets;
    // bottom disconnected components
    std::vector<std::unordered_set<int>> vec_bottom_node_sets;

    // Clump cut iterations
    if (cut_type == Balanced) {
        const unsigned num_clumps = current_superstep.get_number_of_clumps();
        const unsigned clump_split_limit = 1 + unsigned(num_clumps*params.balanced_cut_ratio);

        const int min_weight_for_balanced_split  = int(current_superstep.get_avg_weight_of_partition() * params.min_weight_for_split );

        std::vector<Clump> non_split_fat_clumps;

        // Splitting fat clumps
        unsigned num_split_clumps = 0;
        std::multiset<Clump, typename Clump::Comparator>::iterator it = current_superstep.collection.begin();
        while ( it != current_superstep.collection.cend() && (num_split_clumps <= clump_split_limit) && ((*it).total_weight >= min_weight_for_balanced_split) ) {
            if ( (*it).node_set.size() <= 1 ) {
                non_split_fat_clumps.push_back( (*it) );
                it++;
                continue;
            }
            
            SubDAG clump_subgraph = SubDAG(graph, it->node_set );

            std::pair<std::unordered_set<int>, std::unordered_set<int>> split = dag_weight_bal_cut(clump_subgraph);
            if ( split.first.empty() || split.second.empty() ) {
                non_split_fat_clumps.push_back( (*it) );
                it++;
                continue;
            }

            std::vector<std::unordered_set<int>> top_components = clump_subgraph.weakly_connected_components(split.first);
            std::vector<std::unordered_set<int>> bottom_components = clump_subgraph.weakly_connected_components(split.second);

            vec_top_node_sets.insert(vec_top_node_sets.end(), top_components.begin(), top_components.end());
            vec_bottom_node_sets.insert(vec_bottom_node_sets.end(), bottom_components.begin(), bottom_components.end());

            it++;
            num_split_clumps ++;
        }

        // assigning other clumps
        while ( it != current_superstep.collection.cend() ) {
            bool flip = coin->get_flip();
            if (flip)
                vec_top_node_sets.push_back( (*it).node_set );
            else
                vec_bottom_node_sets.push_back( (*it).node_set );

            it++;
        }

        // assigning non-split fat clumps
        for (auto& clump : non_split_fat_clumps) {
            bool flip = coin->get_flip();
            if (flip)
                vec_top_node_sets.push_back( clump.node_set );
            else
                vec_bottom_node_sets.push_back( clump.node_set );
        }
    }
    else if (cut_type == Shaving) {
        for (auto& clump : current_superstep.collection) {
            std::pair<std::unordered_set<int>, std::unordered_set<int>> split;

            SubDAG clump_subgraph = SubDAG(graph, clump.node_set );
            
            bool flip = coin->get_flip();
            if (flip) {
                // std::cout << "Clumpsize top: " << clump.node_set.size() << std::endl;
                split = top_shave_few_sources(clump_subgraph, params.min_comp_generation_when_shaving);
            }
            else {
                // std::cout << "Clumpsize bottom: " << clump.node_set.size() << std::endl;
                split = bottom_shave_few_sinks(clump_subgraph, params.min_comp_generation_when_shaving);
            }

            std::vector<std::unordered_set<int>> top_components = clump_subgraph.weakly_connected_components(split.first);
            std::vector<std::unordered_set<int>> bottom_components = clump_subgraph.weakly_connected_components(split.second);

            vec_top_node_sets.insert(vec_top_node_sets.end(), top_components.begin(), top_components.end());
            vec_bottom_node_sets.insert(vec_bottom_node_sets.end(), bottom_components.begin(), bottom_components.end());
        }
    }
    else {
        throw std::runtime_error("Cut type not (correctly) specified.");
    }


    // delete old superstep
    superstep_ordered_ids.erase(superstep_location);
    supersteps.erase(supersteps_iterator);


    int change_bot = false;
    int change_top = false;
    
    // add new supersteps
    if ( (!vec_bottom_node_sets.empty()) && std::any_of(vec_bottom_node_sets.cbegin(), vec_bottom_node_sets.cend(), [](const std::unordered_set<int>& node_set) { return !node_set.empty();}) ) {
        add_loose_superstep( superstep_location_integer , vec_bottom_node_sets);
        change_bot = true;
    }
    if ( (!vec_top_node_sets.empty()) && std::any_of(vec_top_node_sets.cbegin(), vec_top_node_sets.cend(), [](const std::unordered_set<int>& node_set) { return !node_set.empty();}) ) {
        add_loose_superstep( superstep_location_integer , vec_top_node_sets);
        change_top = true;
    }
    return (change_bot && change_top);
}

std::unordered_map<int, std::pair<unsigned, unsigned>> LooseSchedule::get_current_node_schedule_allocation() const {
    std::unordered_map<int, std::pair<unsigned, unsigned>> node_allocation;
    node_allocation.reserve(graph.n);

    std::unordered_map<unsigned, unsigned> superstep_id_to_order;
    superstep_id_to_order.reserve(superstep_ordered_ids.size());
    for (unsigned i = 0; i < superstep_ordered_ids.size(); i++) {
        assert( superstep_id_to_order.find(superstep_ordered_ids[i]) == superstep_id_to_order.cend() );
        superstep_id_to_order[superstep_ordered_ids[i]] = i;
    }

    for (auto& sstep: supersteps) {
        assert( superstep_id_to_order.find(sstep.id) != superstep_id_to_order.cend() );
    }

    for(auto& sstep: supersteps) {
        auto clump_it = sstep.collection.begin();
        std::vector<unsigned> clump_to_processor_allocation = sstep.get_current_allocation();
        for (unsigned i = 0 ; i < sstep.collection.size(); i++) {
            for (auto& node :  clump_it->node_set) {
                assert( node_allocation.find(node) == node_allocation.cend() );
                node_allocation[node] = std::pair<unsigned, unsigned>{superstep_id_to_order.at(sstep.id), clump_to_processor_allocation[i] };
            }
            clump_it++;
        }
    }

    return node_allocation;
}

std::vector<std::vector<std::vector<int>>> LooseSchedule::get_current_schedule() const {
    std::vector<std::vector<std::vector<int>>> output;
    output.resize(superstep_ordered_ids.size());

    std::unordered_map<unsigned, unsigned> superstep_id_to_order;
    superstep_id_to_order.reserve(superstep_ordered_ids.size());
    for (unsigned i = 0; i < superstep_ordered_ids.size(); i++) {
        assert( superstep_id_to_order.find(superstep_ordered_ids[i]) == superstep_id_to_order.cend() );
        superstep_id_to_order[superstep_ordered_ids[i]] = i;
    }

    for (auto& sstep: supersteps) {
        assert( superstep_id_to_order.find(sstep.id) != superstep_id_to_order.cend() );
    }

    for(auto& sstep: supersteps) {
        output[superstep_id_to_order.at(sstep.id)] = sstep.get_current_processors_with_nodes();
    }

    return output;
}

std::vector<float> LooseSchedule::get_current_superstep_imbalances_in_order() const {
    std::vector<float> output(superstep_ordered_ids.size());
    
    std::unordered_map<unsigned, unsigned> superstep_id_to_order;
    superstep_id_to_order.reserve(superstep_ordered_ids.size());
    for (unsigned i = 0; i < superstep_ordered_ids.size(); i++) {
        assert( superstep_id_to_order.find(superstep_ordered_ids[i]) == superstep_id_to_order.cend() );
        superstep_id_to_order[superstep_ordered_ids[i]] = i;
    }

    for (auto& sstep: supersteps) {
        assert( superstep_id_to_order.find(sstep.id) != superstep_id_to_order.cend() );
    }

    for(auto& sstep: supersteps) {
        output[superstep_id_to_order.at(sstep.id)] = sstep.get_imbalance();
    }

    return output;
}

void LooseSchedule::print_current_schedule() const {
    std::cout << std::endl << "Schedule:" << std::endl;
    std::vector<std::vector<std::vector<int>>> schedule = get_current_schedule();
    std::vector<float> superstep_imbalances = get_current_superstep_imbalances_in_order();

    for (size_t i =0; i < schedule.size(); i++) {
        std::cout << std::endl << "Superstep " << i << ", imbalance: " << superstep_imbalances[i] << std::endl;
        for (size_t j = 0; j < schedule[i].size(); j++) {
            std::cout << "Processor " << j << ": ";
            for (auto& node : schedule[i][j]) {
                std::cout << node << ", ";
            }
            std::cout << std::endl;
        }
    }
}



bool LooseSchedule::run_hill_climb(std::multiset< LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator supersteps_iterator, int hill_climb_iterations)
{
    LooseSuperStep<Clump> superstep_copy = *supersteps_iterator;
    bool improvement = superstep_copy.run_allocation_improvement(hill_climb_iterations);

    if (improvement) {
        std::vector<unsigned>::iterator superstep_location = superstep_ordered_ids.begin();
        unsigned superstep_location_integer = 0;
        while ( *superstep_location != superstep_copy.id && superstep_location != superstep_ordered_ids.cend() ) { // TODO later: improve with better structure
            superstep_location++;
            superstep_location_integer++;
        }
        if (superstep_location == superstep_ordered_ids.cend()) {
            throw std::runtime_error("Superstep id not found");
        }

        // deleting old superstep
        superstep_ordered_ids.erase(superstep_location);
        supersteps.erase(supersteps_iterator);

        // adding new superstep
        add_loose_superstep_with_allocation( superstep_location_integer , superstep_copy);
    }

    return improvement;
}



bool LooseSchedule::run_superstep_improvement_iteration() {
    std::unordered_set<unsigned> id_superstep_run_hill_climb;
    std::unordered_set<unsigned> id_superstep_to_split;
    std::unordered_map<unsigned, CutType> id_cuttype_lookup;

    std::multiset< LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator sstep_it = supersteps.begin();

    while (sstep_it != supersteps.cend() && sstep_it->get_imbalance() > params.balance_threshhold ) {
        // std::cout << "Superstep id: " << sstep_it->id << " Imbalance: " << sstep_it->get_imbalance() << std::endl;

        if (sstep_it->too_few_nodes()) {
            std::advance(sstep_it,1);
            continue;    
        }
        // if ( (! wait_for_node_increase) && sstep_it->get_number_of_nodes() <= sstep_it->collection.size() * params.nodes_per_clump_no_wait ) {
        //     std::cout << "Too few nodes no wait" << std::endl;
        //     std::advance(sstep_it,1);
        //     continue;
        // }

        if (sstep_it->too__few_clumps()) {
            id_superstep_to_split.emplace(sstep_it->id);
            id_cuttype_lookup[sstep_it->id] = Shaving;
        }
        else if (sstep_it->fat_nodes()) {
            id_superstep_to_split.emplace(sstep_it->id);
            id_cuttype_lookup[sstep_it->id] = Balanced;
        }
        else if ((params.hill_climb_simple_improvement_attemps > 0) && (params.number_of_partitions >= 2) ) {
            id_superstep_run_hill_climb.emplace(sstep_it->id);
        }
        std::advance(sstep_it,1);
    }

    bool change = false;
    size_t i = 0;
    while( i < supersteps.size() ) {
        std::multiset< LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator sstep_change_it = supersteps.begin();
        std::advance(sstep_change_it,i);
        if (sstep_change_it->get_imbalance() < params.balance_threshhold) {
            break;
        }

        bool minus_flag = false;

        if ( id_superstep_to_split.find(sstep_change_it->id) != id_superstep_to_split.cend() ) {
            bool temp_change = split_into_two_supersteps(sstep_change_it, id_cuttype_lookup.at(sstep_change_it->id));
            change = change || temp_change;
            // std::cout << "Split change: " << temp_change << std::endl;
            if (temp_change) {
                minus_flag = true;
            }
        }
        else if ( id_superstep_run_hill_climb.find( sstep_change_it->id ) != id_superstep_run_hill_climb.cend() ) {
            bool improvement = run_hill_climb(sstep_change_it);
            if (improvement) {
                minus_flag = true;
            }
            change = change || improvement;
            // std::cout << "Hill climb change: " << improvement << std::endl;
        }

        if (!minus_flag) {
            i++;
        }
    }

    return change;
}


Coarse_Scheduler_Params Coarse_Scheduler_Params::lin_comb( const Coarse_Scheduler_Params& first, const Coarse_Scheduler_Params& second, const std::pair<unsigned, unsigned>& ratio ) {
    if ( first.number_of_partitions != second.number_of_partitions ) {
              throw std::logic_error( "Number of parititions/processors needs to agree in both Coarse_Schedule_Params" );
    }
    unsigned number_of_partitions = first.number_of_partitions;

    unsigned ratio_sum = ratio.first + ratio.second;

    float balance_threshhold = (first.balance_threshhold*ratio.first+second.balance_threshhold*ratio.second)/ratio_sum;


    float nodes_per_clump = (first.nodes_per_clump*ratio.first+second.nodes_per_clump*ratio.second)/ratio_sum ;

    float nodes_per_partition = (first.nodes_per_partition*ratio.first+second.nodes_per_partition*ratio.second)/ratio_sum;

    float clumps_per_partition = (first.clumps_per_partition*ratio.first+second.clumps_per_partition*ratio.second)/ratio_sum;
    
    float max_weight_for_flag = (first.max_weight_for_flag*ratio.first+second.max_weight_for_flag*ratio.second)/ratio_sum;
    
    float balanced_cut_ratio = (first.balanced_cut_ratio*ratio.first+second.balanced_cut_ratio*ratio.second)/ratio_sum;
    
    float min_weight_for_split = (first.min_weight_for_split*ratio.first+second.min_weight_for_split*ratio.second)/ratio_sum;
    

    long unsigned hill_climb = (first.hill_climb_simple_improvement_attemps*ratio.first+second.hill_climb_simple_improvement_attemps*ratio.second)/ratio_sum;
    unsigned hill_climb_simple_improvement_attemps = hill_climb;

    long int min_comp = (first.min_comp_generation_when_shaving*ratio.first+second.min_comp_generation_when_shaving*ratio.second)/ratio_sum;
    int min_comp_generation_when_shaving = min_comp;

    PartitionAlgorithm part_algo;
    CoinType coin_type;

    unsigned die_roll = randInt(ratio_sum);
    if (die_roll < ratio.first) {
        part_algo = first.part_algo;
        coin_type = first.coin_type;
    }
    else {
        part_algo = second.part_algo;
        coin_type = second.coin_type;
    }

    
    return Coarse_Scheduler_Params( number_of_partitions,
                                    balance_threshhold,
                                    part_algo,
                                    coin_type,
                                    clumps_per_partition,
                                    nodes_per_clump,
                                    nodes_per_partition,
                                    max_weight_for_flag,
                                    balanced_cut_ratio,
                                    min_weight_for_split,
                                    hill_climb_simple_improvement_attemps,
                                    min_comp_generation_when_shaving);
}


std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator LooseSchedule::find_id( const unsigned id ) const {
    std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator iterator = supersteps.begin();

    while ( iterator->id != id && iterator != supersteps.cend() ) {
        iterator++;
    } 

    return iterator;
}


bool LooseSchedule::combine_superstep_attempt( const unsigned superstep_position, const unsigned comm_cost_multiplier, const unsigned com_cost_addition, const bool true_costs) {
    if (superstep_position >= superstep_ordered_ids.size()-1) return false;

    unsigned ss_id_1 = superstep_ordered_ids[superstep_position];
    unsigned ss_id_2 = superstep_ordered_ids[superstep_position+1];

    std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator superstep_1 = find_id(ss_id_1);
    std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator superstep_2 = find_id(ss_id_2);
    assert( superstep_1 != supersteps.cend() );
    assert( superstep_2 != supersteps.cend() );

    // Computing current costs
    unsigned compute_cost_1 = superstep_1->get_curretn_max_weight_of_partition();
    unsigned compute_cost_2 = superstep_2->get_curretn_max_weight_of_partition();
    unsigned comm_cost = 0;

    // compute comm costs
    if (comm_cost_multiplier != 0) {
        std::unordered_set<int> nodes_1({});
        std::unordered_set<int> nodes_2({});

        for (auto& clmp : superstep_1->collection) {
            nodes_1.insert( clmp.node_set.cbegin(), clmp.node_set.cend() );
        }
        for (auto& clmp : superstep_2->collection) {
            nodes_2.insert( clmp.node_set.cbegin(), clmp.node_set.cend() );
        }

        std::unordered_map<int, unsigned> node_alloc1;
        node_alloc1.reserve(nodes_1.size());
        unsigned clmp_index = 0;
        for (auto& clmp : superstep_1->collection) {
            for (auto& node : clmp.node_set) {
                node_alloc1[node] = superstep_1->get_current_allocation()[ clmp_index ];
            }
            clmp_index++;
        }

        std::unordered_map<int, unsigned> node_alloc2;
        node_alloc2.reserve(nodes_2.size());
        clmp_index = 0;
        for (auto& clmp : superstep_2->collection) {
            for (auto& node : clmp.node_set) {
                node_alloc2[node] = superstep_2->get_current_allocation()[ clmp_index ];
            }
            clmp_index++;
        }

        std::vector<unsigned> send_cost( params.number_of_partitions, 0);
        std::vector<unsigned> receive_cost( params.number_of_partitions, 0);

        unsigned clmp_ind = 0;
        for ( auto& clmp : superstep_1->collection ) {
            for (auto& node : clmp.node_set) {
                for ( auto& sub_out_node : graph.Out[graph.super_to_sub.at(node)] ) {
                    int super_out_node = graph.sub_to_super.at( sub_out_node );
                    if ( nodes_2.find(super_out_node) != nodes_2.cend() ) {
                        if (true_costs && (superstep_1->get_current_allocation()[clmp_ind] != node_alloc2.at(super_out_node) ) ) continue;
                        send_cost[ superstep_1->get_current_allocation()[ clmp_ind ] ] += graph.commW[ graph.super_to_sub.at(node) ]; 
                    }
                }
            }
            clmp_ind++;
        }

        clmp_ind = 0;
        for ( auto& clmp : superstep_2->collection ) {
            for (auto& node : clmp.node_set) {
                for ( auto& sub_in_node : graph.In[graph.super_to_sub.at(node)] ) {
                    int super_in_node = graph.sub_to_super.at( sub_in_node );
                    if ( nodes_1.find(super_in_node) != nodes_1.cend() ) {
                        if (true_costs && (superstep_2->get_current_allocation()[clmp_ind] != node_alloc1.at(super_in_node) ) ) continue;
                        receive_cost[ superstep_2->get_current_allocation()[ clmp_ind ] ] += graph.commW[ sub_in_node ]; 
                    }
                }
            }
            clmp_ind++;
        }

        for ( auto& cost : send_cost ) {
            comm_cost = std::max(comm_cost, cost);
        }
        for ( auto& cost : receive_cost ) {
            comm_cost = std::max(comm_cost, cost);
        }

    }

    std::unordered_set<int> combined_node_set({});
    for (auto& clmp : superstep_1->collection) {
        combined_node_set.insert( clmp.node_set.cbegin(), clmp.node_set.cend() );
    }
    for (auto& clmp : superstep_2->collection) {
        combined_node_set.insert( clmp.node_set.cbegin(), clmp.node_set.cend() );
    }

    std::vector<std::unordered_set<int>> components = graph.weakly_connected_components( combined_node_set );
    std::vector<Clump> clump_collection;
    clump_collection.reserve(components.size());
    for (auto& node_set : components) {
        if (! node_set.empty())
            clump_collection.emplace_back(graph, node_set);
    }
    LooseSuperStep<Clump> combine_superstep( step_id_counter, clump_collection, params );
    combine_superstep.run_allocation_improvement();

    unsigned compute_cost_new = combine_superstep.get_curretn_max_weight_of_partition();

    if ( compute_cost_new > compute_cost_1+compute_cost_2+ (comm_cost_multiplier*comm_cost)+com_cost_addition ) {
        return false;
    }
    else {
        // delete old supersteps
        auto ord_id_iter = superstep_ordered_ids.begin();
        std::advance(ord_id_iter, superstep_position);
        assert( *ord_id_iter = ss_id_1 );
        ord_id_iter = superstep_ordered_ids.erase( ord_id_iter );
        assert( *ord_id_iter = ss_id_2 );
        superstep_ordered_ids.erase( ord_id_iter );

        supersteps.erase( find_id(ss_id_1) );
        supersteps.erase( find_id(ss_id_2) );

        // insert new superstep
        add_loose_superstep_with_allocation( superstep_position, combine_superstep);

        return true;
    }
}

bool LooseSchedule::run_joining_supersteps_improvements(const bool parity, const bool only_above_thresh, const unsigned comm_cost_multiplier, const unsigned com_cost_addition, const bool true_costs) {
    bool change = false;

    unsigned i = 0;
    if (parity) {
        i++;
    }

    while ( i < superstep_ordered_ids.size()-1) {
        if ( only_above_thresh ) {
            auto iter_1 = find_id( superstep_ordered_ids[i] );
            assert( iter_1 != supersteps.cend() );
            auto iter_2 = find_id( superstep_ordered_ids[i+1] );
            assert( iter_2 != supersteps.cend() );

            if ( (iter_1->get_imbalance() > params.balance_threshhold) && (iter_2->get_imbalance() > params.balance_threshhold)) {
                bool temp_change = combine_superstep_attempt(i, comm_cost_multiplier, com_cost_addition, true_costs);
                if (temp_change) {
                    change = true;
                    i--;
                }
            }

        }
        else {
            bool temp_change = combine_superstep_attempt(i, comm_cost_multiplier, com_cost_addition, true_costs);
            if (temp_change) {
                change = true;
                i--;
            }
        }


        i += 2;
    }

    return change;
}

void LooseSchedule::permute_allocation_of_superstep(const unsigned index, std::vector<unsigned> permutation) {
    // checking whether permutation
    assert( permutation.size() == params.number_of_partitions );
    std::vector<bool> check(params.number_of_partitions, false);
    for (size_t i = 0; i < permutation.size(); i++) {
        assert( permutation[i] < params.number_of_partitions );
        assert( check[ permutation[i] ] == false );
        check[permutation[i]] = true;
    }

    auto ss_iter = supersteps.begin();
    std::advance(ss_iter, index);
    LooseSuperStep<Clump> permuted_ss = *ss_iter;
    unsigned id_ss = ss_iter->id;
    permuted_ss.permute_allocation(permutation);

    // erasing
    supersteps.erase(ss_iter);
    // inserting
    supersteps.insert(permuted_ss);

    // correctness checking
    ss_iter = supersteps.begin();
    std::advance(ss_iter, index);
    assert( id_ss == ss_iter->id  ); //should be ok - may make problems because of float comparisons - would be able to comment out
}


void LooseSchedule::run_processor_assignment() {
    std::vector< std::vector< unsigned> > cost_v( params.number_of_partitions, std::vector<unsigned>(params.number_of_partitions,1) );
    run_processor_assignment(cost_v);
}

void LooseSchedule::run_processor_assignment(const std::vector<std::vector<unsigned>>& processsor_comm_costs) {
    assert( processsor_comm_costs.size() == params.number_of_partitions );
    for (long unsigned p = 0; p<processsor_comm_costs.size(); p++) {
        assert( processsor_comm_costs[p].size() == params.number_of_partitions );
    }

    std::vector<unsigned int> node_to_superstep_assignment(graph.n);
    std::vector<unsigned int> node_to_processor_assignment(graph.n);

    // generating efficient map superstep_id to which superstep
    std::unordered_map<unsigned, unsigned> superstep_id_to_superstep_sequence;
    superstep_id_to_superstep_sequence.reserve(superstep_ordered_ids.size());
    for(unsigned i = 0; i< superstep_ordered_ids.size(); i++) {
        superstep_id_to_superstep_sequence[superstep_ordered_ids[i]] = i;
    }
    std::unordered_map<unsigned, unsigned> id_to_collection_ind;
    id_to_collection_ind.reserve(superstep_ordered_ids.size());
    unsigned sstep_ind = 0;
    for ( auto& sstep : supersteps ) {
        id_to_collection_ind[ sstep.id ] = sstep_ind;
        sstep_ind++;
    }

    // assigning node to superstep
    for( auto& sstep : supersteps ) {
        for ( auto& clmp : sstep.collection ) {
            for (auto& node : clmp.node_set) {
                node_to_superstep_assignment[node] = superstep_id_to_superstep_sequence.at(sstep.id);
            }
        }
    }

    // assigning node to processor
    std::vector<bool> node_assigned_processor(graph.n, false);
    for (auto& id : superstep_ordered_ids) {
        std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator sstep = supersteps.begin();
        std::advance(sstep, id_to_collection_ind.at(id) );

        std::vector<std::vector<long long unsigned>> partition_to_processor_comm_cost( params.number_of_partitions, std::vector<long long unsigned>(params.number_of_partitions, 0) );

        // computing partition to processor communication savings
        unsigned clmp_index = 0;
        std::vector<unsigned> allocation = sstep->get_current_allocation();
        for ( auto& clmp : sstep->collection ) {
            for ( auto& node : clmp.node_set ) {
                for ( auto& parent : graph.In[node]) {
                    if (node_assigned_processor[parent]) {
                        for (long unsigned procssr = 0; procssr< params.number_of_partitions; procssr++) {
                            if ( procssr ==  node_to_processor_assignment[parent] ) continue;

                            partition_to_processor_comm_cost[ allocation[clmp_index] ][ procssr ] += graph.commW[parent]*processsor_comm_costs[node_to_processor_assignment[parent]][procssr];
                        }
                    }
                }
            }
            clmp_index++;
        }

        // deciding partition to processor allocation
        std::vector<unsigned> partition_to_processor_allocation = min_perfect_matching_for_complete_bipartite(partition_to_processor_comm_cost);

        // allocating nodes
        clmp_index = 0;
        for ( auto& clmp : sstep->collection ) {
            for ( auto& node : clmp.node_set ) {
                node_to_processor_assignment[node] = partition_to_processor_allocation[ allocation[clmp_index] ];
                node_assigned_processor[node] = true;
            }
            clmp_index++;
        }
        permute_allocation_of_superstep( id_to_collection_ind.at(id), partition_to_processor_allocation );
    }

    // check that all nodes have been assigned a processor
    assert( std::all_of(node_assigned_processor.cbegin(), node_assigned_processor.cend(), []( const bool& has_been ) {return has_been;} ) );
}