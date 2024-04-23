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

#include "algorithms/Coarsers/SquashA.hpp"

int SquashA::run_and_add_contraction(const contract_edge_sort edge_sort_type) {
    // TODO implement with ComputationalDAG
    const DAG last_dag(dag_history.back()->getComputationalDag());

    std::vector<int> poset_int_mapping;
    if ( thue_coin.get_flip() ) {
        poset_int_mapping = last_dag.get_strict_poset_integer_map( coarsen_par.noise, coarsen_par.poisson_par );
    }
    else {
        if ( balanced_random.get_flip() ) {
            poset_int_mapping = last_dag.get_top_node_distance();
        }
        else
        {
            std::vector<int> bot_dist = last_dag.get_bottom_node_distance();
            poset_int_mapping.resize( bot_dist.size() );
            for (size_t i = 0; i<  bot_dist.size(); i++) {
                poset_int_mapping[i] = - bot_dist[i];
            }
        }
    }

    std::multiset<Edge_Weighted, Edge_Weighted::Comparator> contractable_edges = last_dag.get_contractable_edges(edge_sort_type, poset_int_mapping);
    // std::cout << "Contractable edges size: " << contractable_edges.size() << std::endl;

    std::multiset<int> sorted_weights;
    for (auto& wt : last_dag.workW) {
        sorted_weights.emplace(wt);
    }
    // int median = Get_upper_third_percentile(sorted_weights)+1; // +1 for safety as it could be zero
    // int median = Get_Median(sorted_weights)+1; // +1 for safety as it could be zero
    int median = Get_lower_third_percentile(sorted_weights)+1; // +1 for safety as it could be zero

    Union_Find_Universe<int> connected_components;
    for (unsigned i = 0; i < last_dag.n; i++) {
        connected_components.add_object(i, last_dag.workW[i]);
    }
    
    // std::unordered_set<int> merged_nodes;
    std::vector<bool> merged_nodes( last_dag.n, false );

    int num_nodes_decrease = 0;
    int num_nodes_aim = last_dag.n - int( last_dag.n / coarsen_par.geom_decay_num_nodes );

    float temperature = 1;
    int temperature_increase_iteration = 0;
    while( num_nodes_decrease < num_nodes_aim && temperature_increase_iteration <= coarsen_par.number_of_temperature_increases ) {
        for (auto& wt_edge : contractable_edges) {
            // Previously merged
            if (merged_nodes[ wt_edge.edge_pair.first ]) continue;
            if (merged_nodes[ wt_edge.edge_pair.second ]) continue;
            
            // weight check
            if ( connected_components.get_weight_of_component_by_name( wt_edge.edge_pair.first )
                + connected_components.get_weight_of_component_by_name( wt_edge.edge_pair.second ) > median*temperature ) continue;
            
            // no loops criteria check
            bool check_failed = false;
            // safety check - this should already be the case
            assert( abs( poset_int_mapping[wt_edge.edge_pair.first] - poset_int_mapping[wt_edge.edge_pair.second] ) <= 1 );
            // Checks over all affected edges
            // In edges first
            for (auto& node : last_dag.In[ wt_edge.edge_pair.first ]) {
                if (node == wt_edge.edge_pair.second) continue;
                if (! merged_nodes[ node ]) continue;
                if ( poset_int_mapping[ wt_edge.edge_pair.first ] >= poset_int_mapping[node] +2 ) continue;
                check_failed = true;
                break;
            }
            if (check_failed) continue;
            // Out edges first
            for (auto& node : last_dag.Out[ wt_edge.edge_pair.first ]) {
                if (node == wt_edge.edge_pair.second) continue;
                if (! merged_nodes[ node ]) continue;
                if ( poset_int_mapping[ node ] >= poset_int_mapping[ wt_edge.edge_pair.first ] +2 ) continue;
                check_failed = true;
                break;
            }
            if (check_failed) continue;
            // In edges second
            for (auto& node : last_dag.In[ wt_edge.edge_pair.second ]) {
                if (node == wt_edge.edge_pair.first) continue;
                if (! merged_nodes[ node ]) continue;
                if ( poset_int_mapping[ wt_edge.edge_pair.second ] >= poset_int_mapping[node] +2 ) continue;
                check_failed = true;
                break;
            }
            if (check_failed) continue;
            // Out edges second
            for (auto& node : last_dag.Out[ wt_edge.edge_pair.second ]) {
                if (node == wt_edge.edge_pair.first) continue;
                if (! merged_nodes[ node ]) continue;
                if ( poset_int_mapping[ node ] >= poset_int_mapping[ wt_edge.edge_pair.second ] +2 ) continue;
                check_failed = true;
                break;
            }
            if (check_failed) continue;


            // merging
            connected_components.join_by_name( wt_edge.edge_pair.first, wt_edge.edge_pair.second );
            merged_nodes[ wt_edge.edge_pair.first ] = true;
            merged_nodes[ wt_edge.edge_pair.second ] = true;
            num_nodes_decrease++;
        }


        temperature *= coarsen_par.temperature_multiplier;
        temperature_increase_iteration++;
    }


    // Getting components to contract and adding graph contraction
    if (num_nodes_decrease > 0 ) {
        std::vector<std::vector<int>> partition_vec = connected_components.get_connected_components();
        std::vector<std::unordered_set<VertexType>> partition;
        for ( auto& vec : partition_vec ) {
            partition.emplace_back(vec.cbegin(), vec.cend());
        }
        add_contraction(partition);
    }
    return num_nodes_decrease;

}

RETURN_STATUS SquashA::run_contractions() {
    if (min_nodes == 0) {
        min_nodes = original_inst->numberOfProcessors() * 1000;
    }
    std::cout << "Coarsen Step: " << dag_history.size() << ", Number of nodes: " << dag_history.back()->numberOfVertices() << std::endl;

    Biased_Random_with_side_bias coin( coarsen_par.edge_sort_ratio );
    int no_change_in_a_row = 0;
    while( no_change_in_a_row < coarsen_par.num_rep_without_node_decrease && dag_history.back()->numberOfVertices() > min_nodes ) {
        int diff;
        if ( coin.get_flip() ) {
            diff = run_and_add_contraction(Contract_Edge_Decrease);
        }
        else {
            diff = run_and_add_contraction(Contract_Edge_Weight);
        }
        std::cout << "Coarsen Step: " << dag_history.size() << ", Number of nodes: " << dag_history.back()->numberOfVertices() << std::endl;
        
        if (diff == 0) {
            no_change_in_a_row++;
        }
        else {
            no_change_in_a_row = 0;
        }
    }

    return SUCCESS;
}