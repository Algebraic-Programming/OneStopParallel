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

#include "scheduler/ContractRefineScheduler/coarsen/coarsen_history.hpp"

CoarsenHistory::CoarsenHistory(const DAG& graph, const CoarsenParams coarsen_par_) : coarsen_par(coarsen_par_) {
    dag_evolution.push_back( std::make_unique<DAG>(graph) );
};

CoarsenHistory::CoarsenHistory(const ComputationalDag& graph, const CoarsenParams coarsen_par_) : coarsen_par(coarsen_par_) {
    dag_evolution.push_back( std::make_unique<DAG>(graph) );
};


void CoarsenHistory::add_dag( std::unique_ptr<DAG> graph, std::unique_ptr<std::unordered_map<int, int>> contraction_map ) {

    // Begin checking for correct history
    const DAG& previous_graph = *(dag_evolution.back());
    std::vector<bool> is_being_mapped_to(graph->n ,false);
    for (int i = 0; i < previous_graph.n; i++) {
        if ((contraction_map->at(i) < 0) || (contraction_map->at(i) >= graph->n)) {
            throw std::invalid_argument("Image of contraction map is not contained in the vertices of incoming graph.");
        }
        is_being_mapped_to[contraction_map->at(i)] = true;
    }
    if (!(std::all_of(is_being_mapped_to.cbegin(), is_being_mapped_to.cend(), [](bool is_mapped_to) { return is_mapped_to; }))) {
        throw std::invalid_argument("The contraction map is not surjective.");
    }
    // End checking for correct history

    // inverse generation
    std::unique_ptr<std::unordered_map< int , std::unordered_set<int> >> expansion_map = std::make_unique<std::unordered_map< int , std::unordered_set<int> >>();
    expansion_map->reserve(graph->n);
    for (int i = 0; i < previous_graph.n; i++) {
        if ( expansion_map->find( contraction_map->at(i) ) !=  expansion_map->cend() ) {
            expansion_map->at(contraction_map->at(i)).emplace( i );
        } else {
            expansion_map->insert( { contraction_map->at(i), std::unordered_set<int>({i})} );
        }
    }

    // Adding to history
    dag_evolution.push_back(move(graph));
    contraction_maps.push_back(move(contraction_map));
    expansion_maps.push_back(move(expansion_map));
}

void CoarsenHistory::add_graph_contraction( const std::vector<std::unordered_set<int>>& sets_to_contract )
{
    const DAG& last_dag = *(dag_evolution.back());
    std::pair<DAG, std::unordered_map<int, int >> contraction_pair = last_dag.contracted_graph_without_loops(sets_to_contract);
    std::unique_ptr<DAG> dag_ptr = std::make_unique<DAG>( contraction_pair.first );
    std::unique_ptr<std::unordered_map<int, int >> contraction_map_ptr = std::make_unique<std::unordered_map<int, int >>( contraction_pair.second.cbegin(), contraction_pair.second.cend() ) ;
    add_dag( move(dag_ptr), move(contraction_map_ptr) );
}

void CoarsenHistory::add_graph_contraction( const std::vector<std::vector<int>>& sets_to_contract )
{
    const DAG& last_dag = *(dag_evolution.back());
    std::pair<DAG, std::unordered_map<int, int >> contraction_pair = last_dag.contracted_graph_without_loops(sets_to_contract);
    std::unique_ptr<DAG> dag_ptr = std::make_unique<DAG>( contraction_pair.first );
    std::unique_ptr<std::unordered_map<int, int >> contraction_map_ptr = std::make_unique<std::unordered_map<int, int >>( contraction_pair.second.cbegin(), contraction_pair.second.cend() ) ;
    add_dag( move(dag_ptr), move(contraction_map_ptr) );
}

int CoarsenHistory::run_and_add_contraction(const contract_edge_sort edge_sort_type , const CoarsenParams& override_coarsen_param)
{
    const DAG& last_dag = *(dag_evolution.back());

    std::vector<int> poset_int_mapping;
    if ( thue_coin.get_flip() ) {
        poset_int_mapping = last_dag.get_strict_poset_integer_map( override_coarsen_param.noise, override_coarsen_param.poisson_par );
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
    for (int i = 0; i < last_dag.n; i++) {
        connected_components.add_object(i, last_dag.workW[i]);
    }

    // std::unordered_set<int> merged_nodes;
    std::vector<bool> merged_nodes( last_dag.n, false );

    int num_nodes_decrease = 0;
    int num_nodes_aim = last_dag.n - int( last_dag.n / override_coarsen_param.geom_decay_num_nodes );

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
            if (num_nodes_decrease >= num_nodes_aim) break;
        }


        temperature *= coarsen_par.temperature_multiplier;
        temperature_increase_iteration++;
    }


    // Getting components to contract and adding graph contraction
    if (num_nodes_decrease > 0 ) {
        add_graph_contraction( connected_components.get_connected_components() );
    }
    return num_nodes_decrease;
}

void CoarsenHistory::run_dag_evolution(const int min_number_of_nodes)
{
    std::cout   << "Coarsen Step: " << dag_evolution.size()
                << ", Number of nodes: " << dag_evolution.back()->n
                << ", Number of edges: " << dag_evolution.back()->comm_edge_W.size()
                << ", Log ratio: " << std::log(dag_evolution.back()->comm_edge_W.size()) / std::log(dag_evolution.back()->n) << std::endl;

    Biased_Random_with_side_bias coin( coarsen_par.edge_sort_ratio );
    int no_change_in_a_row = 0;
    while( no_change_in_a_row < coarsen_par.num_rep_without_node_decrease
        && dag_evolution.back()->n > min_number_of_nodes ) {
        int diff;
        if ( coin.get_flip() ) {
            diff = run_and_add_contraction(Contract_Edge_Decrease);
        }
        else {
            diff = run_and_add_contraction(Contract_Edge_Weight);
        }
        std::cout   << "Coarsen Step: " << dag_evolution.size()
                    << ", Number of nodes: " << dag_evolution.back()->n
                    << ", Number of edges: " << dag_evolution.back()->comm_edge_W.size()
                    << ", Log ratio: " << std::log(dag_evolution.back()->comm_edge_W.size()) / std::log(dag_evolution.back()->n) << std::endl;
        
        if (diff == 0) {
            no_change_in_a_row++;
        }
        else {
            no_change_in_a_row = 0;
        }
    }
}