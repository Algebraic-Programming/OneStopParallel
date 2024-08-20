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

#include "scheduler/ContractRefineScheduler/contract_refine_scheduler.hpp"

void CoarseRefineScheduler::run_coarsen()
{
    std::cout << "Begin Coarsening" << std::endl;
    dag_evolution.run_dag_evolution(params.min_nodes_after_coarsen);
    std::cout << "End Coarsening" << std::endl << std::endl;
}

void CoarseRefineScheduler::run_schedule_initialise()
{
    std::cout << "Generating initial schedule" << std::endl;
    subdag_conversion.reserve(dag_evolution.dag_evolution.size());
    for (auto&& graph : dag_evolution.dag_evolution) {
        std::unique_ptr<SubDAG> new_subdag = std::make_unique<SubDAG>(*graph);
        subdag_conversion.push_back(move(new_subdag));
    }
    assert( dag_evolution.dag_evolution.size() == subdag_conversion.size() );

    active_subdag = subdag_conversion.size()-1;
    active_loose_schedule = std::make_unique<LooseSchedule>( *(subdag_conversion[active_subdag]) , params.coarse_schedule_params_initial);

    std::vector<std::unordered_set<int>> connected_comp = subdag_conversion[active_subdag]->weakly_connected_components();
    std::cout << "Number of connected components: " << connected_comp.size() << std::endl;
    active_loose_schedule->add_loose_superstep(0, connected_comp);

    unsigned queue_length = 10;
    unsigned max_stable_length = 2*queue_length;
    std::list<unsigned> num_supersteps;
    num_supersteps.push_back(1);
    unsigned running_max = std::accumulate(num_supersteps.begin(), num_supersteps.end(), 0, [](const unsigned maxi, const unsigned valu) { return std::max(maxi,valu); });
    unsigned stable_length = 0;
    while ((stable_length<max_stable_length) || (num_supersteps.size()<queue_length)) {
        std::cout << "Refining: current number of supersteps: " << active_loose_schedule->supersteps.size() << std::endl;
        active_loose_schedule->run_superstep_improvement_iteration();
        if (randInt(2) == 0 ) {
            run_schedule_superstep_joinings(randInt(2)==0);
        }

        num_supersteps.push_back(active_loose_schedule->supersteps.size());
        if (num_supersteps.size() > queue_length) {
            num_supersteps.pop_front();
        }
        unsigned new_running_max = std::accumulate(num_supersteps.begin(), num_supersteps.end(), 0, [](const unsigned maxi, const unsigned valu) { return std::max(maxi,valu); });
        if ( new_running_max > running_max ) {
            stable_length = 0;
        } else if (new_running_max == running_max) {
            stable_length++;
        } else {
            stable_length = max_stable_length;
        }
        running_max = new_running_max;
    }

    // bool change = true;
    // while (change) {
    //     std::cout << "Refining: current number of supersteps: " << active_loose_schedule->supersteps.size() << std::endl;
    //     change = active_loose_schedule->run_superstep_improvement_iteration();
    // }

    int no_change_counter = 0;
    bool parity = false;
    while (no_change_counter < 2) {
        std::cout << "Combining: current number of supersteps: " << active_loose_schedule->supersteps.size() << std::endl;
        if (run_schedule_superstep_joinings(parity)) {
            no_change_counter = 0;
        }
        else {
            no_change_counter++;
        }
        parity = !parity;
    }

    std::cout << "Finished initialising schedule" << std::endl << std::endl;
}

bool CoarseRefineScheduler::run_schedule_refine()
{
    bool output = active_loose_schedule->run_superstep_improvement_iteration();
    return output;
}

void CoarseRefineScheduler::run_schedule_evolve()
{
    if (active_subdag <= 0) return;

    std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>& old_supersteps = active_loose_schedule->supersteps;

    int new_active_subdag = active_subdag-1;
    std::vector<unsigned> new_superstep_ordered_ids = active_loose_schedule->superstep_ordered_ids;
    std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator> new_supersteps;

    int num_coarsened_graphs = dag_evolution.dag_evolution.size();
    const Coarse_Scheduler_Params new_params = Coarse_Scheduler_Params::lin_comb(  params.coarse_schedule_params_initial,
                                                                                        params.coarse_schedule_params_final,
                                                                                        std::make_pair( num_coarsened_graphs-active_subdag , active_subdag-1  ));

    for (auto& old_sstep : old_supersteps) {
        // std::vector<unsigned> new_allocation = old_sstep.get_current_allocation();
        std::multiset<Clump, typename Clump::Comparator> new_collection;
        
        for (auto& clump : old_sstep.collection) {
            std::unordered_set<int> new_node_set;

            for (auto& old_node : clump.node_set) {
                new_node_set.insert( dag_evolution.expansion_maps[new_active_subdag]->at(old_node).cbegin(), dag_evolution.expansion_maps[new_active_subdag]->at(old_node).cend() );
            }

            Clump new_clump( *(subdag_conversion[new_active_subdag]), new_node_set );
            // std::cout << clump.total_weight << " " << new_clump.total_weight << std::endl;
            assert( clump.total_weight == new_clump.total_weight );

            new_collection.insert(new_clump);
        }

        new_supersteps.emplace(old_sstep.id, new_collection, old_sstep.get_current_allocation(), old_sstep.get_imbalance(), new_params);
    }

    active_loose_schedule = std::make_unique<LooseSchedule>( *(subdag_conversion[new_active_subdag]), new_params, new_superstep_ordered_ids, new_supersteps);

    active_subdag--;
}

void CoarseRefineScheduler::run_schedule_evolution( const bool only_above_thresh, const unsigned comm_cost_multiplier, const unsigned com_cost_addition )
{
    if (active_subdag < 0) return;

    std::cout << "Starting expansion" << std::endl;

    while (active_subdag > 0) {
        std::cout << "Refining coarsened graph " << active_subdag << ": current number of supersteps: " << active_loose_schedule->supersteps.size() << std::endl;
        run_schedule_evolve();
        run_schedule_refine();
        if (randInt(3) != 0) {
            run_schedule_superstep_joinings( randInt(2) == 0, only_above_thresh, comm_cost_multiplier, com_cost_addition );
        }
    }

    std::cout << "Finished expansion" << std::endl << std::endl;
    std::cout << "Final refinements" << std::endl;

    int no_change = 0;
    while (no_change < params.number_of_final_no_change_reps) {
        std::cout << "Refining: current number of supersteps: " << active_loose_schedule->supersteps.size() << std::endl;
        if (run_schedule_refine()) {
            no_change = 0;
        }
        else {
            no_change++;
        }
    }
}


bool CoarseRefineScheduler::run_schedule_superstep_joinings(const bool parity, const bool only_above_thresh, const unsigned comm_cost_multiplier, const unsigned com_cost_addition)
{
    return active_loose_schedule->run_joining_supersteps_improvements(parity, only_above_thresh, comm_cost_multiplier, com_cost_addition);
}

void CoarseRefineScheduler::run_all(const bool only_above_thresh_initial, const bool only_above_thresh_final, const unsigned comm_cost_multiplier, const unsigned com_cost_addition)
{
    run_coarsen();
    run_schedule_initialise();
    run_schedule_evolution(only_above_thresh_initial, comm_cost_multiplier, com_cost_addition);

    int no_change_counter = 0;
    bool parity = false;
    while (no_change_counter < 2) {
        std::cout << "Collecting: current number of supersteps: " << active_loose_schedule->supersteps.size() << std::endl;
        if (run_schedule_superstep_joinings(parity, only_above_thresh_final, comm_cost_multiplier, com_cost_addition)) {
            no_change_counter = 0;
        }
        else {
            no_change_counter++;
        }
        parity = !parity;
    }
    std::cout << "Schedule complete" << std::endl << std::endl;
}

std::vector<std::vector<std::vector<int>>> CoarseRefineScheduler::get_loose_schedule() const
{
    if (active_subdag != 0) {
        std::runtime_error("Schedule Computation is not complete.");
    }

    return active_loose_schedule->get_current_schedule();
}

std::unordered_map<int, std::pair<unsigned, unsigned>> CoarseRefineScheduler::get_loose_node_schedule_allocation() const
{
    if (active_subdag != 0) {
        std::runtime_error("Schedule Computation is not complete.");
    }

    return active_loose_schedule->get_current_node_schedule_allocation();
}

void CoarseRefineScheduler::print_loose_schedule() const {
    if (active_subdag != 0) {
        std::runtime_error("Schedule Computation is not complete.");
    }

    active_loose_schedule->print_current_schedule();
}


std::vector< std::pair<unsigned, unsigned>> CoarseRefineScheduler::produce_node_computing_schedule() const {
    if (active_subdag != 0) {
        std::runtime_error("Schedule Computation is not complete.");
    }

    std::vector<unsigned int> node_to_superstep_assignment(original_graph.n);
    std::vector<unsigned int> node_to_processor_assignment(original_graph.n);

    // generating efficient map superstep_id to which superstep
    std::unordered_map<unsigned, unsigned> superstep_id_to_superstep_sequence;
    superstep_id_to_superstep_sequence.reserve(active_loose_schedule->superstep_ordered_ids.size());
    for(unsigned i = 0; i< active_loose_schedule->superstep_ordered_ids.size(); i++) {
        superstep_id_to_superstep_sequence[active_loose_schedule->superstep_ordered_ids[i]] = i;
    }
    std::unordered_map<unsigned, unsigned> id_to_collection_ind;
    id_to_collection_ind.reserve(active_loose_schedule->superstep_ordered_ids.size());
    unsigned sstep_ind = 0;
    for ( auto& sstep : active_loose_schedule->supersteps ) {
        id_to_collection_ind[ sstep.id ] = sstep_ind;
        sstep_ind++;
    }

    // assigning node to superstep
    for( auto& sstep : active_loose_schedule->supersteps ) {
        for ( auto& clmp : sstep.collection ) {
            for (auto& node : clmp.node_set) {
                node_to_superstep_assignment[node] = superstep_id_to_superstep_sequence.at(sstep.id);
            }
        }
    }

    // assigning node to processor
    std::vector<bool> node_assigned_processor(original_graph.n, false);
    for (auto& id : active_loose_schedule->superstep_ordered_ids) {
        std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator sstep = active_loose_schedule->supersteps.begin();
        std::advance(sstep, id_to_collection_ind.at(id) );
    // for( auto& sstep : active_loose_schedule->supersteps ) {
        std::vector<std::vector<long long unsigned>> partition_to_processor_comm_saving( params.number_of_partitions, std::vector<long long unsigned>(params.number_of_partitions, 0) );

        // computing partition to processor communication savings
        unsigned clmp_index = 0;
        std::vector<unsigned> allocation = sstep->get_current_allocation();
        for ( auto& clmp : sstep->collection ) {
            for ( auto& node : clmp.node_set ) {
                for ( auto& parent : original_graph.In[node]) {
                    if (node_assigned_processor[parent]) {
                        partition_to_processor_comm_saving[ allocation[clmp_index] ][ node_to_processor_assignment[parent] ] += original_graph.commW[parent];
                    }
                }
            }
            clmp_index++;
        }

        // deciding partition to processor allocation
        std::vector<unsigned> partition_to_processor_allocation = max_perfect_matching_for_complete_bipartite(partition_to_processor_comm_saving);

        // allocating nodes
        clmp_index = 0;
        for ( auto& clmp : sstep->collection ) {
            for ( auto& node : clmp.node_set ) {
                node_to_processor_assignment[node] = partition_to_processor_allocation[ allocation[clmp_index] ];
                node_assigned_processor[node] = true;
            }
            clmp_index++;
        }
    }

    // check that all nodes have been assigned a processor
    assert( std::all_of(node_assigned_processor.cbegin(), node_assigned_processor.cend(), []( const bool& has_been ) {return has_been;} ) );

    // producing output
    std::vector<std::pair<unsigned, unsigned>> output;
    output.reserve(original_graph.n);
    for (unsigned i = 0; i < original_graph.n; i++) {
        output.emplace_back( node_to_superstep_assignment[i], node_to_processor_assignment[i] );
    }

    return output;
}



std::vector<std::vector<std::vector<unsigned>>> CoarseRefineScheduler::get_computing_schedule() const {
    unsigned number_of_supersteps = active_loose_schedule->superstep_ordered_ids.size();

    std::vector<std::vector<std::vector<unsigned>>> sched(number_of_supersteps, std::vector< std::vector<unsigned> >( params.number_of_partitions ));

    std::vector< std::pair<unsigned, unsigned>> node_sched  = produce_node_computing_schedule();
    for (unsigned i = 0; i < node_sched.size(); i++) {
        sched[ node_sched[i].first ][ node_sched[i].second ].emplace_back(i);
    }

    return sched;
}

void CoarseRefineScheduler::print_computing_schedule() const {
    std::cout << std::endl << "Schedule:" << std::endl;
    std::vector<std::vector<std::vector<unsigned>>> schedule = get_computing_schedule();
    std::vector<float> superstep_imbalances =  active_loose_schedule->get_current_superstep_imbalances_in_order();

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




std::vector< std::pair<unsigned, unsigned>> CoarseRefineScheduler::produce_node_computing_schedule( const std::vector<std::vector<unsigned>>& processsor_comm_costs ) const {
    assert( processsor_comm_costs.size() == params.number_of_partitions );
    for (long unsigned p = 0; p<processsor_comm_costs.size(); p++) {
        assert( processsor_comm_costs[p].size() == params.number_of_partitions );
    }

    if (active_subdag != 0) {
        std::runtime_error("Schedule Computation is not complete.");
    }

    std::vector<unsigned int> node_to_superstep_assignment(original_graph.n);
    std::vector<unsigned int> node_to_processor_assignment(original_graph.n);

    // generating efficient map superstep_id to which superstep
    std::unordered_map<unsigned, unsigned> superstep_id_to_superstep_sequence;
    superstep_id_to_superstep_sequence.reserve(active_loose_schedule->superstep_ordered_ids.size());
    for(unsigned i = 0; i< active_loose_schedule->superstep_ordered_ids.size(); i++) {
        superstep_id_to_superstep_sequence[active_loose_schedule->superstep_ordered_ids[i]] = i;
    }
    std::unordered_map<unsigned, unsigned> id_to_collection_ind;
    id_to_collection_ind.reserve(active_loose_schedule->superstep_ordered_ids.size());
    unsigned sstep_ind = 0;
    for ( auto& sstep : active_loose_schedule->supersteps ) {
        id_to_collection_ind[ sstep.id ] = sstep_ind;
        sstep_ind++;
    }

    // assigning node to superstep
    for( auto& sstep : active_loose_schedule->supersteps ) {
        for ( auto& clmp : sstep.collection ) {
            for (auto& node : clmp.node_set) {
                node_to_superstep_assignment[node] = superstep_id_to_superstep_sequence.at(sstep.id);
            }
        }
    }

    // assigning node to processor
    std::vector<bool> node_assigned_processor(original_graph.n, false);
    for (auto& id : active_loose_schedule->superstep_ordered_ids) {
        std::multiset<LooseSuperStep<Clump>, typename LooseSuperStep<Clump>::Comparator>::iterator sstep = active_loose_schedule->supersteps.begin();
        std::advance(sstep, id_to_collection_ind.at(id) );
    // for( auto& sstep : active_loose_schedule->supersteps ) {
        std::vector<std::vector<long long unsigned>> partition_to_processor_comm_cost( params.number_of_partitions, std::vector<long long unsigned>(params.number_of_partitions, 0) );

        // computing partition to processor communication savings
        unsigned clmp_index = 0;
        std::vector<unsigned> allocation = sstep->get_current_allocation();
        for ( auto& clmp : sstep->collection ) {
            for ( auto& node : clmp.node_set ) {
                for ( auto& parent : original_graph.In[node]) {
                    if (node_assigned_processor[parent]) {
                        for (long unsigned procssr = 0; procssr < params.number_of_partitions; procssr++) {
                            if ( procssr ==  node_to_processor_assignment[parent] ) continue;

                            partition_to_processor_comm_cost[ allocation[clmp_index] ][ procssr ] += original_graph.commW[parent]*processsor_comm_costs[node_to_processor_assignment[parent]][procssr];
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
    }

    // check that all nodes have been assigned a processor
    assert( std::all_of(node_assigned_processor.cbegin(), node_assigned_processor.cend(), []( const bool& has_been ) {return has_been;} ) );

    // producing output
    std::vector<std::pair<unsigned, unsigned>> output;
    output.reserve(original_graph.n);
    for (unsigned i = 0; i < original_graph.n; i++) {
        output.emplace_back( node_to_superstep_assignment[i], node_to_processor_assignment[i] );
    }

    return output;
}



std::vector<std::vector<std::vector<unsigned>>> CoarseRefineScheduler::get_computing_schedule( const std::vector<std::vector<unsigned>>& processsor_comm_costs ) const {
    unsigned number_of_supersteps = active_loose_schedule->superstep_ordered_ids.size();

    std::vector<std::vector<std::vector<unsigned>>> sched(number_of_supersteps, std::vector< std::vector<unsigned> >( params.number_of_partitions ));

    std::vector< std::pair<unsigned, unsigned>> node_sched  = produce_node_computing_schedule( processsor_comm_costs );
    for (unsigned i = 0; i < node_sched.size(); i++) {
        sched[ node_sched[i].first ][ node_sched[i].second ].emplace_back(i);
    }

    return sched;
}

void CoarseRefineScheduler::print_computing_schedule( const std::vector<std::vector<unsigned>>& processsor_comm_costs ) const {
    std::cout << std::endl << "Schedule:" << std::endl;
    std::vector<std::vector<std::vector<unsigned>>> schedule = get_computing_schedule( processsor_comm_costs );
    std::vector<float> superstep_imbalances =  active_loose_schedule->get_current_superstep_imbalances_in_order();

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







BspSchedule CoarseRefineScheduler::produce_bsp_schedule(const BspInstance &bsp_instance) const {
    if (active_subdag != 0) {
        std::runtime_error("Schedule Computation is not complete.");
    }
    
    std::vector<unsigned int> node_to_superstep_assignment(original_graph.n);
    std::vector<unsigned int> node_to_processor_assignment(original_graph.n);

    std::vector<std::pair<unsigned, unsigned>> node_comp_sched = produce_node_computing_schedule( bsp_instance.sendCostMatrix() );

    for (unsigned i = 0; i < original_graph.n; i++) {
        node_to_superstep_assignment[i] = node_comp_sched[i].first;
        node_to_processor_assignment[i] = node_comp_sched[i].second;
    }

    BspSchedule bspsched(bsp_instance, node_to_processor_assignment, node_to_superstep_assignment);

    bspsched.setAutoCommunicationSchedule();

    return bspsched;
}
