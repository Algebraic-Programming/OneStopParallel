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

#include "algorithms/ContractRefineScheduler/BalDMixR.hpp"

BalDMixR::BalDMixR(const Coarse_Scheduler_Params params_, unsigned mixing_loops_)
    :   Scheduler(), params( std::make_unique<Coarse_Scheduler_Params>(params_) ), mixing_loops(mixing_loops_) { };


BalDMixR::BalDMixR(unsigned timelimit, const Coarse_Scheduler_Params params_, unsigned mixing_loops_)
    :   Scheduler(timelimit), params( std::make_unique<Coarse_Scheduler_Params>(params_) ), mixing_loops(mixing_loops_) { };

void BalDMixR::auto_mixing_loops(const BspInstance& instance) {
    float mixing_amount = std::log(instance.getComputationalDag().numberOfEdges()+2);
    mixing_loops = unsigned(10*(mixing_amount+1));

    mixing_loops = std::max(unsigned(10), mixing_loops);
    mixing_loops = std::max(instance.numberOfProcessors(), mixing_loops);
}

void BalDMixR::run_initialise() {
    std::cout << "Generating initial schedule" << std::endl;

    unsigned queue_length = 10;
    unsigned max_stable_length = 2*queue_length;
    std::list<unsigned> num_supersteps;
    num_supersteps.push_back(1);
    unsigned running_max = std::accumulate(num_supersteps.begin(), num_supersteps.end(), 0, [](const unsigned maxi, const unsigned valu) { return std::max(maxi,valu); });
    unsigned stable_length = 0;
    while ((stable_length<max_stable_length) || (num_supersteps.size()<queue_length)) {
        std::cout << "Refining: current number of supersteps: " << helper_vessel->loose_schedule.supersteps.size() << std::endl;
        helper_vessel->loose_schedule.run_superstep_improvement_iteration();
        if (randInt(2) == 0 ) {
            helper_vessel->loose_schedule.run_joining_supersteps_improvements(randInt(2)==0);
        }

        num_supersteps.push_back(helper_vessel->loose_schedule.supersteps.size());
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
    //     std::cout << "Refining: current number of supersteps: " << helper_vessel->loose_schedule.supersteps.size() << std::endl;
    //     change = helper_vessel->loose_schedule.run_superstep_improvement_iteration();
    // }

    int no_change_counter = 0;
    bool parity = false;
    while (no_change_counter < 2) {
        std::cout << "Combining: current number of supersteps: " << helper_vessel->loose_schedule.supersteps.size() << std::endl;
        if (helper_vessel->loose_schedule.run_joining_supersteps_improvements(parity)) {
            no_change_counter = 0;
        }
        else {
            no_change_counter++;
        }
        parity = !parity;
    }
    std::cout << "Finished initialising schedule" << std::endl << std::endl;
}


void BalDMixR::run_mix_loops() {
    unsigned no_changes = 0;
    for (unsigned i = 0; i < mixing_loops; i++) {
        bool temp1 = helper_vessel->loose_schedule.run_superstep_improvement_iteration();
        bool temp2 = helper_vessel->loose_schedule.run_joining_supersteps_improvements(randInt(2) == 0);
        std::cout << "Mixing: current number of supersteps: " << helper_vessel->loose_schedule.supersteps.size() << std::endl;

        if (temp1 || temp2) {
            no_changes = 0;
        } else {
            no_changes++;
        }

        if (no_changes > 4) break;
    }

    bool change = true;
    while(change) {
        change = helper_vessel->loose_schedule.run_superstep_improvement_iteration();
    }
}

void BalDMixR::run_processor_assingment(const std::vector<std::vector<unsigned>>& processsor_comm_costs) {
    helper_vessel->loose_schedule.run_processor_assignment( processsor_comm_costs );
}

void BalDMixR::run_superstep_collapses(const BspInstance& instance) {
    run_processor_assingment(instance.getArchitecture().sendCostMatrix());

    bool parity = false;
    unsigned no_changes = 0;
    while( no_changes < 2 ) {
        bool temp2 = helper_vessel->loose_schedule.run_joining_supersteps_improvements(parity,
                                                                        false,
                                                                        (instance.getArchitecture().communicationCosts()+1) / 2, // this is dangerous as it does not take the whole communication schedule into account which is why it is halved
                                                                        instance.getArchitecture().synchronisationCosts(),
                                                                        true);

        parity = !parity;

        if (temp2) {
            no_changes = 0;
            run_processor_assingment(instance.getArchitecture().sendCostMatrix());
            std::cout << "Collapsing Supersteps: current number of supersteps: " << helper_vessel->loose_schedule.supersteps.size() << std::endl;
        } else {
            no_changes++;
        }
    }
}

BspSchedule BalDMixR::produce_bsp_schedule(const BspInstance& instance) {
    
    std::vector<unsigned int> node_to_superstep_assignment(helper_vessel->G.n);
    std::vector<unsigned int> node_to_processor_assignment(helper_vessel->G.n);

    std::unordered_map<unsigned, unsigned> superstep_id_to_superstep_sequence;
    superstep_id_to_superstep_sequence.reserve(helper_vessel->loose_schedule.superstep_ordered_ids.size());
    for(unsigned i = 0; i< helper_vessel->loose_schedule.superstep_ordered_ids.size(); i++) {
        superstep_id_to_superstep_sequence[helper_vessel->loose_schedule.superstep_ordered_ids[i]] = i;
    }

    for (auto& sstep : helper_vessel->loose_schedule.supersteps) {
        unsigned clmp_ind = 0;
        for (auto& clmp : sstep.collection ) {
            for (auto& node : clmp.node_set) {
                node_to_superstep_assignment[node] = superstep_id_to_superstep_sequence.at(sstep.id);
                node_to_processor_assignment[node] = sstep.get_current_allocation()[clmp_ind];
            }
            clmp_ind++;
        }
    }

    BspSchedule bspsched(instance, node_to_processor_assignment, node_to_superstep_assignment);

    bspsched.setAutoCommunicationSchedule();

    return bspsched;
}

std::pair<RETURN_STATUS, BspSchedule> BalDMixR::computeSchedule(const BspInstance& instance) {
    if (instance.numberOfProcessors() == 1) {
        Serial dummy;
        return dummy.computeSchedule(instance);
    }


    // Updating parameters
    if (mixing_loops == 0) auto_mixing_loops(instance);
    params = std::make_unique<Coarse_Scheduler_Params>( instance.numberOfProcessors(),
                                                        params->balance_threshhold,
                                                        params->part_algo,
                                                        params->coin_type,
                                                        params->clumps_per_partition,
                                                        params->nodes_per_clump,
                                                        params->nodes_per_partition,
                                                        params->max_weight_for_flag,
                                                        params->balanced_cut_ratio,
                                                        params->min_weight_for_split,
                                                        params->hill_climb_simple_improvement_attemps);

    // Setting up structures
    helper_vessel = std::make_unique<support_structure>( instance.getComputationalDag(), *params );

    // Running algorithm
    run_initialise();
    run_mix_loops();
    run_processor_assingment( instance.getArchitecture().sendCostMatrix() );
    run_superstep_collapses(instance);
    std::cout << "Schedule Complete" << std::endl;
    BspSchedule sched = produce_bsp_schedule(instance);

    // Deleting structures
    helper_vessel.reset();

    return std::make_pair(SUCCESS , sched);
}

BalDMixR::support_structure::support_structure(const ComputationalDag& cdag, const Coarse_Scheduler_Params& para)
    :   G( DAG( cdag )), graph(G), loose_schedule( LooseSchedule(graph, para) )
{
    std::vector<std::unordered_set<int>> connected_comp = graph.weakly_connected_components();
    loose_schedule.add_loose_superstep(0, connected_comp);
};
