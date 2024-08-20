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

#include "scheduler/ContractRefineScheduler/CoBalDMixR.hpp"

CoBalDMixR::support_structure::support_structure(const ComputationalDag& cdag, const CoarseRefineScheduler_parameters& para)
    :   original_graph( DAG( cdag )), coarse_refiner(original_graph, para) { };

CoBalDMixR::CoBalDMixR(const CoarseRefineScheduler_parameters params_)
    : Scheduler(), params( std::make_unique<CoarseRefineScheduler_parameters>(params_) ) { };

CoBalDMixR::CoBalDMixR(unsigned timelimit, const CoarseRefineScheduler_parameters params_)
    : Scheduler(timelimit), params( std::make_unique<CoarseRefineScheduler_parameters>(params_)) { };

void CoBalDMixR::run_processor_assingment(const std::vector<std::vector<unsigned>>& processsor_comm_costs) {
    helper_vessel->coarse_refiner.active_loose_schedule->run_processor_assignment( processsor_comm_costs );
}

void CoBalDMixR::run_superstep_collapses(const BspInstance& instance) {
    run_processor_assingment(instance.getArchitecture().sendCostMatrix());

    bool parity = false;
    unsigned no_changes = 0;
    while( no_changes < 2 ) {
        bool temp2 = helper_vessel->coarse_refiner.active_loose_schedule->run_joining_supersteps_improvements(parity,
                                                                        false,
                                                                        (instance.getArchitecture().communicationCosts()+1) / 2, // this is dangerous as it does not take the whole communication schedule into account which is why it is halved
                                                                        instance.getArchitecture().synchronisationCosts(),
                                                                        true);

        parity = !parity;

        if (temp2) {
            no_changes = 0;
            run_processor_assingment(instance.getArchitecture().sendCostMatrix());
            std::cout << "Collapsing Supersteps: current number of supersteps: " << helper_vessel->coarse_refiner.active_loose_schedule->supersteps.size() << std::endl;
        } else {
            no_changes++;
        }
    }
}

BspSchedule CoBalDMixR::produce_bsp_schedule(const BspInstance& instance) {
    
    std::vector<unsigned int> node_to_superstep_assignment(helper_vessel->original_graph.n);
    std::vector<unsigned int> node_to_processor_assignment(helper_vessel->original_graph.n);

    std::unordered_map<unsigned, unsigned> superstep_id_to_superstep_sequence;
    superstep_id_to_superstep_sequence.reserve(helper_vessel->coarse_refiner.active_loose_schedule->superstep_ordered_ids.size());
    for(unsigned i = 0; i< helper_vessel->coarse_refiner.active_loose_schedule->superstep_ordered_ids.size(); i++) {
        superstep_id_to_superstep_sequence[helper_vessel->coarse_refiner.active_loose_schedule->superstep_ordered_ids[i]] = i;
    }

    for (auto& sstep : helper_vessel->coarse_refiner.active_loose_schedule->supersteps) {
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

std::pair<RETURN_STATUS, BspSchedule> CoBalDMixR::computeSchedule(const BspInstance &instance) {
    if (instance.numberOfProcessors() == 1) {
        Serial dummy;
        return dummy.computeSchedule(instance);
    }

    // Updating parameters
    params = std::make_unique<CoarseRefineScheduler_parameters>(
        Coarse_Scheduler_Params(
            instance.numberOfProcessors(),
            params->coarse_schedule_params_initial.balance_threshhold,
            params->coarse_schedule_params_initial.part_algo,
            params->coarse_schedule_params_initial.coin_type,
            params->coarse_schedule_params_initial.clumps_per_partition,
            params->coarse_schedule_params_initial.nodes_per_clump,
            params->coarse_schedule_params_initial.nodes_per_partition,
            params->coarse_schedule_params_initial.max_weight_for_flag,
            params->coarse_schedule_params_initial.balanced_cut_ratio,
            params->coarse_schedule_params_initial.min_weight_for_split,
            params->coarse_schedule_params_initial.hill_climb_simple_improvement_attemps
        ),
        Coarse_Scheduler_Params(
            instance.numberOfProcessors(),
            params->coarse_schedule_params_final.balance_threshhold,
            params->coarse_schedule_params_final.part_algo,
            params->coarse_schedule_params_final.coin_type,
            params->coarse_schedule_params_final.clumps_per_partition,
            params->coarse_schedule_params_final.nodes_per_clump,
            params->coarse_schedule_params_final.nodes_per_partition,
            params->coarse_schedule_params_final.max_weight_for_flag,
            params->coarse_schedule_params_final.balanced_cut_ratio,
            params->coarse_schedule_params_final.min_weight_for_split,
            params->coarse_schedule_params_final.hill_climb_simple_improvement_attemps
        ),
        params->coarsen_param,
        (params->min_nodes_after_coarsen + std::max(params->number_of_partitions, 1) - 1) / std::max(params->number_of_partitions, 1),
        params->number_of_final_no_change_reps
    );

    // Setting up structures
    helper_vessel = std::make_unique<support_structure>( instance.getComputationalDag(), *params );

    helper_vessel->coarse_refiner.run_coarsen();
    helper_vessel->coarse_refiner.run_schedule_initialise();
    helper_vessel->coarse_refiner.run_schedule_evolution();
    run_processor_assingment( instance.getArchitecture().sendCostMatrix());
    run_superstep_collapses(instance);

    std::cout << "Schedule Complete" << std::endl;
    BspSchedule sched = produce_bsp_schedule(instance);

    // Deleting structures
    helper_vessel.reset();

    return std::make_pair(SUCCESS , sched);
}