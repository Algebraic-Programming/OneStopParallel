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

#include "scheduler/SSPInstanceContractor.hpp"


void SSPInstanceContractor::setTimeLimitSeconds(unsigned int limit) {
    timeLimitSeconds = limit;
    if (sched) sched->setTimeLimitSeconds(limit);
    if (improver) improver->setTimeLimitSeconds(limit);
}

void SSPInstanceContractor::setTimeLimitHours(unsigned int limit) {
    timeLimitSeconds = limit * 3600;
    if (sched) sched->setTimeLimitHours(limit);
    if (improver) improver->setTimeLimitHours(limit);
}

void SSPInstanceContractor::setUseMemoryConstraint(bool use_memory_constraint_) {
    if (sched) sched->setUseMemoryConstraint(use_memory_constraint_);
    if (improver) improver->setUseMemoryConstraint(use_memory_constraint_);
}

RETURN_STATUS SSPInstanceContractor::add_contraction( const std::vector<std::unordered_set<VertexType>>& partition ) {
    assert( ! dag_history.empty() );
    std::pair<ComputationalDag, std::unordered_map<VertexType, VertexType>> graph_and_contraction_map = dag_history.back()->getComputationalDag().contracted_graph_without_loops(partition);

    std::unique_ptr<BspInstance> new_inst = std::make_unique<BspInstance>(graph_and_contraction_map.first, dag_history.back()->getArchitecture());

    std::unique_ptr<std::unordered_map<VertexType,std::set<VertexType>>> expansion_map = std::make_unique<std::unordered_map<VertexType,std::set<VertexType>>>();
    for (auto& [origin, destination]: graph_and_contraction_map.second) {
        if ( expansion_map->find(destination) == expansion_map->cend() ) {
            expansion_map->insert( std::make_pair( destination, std::set<VertexType>({origin})));
        } else {
            (*expansion_map)[destination].emplace(origin);
        }
    }

    std::unique_ptr<std::unordered_map<VertexType, VertexType>> new_contr = std::make_unique<std::unordered_map<VertexType, VertexType>>(graph_and_contraction_map.second);

    // New vertex types
    unsigned vertex_type_count = 0;
    std::map<std::set<unsigned>, unsigned> allowed_proc_type_to_vertex_type;
    for (const VertexType coarse_vert : new_inst->getComputationalDag().vertices() ) {
        std::set<unsigned> allowed_proc_types;
        for (unsigned proc_type = 0; proc_type < new_inst->getArchitecture().getNumberOfProcessorTypes(); proc_type++) {
            allowed_proc_types.emplace(proc_type);
        }
        for (const VertexType fine_vert : (*expansion_map)[coarse_vert]) {
            for (auto proc_type_iter = allowed_proc_types.begin(); proc_type_iter != allowed_proc_types.end(); ) {
                if (dag_history.back()->isCompatibleType( dag_history.back()->getComputationalDag().nodeType(fine_vert), *proc_type_iter)) {
                    proc_type_iter++;
                } else {
                    proc_type_iter = allowed_proc_types.erase(proc_type_iter);
                }
            }
        }

        if (allowed_proc_types.empty()) {
            throw std::logic_error("Coarsening algorithm error: no processor type can schedule the coarse vertex!");
        }

        if (allowed_proc_type_to_vertex_type.find(allowed_proc_types) == allowed_proc_type_to_vertex_type.end()) {
            allowed_proc_type_to_vertex_type[allowed_proc_types] = vertex_type_count;
            new_inst->getComputationalDag().setNodeType(coarse_vert, vertex_type_count);
            vertex_type_count++;
        } else {
            new_inst->getComputationalDag().setNodeType(coarse_vert, allowed_proc_type_to_vertex_type.at(allowed_proc_types));
        }
    }
    std::vector<std::vector<bool>> vert_type_to_proc_type_compatability(vertex_type_count, std::vector<bool>(new_inst->getArchitecture().getNumberOfProcessorTypes(), false));
    for (const auto& [proc_types, vertex_type] : allowed_proc_type_to_vertex_type) {
        for (const auto proc_type : proc_types) {
            vert_type_to_proc_type_compatability[vertex_type][proc_type] = true;
        }
    }
    new_inst->setNodeProcessorCompatibility(vert_type_to_proc_type_compatability);

    // // tests
    // for (auto& [orig, dest] : *new_contr) {
    //     assert( expansion_map->find(dest) != expansion_map->cend() );
    //     assert( expansion_map->at(dest).find(orig) != expansion_map->at(dest).cend() );
    // }
    // unsigned counter = 0;
    // for (auto& [dest, orig_set] : *expansion_map) {
    //     counter += orig_set.size();
    // }
    // assert(counter == new_contr->size());
    // assert(counter == dag_history.back()->getComputationalDag().numberOfVertices());
    

    dag_history.push_back( move(new_inst) );
    contraction_maps.push_back( move(new_contr) );
    expansion_maps.push_back(move( expansion_map ));

    compactify_dag_history();

    return SUCCESS;
}

RETURN_STATUS SSPInstanceContractor::compute_initial_schedule(unsigned stale) {
    active_graph = dag_history.size()-1;
    std::pair<RETURN_STATUS, SspSchedule> initial_schedule;
    if (active_graph == 0) {
        initial_schedule = sched->computeSspSchedule( *getOriginalInstance(), stale );
    } else {
        initial_schedule = sched->computeSspSchedule( *(dag_history.back()), stale );
    }
    active_schedule = initial_schedule.second;
    auto ret = improve_active_schedule();


    return std::max(ret,initial_schedule.first);
}


SspSchedule SSPInstanceContractor::expand_schedule(const SspSchedule& schedule, std::pair< ComputationalDag, std::unordered_map<VertexType, VertexType>> pair, const BspInstance& instance) {

    
    SspSchedule expanded_schedule(instance);

    for ( auto node : instance.getComputationalDag().vertices() ) {
        expanded_schedule.setAssignedProcessor(node, schedule.assignedProcessor(pair.second.at(node)) );
        expanded_schedule.setAssignedSuperstep(node, schedule.assignedSuperstep(pair.second.at(node)) );
    }

    // for (auto& [triple, step] : schedule.getCommunicationSchedule()) {
    //     for (auto& node : expansion_maps[active_graph-1]->at(std::get<0>(triple))) {
    //         expanded_schedule.addCommunicationScheduleEntry( std::tuple<unsigned int, unsigned int, unsigned int>({node, std::get<1>(triple), std::get<2>(triple) }), step);
    //     }
    // }
    expanded_schedule.setAutoCommunicationSchedule();


    return expanded_schedule;
}





RETURN_STATUS SSPInstanceContractor::expand_active_schedule() {
    assert((active_graph >= 1) && ( (long unsigned) active_graph < dag_history.size()));

    SspSchedule expanded_schedule;
    if (active_graph == 1) {
        expanded_schedule = SspSchedule(*original_inst);
    } else {
        expanded_schedule = SspSchedule( *(dag_history[ active_graph-1 ]));
    }

    for ( auto node : dag_history[active_graph-1]->getComputationalDag().vertices() ) {
        expanded_schedule.setAssignedProcessor(node, active_schedule.assignedProcessor(contraction_maps[active_graph-1]->at(node)) );
        expanded_schedule.setAssignedSuperstep(node, active_schedule.assignedSuperstep(contraction_maps[active_graph-1]->at(node)) );
    }

    for (auto& [triple, step] : active_schedule.getCommunicationSchedule()) {
        for (auto& node : expansion_maps[active_graph-1]->at(std::get<0>(triple))) {
            expanded_schedule.addCommunicationScheduleEntry( std::tuple<unsigned int, unsigned int, unsigned int>({node, std::get<1>(triple), std::get<2>(triple) }), step);
        }
    }

    active_graph--;
    active_schedule = expanded_schedule;
    return SUCCESS;
}

RETURN_STATUS SSPInstanceContractor::improve_active_schedule() {
    if (improver) {
        return improver->improveSSPSchedule( active_schedule );
    }
    return SUCCESS;
}

RETURN_STATUS SSPInstanceContractor::run_expansions() {
    assert(active_graph >= 0 && (long unsigned) active_graph == dag_history.size()-1);
    RETURN_STATUS status = SUCCESS;
    while(active_graph>0) {
        status = std::max(status, expand_active_schedule());
        status = std::max(status, improve_active_schedule());
    }
    return status;
}


std::pair< ComputationalDag, std::unordered_map<VertexType, VertexType> > SSPInstanceContractor::get_contracted_graph_and_mapping( const ComputationalDag& graph ) {
    clear_computation_data();

    BspInstance tmp(graph, BspArchitecture());
    original_inst = &tmp;

    dag_history.emplace_back( std::make_unique<BspInstance>( graph, BspArchitecture() ));
    run_contractions();

    std::unordered_map<VertexType, VertexType> contraction_map;
    contraction_map.reserve(graph.numberOfVertices());
    for (const VertexType& node : graph.vertices()) {
        contraction_map[node] = node;
    }

    for (long unsigned iteration = 0; iteration < contraction_maps.size(); iteration++) {
        for (const VertexType& node : graph.vertices()) {
            contraction_map[node] = (*(contraction_maps[iteration]))[ contraction_map[node] ];
        }
    }

    return std::make_pair( dag_history.back()->getComputationalDag(), contraction_map );    
}

std::pair<RETURN_STATUS, SspSchedule> SSPInstanceContractor::computeSspSchedule(const BspInstance& instance, unsigned stale) {
    clear_computation_data();
    original_inst = &instance;
    dag_history.emplace_back(std::make_unique<BspInstance>(instance));

    RETURN_STATUS status = SUCCESS;
    status = std::max(status, run_contractions());
    assert( (dag_history.size()-1 == expansion_maps.size()) && (expansion_maps.size() == contraction_maps.size()) );

    status = std::max(status, compute_initial_schedule(stale));
    
    status = std::max(status, run_expansions());

    SspSchedule output = active_schedule;

    clear_computation_data();

    return std::make_pair(status, output);
}

void SSPInstanceContractor::clear_computation_data() {
    dag_history.clear();
    dag_history.shrink_to_fit();
    
    contraction_maps.clear();
    contraction_maps.shrink_to_fit();

    expansion_maps.clear();
    expansion_maps.shrink_to_fit();

    active_graph = -1;
    active_schedule = SspSchedule();
}

void SSPInstanceContractor::compactify_dag_history() {
    if (dag_history.size() < 3) return;

    size_t dag_indx_first = dag_history.size()-2;
    size_t map_indx_first = contraction_maps.size()-2;

    size_t dag_indx_second = dag_history.size()-1;
    size_t map_indx_second = contraction_maps.size()-1;

    if ( ((double) dag_history[dag_indx_first-1]->numberOfVertices() / (double) dag_history[dag_indx_second-1]->numberOfVertices()) > 1.25 ) return;

    // Compute combined contraction_map
    std::unique_ptr<std::unordered_map<VertexType, VertexType>> combi_contraction_map = std::make_unique<std::unordered_map<VertexType, VertexType>>();
    for (auto map_pair_it = contraction_maps[map_indx_first]->begin(); map_pair_it != contraction_maps[map_indx_first]->cend(); map_pair_it++) {
        combi_contraction_map->emplace( map_pair_it->first, contraction_maps[map_indx_second]->at( map_pair_it->second ) );
    }

    // Compute combined expansion_map
    std::unique_ptr<std::unordered_map<VertexType, std::set<VertexType>>> combi_expansion_map = std::make_unique<std::unordered_map<VertexType, std::set<VertexType>>>();
    for (auto map_pair_it = combi_contraction_map->begin(); map_pair_it != combi_contraction_map->cend(); map_pair_it++) {
        if ( combi_expansion_map->find(map_pair_it->second) == combi_expansion_map->cend() ) {
            combi_expansion_map->insert( std::make_pair( map_pair_it->second, std::set<VertexType>({map_pair_it->first})));
        } else {
            (*combi_expansion_map)[map_pair_it->second].emplace(map_pair_it->first);
        }
    }

    // Delete ComputationalDag
    auto dag_it = dag_history.begin();
    std::advance(dag_it, dag_indx_first);
    dag_history.erase(dag_it);

    // Delete contraction map
    auto contr_map_it = contraction_maps.begin();
    std::advance(contr_map_it, map_indx_second);
    contraction_maps.erase(contr_map_it);

    // Replace contraction map
    contraction_maps[map_indx_first] = move(combi_contraction_map);

    // Delete expansion map
    auto exp_map_it = expansion_maps.begin();
    std::advance(exp_map_it, map_indx_second);
    expansion_maps.erase(exp_map_it);

    // Replace expansion map
    expansion_maps[map_indx_first] = move(combi_expansion_map);
}