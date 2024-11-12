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

#include "scheduler/HDagg/HCoreHDagg.hpp"

std::vector<VertexType> HCoreHDagg::wavefront_initial_vertices(const std::vector<unsigned> &number_of_unprocessed_parents, const BspInstance &instance) {
    std::vector<VertexType> initial_tasks;

    for (VertexType vert = 0; vert < instance.getComputationalDag().numberOfVertices(); vert++) {
        if (number_of_unprocessed_parents[vert] == 0) {
            initial_tasks.emplace_back(vert);
        }
    }

    return initial_tasks;
}

std::vector<VertexType> HCoreHDagg::wavefront_get_next_vertices_forward_check(const std::vector<VertexType> &task_to_schedule_forward_check, std::vector<unsigned> &number_of_unprocessed_parents_forward_check, Union_Find_Universe<VertexType> &uf_structure, const BspInstance &instance) {
    std::vector<VertexType> next_vertices_to_schedule;

    for (const VertexType &vert : task_to_schedule_forward_check) {
        for (const VertexType &child : instance.getComputationalDag().children(vert)) {
            number_of_unprocessed_parents_forward_check[child]--;
            if (number_of_unprocessed_parents_forward_check[child] == 0) {
                bool can_add = true;
                for (const VertexType &parent : instance.getComputationalDag().parents(child)) {
                    if (!uf_structure.is_in_universe(parent)) continue;
                    if (instance.getComputationalDag().nodeType(parent) != instance.getComputationalDag().nodeType(child)) {
                        can_add = false;
                        break;
                    }
                }
                if (use_memory_constraint) {
                    std::set<VertexType> parent_components;
                    for (const VertexType &parent : instance.getComputationalDag().parents(child)) {
                        if (!uf_structure.is_in_universe(parent)) continue;
                        parent_components.emplace(uf_structure.find_origin_by_name(parent));
                    }

                    unsigned combined_memory = instance.getComputationalDag().nodeMemoryWeight(child);
                    for (const VertexType &par_in_comp : parent_components) {
                        combined_memory += uf_structure.get_memory_of_component_by_name(par_in_comp);
                    }
                    if (combined_memory > instance.maxMemoryBoundNodeType(instance.getComputationalDag().nodeType(child))) {
                        can_add = false;
                    }
                }
                if (can_add) {
                    next_vertices_to_schedule.emplace_back(child);
                }
            }
        }
    }

    return next_vertices_to_schedule;
}

std::vector<VertexType> HCoreHDagg::wavefront_get_next_vertices(const std::vector<std::vector<VertexType>> &best_allocation, std::vector<unsigned> &number_of_unprocessed_parents, std::vector<bool> &vertex_scheduled, std::set<VertexType> &ready_vertices, const BspInstance &instance) {

    std::vector<VertexType> vertices_to_schedule;

    for (unsigned proc = 0; proc < best_allocation.size(); proc++) {
        for (const VertexType &vert : best_allocation[proc]) {
            for (const VertexType &child : instance.getComputationalDag().children(vert)) {
                number_of_unprocessed_parents[child]--;
                if (number_of_unprocessed_parents[child] == 0 && (!vertex_scheduled[child])) {
                    vertices_to_schedule.emplace_back(child);
                    ready_vertices.emplace(child);
                }
            }
        }
    }

    return vertices_to_schedule;
}




std::vector<VertexType> HCoreHDagg::wavefront_vertextype_initial_vertices(const std::vector<unsigned> &number_of_unprocessed_parents, const BspInstance &instance) {

    return wavefront_initial_vertices(number_of_unprocessed_parents, instance);
}

std::vector<VertexType> HCoreHDagg::wavefront_vertextype_get_next_vertices_forward_check(const std::vector<VertexType> &task_to_schedule_forward_check, std::vector<unsigned> &number_of_unprocessed_parents_forward_check, std::set<VertexType> &ready_vertices_forward_check, const std::vector<unsigned> &preferred_order_of_adding_vertex_type_wavefront, Union_Find_Universe<VertexType> &uf_structure, const BspInstance &instance) {
    std::vector<VertexType> next_vertices_to_schedule;

    for (const VertexType &vert : task_to_schedule_forward_check) {
        for (const VertexType &child : instance.getComputationalDag().children(vert)) {
            number_of_unprocessed_parents_forward_check[child]--;
            if (number_of_unprocessed_parents_forward_check[child] == 0) {
                bool can_add = true;
                for (const VertexType &parent : instance.getComputationalDag().parents(child)) {
                    if (!uf_structure.is_in_universe(parent)) continue;
                    if (instance.getComputationalDag().nodeType(parent) != instance.getComputationalDag().nodeType(child)) {
                        can_add = false;
                        break;
                    }
                }
                if (use_memory_constraint) {
                    std::set<VertexType> parent_components;
                    for (const VertexType &parent : instance.getComputationalDag().parents(child)) {
                        if (!uf_structure.is_in_universe(parent)) continue;
                        parent_components.emplace(uf_structure.find_origin_by_name(parent));
                    }

                    unsigned combined_memory = instance.getComputationalDag().nodeMemoryWeight(child);
                    for (const VertexType &par_in_comp : parent_components) {
                        combined_memory += uf_structure.get_memory_of_component_by_name(par_in_comp);
                    }
                    if (combined_memory > instance.maxMemoryBoundNodeType(instance.getComputationalDag().nodeType(child))) {
                        can_add = false;
                    }
                }
                if (can_add) {
                    ready_vertices_forward_check.emplace(child);
                }
            }
        }
    }

    for (const unsigned &vertType : preferred_order_of_adding_vertex_type_wavefront) {
        for (const VertexType &vert : ready_vertices_forward_check) {
            if (instance.getComputationalDag().nodeType(vert) != vertType) continue;

            next_vertices_to_schedule.emplace_back(vert);
        }

        if (!next_vertices_to_schedule.empty()) break;
    }

    return next_vertices_to_schedule;
}

std::vector<VertexType> HCoreHDagg::wavefront_vertextype_get_next_vertices(const std::vector<std::vector<VertexType>> &best_allocation, std::vector<unsigned> &number_of_unprocessed_parents, std::vector<bool> &vertex_scheduled, std::set<VertexType> &ready_vertices, const BspInstance &instance) {
    
    return wavefront_get_next_vertices(best_allocation, number_of_unprocessed_parents, vertex_scheduled, ready_vertices, instance);
}








std::vector<unsigned> HCoreHDagg::component_allocation(const std::vector<std::tuple<std::vector<VertexType>, unsigned, unsigned>> &components_weights_and_memory, unsigned &max_work_weight, std::vector<unsigned> &preferred_order_of_adding_vertex_type_wavefront, const BspInstance &instance, RETURN_STATUS &status) {
    std::set<weighted_memory_component, std::greater<weighted_memory_component>> components;
    for (unsigned i = 0; i < components_weights_and_memory.size(); i++) {
        components.emplace( i,
                            std::get<1>(components_weights_and_memory[i]),
                            std::get<2>(components_weights_and_memory[i]),
                            instance.getComputationalDag().nodeType( std::get<0>(components_weights_and_memory[i])[0] ));
    }

    std::set<weighted_memory_bin> bins;
    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
        bins.emplace_hint(bins.cend(), proc, instance.getArchitecture().processorType(proc));
    }

    std::vector<unsigned> allocation(components_weights_and_memory.size());

    for (const weighted_memory_component &comp : components) {
        bool allocated = false;
        for (auto bin_iter = bins.begin(); bin_iter != bins.cend(); bin_iter++) {
            if ( !instance.isCompatibleType(comp.vertex_type, bin_iter->bin_type) ) continue;
            if ( use_memory_constraint && (comp.memory + bin_iter->memory > instance.getArchitecture().memoryBound(bin_iter->bin_type))) continue;

            weighted_memory_bin new_bin = *bin_iter;
            new_bin += comp;
            allocation[comp.id] = new_bin.id;
            
            bins.erase(bin_iter);
            bins.emplace(new_bin);

            allocated = true;
            break;
        }
        if (!allocated) {
            status = std::max(status, RETURN_STATUS::BEST_FOUND);
            for (auto bin_iter = bins.begin(); bin_iter != bins.cend(); bin_iter++) {
                if ( !instance.isCompatibleType(comp.vertex_type, bin_iter->bin_type) ) continue;

                weighted_memory_bin new_bin = *bin_iter;
                new_bin += comp;
                allocation[comp.id] = new_bin.id;
                
                bins.erase(bin_iter);
                bins.emplace(new_bin);

                allocated = true;
                break;
            }
        }
        if (!allocated) {
            status = std::max(status, RETURN_STATUS::ERROR);
        }
        assert(allocated && "Must be able to allocate component.");
    }

    max_work_weight = bins.rbegin()->weight;

    if (params.front_type == HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE) {
        std::vector<bool> inserted_vertex_type(instance.getComputationalDag().getNumberOfNodeTypes(), false);

        preferred_order_of_adding_vertex_type_wavefront.clear();
        preferred_order_of_adding_vertex_type_wavefront.reserve(instance.getComputationalDag().getNumberOfNodeTypes());

        for (auto bin_iter = bins.begin(); bin_iter != bins.cend(); bin_iter++) {
            for (unsigned vertType = 0; vertType < instance.getComputationalDag().getNumberOfNodeTypes(); vertType++) {
                if (inserted_vertex_type[vertType]) continue;

                if (instance.isCompatibleType(vertType, bin_iter->bin_type)) {
                    preferred_order_of_adding_vertex_type_wavefront.emplace_back(vertType);
                    inserted_vertex_type[vertType] = true;
                }
            }
        }
    }

    return allocation;
}


float HCoreHDagg::score_weight_balance(unsigned max_work_weight, unsigned total_work_weight) {
    return static_cast<float>(max_work_weight) / static_cast<float>(total_work_weight);
}


float HCoreHDagg::score_scaled_superstep_cost(unsigned max_work_weight, unsigned total_work_weight, const BspInstance &instance) {
    return static_cast<float>(max_work_weight + instance.getArchitecture().synchronisationCosts()) / static_cast<float>(total_work_weight);;
}

float HCoreHDagg::future_score_weight_balance(const std::set<VertexType> &ready_vertices_forward_check, const BspInstance &instance) {
    const ComputationalDag &graph = instance.getComputationalDag();
    const BspArchitecture &arch = instance.getArchitecture();

    unsigned largest_work_weight = 0;
    unsigned total_work = 0;
    std::vector<float> total_work_weight_per_processor_type(arch.getNumberOfProcessorTypes(), 0);

    for (const VertexType &vert : ready_vertices_forward_check) {
        largest_work_weight = std::max(largest_work_weight, static_cast<unsigned>(graph.nodeWorkWeight(vert)));
        total_work += graph.nodeWorkWeight(vert);
    }

    for (const VertexType &vert : ready_vertices_forward_check) {
        std::vector<unsigned> compatible_processor_types;

        for (unsigned proc_type = 0; proc_type < arch.getNumberOfProcessorTypes(); proc_type++) {
            if (instance.isCompatibleType(graph.nodeType(vert), proc_type)) {
                compatible_processor_types.emplace_back(proc_type);
            }
        }

        for (const unsigned proc_type : compatible_processor_types) {
            total_work_weight_per_processor_type[proc_type] += static_cast<float>(graph.nodeWorkWeight(vert)) / static_cast<float>(compatible_processor_types.size());
        }
    }

    std::vector<unsigned> number_of_processors_of_each_type(arch.getNumberOfProcessorTypes(), 0);
    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
        number_of_processors_of_each_type[arch.processorType(proc)]++;
    }

    float max_work = static_cast<float>(largest_work_weight);
    for (unsigned proc_type = 0; proc_type < arch.getNumberOfProcessorTypes(); proc_type++) {
        max_work = std::max(max_work, total_work_weight_per_processor_type[proc_type] / static_cast<float>(number_of_processors_of_each_type[proc_type]));
    }

    return max_work / static_cast<float>(total_work);
}

float HCoreHDagg::future_score_scaled_superstep_cost(const std::set<VertexType> &ready_vertices_forward_check, const BspInstance &instance) {
    const ComputationalDag &graph = instance.getComputationalDag();
    const BspArchitecture &arch = instance.getArchitecture();

    unsigned largest_work_weight = 0;
    unsigned total_work = 0;
    std::vector<float> total_work_weight_per_processor_type(arch.getNumberOfProcessorTypes(), 0);

    for (const VertexType &vert : ready_vertices_forward_check) {
        largest_work_weight = std::max(largest_work_weight, static_cast<unsigned>(graph.nodeWorkWeight(vert)));
        total_work += graph.nodeWorkWeight(vert);
    }

    for (const VertexType &vert : ready_vertices_forward_check) {
        std::vector<unsigned> compatible_processor_types;

        for (unsigned proc_type = 0; proc_type < arch.getNumberOfProcessorTypes(); proc_type++) {
            if (instance.isCompatibleType(graph.nodeType(vert), proc_type)) {
                compatible_processor_types.emplace_back(proc_type);
            }
        }

        for (const unsigned proc_type : compatible_processor_types) {
            total_work_weight_per_processor_type[proc_type] += static_cast<float>(graph.nodeWorkWeight(vert)) / static_cast<float>(compatible_processor_types.size());
        }
    }

    std::vector<unsigned> number_of_processors_of_each_type(arch.getNumberOfProcessorTypes(), 0);
    for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
        number_of_processors_of_each_type[arch.processorType(proc)]++;
    }

    float max_work = static_cast<float>(largest_work_weight);
    for (unsigned proc_type = 0; proc_type < arch.getNumberOfProcessorTypes(); proc_type++) {
        max_work = std::max(max_work, total_work_weight_per_processor_type[proc_type] / static_cast<float>(number_of_processors_of_each_type[proc_type]));
    }

    return (max_work + static_cast<float>(arch.synchronisationCosts())) / static_cast<float>(total_work);
}


std::pair<RETURN_STATUS, BspSchedule> HCoreHDagg::computeSchedule(const BspInstance &instance) {

    BspSchedule sched(instance);
    RETURN_STATUS status = RETURN_STATUS::SUCCESS;

    if (use_memory_constraint && instance.getArchitecture().getMemoryConstraintType() == NONE) {
        use_memory_constraint = false;
    }
    if (use_memory_constraint && (instance.getArchitecture().getMemoryConstraintType() != LOCAL)) {
        std::cerr << "Memory constraint type not supported. Only LOCAL and NONE supported." << std::endl;
        std::cerr << "Assumes memory constraint type to be LOCAL for best effort." << std::endl;
        status = std::max(status, RETURN_STATUS::BEST_FOUND);
    }
    
    const ComputationalDag& graph = instance.getComputationalDag();

    std::vector<bool> vertex_scheduled(graph.numberOfVertices(), false);

    std::set<VertexType> ready_vertices;
    std::vector<unsigned> number_of_unprocessed_parents(graph.numberOfVertices());
    for (VertexType vert = 0; vert < graph.numberOfVertices(); vert++) {
        number_of_unprocessed_parents[vert] = graph.numberOfParents(vert);
        if (number_of_unprocessed_parents[vert] == 0) {
            ready_vertices.emplace_hint(ready_vertices.cend(), vert);
        }
    }


    std::vector<VertexType> task_to_schedule;
    switch (params.front_type) {
        case HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT:
            task_to_schedule = wavefront_initial_vertices(number_of_unprocessed_parents, instance);
            break;
        case HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE:
            task_to_schedule = wavefront_vertextype_initial_vertices(number_of_unprocessed_parents, instance);
            break;
        default:
            // no default
            break;
    }

    unsigned superstep_counter = 0;

    while (!ready_vertices.empty()) {
        Union_Find_Universe<VertexType> uf_structure;

        unsigned total_memory = 0;
        unsigned total_work_weight = 0;

        std::vector<std::vector<VertexType>> best_allocation;
        float best_score = std::numeric_limits<float>::max();

        std::vector<unsigned> number_of_unprocessed_parents_forward_check = number_of_unprocessed_parents;
        std::vector<bool> vertex_scheduled_forward_check = vertex_scheduled;

        std::set<VertexType> ready_vertices_forward_check;
        std::vector<unsigned> number_of_unprocessed_parents_future;
        if (params.consider_future_score) {     
            ready_vertices_forward_check = ready_vertices;
            number_of_unprocessed_parents_future = number_of_unprocessed_parents;
        }

        std::set<VertexType> ready_vertices_for_node_type_forward_check;
        if (params.front_type == HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE) {
            ready_vertices_for_node_type_forward_check = ready_vertices;
        }

        bool compute_next_iteration = true;
        unsigned repeated_improve_failures = 0;
        
        while (compute_next_iteration) {
            compute_next_iteration = false;

            for (const VertexType& vert : task_to_schedule) {
                uf_structure.add_object(vert, graph.nodeWorkWeight(vert), graph.nodeMemoryWeight(vert));
            }

            for (const VertexType& vert : task_to_schedule) {
                for (const VertexType& parent : graph.parents(vert)) {
                    if (uf_structure.is_in_universe(parent)) {
                        assert((graph.nodeType(vert) == graph.nodeType(parent)) && "Tried to join two vertices of different types.");
                        uf_structure.join_by_name(vert, parent);
                    }
                }
            }

            for (const VertexType& vert : task_to_schedule) {
                total_work_weight += graph.nodeWorkWeight(vert);
            }

            if (use_memory_constraint) {
                for (const VertexType& vert : task_to_schedule) {
                    total_memory += graph.nodeMemoryWeight(vert);
                }
            }

            // Get connected components
            std::vector<std::tuple<std::vector<VertexType>, unsigned, unsigned>> components_weights_and_memory = uf_structure.get_connected_components_weights_and_memories();

            // Do allocation
            unsigned max_work_weight;
            std::vector<unsigned> preferred_order_of_adding_vertex_type_wavefront;
            std::vector<unsigned> component_to_processor_allocation = component_allocation(components_weights_and_memory, max_work_weight, preferred_order_of_adding_vertex_type_wavefront, instance, status);

            float score = std::numeric_limits<float>::max();
            switch (params.score_func) {
                case HCoreHDagg_parameters::SCORE_FUNC::WEIGHT_BALANCE:
                    score = score_weight_balance(max_work_weight, total_work_weight);
                    break;
                case HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST:
                    score = score_scaled_superstep_cost(max_work_weight, total_work_weight, instance);
                    break;
                default:
                    // no default
                    break;
            }

            for (const VertexType& vert : task_to_schedule) {
                vertex_scheduled_forward_check[vert] = true;
            }

            if (params.consider_future_score) {
                for (const VertexType &vert : task_to_schedule) {
                    ready_vertices_forward_check.erase(vert);
                }
                for (const VertexType &vert : task_to_schedule) {
                    for (const VertexType &child : graph.children(vert)) {
                        number_of_unprocessed_parents_future[child]--;
                        if (number_of_unprocessed_parents_future[child] == 0 && (!vertex_scheduled_forward_check[child])) {
                            ready_vertices_forward_check.emplace(child);
                        }
                    }
                }
            }

            if (score <= best_score) {
                compute_next_iteration = true;
                
                best_allocation.clear();
                best_allocation.resize(instance.numberOfProcessors());
                for (unsigned i = 0; i < components_weights_and_memory.size(); i++) {
                    std::vector<VertexType> &component = std::get<0>(components_weights_and_memory[i]);
                    unsigned &allocate_to_proc = component_to_processor_allocation[i];

                    best_allocation[allocate_to_proc].insert(best_allocation[allocate_to_proc].cend(), component.begin(), component.end());
                }

                repeated_improve_failures = 0;
                best_score = score;

                if (params.consider_future_score) {
                    float future_score = std::numeric_limits<float>::max();
                    switch (params.score_func) {
                        case HCoreHDagg_parameters::SCORE_FUNC::WEIGHT_BALANCE:
                            future_score = future_score_weight_balance(ready_vertices_forward_check, instance);
                            break;
                        case HCoreHDagg_parameters::SCORE_FUNC::SCALED_SUPERSTEP_COST:
                            future_score = future_score_scaled_superstep_cost(ready_vertices_forward_check, instance);
                            break;
                        default:
                            // no default
                            break;
                    }
                    if (future_score < best_score * 0.875) {
                        compute_next_iteration = false;
                    }

                }
            } else {
                repeated_improve_failures++;
                if (repeated_improve_failures < params.max_repeated_failures_to_improve) {
                    compute_next_iteration = true;
                }
            }

            if (total_work_weight < params.min_total_work_weight_check || max_work_weight < params.min_max_work_weight_check) {
                compute_next_iteration = true;
            }
            if (use_memory_constraint && (total_memory > instance.getArchitecture().sumMemoryBound())) {
                compute_next_iteration = false;
            }
            if (!compute_next_iteration) break;


            // Get the new vertices
            switch (params.front_type) {
                case HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT:
                    task_to_schedule = wavefront_get_next_vertices_forward_check(task_to_schedule, number_of_unprocessed_parents_forward_check, uf_structure, instance);
                    break;
                case HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE:
                    for (const VertexType &vert : task_to_schedule) {
                        ready_vertices_for_node_type_forward_check.erase(vert);
                    }
                    task_to_schedule = wavefront_vertextype_get_next_vertices_forward_check(task_to_schedule, number_of_unprocessed_parents_forward_check, ready_vertices_for_node_type_forward_check, preferred_order_of_adding_vertex_type_wavefront, uf_structure, instance);
                    break;
                default:
                    // no default
                    break;
            }

            if (task_to_schedule.empty()) break;
        }
        for (unsigned proc = 0; proc < best_allocation.size(); proc++) {
            for (const VertexType& vert : best_allocation[proc]) {
                assert(instance.isCompatible(vert, proc) && "Vertices must be allocated to compatible processor type!");
                sched.setAssignedProcessor(vert, proc);
                sched.setAssignedSuperstep(vert, superstep_counter);
                vertex_scheduled[vert] = true;
                if (ready_vertices.find(vert) != ready_vertices.end()) {
                    ready_vertices.erase(vert);
                }
            }
        }

        // Get the new vertices and add them to the union-find structure
        switch (params.front_type) {
            case HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT:
                task_to_schedule = wavefront_get_next_vertices(best_allocation, number_of_unprocessed_parents, vertex_scheduled, ready_vertices, instance);
                break;
            case HCoreHDagg_parameters::FRONT_TYPE::WAVEFRONT_VERTEXTYPE:
                task_to_schedule = wavefront_vertextype_get_next_vertices(best_allocation, number_of_unprocessed_parents, vertex_scheduled, ready_vertices, instance);
                break;
            default:
                // no default
                break;
        }

        superstep_counter++;
    }



    sched.setAutoCommunicationSchedule();

    return std::make_pair(status, sched);
}