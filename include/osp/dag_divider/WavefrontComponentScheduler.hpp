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

#pragma once
#include "DagDivider.hpp"
#include "IsomorphismGroups.hpp"
#include "WavefrontComponentDivider.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

namespace osp {

template<typename Graph_t, typename constr_graph_t>
class WavefrontComponentScheduler : public Scheduler<Graph_t> {

    //bool set_num_proc_crit_path = false;

    IDagDivider<Graph_t> *divider;
    Scheduler<constr_graph_t> *scheduler;
    bool check_isomorphism_groups = true;
    // Use static constexpr for compile-time debugging code removal
    static constexpr bool enable_debug_prints = true;

    /**
     * @brief Distributes processors proportionally, ensuring active components get at least one if possible.
     */
    std::vector<unsigned> distributeProcessors(
        unsigned total_processors_of_type,
        const std::vector<double>& work_weights,
        double total_work) {
        
        if (total_work <= 1e-9 || total_processors_of_type == 0) {
            return std::vector<unsigned>(work_weights.size(), 0);
        }

        std::vector<unsigned> allocation(work_weights.size(), 0);
        std::vector<size_t> active_indices;
        for(size_t i = 0; i < work_weights.size(); ++i) {
            if (work_weights[i] > 1e-9) {
                active_indices.push_back(i);
            }
        }

        if (active_indices.empty()) {
            return allocation;
        }

        unsigned num_active_components = active_indices.size();
        unsigned remaining_procs = total_processors_of_type;
        
        // If we have enough processors, guarantee one for each active component first.
        if (total_processors_of_type >= num_active_components) {
            for (size_t idx : active_indices) {
                allocation[idx] = 1;
            }
            remaining_procs -= num_active_components;
        }

        if (remaining_procs > 0) {
            // Distribute the rest (or all, in case of scarcity) proportionally.
            std::vector<double> active_work_weights;
            double active_total_work = 0;
            for(size_t idx : active_indices) {
                active_work_weights.push_back(work_weights[idx]);
                active_total_work += work_weights[idx];
            }

            if (active_total_work > 1e-9) {
                std::vector<std::pair<double, size_t>> remainders;
                unsigned allocated_count = 0;

                for (size_t i = 0; i < active_indices.size(); ++i) {
                    double exact_share = (active_work_weights[i] / active_total_work) * remaining_procs;
                    unsigned additional_alloc = static_cast<unsigned>(std::floor(exact_share));
                    allocation[active_indices[i]] += additional_alloc;
                    remainders.push_back({exact_share - additional_alloc, active_indices[i]});
                    allocated_count += additional_alloc;
                }

                std::sort(remainders.rbegin(), remainders.rend());

                unsigned remainder_processors = remaining_procs - allocated_count;
                for (unsigned i = 0; i < remainder_processors; ++i) {
                    allocation[remainders[i].second]++;
                }
            }
        }
        
        return allocation;
    }

    /**
     * @brief Creates a BspArchitecture for a sub-problem given a specific processor allocation.
     */
    BspArchitecture<constr_graph_t> createSubArchitecture(
        const BspArchitecture<Graph_t> &original_arch,
        const std::vector<unsigned>& sub_dag_proc_types) {

        BspArchitecture<constr_graph_t> sub_architecture(original_arch);
        std::vector<v_memw_t<Graph_t>> sub_dag_processor_memory(original_arch.getProcessorTypeCount().size(),
                                                                std::numeric_limits<v_memw_t<Graph_t>>::max());
        for (unsigned i = 0; i < original_arch.numberOfProcessors(); ++i) {
            sub_dag_processor_memory[original_arch.processorType(i)] =
                std::min(original_arch.memoryBound(i), sub_dag_processor_memory[original_arch.processorType(i)]);
        }
        sub_architecture.set_processors_consequ_types(sub_dag_proc_types, sub_dag_processor_memory);
        return sub_architecture;
    }


    RETURN_STATUS computeSchedule_with_isomorphism_groups(BspSchedule<Graph_t> &schedule) {
        const auto &instance = schedule.getInstance();
        const auto &original_arch = instance.getArchitecture();
        const auto& original_proc_type_count = original_arch.getProcessorTypeCount();

        std::vector<std::vector<unsigned>> global_ids_by_type(original_arch.getNumberOfProcessorTypes());
        for (unsigned i = 0; i < original_arch.numberOfProcessors(); ++i) {
            global_ids_by_type[original_arch.processorType(i)].push_back(i);
        }

        IsomorphismGroups<Graph_t, constr_graph_t> iso_groups;
        std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> vertex_maps = divider->divide(instance.getComputationalDag());
        iso_groups.compute_isomorphism_groups(vertex_maps, instance.getComputationalDag());

        const auto &isomorphism_groups = iso_groups.get_isomorphism_groups();
        unsigned superstep_offset = 0;

        for (std::size_t i = 0; i < isomorphism_groups.size(); ++i) {
            if constexpr (enable_debug_prints) std::cout << "\n--- Processing Wavefront Set " << i << " ---" << std::endl;

            std::vector<std::vector<double>> group_work_by_type(
                isomorphism_groups[i].size(), std::vector<double>(original_proc_type_count.size(), 0.0));

            for (std::size_t j = 0; j < isomorphism_groups[i].size(); ++j) {
                const constr_graph_t &rep_sub_dag = iso_groups.get_isomorphism_groups_subgraphs()[i][j];
                for (unsigned type_idx = 0; type_idx < original_proc_type_count.size(); ++type_idx) {
                    const double rep_work_for_type = sumOfCompatibleWorkWeights(rep_sub_dag, instance, type_idx);
                    group_work_by_type[j][type_idx] = rep_work_for_type * static_cast<double>(isomorphism_groups[i][j].size());
                }
            }

            // --- VALIDATION STEP ---
            for (std::size_t j = 0; j < isomorphism_groups[i].size(); ++j) {
                const constr_graph_t &rep_sub_dag = iso_groups.get_isomorphism_groups_subgraphs()[i][j];
                const double total_rep_work = sumOfVerticesWorkWeights(rep_sub_dag);
                
                double sum_of_compatible_works_for_rep = 0.0;
                for (unsigned type_idx = 0; type_idx < original_proc_type_count.size(); ++type_idx) {
                    sum_of_compatible_works_for_rep += sumOfCompatibleWorkWeights(rep_sub_dag, instance, type_idx);
                }

                if (sum_of_compatible_works_for_rep > total_rep_work + 1e-9) {
                    if constexpr (enable_debug_prints) {
                        std::cerr << "ERROR: Sum of compatible work (" << sum_of_compatible_works_for_rep 
                                  << ") exceeds total work (" << total_rep_work 
                                  << ") for a sub-dag. Aborting." << std::endl;
                    }
                    return RETURN_STATUS::ERROR;
                }
            }

            std::vector<std::vector<unsigned>> group_proc_allocations(isomorphism_groups[i].size(), std::vector<unsigned>(original_proc_type_count.size()));
            for(unsigned type_idx = 0; type_idx < original_proc_type_count.size(); ++type_idx) {
                std::vector<double> work_for_this_type(isomorphism_groups[i].size());
                double total_work_for_this_type = 0;
                for(size_t group_idx = 0; group_idx < isomorphism_groups[i].size(); ++group_idx) {
                    work_for_this_type[group_idx] = group_work_by_type[group_idx][type_idx];
                    total_work_for_this_type += work_for_this_type[group_idx];
                }
                
                auto type_alloc = distributeProcessors(original_proc_type_count[type_idx], work_for_this_type, total_work_for_this_type);
                for(size_t group_idx = 0; group_idx < isomorphism_groups[i].size(); ++group_idx) {
                    group_proc_allocations[group_idx][type_idx] = type_alloc[group_idx];
                }
            }

            if constexpr (enable_debug_prints) {
                std::cout << "Processor Allocation for this Wavefront Set:" << std::endl;
                for (size_t j = 0; j < group_proc_allocations.size(); ++j) {
                    std::cout << "  Iso Group " << j << " (" << isomorphism_groups[i][j].size() << " copies): { ";
                    for (unsigned type_idx = 0; type_idx < group_proc_allocations[j].size(); ++type_idx) {
                        std::cout << "Type " << type_idx << ": " << group_proc_allocations[j][type_idx] << "; ";
                    }
                    std::cout << "}" << std::endl;
                }
            }

            unsigned max_number_supersteps = 0;
            std::vector<unsigned> proc_type_offsets(original_arch.getNumberOfProcessorTypes(), 0);
            
            for (std::size_t j = 0; j < isomorphism_groups[i].size(); ++j) {
                constr_graph_t &sub_dag = iso_groups.get_isomorphism_groups_subgraphs()[i][j];

                std::vector<unsigned> single_sub_dag_proc_types = group_proc_allocations[j];
                size_t num_members = isomorphism_groups[i][j].size();
                if (num_members > 0) {
                    for(unsigned type_idx = 0; type_idx < single_sub_dag_proc_types.size(); ++type_idx) {
                        auto& count = single_sub_dag_proc_types[type_idx];
                        if (count > 0 && (count / static_cast<unsigned>(num_members) == 0)) {
                            count = 1;
                        } else {
                            count /= static_cast<unsigned>(num_members);
                        }
                    }
                }

                if constexpr (enable_debug_prints) {
                    std::cout << "  Creating sub-architecture for Iso Group " << j << " representative: { ";
                    for (unsigned type_idx = 0; type_idx < single_sub_dag_proc_types.size(); ++type_idx) {
                        std::cout << "Type " << type_idx << ": " << single_sub_dag_proc_types[type_idx] << "; ";
                    }
                    std::cout << "}" << std::endl;
                }

                BspArchitecture<constr_graph_t> sub_architecture = createSubArchitecture(original_arch, single_sub_dag_proc_types);
                BspInstance<constr_graph_t> sub_instance(sub_dag, sub_architecture);
                sub_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());
                BspSchedule<constr_graph_t> sub_schedule(sub_instance);
                auto status = scheduler->computeSchedule(sub_schedule);
                
                if (status != RETURN_STATUS::OSP_SUCCESS && status != RETURN_STATUS::BEST_FOUND) return status;

                const auto sub_proc_type_count = sub_architecture.getProcessorTypeCount();
                std::vector<unsigned> sub_proc_type_corrections(sub_architecture.getNumberOfProcessorTypes(), 0);
                for (std::size_t k = 1; k < sub_proc_type_corrections.size(); ++k) {
                    sub_proc_type_corrections[k] = sub_proc_type_corrections[k - 1] + sub_proc_type_count[k - 1];
                }

                for (const auto &group_member_idx : isomorphism_groups[i][j]) {
                    vertex_idx_t<constr_graph_t> subdag_vertex = 0;
                    for (const auto &vertex : vertex_maps[i][group_member_idx]) {
                        const unsigned proc_in_sub_sched = sub_schedule.assignedProcessor(subdag_vertex);
                        const unsigned proc_type = sub_architecture.processorType(proc_in_sub_sched);
                        const unsigned local_proc_id_within_type = proc_in_sub_sched - sub_proc_type_corrections[proc_type];
                        unsigned global_proc_id = global_ids_by_type[proc_type][proc_type_offsets[proc_type] + local_proc_id_within_type];
                        
                        schedule.setAssignedProcessor(vertex, global_proc_id);
                        schedule.setAssignedSuperstep(vertex, superstep_offset + sub_schedule.assignedSuperstep(subdag_vertex));
                        subdag_vertex++;
                    }

                    for (size_t k = 0; k < sub_proc_type_count.size(); ++k) {
                        proc_type_offsets[k] += sub_proc_type_count[k];
                    }
                }
                max_number_supersteps = std::max(max_number_supersteps, sub_schedule.numberOfSupersteps());
            }
            superstep_offset += max_number_supersteps;
        }
        return RETURN_STATUS::OSP_SUCCESS;
    }

    RETURN_STATUS computeSchedule_without_isomorphism_groups(BspSchedule<Graph_t> &schedule) {
        const auto &instance = schedule.getInstance();
        const auto &original_arch = instance.getArchitecture();
        const auto& original_proc_type_count = original_arch.getProcessorTypeCount();
        const auto& computational_dag = instance.getComputationalDag();

        std::vector<std::vector<unsigned>> global_ids_by_type(original_arch.getNumberOfProcessorTypes());
        for (unsigned i = 0; i < original_arch.numberOfProcessors(); ++i) {
            global_ids_by_type[original_arch.processorType(i)].push_back(i);
        }

        auto vertex_maps = divider->divide(computational_dag);
        unsigned superstep_offset = 0;

        for (std::size_t i = 0; i < vertex_maps.size(); ++i) { // For each wavefront set
            if constexpr (enable_debug_prints) std::cout << "\n--- Processing Wavefront Set " << i << " (No Isomorphism) ---" << std::endl;
            
            const auto& components = vertex_maps[i];
            std::vector<constr_graph_t> sub_dags(components.size());
            std::vector<std::vector<double>> work_by_type(components.size(), std::vector<double>(original_proc_type_count.size(), 0.0));

            for(size_t j = 0; j < components.size(); ++j) {
                create_induced_subgraph(computational_dag, sub_dags[j], components[j]);
                for (unsigned type_idx = 0; type_idx < original_proc_type_count.size(); ++type_idx) {
                    work_by_type[j][type_idx] = sumOfCompatibleWorkWeights(sub_dags[j], instance, type_idx);
                }
            }

            // Validation Step
            for (std::size_t j = 0; j < components.size(); ++j) {
                const double total_work = sumOfVerticesWorkWeights(sub_dags[j]);
                double sum_compatible = 0.0;
                for (unsigned type_idx = 0; type_idx < original_proc_type_count.size(); ++type_idx) {
                    sum_compatible += sumOfCompatibleWorkWeights(sub_dags[j], instance, type_idx);
                }
                if (sum_compatible > total_work + 1e-9) {
                    if constexpr (enable_debug_prints) std::cerr << "ERROR: Sum of compatible work exceeds total work. Aborting." << std::endl;
                    return RETURN_STATUS::ERROR;
                }
            }

            // Distribute Processors
            std::vector<std::vector<unsigned>> proc_allocations(components.size(), std::vector<unsigned>(original_proc_type_count.size()));
            for(unsigned type_idx = 0; type_idx < original_proc_type_count.size(); ++type_idx) {
                std::vector<double> work_for_this_type(components.size());
                double total_work_for_this_type = 0;
                for(size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
                    work_for_this_type[comp_idx] = work_by_type[comp_idx][type_idx];
                    total_work_for_this_type += work_for_this_type[comp_idx];
                }
                auto type_alloc = distributeProcessors(original_proc_type_count[type_idx], work_for_this_type, total_work_for_this_type);
                for(size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
                    proc_allocations[comp_idx][type_idx] = type_alloc[comp_idx];
                }
            }

            unsigned max_number_supersteps = 0;
            std::vector<unsigned> proc_type_offsets(original_arch.getNumberOfProcessorTypes(), 0);

            for (std::size_t j = 0; j < components.size(); ++j) {
                BspArchitecture<constr_graph_t> sub_architecture = createSubArchitecture(original_arch, proc_allocations[j]);
                BspInstance<constr_graph_t> sub_instance(sub_dags[j], sub_architecture);
                sub_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

                BspSchedule<constr_graph_t> sub_schedule(sub_instance);
                const auto status = scheduler->computeSchedule(sub_schedule);
                if (status != RETURN_STATUS::OSP_SUCCESS && status != RETURN_STATUS::BEST_FOUND) return status;

                const auto sub_proc_type_count = sub_architecture.getProcessorTypeCount();
                std::vector<unsigned> sub_proc_type_corrections(sub_architecture.getNumberOfProcessorTypes(), 0);
                for (std::size_t k = 1; k < sub_proc_type_corrections.size(); ++k) {
                    sub_proc_type_corrections[k] = sub_proc_type_corrections[k - 1] + sub_proc_type_count[k - 1];
                }

                vertex_idx_t<constr_graph_t> subdag_vertex = 0;
                // This needs to be a sorted list of vertices to map correctly to the induced subgraph
                std::vector<vertex_idx_t<Graph_t>> sorted_component_vertices(components[j].begin(), components[j].end());
                std::sort(sorted_component_vertices.begin(), sorted_component_vertices.end());

                for (const auto &vertex : sorted_component_vertices) {
                    const unsigned proc_in_sub_sched = sub_schedule.assignedProcessor(subdag_vertex);
                    const unsigned proc_type = sub_architecture.processorType(proc_in_sub_sched);
                    const unsigned local_proc_id_within_type = proc_in_sub_sched - sub_proc_type_corrections[proc_type];
                    unsigned global_proc_id = global_ids_by_type[proc_type][proc_type_offsets[proc_type] + local_proc_id_within_type];
                    
                    schedule.setAssignedProcessor(vertex, global_proc_id);
                    schedule.setAssignedSuperstep(vertex, superstep_offset + sub_schedule.assignedSuperstep(subdag_vertex));
                    subdag_vertex++;
                }

                for (size_t k = 0; k < sub_proc_type_count.size(); ++k) {
                    proc_type_offsets[k] += sub_proc_type_count[k];
                }
                max_number_supersteps = std::max(max_number_supersteps, sub_schedule.numberOfSupersteps());
            }
            superstep_offset += max_number_supersteps;
        }
        return RETURN_STATUS::OSP_SUCCESS;
    }

  public:
    WavefrontComponentScheduler(IDagDivider<Graph_t> &div, Scheduler<constr_graph_t> &scheduler)
        : divider(&div), scheduler(&scheduler) {}

    void set_check_isomorphism_groups(bool check) { check_isomorphism_groups = check; }

    std::string getScheduleName() const override { return "WavefrontComponentScheduler"; }

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {
        if (check_isomorphism_groups) {
            return computeSchedule_with_isomorphism_groups(schedule);
        } else {
            return computeSchedule_without_isomorphism_groups(schedule);
        }
    }
};

template<typename Graph_t>
using WavefrontComponentScheduler_def_int_t = WavefrontComponentScheduler<Graph_t, boost_graph_int_t>;

template<typename Graph_t>
using WavefrontComponentScheduler_def_uint_t = WavefrontComponentScheduler<Graph_t, boost_graph_uint_t>;

} // namespace osp