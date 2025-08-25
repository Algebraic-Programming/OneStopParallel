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
#include "AbstractWavefrontScheduler.hpp"
#include "IsomorphismGroups.hpp"

namespace osp {

/**
 * @class IsomorphicWavefrontComponentScheduler
 * @brief Schedules wavefronts by grouping isomorphic components.
 */
template<typename Graph_t, typename constr_graph_t>
class IsomorphicWavefrontComponentScheduler : public AbstractWavefrontScheduler<Graph_t, constr_graph_t> {
public:
    IsomorphicWavefrontComponentScheduler(IDagDivider<Graph_t> &div, Scheduler<constr_graph_t> &scheduler)
        : AbstractWavefrontScheduler<Graph_t, constr_graph_t>(div, scheduler) {}

    std::string getScheduleName() const override { return "IsomorphicWavefrontComponentScheduler"; }

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {
        const auto &instance = schedule.getInstance();
        const auto &original_arch = instance.getArchitecture();

        std::vector<std::vector<unsigned>> global_ids_by_type(original_arch.getNumberOfProcessorTypes());
        for (unsigned i = 0; i < original_arch.numberOfProcessors(); ++i) {
            global_ids_by_type[original_arch.processorType(i)].push_back(i);
        }

        IsomorphismGroups<Graph_t, constr_graph_t> iso_groups;
        std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> vertex_maps = this->divider->divide(instance.getComputationalDag());
        iso_groups.compute_isomorphism_groups(vertex_maps, instance.getComputationalDag());

        unsigned superstep_offset = 0;
        for (std::size_t i = 0; i < vertex_maps.size(); ++i) {
            if (this->enable_debug_prints) std::cout << "\n--- Processing Wavefront Set " << i << " ---" << std::endl;
            
            unsigned supersteps_in_set = 0;
            auto status = process_wavefront_set(schedule, vertex_maps[i], 
                                                iso_groups.get_isomorphism_groups()[i],
                                                iso_groups.get_isomorphism_groups_subgraphs()[i],
                                                global_ids_by_type,
                                                superstep_offset, supersteps_in_set);
            if (status != RETURN_STATUS::OSP_SUCCESS) {
                return status;
            }
            superstep_offset += supersteps_in_set;
        }
        return RETURN_STATUS::OSP_SUCCESS;
    }

private:
    RETURN_STATUS process_wavefront_set(
        BspSchedule<Graph_t>& schedule,
        const std::vector<std::vector<vertex_idx_t<Graph_t>>>& vertex_map_for_set,
        const std::vector<std::vector<size_t>>& iso_groups_for_set,
        const std::vector<constr_graph_t>& subgraphs_for_set,
        const std::vector<std::vector<unsigned>>& global_ids_by_type,
        unsigned superstep_offset,
        unsigned& supersteps_in_set) {

        const auto &instance = schedule.getInstance();
        const auto &original_arch = instance.getArchitecture();
        const auto& original_proc_type_count = original_arch.getProcessorTypeCount();

        // Calculate work for each isomorphism group
        std::vector<std::vector<double>> group_work_by_type(
            iso_groups_for_set.size(), std::vector<double>(original_proc_type_count.size(), 0.0));

        for (std::size_t j = 0; j < iso_groups_for_set.size(); ++j) {
            const constr_graph_t &rep_sub_dag = subgraphs_for_set[j];
            for (unsigned type_idx = 0; type_idx < original_proc_type_count.size(); ++type_idx) {
                const double rep_work_for_type = sumOfCompatibleWorkWeights(rep_sub_dag, instance, type_idx);
                group_work_by_type[j][type_idx] = rep_work_for_type * static_cast<double>(iso_groups_for_set[j].size());
            }
        }

        assert(this->validateWorkDistribution(subgraphs_for_set, instance));

        // Distribute processors among isomorphism groups
        std::vector<std::vector<unsigned>> group_proc_allocations(iso_groups_for_set.size(), std::vector<unsigned>(original_proc_type_count.size()));
        
        for(unsigned type_idx = 0; type_idx < original_proc_type_count.size(); ++type_idx) {
            std::vector<double> work_for_this_type;
            for(size_t group_idx = 0; group_idx < iso_groups_for_set.size(); ++group_idx) {
                work_for_this_type.push_back(group_work_by_type[group_idx][type_idx]);
            }
            
            std::vector<unsigned> type_allocation;
            bool starvation_hit = this->distributeProcessors(original_proc_type_count[type_idx], work_for_this_type, type_allocation);

            if (starvation_hit) {
                if constexpr (this->enable_debug_prints) {
                    std::cerr << "ERROR: Processor starvation detected for type " << type_idx 
                              << ". Not enough processors to assign one to each active isomorphism group." << std::endl;
                }
                return RETURN_STATUS::ERROR;
            }
            
            for(size_t group_idx = 0; group_idx < iso_groups_for_set.size(); ++group_idx) {
                group_proc_allocations[group_idx][type_idx] = type_allocation[group_idx];
            }
        }

        // Schedule each group
        unsigned max_supersteps = 0;
        std::vector<unsigned> proc_type_offsets(original_arch.getNumberOfProcessorTypes(), 0);
        
        for (std::size_t j = 0; j < iso_groups_for_set.size(); ++j) {
            unsigned supersteps_for_group = 0;
            auto status = schedule_isomorphism_group(schedule, vertex_map_for_set, iso_groups_for_set[j], subgraphs_for_set[j],
                                                     group_proc_allocations[j], global_ids_by_type, proc_type_offsets,
                                                     superstep_offset, supersteps_for_group);
            if (status != RETURN_STATUS::OSP_SUCCESS) return status;
            max_supersteps = std::max(max_supersteps, supersteps_for_group);

            // Advance offsets for the next group
            for (size_t k = 0; k < group_proc_allocations[j].size(); ++k) {
                proc_type_offsets[k] += group_proc_allocations[j][k];
            }
        }
        supersteps_in_set = max_supersteps;
        return RETURN_STATUS::OSP_SUCCESS;
    }

    RETURN_STATUS schedule_isomorphism_group(
        BspSchedule<Graph_t>& schedule,
        const std::vector<std::vector<vertex_idx_t<Graph_t>>>& vertex_map_for_set,
        const std::vector<size_t>& group_members,
        const constr_graph_t& rep_sub_dag,
        const std::vector<unsigned>& procs_for_group,
        const std::vector<std::vector<unsigned>>& global_ids_by_type,
        const std::vector<unsigned>& proc_type_offsets,
        unsigned superstep_offset,
        unsigned& supersteps_for_group) {

        const auto &instance = schedule.getInstance();
        const auto &original_arch = instance.getArchitecture();
        const size_t num_members = group_members.size();
        supersteps_for_group = 0;

        bool scarcity_found = false;
        if (num_members > 0) {
            for (unsigned type_idx = 0; type_idx < procs_for_group.size(); ++type_idx) {
                // Scarcity exists if processors are not perfectly divisible among members.
                if (procs_for_group[type_idx] % num_members != 0) {
                    scarcity_found = true;
                    break;
                }
            }
        }

        if (scarcity_found) {
            // --- SCARCITY/INDIVISIBLE CASE: Schedule sequentially on the shared processor block ---
            BspInstance<constr_graph_t> sub_instance(rep_sub_dag, this->createSubArchitecture(original_arch, procs_for_group));
            sub_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());
            auto & sub_architecture = sub_instance.getArchitecture();

            unsigned sequential_superstep_offset = 0;
            for (const auto &group_member_idx : group_members) {
                BspSchedule<constr_graph_t> sub_schedule(sub_instance);
                auto status = this->scheduler->computeSchedule(sub_schedule);
                if (status != RETURN_STATUS::OSP_SUCCESS && status != RETURN_STATUS::BEST_FOUND) return status;

                const auto sub_proc_type_count = sub_architecture.getProcessorTypeCount();
                std::vector<unsigned> sub_proc_type_corrections(sub_architecture.getNumberOfProcessorTypes(), 0);
                for (std::size_t k = 1; k < sub_proc_type_corrections.size(); ++k) {
                    sub_proc_type_corrections[k] = sub_proc_type_corrections[k - 1] + sub_proc_type_count[k - 1];
                }

                std::vector<vertex_idx_t<Graph_t>> sorted_component_vertices(vertex_map_for_set[group_member_idx].begin(), vertex_map_for_set[group_member_idx].end());
                std::sort(sorted_component_vertices.begin(), sorted_component_vertices.end());
                
                vertex_idx_t<constr_graph_t> subdag_vertex = 0;
                for (const auto &vertex : sorted_component_vertices) {
                    const unsigned proc_in_sub_sched = sub_schedule.assignedProcessor(subdag_vertex);
                    const unsigned proc_type = sub_architecture.processorType(proc_in_sub_sched);
                    const unsigned local_proc_id_within_type = proc_in_sub_sched - sub_proc_type_corrections[proc_type];
                    unsigned global_proc_id = global_ids_by_type[proc_type][proc_type_offsets[proc_type] + local_proc_id_within_type];
                    
                    schedule.setAssignedProcessor(vertex, global_proc_id);
                    schedule.setAssignedSuperstep(vertex, superstep_offset + sequential_superstep_offset + sub_schedule.assignedSuperstep(subdag_vertex));
                    subdag_vertex++;
                }
                sequential_superstep_offset += sub_schedule.numberOfSupersteps();
            }
            supersteps_for_group = sequential_superstep_offset;

        } else {
            // --- ABUNDANCE/DIVISIBLE CASE: Replicate Schedule ---
            std::vector<unsigned> single_sub_dag_proc_types = procs_for_group;
            if (num_members > 0) {
                for(auto& count : single_sub_dag_proc_types) count /= static_cast<unsigned>(num_members);
            }

            BspInstance<constr_graph_t> sub_instance(rep_sub_dag, this->createSubArchitecture(original_arch, single_sub_dag_proc_types));
            sub_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());
            
            BspSchedule<constr_graph_t> sub_schedule(sub_instance);
            auto status = this->scheduler->computeSchedule(sub_schedule);
            if (status != RETURN_STATUS::OSP_SUCCESS && status != RETURN_STATUS::BEST_FOUND) return status;
            
            const auto sub_proc_type_count = sub_schedule.getInstance().getArchitecture().getProcessorTypeCount();
            std::vector<unsigned> sub_proc_type_corrections(sub_proc_type_count.size(), 0);
            for (std::size_t k = 1; k < sub_proc_type_corrections.size(); ++k) {
                sub_proc_type_corrections[k] = sub_proc_type_corrections[k - 1] + sub_proc_type_count[k - 1];
            }

            std::vector<unsigned> current_member_proc_offsets = proc_type_offsets;
            for (const auto &group_member_idx : group_members) {
                std::vector<vertex_idx_t<Graph_t>> sorted_component_vertices(vertex_map_for_set[group_member_idx].begin(), vertex_map_for_set[group_member_idx].end());
                std::sort(sorted_component_vertices.begin(), sorted_component_vertices.end());

                vertex_idx_t<constr_graph_t> subdag_vertex = 0;
                for (const auto &vertex : sorted_component_vertices) {
                    const unsigned proc_in_sub_sched = sub_schedule.assignedProcessor(subdag_vertex);
                    const unsigned proc_type = sub_schedule.getInstance().getArchitecture().processorType(proc_in_sub_sched);
                    const unsigned local_proc_id_within_type = proc_in_sub_sched - sub_proc_type_corrections[proc_type];
                    unsigned global_proc_id = global_ids_by_type[proc_type][current_member_proc_offsets[proc_type] + local_proc_id_within_type];
                    
                    schedule.setAssignedProcessor(vertex, global_proc_id);
                    schedule.setAssignedSuperstep(vertex, superstep_offset + sub_schedule.assignedSuperstep(subdag_vertex));
                    subdag_vertex++;
                }
                for (size_t k = 0; k < sub_proc_type_count.size(); ++k) {
                    current_member_proc_offsets[k] += sub_proc_type_count[k];
                }
            }
            supersteps_for_group = sub_schedule.numberOfSupersteps();
        }
        return RETURN_STATUS::OSP_SUCCESS;
    }
};

template<typename Graph_t>
using IsomorphicWavefrontComponentScheduler_def_int_t = IsomorphicWavefrontComponentScheduler<Graph_t, boost_graph_int_t>;

}