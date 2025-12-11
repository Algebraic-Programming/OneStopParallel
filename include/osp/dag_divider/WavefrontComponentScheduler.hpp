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

namespace osp {

/**
 * @class WavefrontComponentScheduler
 * @brief Schedules wavefronts by treating each component individually.
 */
template <typename Graph_t, typename constr_graph_t>
class WavefrontComponentScheduler : public AbstractWavefrontScheduler<Graph_t, constr_graph_t> {
  public:
    WavefrontComponentScheduler(IDagDivider<Graph_t> &div, Scheduler<constr_graph_t> &scheduler_)
        : AbstractWavefrontScheduler<Graph_t, constr_graph_t>(div, scheduler_) {}

    std::string getScheduleName() const override { return "WavefrontComponentScheduler"; }

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {
        const auto &instance = schedule.getInstance();
        const auto &original_arch = instance.getArchitecture();
        const auto &original_proc_type_count = original_arch.getProcessorTypeCount();
        const auto &computational_dag = instance.getComputationalDag();

        std::vector<std::vector<unsigned>> global_ids_by_type(original_arch.getNumberOfProcessorTypes());
        for (unsigned i = 0; i < original_arch.numberOfProcessors(); ++i) {
            global_ids_by_type[original_arch.processorType(i)].push_back(i);
        }

        auto vertex_maps = this->divider->divide(computational_dag);
        unsigned superstep_offset = 0;

        for (std::size_t i = 0; i < vertex_maps.size(); ++i) {    // For each wavefront set
            if (this->enable_debug_prints) {
                std::cout << "\n--- Processing Wavefront Set " << i << " (No Isomorphism) ---" << std::endl;
            }

            const auto &components = vertex_maps[i];
            std::vector<constr_graph_t> sub_dags(components.size());
            std::vector<std::vector<double>> work_by_type(components.size(),
                                                          std::vector<double>(original_proc_type_count.size(), 0.0));

            for (size_t j = 0; j < components.size(); ++j) {
                create_induced_subgraph(computational_dag, sub_dags[j], components[j]);
                for (unsigned type_idx = 0; type_idx < original_proc_type_count.size(); ++type_idx) {
                    work_by_type[j][type_idx] = sumOfCompatibleWorkWeights(sub_dags[j], instance, type_idx);
                }
            }

            assert(this->validateWorkDistribution(sub_dags, instance));

            // Distribute Processors
            std::vector<std::vector<unsigned>> proc_allocations(components.size(),
                                                                std::vector<unsigned>(original_proc_type_count.size()));
            for (unsigned type_idx = 0; type_idx < original_proc_type_count.size(); ++type_idx) {
                std::vector<double> work_for_this_type(components.size());
                for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
                    work_for_this_type[comp_idx] = work_by_type[comp_idx][type_idx];
                }

                std::vector<unsigned> type_allocation;
                bool starvation_hit
                    = this->distributeProcessors(original_proc_type_count[type_idx], work_for_this_type, type_allocation);

                if (starvation_hit) {
                    if constexpr (this->enable_debug_prints) {
                        std::cerr << "ERROR: Processor starvation detected for type " << type_idx << " in wavefront set " << i
                                  << ". Not enough processors to assign one to each active component." << std::endl;
                    }
                    return RETURN_STATUS::ERROR;
                }

                for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
                    proc_allocations[comp_idx][type_idx] = type_allocation[comp_idx];
                }
            }

            unsigned max_number_supersteps = 0;
            std::vector<unsigned> proc_type_offsets(original_arch.getNumberOfProcessorTypes(), 0);

            for (std::size_t j = 0; j < components.size(); ++j) {
                BspArchitecture<constr_graph_t> sub_architecture = this->createSubArchitecture(original_arch, proc_allocations[j]);
                if constexpr (this->enable_debug_prints) {
                    std::cout << "  Component " << j << " sub-architecture: { ";
                    for (unsigned type_idx = 0; type_idx < sub_architecture.getNumberOfProcessorTypes(); ++type_idx) {
                        std::cout << "Type " << type_idx << ": " << sub_architecture.getProcessorTypeCount()[type_idx] << "; ";
                    }
                    std::cout << "}" << std::endl;
                }

                BspInstance<constr_graph_t> sub_instance(sub_dags[j], sub_architecture);
                sub_instance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

                BspSchedule<constr_graph_t> sub_schedule(sub_instance);
                const auto status = this->scheduler->computeSchedule(sub_schedule);
                if (status != RETURN_STATUS::OSP_SUCCESS && status != RETURN_STATUS::BEST_FOUND) {
                    return status;
                }

                const auto sub_proc_type_count = sub_architecture.getProcessorTypeCount();
                std::vector<unsigned> sub_proc_type_corrections(sub_architecture.getNumberOfProcessorTypes(), 0);
                for (std::size_t k = 1; k < sub_proc_type_corrections.size(); ++k) {
                    sub_proc_type_corrections[k] = sub_proc_type_corrections[k - 1] + sub_proc_type_count[k - 1];
                }

                vertex_idx_t<constr_graph_t> subdag_vertex = 0;
                std::vector<vertex_idx_t<Graph_t>> sorted_component_vertices(components[j].begin(), components[j].end());
                std::sort(sorted_component_vertices.begin(), sorted_component_vertices.end());

                for (const auto &vertex : sorted_component_vertices) {
                    const unsigned proc_in_sub_sched = sub_schedule.assignedProcessor(subdag_vertex);
                    const unsigned proc_type = sub_architecture.processorType(proc_in_sub_sched);
                    const unsigned local_proc_id_within_type = proc_in_sub_sched - sub_proc_type_corrections[proc_type];
                    unsigned global_proc_id
                        = global_ids_by_type[proc_type][proc_type_offsets[proc_type] + local_proc_id_within_type];

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
};

template <typename Graph_t>
using WavefrontComponentScheduler_def_int_t = WavefrontComponentScheduler<Graph_t, boost_graph_int_t>;

}    // namespace osp
