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

#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include <iostream>
#include <numeric>

namespace osp {

/**
 * @brief A scheduler for a single trimmed group, which consists of multiple isomorphic connected components.
 *
 * @class TrimmedGroupScheduler
 *
 * This scheduler functions similarly to the ConnectedComponentScheduler but is tailored for a single,
 * potentially disconnected, subgraph that resulted from merging smaller isomorphic subgraphs. It divides
 * the input graph into its weakly connected components and schedules them on proportionally allocated processors.
 */
template<typename Constr_Graph_t>
class TrimmedGroupScheduler : public Scheduler<Constr_Graph_t> {

    Scheduler<Constr_Graph_t> *sub_scheduler;
    unsigned min_non_zero_procs_;

    static constexpr bool verbose = false;

  public:
    TrimmedGroupScheduler(Scheduler<Constr_Graph_t> &scheduler, unsigned min_non_zero_procs)
        : sub_scheduler(&scheduler), min_non_zero_procs_(min_non_zero_procs) {}

    std::string getScheduleName() const override { return "TrimmedGroupScheduler"; }

    RETURN_STATUS computeSchedule(BspSchedule<Constr_Graph_t> &schedule) override {
        const auto &instance = schedule.getInstance();
        const Constr_Graph_t &dag = instance.getComputationalDag();
        const BspArchitecture<Constr_Graph_t> &arch = instance.getArchitecture();

        // Find the weakly connected components. These are assumed to be isomorphic subgraphs.
        std::vector<vertex_idx_t<Constr_Graph_t>> component_map(dag.num_vertices());
        size_t num_components = compute_weakly_connected_components(dag, component_map);

        if (num_components == 0) {
            schedule.setNumberOfSupersteps(0);
            return RETURN_STATUS::OSP_SUCCESS;
        }

        if constexpr (verbose) {
            std::cout << "  [TrimmedGroupScheduler] min_non_zero_procs: " << min_non_zero_procs_
                      << ", num_components: " << num_components << std::endl;
        }

        // Group vertices by component.
        std::vector<std::vector<vertex_idx_t<Constr_Graph_t>>> components_vertices(num_components);
        for (vertex_idx_t<Constr_Graph_t> v = 0; v < dag.num_vertices(); ++v) {
            components_vertices[component_map[v]].push_back(v);
        }

        // Distribute components among processor types.
        // The goal is to assign `base_count` components to each processor type group,
        // plus one extra for the first `remainder` groups.
        const unsigned base_count = static_cast<unsigned>(num_components) / min_non_zero_procs_;
        const unsigned remainder = static_cast<unsigned>(num_components) % min_non_zero_procs_;

        std::vector<std::vector<unsigned>> component_indices_per_group(min_non_zero_procs_);
        unsigned component_cursor = 0;
        for (unsigned i = 0; i < min_non_zero_procs_; ++i) {
            unsigned num_to_assign = base_count + (i < remainder ? 1 : 0);
            for (unsigned j = 0; j < num_to_assign; ++j) {
                if (component_cursor < num_components) {
                    component_indices_per_group[i].push_back(component_cursor++);
                }
            }
        }

        // Determine the processor allocation for a single sub-problem.
        // Calculate offsets for processor types within the main 'arch' (passed to TrimmedGroupScheduler)
        std::vector<unsigned> arch_proc_type_offsets(arch.getNumberOfProcessorTypes(), 0);
        const auto &arch_proc_type_counts = arch.getProcessorTypeCount();
        for (unsigned type_idx = 1; type_idx < arch.getNumberOfProcessorTypes(); ++type_idx) {
            arch_proc_type_offsets[type_idx] = arch_proc_type_offsets[type_idx - 1] + arch_proc_type_counts[type_idx - 1];
        }

        std::vector<unsigned> sub_proc_counts(arch.getNumberOfProcessorTypes());
        std::vector<v_memw_t<Constr_Graph_t>> mem_weights(arch.getNumberOfProcessorTypes(), 0);
        for (unsigned type_idx = 0; type_idx < arch.getNumberOfProcessorTypes(); ++type_idx) {
            sub_proc_counts[type_idx] = arch.getProcessorTypeCount()[type_idx] / min_non_zero_procs_;
            mem_weights[type_idx] = static_cast<v_memw_t<Constr_Graph_t>>(arch.maxMemoryBoundProcType(type_idx));
        }

        if constexpr (verbose) {
            std::cout << "  [TrimmedGroupScheduler] Sub-problem processor counts per type: ";
            for (size_t type_idx = 0; type_idx < sub_proc_counts.size(); ++type_idx) {
                std::cout << "T" << type_idx << ":" << sub_proc_counts[type_idx] << " ";
            }
            std::cout << std::endl;
        }

        // Create the sub-architecture for one sub-problem.
        BspArchitecture<Constr_Graph_t> sub_arch(arch);
        sub_arch.SetProcessorsConsequTypes(sub_proc_counts, mem_weights);

        // Calculate offsets for processor types within the 'sub_arch'
        std::vector<unsigned> sub_arch_proc_type_offsets(sub_arch.getNumberOfProcessorTypes(), 0);
        const auto &sub_arch_proc_type_counts = sub_arch.getProcessorTypeCount();
        for (unsigned type_idx = 1; type_idx < sub_arch.getNumberOfProcessorTypes(); ++type_idx) {
            sub_arch_proc_type_offsets[type_idx] = sub_arch_proc_type_offsets[type_idx - 1] + sub_arch_proc_type_counts[type_idx - 1];
        }

        unsigned max_supersteps = 0;
        for (unsigned i = 0; i < min_non_zero_procs_; ++i) {

            std::vector<vertex_idx_t<Constr_Graph_t>> group_vertices;
            for (unsigned comp_idx : component_indices_per_group[i]) {
                group_vertices.insert(group_vertices.end(), components_vertices[comp_idx].begin(), components_vertices[comp_idx].end());
            }
            std::sort(group_vertices.begin(), group_vertices.end());

            BspInstance<Constr_Graph_t> sub_instanc;
            sub_instanc.setArchitecture(sub_arch);                                                                          // Set the sub-architecture
            sub_instanc.setNodeProcessorCompatibility(instance.getNodeProcessorCompatibilityMatrix());                      // Inherit compatibility
            auto global_to_local_map = create_induced_subgraph_map(dag, sub_instanc.getComputationalDag(), group_vertices); // Create induced subgraph

            // Create a schedule object for the sub-problem
            BspSchedule<Constr_Graph_t> sub_schedule(sub_instanc);

            // Call the sub-scheduler to compute the schedule for this group of components
            auto status = sub_scheduler->computeSchedule(sub_schedule);
            if (status != RETURN_STATUS::OSP_SUCCESS && status != RETURN_STATUS::BEST_FOUND)
                return status;

            // Map the sub-schedule back to the main schedule.
            for (const auto &v_global : group_vertices) {
                const auto v_local = global_to_local_map.at(v_global);
                const unsigned sub_proc = sub_schedule.assignedProcessor(v_local);
                const unsigned sub_superstep = sub_schedule.assignedSuperstep(v_local);

                // Determine the processor type and its local index within that type in the sub_arch
                const unsigned proc_type = sub_arch.processorType(sub_proc);
                const unsigned local_idx_within_type = sub_proc - sub_arch_proc_type_offsets[proc_type];

                // Calculate the global processor ID by combining:
                // The base offset of this processor type in the main 'arch'.
                // The offset for the current 'i'-th block of processors of this type.
                // The local index within that type block.
                const unsigned global_proc = arch_proc_type_offsets[proc_type] +
                                             (i * sub_proc_counts[proc_type]) +
                                             local_idx_within_type;
                schedule.setAssignedProcessor(v_global, global_proc);
                schedule.setAssignedSuperstep(v_global, sub_superstep);
            }
            max_supersteps = std::max(max_supersteps, sub_schedule.numberOfSupersteps());
        }

        schedule.setNumberOfSupersteps(max_supersteps);
        return RETURN_STATUS::OSP_SUCCESS;
    }
};

} // namespace osp