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

#include <iostream>
#include <numeric>

#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

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
template <typename ConstrGraphT>
class TrimmedGroupScheduler : public Scheduler<ConstrGraphT> {
    Scheduler<ConstrGraphT> *subScheduler_;
    unsigned minNonZeroProcs_;

    static constexpr bool verbose_ = false;

  public:
    TrimmedGroupScheduler(Scheduler<ConstrGraphT> &scheduler, unsigned minNonZeroProcs)
        : subScheduler_(&scheduler), minNonZeroProcs_(minNonZeroProcs) {}

    std::string getScheduleName() const override { return "TrimmedGroupScheduler"; }

    ReturnStatus computeSchedule(BspSchedule<ConstrGraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();
        const ConstrGraphT &dag = instance.GetComputationalDag();
        const BspArchitecture<ConstrGraphT> &arch = instance.GetArchitecture();

        // Find the weakly connected components. These are assumed to be isomorphic subgraphs.
        std::vector<VertexIdxT<Constr_Graph_t>> componentMap(dag.NumVertices());
        size_t numComponents = compute_weakly_connected_components(dag, component_map);

        if (numComponents == 0) {
            schedule.setNumberOfSupersteps(0);
            return ReturnStatus::OSP_SUCCESS;
        }

        if constexpr (verbose_) {
            std::cout << "  [TrimmedGroupScheduler] min_non_zero_procs: " << minNonZeroProcs_
                      << ", num_components: " << numComponents << std::endl;
        }

        // Group vertices by component.
        std::vector<std::vector<VertexIdxT<Constr_Graph_t>>> componentsVertices(numComponents);
        for (VertexIdxT<Constr_Graph_t> v = 0; v < dag.NumVertices(); ++v) {
            componentsVertices[component_map[v]].push_back(v);
        }

        // Distribute components among processor types.
        // The goal is to assign `base_count` components to each processor type group,
        // plus one extra for the first `remainder` groups.
        const unsigned baseCount = static_cast<unsigned>(numComponents) / minNonZeroProcs_;
        const unsigned remainder = static_cast<unsigned>(numComponents) % minNonZeroProcs_;

        std::vector<std::vector<unsigned>> componentIndicesPerGroup(minNonZeroProcs_);
        unsigned componentCursor = 0;
        for (unsigned i = 0; i < minNonZeroProcs_; ++i) {
            unsigned numToAssign = baseCount + (i < remainder ? 1 : 0);
            for (unsigned j = 0; j < numToAssign; ++j) {
                if (componentCursor < numComponents) {
                    componentIndicesPerGroup[i].push_back(componentCursor++);
                }
            }
        }

        // Determine the processor allocation for a single sub-problem.
        // Calculate offsets for processor types within the main 'arch' (passed to TrimmedGroupScheduler)
        std::vector<unsigned> archProcTypeOffsets(arch.getNumberOfProcessorTypes(), 0);
        const auto &archProcTypeCounts = arch.getProcessorTypeCount();
        for (unsigned typeIdx = 1; typeIdx < arch.getNumberOfProcessorTypes(); ++typeIdx) {
            archProcTypeOffsets[typeIdx] = archProcTypeOffsets[typeIdx - 1] + archProcTypeCounts[typeIdx - 1];
        }

        std::vector<unsigned> subProcCounts(arch.getNumberOfProcessorTypes());
        std::vector<VMemwT<Constr_Graph_t>> memWeights(arch.getNumberOfProcessorTypes(), 0);
        for (unsigned typeIdx = 0; typeIdx < arch.getNumberOfProcessorTypes(); ++typeIdx) {
            subProcCounts[typeIdx] = arch.getProcessorTypeCount()[typeIdx] / minNonZeroProcs_;
            memWeights[typeIdx] = static_cast<VMemwT<Constr_Graph_t>>(arch.maxMemoryBoundProcType(typeIdx));
        }

        if constexpr (verbose_) {
            std::cout << "  [TrimmedGroupScheduler] Sub-problem processor counts per type: ";
            for (size_t typeIdx = 0; typeIdx < subProcCounts.size(); ++typeIdx) {
                std::cout << "T" << typeIdx << ":" << subProcCounts[typeIdx] << " ";
            }
            std::cout << std::endl;
        }

        // Create the sub-architecture for one sub-problem.
        BspArchitecture<ConstrGraphT> subArch(arch);
        subArch.SetProcessorsConsequTypes(subProcCounts, mem_weights);

        // Calculate offsets for processor types within the 'sub_arch'
        std::vector<unsigned> subArchProcTypeOffsets(subArch.getNumberOfProcessorTypes(), 0);
        const auto &subArchProcTypeCounts = subArch.getProcessorTypeCount();
        for (unsigned typeIdx = 1; typeIdx < subArch.getNumberOfProcessorTypes(); ++typeIdx) {
            subArchProcTypeOffsets[typeIdx] = subArchProcTypeOffsets[typeIdx - 1] + subArchProcTypeCounts[typeIdx - 1];
        }

        unsigned maxSupersteps = 0;
        for (unsigned i = 0; i < minNonZeroProcs_; ++i) {
            std::vector<VertexIdxT<Constr_Graph_t>> groupVertices;
            for (unsigned compIdx : componentIndicesPerGroup[i]) {
                groupVertices.insert(
                    group_vertices.end(), components_vertices[compIdx].begin(), components_vertices[compIdx].end());
            }
            std::sort(group_vertices.begin(), group_vertices.end());

            BspInstance<ConstrGraphT> subInstanc;
            subInstanc.GetArchitecture() = subArch;
            subInstanc.setNodeProcessorCompatibility(instance.getNodeProcessorCompatibilityMatrix());    // Inherit compatibility
            auto globalToLocalMap = create_induced_subgraph_map(
                dag, subInstanc.GetComputationalDag(), group_vertices);    // Create induced subgraph

            // Create a schedule object for the sub-problem
            BspSchedule<ConstrGraphT> subSchedule(subInstanc);

            // Call the sub-scheduler to compute the schedule for this group of components
            auto status = subScheduler_->computeSchedule(subSchedule);
            if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::BEST_FOUND) {
                return status;
            }

            // Map the sub-schedule back to the main schedule.
            for (const auto &v_global : group_vertices) {
                const auto v_local = global_to_local_map.at(v_global);
                const unsigned sub_proc = sub_schedule.AssignedProcessor(v_local);
                const unsigned sub_superstep = sub_schedule.AssignedSuperstep(v_local);

                // Determine the processor type and its local index within that type in the sub_arch
                const unsigned proc_type = sub_arch.processorType(sub_proc);
                const unsigned local_idx_within_type = sub_proc - sub_arch_proc_type_offsets[proc_type];

                // Calculate the global processor ID by combining:
                // The base offset of this processor type in the main 'arch'.
                // The offset for the current 'i'-th block of processors of this type.
                // The local index within that type block.
                const unsigned global_proc
                    = arch_proc_type_offsets[proc_type] + (i * sub_proc_counts[proc_type]) + local_idx_within_type;
                schedule.setAssignedProcessor(v_global, global_proc);
                schedule.setAssignedSuperstep(v_global, sub_superstep);
            }
            maxSupersteps = std::max(maxSupersteps, subSchedule.NumberOfSupersteps());
        }

        schedule.setNumberOfSupersteps(maxSupersteps);
        return ReturnStatus::OSP_SUCCESS;
    }
};

}    // namespace osp
