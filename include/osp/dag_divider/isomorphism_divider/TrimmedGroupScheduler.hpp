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

    std::string GetScheduleName() const override { return "TrimmedGroupScheduler"; }

    ReturnStatus ComputeSchedule(BspSchedule<ConstrGraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();
        const ConstrGraphT &dag = instance.GetComputationalDag();
        const BspArchitecture<ConstrGraphT> &arch = instance.GetArchitecture();

        // Find the weakly connected components. These are assumed to be isomorphic subgraphs.
        std::vector<VertexIdxT<Constr_Graph_t>> componentMap(dag.NumVertices());
        size_t numComponents = ComputeWeaklyConnectedComponents(dag, componentMap);

        if (numComponents == 0) {
            schedule.SetNumberOfSupersteps(0);
            return ReturnStatus::OSP_SUCCESS;
        }

        if constexpr (verbose_) {
            std::cout << "  [TrimmedGroupScheduler] min_non_zero_procs: " << minNonZeroProcs_
                      << ", num_components: " << numComponents << std::endl;
        }

        // Group vertices by component.
        std::vector<std::vector<VertexIdxT<Constr_Graph_t>>> componentsVertices(numComponents);
        for (VertexIdxT<Constr_Graph_t> v = 0; v < dag.NumVertices(); ++v) {
            componentsVertices[componentMap[v]].push_back(v);
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
        std::vector<unsigned> archProcTypeOffsets(arch.GetNumberOfProcessorTypes(), 0);
        const auto &archProcTypeCounts = arch.GetProcessorTypeCount();
        for (unsigned typeIdx = 1; typeIdx < arch.GetNumberOfProcessorTypes(); ++typeIdx) {
            archProcTypeOffsets[typeIdx] = archProcTypeOffsets[typeIdx - 1] + archProcTypeCounts[typeIdx - 1];
        }

        std::vector<unsigned> subProcCounts(arch.GetNumberOfProcessorTypes());
        std::vector<VMemwT<Constr_Graph_t>> memWeights(arch.GetNumberOfProcessorTypes(), 0);
        for (unsigned typeIdx = 0; typeIdx < arch.GetNumberOfProcessorTypes(); ++typeIdx) {
            subProcCounts[typeIdx] = arch.GetProcessorTypeCount()[typeIdx] / minNonZeroProcs_;
            memWeights[typeIdx] = static_cast<VMemwT<Constr_Graph_t>>(arch.MaxMemoryBoundProcType(typeIdx));
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
        subArch.SetProcessorsConsequTypes(subProcCounts, memWeights);

        // Calculate offsets for processor types within the 'sub_arch'
        std::vector<unsigned> subArchProcTypeOffsets(subArch.GetNumberOfProcessorTypes(), 0);
        const auto &subArchProcTypeCounts = subArch.GetProcessorTypeCount();
        for (unsigned typeIdx = 1; typeIdx < subArch.GetNumberOfProcessorTypes(); ++typeIdx) {
            subArchProcTypeOffsets[typeIdx] = subArchProcTypeOffsets[typeIdx - 1] + subArchProcTypeCounts[typeIdx - 1];
        }

        unsigned maxSupersteps = 0;
        for (unsigned i = 0; i < minNonZeroProcs_; ++i) {
            std::vector<VertexIdxT<Constr_Graph_t>> groupVertices;
            for (unsigned compIdx : componentIndicesPerGroup[i]) {
                groupVertices.insert(groupVertices.end(), componentsVertices[compIdx].begin(), componentsVertices[compIdx].end());
            }
            std::sort(groupVertices.begin(), groupVertices.end());

            BspInstance<ConstrGraphT> subInstance;
            subInstance.GetArchitecture() = subArch;
            subInstance.SetNodeProcessorCompatibility(instance.GetNodeProcessorCompatibilityMatrix());    // Inherit compatibility
            auto globalToLocalMap
                = CreateInducedSubgraphMap(dag, subInstance.GetComputationalDag(), groupVertices);    // Create induced subgraph

            // Create a schedule object for the sub-problem
            BspSchedule<ConstrGraphT> subSchedule(subInstance);

            // Call the sub-scheduler to compute the schedule for this group of components
            auto status = subScheduler_->ComputeSchedule(subSchedule);
            if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::BEST_FOUND) {
                return status;
            }

            // Map the sub-schedule back to the main schedule.
            for (const auto &vGlobal : groupVertices) {
                const auto vLocal = globalToLocalMap.at(vGlobal);
                const unsigned subProc = subSchedule.AssignedProcessor(vLocal);
                const unsigned subSuperstep = subSchedule.AssignedSuperstep(vLocal);

                // Determine the processor type and its local index within that type in the sub_arch
                const unsigned procType = subArch.ProcessorType(subProc);
                const unsigned localIdxWithinType = subProc - subArchProcTypeOffsets[procType];

                // Calculate the global processor ID by combining:
                // The base offset of this processor type in the main 'arch'.
                // The offset for the current 'i'-th block of processors of this type.
                // The local index within that type block.
                const unsigned globalProc = archProcTypeOffsets[procType] + (i * subProcCounts[procType]) + localIdxWithinType;
                schedule.SetAssignedProcessor(vGlobal, globalProc);
                schedule.SetAssignedSuperstep(vGlobal, subSuperstep);
            }
            maxSupersteps = std::max(maxSupersteps, subSchedule.NumberOfSupersteps());
        }

        schedule.SetNumberOfSupersteps(maxSupersteps);
        return ReturnStatus::OSP_SUCCESS;
    }
};

}    // namespace osp
