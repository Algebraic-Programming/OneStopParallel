/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <numeric>
#include <string>
#include <vector>

#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

namespace osp {

/**
 * @class TrimmedGroupScheduler
 * @brief A scheduler for a single trimmed group consisting of multiple isomorphic connected components.
 *
 * This scheduler partitions a disconnected subgraph (a pruned group) into its weakly connected components.
 * It assumes these components are isomorphic and distributes them among the available processor groups
 * to balance the load.
 *
 * @tparam ConstrGraphT The type of the graph.
 */
template <typename ConstrGraphT>
class TrimmedGroupScheduler : public Scheduler<ConstrGraphT> {
    Scheduler<ConstrGraphT> *subScheduler_;
    unsigned minNonZeroProcs_;

  public:
    /**
     * @brief Constructs a TrimmedGroupScheduler.
     * @param scheduler The sub-scheduler to use for scheduling individual component groups.
     * @param minNonZeroProcs The minimum number of non-zero processors to utilize.
     */
    TrimmedGroupScheduler(Scheduler<ConstrGraphT> &scheduler, unsigned minNonZeroProcs)
        : subScheduler_(&scheduler), minNonZeroProcs_(minNonZeroProcs) {}

    std::string GetScheduleName() const override { return "TrimmedGroupScheduler"; }

    ReturnStatus ComputeSchedule(BspSchedule<ConstrGraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();
        const ConstrGraphT &dag = instance.GetComputationalDag();

        std::vector<VertexIdxT<ConstrGraphT>> componentMap(dag.NumVertices());
        size_t numComponents = ComputeWeaklyConnectedComponents(dag, componentMap);

        if (numComponents == 0) {
            schedule.SetNumberOfSupersteps(0);
            return ReturnStatus::OSP_SUCCESS;
        }

        std::vector<std::vector<VertexIdxT<ConstrGraphT>>> componentsVertices(numComponents);
        for (VertexIdxT<ConstrGraphT> v = 0; v < dag.NumVertices(); ++v) {
            componentsVertices[componentMap[v]].push_back(v);
        }

        auto componentIndicesPerGroup = DistributeComponents(numComponents);
        auto subArch = BuildSubArchitecture(instance.GetArchitecture());

        return SolveAndMapSubProblems(schedule, componentIndicesPerGroup, componentsVertices, subArch);
    }

  private:
    /**
     * @brief Distributes components among the processor groups.
     * @param numComponents Total number of components.
     * @return A vector where each element is a list of component indices assigned to a processor group.
     */
    std::vector<std::vector<unsigned>> DistributeComponents(size_t numComponents) {
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
        return componentIndicesPerGroup;
    }

    /**
     * @brief Builds the architecture for a single sub-problem (one processor group).
     * @param arch The global architecture.
     * @return The sub-architecture.
     */
    BspArchitecture<ConstrGraphT> BuildSubArchitecture(const BspArchitecture<ConstrGraphT> &arch) {
        std::vector<unsigned> subProcCounts(arch.GetNumberOfProcessorTypes());
        std::vector<VMemwT<ConstrGraphT>> memWeights(arch.GetNumberOfProcessorTypes(), 0);

        for (unsigned typeIdx = 0; typeIdx < arch.GetNumberOfProcessorTypes(); ++typeIdx) {
            subProcCounts[typeIdx] = arch.GetProcessorTypeCount()[typeIdx] / minNonZeroProcs_;
            memWeights[typeIdx] = static_cast<VMemwT<ConstrGraphT>>(arch.MaxMemoryBoundProcType(typeIdx));
        }

        BspArchitecture<ConstrGraphT> subArch(arch);
        subArch.SetProcessorsConsequTypes(subProcCounts, memWeights);
        return subArch;
    }

    /**
     * @brief Solves the sub-schedule for each group and maps the results back to the global schedule.
     */
    ReturnStatus SolveAndMapSubProblems(BspSchedule<ConstrGraphT> &schedule,
                                        const std::vector<std::vector<unsigned>> &componentIndicesPerGroup,
                                        const std::vector<std::vector<VertexIdxT<ConstrGraphT>>> &componentsVertices,
                                        const BspArchitecture<ConstrGraphT> &subArch) {
        const auto &instance = schedule.GetInstance();
        const auto &arch = instance.GetArchitecture();
        const auto &dag = instance.GetComputationalDag();

        // Calculate offsets for mapping local sub-processor IDs to global processor IDs
        std::vector<unsigned> archProcTypeOffsets(arch.GetNumberOfProcessorTypes(), 0);
        const auto &archProcTypeCounts = arch.GetProcessorTypeCount();
        for (unsigned typeIdx = 1; typeIdx < arch.GetNumberOfProcessorTypes(); ++typeIdx) {
            archProcTypeOffsets[typeIdx] = archProcTypeOffsets[typeIdx - 1] + archProcTypeCounts[typeIdx - 1];
        }

        std::vector<unsigned> subArchProcTypeOffsets(subArch.GetNumberOfProcessorTypes(), 0);
        const auto &subArchProcTypeCounts = subArch.GetProcessorTypeCount();
        for (unsigned typeIdx = 1; typeIdx < subArch.GetNumberOfProcessorTypes(); ++typeIdx) {
            subArchProcTypeOffsets[typeIdx] = subArchProcTypeOffsets[typeIdx - 1] + subArchProcTypeCounts[typeIdx - 1];
        }

        std::vector<unsigned> subProcCounts = subArch.GetProcessorTypeCount();
        unsigned maxSupersteps = 0;

        for (unsigned i = 0; i < minNonZeroProcs_; ++i) {
            if (componentIndicesPerGroup[i].empty()) continue;

            std::vector<VertexIdxT<ConstrGraphT>> groupVertices;
            for (unsigned compIdx : componentIndicesPerGroup[i]) {
                groupVertices.insert(groupVertices.end(), componentsVertices[compIdx].begin(), componentsVertices[compIdx].end());
            }
            std::sort(groupVertices.begin(), groupVertices.end());

            BspInstance<ConstrGraphT> subInstance;
            subInstance.GetArchitecture() = subArch;
            subInstance.SetNodeProcessorCompatibility(instance.GetNodeProcessorCompatibilityMatrix());

            auto globalToLocalMap = CreateInducedSubgraphMap(dag, subInstance.GetComputationalDag(), groupVertices);

            BspSchedule<ConstrGraphT> subSchedule(subInstance);
            auto status = subScheduler_->ComputeSchedule(subSchedule);

            if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::BEST_FOUND) {
                return status;
            }

            for (const auto &vGlobal : groupVertices) {
                const auto vLocal = globalToLocalMap.at(vGlobal);
                const unsigned subProc = subSchedule.AssignedProcessor(vLocal);
                const unsigned subSuperstep = subSchedule.AssignedSuperstep(vLocal);

                const unsigned procType = subArch.ProcessorType(subProc);
                const unsigned localIdxWithinType = subProc - subArchProcTypeOffsets[procType];
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
