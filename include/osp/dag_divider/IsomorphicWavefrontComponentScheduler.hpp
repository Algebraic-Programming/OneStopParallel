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
template <typename GraphT, typename ConstrGraphT>
class IsomorphicWavefrontComponentScheduler : public AbstractWavefrontScheduler<GraphT, ConstrGraphT> {
  public:
    IsomorphicWavefrontComponentScheduler(IDagDivider<GraphT> &div, Scheduler<ConstrGraphT> &scheduler)
        : AbstractWavefrontScheduler<GraphT, ConstrGraphT>(div, scheduler) {}

    std::string GetScheduleName() const override { return "IsomorphicWavefrontComponentScheduler"; }

    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();
        const auto &originalArch = instance.GetArchitecture();

        std::vector<std::vector<unsigned>> globalIdsByType(originalArch.GetNumberOfProcessorTypes());
        for (unsigned i = 0; i < originalArch.NumberOfProcessors(); ++i) {
            globalIdsByType[originalArch.ProcessorType(i)].push_back(i);
        }

        IsomorphismGroups<GraphT, ConstrGraphT> isoGroups;
        std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> vertexMaps
            = this->divider_->divide(instance.GetComputationalDag());
        isoGroups.compute_isomorphism_groups(vertex_maps, instance.GetComputationalDag());

        unsigned superstepOffset = 0;
        for (std::size_t i = 0; i < vertexMaps.size(); ++i) {
            if (this->enableDebugPrints_) {
                std::cout << "\n--- Processing Wavefront Set " << i << " ---" << std::endl;
            }

            unsigned superstepsInSet = 0;
            auto status = process_wavefront_set(schedule,
                                                vertex_maps[i],
                                                iso_groups.get_isomorphism_groups()[i],
                                                iso_groups.get_isomorphism_groups_subgraphs()[i],
                                                global_ids_by_type,
                                                superstep_offset,
                                                supersteps_in_set);
            if (status != ReturnStatus::OSP_SUCCESS) {
                return status;
            }
            superstepOffset += superstepsInSet;
        }
        return ReturnStatus::OSP_SUCCESS;
    }

  private:
    ReturnStatus ProcessWavefrontSet(BspSchedule<GraphT> &schedule,
                                     const std::vector<std::vector<VertexIdxT<GraphT>>> &vertexMapForSet,
                                     const std::vector<std::vector<size_t>> &isoGroupsForSet,
                                     const std::vector<ConstrGraphT> &subgraphsForSet,
                                     const std::vector<std::vector<unsigned>> &globalIdsByType,
                                     unsigned superstepOffset,
                                     unsigned &superstepsInSet) {
        const auto &instance = schedule.GetInstance();
        const auto &originalArch = instance.GetArchitecture();
        const auto &originalProcTypeCount = originalArch.getProcessorTypeCount();

        if constexpr (this->enableDebugPrints_) {
            std::cout << "  Found " << isoGroupsForSet.size() << " isomorphism groups in this wavefront set." << std::endl;
        }

        // Calculate work for each isomorphism group
        std::vector<std::vector<double>> groupWorkByType(isoGroupsForSet.size(),
                                                         std::vector<double>(originalProcTypeCount.size(), 0.0));

        for (std::size_t j = 0; j < isoGroupsForSet.size(); ++j) {
            const ConstrGraphT &repSubDag = subgraphsForSet[j];
            for (unsigned typeIdx = 0; typeIdx < originalProcTypeCount.size(); ++typeIdx) {
                const double repWorkForType = sumOfCompatibleWorkWeights(repSubDag, instance, typeIdx);
                groupWorkByType[j][typeIdx] = repWorkForType * static_cast<double>(isoGroupsForSet[j].size());
            }
        }

        assert(this->ValidateWorkDistribution(subgraphsForSet, instance));

        // Distribute processors among isomorphism groups
        std::vector<std::vector<unsigned>> groupProcAllocations(isoGroupsForSet.size(),
                                                                std::vector<unsigned>(originalProcTypeCount.size()));

        for (unsigned typeIdx = 0; typeIdx < originalProcTypeCount.size(); ++typeIdx) {
            std::vector<double> workForThisType;
            for (size_t groupIdx = 0; groupIdx < isoGroupsForSet.size(); ++groupIdx) {
                workForThisType.push_back(groupWorkByType[groupIdx][typeIdx]);
            }

            std::vector<unsigned> typeAllocation;
            bool starvationHit = this->DistributeProcessors(originalProcTypeCount[typeIdx], workForThisType, typeAllocation);

            if (starvationHit) {
                if constexpr (this->enableDebugPrints_) {
                    std::cerr << "ERROR: Processor starvation detected for type " << typeIdx
                              << ". Not enough processors to assign one to each active isomorphism group." << std::endl;
                }
                return ReturnStatus::ERROR;
            }

            for (size_t groupIdx = 0; groupIdx < isoGroupsForSet.size(); ++groupIdx) {
                groupProcAllocations[groupIdx][typeIdx] = typeAllocation[groupIdx];
            }
        }

        // Schedule each group
        unsigned maxSupersteps = 0;
        std::vector<unsigned> procTypeOffsets(originalArch.GetNumberOfProcessorTypes(), 0);

        std::vector<unsigned> numSuperstepsPerIsoGroup(isoGroupsForSet.size());

        for (std::size_t j = 0; j < isoGroupsForSet.size(); ++j) {
            unsigned superstepsForGroup = 0;
            auto status = schedule_isomorphism_group(schedule,
                                                     vertex_map_for_set,
                                                     iso_groups_for_set[j],
                                                     subgraphs_for_set[j],
                                                     group_proc_allocations[j],
                                                     global_ids_by_type,
                                                     proc_type_offsets,
                                                     superstep_offset,
                                                     supersteps_for_group);
            if (status != ReturnStatus::OSP_SUCCESS) {
                return status;
            }
            numSuperstepsPerIsoGroup[j] = superstepsForGroup;
            maxSupersteps = std::max(maxSupersteps, superstepsForGroup);

            // Advance offsets for the next group
            for (size_t k = 0; k < groupProcAllocations[j].size(); ++k) {
                procTypeOffsets[k] += groupProcAllocations[j][k];
            }
        }

        for (std::size_t j = 0; j < isoGroupsForSet.size(); ++j) {
            numSuperstepsPerIsoGroup[j] = maxSupersteps - numSuperstepsPerIsoGroup[j];

            if (numSuperstepsPerIsoGroup[j] > 0) {    // This is the padding
                const auto &groupMembers = isoGroupsForSet[j];
                for (const auto &originalCompIdx : groupMembers) {
                    const auto &componentVertices = vertex_map_for_set[originalCompIdx];
                    for (const auto &vertex : component_vertices) {
                        schedule.SetAssignedSuperstep(vertex, schedule.AssignedSuperstep(vertex) + num_supersteps_per_iso_group[j]);
                    }
                }
            }
        }

        superstepsInSet = maxSupersteps;
        return ReturnStatus::OSP_SUCCESS;
    }

    ReturnStatus ScheduleIsomorphismGroup(BspSchedule<GraphT> &schedule,
                                          const std::vector<std::vector<VertexIdxT<GraphT>>> &vertexMapForSet,
                                          const std::vector<size_t> &groupMembers,
                                          const ConstrGraphT &repSubDag,
                                          const std::vector<unsigned> &procsForGroup,
                                          const std::vector<std::vector<unsigned>> &globalIdsByType,
                                          const std::vector<unsigned> &procTypeOffsets,
                                          unsigned superstepOffset,
                                          unsigned &superstepsForGroup) {
        const auto &instance = schedule.GetInstance();
        const auto &originalArch = instance.GetArchitecture();
        const size_t numMembers = groupMembers.size();
        superstepsForGroup = 0;

        bool scarcityFound = false;
        if (numMembers > 0) {
            for (unsigned typeIdx = 0; typeIdx < procsForGroup.size(); ++typeIdx) {
                if (procsForGroup[typeIdx] % numMembers != 0) {
                    scarcityFound = true;
                    break;
                }
            }
        }

        if (scarcityFound) {
            // --- SCARCITY/INDIVISIBLE CASE: Schedule sequentially on the shared processor block ---
            if constexpr (this->enableDebugPrints_) {
                std::cout << "  Group with " << numMembers << " members: Scarcity/Indivisible case. Scheduling sequentially."
                          << std::endl;
            }

            BspInstance<ConstrGraphT> subInstance(repSubDag, this->CreateSubArchitecture(originalArch, procsForGroup));
            subInstance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());
            auto &subArchitecture = subInstance.GetArchitecture();

            if constexpr (this->enableDebugPrints_) {
                std::cout << "    Sub-architecture for sequential scheduling: { ";
                for (unsigned typeIdx = 0; typeIdx < subArchitecture.GetNumberOfProcessorTypes(); ++typeIdx) {
                    std::cout << "Type " << typeIdx << ": " << subArchitecture.getProcessorTypeCount()[typeIdx] << "; ";
                }
                std::cout << "}" << std::endl;
            }

            unsigned sequentialSuperstepOffset = 0;
            for (const auto &groupMemberIdx : groupMembers) {
                BspSchedule<ConstrGraphT> subSchedule(subInstance);
                auto status = this->scheduler_->ComputeSchedule(subSchedule);
                if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::BEST_FOUND) {
                    return status;
                }

                const auto subProcTypeCount = subArchitecture.getProcessorTypeCount();
                std::vector<unsigned> subProcTypeCorrections(subArchitecture.GetNumberOfProcessorTypes(), 0);
                for (std::size_t k = 1; k < subProcTypeCorrections.size(); ++k) {
                    subProcTypeCorrections[k] = subProcTypeCorrections[k - 1] + subProcTypeCount[k - 1];
                }

                std::vector<VertexIdxT<GraphT>> sortedComponentVertices(vertexMapForSet[groupMemberIdx].begin(),
                                                                        vertex_map_for_set[groupMemberIdx].end());
                std::sort(sorted_component_vertices.begin(), sorted_component_vertices.end());

                VertexIdxT<constr_graph_t> subdagVertex = 0;
                for (const auto &vertex : sorted_component_vertices) {
                    const unsigned proc_in_sub_sched = sub_schedule.AssignedProcessor(subdag_vertex);
                    const unsigned proc_type = sub_architecture.ProcessorType(proc_in_sub_sched);
                    const unsigned local_proc_id_within_type = proc_in_sub_sched - sub_proc_type_corrections[proc_type];
                    unsigned global_proc_id
                        = global_ids_by_type[proc_type][proc_type_offsets[proc_type] + local_proc_id_within_type];

                    schedule.SetAssignedProcessor(vertex, global_proc_id);
                    schedule.SetAssignedSuperstep(
                        vertex, superstep_offset + sequential_superstep_offset + sub_schedule.AssignedSuperstep(subdag_vertex));
                    subdag_vertex++;
                }
                sequentialSuperstepOffset += subSchedule.NumberOfSupersteps();
            }
            superstepsForGroup = sequentialSuperstepOffset;

        } else {
            // --- ABUNDANCE/DIVISIBLE CASE: Replicate Schedule ---
            if constexpr (this->enableDebugPrints_) {
                std::cout << "  Group with " << numMembers << " members: Abundance/Divisible case. Replicating schedule."
                          << std::endl;
            }

            std::vector<unsigned> singleSubDagProcTypes = procsForGroup;
            if (numMembers > 0) {
                for (auto &count : singleSubDagProcTypes) {
                    count /= static_cast<unsigned>(numMembers);
                }
            }

            BspInstance<ConstrGraphT> subInstance(repSubDag, this->CreateSubArchitecture(originalArch, singleSubDagProcTypes));
            subInstance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

            if constexpr (this->enableDebugPrints_) {
                const auto &subArch = subInstance.GetArchitecture();
                std::cout << "    Sub-architecture for replication (per member): { ";
                for (unsigned typeIdx = 0; typeIdx < subArch.GetNumberOfProcessorTypes(); ++typeIdx) {
                    std::cout << "Type " << typeIdx << ": " << subArch.getProcessorTypeCount()[typeIdx] << "; ";
                }
                std::cout << "}" << std::endl;
            }

            BspSchedule<ConstrGraphT> subSchedule(subInstance);
            auto status = this->scheduler_->ComputeSchedule(subSchedule);
            if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::BEST_FOUND) {
                return status;
            }

            const auto subProcTypeCount = subSchedule.GetInstance().GetArchitecture().getProcessorTypeCount();
            std::vector<unsigned> subProcTypeCorrections(subProcTypeCount.size(), 0);
            for (std::size_t k = 1; k < subProcTypeCorrections.size(); ++k) {
                subProcTypeCorrections[k] = subProcTypeCorrections[k - 1] + subProcTypeCount[k - 1];
            }

            std::vector<unsigned> currentMemberProcOffsets = procTypeOffsets;
            for (const auto &groupMemberIdx : groupMembers) {
                std::vector<VertexIdxT<GraphT>> sortedComponentVertices(vertexMapForSet[groupMemberIdx].begin(),
                                                                        vertex_map_for_set[groupMemberIdx].end());
                std::sort(sorted_component_vertices.begin(), sorted_component_vertices.end());

                VertexIdxT<constr_graph_t> subdagVertex = 0;
                for (const auto &vertex : sorted_component_vertices) {
                    const unsigned proc_in_sub_sched = sub_schedule.AssignedProcessor(subdag_vertex);
                    const unsigned proc_type = sub_schedule.GetInstance().GetArchitecture().ProcessorType(proc_in_sub_sched);
                    const unsigned local_proc_id_within_type = proc_in_sub_sched - sub_proc_type_corrections[proc_type];
                    unsigned global_proc_id
                        = global_ids_by_type[proc_type][current_member_proc_offsets[proc_type] + local_proc_id_within_type];

                    schedule.SetAssignedProcessor(vertex, global_proc_id);
                    schedule.SetAssignedSuperstep(vertex, superstep_offset + sub_schedule.AssignedSuperstep(subdag_vertex));
                    subdag_vertex++;
                }
                for (size_t k = 0; k < subProcTypeCount.size(); ++k) {
                    currentMemberProcOffsets[k] += subProcTypeCount[k];
                }
            }
            superstepsForGroup = subSchedule.NumberOfSupersteps();
        }
        return ReturnStatus::OSP_SUCCESS;
    }
};

template <typename Graph_t>
using IsomorphicWavefrontComponentScheduler_def_int_t = IsomorphicWavefrontComponentScheduler<Graph_t, boost_graph_int_t>;

}    // namespace osp
