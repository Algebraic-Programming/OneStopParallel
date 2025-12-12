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
template <typename GraphT, typename ConstrGraphT>
class WavefrontComponentScheduler : public AbstractWavefrontScheduler<GraphT, ConstrGraphT> {
  public:
    WavefrontComponentScheduler(IDagDivider<GraphT> &div, Scheduler<ConstrGraphT> &scheduler)
        : AbstractWavefrontScheduler<GraphT, ConstrGraphT>(div, scheduler) {}

    std::string getScheduleName() const override { return "WavefrontComponentScheduler"; }

    ReturnStatus computeSchedule(BspSchedule<GraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();
        const auto &originalArch = instance.GetArchitecture();
        const auto &originalProcTypeCount = originalArch.getProcessorTypeCount();
        const auto &computationalDag = instance.GetComputationalDag();

        std::vector<std::vector<unsigned>> globalIdsByType(originalArch.getNumberOfProcessorTypes());
        for (unsigned i = 0; i < originalArch.NumberOfProcessors(); ++i) {
            globalIdsByType[originalArch.processorType(i)].push_back(i);
        }

        auto vertexMaps = this->divider_->divide(computationalDag);
        unsigned superstepOffset = 0;

        for (std::size_t i = 0; i < vertexMaps.size(); ++i) {    // For each wavefront set
            if (this->enableDebugPrints_) {
                std::cout << "\n--- Processing Wavefront Set " << i << " (No Isomorphism) ---" << std::endl;
            }

            const auto &components = vertexMaps[i];
            std::vector<ConstrGraphT> subDags(components.size());
            std::vector<std::vector<double>> workByType(components.size(), std::vector<double>(originalProcTypeCount.size(), 0.0));

            for (size_t j = 0; j < components.size(); ++j) {
                create_induced_subgraph(computationalDag, subDags[j], components[j]);
                for (unsigned typeIdx = 0; typeIdx < originalProcTypeCount.size(); ++typeIdx) {
                    workByType[j][typeIdx] = sumOfCompatibleWorkWeights(subDags[j], instance, typeIdx);
                }
            }

            assert(this->ValidateWorkDistribution(subDags, instance));

            // Distribute Processors
            std::vector<std::vector<unsigned>> procAllocations(components.size(),
                                                               std::vector<unsigned>(originalProcTypeCount.size()));
            for (unsigned typeIdx = 0; typeIdx < originalProcTypeCount.size(); ++typeIdx) {
                std::vector<double> workForThisType(components.size());
                for (size_t compIdx = 0; compIdx < components.size(); ++compIdx) {
                    workForThisType[compIdx] = workByType[compIdx][typeIdx];
                }

                std::vector<unsigned> typeAllocation;
                bool starvationHit = this->DistributeProcessors(originalProcTypeCount[typeIdx], workForThisType, typeAllocation);

                if (starvationHit) {
                    if constexpr (this->enableDebugPrints_) {
                        std::cerr << "ERROR: Processor starvation detected for type " << typeIdx << " in wavefront set " << i
                                  << ". Not enough processors to assign one to each active component." << std::endl;
                    }
                    return ReturnStatus::ERROR;
                }

                for (size_t compIdx = 0; compIdx < components.size(); ++compIdx) {
                    procAllocations[compIdx][typeIdx] = typeAllocation[compIdx];
                }
            }

            unsigned maxNumberSupersteps = 0;
            std::vector<unsigned> procTypeOffsets(originalArch.getNumberOfProcessorTypes(), 0);

            for (std::size_t j = 0; j < components.size(); ++j) {
                BspArchitecture<ConstrGraphT> subArchitecture = this->CreateSubArchitecture(originalArch, procAllocations[j]);
                if constexpr (this->enableDebugPrints_) {
                    std::cout << "  Component " << j << " sub-architecture: { ";
                    for (unsigned typeIdx = 0; typeIdx < subArchitecture.getNumberOfProcessorTypes(); ++typeIdx) {
                        std::cout << "Type " << typeIdx << ": " << subArchitecture.getProcessorTypeCount()[typeIdx] << "; ";
                    }
                    std::cout << "}" << std::endl;
                }

                BspInstance<ConstrGraphT> subInstance(subDags[j], subArchitecture);
                subInstance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

                BspSchedule<ConstrGraphT> subSchedule(subInstance);
                const auto status = this->scheduler_->computeSchedule(subSchedule);
                if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::BEST_FOUND) {
                    return status;
                }

                const auto subProcTypeCount = subArchitecture.getProcessorTypeCount();
                std::vector<unsigned> subProcTypeCorrections(subArchitecture.getNumberOfProcessorTypes(), 0);
                for (std::size_t k = 1; k < subProcTypeCorrections.size(); ++k) {
                    subProcTypeCorrections[k] = subProcTypeCorrections[k - 1] + subProcTypeCount[k - 1];
                }

                VertexIdxT<constr_graph_t> subdagVertex = 0;
                std::vector<VertexIdxT<GraphT>> sortedComponentVertices(components[j].begin(), components[j].end());
                std::sort(sorted_component_vertices.begin(), sorted_component_vertices.end());

                for (const auto &vertex : sorted_component_vertices) {
                    const unsigned proc_in_sub_sched = sub_schedule.assignedProcessor(subdag_vertex);
                    const unsigned proc_type = sub_architecture.processorType(proc_in_sub_sched);
                    const unsigned local_proc_id_within_type = proc_in_sub_sched - sub_proc_type_corrections[proc_type];
                    unsigned global_proc_id
                        = global_ids_by_type[proc_type][proc_type_offsets[proc_type] + local_proc_id_within_type];

                    schedule.setAssignedProcessor(vertex, global_proc_id);
                    schedule.setAssignedSuperstep(vertex, superstep_offset + sub_schedule.AssignedSuperstep(subdag_vertex));
                    subdag_vertex++;
                }

                for (size_t k = 0; k < subProcTypeCount.size(); ++k) {
                    procTypeOffsets[k] += subProcTypeCount[k];
                }
                maxNumberSupersteps = std::max(maxNumberSupersteps, subSchedule.NumberOfSupersteps());
            }
            superstepOffset += maxNumberSupersteps;
        }
        return ReturnStatus::OSP_SUCCESS;
    }
};

template <typename Graph_t>
using WavefrontComponentScheduler_def_int_t = WavefrontComponentScheduler<Graph_t, boost_graph_int_t>;

}    // namespace osp
