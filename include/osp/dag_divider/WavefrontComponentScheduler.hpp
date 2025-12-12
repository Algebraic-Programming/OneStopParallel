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

    std::string GetScheduleName() const override { return "WavefrontComponentScheduler"; }

    RETURN_STATUS ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();
        const auto &originalArch = instance.GetArchitecture();
        const auto &originalProcTypeCount = originalArch.getProcessorTypeCount();
        const auto &computationalDag = instance.GetComputationalDag();

        std::vector<std::vector<unsigned>> globalIdsByType(originalArch.getNumberOfProcessorTypes());
        for (unsigned i = 0; i < originalArch.NumberOfProcessors(); ++i) {
            globalIdsByType[originalArch.processorType(i)].push_back(i);
        }

        auto vertexMaps = this->divider->divide(computationalDag);
        unsigned superstepOffset = 0;

        for (std::size_t i = 0; i < vertexMaps.size(); ++i) {    // For each wavefront set
            if (this->enable_debug_prints) {
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

            assert(this->validateWorkDistribution(subDags, instance));

            // Distribute Processors
            std::vector<std::vector<unsigned>> procAllocations(components.size(),
                                                               std::vector<unsigned>(originalProcTypeCount.size()));
            for (unsigned typeIdx = 0; typeIdx < originalProcTypeCount.size(); ++typeIdx) {
                std::vector<double> workForThisType(components.size());
                for (size_t compIdx = 0; compIdx < components.size(); ++compIdx) {
                    workForThisType[compIdx] = workByType[compIdx][typeIdx];
                }

                std::vector<unsigned> typeAllocation;
                bool starvationHit = this->distributeProcessors(originalProcTypeCount[typeIdx], workForThisType, typeAllocation);

                if (starvationHit) {
                    if constexpr (this->enable_debug_prints) {
                        std::cerr << "ERROR: Processor starvation detected for type " << typeIdx << " in wavefront set " << i
                                  << ". Not enough processors to assign one to each active component." << std::endl;
                    }
                    return RETURN_STATUS::ERROR;
                }

                for (size_t compIdx = 0; compIdx < components.size(); ++compIdx) {
                    procAllocations[compIdx][typeIdx] = typeAllocation[compIdx];
                }
            }

            unsigned maxNumberSupersteps = 0;
            std::vector<unsigned> procTypeOffsets(originalArch.getNumberOfProcessorTypes(), 0);

            for (std::size_t j = 0; j < components.size(); ++j) {
                BspArchitecture<ConstrGraphT> subArchitecture = this->createSubArchitecture(originalArch, procAllocations[j]);
                if constexpr (this->enable_debug_prints) {
                    std::cout << "  Component " << j << " sub-architecture: { ";
                    for (unsigned typeIdx = 0; typeIdx < subArchitecture.getNumberOfProcessorTypes(); ++typeIdx) {
                        std::cout << "Type " << typeIdx << ": " << subArchitecture.getProcessorTypeCount()[typeIdx] << "; ";
                    }
                    std::cout << "}" << std::endl;
                }

                BspInstance<ConstrGraphT> subInstance(subDags[j], subArchitecture);
                subInstance.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

                BspSchedule<ConstrGraphT> subSchedule(subInstance);
                const auto status = this->scheduler->computeSchedule(subSchedule);
                if (status != RETURN_STATUS::OSP_SUCCESS && status != RETURN_STATUS::BEST_FOUND) {
                    return status;
                }

                const auto subProcTypeCount = subArchitecture.getProcessorTypeCount();
                std::vector<unsigned> subProcTypeCorrections(subArchitecture.getNumberOfProcessorTypes(), 0);
                for (std::size_t k = 1; k < subProcTypeCorrections.size(); ++k) {
                    subProcTypeCorrections[k] = subProcTypeCorrections[k - 1] + subProcTypeCount[k - 1];
                }

                VertexIdxT<ConstrGraphT> subdagVertex = 0;
                std::vector<VertexIdxT<GraphT>> sortedComponentVertices(components[j].begin(), components[j].end());
                std::sort(sortedComponentVertices.begin(), sortedComponentVertices.end());

                for (const auto &vertex : sortedComponentVertices) {
                    const unsigned procInSubSched = subSchedule.assignedProcessor(subdagVertex);
                    const unsigned procType = subArchitecture.processorType(procInSubSched);
                    const unsigned localProcIdWithinType = procInSubSched - subProcTypeCorrections[procType];
                    unsigned globalProcId = globalIdsByType[procType][procTypeOffsets[procType] + localProcIdWithinType];

                    schedule.setAssignedProcessor(vertex, globalProcId);
                    schedule.setAssignedSuperstep(vertex, superstepOffset + subSchedule.assignedSuperstep(subdagVertex));
                    subdagVertex++;
                }

                for (size_t k = 0; k < subProcTypeCount.size(); ++k) {
                    procTypeOffsets[k] += subProcTypeCount[k];
                }
                maxNumberSupersteps = std::max(maxNumberSupersteps, subSchedule.NumberOfSupersteps());
            }
            superstepOffset += maxNumberSupersteps;
        }
        return RETURN_STATUS::OSP_SUCCESS;
    }
};

template <typename GraphT>
using WavefrontComponentSchedulerDefIntT = WavefrontComponentScheduler<GraphT, BoostGraphIntT>;

}    // namespace osp
