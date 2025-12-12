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
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>

#include "DagDivider.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/computational_dag_util.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

namespace osp {

/**
 * @class AbstractWavefrontScheduler
 * @brief Base class for schedulers that operate on wavefronts of a DAG.
 */
template <typename GraphT, typename ConstrGraphT>
class AbstractWavefrontScheduler : public Scheduler<GraphT> {
  protected:
    IDagDivider<GraphT> *divider_;
    Scheduler<ConstrGraphT> *scheduler_;
    static constexpr bool enableDebugPrints_ = true;

    /**
     * @brief Distributes processors proportionally, ensuring active components get at least one if possible.
     * @param allocation A reference to the vector that will be filled with the processor allocation.
     * @return True if the scarcity case was hit (fewer processors than active components), false otherwise.
     */
    bool DistributeProcessors(unsigned totalProcessorsOfType,
                              const std::vector<double> &workWeights,
                              std::vector<unsigned> &allocation) const {
        allocation.assign(workWeights.size(), 0);
        double totalWork = std::accumulate(workWeights.begin(), workWeights.end(), 0.0);
        if (totalWork <= 1e-9 || totalProcessorsOfType == 0) {
            return false;
        }

        std::vector<size_t> activeIndices;
        for (size_t i = 0; i < workWeights.size(); ++i) {
            if (workWeights[i] > 1e-9) {
                activeIndices.push_back(i);
            }
        }

        if (activeIndices.empty()) {
            return false;
        }

        size_t numActiveComponents = activeIndices.size();
        unsigned remainingProcs = totalProcessorsOfType;

        // --- Stage 1: Guarantee at least one processor if possible (anti-starvation) ---
        if (totalProcessorsOfType >= numActiveComponents) {
            // Abundance case: Give one processor to each active component first.
            for (size_t idx : activeIndices) {
                allocation[idx] = 1;
            }
            remainingProcs -= static_cast<unsigned>(numActiveComponents);
        } else {
            // Scarcity case: Not enough processors for each active component.
            std::vector<std::pair<double, size_t>> sortedWork;
            for (size_t idx : activeIndices) {
                sortedWork.push_back({workWeights[idx], idx});
            }
            std::sort(sortedWork.rbegin(), sortedWork.rend());
            for (unsigned i = 0; i < remainingProcs; ++i) {
                allocation[sortedWork[i].second]++;
            }
            return true;    // Scarcity case was hit.
        }

        // --- Stage 2: Proportional Distribution of Remaining Processors ---
        if (remainingProcs > 0) {
            std::vector<double> adjustedWorkWeights;
            double adjustedTotalWork = 0;

            double workPerProc = totalWork / static_cast<double>(totalProcessorsOfType);

            for (size_t idx : activeIndices) {
                double adjustedWork = std::max(0.0, workWeights[idx] - workPerProc);
                adjustedWorkWeights.push_back(adjustedWork);
                adjustedTotalWork += adjustedWork;
            }

            if (adjustedTotalWork > 1e-9) {
                std::vector<std::pair<double, size_t>> remainders;
                unsigned allocatedCount = 0;

                for (size_t i = 0; i < activeIndices.size(); ++i) {
                    double exactShare = (adjustedWorkWeights[i] / adjustedTotalWork) * remainingProcs;
                    unsigned additionalAlloc = static_cast<unsigned>(std::floor(exactShare));
                    allocation[activeIndices[i]] += additionalAlloc;    // Add to the base allocation of 1
                    remainders.push_back({exactShare - additionalAlloc, activeIndices[i]});
                    allocatedCount += additionalAlloc;
                }

                std::sort(remainders.rbegin(), remainders.rend());

                unsigned remainderProcessors = remainingProcs - allocatedCount;
                for (unsigned i = 0; i < remainderProcessors; ++i) {
                    if (i < remainders.size()) {
                        allocation[remainders[i].second]++;
                    }
                }
            }
        }
        return false;    // Scarcity case was not hit.
    }

    BspArchitecture<ConstrGraphT> CreateSubArchitecture(const BspArchitecture<GraphT> &originalArch,
                                                        const std::vector<unsigned> &subDagProcTypes) const {
        // The calculation is now inside the assert, so it only happens in debug builds.
        assert(std::accumulate(subDagProcTypes.begin(), subDagProcTypes.end(), 0u) > 0
               && "Attempted to create a sub-architecture with zero processors.");

        BspArchitecture<ConstrGraphT> subArchitecture(originalArch);
        std::vector<VMemwT<Graph_t>> subDagProcessorMemory(original_arch.getProcessorTypeCount().size(),
                                                           std::numeric_limits<VMemwT<Graph_t>>::max());
        for (unsigned i = 0; i < originalArch.NumberOfProcessors(); ++i) {
            subDagProcessorMemory[originalArch.processorType(i)]
                = std::min(originalArch.memoryBound(i), sub_dag_processor_memory[originalArch.processorType(i)]);
        }
        subArchitecture.SetProcessorsConsequTypes(subDagProcTypes, sub_dag_processor_memory);
        return subArchitecture;
    }

    bool ValidateWorkDistribution(const std::vector<ConstrGraphT> &subDags, const BspInstance<GraphT> &instance) const {
        const auto &originalArch = instance.GetArchitecture();
        for (const auto &repSubDag : subDags) {
            const double totalRepWork = sumOfVerticesWorkWeights(repSubDag);

            double sumOfCompatibleWorksForRep = 0.0;
            for (unsigned typeIdx = 0; typeIdx < originalArch.getNumberOfProcessorTypes(); ++typeIdx) {
                sumOfCompatibleWorksForRep += sumOfCompatibleWorkWeights(repSubDag, instance, typeIdx);
            }

            if (sumOfCompatibleWorksForRep > totalRepWork + 1e-9) {
                if constexpr (enableDebugPrints_) {
                    std::cerr << "ERROR: Sum of compatible work (" << sumOfCompatibleWorksForRep << ") exceeds total work ("
                              << totalRepWork << ") for a sub-dag. Aborting." << std::endl;
                }
                return false;
            }
        }
        return true;
    }

  public:
    AbstractWavefrontScheduler(IDagDivider<GraphT> &div, Scheduler<ConstrGraphT> &sched) : divider_(&div), scheduler_(&sched) {}
};

}    // namespace osp
