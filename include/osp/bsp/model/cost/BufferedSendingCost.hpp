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
#include <vector>

#include "osp/bsp/model/cost/CostModelHelpers.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

/**
 * @struct BufferedSendingCost
 * @brief Implements the buffered sending cost model.
 */
template <typename GraphT>
struct BufferedSendingCost {
    using CostType = VCommwT<GraphT>;

    CostType operator()(const BspSchedule<GraphT> &schedule) const {
        const auto &instance = schedule.GetInstance();
        unsigned numberOfSupersteps = schedule.NumberOfSupersteps();
        const auto &nodeToProcessorAssignment = schedule.AssignedProcessors();
        const auto &nodeToSuperstepAssignment = schedule.AssignedSupersteps();
        const auto staleness = schedule.Staleness();

        std::vector<std::vector<VCommwT<GraphT>>> rec(instance.NumberOfProcessors(),
                                                      std::vector<VCommwT<GraphT>>(numberOfSupersteps, 0));
        std::vector<std::vector<VCommwT<GraphT>>> send(instance.NumberOfProcessors(),
                                                       std::vector<VCommwT<GraphT>>(numberOfSupersteps, 0));

        for (VertexIdxT<GraphT> node = 0; node < instance.NumberOfVertices(); node++) {
            std::vector<unsigned> stepNeeded(instance.NumberOfProcessors(), numberOfSupersteps);
            for (const auto &target : instance.GetComputationalDag().Children(node)) {
                if (nodeToProcessorAssignment[node] != nodeToProcessorAssignment[target]) {
                    stepNeeded[nodeToProcessorAssignment[target]]
                        = std::min(stepNeeded[nodeToProcessorAssignment[target]], nodeToSuperstepAssignment[target]);
                }
            }

            for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
                if (stepNeeded[proc] < numberOfSupersteps) {
                    send[nodeToProcessorAssignment[node]][nodeToSuperstepAssignment[node]]
                        += instance.SendCosts(nodeToProcessorAssignment[node], proc)
                           * instance.GetComputationalDag().VertexCommWeight(node);

                    if (stepNeeded[proc] >= staleness) {
                        rec[proc][stepNeeded[proc] - staleness] += instance.SendCosts(nodeToProcessorAssignment[node], proc)
                                                                   * instance.GetComputationalDag().VertexCommWeight(node);
                    }
                }
            }
        }

        const auto maxCommPerStep = cost_helpers::ComputeMaxCommPerStep(schedule, rec, send);
        VCommwT<GraphT> commCosts = 0;
        for (unsigned step = 0; step < numberOfSupersteps; step++) {
            const auto stepCommCost = maxCommPerStep[step];
            commCosts += stepCommCost;

            if (stepCommCost > 0) {
                commCosts += instance.SynchronisationCosts();
            }
        }

        return commCosts + cost_helpers::ComputeWorkCosts(schedule);
    }
};

}    // namespace osp
