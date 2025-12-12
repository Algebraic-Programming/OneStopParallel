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

template <typename GraphT>
void ComputeLazyCommunicationCosts(const BspInstance<GraphT> &instance,
                                   unsigned numberOfSupersteps,
                                   const std::vector<unsigned> &nodeToProcessorAssignment,
                                   const std::vector<unsigned> &nodeToSuperstepAssignment,
                                   const unsigned staleness,
                                   std::vector<std::vector<VCommwT<GraphT>>> &rec,
                                   std::vector<std::vector<VCommwT<GraphT>>> &send) {
    for (const auto &node : instance.Vertices()) {
        std::vector<unsigned> stepNeeded(instance.NumberOfProcessors(), numberOfSupersteps);
        for (const auto &target : instance.GetComputationalDag().Children(node)) {
            if (nodeToProcessorAssignment[node] != nodeToProcessorAssignment[target]) {
                stepNeeded[nodeToProcessorAssignment[target]]
                    = std::min(stepNeeded[nodeToProcessorAssignment[target]], nodeToSuperstepAssignment[target]);
            }
        }

        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); proc++) {
            if (stepNeeded[proc] < numberOfSupersteps) {
                send[nodeToProcessorAssignment[node]][stepNeeded[proc] - staleness]
                    += instance.SendCosts(nodeToProcessorAssignment[node], proc)
                       * instance.GetComputationalDag().VertexCommWeight(node);
                rec[proc][stepNeeded[proc] - staleness] += instance.SendCosts(nodeToProcessorAssignment[node], proc)
                                                           * instance.GetComputationalDag().VertexCommWeight(node);
            }
        }
    }
}

template <typename GraphT>
void ComputeLazyCommunicationCosts(const BspSchedule<GraphT> &schedule,
                                   std::vector<std::vector<VCommwT<GraphT>>> &rec,
                                   std::vector<std::vector<VCommwT<GraphT>>> &send) {
    ComputeLazyCommunicationCosts(schedule.getInstance(),
                                  schedule.NumberOfSupersteps(),
                                  schedule.AssignedProcessors(),
                                  schedule.AssignedSupersteps(),
                                  schedule.getStaleness(),
                                  rec,
                                  send);
}

/**
 * @struct LazyCommunicationCost
 * @brief Implements the lazy communication cost model.
 */
template <typename GraphT>
struct LazyCommunicationCost {
    using CostType = VWorkwT<GraphT>;

    CostType operator()(const BspSchedule<GraphT> &schedule) const {
        const auto &numberOfProcessors = schedule.GetInstance().NumberOfProcessors();
        const auto &numberOfSupersteps = schedule.NumberOfSupersteps();

        std::vector<std::vector<VCommwT<GraphT>>> rec(numberOfProcessors, std::vector<VCommwT<GraphT>>(numberOfSupersteps, 0));
        std::vector<std::vector<VCommwT<GraphT>>> send(numberOfProcessors, std::vector<VCommwT<GraphT>>(numberOfSupersteps, 0));

        ComputeLazyCommunicationCosts(schedule, rec, send);
        const auto maxCommPerStep = cost_helpers::ComputeMaxCommPerStep(schedule, rec, send);

        VCommwT<GraphT> commCosts = 0;
        for (unsigned step = 0; step < numberOfSupersteps; step++) {
            const auto stepCommCost = maxCommPerStep[step];
            commCosts += stepCommCost;

            if (stepCommCost > 0) {
                commCosts += schedule.GetInstance().SynchronisationCosts();
            }
        }

        return commCosts + cost_helpers::ComputeWorkCosts(schedule);
    }
};

}    // namespace osp
