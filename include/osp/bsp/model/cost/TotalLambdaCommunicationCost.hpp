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

#include <unordered_set>

#include "osp/bsp/model/cost/CostModelHelpers.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

/**
 * @struct TotalLambdaCommunicationCost
 * @brief Implements the total lambda communication cost model.
 */
template <typename GraphT>
struct TotalLambdaCommunicationCost {
    using CostType = double;

    CostType operator()(const BspSchedule<GraphT> &schedule) const {
        const auto &instance = schedule.GetInstance();
        const auto &nodeToProcessorAssignment = schedule.AssignedProcessors();

        VCommwT<GraphT> commCosts = 0;
        const double commMultiplier = 1.0 / instance.NumberOfProcessors();

        for (const auto &v : instance.Vertices()) {
            if (instance.GetComputationalDag().OutDegree(v) == 0) {
                continue;
            }

            std::unordered_set<unsigned> targetProcs;
            for (const auto &target : instance.GetComputationalDag().Children(v)) {
                targetProcs.insert(nodeToProcessorAssignment[target]);
            }

            const unsigned sourceProc = nodeToProcessorAssignment[v];
            const auto vCommCost = instance.GetComputationalDag().VertexCommWeight(v);

            for (const auto &targetProc : targetProcs) {
                commCosts += vCommCost * instance.SendCosts(sourceProc, targetProc);
            }
        }

        const unsigned numberOfSupersteps = schedule.NumberOfSupersteps();

        auto commCost = commCosts * commMultiplier * static_cast<double>(instance.CommunicationCosts());
        auto workCost = cost_helpers::ComputeWorkCosts(schedule);
        auto syncCost
            = static_cast<VCommwT<GraphT>>(numberOfSupersteps > 1 ? numberOfSupersteps - 1 : 0) * instance.SynchronisationCosts();

        return commCost + static_cast<double>(workCost) + static_cast<double>(syncCost);
    }
};

}    // namespace osp
