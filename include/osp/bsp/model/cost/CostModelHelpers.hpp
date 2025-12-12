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

#include "osp/bsp/model/BspInstance.hpp"

namespace osp {

template <typename GraphT>
class BspSchedule;

namespace cost_helpers {

template <typename GraphT>
std::vector<VCommwT<GraphT>> ComputeMaxCommPerStep(const BspInstance<GraphT> &instance,
                                                   unsigned numberOfSupersteps,
                                                   const std::vector<std::vector<VCommwT<GraphT>>> &rec,
                                                   const std::vector<std::vector<VCommwT<GraphT>>> &send) {
    std::vector<VCommwT<GraphT>> maxCommPerStep(numberOfSupersteps, 0);
    for (unsigned step = 0; step < numberOfSupersteps; step++) {
        VCommwT<GraphT> maxSend = 0;
        VCommwT<GraphT> maxRec = 0;

        for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
            if (maxSend < send[proc][step]) {
                maxSend = send[proc][step];
            }
            if (maxRec < rec[proc][step]) {
                maxRec = rec[proc][step];
            }
        }
        maxCommPerStep[step] = std::max(maxSend, maxRec) * instance.communicationCosts();
    }
    return maxCommPerStep;
}

template <typename GraphT>
std::vector<VCommwT<GraphT>> ComputeMaxCommPerStep(const BspSchedule<GraphT> &schedule,
                                                   const std::vector<std::vector<VCommwT<GraphT>>> &rec,
                                                   const std::vector<std::vector<VCommwT<GraphT>>> &send) {
    return compute_max_comm_per_step(schedule.getInstance(), schedule.numberOfSupersteps(), rec, send);
}

template <typename GraphT>
std::vector<VWorkwT<GraphT>> ComputeMaxWorkPerStep(const BspInstance<GraphT> &instance,
                                                   unsigned numberOfSupersteps,
                                                   const std::vector<unsigned> &nodeToProcessorAssignment,
                                                   const std::vector<unsigned> &nodeToSuperstepAssignment) {
    std::vector<std::vector<VWorkwT<GraphT>>> work = std::vector<std::vector<VWorkwT<GraphT>>>(
        numberOfSupersteps, std::vector<VWorkwT<GraphT>>(instance.numberOfProcessors(), 0));
    for (const auto &node : instance.vertices()) {
        work[nodeToSuperstepAssignment[node]][nodeToProcessorAssignment[node]]
            += instance.getComputationalDag().vertex_work_weight(node);
    }

    std::vector<VWorkwT<GraphT>> maxWorkPerStep(numberOfSupersteps, 0);
    for (unsigned step = 0; step < numberOfSupersteps; step++) {
        VWorkwT<GraphT> maxWork = 0;
        for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
            if (maxWork < work[step][proc]) {
                maxWork = work[step][proc];
            }
        }

        maxWorkPerStep[step] = maxWork;
    }

    return maxWorkPerStep;
}

template <typename GraphT>
std::vector<VWorkwT<GraphT>> ComputeMaxWorkPerStep(const BspSchedule<GraphT> &schedule) {
    return compute_max_work_per_step(
        schedule.getInstance(), schedule.numberOfSupersteps(), schedule.assignedProcessors(), schedule.assignedSupersteps());
}

template <typename GraphT>
VWorkwT<GraphT> ComputeWorkCosts(const BspInstance<GraphT> &instance,
                                 unsigned numberOfSupersteps,
                                 const std::vector<unsigned> &nodeToProcessorAssignment,
                                 const std::vector<unsigned> &nodeToSuperstepAssignment) {
    std::vector<VWorkwT<GraphT>> maxWorkPerStep
        = compute_max_work_per_step(instance, numberOfSupersteps, nodeToProcessorAssignment, nodeToSuperstepAssignment);

    return std::accumulate(maxWorkPerStep.begin(), maxWorkPerStep.end(), static_cast<VWorkwT<GraphT>>(0));
}

template <typename GraphT>
VWorkwT<GraphT> ComputeWorkCosts(const BspSchedule<GraphT> &schedule) {
    return compute_work_costs(
        schedule.getInstance(), schedule.numberOfSupersteps(), schedule.assignedProcessors(), schedule.assignedSupersteps());
}

}    // namespace cost_helpers
}    // namespace osp
