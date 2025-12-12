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
    using cost_type = v_commw_t<Graph_t>;

    cost_type operator()(const BspSchedule<GraphT> &schedule) const {
        const auto &instance = schedule.getInstance();
        unsigned numberOfSupersteps = schedule.numberOfSupersteps();
        const auto &nodeToProcessorAssignment = schedule.assignedProcessors();
        const auto &nodeToSuperstepAssignment = schedule.assignedSupersteps();
        const auto staleness = schedule.getStaleness();

        std::vector<std::vector<v_commw_t<Graph_t>>> rec(instance.numberOfProcessors(),
                                                         std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));
        std::vector<std::vector<v_commw_t<Graph_t>>> send(instance.numberOfProcessors(),
                                                          std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));

        for (vertex_idx_t<Graph_t> node = 0; node < instance.numberOfVertices(); node++) {
            std::vector<unsigned> stepNeeded(instance.numberOfProcessors(), numberOfSupersteps);
            for (const auto &target : instance.getComputationalDag().children(node)) {
                if (node_to_processor_assignment[node] != node_to_processor_assignment[target]) {
                    step_needed[node_to_processor_assignment[target]]
                        = std::min(step_needed[node_to_processor_assignment[target]], node_to_superstep_assignment[target]);
                }
            }

            for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
                if (stepNeeded[proc] < numberOfSupersteps) {
                    send[nodeToProcessorAssignment[node]][nodeToSuperstepAssignment[node]]
                        += instance.sendCosts(nodeToProcessorAssignment[node], proc)
                           * instance.getComputationalDag().vertex_comm_weight(node);

                    if (stepNeeded[proc] >= staleness) {
                        rec[proc][stepNeeded[proc] - staleness] += instance.sendCosts(nodeToProcessorAssignment[node], proc)
                                                                   * instance.getComputationalDag().vertex_comm_weight(node);
                    }
                }
            }
        }

        const auto maxCommPerStep = cost_helpers::compute_max_comm_per_step(schedule, rec, send);
        v_commw_t<Graph_t> commCosts = 0;
        for (unsigned step = 0; step < numberOfSupersteps; step++) {
            const auto stepCommCost = max_comm_per_step[step];
            commCosts += step_comm_cost;

            if (stepCommCost > 0) {
                commCosts += instance.synchronisationCosts();
            }
        }

        return comm_costs + cost_helpers::compute_work_costs(schedule);
    }
};

}    // namespace osp
