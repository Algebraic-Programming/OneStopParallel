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

#include "osp/bsp/model/cost/CostModelHelpers.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

/**
 * @struct TotalCommunicationCost
 * @brief Implements the total communication cost model.
 */
template<typename Graph_t>
struct TotalCommunicationCost {

    using cost_type = double;

    cost_type operator()(const BspSchedule<Graph_t> &schedule) const {

        const auto &instance = schedule.getInstance();
        const auto &node_to_processor_assignment = schedule.assignedProcessors();

        v_commw_t<Graph_t> total_communication = 0;

        for (const auto &v : instance.vertices()) {
            for (const auto &target : instance.getComputationalDag().children(v)) {

                if (node_to_processor_assignment[v] != node_to_processor_assignment[target]) {
                    total_communication += instance.sendCosts(node_to_processor_assignment[v], node_to_processor_assignment[target]) * instance.getComputationalDag().vertex_comm_weight(v);
                }
            }
        }

        auto comm_cost = total_communication * static_cast<double>(instance.communicationCosts()) / static_cast<double>(instance.numberOfProcessors());

        const unsigned number_of_supersteps = schedule.numberOfSupersteps();

        auto work_cost = cost_helpers::compute_work_costs(schedule);
        auto sync_cost = static_cast<v_commw_t<Graph_t>>(number_of_supersteps > 1 ? number_of_supersteps - 1 : 0) * instance.synchronisationCosts();

        return comm_cost + work_cost + sync_cost;
    }
};

} // namespace osp
