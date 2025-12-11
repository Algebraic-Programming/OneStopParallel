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
template <typename Graph_t>
struct TotalLambdaCommunicationCost {
    using cost_type = double;

    cost_type operator()(const BspSchedule<Graph_t> &schedule) const {
        const auto &instance = schedule.getInstance();
        const auto &node_to_processor_assignment = schedule.assignedProcessors();

        v_commw_t<Graph_t> comm_costs = 0;
        const double comm_multiplier = 1.0 / instance.numberOfProcessors();

        for (const auto &v : instance.vertices()) {
            if (instance.getComputationalDag().out_degree(v) == 0) {
                continue;
            }

            std::unordered_set<unsigned> target_procs;
            for (const auto &target : instance.getComputationalDag().children(v)) {
                target_procs.insert(node_to_processor_assignment[target]);
            }

            const unsigned source_proc = node_to_processor_assignment[v];
            const auto v_comm_cost = instance.getComputationalDag().vertex_comm_weight(v);

            for (const auto &target_proc : target_procs) {
                comm_costs += v_comm_cost * instance.sendCosts(source_proc, target_proc);
            }
        }

        const unsigned number_of_supersteps = schedule.numberOfSupersteps();

        auto comm_cost = comm_costs * comm_multiplier * static_cast<double>(instance.communicationCosts());
        auto work_cost = cost_helpers::compute_work_costs(schedule);
        auto sync_cost = static_cast<v_commw_t<Graph_t>>(number_of_supersteps > 1 ? number_of_supersteps - 1 : 0)
                         * instance.synchronisationCosts();

        return comm_cost + static_cast<double>(work_cost) + static_cast<double>(sync_cost);
    }
};

}    // namespace osp
