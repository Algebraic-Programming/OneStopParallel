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
#include <algorithm>
#include <vector>

namespace osp {

template<typename Graph_t>
void compute_lazy_communication_costs(
    const BspInstance<Graph_t> &instance,
    unsigned number_of_supersteps,
    const std::vector<unsigned> &node_to_processor_assignment,
    const std::vector<unsigned> &node_to_superstep_assignment,
    const unsigned staleness,
    std::vector<std::vector<v_commw_t<Graph_t>>> &rec,
    std::vector<std::vector<v_commw_t<Graph_t>>> &send) {
    for (const auto &node : instance.vertices()) {

        std::vector<unsigned> step_needed(instance.numberOfProcessors(), number_of_supersteps);
        for (const auto &target : instance.getComputationalDag().children(node)) {

            if (node_to_processor_assignment[node] != node_to_processor_assignment[target]) {
                step_needed[node_to_processor_assignment[target]] = std::min(step_needed[node_to_processor_assignment[target]], node_to_superstep_assignment[target]);
            }
        }

        for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {

            if (step_needed[proc] < number_of_supersteps) {
                send[node_to_processor_assignment[node]][step_needed[proc] - staleness] += instance.sendCosts(node_to_processor_assignment[node], proc) * instance.getComputationalDag().vertex_comm_weight(node);
                rec[proc][step_needed[proc] - staleness] += instance.sendCosts(node_to_processor_assignment[node], proc) * instance.getComputationalDag().vertex_comm_weight(node);
            }
        }
    }
}

template<typename Graph_t>
void compute_lazy_communication_costs(
    const BspSchedule<Graph_t> &schedule,
    std::vector<std::vector<v_commw_t<Graph_t>>> &rec,
    std::vector<std::vector<v_commw_t<Graph_t>>> &send) {
    compute_lazy_communication_costs(schedule.getInstance(), schedule.numberOfSupersteps(), schedule.assignedProcessors(), schedule.assignedSupersteps(), schedule.getStaleness(), rec, send);
}

/**
 * @struct LazyCommunicationCost
 * @brief Implements the lazy communication cost model.
 */
template<typename Graph_t>
struct LazyCommunicationCost {

    using cost_type = v_workw_t<Graph_t>;

    cost_type operator()(const BspSchedule<Graph_t> &schedule) const {
        const auto &number_of_processors = schedule.getInstance().numberOfProcessors();
        const auto &number_of_supersteps = schedule.numberOfSupersteps();

        std::vector<std::vector<v_commw_t<Graph_t>>> rec(number_of_processors, std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));
        std::vector<std::vector<v_commw_t<Graph_t>>> send(number_of_processors, std::vector<v_commw_t<Graph_t>>(number_of_supersteps, 0));

        compute_lazy_communication_costs(schedule, rec, send);
        const auto max_comm_per_step = cost_helpers::compute_max_comm_per_step(schedule, rec, send);

        v_commw_t<Graph_t> comm_costs = 0;
        for (unsigned step = 0; step < number_of_supersteps; step++) {
            const auto step_comm_cost = max_comm_per_step[step];
            comm_costs += step_comm_cost;

            if (step_comm_cost > 0) {
                comm_costs += schedule.getInstance().synchronisationCosts();
            }
        }

        return comm_costs + cost_helpers::compute_work_costs(schedule);
    }
};

} // namespace osp
