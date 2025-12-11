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

template <typename Graph_t>
class BspSchedule;

namespace cost_helpers {

template <typename Graph_t>
std::vector<v_commw_t<Graph_t>> compute_max_comm_per_step(const BspInstance<Graph_t> &instance,
                                                          unsigned number_of_supersteps,
                                                          const std::vector<std::vector<v_commw_t<Graph_t>>> &rec,
                                                          const std::vector<std::vector<v_commw_t<Graph_t>>> &send) {
    std::vector<v_commw_t<Graph_t>> max_comm_per_step(number_of_supersteps, 0);
    for (unsigned step = 0; step < number_of_supersteps; step++) {
        v_commw_t<Graph_t> max_send = 0;
        v_commw_t<Graph_t> max_rec = 0;

        for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
            if (max_send < send[proc][step]) { max_send = send[proc][step]; }
            if (max_rec < rec[proc][step]) { max_rec = rec[proc][step]; }
        }
        max_comm_per_step[step] = std::max(max_send, max_rec) * instance.communicationCosts();
    }
    return max_comm_per_step;
}

template <typename Graph_t>
std::vector<v_commw_t<Graph_t>> compute_max_comm_per_step(const BspSchedule<Graph_t> &schedule,
                                                          const std::vector<std::vector<v_commw_t<Graph_t>>> &rec,
                                                          const std::vector<std::vector<v_commw_t<Graph_t>>> &send) {
    return compute_max_comm_per_step(schedule.getInstance(), schedule.numberOfSupersteps(), rec, send);
}

template <typename Graph_t>
std::vector<v_workw_t<Graph_t>> compute_max_work_per_step(const BspInstance<Graph_t> &instance,
                                                          unsigned number_of_supersteps,
                                                          const std::vector<unsigned> &node_to_processor_assignment,
                                                          const std::vector<unsigned> &node_to_superstep_assignment) {
    std::vector<std::vector<v_workw_t<Graph_t>>> work = std::vector<std::vector<v_workw_t<Graph_t>>>(
        number_of_supersteps, std::vector<v_workw_t<Graph_t>>(instance.numberOfProcessors(), 0));
    for (const auto &node : instance.vertices()) {
        work[node_to_superstep_assignment[node]][node_to_processor_assignment[node]]
            += instance.getComputationalDag().vertex_work_weight(node);
    }

    std::vector<v_workw_t<Graph_t>> max_work_per_step(number_of_supersteps, 0);
    for (unsigned step = 0; step < number_of_supersteps; step++) {
        v_workw_t<Graph_t> max_work = 0;
        for (unsigned proc = 0; proc < instance.numberOfProcessors(); proc++) {
            if (max_work < work[step][proc]) { max_work = work[step][proc]; }
        }

        max_work_per_step[step] = max_work;
    }

    return max_work_per_step;
}

template <typename Graph_t>
std::vector<v_workw_t<Graph_t>> compute_max_work_per_step(const BspSchedule<Graph_t> &schedule) {
    return compute_max_work_per_step(
        schedule.getInstance(), schedule.numberOfSupersteps(), schedule.assignedProcessors(), schedule.assignedSupersteps());
}

template <typename Graph_t>
v_workw_t<Graph_t> compute_work_costs(const BspInstance<Graph_t> &instance,
                                      unsigned number_of_supersteps,
                                      const std::vector<unsigned> &node_to_processor_assignment,
                                      const std::vector<unsigned> &node_to_superstep_assignment) {
    std::vector<v_workw_t<Graph_t>> max_work_per_step
        = compute_max_work_per_step(instance, number_of_supersteps, node_to_processor_assignment, node_to_superstep_assignment);

    return std::accumulate(max_work_per_step.begin(), max_work_per_step.end(), static_cast<v_workw_t<Graph_t>>(0));
}

template <typename Graph_t>
v_workw_t<Graph_t> compute_work_costs(const BspSchedule<Graph_t> &schedule) {
    return compute_work_costs(
        schedule.getInstance(), schedule.numberOfSupersteps(), schedule.assignedProcessors(), schedule.assignedSupersteps());
}

}    // namespace cost_helpers
}    // namespace osp
