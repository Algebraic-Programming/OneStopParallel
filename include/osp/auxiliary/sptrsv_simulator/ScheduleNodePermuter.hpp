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

@author Toni Boehnlein, Christos Matzoros, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <algorithm>
#include <queue>
#include <set>
#include <vector>

#include "osp/bsp/model/BspSchedule.hpp"

namespace osp {

enum ScheduleNodePermutationModes { LOOP_PROCESSORS, SNAKE_PROCESSORS, PROCESSOR_FIRST, NO_PERMUTE };

/**
 * @brief Computes a permutation to improve locality of a schedule, looping through processors
 *
 * @param sched BSP Schedule
 * @param mode ordering of processors
 * @return std::vector<size_t> vec[prev_node_name] = new_node_name(location)
 */
template <typename GraphT>
std::vector<size_t> ScheduleNodePermuterBasic(const BspSchedule<GraphT> &sched,
                                              const ScheduleNodePermutationModes mode = LOOP_PROCESSORS) {
    // superstep, processor, nodes
    std::vector<std::vector<std::vector<size_t>>> allocation(
        sched.numberOfSupersteps(),
        std::vector<std::vector<size_t>>(sched.getInstance().numberOfProcessors(), std::vector<size_t>({})));
    for (size_t node = 0; node < sched.getInstance().numberOfVertices(); node++) {
        allocation[sched.assignedSuperstep(node)][sched.assignedProcessor(node)].emplace_back(node);
    }

    // reordering and allocating into permutation
    std::vector<size_t> permutation(sched.getInstance().numberOfVertices());

    if (mode == LOOP_PROCESSORS || mode == SNAKE_PROCESSORS) {
        bool forward = true;
        size_t counter = 0;
        for (auto stepIt = allocation.begin(); stepIt != allocation.cend(); stepIt++) {
            if (forward) {
                for (auto procIt = stepIt->begin(); procIt != stepIt->cend(); procIt++) {
                    // topological_sort_for_data_locality_interior_basic(*proc_it, sched);
                    for (const auto &node : *procIt) {
                        permutation[node] = counter;
                        counter++;
                    }
                }
            } else {
                for (auto procIt = stepIt->rbegin(); procIt != stepIt->crend(); procIt++) {
                    // topological_sort_for_data_locality_interior_basic(*proc_it, sched);
                    for (const auto &node : *procIt) {
                        permutation[node] = counter;
                        counter++;
                    }
                }
            }

            if (mode == SNAKE_PROCESSORS) {
                forward = !forward;
            }
        }
    } else {
        throw std::logic_error("Permutation mode not implemented.");
    }

    return permutation;
}

}    // namespace osp
