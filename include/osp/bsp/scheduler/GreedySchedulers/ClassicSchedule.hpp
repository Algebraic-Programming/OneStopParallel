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

#include <deque>
#include <limits>
#include <vector>

#include "osp/bsp/model/BspSchedule.hpp"

namespace osp {

/**
 * @class CSchedule
 * @brief Represents a classic schedule for scheduling scheduler.
 *
 * This class stores the processor and time information for a schedule.
 */
template <typename GraphT>
class CSchedule {
  private:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using workw_t = v_workw_t<Graph_t>;

  public:
    std::vector<unsigned> proc_; /**< The processor assigned to each task. */
    std::vector<workw_t> time_;  /**< The time at which each task starts. */

    /**
     * @brief Constructs a CSchedule object with the given size.
     * @param size The size of the schedule.
     */
    CSchedule(std::size_t size)
        : proc_(std::vector<unsigned>(size, std::numeric_limits<unsigned>::max())), time(std::vector<workw_t>(size, 0)) {}

    /**
     * @brief Converts the CSchedule object to a BspSchedule object.
     * @param instance The BspInstance object representing the BSP instance.
     * @param local_greedyProcLists The local greedy processor lists.
     * @return The converted BspSchedule object.
     */

    void ConvertToBspSchedule(const BspInstance<GraphT> &instance,
                              const std::vector<std::deque<vertex_idx>> &procAssignmentLists,
                              BspSchedule<GraphT> &bspSchedule) {
        for (const auto &v : instance.vertices()) {
            bspSchedule.setAssignedProcessor(v, proc_[v]);
        }

        const vertex_idx n = instance.numberOfVertices();
        const unsigned p = instance.NumberOfProcessors();

        unsigned superStepIdx = 0, totalNodesDone = 0;
        std::vector<bool> processed(n, false);

        std::vector<decltype(procAssignmentLists[0].cbegin())> done(p), limit(P);

        for (unsigned j = 0; j < p; ++j) {
            done[j] = procAssignmentLists[j].cbegin();
        }

        while (totalNodesDone < n) {
            // create next superstep
            workw_t timeLimit = std::numeric_limits<workw_t>::max();
            for (unsigned j = 0; j < p; ++j) {
                for (limit[j] = done[j]; limit[j] != procAssignmentLists[j].end(); ++limit[j]) {
                    const vertex_idx node = *limit[j];
                    bool cut = false;

                    for (const auto &source : instance.getComputationalDag().parents(node)) {
                        if (!processed[source] && proc[source] != proc[node]) {
                            cut = true;
                        }
                    }

                    if (cut) {
                        break;
                    }
                }
                if (limit[j] != procAssignmentLists[j].end() && time[*limit[j]] < timeLimit) {
                    timeLimit = time[*limit[j]];
                }
            }

            for (unsigned j = 0; j < p; ++j) {
                for (; done[j] != limit[j]
                       && (time[*done[j]] < timeLimit
                           || (time[*done[j]] == timeLimit && instance.getComputationalDag().VertexWorkWeight(*done[j]) == 0));
                     ++done[j]) {
                    processed[*done[j]] = true;
                    bspSchedule.setAssignedSuperstep(*done[j], superStepIdx);
                    ++totalNodesDone;
                }
            }

            ++superStepIdx;
        }
    }
};

}    // namespace osp
