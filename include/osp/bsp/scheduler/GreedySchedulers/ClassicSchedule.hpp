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
    using VertexIdx = VertexIdxT<GraphT>;
    using WorkwT = VWorkwT<GraphT>;

  public:
    std::vector<unsigned> proc_; /**< The processor assigned to each task. */
    std::vector<WorkwT> time_;   /**< The time at which each task starts. */

    /**
     * @brief Constructs a CSchedule object with the given size.
     * @param size The size of the schedule.
     */
    CSchedule(std::size_t size)
        : proc_(std::vector<unsigned>(size, std::numeric_limits<unsigned>::max())), time_(std::vector<WorkwT>(size, 0)) {}

    /**
     * @brief Converts the CSchedule object to a BspSchedule object.
     * @param instance The BspInstance object representing the BSP instance.
     * @param local_greedyProcLists The local greedy processor lists.
     * @return The converted BspSchedule object.
     */

    void ConvertToBspSchedule(const BspInstance<GraphT> &instance,
                              const std::vector<std::deque<VertexIdx>> &procAssignmentLists,
                              BspSchedule<GraphT> &bspSchedule) {
        for (const auto &v : instance.Vertices()) {
            bspSchedule.SetAssignedProcessor(v, proc_[v]);
        }

        const VertexIdx n = instance.NumberOfVertices();
        const unsigned p = instance.NumberOfProcessors();

        unsigned superStepIdx = 0, totalNodesDone = 0;
        std::vector<bool> processed(n, false);

        std::vector<decltype(procAssignmentLists[0].cbegin())> done(p), limit(p);

        for (unsigned j = 0; j < p; ++j) {
            done[j] = procAssignmentLists[j].cbegin();
        }

        while (totalNodesDone < n) {
            // create next superstep
            WorkwT timeLimit = std::numeric_limits<WorkwT>::max();
            for (unsigned j = 0; j < p; ++j) {
                for (limit[j] = done[j]; limit[j] != procAssignmentLists[j].end(); ++limit[j]) {
                    const VertexIdx node = *limit[j];
                    bool cut = false;

                    for (const auto &source : instance.GetComputationalDag().Parents(node)) {
                        if (!processed[source] && proc_[source] != proc_[node]) {
                            cut = true;
                        }
                    }

                    if (cut) {
                        break;
                    }
                }
                if (limit[j] != procAssignmentLists[j].end() && time_[*limit[j]] < timeLimit) {
                    timeLimit = time_[*limit[j]];
                }
            }

            for (unsigned j = 0; j < p; ++j) {
                for (; done[j] != limit[j]
                       && (time_[*done[j]] < timeLimit
                           || (time_[*done[j]] == timeLimit && instance.GetComputationalDag().VertexWorkWeight(*done[j]) == 0));
                     ++done[j]) {
                    processed[*done[j]] = true;
                    bspSchedule.SetAssignedSuperstep(*done[j], superStepIdx);
                    ++totalNodesDone;
                }
            }

            ++superStepIdx;
        }
    }
};

}    // namespace osp
