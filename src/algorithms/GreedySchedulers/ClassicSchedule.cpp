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

#include "algorithms/GreedySchedulers/ClassicSchedule.hpp"


BspSchedule CSchedule::convertToBspSchedule(const BspInstance &instance,
                                            const std::vector<std::deque<unsigned>> &procAssignmentLists) {

    BspSchedule bsp_schedule(instance);

    for (unsigned i = 0; i < instance.numberOfVertices(); i++)
        bsp_schedule.setAssignedProcessor(i, proc[i]);

    const int N = instance.numberOfVertices();
    const int P = instance.numberOfProcessors();

    int superStepIdx = 0, totalNodesDone = 0;
    std::vector<bool> processed(N, false);
    std::vector<std::deque<unsigned>::const_iterator> done(P), limit(P);
    for (int j = 0; j < P; ++j)
        done[j] = procAssignmentLists[j].begin();

    while (totalNodesDone < N) {
        // create next superstep
        int timeLimit = INT_MAX;
        for (int j = 0; j < P; ++j) {
            for (limit[j] = done[j]; limit[j] != procAssignmentLists[j].end(); ++limit[j]) {
                const int node = *limit[j];
                bool cut = false;

                for (const auto &source : instance.getComputationalDag().parents(node)) {
                    if (!processed[source] && proc[source] != proc[node])
                        cut = true;
                }

                //                for (const int source : G.In[node])
                //                    if (!processed[source] && proc[source] != proc[node])
                //                        cut = true;

                if (cut)
                    break;
            }
            if (limit[j] != procAssignmentLists[j].end() && time[*limit[j]] < timeLimit)
                timeLimit = time[*limit[j]];
        }

        for (int j = 0; j < P; ++j)
            for (; done[j] != limit[j] &&
                   (time[*done[j]] < timeLimit ||
                    (time[*done[j]] == timeLimit && instance.getComputationalDag().nodeWorkWeight(*done[j]) == 0));
                 ++done[j]) {
                processed[*done[j]] = true;
                bsp_schedule.setAssignedSuperstep(*done[j], superStepIdx);
                ++totalNodesDone;
            }

        ++superStepIdx;
    }
    bsp_schedule.setAutoCommunicationSchedule();
    return bsp_schedule;
};