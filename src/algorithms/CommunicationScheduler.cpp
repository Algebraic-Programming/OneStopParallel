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

#include "algorithms/CommunicationScheduler.hpp"


std::map<KeyTriple, unsigned>  LazyCommunicationScheduler::computeCommunicationSchedule(const IBspSchedule &sched) {

    std::map<KeyTriple, unsigned> commSchedule;

    for (const auto &ep : sched.getInstance().getComputationalDag().edges()) {
        const unsigned int source = sched.getInstance().getComputationalDag().source(ep);
        const unsigned int target = sched.getInstance().getComputationalDag().target(ep);

        if (sched.assignedProcessor(source) != sched.assignedProcessor(target)) {

            const auto &tmp =
                std::make_tuple(source, sched.assignedProcessor(source), sched.assignedProcessor(target));
            if (commSchedule.find(tmp) == commSchedule.end()) {
                commSchedule[tmp] = sched.assignedSuperstep(target) - 1;

            } else {
                commSchedule[tmp] = std::min(sched.assignedSuperstep(target) - 1, commSchedule[tmp]);
            }
        }
    }

    return commSchedule;
}


std::map<KeyTriple, unsigned>  EagerCommunicationScheduler::computeCommunicationSchedule(const IBspSchedule &sched) {

    std::map<KeyTriple, unsigned> commSchedule;

    for (const auto &ep : sched.getInstance().getComputationalDag().edges()) {
        const unsigned int source = sched.getInstance().getComputationalDag().source(ep);
        const unsigned int target = sched.getInstance().getComputationalDag().target(ep);

        if (sched.assignedProcessor(source) != sched.assignedProcessor(target)) {

            commSchedule[std::make_tuple(source, sched.assignedProcessor(source),
                                         sched.assignedProcessor(target))] = sched.assignedSuperstep(source);
        }
    }

    return commSchedule;
}