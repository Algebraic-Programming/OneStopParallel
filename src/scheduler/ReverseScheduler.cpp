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

#include "scheduler/ReverseScheduler.hpp"

void ReverseScheduler::setTimeLimitSeconds(unsigned int limit) {
    timeLimitSeconds = limit;
    if (base_scheduler) base_scheduler->setTimeLimitHours(limit);
}

void ReverseScheduler::setTimeLimitHours(unsigned int limit) {
    timeLimitSeconds = limit * 3600;
    if (base_scheduler) base_scheduler->setTimeLimitHours(limit);
}

std::pair<RETURN_STATUS, BspSchedule> ReverseScheduler::computeSchedule(const BspInstance &instance) {
    BspInstance rev_inst;
    std::vector<VertexType> vertex_mapping;
    std::tie(vertex_mapping, rev_inst.getComputationalDag()) = instance.getComputationalDag().reverse_graph();
    rev_inst.setArchitecture(instance.getArchitecture());
    rev_inst.setNodeProcessorCompatibility(instance.getNodeProcessorCompatibilityMatrix());

    RETURN_STATUS rev_status;
    BspSchedule rev_schedule;

    std::tie(rev_status, rev_schedule) = base_scheduler->computeSchedule(rev_inst);

    if (rev_status != SUCCESS && rev_status != BEST_FOUND) {
        return {rev_status, rev_schedule};
    }

    rev_schedule.updateNumberOfSupersteps();
    const unsigned total_supersteps = rev_schedule.numberOfSupersteps() - 1;

    BspSchedule schedule(instance);
    for (VertexType vert = 0; vert < instance.numberOfVertices(); vert++) {
        schedule.setAssignedProcessor(vert, rev_schedule.assignedProcessor(vertex_mapping[vert]));
    }
    for (VertexType vert = 0; vert < instance.numberOfVertices(); vert++) {
        schedule.setAssignedSuperstep(vert, total_supersteps - rev_schedule.assignedSuperstep(vertex_mapping[vert]));
    }
    schedule.setAutoCommunicationSchedule();

    return {rev_status, schedule};
}