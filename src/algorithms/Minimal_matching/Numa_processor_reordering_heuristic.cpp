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

#include "algorithms/Minimal_matching/Numa_processor_reordering_heuristic.hpp"

std::vector<std::vector<unsigned>> numa_processor_reordering_heuristic::comp_p2p_comm(const BspSchedule &schedule) {
    std::vector<std::vector<unsigned>> proc_to_proc_comm =
        std::vector<std::vector<unsigned>>(schedule.getInstance().numberOfProcessors(),
                                           std::vector<unsigned>(schedule.getInstance().numberOfProcessors(), 0));

    for (size_t node = 0; node < schedule.getInstance().numberOfVertices(); node++) {
        for (const auto &chld_edge : schedule.getInstance().getComputationalDag().out_edges(node)) {
            if (schedule.assignedProcessor(node) != schedule.assignedProcessor(chld_edge.m_target)) {
                proc_to_proc_comm[schedule.assignedProcessor(node)][schedule.assignedProcessor(chld_edge.m_target)] +=
                    schedule.getInstance().getComputationalDag().nodeCommunicationWeight(node);
            }
        }
    }

    return proc_to_proc_comm;
}

RETURN_STATUS numa_processor_reordering_heuristic::improveSchedule(BspSchedule &schedule) {
    std::pair<RETURN_STATUS, BspSchedule> out = constructImprovedSchedule(schedule);
    schedule = out.second;
    return out.first;
}

std::pair<RETURN_STATUS, BspSchedule>
numa_processor_reordering_heuristic::constructImprovedSchedule(const BspSchedule &schedule) {
    if (!schedule.getInstance().getArchitecture().isNumaArchitecture())
        return {SUCCESS, schedule};

    BspSchedule new_sched(schedule.getInstance());
    new_sched.setAssignedSupersteps(schedule.assignedSupersteps());

    std::vector<unsigned> proc_reorder = compute_best_reordering(schedule);
    for (size_t node = 0; node < schedule.getInstance().numberOfVertices(); node++) {
        new_sched.setAssignedProcessor(node, proc_reorder[schedule.assignedProcessor(node)]);
    }

    new_sched.setAutoCommunicationSchedule();
    return std::make_pair(SUCCESS, new_sched);
}

std::vector<unsigned> numa_processor_reordering_heuristic::compute_best_reordering(const BspSchedule &schedule) {

    throw std::runtime_error("ILP partitioner did not find a solution :(");
}