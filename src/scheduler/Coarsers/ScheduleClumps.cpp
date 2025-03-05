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

#include "scheduler/Coarsers/ScheduleClumps.hpp"

RETURN_STATUS ScheduleClumps::run_contractions() {

    const ComputationalDag &graph = dag_history.back()->getComputationalDag();

    std::pair<RETURN_STATUS, BspSchedule> clumping_return = clumpingScheduler->computeSchedule( *(dag_history.back()) );

    RETURN_STATUS &status = clumping_return.first;
    BspSchedule &clumpingSchedule = clumping_return.second;

    Union_Find_Universe<VertexType> universe;

    for (VertexType vert = 0; vert < graph.numberOfVertices(); vert++) {
        universe.add_object(vert);
    }
    for (VertexType vert = 0; vert < graph.numberOfVertices(); vert++) {
        for (const VertexType &child : graph.children(vert)) {
            if ( (clumpingSchedule.assignedSuperstep(vert) == clumpingSchedule.assignedSuperstep(child))
                    && (clumpingSchedule.assignedProcessor(vert) == clumpingSchedule.assignedProcessor(child)) ) {
                
                universe.join_by_name(vert, child);
            }
        }
    }

    std::vector<std::vector<VertexType>> partition = universe.get_connected_components();
    
    std::vector<std::unordered_set<VertexType>> partition_other_format(partition.size());
    for (size_t i = 0; i < partition.size(); i++) {
        for (auto vert : partition[i]) {
            partition_other_format[i].emplace(vert);
        }
    }

    add_contraction(partition_other_format);
    return status;
}


void ScheduleClumps::setUseMemoryConstraint(bool use_memory_constraint_) {
    clumpingScheduler->setUseMemoryConstraint(use_memory_constraint_);
    if (sched) sched->setUseMemoryConstraint(use_memory_constraint_);
    if (improver) improver->setUseMemoryConstraint(use_memory_constraint_);
}