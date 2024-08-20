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

#include "scheduler/MultiBspScheduler/MultiBspScheduler.hpp"

std::vector<std::vector<std::unordered_set<VertexType>>> HierarchichalMultiBspScheduler::split_dag(const ComputationalDag &dag, const BspSchedule &schedule) {

    std::vector<std::vector<std::unordered_set<VertexType>>> vertex_sets(
        schedule.getInstance().numberOfProcessors(),
        std::vector<std::unordered_set<VertexType>>(schedule.numberOfSupersteps()));

    for (unsigned node = 0; node < schedule.getInstance().numberOfVertices(); node++) {

        vertex_sets[schedule.assignedProcessor(node)][schedule.assignedSuperstep(node)].insert(node);
    }

    return vertex_sets;
}