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

#include "coarser/Coarser.hpp"


std::pair<RETURN_STATUS, BspSchedule> pull_back_schedule(const BspInstance &instance_large, const BspSchedule &schedule_in, const std::vector<std::vector<VertexType>>& vertex_map) {

    BspSchedule schedule_out(instance_large);

    for (unsigned v = 0; v < vertex_map.size(); ++v) {

        for (unsigned i = 0; i < vertex_map[v].size(); ++i) {
            schedule_out.setAssignedSuperstep(vertex_map[v][i], schedule_in.assignedSuperstep(v));
            schedule_out.setAssignedProcessor(vertex_map[v][i], schedule_in.assignedProcessor(v));
        }
    }

    return std::make_pair(RETURN_STATUS::SUCCESS, schedule_out);

}

std::pair<RETURN_STATUS, DAGPartition> pull_back_partition(const BspInstance &instance_large, const DAGPartition &partition_in, const std::vector<std::vector<VertexType>>& vertex_map) {

    DAGPartition partition_out(instance_large);

    for (unsigned v = 0; v < vertex_map.size(); ++v) {

        for (unsigned i = 0; i < vertex_map[v].size(); ++i) {
            partition_out.setAssignedProcessor(vertex_map[v][i], partition_in.assignedProcessor(v));
        }
    }

    return std::make_pair(RETURN_STATUS::SUCCESS, partition_out);

}