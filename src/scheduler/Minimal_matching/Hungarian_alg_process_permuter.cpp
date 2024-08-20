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

#include "scheduler/Minimal_matching/Hungarian_alg_process_permuter.hpp"

RETURN_STATUS Hungarian_alg_process_permuter::improveSchedule(BspSchedule &schedule) {
    std::pair<RETURN_STATUS, BspSchedule> out = constructImprovedSchedule(schedule);
    schedule = out.second;
    return out.first;
}

std::pair<RETURN_STATUS, BspSchedule> Hungarian_alg_process_permuter::constructImprovedSchedule(const BspSchedule &schedule) {
    BspSchedule new_sched(schedule.getInstance());
    new_sched.setAssignedSupersteps( schedule.assignedSupersteps() );

    std::vector<std::vector<size_t>> superstep_nodes(schedule.numberOfSupersteps());
    for (size_t node = 0; node < schedule.getInstance().numberOfVertices(); node++) {
        superstep_nodes[ schedule.assignedSuperstep(node) ].emplace_back(node);
    }

    std::vector<bool> node_assigned_processor(schedule.getInstance().numberOfVertices(), false);
    for (size_t sstep = 0; sstep < superstep_nodes.size(); sstep++) {
        std::vector<std::vector<long long unsigned>> partition_to_processor_comm_cost(  schedule.getInstance().numberOfProcessors(),
                                                                                        std::vector<long long unsigned>(schedule.getInstance().numberOfProcessors(), 0) );

        // computing costs
        for (const auto& node : superstep_nodes[sstep]) {
            for (const auto& in_edge : schedule.getInstance().getComputationalDag().in_edges(node)) {
                if (node_assigned_processor[in_edge.m_source]) {
                    for (long unsigned procssr = 0; procssr< schedule.getInstance().numberOfProcessors(); procssr++) {
                        if ( procssr ==  new_sched.assignedProcessor( in_edge.m_source ) ) continue;

                        partition_to_processor_comm_cost[ new_sched.assignedProcessor(node) ][ procssr ]
                            += schedule.getInstance().getComputationalDag().nodeCommunicationWeight(in_edge.m_source)
                                * schedule.getInstance().getArchitecture().sendCostMatrix()[new_sched.assignedProcessor(in_edge.m_source)][procssr];
                    }
                }
            }
        }

        // deciding partition to processor allocation
        std::vector<unsigned> partition_to_processor_allocation = min_perfect_matching_for_complete_bipartite(partition_to_processor_comm_cost);

        // allocating nodes
        for (const auto& node : superstep_nodes[sstep]) {
            new_sched.setAssignedProcessor( node, partition_to_processor_allocation[ schedule.assignedProcessor(node) ] );
            node_assigned_processor[node] = true;
        }
    }

    // check that all nodes have been assigned a processor
    assert( std::all_of(node_assigned_processor.cbegin(), node_assigned_processor.cend(), []( const bool& has_been ) {return has_been;} ) );


    new_sched.setAutoCommunicationSchedule();
    return std::make_pair(SUCCESS, new_sched);
}
