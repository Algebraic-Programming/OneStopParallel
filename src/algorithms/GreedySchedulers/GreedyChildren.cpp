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

#include "algorithms/GreedySchedulers/GreedyChildren.hpp"

std::pair<RETURN_STATUS, BspSchedule> GreedyChildren::computeSchedule(const BspInstance& instance) {
    const auto& graph = instance.getComputationalDag();
    BspSchedule sched(instance);

    unsigned superstep_counter = 0;

    std::vector<VertexType> predecessors_count(graph.numberOfVertices(), 0);
    std::multiset<std::pair<unsigned, VertexType>, std::greater<>> next;
    for (const VertexType &i : graph.sourceVertices()) {
            next.emplace(graph.children(i).size(), i);
    }

    while (!next.empty()) {
        std::unordered_set<VertexType> nodes_assigned_this_superstep;
        std::vector<size_t> processor_weights(instance.numberOfProcessors(), 0);

        bool few_sources = next.size() < instance.numberOfProcessors() ? true : false ;
        bool node_added = true;
        while (!next.empty() && node_added) {
            node_added = false;
            for (auto iter = next.begin(); iter != next.cend(); iter++) {
                const auto& node = iter->second;
                bool processor_set = false;
                bool failed_to_allocate = false;
                unsigned processor_to_be_allocated;

                for (const auto& par : graph.parents(node)) {
                    if (processor_set && (nodes_assigned_this_superstep.find(par) != nodes_assigned_this_superstep.cend()) && (sched.assignedProcessor(par) != processor_to_be_allocated)) {
                        failed_to_allocate = true;
                        break;
                    }
                    if ((!processor_set) && (nodes_assigned_this_superstep.find(par) != nodes_assigned_this_superstep.cend())) {
                        processor_set = true;
                        processor_to_be_allocated = sched.assignedProcessor(par);
                    }
                }
                if (failed_to_allocate) continue;

                sched.setAssignedSuperstep(node, superstep_counter);
                if (processor_set) {
                    sched.setAssignedProcessor(node, processor_to_be_allocated);
                } else {
                    auto min_iter = std::min_element(processor_weights.begin(), processor_weights.end());
                    sched.setAssignedProcessor(node, std::distance(processor_weights.begin(), min_iter));
                }

                nodes_assigned_this_superstep.emplace(node);
                processor_weights[sched.assignedProcessor(node)] += graph.nodeWorkWeight(node);
                std::vector<VertexType> new_nodes;
                for (const auto& chld : graph.children(node)) {
                    predecessors_count[chld]++;
                    if ( predecessors_count[chld] == graph.parents(chld).size() ) {
                        new_nodes.emplace_back(chld);
                    }
                }
                next.erase(iter);
                for (const auto& vrt : new_nodes) {
                    next.emplace(graph.children(vrt).size(), vrt);
                }
                node_added = true;
                break;
            }
            if (ensure_enough_sources && few_sources && next.size() >= instance.numberOfProcessors() ) break;
        }

        superstep_counter++;
    }

    sched.setAutoCommunicationSchedule();
    return {SUCCESS, sched};
}

