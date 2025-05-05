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

#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <vector>

#include "bsp/scheduler/Scheduler.hpp"
#include "graph_algorithms/directed_graph_util.hpp"

namespace osp {

template<typename Graph_t>
class GreedyChildren : public Scheduler<Graph_t> {
  private:
    bool ensure_enough_sources;

  public:
    GreedyChildren(bool ensure_enough_sources_ = true) : ensure_enough_sources(ensure_enough_sources_) {};
    GreedyChildren(unsigned time_limit, bool ensure_enough_sources_ = true)
        : Scheduler<Graph_t>(time_limit), ensure_enough_sources(ensure_enough_sources_) {};

    std::pair<RETURN_STATUS, BspSchedule<Graph_t>> computeSchedule(const BspInstance<Graph_t> &instance) override {

        using VertexType = vertex_idx_t<Graph_t>;

        const auto &graph = instance.getComputationalDag();

        BspSchedule<Graph_t> sched(instance);

        unsigned superstep_counter = 0;

        std::vector<VertexType> predecessors_count(instance.numberOfVertices(), 0);
        std::multiset<std::pair<unsigned, VertexType>, std::greater<>> next;
        for (const VertexType &i : source_vertices_view(graph)) {
            next.emplace(graph.out_degree(i), i);
        }

        while (!next.empty()) {
            std::unordered_set<VertexType> nodes_assigned_this_superstep;
            std::vector<v_workw_t<Graph_t>> processor_weights(instance.numberOfProcessors(), 0);

            bool few_sources = next.size() < instance.numberOfProcessors() ? true : false;
            bool node_added = true;
            while (!next.empty() && node_added) {
                node_added = false;
                for (auto iter = next.begin(); iter != next.cend(); iter++) {
                    const auto &node = iter->second;
                    bool processor_set = false;
                    bool failed_to_allocate = false;
                    unsigned processor_to_be_allocated;

                    for (const auto &par : graph.parents(node)) {
                        if (processor_set &&
                            (nodes_assigned_this_superstep.find(par) != nodes_assigned_this_superstep.cend()) &&
                            (sched.assignedProcessor(par) != processor_to_be_allocated)) {
                            failed_to_allocate = true;
                            break;
                        }
                        if ((!processor_set) &&
                            (nodes_assigned_this_superstep.find(par) != nodes_assigned_this_superstep.cend())) {
                            processor_set = true;
                            processor_to_be_allocated = sched.assignedProcessor(par);
                        }
                    }
                    if (failed_to_allocate)
                        continue;

                    sched.setAssignedSuperstep(node, superstep_counter);
                    if (processor_set) {
                        sched.setAssignedProcessor(node, processor_to_be_allocated);
                    } else {

                        auto min_iter = std::min_element(processor_weights.begin(), processor_weights.end());
                        assert(std::distance(processor_weights.begin(), min_iter) >= 0);
                        sched.setAssignedProcessor(
                            node, static_cast<unsigned>(std::distance(processor_weights.begin(), min_iter)));
                    }

                    nodes_assigned_this_superstep.emplace(node);
                    processor_weights[sched.assignedProcessor(node)] += graph.vertex_work_weight(node);
                    std::vector<VertexType> new_nodes;
                    for (const auto &chld : graph.children(node)) {
                        predecessors_count[chld]++;
                        if (predecessors_count[chld] == graph.in_degree(chld)) {
                            new_nodes.emplace_back(chld);
                        }
                    }
                    next.erase(iter);
                    for (const auto &vrt : new_nodes) {
                        next.emplace(graph.out_degree(vrt), vrt);
                    }
                    node_added = true;
                    break;
                }
                if (ensure_enough_sources && few_sources && next.size() >= instance.numberOfProcessors())
                    break;
            }

            superstep_counter++;
        }

        return {SUCCESS, sched};
    }

    std::string getScheduleName() const override {
        return ensure_enough_sources ? "GreedyChildrenS" : "GreedyChildren";
    }
};

} // namespace osp