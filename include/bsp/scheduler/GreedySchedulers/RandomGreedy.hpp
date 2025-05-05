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
class RandomGreedy : public Scheduler<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "RandomGreedy can only be used with computational DAGs.");

  private:
    bool ensure_enough_sources;

  public:
    RandomGreedy(bool ensure_enough_sources_ = true) : ensure_enough_sources(ensure_enough_sources_) {};
    RandomGreedy(unsigned time_limit, bool ensure_enough_sources_)
        : Scheduler<Graph_t>(time_limit), ensure_enough_sources(ensure_enough_sources_) {};

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &sched) override {

        using VertexType = vertex_idx_t<Graph_t>;

        const auto &instance = sched.getInstance();

        for (const auto &v : instance.getComputationalDag().vertices()) {
            sched.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
            sched.setAssignedSuperstep(v, std::numeric_limits<unsigned>::max());
        }

        const auto &graph = instance.getComputationalDag();

        unsigned superstep_counter = 0;

        std::vector<VertexType> predecessors_count(instance.numberOfVertices(), 0);
        std::vector<VertexType> next;
        for (const auto &i : source_vertices_view(graph)) {
            next.push_back(i);
        }

        std::random_device rd;
        std::mt19937 g(rd());

        while (!next.empty()) {
            std::shuffle(next.begin(), next.end(), g);
            std::unordered_set<VertexType> nodes_assigned_this_superstep;
            std::vector<v_workw_t<Graph_t>> processor_weights(instance.numberOfProcessors(), 0);

            bool few_sources = next.size() < instance.numberOfProcessors() ? true : false;
            unsigned fail_counter = 0;
            while (!next.empty() && fail_counter < 20) {

                std::uniform_int_distribution<VertexType> rand_node_idx(0, next.size() - 1);
                size_t node_ind = rand_node_idx(g);
                const auto &node = next[node_ind];
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
                if (failed_to_allocate) {
                    fail_counter++;
                    continue;
                } else {
                    fail_counter = 0;
                }

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
                assert(node_ind < std::numeric_limits<long>::max());
                next.erase(std::next(next.begin(), static_cast<long>(node_ind)));
                next.insert(next.end(), new_nodes.cbegin(), new_nodes.cend());

                if (ensure_enough_sources && few_sources && next.size() >= instance.numberOfProcessors())
                    break;
            }

            superstep_counter++;
        }

        return SUCCESS;
    }

    std::string getScheduleName() const override { return ensure_enough_sources ? "RandomGreedyS" : "RandomGreedy"; }
};

} // namespace osp