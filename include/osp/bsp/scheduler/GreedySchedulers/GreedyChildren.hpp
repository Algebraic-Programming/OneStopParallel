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

#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

template <typename Graph_t>
class GreedyChildren : public Scheduler<Graph_t> {
  private:
    bool ensure_enough_sources;

  public:
    GreedyChildren(bool ensure_enough_sources_ = true) : Scheduler<Graph_t>(), ensure_enough_sources(ensure_enough_sources_) {};

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &sched) override {
        using VertexType = vertex_idx_t<Graph_t>;
        const auto &instance = sched.getInstance();

        for (const auto &v : instance.getComputationalDag().vertices()) {
            sched.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
        }

        const auto &graph = instance.getComputationalDag();

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
                    unsigned processor_to_be_allocated = 0;

                    for (const auto &par : graph.parents(node)) {
                        if (nodes_assigned_this_superstep.count(par)) {
                            if (!processor_set) {
                                const unsigned par_proc = sched.assignedProcessor(par);
                                if (!instance.isCompatible(node, par_proc)) {
                                    failed_to_allocate = true;
                                    break;
                                }
                                processor_set = true;
                                processor_to_be_allocated = par_proc;
                            } else if (sched.assignedProcessor(par) != processor_to_be_allocated) {
                                failed_to_allocate = true;
                                break;
                            }
                        }
                    }

                    if (failed_to_allocate) {
                        continue;
                    }

                    sched.setAssignedSuperstep(node, superstep_counter);
                    if (processor_set) {
                        sched.setAssignedProcessor(node, processor_to_be_allocated);
                    } else {
                        v_workw_t<Graph_t> min_weight = std::numeric_limits<v_workw_t<Graph_t>>::max();
                        unsigned best_proc = std::numeric_limits<unsigned>::max();
                        for (unsigned p = 0; p < instance.numberOfProcessors(); ++p) {
                            if (instance.isCompatible(node, p)) {
                                if (processor_weights[p] < min_weight) {
                                    min_weight = processor_weights[p];
                                    best_proc = p;
                                }
                            }
                        }
                        sched.setAssignedProcessor(node, best_proc);
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
                if (ensure_enough_sources && few_sources && next.size() >= instance.numberOfProcessors()) {
                    break;
                }
            }

            superstep_counter++;
        }

        return RETURN_STATUS::OSP_SUCCESS;
    }

    std::string getScheduleName() const override { return ensure_enough_sources ? "GreedyChildrenS" : "GreedyChildren"; }
};

}    // namespace osp
