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

template <typename GraphT>
class RandomGreedy : public Scheduler<GraphT> {
    static_assert(IsComputationalDagV<Graph_t>, "RandomGreedy can only be used with computational DAGs.");

  private:
    bool ensureEnoughSources_;

  public:
    RandomGreedy(bool ensureEnoughSources = true) : Scheduler<GraphT>(), ensureEnoughSources_(ensureEnoughSources) {};

    RETURN_STATUS computeSchedule(BspSchedule<GraphT> &sched) override {
        using VertexType = vertex_idx_t<Graph_t>;

        const auto &instance = sched.GetInstance();

        for (const auto &v : instance.getComputationalDag().vertices()) {
            sched.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
            sched.setAssignedSuperstep(v, std::numeric_limits<unsigned>::max());
        }

        const auto &graph = instance.getComputationalDag();

        unsigned superstepCounter = 0;

        std::vector<VertexType> predecessorsCount(instance.numberOfVertices(), 0);
        std::vector<VertexType> next;
        for (const auto &i : source_vertices_view(graph)) {
            next.push_back(i);
        }

        std::random_device rd;
        std::mt19937 g(rd());

        while (!next.empty()) {
            std::shuffle(next.begin(), next.end(), g);
            std::unordered_set<VertexType> nodesAssignedThisSuperstep;
            std::vector<v_workw_t<Graph_t>> processorWeights(instance.NumberOfProcessors(), 0);

            bool fewSources = next.size() < instance.NumberOfProcessors() ? true : false;
            unsigned failCounter = 0;
            while (!next.empty() && failCounter < 20) {
                std::uniform_int_distribution<VertexType> randNodeIdx(0, next.size() - 1);
                VertexType nodeInd = rand_node_idx(g);
                const auto &node = next[node_ind];
                bool processorSet = false;
                bool failedToAllocate = false;
                unsigned processorToBeAllocated = 0;

                for (const auto &par : graph.parents(node)) {
                    if (processor_set && (nodes_assigned_this_superstep.find(par) != nodes_assigned_this_superstep.cend())
                        && (sched.assignedProcessor(par) != processor_to_be_allocated)) {
                        failed_to_allocate = true;
                        break;
                    }
                    if ((!processor_set) && (nodes_assigned_this_superstep.find(par) != nodes_assigned_this_superstep.cend())) {
                        processor_set = true;
                        processor_to_be_allocated = sched.assignedProcessor(par);
                    }
                }
                if (failedToAllocate) {
                    failCounter++;
                    continue;
                } else {
                    failCounter = 0;
                }

                sched.setAssignedSuperstep(node, superstepCounter);
                if (processorSet) {
                    sched.setAssignedProcessor(node, processorToBeAllocated);
                } else {
                    auto minIter = std::min_element(processor_weights.begin(), processor_weights.end());

                    assert(std::distance(processor_weights.begin(), min_iter) >= 0);

                    sched.setAssignedProcessor(node, static_cast<unsigned>(std::distance(processor_weights.begin(), min_iter)));
                }

                nodesAssignedThisSuperstep.emplace(node);
                processorWeights[sched.assignedProcessor(node)] += graph.VertexWorkWeight(node);
                std::vector<VertexType> newNodes;
                for (const auto &chld : graph.children(node)) {
                    predecessors_count[chld]++;
                    if (predecessors_count[chld] == graph.in_degree(chld)) {
                        new_nodes.emplace_back(chld);
                    }
                }

                auto it = next.begin();
                std::advance(it, node_ind);
                next.erase(it);
                next.insert(next.end(), new_nodes.cbegin(), new_nodes.cend());

                if (ensureEnoughSources_ && fewSources && next.size() >= instance.NumberOfProcessors()) {
                    break;
                }
            }

            superstepCounter++;
        }

        return RETURN_STATUS::OSP_SUCCESS;
    }

    std::string getScheduleName() const override { return ensureEnoughSources_ ? "RandomGreedyS" : "RandomGreedy"; }
};

}    // namespace osp
