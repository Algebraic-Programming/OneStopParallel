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
class GreedyChildren : public Scheduler<GraphT> {
  private:
    bool ensureEnoughSources_;

  public:
    GreedyChildren(bool ensureEnoughSources = true) : Scheduler<GraphT>(), ensureEnoughSources_(ensureEnoughSources) {};

    ReturnStatus computeSchedule(BspSchedule<GraphT> &sched) override {
        using VertexType = VertexIdxT<GraphT>;
        const auto &instance = sched.GetInstance();

        for (const auto &v : instance.GetComputationalDag().vertices()) {
            sched.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
        }

        const auto &graph = instance.GetComputationalDag();

        unsigned superstepCounter = 0;

        std::vector<VertexType> predecessorsCount(instance.NumberOfVertices(), 0);
        std::multiset<std::pair<unsigned, VertexType>, std::greater<>> next;
        for (const VertexType &i : source_vertices_view(graph)) {
            next.emplace(graph.OutDegree(i), i);
        }

        while (!next.empty()) {
            std::unordered_set<VertexType> nodesAssignedThisSuperstep;
            std::vector<VWorkwT<GraphT>> processorWeights(instance.NumberOfProcessors(), 0);

            bool fewSources = next.size() < instance.NumberOfProcessors() ? true : false;
            bool nodeAdded = true;
            while (!next.empty() && node_added) {
                nodeAdded = false;
                for (auto iter = next.begin(); iter != next.cend(); iter++) {
                    const auto &node = iter->second;
                    bool processorSet = false;
                    bool failedToAllocate = false;
                    unsigned processorToBeAllocated = 0;

                    for (const auto &par : graph.Parents(node)) {
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

                    if (failedToAllocate) {
                        continue;
                    }

                    sched.setAssignedSuperstep(node, superstepCounter);
                    if (processorSet) {
                        sched.setAssignedProcessor(node, processorToBeAllocated);
                    } else {
                        VWorkwT<GraphT> minWeight = std::numeric_limits<VWorkwT<GraphT>>::max();
                        unsigned bestProc = std::numeric_limits<unsigned>::max();
                        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
                            if (instance.isCompatible(node, p)) {
                                if (processorWeights[p] < min_weight) {
                                    minWeight = processor_weights[p];
                                    bestProc = p;
                                }
                            }
                        }
                        sched.setAssignedProcessor(node, bestProc);
                    }

                    nodesAssignedThisSuperstep.emplace(node);
                    processorWeights[sched.assignedProcessor(node)] += graph.VertexWorkWeight(node);
                    std::vector<VertexType> newNodes;
                    for (const auto &chld : graph.Children(node)) {
                        predecessors_count[chld]++;
                        if (predecessors_count[chld] == graph.in_degree(chld)) {
                            new_nodes.emplace_back(chld);
                        }
                    }
                    next.erase(iter);
                    for (const auto &vrt : new_nodes) {
                        next.emplace(graph.OutDegree(vrt), vrt);
                    }
                    nodeAdded = true;
                    break;
                }
                if (ensure_enough_sources && few_sources && next.size() >= instance.NumberOfProcessors()) {
                    break;
                }
            }

            superstepCounter++;
        }

        return ReturnStatus::OSP_SUCCESS;
    }

    std::string getScheduleName() const override { return ensureEnoughSources_ ? "GreedyChildrenS" : "GreedyChildren"; }
};

}    // namespace osp
