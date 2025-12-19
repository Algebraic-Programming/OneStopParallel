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

    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &sched) override {
        using VertexType = VertexIdxT<GraphT>;
        const auto &instance = sched.GetInstance();

        for (const auto &v : instance.GetComputationalDag().Vertices()) {
            sched.SetAssignedProcessor(v, std::numeric_limits<unsigned>::max());
        }

        const auto &graph = instance.GetComputationalDag();

        unsigned superstepCounter = 0;

        std::vector<VertexType> predecessorsCount(instance.NumberOfVertices(), 0);
        std::multiset<std::pair<unsigned, VertexType>, std::greater<>> next;
        for (const VertexType &i : SourceVerticesView(graph)) {
            next.emplace(graph.OutDegree(i), i);
        }

        while (!next.empty()) {
            std::unordered_set<VertexType> nodesAssignedThisSuperstep;
            std::vector<VWorkwT<GraphT>> processorWeights(instance.NumberOfProcessors(), 0);

            bool fewSources = next.size() < instance.NumberOfProcessors() ? true : false;
            bool nodeAdded = true;
            while (!next.empty() && nodeAdded) {
                nodeAdded = false;
                for (auto iter = next.begin(); iter != next.cend(); iter++) {
                    const auto &node = iter->second;
                    bool processorSet = false;
                    bool failedToAllocate = false;
                    unsigned processorToBeAllocated = 0;

                    for (const auto &par : graph.Parents(node)) {
                        if (nodesAssignedThisSuperstep.count(par)) {
                            if (!processorSet) {
                                const unsigned parProc = sched.AssignedProcessor(par);
                                if (!instance.IsCompatible(node, parProc)) {
                                    failedToAllocate = true;
                                    break;
                                }
                                processorSet = true;
                                processorToBeAllocated = parProc;
                            } else if (sched.AssignedProcessor(par) != processorToBeAllocated) {
                                failedToAllocate = true;
                                break;
                            }
                        }
                    }

                    if (failedToAllocate) {
                        continue;
                    }

                    sched.SetAssignedSuperstep(node, superstepCounter);
                    if (processorSet) {
                        sched.SetAssignedProcessor(node, processorToBeAllocated);
                    } else {
                        VWorkwT<GraphT> minWeight = std::numeric_limits<VWorkwT<GraphT>>::max();
                        unsigned bestProc = std::numeric_limits<unsigned>::max();
                        for (unsigned p = 0; p < instance.NumberOfProcessors(); ++p) {
                            if (instance.IsCompatible(node, p)) {
                                if (processorWeights[p] < minWeight) {
                                    minWeight = processorWeights[p];
                                    bestProc = p;
                                }
                            }
                        }
                        sched.SetAssignedProcessor(node, bestProc);
                    }

                    nodesAssignedThisSuperstep.emplace(node);
                    processorWeights[sched.AssignedProcessor(node)] += graph.VertexWorkWeight(node);
                    std::vector<VertexType> newNodes;
                    for (const auto &chld : graph.Children(node)) {
                        predecessorsCount[chld]++;
                        if (predecessorsCount[chld] == graph.InDegree(chld)) {
                            newNodes.emplace_back(chld);
                        }
                    }
                    next.erase(iter);
                    for (const auto &vrt : newNodes) {
                        next.emplace(graph.OutDegree(vrt), vrt);
                    }
                    nodeAdded = true;
                    break;
                }
                if (ensureEnoughSources_ && fewSources && next.size() >= instance.NumberOfProcessors()) {
                    break;
                }
            }

            superstepCounter++;
        }

        return ReturnStatus::OSP_SUCCESS;
    }

    std::string GetScheduleName() const override { return ensureEnoughSources_ ? "GreedyChildrenS" : "GreedyChildren"; }
};

}    // namespace osp
