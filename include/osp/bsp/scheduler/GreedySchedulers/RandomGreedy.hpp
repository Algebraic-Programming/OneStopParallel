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
    static_assert(IsComputationalDagV<GraphT>, "RandomGreedy can only be used with computational DAGs.");

  private:
    bool ensureEnoughSources_;

  public:
    RandomGreedy(bool ensureEnoughSources = true) : Scheduler<GraphT>(), ensureEnoughSources_(ensureEnoughSources) {};

    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &sched) override {
        using VertexType = VertexIdxT<GraphT>;

        const auto &instance = sched.GetInstance();

        for (const auto &v : instance.GetComputationalDag().Vertices()) {
            sched.SetAssignedProcessor(v, std::numeric_limits<unsigned>::max());
            sched.SetAssignedSuperstep(v, std::numeric_limits<unsigned>::max());
        }

        const auto &graph = instance.GetComputationalDag();

        unsigned superstepCounter = 0;

        std::vector<VertexType> predecessorsCount(instance.NumberOfVertices(), 0);
        std::vector<VertexType> next;
        for (const auto &i : SourceVerticesView(graph)) {
            next.push_back(i);
        }

        std::random_device rd;
        std::mt19937 g(rd());

        while (!next.empty()) {
            std::shuffle(next.begin(), next.end(), g);
            std::unordered_set<VertexType> nodesAssignedThisSuperstep;
            std::vector<VWorkwT<GraphT>> processorWeights(instance.NumberOfProcessors(), 0);

            bool fewSources = next.size() < instance.NumberOfProcessors() ? true : false;
            unsigned failCounter = 0;
            while (!next.empty() && failCounter < 20) {
                std::uniform_int_distribution<VertexType> randNodeIdx(0, next.size() - 1);
                VertexType nodeInd = randNodeIdx(g);
                const auto &node = next[nodeInd];
                bool processorSet = false;
                bool failedToAllocate = false;
                unsigned processorToBeAllocated = 0;

                for (const auto &par : graph.Parents(node)) {
                    if (processorSet && (nodesAssignedThisSuperstep.find(par) != nodesAssignedThisSuperstep.cend())
                        && (sched.AssignedProcessor(par) != processorToBeAllocated)) {
                        failedToAllocate = true;
                        break;
                    }
                    if ((!processorSet) && (nodesAssignedThisSuperstep.find(par) != nodesAssignedThisSuperstep.cend())) {
                        processorSet = true;
                        processorToBeAllocated = sched.AssignedProcessor(par);
                    }
                }
                if (failedToAllocate) {
                    failCounter++;
                    continue;
                } else {
                    failCounter = 0;
                }

                sched.SetAssignedSuperstep(node, superstepCounter);
                if (processorSet) {
                    sched.SetAssignedProcessor(node, processorToBeAllocated);
                } else {
                    auto minIter = std::min_element(processorWeights.begin(), processorWeights.end());

                    assert(std::distance(processorWeights.begin(), minIter) >= 0);

                    sched.SetAssignedProcessor(node, static_cast<unsigned>(std::distance(processorWeights.begin(), minIter)));
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

                auto it = next.begin();
                std::advance(it, nodeInd);
                next.erase(it);
                next.insert(next.end(), newNodes.cbegin(), newNodes.cend());

                if (ensureEnoughSources_ && fewSources && next.size() >= instance.NumberOfProcessors()) {
                    break;
                }
            }

            superstepCounter++;
        }

        return ReturnStatus::OSP_SUCCESS;
    }

    std::string GetScheduleName() const override { return ensureEnoughSources_ ? "RandomGreedyS" : "RandomGreedy"; }
};

}    // namespace osp
