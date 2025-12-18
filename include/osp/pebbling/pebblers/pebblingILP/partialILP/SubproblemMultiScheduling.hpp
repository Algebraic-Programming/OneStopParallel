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

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

template <typename GraphT>
class SubproblemMultiScheduling : public Scheduler<GraphT> {
    static_assert(isComputationalDagV<GraphT>, "PebblingSchedule can only be used with computational DAGs.");

  private:
    using VertexIdx = VertexIdxT<GraphT>;
    using commweight_type = VCommwT<GraphT>;
    using workweight_type = VWorkwT<GraphT>;

    std::vector<VertexIdx> lastNodeOnProc_;
    std::vector<std::vector<VertexIdx>> procTaskLists_;
    std::vector<workweight_type> longestOutgoingPath_;

  public:
    SubproblemMultiScheduling() {}

    virtual ~SubproblemMultiScheduling() = default;

    ReturnStatus ComputeMultiSchedule(const BspInstance<GraphT> &instance, std::vector<std::set<unsigned>> &processorsToNode);

    std::vector<std::pair<VertexIdx, unsigned>> MakeAssignment(const BspInstance<GraphT> &instance,
                                                               const std::set<std::pair<unsigned, VertexIdx>> &nodesAvailable,
                                                               const std::set<unsigned> &procsAvailable) const;

    std::vector<workweight_type> static GetLongestPath(const GraphT &graph);

    // not used, only here for using scheduler class base functionality (status enums, timelimits, etc)
    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override;

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string GetScheduleName() const override { return "SubproblemMultiScheduling"; }

    inline const std::vector<std::vector<unsigned>> &GetProcTaskLists() const { return procTaskLists_; }
};

// currently duplicated from BSP locking scheduler's code
template <typename GraphT>
std::vector<VWorkwT<GraphT>> SubproblemMultiScheduling<GraphT>::GetLongestPath(const GraphT &graph) {
    std::vector<workweight_type> longestPath(graph.NumVertices(), 0);

    std::vector<VertexIdx> topOrder = GetTopOrder(graph);

    for (auto rIter = topOrder.rbegin(); rIter != topOrder.crend(); rIter++) {
        longestPath[*rIter] = graph.VertexWorkWeight(*rIter);
        if (graph.OutDegree(*rIter) > 0) {
            workweight_type max = 0;
            for (const auto &child : graph.Children(*rIter)) {
                if (max <= longestPath[child]) {
                    max = longestPath[child];
                }
            }
            longestPath[*rIter] += max;
        }
    }

    return longestPath;
}

template <typename GraphT>
ReturnStatus SubproblemMultiScheduling<GraphT>::ComputeMultiSchedule(const BspInstance<GraphT> &instance,
                                                                     std::vector<std::set<unsigned>> &processorsToNode) {
    const unsigned &n = static_cast<unsigned>(instance.NumberOfVertices());
    const unsigned &p = instance.NumberOfProcessors();
    const auto &g = instance.GetComputationalDag();

    processorsToNode.clear();
    processorsToNode.resize(n);

    procTaskLists_.clear();
    procTaskLists_.resize(p);

    lastNodeOnProc_.clear();
    lastNodeOnProc_.resize(p, UINT_MAX);

    longestOutgoingPath_ = GetLongestPath(g);

    std::set<std::pair<unsigned, VertexIdx>> readySet;

    std::vector<unsigned> nrPredecRemain(n);
    for (VertexIdx node = 0; node < n; node++) {
        nrPredecRemain[node] = static_cast<unsigned>(g.InDegree(node));
        if (g.InDegree(node) == 0) {
            readySet.emplace(-longestOutgoingPath_[node], node);
        }
    }

    std::set<unsigned> freeProcs;
    for (unsigned proc = 0; proc < p; ++proc) {
        freeProcs.insert(proc);
    }

    std::vector<double> nodeFinishTime(n, 0);

    std::set<std::pair<double, VertexIdx>> finishTimes;
    finishTimes.emplace(0, std::numeric_limits<unsigned>::max());

    while (!readySet.empty() || !finishTimes.empty()) {
        const double time = finishTimes.begin()->first;

        // Find new ready jobs
        while (!finishTimes.empty() && fabs(finishTimes.begin()->first - time) < 0.0001) {
            const VertexIdx node = finishTimes.begin()->second;
            finishTimes.erase(finishTimes.begin());

            if (node != std::numeric_limits<unsigned>::max()) {
                for (const VertexIdx &succ : g.Children(node)) {
                    nrPredecRemain[succ]--;
                    if (nrPredecRemain[succ] == 0) {
                        readySet.emplace(-longestOutgoingPath_[succ], succ);
                    }
                }
                for (unsigned proc : processorsToNode[node]) {
                    freeProcs.insert(proc);
                }
            }
        }

        // Assign new jobs to idle processors

        // first assign free processors to ready nodes
        std::vector<std::pair<VertexIdx, unsigned>> newAssingments = MakeAssignment(instance, readySet, freeProcs);

        for (auto entry : newAssingments) {
            VertexIdx node = entry.first;
            unsigned proc = entry.second;

            processorsToNode[node].insert(proc);
            procTaskLists_[proc].push_back(node);
            finishTimes.emplace(time + g.VertexWorkWeight(node), node);
            nodeFinishTime[node] = time + g.VertexWorkWeight(node);
            lastNodeOnProc_[proc] = node;
            freeProcs.erase(proc);
            readySet.erase({-longestOutgoingPath_[node], node});
        }

        // assign remaining free processors to already started nodes, if it helps
        decltype(finishTimes.rbegin()) itr = finishTimes.rbegin();
        while (!freeProcs.empty() && itr != finishTimes.rend()) {
            double lastFinishTime = itr->first;

            decltype(finishTimes.rbegin()) itrLatest = itr;
            std::set<std::pair<workweight_type, VertexIdx>> possibleNodes;
            while (itrLatest != finishTimes.rend() && itrLatest->first + 0.0001 > lastFinishTime) {
                VertexIdx node = itrLatest->second;
                double newFinishTime
                    = time
                      + static_cast<double>(g.VertexWorkWeight(node)) / (static_cast<double>(processorsToNode[node].size()) + 1);
                if (newFinishTime + 0.0001 < itrLatest->first) {
                    possibleNodes.emplace(-longestOutgoingPath_[node], node);
                }

                ++itrLatest;
            }
            newAssingments = MakeAssignment(instance, possibleNodes, freeProcs);
            for (auto entry : newAssingments) {
                VertexIdx node = entry.first;
                unsigned proc = entry.second;

                processorsToNode[node].insert(proc);
                procTaskLists_[proc].push_back(node);
                finishTimes.erase({nodeFinishTime[node], node});
                double newFinishTime
                    = time + static_cast<double>(g.VertexWorkWeight(node)) / (static_cast<double>(processorsToNode[node].size()));
                finishTimes.emplace(newFinishTime, node);
                nodeFinishTime[node] = newFinishTime;
                lastNodeOnProc_[proc] = node;
                freeProcs.erase(proc);
            }
            if (newAssingments.empty()) {
                itr = itrLatest;
            }
        }
    }

    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT>
std::vector<std::pair<VertexIdxT<GraphT>, unsigned>> SubproblemMultiScheduling<GraphT>::MakeAssignment(
    const BspInstance<GraphT> &instance,
    const std::set<std::pair<unsigned, VertexIdx>> &nodesAvailable,
    const std::set<unsigned> &procsAvailable) const {
    std::vector<std::pair<VertexIdx, unsigned>> assignments;
    if (nodesAvailable.empty() || procsAvailable.empty()) {
        return assignments;
    }

    std::set<VertexIdx> assignedNodes;
    std::vector<bool> assignedProcs(instance.NumberOfProcessors(), false);

    for (unsigned proc : procsAvailable) {
        if (lastNodeOnProc_[proc] == UINT_MAX) {
            continue;
        }

        for (const auto &succ : instance.GetComputationalDag().Children(lastNodeOnProc_[proc])) {
            if (nodesAvailable.find({-longestOutgoingPath_[succ], succ}) != nodesAvailable.end()
                && instance.IsCompatible(succ, proc) && assignedNodes.find(succ) == assignedNodes.end()) {
                assignments.emplace_back(succ, proc);
                assignedNodes.insert(succ);
                assignedProcs[proc] = true;
                break;
            }
        }
    }

    for (unsigned proc : procsAvailable) {
        if (!assignedProcs[proc]) {
            for (auto itr = nodesAvailable.begin(); itr != nodesAvailable.end(); ++itr) {
                VertexIdx node = itr->second;
                if (instance.IsCompatible(node, proc) && assignedNodes.find(node) == assignedNodes.end()) {
                    assignments.emplace_back(node, proc);
                    assignedNodes.insert(node);
                    break;
                }
            }
        }
    }

    return assignments;
}

template <typename GraphT>
ReturnStatus SubproblemMultiScheduling<GraphT>::ComputeSchedule(BspSchedule<GraphT> &) {
    return ReturnStatus::ERROR;
}

}    // namespace osp
