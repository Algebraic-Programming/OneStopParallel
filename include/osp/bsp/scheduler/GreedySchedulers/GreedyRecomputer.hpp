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

#include "osp/bsp/model/BspScheduleRecomp.hpp"

namespace osp {

/**
 * @brief The GreedyReccomputer class applies a greedy algorithm to remove some of the communciation steps in
 * a BspSchedule by recomputation steps if this decreases the cost.
 */
template <typename GraphT>
class GreedyRecomputer {
    static_assert(isComputationalDagV<GraphT>, "GreedyRecomputer can only be used with computational DAGs.");

  private:
    using VertexIdx = VertexIdxT<GraphT>;
    using cost_type = VWorkwT<GraphT>;
    using KeyTriple = std::tuple<VertexIdxT<GraphT>, unsigned int, unsigned int>;

    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "GreedyRecomputer requires work and comm. weights to have the same type.");

  public:
    /**
     * @brief Default destructor for GreedyRecomputer.
     */
    virtual ~GreedyRecomputer() = default;

    ReturnStatus ComputeRecompSchedule(BspScheduleCS<GraphT> &initialSchedule, BspScheduleRecomp<GraphT> &outSchedule) const;
};

template <typename GraphT>
ReturnStatus GreedyRecomputer<GraphT>::ComputeRecompSchedule(BspScheduleCS<GraphT> &initialSchedule,
                                                             BspScheduleRecomp<GraphT> &outSchedule) const {
    const VertexIdx &n = initialSchedule.GetInstance().NumberOfVertices();
    const unsigned &p = initialSchedule.GetInstance().NumberOfProcessors();
    const unsigned &s = initialSchedule.NumberOfSupersteps();
    const GraphT &g = initialSchedule.GetInstance().GetComputationalDag();

    outSchedule = BspScheduleRecomp<GraphT>(initialSchedule.GetInstance());
    outSchedule.SetNumberOfSupersteps(initialSchedule.NumberOfSupersteps());

    // Initialize required data structures
    std::vector<std::vector<cost_type>> workCost(p, std::vector<cost_type>(s, 0)), sendCost(p, std::vector<cost_type>(s, 0)),
        recCost(p, std::vector<cost_type>(s, 0));

    std::vector<std::vector<unsigned>> firstComputable(n, std::vector<unsigned>(p, 0U)),
        firstPresent(n, std::vector<unsigned>(p, std::numeric_limits<unsigned>::max()));

    std::vector<std::vector<std::multiset<unsigned>>> neededOnProc(n, std::vector<std::multiset<unsigned>>(p, {s}));

    std::vector<cost_type> maxWork(s, 0), maxComm(s, 0);

    std::vector<std::set<KeyTriple>> commSteps(s);

    for (VertexIdx node = 0; node < n; ++node) {
        const unsigned &proc = initialSchedule.AssignedProcessor(node);
        const unsigned &step = initialSchedule.AssignedSuperstep(node);

        workCost[proc][step] += g.VertexWorkWeight(node);
        firstPresent[node][proc] = std::min(firstPresent[node][proc], step);
        for (VertexIdx pred : g.Parents(node)) {
            neededOnProc[pred][proc].insert(step);
        }

        outSchedule.Assignments(node).emplace_back(proc, step);
    }
    for (const std::pair<KeyTriple, unsigned> item : initialSchedule.GetCommunicationSchedule()) {
        const VertexIdx &node = std::get<0>(item.first);
        const unsigned &fromProc = std::get<1>(item.first);
        const unsigned &toProc = std::get<2>(item.first);
        const unsigned &step = item.second;
        sendCost[fromProc][step]
            += g.VertexCommWeight(node) * initialSchedule.GetInstance().GetArchitecture().CommunicationCosts(fromProc, toProc);
        recCost[toProc][step]
            += g.VertexCommWeight(node) * initialSchedule.GetInstance().GetArchitecture().CommunicationCosts(fromProc, toProc);

        commSteps[step].emplace(item.first);
        neededOnProc[node][fromProc].insert(step);
        firstPresent[node][toProc] = std::min(firstPresent[node][toProc], step + 1);
    }
    for (unsigned step = 0; step < s; ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            maxWork[step] = std::max(maxWork[step], workCost[proc][step]);
            maxComm[step] = std::max(maxComm[step], sendCost[proc][step]);
            maxComm[step] = std::max(maxComm[step], recCost[proc][step]);
        }
    }

    for (VertexIdx node = 0; node < n; ++node) {
        for (const VertexIdx &pred : g.Parents(node)) {
            for (unsigned proc = 0; proc < p; ++proc) {
                firstComputable[node][proc] = std::max(firstComputable[node][proc], firstPresent[pred][proc]);
            }
        }
    }

    // Find improvement steps
    bool stillImproved = true;
    while (stillImproved) {
        stillImproved = false;

        for (unsigned step = 0; step < s; ++step) {
            std::vector<KeyTriple> toErase;
            for (const KeyTriple &entry : commSteps[step]) {
                const VertexIdx &node = std::get<0>(entry);
                const unsigned &fromProc = std::get<1>(entry);
                const unsigned &toProc = std::get<2>(entry);

                // check how much comm cost we save by removing comm schedule entry
                cost_type commInduced = g.VertexCommWeight(node)
                                        * initialSchedule.GetInstance().GetArchitecture().CommunicationCosts(fromProc, toProc);

                cost_type newMaxComm = 0;
                for (unsigned proc = 0; proc < p; ++proc) {
                    if (proc == fromProc) {
                        newMaxComm = std::max(newMaxComm, sendCost[proc][step] - commInduced);
                    } else {
                        newMaxComm = std::max(newMaxComm, sendCost[proc][step]);
                    }
                    if (proc == toProc) {
                        newMaxComm = std::max(newMaxComm, recCost[proc][step] - commInduced);
                    } else {
                        newMaxComm = std::max(newMaxComm, recCost[proc][step]);
                    }
                }
                if (newMaxComm == maxComm[step]) {
                    continue;
                }

                if (!initialSchedule.GetInstance().IsCompatible(node, toProc)) {
                    continue;
                }

                cost_type decrease = maxComm[step] - newMaxComm;
                if (maxComm[step] > 0 && newMaxComm == 0) {
                    decrease += initialSchedule.GetInstance().GetArchitecture().SynchronisationCosts();
                }

                // check how much it would increase the work cost instead
                unsigned bestStep = s;
                cost_type smallestIncrease = std::numeric_limits<cost_type>::max();
                for (unsigned compStep = firstComputable[node][toProc]; compStep <= *neededOnProc[node][toProc].begin();
                     ++compStep) {
                    cost_type increase = workCost[toProc][compStep] + g.VertexWorkWeight(node) > maxWork[compStep]
                                             ? workCost[toProc][compStep] + g.VertexWorkWeight(node) - maxWork[compStep]
                                             : 0;

                    if (increase < smallestIncrease) {
                        bestStep = compStep;
                        smallestIncrease = increase;
                    }
                }

                // check if this modification is beneficial
                if (bestStep == s || smallestIncrease > decrease) {
                    continue;
                }

                // execute the modification
                toErase.emplace_back(entry);
                outSchedule.Assignments(node).emplace_back(toProc, bestStep);

                sendCost[fromProc][step] -= commInduced;
                recCost[toProc][step] -= commInduced;
                maxComm[step] = newMaxComm;

                workCost[toProc][bestStep] += g.VertexWorkWeight(node);
                maxWork[bestStep] += smallestIncrease;

                // update movability bounds
                for (const VertexIdx &pred : g.Parents(node)) {
                    neededOnProc[pred][toProc].insert(bestStep);
                }

                neededOnProc[node][fromProc].erase(neededOnProc[node][fromProc].lower_bound(step));

                firstPresent[node][toProc] = bestStep;
                for (const VertexIdx &succ : g.Children(node)) {
                    for (const VertexIdx &pred : g.Parents(node)) {
                        firstComputable[succ][toProc] = std::max(firstComputable[succ][toProc], firstPresent[pred][toProc]);
                    }
                }

                stillImproved = true;
            }
            for (const KeyTriple &entry : toErase) {
                commSteps[step].erase(entry);
            }
        }
    }

    for (unsigned step = 0; step < s; ++step) {
        for (const KeyTriple &entry : commSteps[step]) {
            outSchedule.GetCommunicationSchedule().emplace(entry, step);
        }
    }

    outSchedule.MergeSupersteps();

    return ReturnStatus::OSP_SUCCESS;
}

}    // namespace osp
