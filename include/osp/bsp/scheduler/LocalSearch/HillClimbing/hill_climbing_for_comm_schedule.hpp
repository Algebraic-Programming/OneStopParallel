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

#include "osp/bsp/model/BspScheduleCS.hpp"
#include "osp/bsp/model/cost/CostModelHelpers.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_top_sort.hpp"

namespace osp {

template <typename GraphT>
class HillClimbingForCommSteps {
    static_assert(IsDirectedGraphV<Graph_t>, "Graph_t must satisfy the directed_graph concept");
    static_assert(IsComputationalDagV<Graph_t>, "Graph_t must satisfy the computational_dag concept");

    using vertex_idx = vertex_idx_t<Graph_t>;
    using cost_type = v_commw_t<Graph_t>;

    BspScheduleCS<GraphT> *schedule_;
    cost_type cost_ = 0;

    // Main parameters for runnign algorithm
    bool steepestAscent_ = false;

    // aux data for comm schedule hill climbing
    std::vector<std::vector<unsigned>> commSchedule_;
    std::vector<std::vector<std::list<vertex_idx>>> supsteplists_;
    std::vector<std::set<std::pair<cost_type, unsigned>>> commCostList_;
    std::vector<std::vector<typename std::set<std::pair<cost_type, unsigned>>::iterator>> commCostPointer_;
    std::vector<std::vector<cost_type>> sent_, received_, commCost_;
    std::vector<std::vector<std::pair<unsigned, unsigned>>> commBounds_;
    std::vector<std::vector<std::list<std::pair<vertex_idx, unsigned>>>> commSchedSendLists_;
    std::vector<std::vector<typename std::list<std::pair<vertex_idx, unsigned>>::iterator>> commSchedSendListPointer_;
    std::vector<std::vector<std::list<std::pair<vertex_idx, unsigned>>>> commSchedRecLists_;
    std::vector<std::vector<typename std::list<std::pair<vertex_idx, unsigned>>::iterator>> commSchedRecListPointer_;
    std::vector<cost_type> minimumCostPerSuperstep_;
    unsigned nextSupstep_;

    // Create superstep lists (for convenience) for a BSP schedule
    void CreateSupstepLists();

    // Initialize data structures (based on current schedule)
    void Init();

    // compute cost change incurred by a potential move
    int MoveCostChange(vertex_idx node, unsigned p, unsigned step);

    // execute a move, updating the comm. schedule and the data structures
    void ExecuteMove(vertex_idx node, unsigned p, unsigned step, int changeCost);

    // Single comm. schedule hill climbing step
    bool Improve();

    // Convert communication schedule to new format in the end
    void ConvertCommSchedule();

  public:
    HillClimbingForCommSteps() {}

    virtual ~HillClimbingForCommSteps() = default;

    virtual RETURN_STATUS ImproveSchedule(BspScheduleCS<GraphT> &inputSchedule);

    // call with time limit
    virtual RETURN_STATUS ImproveScheduleWithTimeLimit(BspScheduleCS<GraphT> &inputSchedule, const unsigned timeLimit);

    // setting parameters
    void SetSteepestAscend(bool steepestAscent) { steepestAscent_ = steepestAscent; }

    virtual std::string GetScheduleName() const { return "HillClimbingForCommSchedule"; }
};

template <typename GraphT>
RETURN_STATUS HillClimbingForCommSteps<GraphT>::ImproveSchedule(BspScheduleCS<GraphT> &inputSchedule) {
    return improveScheduleWithTimeLimit(input_schedule, 180);
}

// Main method for hill climbing (with time limit)
template <typename GraphT>
RETURN_STATUS HillClimbingForCommSteps<GraphT>::ImproveScheduleWithTimeLimit(BspScheduleCS<GraphT> &inputSchedule,
                                                                             const unsigned timeLimit) {
    schedule_ = &inputSchedule;

    if (schedule_->NumberOfSupersteps() <= 2) {
        return RETURN_STATUS::OSP_SUCCESS;
    }

    Init();
    // ConvertCommSchedule();
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

    unsigned counter = 0;
    while (Improve()) {
        if ((++counter) == 100) {
            counter = 0;
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
            if (elapsed >= timeLimit) {
                std::cout << "Comm. Sched. Hill Climbing was shut down due to time limit." << std::endl;
                break;
            }
        }
    }

    ConvertCommSchedule();

    return RETURN_STATUS::OSP_SUCCESS;
}

// Initialization for comm. schedule hill climbing
template <typename GraphT>
void HillClimbingForCommSteps<GraphT>::Init() {
    const unsigned n = static_cast<unsigned>(schedule_->GetInstance().GetComputationalDag().NumVertices());
    const unsigned p = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();
    const unsigned m = schedule_->NumberOfSupersteps();
    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    CreateSupstepLists();
    cost = schedule->computeCosts();

    nextSupstep_ = 0;
    commSchedule_.clear();
    commSchedule_.resize(n, std::vector<unsigned>(p, UINT_MAX));
    sent.clear();
    sent.resize(M - 1, std::vector<cost_type>(P, 0));
    received.clear();
    received.resize(M - 1, std::vector<cost_type>(P, 0));
    commCost.clear();
    commCost.resize(M - 1, std::vector<cost_type>(P));
    commCostList.clear();
    commCostList.resize(M - 1);
    commCostPointer.clear();
    commCostPointer.resize(M - 1, std::vector<typename std::set<std::pair<cost_type, unsigned>>::iterator>(P));
    commBounds_.clear();
    commBounds_.resize(n, std::vector<std::pair<unsigned, unsigned>>(p));
    commSchedSendLists.clear();
    commSchedSendLists.resize(M - 1, std::vector<std::list<std::pair<vertex_idx, unsigned>>>(P));
    commSchedRecLists.clear();
    commSchedRecLists.resize(M - 1, std::vector<std::list<std::pair<vertex_idx, unsigned>>>(P));
    commSchedSendListPointer.clear();
    commSchedSendListPointer.resize(N, std::vector<typename std::list<std::pair<vertex_idx, unsigned>>::iterator>(P));
    commSchedRecListPointer.clear();
    commSchedRecListPointer.resize(N, std::vector<typename std::list<std::pair<vertex_idx, unsigned>>::iterator>(P));

    // initialize to lazy comm schedule first - to make sure it's correct even if e.g. com scehdule has indirect sending
    for (unsigned step = 1; step < m; ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            for (const vertex_idx node : supsteplists[step][proc]) {
                for (const vertex_idx &pred : G.parents(node)) {
                    if (schedule->assignedProcessor(pred) != schedule->assignedProcessor(node)
                        && commSchedule[pred][schedule->assignedProcessor(node)] == UINT_MAX) {
                        commSchedule[pred][schedule->assignedProcessor(node)] = step - schedule->getStaleness();
                        commBounds[pred][schedule->assignedProcessor(node)]
                            = std::make_pair(schedule->assignedSuperstep(pred), step - schedule->getStaleness());
                    }
                }
            }
        }
    }

    // overwrite with original comm schedule, wherever possible
    const std::map<std::tuple<vertex_idx, unsigned, unsigned>, unsigned int> originalCommSchedule
        = schedule_->getCommunicationSchedule();
    for (vertex_idx node = 0; node < n; ++node) {
        for (unsigned proc = 0; proc < p; ++proc) {
            if (commSchedule_[node][proc] == UINT_MAX) {
                continue;
            }

            const auto commScheduleKey = std::make_tuple(node, schedule_->assignedProcessor(node), proc);
            auto mapIterator = originalCommSchedule.find(comm_schedule_key);
            if (mapIterator != originalCommSchedule.end()) {
                unsigned originalStep = mapIterator->second;
                if (originalStep >= commBounds_[node][proc].first && originalStep <= commBounds_[node][proc].second) {
                    commSchedule_[node][proc] = originalStep;
                }
            }

            unsigned step = commSchedule_[node][proc];
            commSchedSendLists[step][schedule->assignedProcessor(node)].emplace_front(node, proc);
            commSchedSendListPointer[node][proc] = commSchedSendLists[step][schedule->assignedProcessor(node)].begin();
            commSchedRecLists[step][proc].emplace_front(node, proc);
            commSchedRecListPointer[node][proc] = commSchedRecLists[step][proc].begin();

            sent[step][schedule->assignedProcessor(node)]
                += schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                   * schedule->GetInstance().GetArchitecture().sendCosts(schedule->assignedProcessor(node), proc);
            received[step][proc] += schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                    * schedule->GetInstance().GetArchitecture().sendCosts(schedule->assignedProcessor(node), proc);
        }
    }

    for (unsigned step = 0; step < m - 1; ++step) {
        for (unsigned proc = 0; proc < p; ++proc) {
            commCost[step][proc] = std::max(sent[step][proc], received[step][proc]);
            commCostPointer[step][proc] = commCostList[step].emplace(commCost[step][proc], proc).first;
        }
    }

    // set minimum cost - differs for BSP and MaxBSP
    minimum_cost_per_superstep.clear();
    if (schedule_->getStaleness() == 1) {
        minimum_cost_per_superstep.resize(M - 1, 0);
    } else {
        minimum_cost_per_superstep = cost_helpers::compute_max_work_per_step(*schedule);
        minimum_cost_per_superstep.erase(minimum_cost_per_superstep.begin());
    }
}

// compute cost change incurred by a potential move
template <typename GraphT>
int HillClimbingForCommSteps<GraphT>::MoveCostChange(const vertex_idx node, const unsigned p, const unsigned step) {
    const unsigned oldStep = commSchedule_[node][p];
    const unsigned sourceProc = schedule_->assignedProcessor(node);
    int change = 0;

    // Change at old place
    auto itr = commCostList[oldStep].rbegin();
    cost_type oldMax = std::max(itr->first * schedule->GetInstance().GetArchitecture().CommunicationCosts(),
                                minimum_cost_per_superstep[oldStep])
                       + schedule->GetInstance().GetArchitecture().SynchronisationCosts();
    cost_type maxSource = std::max(sent[oldStep][sourceProc]
                                       - schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                             * schedule->GetInstance().GetArchitecture().sendCosts(sourceProc, p),
                                   received[oldStep][sourceProc]);
    cost_type maxTarget = std::max(sent[oldStep][p],
                                   received[oldStep][p]
                                       - schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                             * schedule->GetInstance().GetArchitecture().sendCosts(sourceProc, p));
    cost_type maxOther = 0;
    for (; itr != commCostList[oldStep].rend(); ++itr) {
        if (itr->second != sourceProc && itr->second != p) {
            maxOther = itr->first;
            break;
        }
    }

    cost_type newMax
        = std::max(std::max(maxSource, maxTarget), maxOther) * schedule_->GetInstance().GetArchitecture().CommunicationCosts();
    cost_type newSync = (newMax > 0) ? schedule_->GetInstance().GetArchitecture().SynchronisationCosts() : 0;
    newMax = std::max(newMax, minimum_cost_per_superstep[oldStep]) + newSync;
    change += static_cast<int>(newMax) - static_cast<int>(oldMax);

    // Change at new place
    oldMax = commCostList[step].rbegin()->first * schedule->GetInstance().GetArchitecture().CommunicationCosts();
    cost_type oldSync = (oldMax > 0) ? schedule_->GetInstance().GetArchitecture().SynchronisationCosts() : 0;
    oldMax = std::max(oldMax, minimum_cost_per_superstep[step]);
    maxSource = schedule->GetInstance().GetArchitecture().CommunicationCosts()
                * (sent[step][sourceProc]
                   + schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                         * schedule->GetInstance().GetArchitecture().sendCosts(sourceProc, p));
    maxTarget = schedule->GetInstance().GetArchitecture().CommunicationCosts()
                * (received[step][p]
                   + schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                         * schedule->GetInstance().GetArchitecture().sendCosts(sourceProc, p));

    newMax = std::max(std::max(oldMax, maxSource), maxTarget);
    change += static_cast<int>(newMax + schedule_->GetInstance().GetArchitecture().SynchronisationCosts())
              - static_cast<int>(oldMax + oldSync);

    return change;
}

// execute a move, updating the comm. schedule and the data structures
template <typename GraphT>
void HillClimbingForCommSteps<GraphT>::ExecuteMove(vertex_idx node, unsigned p, const unsigned step, const int changeCost) {
    const unsigned oldStep = commSchedule_[node][p];
    const unsigned sourceProc = schedule_->assignedProcessor(node);
    cost = static_cast<cost_type>(static_cast<int>(cost) + changeCost);

    // Old step update
    if (sent[oldStep][sourceProc] > received[oldStep][sourceProc]) {
        commCostList[oldStep].erase(commCostPointer[oldStep][sourceProc]);
        sent[oldStep][sourceProc] -= schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                     * schedule->GetInstance().GetArchitecture().sendCosts(sourceProc, p);
        commCost[oldStep][sourceProc] = std::max(sent[oldStep][sourceProc], received[oldStep][sourceProc]);
        commCostPointer[oldStep][sourceProc] = commCostList[oldStep].emplace(commCost[oldStep][sourceProc], sourceProc).first;
    } else {
        sent[oldStep][sourceProc] -= schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                     * schedule->GetInstance().GetArchitecture().sendCosts(sourceProc, p);
    }

    if (received[oldStep][p] > sent[oldStep][p]) {
        commCostList[oldStep].erase(commCostPointer[oldStep][p]);
        received[oldStep][p] -= schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                * schedule->GetInstance().GetArchitecture().sendCosts(sourceProc, p);
        commCost[oldStep][p] = std::max(sent[oldStep][p], received[oldStep][p]);
        commCostPointer[oldStep][p] = commCostList[oldStep].emplace(commCost[oldStep][p], p).first;
    } else {
        received[oldStep][p] -= schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                * schedule->GetInstance().GetArchitecture().sendCosts(sourceProc, p);
    }

    // New step update
    sent[step][sourceProc] += schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                              * schedule->GetInstance().GetArchitecture().sendCosts(sourceProc, p);
    if (sent[step][sourceProc] > received[step][sourceProc]) {
        commCostList[step].erase(commCostPointer[step][sourceProc]);
        commCost[step][sourceProc] = sent[step][sourceProc];
        commCostPointer[step][sourceProc] = commCostList[step].emplace(commCost[step][sourceProc], sourceProc).first;
    }

    received[step][p] += schedule->GetInstance().GetComputationalDag().VertexCommWeight(node)
                         * schedule->GetInstance().GetArchitecture().sendCosts(sourceProc, p);
    if (received[step][p] > sent[step][p]) {
        commCostList[step].erase(commCostPointer[step][p]);
        commCost[step][p] = received[step][p];
        commCostPointer[step][p] = commCostList[step].emplace(commCost[step][p], p).first;
    }

    // CommSched update
    commSchedule_[node][p] = step;

    // Comm lists
    commSchedSendLists[oldStep][sourceProc].erase(commSchedSendListPointer[node][p]);
    commSchedSendLists[step][sourceProc].emplace_front(node, p);
    commSchedSendListPointer[node][p] = commSchedSendLists[step][sourceProc].begin();

    commSchedRecLists[oldStep][p].erase(commSchedRecListPointer[node][p]);
    commSchedRecLists[step][p].emplace_front(node, p);
    commSchedRecListPointer[node][p] = commSchedRecLists[step][p].begin();
}

// Single comm. schedule hill climbing step
template <typename GraphT>
bool HillClimbingForCommSteps<GraphT>::Improve() {
    const unsigned m = static_cast<unsigned>(schedule_->NumberOfSupersteps());
    int bestDiff = 0;
    vertex_idx bestNode = 0;
    unsigned bestProc = 0, bestStep = 0;
    unsigned startingSupstep = nextSupstep_;

    // iterate over supersteps
    while (true) {
        auto itr = commCostList[nextSupstep].rbegin();

        if (itr == commCostList[nextSupstep].crend()) {
            break;
        }

        // find maximal comm cost that dominates the h-relation
        const cost_type commMax = itr->first;
        if (commMax == 0) {
            nextSupstep_ = (nextSupstep_ + 1) % (m - 1);
            if (nextSupstep_ == startingSupstep) {
                break;
            } else {
                continue;
            }
        }

        // go over all processors that incur this maximal comm cost in superstep nextSupstep
        for (; itr != commCostList[nextSupstep].rend() && itr->first == commMax; ++itr) {
            const unsigned maxProc = itr->second;

            if (sent[nextSupstep][maxProc] == commMax) {
                for (const std::pair<vertex_idx, unsigned> &entry : commSchedSendLists[nextSupstep][maxProc]) {
                    const vertex_idx node = entry.first;
                    const unsigned p = entry.second;
                    // iterate over alternative supsteps to place this communication step
                    for (unsigned step = commBounds[node][p].first; step < commBounds[node][p].second; ++step) {
                        if (step == commSchedule[node][p]) {
                            continue;
                        }

                        const int costDiff = moveCostChange(node, p, step);

                        if (!steepestAscent && costDiff < 0) {
                            executeMove(node, p, step, costDiff);
                            return true;
                        } else if (costDiff < bestDiff) {
                            bestNode = node;
                            bestProc = p;
                            bestStep = step;
                            bestDiff = costDiff;
                        }
                    }
                }
            }

            if (received[nextSupstep][maxProc] == commMax) {
                for (const std::pair<vertex_idx, unsigned> &entry : commSchedRecLists[nextSupstep][maxProc]) {
                    const vertex_idx node = entry.first;
                    const unsigned p = entry.second;
                    // iterate over alternative supsteps to place this communication step
                    for (unsigned step = commBounds[node][p].first; step < commBounds[node][p].second; ++step) {
                        if (step == commSchedule[node][p]) {
                            continue;
                        }

                        const int costDiff = moveCostChange(node, p, step);

                        if (!steepestAscent && costDiff < 0) {
                            executeMove(node, p, step, costDiff);
                            return true;
                        }
                        if (costDiff < bestDiff) {
                            bestNode = node;
                            bestProc = p;
                            bestStep = step;
                            bestDiff = costDiff;
                        }
                    }
                }
            }
        }

        nextSupstep_ = (nextSupstep_ + 1) % (m - 1);
        if (nextSupstep_ == startingSupstep) {
            break;
        }
    }

    if (bestDiff == 0) {
        return false;
    }

    executeMove(bestNode, bestProc, bestStep, bestDiff);

    return true;
}

template <typename GraphT>
void HillClimbingForCommSteps<GraphT>::CreateSupstepLists() {
    const unsigned p = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();
    const GraphT &g = schedule_->GetInstance().GetComputationalDag();

    schedule_->updateNumberOfSupersteps();
    const unsigned m = schedule_->NumberOfSupersteps();

    supsteplists.clear();
    supsteplists.resize(M, std::vector<std::list<vertex_idx>>(P));

    const std::vector<vertex_idx> topOrder = GetTopOrder(g);
    for (vertex_idx node : topOrder) {
        supsteplists[schedule->assignedSuperstep(node)][schedule->assignedProcessor(node)].push_back(node);
    }
}

template <typename GraphT>
void HillClimbingForCommSteps<GraphT>::ConvertCommSchedule() {
    const vertex_idx n = static_cast<vertex_idx>(schedule_->GetInstance().GetComputationalDag().NumVertices());
    const unsigned p = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();

    std::map<std::tuple<vertex_idx, unsigned, unsigned>, unsigned> newCommSchedule;

    for (vertex_idx node = 0; node < N; ++node) {
        for (unsigned proc = 0; proc < p; ++proc) {
            if (commSchedule_[node][proc] != UINT_MAX) {
                const auto commScheduleKey = std::make_tuple(node, schedule_->assignedProcessor(node), proc);
                newCommSchedule[comm_schedule_key] = commSchedule_[node][proc];
            }
        }
    }

    schedule_->setCommunicationSchedule(newCommSchedule);
}

}    // namespace osp
