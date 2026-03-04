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
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");
    static_assert(isComputationalDagV<GraphT>, "GraphT must satisfy the computational_dag concept");

    using VertexIdx = VertexIdxT<GraphT>;
    using CostType = VCommwT<GraphT>;

    BspScheduleCS<GraphT> *schedule_;
    CostType cost_ = 0;

    // Main parameters for runnign algorithm
    bool steepestAscent_ = false;

    // aux data for comm schedule hill climbing
    std::vector<std::vector<unsigned>> commSchedule_;
    std::vector<std::vector<std::list<VertexIdx>>> supsteplists_;
    std::vector<std::set<std::pair<CostType, unsigned>>> commCostList_;
    std::vector<std::vector<typename std::set<std::pair<CostType, unsigned>>::iterator>> commCostPointer_;
    std::vector<std::vector<CostType>> sent_, received_, commCost_;
    std::vector<std::vector<std::pair<unsigned, unsigned>>> commBounds_;
    std::vector<std::vector<std::list<std::pair<VertexIdx, unsigned>>>> commSchedSendLists_;
    std::vector<std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>> commSchedSendListPointer_;
    std::vector<std::vector<std::list<std::pair<VertexIdx, unsigned>>>> commSchedRecLists_;
    std::vector<std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>> commSchedRecListPointer_;
    std::vector<CostType> minimumCostPerSuperstep_;
    unsigned nextSupstep_;

    // Create superstep lists (for convenience) for a BSP schedule
    void CreateSupstepLists();

    // Initialize data structures (based on current schedule)
    void Init();

    // compute cost change incurred by a potential move
    int MoveCostChange(VertexIdx node, unsigned targetProc, unsigned newStep);

    // execute a move, updating the comm. schedule and the data structures
    void ExecuteMove(VertexIdx node, unsigned targetProc, unsigned newStep, int changeCost);

    // Single comm. schedule hill climbing step
    bool Improve();

    // Convert communication schedule to new format in the end
    void ConvertCommSchedule();

  public:
    HillClimbingForCommSteps() {}

    virtual ~HillClimbingForCommSteps() = default;

    virtual ReturnStatus ImproveSchedule(BspScheduleCS<GraphT> &inputSchedule);

    // call with time limit
    virtual ReturnStatus ImproveScheduleWithTimeLimit(BspScheduleCS<GraphT> &inputSchedule, const unsigned timeLimit);

    // setting parameters
    void SetSteepestAscend(bool steepestAscent) { steepestAscent_ = steepestAscent; }

    virtual std::string GetScheduleName() const { return "HillClimbingForCommSchedule"; }
};

template <typename GraphT>
ReturnStatus HillClimbingForCommSteps<GraphT>::ImproveSchedule(BspScheduleCS<GraphT> &inputSchedule) {
    return ImproveScheduleWithTimeLimit(inputSchedule, 180);
}

// Main method for hill climbing (with time limit)
template <typename GraphT>
ReturnStatus HillClimbingForCommSteps<GraphT>::ImproveScheduleWithTimeLimit(BspScheduleCS<GraphT> &inputSchedule,
                                                                            const unsigned timeLimit) {
    schedule_ = &inputSchedule;

    if (schedule_->NumberOfSupersteps() <= 2) {
        return ReturnStatus::OSP_SUCCESS;
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

    return ReturnStatus::OSP_SUCCESS;
}

// Initialization for comm. schedule hill climbing
template <typename GraphT>
void HillClimbingForCommSteps<GraphT>::Init() {
    const unsigned numVertices = static_cast<unsigned>(schedule_->GetInstance().GetComputationalDag().NumVertices());
    const unsigned numProcs = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();
    const unsigned numSupersteps = schedule_->NumberOfSupersteps();
    const GraphT &G = schedule_->GetInstance().GetComputationalDag();

    CreateSupstepLists();
    cost_ = schedule_->ComputeCosts();

    nextSupstep_ = 0;
    commSchedule_.clear();
    commSchedule_.resize(numVertices, std::vector<unsigned>(numProcs, UINT_MAX));
    sent_.clear();
    sent_.resize(numSupersteps - 1, std::vector<CostType>(numProcs, 0));
    received_.clear();
    received_.resize(numSupersteps - 1, std::vector<CostType>(numProcs, 0));
    commCost_.clear();
    commCost_.resize(numSupersteps - 1, std::vector<CostType>(numProcs));
    commCostList_.clear();
    commCostList_.resize(numSupersteps - 1);
    commCostPointer_.clear();
    commCostPointer_.resize(numSupersteps - 1, std::vector<typename std::set<std::pair<CostType, unsigned>>::iterator>(numProcs));
    commBounds_.clear();
    commBounds_.resize(numVertices, std::vector<std::pair<unsigned, unsigned>>(numProcs));
    commSchedSendLists_.clear();
    commSchedSendLists_.resize(numSupersteps - 1, std::vector<std::list<std::pair<VertexIdx, unsigned>>>(numProcs));
    commSchedRecLists_.clear();
    commSchedRecLists_.resize(numSupersteps - 1, std::vector<std::list<std::pair<VertexIdx, unsigned>>>(numProcs));
    commSchedSendListPointer_.clear();
    commSchedSendListPointer_.resize(numVertices, std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>(numProcs));
    commSchedRecListPointer_.clear();
    commSchedRecListPointer_.resize(numVertices, std::vector<typename std::list<std::pair<VertexIdx, unsigned>>::iterator>(numProcs));

    // initialize to lazy comm schedule first - to make sure it's correct even if e.g. com scehdule has indirect sending
    for (unsigned step = 1; step < numSupersteps; ++step) {
        for (unsigned proc = 0; proc < numProcs; ++proc) {
            for (const VertexIdx node : supsteplists_[step][proc]) {
                for (const VertexIdx &pred : G.Parents(node)) {
                    if (schedule_->AssignedProcessor(pred) != schedule_->AssignedProcessor(node)
                        && commSchedule_[pred][schedule_->AssignedProcessor(node)] == UINT_MAX) {
                        commSchedule_[pred][schedule_->AssignedProcessor(node)] = step - schedule_->GetStaleness();
                        commBounds_[pred][schedule_->AssignedProcessor(node)]
                            = std::make_pair(schedule_->AssignedSuperstep(pred), step - schedule_->GetStaleness());
                    }
                }
            }
        }
    }

    // overwrite with original comm schedule, wherever possible
    const std::map<std::tuple<VertexIdx, unsigned, unsigned>, unsigned int> originalCommSchedule
        = schedule_->GetCommunicationSchedule();
    for (VertexIdx node = 0; node < numVertices; ++node) {
        for (unsigned proc = 0; proc < numProcs; ++proc) {
            if (commSchedule_[node][proc] == UINT_MAX) {
                continue;
            }

            const auto commScheduleKey = std::make_tuple(node, schedule_->AssignedProcessor(node), proc);
            auto mapIterator = originalCommSchedule.find(commScheduleKey);
            if (mapIterator != originalCommSchedule.end()) {
                unsigned originalStep = mapIterator->second;
                if (originalStep >= commBounds_[node][proc].first && originalStep <= commBounds_[node][proc].second) {
                    commSchedule_[node][proc] = originalStep;
                }
            }

            unsigned step = commSchedule_[node][proc];
            commSchedSendLists_[step][schedule_->AssignedProcessor(node)].emplace_front(node, proc);
            commSchedSendListPointer_[node][proc] = commSchedSendLists_[step][schedule_->AssignedProcessor(node)].begin();
            commSchedRecLists_[step][proc].emplace_front(node, proc);
            commSchedRecListPointer_[node][proc] = commSchedRecLists_[step][proc].begin();

            sent_[step][schedule_->AssignedProcessor(node)]
                += G.VertexCommWeight(node)
                   * schedule_->GetInstance().GetArchitecture().SendCosts(schedule_->AssignedProcessor(node), proc);
            received_[step][proc]
                += G.VertexCommWeight(node)
                   * schedule_->GetInstance().GetArchitecture().SendCosts(schedule_->AssignedProcessor(node), proc);
        }
    }

    for (unsigned step = 0; step < numSupersteps - 1; ++step) {
        for (unsigned proc = 0; proc < numProcs; ++proc) {
            commCost_[step][proc] = std::max(sent_[step][proc], received_[step][proc]);
            commCostPointer_[step][proc] = commCostList_[step].emplace(commCost_[step][proc], proc).first;
        }
    }

    // set minimum cost - differs for BSP and MaxBSP
    minimumCostPerSuperstep_.clear();
    if (schedule_->GetStaleness() == 1) {
        minimumCostPerSuperstep_.resize(numSupersteps - 1, 0);
    } else {
        minimumCostPerSuperstep_ = cost_helpers::ComputeMaxWorkPerStep(*schedule_);
        minimumCostPerSuperstep_.erase(minimumCostPerSuperstep_.begin());
    }
}

// compute cost change incurred by a potential move
template <typename GraphT>
int HillClimbingForCommSteps<GraphT>::MoveCostChange(const VertexIdx node, const unsigned targetProc, const unsigned newStep) {
    const unsigned oldStep = commSchedule_[node][targetProc];
    const unsigned sourceProc = schedule_->AssignedProcessor(node);
    int change = 0;

    // Change at old place
    auto itr = commCostList_[oldStep].rbegin();
    CostType oldMax = std::max(itr->first * schedule_->GetInstance().GetArchitecture().CommunicationCosts(),
                               minimumCostPerSuperstep_[oldStep])
                      + schedule_->GetInstance().GetArchitecture().SynchronisationCosts();
    CostType maxSource = std::max(sent_[oldStep][sourceProc]
                                      - schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                            * schedule_->GetInstance().GetArchitecture().SendCosts(sourceProc, targetProc),
                                  received_[oldStep][sourceProc]);
    CostType maxTarget = std::max(sent_[oldStep][targetProc],
                                  received_[oldStep][targetProc]
                                      - schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                            * schedule_->GetInstance().GetArchitecture().SendCosts(sourceProc, targetProc));
    CostType maxOther = 0;
    for (; itr != commCostList_[oldStep].rend(); ++itr) {
        if (itr->second != sourceProc && itr->second != targetProc) {
            maxOther = itr->first;
            break;
        }
    }

    CostType newMax
        = std::max(std::max(maxSource, maxTarget), maxOther) * schedule_->GetInstance().GetArchitecture().CommunicationCosts();
    CostType newSync = (newMax > 0) ? schedule_->GetInstance().GetArchitecture().SynchronisationCosts() : 0;
    newMax = std::max(newMax, minimumCostPerSuperstep_[oldStep]) + newSync;
    change += static_cast<int>(newMax) - static_cast<int>(oldMax);

    // Change at new place
    oldMax = commCostList_[newStep].rbegin()->first * schedule_->GetInstance().GetArchitecture().CommunicationCosts();
    CostType oldSync = (oldMax > 0) ? schedule_->GetInstance().GetArchitecture().SynchronisationCosts() : 0;
    oldMax = std::max(oldMax, minimumCostPerSuperstep_[newStep]);
    maxSource = schedule_->GetInstance().GetArchitecture().CommunicationCosts()
                * (sent_[newStep][sourceProc]
                   + schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                         * schedule_->GetInstance().GetArchitecture().SendCosts(sourceProc, targetProc));
    maxTarget = schedule_->GetInstance().GetArchitecture().CommunicationCosts()
                * (received_[newStep][targetProc]
                   + schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                         * schedule_->GetInstance().GetArchitecture().SendCosts(sourceProc, targetProc));

    newMax = std::max(std::max(oldMax, maxSource), maxTarget);
    change += static_cast<int>(newMax + schedule_->GetInstance().GetArchitecture().SynchronisationCosts())
              - static_cast<int>(oldMax + oldSync);

    return change;
}

// execute a move, updating the comm. schedule and the data structures
template <typename GraphT>
void HillClimbingForCommSteps<GraphT>::ExecuteMove(VertexIdx node, unsigned targetProc, const unsigned newStep, const int changeCost) {
    const unsigned oldStep = commSchedule_[node][targetProc];
    const unsigned sourceProc = schedule_->AssignedProcessor(node);
    cost_ = static_cast<CostType>(static_cast<int>(cost_) + changeCost);

    // Old step update
    if (sent_[oldStep][sourceProc] > received_[oldStep][sourceProc]) {
        commCostList_[oldStep].erase(commCostPointer_[oldStep][sourceProc]);
        sent_[oldStep][sourceProc] -= schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                      * schedule_->GetInstance().GetArchitecture().SendCosts(sourceProc, targetProc);
        commCost_[oldStep][sourceProc] = std::max(sent_[oldStep][sourceProc], received_[oldStep][sourceProc]);
        commCostPointer_[oldStep][sourceProc] = commCostList_[oldStep].emplace(commCost_[oldStep][sourceProc], sourceProc).first;
    } else {
        sent_[oldStep][sourceProc] -= schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                      * schedule_->GetInstance().GetArchitecture().SendCosts(sourceProc, targetProc);
    }

    if (received_[oldStep][targetProc] > sent_[oldStep][targetProc]) {
        commCostList_[oldStep].erase(commCostPointer_[oldStep][targetProc]);
        received_[oldStep][targetProc] -= schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                    * schedule_->GetInstance().GetArchitecture().SendCosts(sourceProc, targetProc);
        commCost_[oldStep][targetProc] = std::max(sent_[oldStep][targetProc], received_[oldStep][targetProc]);
        commCostPointer_[oldStep][targetProc] = commCostList_[oldStep].emplace(commCost_[oldStep][targetProc], targetProc).first;
    } else {
        received_[oldStep][targetProc] -= schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                                    * schedule_->GetInstance().GetArchitecture().SendCosts(sourceProc, targetProc);
    }

    // New step update
    sent_[newStep][sourceProc] += schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                               * schedule_->GetInstance().GetArchitecture().SendCosts(sourceProc, targetProc);
    if (sent_[newStep][sourceProc] > received_[newStep][sourceProc]) {
        commCostList_[newStep].erase(commCostPointer_[newStep][sourceProc]);
        commCost_[newStep][sourceProc] = sent_[newStep][sourceProc];
        commCostPointer_[newStep][sourceProc] = commCostList_[newStep].emplace(commCost_[newStep][sourceProc], sourceProc).first;
    }

    received_[newStep][targetProc] += schedule_->GetInstance().GetComputationalDag().VertexCommWeight(node)
                             * schedule_->GetInstance().GetArchitecture().SendCosts(sourceProc, targetProc);
    if (received_[newStep][targetProc] > sent_[newStep][targetProc]) {
        commCostList_[newStep].erase(commCostPointer_[newStep][targetProc]);
        commCost_[newStep][targetProc] = received_[newStep][targetProc];
        commCostPointer_[newStep][targetProc] = commCostList_[newStep].emplace(commCost_[newStep][targetProc], targetProc).first;
    }

    // CommSched update
    commSchedule_[node][targetProc] = newStep;

    // Comm lists
    commSchedSendLists_[oldStep][sourceProc].erase(commSchedSendListPointer_[node][targetProc]);
    commSchedSendLists_[newStep][sourceProc].emplace_front(node, targetProc);
    commSchedSendListPointer_[node][targetProc] = commSchedSendLists_[newStep][sourceProc].begin();

    commSchedRecLists_[oldStep][targetProc].erase(commSchedRecListPointer_[node][targetProc]);
    commSchedRecLists_[newStep][targetProc].emplace_front(node, targetProc);
    commSchedRecListPointer_[node][targetProc] = commSchedRecLists_[newStep][targetProc].begin();
}

// Single comm. schedule hill climbing step
template <typename GraphT>
bool HillClimbingForCommSteps<GraphT>::Improve() {
    const unsigned numSupersteps = static_cast<unsigned>(schedule_->NumberOfSupersteps());
    int bestDiff = 0;
    VertexIdx bestNode = 0;
    unsigned bestProc = 0, bestStep = 0;
    unsigned startingSupstep = nextSupstep_;

    // iterate over supersteps
    while (true) {
        auto itr = commCostList_[nextSupstep_].rbegin();

        if (itr == commCostList_[nextSupstep_].crend()) {
            break;
        }

        // find maximal comm cost that dominates the h-relation
        const CostType commMax = itr->first;
        if (commMax == 0) {
            nextSupstep_ = (nextSupstep_ + 1) % (numSupersteps - 1);
            if (nextSupstep_ == startingSupstep) {
                break;
            } else {
                continue;
            }
        }

        // go over all processors that incur this maximal comm cost in superstep nextSupstep_
        for (; itr != commCostList_[nextSupstep_].rend() && itr->first == commMax; ++itr) {
            const unsigned maxProc = itr->second;

            if (sent_[nextSupstep_][maxProc] == commMax) {
                for (const std::pair<VertexIdx, unsigned> &entry : commSchedSendLists_[nextSupstep_][maxProc]) {
                    const VertexIdx node = entry.first;
                    const unsigned proc = entry.second;
                    // iterate over alternative supsteps to place this communication step
                    for (unsigned step = commBounds_[node][proc].first; step < commBounds_[node][proc].second; ++step) {
                        if (step == commSchedule_[node][proc]) {
                            continue;
                        }

                        const int costDiff = MoveCostChange(node, proc, step);

                        if (!steepestAscent_ && costDiff < 0) {
                            ExecuteMove(node, proc, step, costDiff);
                            return true;
                        } else if (costDiff < bestDiff) {
                            bestNode = node;
                            bestProc = proc;
                            bestStep = step;
                            bestDiff = costDiff;
                        }
                    }
                }
            }

            if (received_[nextSupstep_][maxProc] == commMax) {
                for (const std::pair<VertexIdx, unsigned> &entry : commSchedRecLists_[nextSupstep_][maxProc]) {
                    const VertexIdx node = entry.first;
                    const unsigned proc = entry.second;
                    // iterate over alternative supsteps to place this communication step
                    for (unsigned step = commBounds_[node][proc].first; step < commBounds_[node][proc].second; ++step) {
                        if (step == commSchedule_[node][proc]) {
                            continue;
                        }

                        const int costDiff = MoveCostChange(node, proc, step);

                        if (!steepestAscent_ && costDiff < 0) {
                            ExecuteMove(node, proc, step, costDiff);
                            return true;
                        }
                        if (costDiff < bestDiff) {
                            bestNode = node;
                            bestProc = proc;
                            bestStep = step;
                            bestDiff = costDiff;
                        }
                    }
                }
            }
        }

        nextSupstep_ = (nextSupstep_ + 1) % (numSupersteps - 1);
        if (nextSupstep_ == startingSupstep) {
            break;
        }
    }

    if (bestDiff == 0) {
        return false;
    }

    ExecuteMove(bestNode, bestProc, bestStep, bestDiff);

    return true;
}

template <typename GraphT>
void HillClimbingForCommSteps<GraphT>::CreateSupstepLists() {
    const unsigned numProcs = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();
    const GraphT &G = schedule_->GetInstance().GetComputationalDag();

    schedule_->UpdateNumberOfSupersteps();
    const unsigned numSupersteps = schedule_->NumberOfSupersteps();

    supsteplists_.clear();
    supsteplists_.resize(numSupersteps, std::vector<std::list<VertexIdx>>(numProcs));

    const std::vector<VertexIdx> topOrder = GetTopOrder(G);
    for (VertexIdx node : topOrder) {
        supsteplists_[schedule_->AssignedSuperstep(node)][schedule_->AssignedProcessor(node)].push_back(node);
    }
}

template <typename GraphT>
void HillClimbingForCommSteps<GraphT>::ConvertCommSchedule() {
    const VertexIdx numVertices = static_cast<VertexIdx>(schedule_->GetInstance().GetComputationalDag().NumVertices());
    const unsigned numProcs = schedule_->GetInstance().GetArchitecture().NumberOfProcessors();

    std::map<std::tuple<VertexIdx, unsigned, unsigned>, unsigned> newCommSchedule;

    for (VertexIdx node = 0; node < numVertices; ++node) {
        for (unsigned proc = 0; proc < numProcs; ++proc) {
            if (commSchedule_[node][proc] != UINT_MAX) {
                const auto commScheduleKey = std::make_tuple(node, schedule_->AssignedProcessor(node), proc);
                newCommSchedule[commScheduleKey] = commSchedule_[node][proc];
            }
        }
    }

    schedule_->SetCommunicationSchedule(newCommSchedule);
}

}    // namespace osp
